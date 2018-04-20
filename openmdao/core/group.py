"""Define the Group class."""
from __future__ import division

from collections import Iterable, Counter, OrderedDict, defaultdict
from itertools import product, chain
from numbers import Number
import warnings
import inspect

from six import iteritems, string_types, itervalues
from six.moves import range

import numpy as np
import networkx as nx

from openmdao.approximation_schemes.complex_step import ComplexStep
from openmdao.approximation_schemes.finite_difference import FiniteDifference
from openmdao.core.system import System
from openmdao.core.component import Component
from openmdao.proc_allocators.default_allocator import DefaultAllocator, ProcAllocationError
from openmdao.jacobians.assembled_jacobian import SUBJAC_META_DEFAULTS
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce
from openmdao.solvers.linear.linear_runonce import LinearRunOnce
from openmdao.utils.array_utils import convert_neg, array_connection_compatible
from openmdao.utils.general_utils import warn_deprecation, ContainsAll, all_ancestors
from openmdao.utils.units import is_compatible
from openmdao.utils.mpi import MPI
from openmdao.utils.graph_utils import all_connected_nodes

# regex to check for valid names.
import re
namecheck_rgx = re.compile('[a-zA-Z][_a-zA-Z0-9]*')


class Group(System):
    """
    Class used to group systems together; instantiate or inherit.

    Attributes
    ----------
    _mpi_proc_allocator : ProcAllocator
        Object used to allocate MPI processes to subsystems.
    _proc_info : dict of subsys_name: (min_procs, max_procs, weight)
        Information used to determine MPI process allocation to subsystems.
    """

    def __init__(self, **kwargs):
        """
        Set the solvers to nonlinear and linear block Gauss--Seidel by default.

        Parameters
        ----------
        **kwargs : dict
            dict of arguments available here and in all descendants of this
            Group.
        """
        super(Group, self).__init__(**kwargs)

        # TODO: we cannot set the solvers with property setters at the moment
        # because our lint check thinks that we are defining new attributes
        # called nonlinear_solver and linear_solver without documenting them.
        if not self._nonlinear_solver:
            self._nonlinear_solver = NonlinearRunOnce()
        if not self._linear_solver:
            self._linear_solver = LinearRunOnce()
        self._mpi_proc_allocator = DefaultAllocator()
        self._proc_info = {}

    def setup(self):
        """
        Build this group.

        This method should be overidden by your Group's method.

        You may call 'add_subsystem' to add systems to this group. You may also issue connections,
        and set the linear and nonlinear solvers for this group level. You cannot safely change
        anything on children systems; use the 'configure' method instead.

        Available attributes:
            name
            pathname
            comm
            metadata
        """
        pass

    def configure(self):
        """
        Configure this group to assign children settings.

        This method may optionally be overidden by your Group's method.

        You may only use this method to change settings on your children subsystems. This includes
        setting solvers in cases where you want to override the defaults.

        You can assume that the full hierarchy below your level has been instantiated and has
        already called its own configure methods.

        Available attributes:
            name
            pathname
            comm
            metadata
            system hieararchy with attribute access
        """
        pass

    def _configure(self):
        """
        Configure our model recursively to assign any children settings.

        Highest system's settings take precedence.
        """
        for subsys in self._subsystems_myproc:
            subsys._configure()

            if subsys._has_guess:
                self._has_guess = True

        self.configure()

    def _setup_procs(self, pathname, comm):
        """
        Distribute processors and assign pathnames.

        Parameters
        ----------
        pathname : str
            Global name of the system, including the path.
        comm : MPI.Comm or <FakeComm>
            MPI communicator object.
        """
        self.pathname = pathname
        self.comm = comm

        self._subsystems_allprocs = []
        self._manual_connections = {}
        self._design_vars = OrderedDict()
        self._responses = OrderedDict()

        self._static_mode = False
        self._subsystems_allprocs.extend(self._static_subsystems_allprocs)
        self._manual_connections.update(self._static_manual_connections)
        self._design_vars.update(self._static_design_vars)
        self._responses.update(self._static_responses)
        self.setup()
        self._static_mode = True

        if MPI:
            proc_info = [self._proc_info[s.name] for s in self._subsystems_allprocs]

            # Call the load balancing algorithm
            try:
                sub_inds, sub_comm, sub_proc_range = self._mpi_proc_allocator(
                    proc_info, len(self._subsystems_allprocs), comm)
            except ProcAllocationError as err:
                subs = self._subsystems_allprocs
                if err.sub_inds is None:
                    raise RuntimeError("%s: %s" % (self.pathname, err.msg))
                else:
                    raise RuntimeError("%s: MPI process allocation failed: %s for the following "
                                       "subsystems: %s" % (self.pathname, err.msg,
                                                           [subs[i].name for i in err.sub_inds]))

            self._subsystems_myproc = [self._subsystems_allprocs[ind] for ind in sub_inds]

            # Define local subsystems
            if np.sum([minp for minp, _, _ in proc_info]) <= comm.size:
                self._subsystems_myproc_inds = sub_inds
            else:
                # reorder the subsystems_allprocs based on which procs they live on. If we don't
                # do this, we can get ordering mismatches in some of our data structures.
                new_allsubs = []
                seen = set()
                gathered = self.comm.allgather(sub_inds)
                for rank, inds in enumerate(gathered):
                    for ind in inds:
                        if ind not in seen:
                            new_allsubs.append(self._subsystems_allprocs[ind])
                            seen.add(ind)
                self._subsystems_allprocs = new_allsubs
                sub_idxs = {s.name: i for i, s in enumerate(self._subsystems_allprocs)}

                # since the subsystems_allprocs order changed, we also have to update
                # subsystems_myproc_inds
                self._subsystems_myproc_inds = [sub_idxs[s.name] for s in self._subsystems_myproc]
        else:
            sub_comm = comm
            self._subsystems_myproc = self._subsystems_allprocs
            self._subsystems_myproc_inds = list(range(len(self._subsystems_myproc)))
            sub_proc_range = (0, 1)

        # Compute _subsystems_proc_range
        self._subsystems_proc_range = [sub_proc_range] * len(self._subsystems_myproc)

        # Perform recursion
        for subsys in self._subsystems_myproc:
            if self.pathname:
                subsys._setup_procs('.'.join((self.pathname, subsys.name)), sub_comm)
            else:
                subsys._setup_procs(subsys.name, sub_comm)

    def _setup_vars(self, recurse=True):
        """
        Call setup in components and count variables, total and by var_set.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(Group, self)._setup_vars()
        num_var = self._num_var
        num_var_byset = self._num_var_byset

        # Recursion
        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_vars(recurse)

        # Compute num_var, num_var_byset, at least locally
        for vec_name in self._lin_rel_vec_name_list:
            num_var[vec_name] = {}
            num_var_byset[vec_name] = {}
            for type_ in ['input', 'output']:
                num_var[vec_name][type_] = np.sum(
                    [subsys._num_var[vec_name][type_] for subsys in self._subsystems_myproc
                     if vec_name in subsys._rel_vec_names], dtype=int)

                num_var_byset[vec_name][type_] = vbyset = {}
                for subsys in self._subsystems_myproc:
                    if vec_name not in subsys._rel_vec_names:
                        continue
                    for set_name, num in iteritems(subsys._num_var_byset[vec_name][type_]):
                        if set_name not in vbyset:
                            vbyset[set_name] = 0
                        vbyset[set_name] += num

        # If running in parallel, allgather
        if self.comm.size > 1:
            # Perform a single allgather
            if self._subsystems_myproc and self._subsystems_myproc[0].comm.rank == 0:
                raw = (num_var, num_var_byset)
            else:
                raw = (None, None)
            gathered = self.comm.allgather(raw)

            for vec_name in self._lin_rel_vec_name_list:
                num_var = self._num_var[vec_name]
                num_var_byset = self._num_var_byset[vec_name]

                # Empty the dictionaries
                for type_ in ['input', 'output']:
                    num_var[type_] = 0
                    num_var_byset[type_] = {}

                # Process the gathered data and update the dictionaries
                for myproc_num_var, myproc_num_var_byset in gathered:
                    if myproc_num_var is None:
                        continue
                    for type_ in ['input', 'output']:
                        num_var[type_] += myproc_num_var[vec_name][type_]
                        for set_name, num in iteritems(myproc_num_var_byset[vec_name][type_]):
                            if set_name not in num_var_byset[type_]:
                                num_var_byset[type_][set_name] = 0
                            num_var_byset[type_][set_name] += num

        self._num_var['nonlinear'] = self._num_var['linear']
        self._num_var_byset['nonlinear'] = self._num_var_byset['linear']

    def _setup_var_index_ranges(self, set2iset, recurse=True):
        """
        Compute the division of variables by subsystem and pass down the set_name-to-iset maps.

        Parameters
        ----------
        set2iset : {'input': dict, 'output': dict}
            Dictionary mapping the var_set name to the var_set index.
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(Group, self)._setup_var_index_ranges(set2iset)

        nsub_allprocs = len(self._subsystems_allprocs)

        subsystems_var_range = self._subsystems_var_range = {}
        subsystems_var_range_byset = self._subsystems_var_range_byset = {}

        # First compute these on one processor for each subsystem
        for vec_name in self._lin_rel_vec_name_list:

            # Here, we count the number of variables (total and by varset) in each subsystem.
            # We do this so that we can compute the offset when we recurse into each subsystem.
            allprocs_counters = {
                type_: np.zeros(nsub_allprocs, int) for type_ in ['input', 'output']}
            allprocs_counters_byset = {
                type_: np.zeros((nsub_allprocs, len(set2iset[type_])), int)
                for type_ in ['input', 'output']}

            for type_ in ['input', 'output']:
                for subsys, isub in zip(self._subsystems_myproc, self._subsystems_myproc_inds):
                    if subsys.comm.rank == 0 and vec_name in subsys._rel_vec_names:
                        allprocs_counters[type_][isub] = subsys._num_var[vec_name][type_]
                        for set_name in subsys._num_var_byset[vec_name][type_]:
                            iset = set2iset[type_][set_name]
                            allprocs_counters_byset[type_][isub, iset] = \
                                subsys._num_var_byset[vec_name][type_][set_name]

            # If running in parallel, allgather
            if self.comm.size > 1:
                raw = (allprocs_counters, allprocs_counters_byset)
                gathered = self.comm.allgather(raw)

                allprocs_counters = {
                    type_: np.zeros(nsub_allprocs, int) for type_ in ['input', 'output']}
                allprocs_counters_byset = {
                    type_: np.zeros((nsub_allprocs, len(set2iset[type_])), int)
                    for type_ in ['input', 'output']
                }
                for myproc_counters, myproc_counters_byset in gathered:
                    for type_ in ['input', 'output']:
                        allprocs_counters[type_] += myproc_counters[type_]
                        allprocs_counters_byset[type_] += myproc_counters_byset[type_]

            # Compute _subsystems_var_range, _subsystems_var_range_byset
            subsystems_var_range[vec_name] = {}
            subsystems_var_range_byset[vec_name] = {}

            for type_ in ['input', 'output']:
                subsystems_var_range[vec_name][type_] = {}
                subsystems_var_range_byset[vec_name][type_] = {
                    set_name: {} for set_name in set2iset[type_]
                }

                for subsys, isub in zip(self._subsystems_myproc, self._subsystems_myproc_inds):
                    if vec_name not in subsys._rel_vec_names:
                        continue
                    subsystems_var_range[vec_name][type_][subsys.name] = (
                        np.sum(allprocs_counters[type_][:isub]),
                        np.sum(allprocs_counters[type_][:isub + 1]))
                    for set_name, rng in iteritems(subsystems_var_range_byset[vec_name][type_]):
                        iset = set2iset[type_][set_name]
                        rng[subsys.name] = (np.sum(allprocs_counters_byset[type_][:isub, iset]),
                                            np.sum(allprocs_counters_byset[type_][:isub + 1,
                                                                                  iset]))

        subsystems_var_range['nonlinear'] = subsystems_var_range['linear']
        subsystems_var_range_byset['nonlinear'] = subsystems_var_range_byset['linear']

        # Recursion
        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_var_index_ranges(set2iset, recurse)

    def _setup_var_data(self, recurse=True):
        """
        Compute the list of abs var names, abs/prom name maps, and metadata dictionaries.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(Group, self)._setup_var_data()
        allprocs_abs_names = self._var_allprocs_abs_names
        abs_names = self._var_abs_names
        allprocs_prom2abs_list = self._var_allprocs_prom2abs_list
        abs2prom = self._var_abs2prom
        allprocs_abs2meta = self._var_allprocs_abs2meta
        abs2meta = self._var_abs2meta

        # Recursion
        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_var_data(recurse)
                self._has_output_scaling |= subsys._has_output_scaling
                self._has_resid_scaling |= subsys._has_resid_scaling

        for subsys in self._subsystems_myproc:
            var_maps = subsys._get_maps(subsys._var_allprocs_prom2abs_list)

            # Assemble allprocs_abs2meta and abs2meta
            allprocs_abs2meta.update(subsys._var_allprocs_abs2meta)
            abs2meta.update(subsys._var_abs2meta)

            for type_ in ['input', 'output']:

                # Assemble abs_names and allprocs_abs_names
                allprocs_abs_names[type_].extend(subsys._var_allprocs_abs_names[type_])
                abs_names[type_].extend(subsys._var_abs_names[type_])

                # Assemble abs2prom
                for abs_name in subsys._var_abs_names[type_]:
                    sub_prom_name = subsys._var_abs2prom[type_][abs_name]
                    abs2prom[type_][abs_name] = var_maps[type_][sub_prom_name]

                # Assemble allprocs_prom2abs_list
                sub_allprocs_prom2abs_list_t = subsys._var_allprocs_prom2abs_list[type_]
                for sub_prom_name in sub_allprocs_prom2abs_list_t:
                    prom_name = var_maps[type_][sub_prom_name]
                    if prom_name not in allprocs_prom2abs_list[type_]:
                        allprocs_prom2abs_list[type_][prom_name] = []
                    allprocs_prom2abs_list[type_][prom_name].extend(
                        sub_allprocs_prom2abs_list_t[sub_prom_name])

        for prom_name, abs_list in iteritems(allprocs_prom2abs_list['output']):
            if len(abs_list) > 1:
                raise RuntimeError("Output name '%s' refers to "
                                   "multiple outputs: %s." %
                                   (prom_name, sorted(abs_list)))

        # If running in parallel, allgather
        if self.comm.size > 1:
            if self._subsystems_myproc and self._subsystems_myproc[0].comm.rank == 0:
                raw = (allprocs_abs_names, allprocs_prom2abs_list, allprocs_abs2meta,
                       self._has_output_scaling, self._has_resid_scaling)
            else:
                raw = (
                    {'input': [], 'output': []},
                    {'input': {}, 'output': {}},
                    {'input': {}, 'output': {}},
                    False,
                    False
                )
            gathered = self.comm.allgather(raw)

            for type_ in ['input', 'output']:
                allprocs_abs_names[type_] = []
                allprocs_prom2abs_list[type_] = defaultdict(list)

            for myproc_abs_names, myproc_prom2abs_list, myproc_abs2meta, oscale, rscale in gathered:
                self._has_output_scaling |= oscale
                self._has_resid_scaling |= rscale

                # Assemble in parallel allprocs_abs2meta
                allprocs_abs2meta.update(myproc_abs2meta)

                for type_ in ['input', 'output']:

                    # Assemble in parallel allprocs_abs_names
                    allprocs_abs_names[type_].extend(myproc_abs_names[type_])

                    # Assemble in parallel allprocs_prom2abs_list
                    for prom_name, abs_names_list in iteritems(myproc_prom2abs_list[type_]):
                        allprocs_prom2abs_list[type_][prom_name].extend(abs_names_list)

    def _setup_var_sizes(self, recurse=True):
        """
        Compute the arrays of local variable sizes for all variables/procs on this system.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(Group, self)._setup_var_sizes()

        iproc = self.comm.rank
        nproc = self.comm.size

        subsystems_proc_range = self._subsystems_proc_range

        # Recursion
        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_var_sizes(recurse)

        sizes = self._var_sizes
        sizes_byset = self._var_sizes_byset

        # Compute _var_sizes
        for vec_name in self._lin_rel_vec_name_list:
            sizes[vec_name] = {}
            sizes_byset[vec_name] = {}
            subsystems_var_range = self._subsystems_var_range[vec_name]
            subsystems_var_range_byset = self._subsystems_var_range_byset[vec_name]

            for type_ in ['input', 'output']:
                sizes[vec_name][type_] = np.zeros((nproc, self._num_var[vec_name][type_]), int)

                sizes_byset[vec_name][type_] = {}
                for set_name, nvars in iteritems(self._num_var_byset[vec_name][type_]):
                    sizes_byset[vec_name][type_][set_name] = np.zeros((nproc, nvars), int)

                for ind, subsys in enumerate(self._subsystems_myproc):
                    if vec_name not in subsys._rel_vec_names:
                        continue
                    proc_slice = slice(*subsystems_proc_range[ind])
                    var_slice = slice(*subsystems_var_range[type_][subsys.name])
                    sizes[vec_name][type_][proc_slice, var_slice] = \
                        subsys._var_sizes[vec_name][type_]

                    for set_name, subsizes in iteritems(subsys._var_sizes_byset[vec_name][type_]):
                        var_slice = slice(*subsystems_var_range_byset[type_][set_name][subsys.name])
                        sizes_byset[vec_name][type_][set_name][proc_slice, var_slice] = subsizes

        # If parallel, all gather
        if self.comm.size > 1:
            for vec_name in self._lin_rel_vec_name_list:
                sizes = self._var_sizes[vec_name]
                sizes_byset = self._var_sizes_byset[vec_name]
                for type_ in ['input', 'output']:
                    self.comm.Allgather(sizes[type_][iproc, :], sizes[type_])
                    for set_name, vsizes in iteritems(sizes_byset[type_]):
                        self.comm.Allgather(sizes_byset[type_][set_name][iproc, :], vsizes)

            # compute owning ranks
            owns = self._owning_rank
            for type_ in ('input', 'output'):
                sizes = self._var_sizes['linear'][type_]
                for i, name in enumerate(self._var_allprocs_abs_names[type_]):
                    for rank in range(self.comm.size):
                        if sizes[rank, i] > 0:
                            owns[name] = rank
                            break

        self._var_sizes['nonlinear'] = self._var_sizes['linear']
        self._var_sizes_byset['nonlinear'] = self._var_sizes_byset['linear']

        self._setup_global_shapes()

    def _setup_global_connections(self, recurse=True, conns=None):
        """
        Compute dict of all connections between this system's inputs and outputs.

        The connections come from 4 sources:
        1. Implicit connections owned by the current system
        2. Explicit connections declared by the current system
        3. Explicit connections declared by parent systems
        4. Implicit / explicit from subsystems

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        conns : dict
            Dictionary of connections passed down from parent group.
        """
        super(Group, self)._setup_global_connections()

        global_abs_in2out = self._conn_global_abs_in2out

        allprocs_prom2abs_list_in = self._var_allprocs_prom2abs_list['input']
        allprocs_prom2abs_list_out = self._var_allprocs_prom2abs_list['output']
        abs2meta = self._var_abs2meta
        pathname = self.pathname

        abs_in2out = {}

        if pathname == '':
            path_len = 0
            nparts = 0
        else:
            path_len = len(pathname) + 1
            nparts = len(pathname.split('.'))

        new_conns = defaultdict(dict)

        if conns is not None:
            for abs_in, abs_out in iteritems(conns):
                inparts = abs_in.split('.')
                outparts = abs_out.split('.')

                if inparts[:nparts] == outparts[:nparts]:
                    global_abs_in2out[abs_in] = abs_out

                    # if connection is contained in a subgroup, add to conns
                    # to pass down to subsystems.
                    if inparts[:nparts + 1] == outparts[:nparts + 1]:
                        new_conns[inparts[nparts]][abs_in] = abs_out

        # Add implicit connections (only ones owned by this group)
        for prom_name in allprocs_prom2abs_list_out:
            if prom_name in allprocs_prom2abs_list_in:
                abs_out = allprocs_prom2abs_list_out[prom_name][0]
                out_subsys = abs_out[path_len:].split('.', 1)[0]
                for abs_in in allprocs_prom2abs_list_in[prom_name]:
                    in_subsys = abs_in[path_len:].split('.', 1)[0]
                    if out_subsys != in_subsys:
                        abs_in2out[abs_in] = abs_out

        # Add explicit connections (only ones declared by this group)
        for prom_in, (prom_out, src_indices, flat_src_indices) in \
                iteritems(self._manual_connections):

            # throw an exception if either output or input doesn't exist
            # (not traceable to a connect statement, so provide context)
            if prom_out not in allprocs_prom2abs_list_out:
                raise NameError(
                    "Output '%s' does not exist for connection in '%s' from '%s' to '%s'." %
                    (prom_out, self.pathname, prom_out, prom_in))

            if prom_in not in allprocs_prom2abs_list_in:
                raise NameError(
                    "Input '%s' does not exist for connection in '%s' from '%s' to '%s'." %
                    (prom_in, self.pathname, prom_out, prom_in))

            # Throw an exception if output and input are in the same system
            # (not traceable to a connect statement, so provide context)
            # and check if src_indices is defined in both connect and add_input.
            abs_out = allprocs_prom2abs_list_out[prom_out][0]
            outparts = abs_out.split('.')
            out_subsys = outparts[:-1]

            for abs_in in allprocs_prom2abs_list_in[prom_in]:
                inparts = abs_in.split('.')
                in_subsys = inparts[:-1]
                if out_subsys == in_subsys:
                    raise RuntimeError("Output and input are in the same System " +
                                       "for connection in '%s' from '%s' to '%s'." %
                                       (self.pathname, prom_out, prom_in))

                if src_indices is not None and abs_in in abs2meta:
                    meta = abs2meta[abs_in]
                    if meta['src_indices'] is not None:
                        raise RuntimeError("%s: src_indices has been defined "
                                           "in both connect('%s', '%s') "
                                           "and add_input('%s', ...)." %
                                           (self.pathname, prom_out,
                                            prom_in, prom_in))
                    meta['src_indices'] = np.atleast_1d(src_indices)
                    meta['flat_src_indices'] = flat_src_indices

                if abs_in in abs_in2out:
                    raise RuntimeError("Input '%s' cannot be connected to '%s' because it's already"
                                       " connected to '%s'" % (abs_in, abs_out, abs_in2out[abs_in]))

                abs_in2out[abs_in] = abs_out

                # if connection is contained in a subgroup, add to conns to pass down to subsystems.
                if inparts[:nparts + 1] == outparts[:nparts + 1]:
                    new_conns[inparts[nparts]][abs_in] = abs_out

        # Recursion
        if recurse:
            for subsys in self._subsystems_myproc:
                if subsys.name in new_conns:
                    subsys._setup_global_connections(recurse=recurse,
                                                     conns=new_conns[subsys.name])
                else:
                    subsys._setup_global_connections(recurse=recurse)

        # Compute global_abs_in2out by first adding this group's contributions,
        # then adding contributions from systems above/below, then allgathering.
        conn_list = list(iteritems(global_abs_in2out))
        conn_list.extend(iteritems(abs_in2out))
        global_abs_in2out.update(abs_in2out)

        for subsys in self._subsystems_myproc:
            global_abs_in2out.update(subsys._conn_global_abs_in2out)
            conn_list.extend(iteritems(subsys._conn_global_abs_in2out))

        if len(conn_list) > len(global_abs_in2out):
            dupes = [n for n, val in iteritems(Counter(tgt for tgt, src in conn_list)) if val > 1]
            dup_info = defaultdict(set)
            for tgt, src in conn_list:
                for dup in dupes:
                    if tgt == dup:
                        dup_info[tgt].add(src)
            dup_info = [(n, srcs) for n, srcs in iteritems(dup_info) if len(srcs) > 1]
            if dup_info:
                msg = ["%s from %s" % (tgt, sorted(srcs)) for tgt, srcs in dup_info]
                raise RuntimeError("The following inputs have multiple connections: %s" %
                                   ", ".join(msg))

        # If running in parallel, allgather
        if self.comm.size > 1:
            if self._subsystems_myproc and self._subsystems_myproc[0].comm.rank == 0:
                raw = global_abs_in2out
            else:
                raw = {}
            gathered = self.comm.allgather(raw)

            for myproc_global_abs_in2out in gathered:
                global_abs_in2out.update(myproc_global_abs_in2out)

    def _init_relevance(self, mode):
        """
        Create the relevance dictionary.

        Parameters
        ----------
        mode : str
            Derivative direction, either 'fwd' or 'rev'.

        Returns
        -------
        dict
            The relevance dictionary.
        """
        desvars = self.get_design_vars(recurse=True, get_sizes=False)
        responses = self.get_responses(recurse=True, get_sizes=False)
        return get_relevant_vars(self._conn_global_abs_in2out, desvars, responses, mode)

    def _setup_connections(self, recurse=True):
        """
        Compute dict of all implicit and explicit connections owned by this system.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(Group, self)._setup_connections()
        abs_in2out = self._conn_abs_in2out

        global_abs_in2out = self._conn_global_abs_in2out
        pathname = self.pathname

        # Recursion
        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_connections(recurse)

        if pathname == '':
            path_len = 0
        else:
            path_len = len(pathname) + 1

        allprocs_abs2meta = self._var_allprocs_abs2meta

        # Check input/output units here, and set _has_input_scaling
        # to True for this Group if units are defined and different, or if
        # ref or ref0 are defined for the output.
        for abs_in, abs_out in iteritems(global_abs_in2out):
            # First, check that this system owns both the input and output.
            if abs_in[:len(pathname)] == pathname and abs_out[:len(pathname)] == pathname:
                # Second, check that they are in different subsystems of this system.
                out_subsys = abs_out[path_len:].split('.', 1)[0]
                in_subsys = abs_in[path_len:].split('.', 1)[0]
                if out_subsys != in_subsys:
                    abs_in2out[abs_in] = abs_out

            # if connected output has scaling then we need input scaling
            if not self._has_input_scaling:
                out_units = allprocs_abs2meta[abs_out]['units']
                in_units = allprocs_abs2meta[abs_in]['units']

                # if units are defined and different, we need input scaling.
                needs_input_scaling = (in_units and out_units and in_units != out_units)

                # we also need it if a connected output has any scaling.
                if not needs_input_scaling:
                    out_meta = allprocs_abs2meta[abs_out]

                    ref = out_meta['ref']
                    if np.isscalar(ref):
                        needs_input_scaling = ref != 1.0
                    else:
                        needs_input_scaling = np.any(ref != 1.0)

                    if not needs_input_scaling:
                        ref0 = out_meta['ref0']
                        if np.isscalar(ref0):
                            needs_input_scaling = ref0 != 0.0
                        else:
                            needs_input_scaling = np.any(ref0)

                        if not needs_input_scaling:
                            res_ref = out_meta['res_ref']
                            if np.isscalar(res_ref):
                                needs_input_scaling = res_ref != 1.0
                            else:
                                needs_input_scaling = np.any(res_ref != 1.0)

                self._has_input_scaling = needs_input_scaling

        # Now that both implicit & explicit connections have been added,
        # check unit/shape compatibility, but only for connections that are
        # either owned by (implicit) or declared by (explicit) this Group.
        # This way, we don't repeat the error checking in multiple groups.
        abs2meta = self._var_abs2meta

        for abs_in, abs_out in iteritems(abs_in2out):
            # check unit compatibility
            out_units = allprocs_abs2meta[abs_out]['units']
            in_units = allprocs_abs2meta[abs_in]['units']

            if out_units:
                if not in_units:
                    warnings.warn("Output '%s' with units of '%s' is "
                                  "connected to input '%s' which has no"
                                  " units." % (abs_out, out_units, abs_in))
                elif not is_compatible(in_units, out_units):
                    raise RuntimeError("Output units of '%s' for '%s' are"
                                       " incompatible with input units of "
                                       "'%s' for '%s'." %
                                       (out_units, abs_out, in_units, abs_in))
            elif in_units is not None:
                warnings.warn("Input '%s' with units of '%s' is "
                              "connected to output '%s' which has "
                              "no units." % (abs_in, in_units, abs_out))

            # check shape compatibility
            if abs_in in abs2meta and abs_out in abs2meta:
                # get output shape from allprocs meta dict, since it may
                # be distributed (we want global shape)
                out_shape = allprocs_abs2meta[abs_out]['global_shape']
                # get input shape and src_indices from the local meta dict
                # (input is always local)
                in_shape = abs2meta[abs_in]['shape']
                src_indices = abs2meta[abs_in]['src_indices']
                flat = abs2meta[abs_in]['flat_src_indices']

                if src_indices is None and out_shape != in_shape:
                    # out_shape != in_shape is allowed if
                    # there's no ambiguity in storage order
                    if not array_connection_compatible(in_shape, out_shape):
                        msg = ("The source and target shapes do not match or are ambiguous"
                               " for the connection '%s' to '%s'. Expected %s but got %s.")
                        raise ValueError(msg % (abs_out, abs_in,
                                                tuple([int(s) for s in in_shape]),
                                                tuple([int(s) for s in out_shape])))

                if src_indices is not None:
                    src_indices = np.atleast_1d(src_indices)

                    # initial dimensions of indices shape must be same shape as target
                    for idx_d, inp_d in zip(src_indices.shape, in_shape):
                        if idx_d != inp_d:
                            msg = ("The source indices %s do not specify a "
                                   "valid shape for the connection '%s' to "
                                   "'%s'. The target shape is "
                                   "%s but indices are %s.")
                            raise ValueError(msg % (str(src_indices).replace('\n', ''),
                                                    abs_out, abs_in,
                                                    in_shape, src_indices.shape))

                    # any remaining dimension of indices must match shape of source
                    if len(src_indices.shape) > len(in_shape):
                        source_dimensions = src_indices.shape[len(in_shape)]
                        if source_dimensions != len(out_shape):
                            msg = ("The source indices %s do not specify a "
                                   "valid shape for the connection '%s' to "
                                   "'%s'. The source has %d "
                                   "dimensions but the indices expect %d.")
                            raise ValueError(msg % (str(src_indices).replace('\n', ''),
                                                    abs_out, abs_in,
                                                    len(out_shape), source_dimensions))
                    else:
                        source_dimensions = 1

                    # check all indices are in range of the source dimensions
                    if flat:
                        out_size = np.prod(out_shape)
                        mx = np.max(src_indices)
                        mn = np.min(src_indices)
                        if mx >= out_size:
                            bad_idx = mx
                        elif mn < -out_size:
                            bad_idx = mn
                        else:
                            bad_idx = None
                        if bad_idx is not None:
                            msg = ("The source indices do not specify "
                                   "a valid index for the connection "
                                   "'%s' to '%s'. Index "
                                   "'%d' is out of range for a flat source "
                                   "of size %d.")
                            raise ValueError(msg % (abs_out, abs_in,
                                             bad_idx, out_size))
                        if src_indices.ndim > 1:
                            abs2meta[abs_in]['src_indices'] = \
                                abs2meta[abs_in]['src_indices'].flatten()
                    else:
                        for d in range(source_dimensions):
                            # when running under MPI, there is a value for each proc
                            d_size = out_shape[d] * self.comm.size
                            for i in src_indices[..., d].flat:
                                if abs(i) >= d_size:
                                    msg = ("The source indices do not specify "
                                           "a valid index for the connection "
                                           "'%s' to '%s'. Index "
                                           "'%d' is out of range for source "
                                           "dimension of size %d.")
                                    raise ValueError(msg % (abs_out, abs_in,
                                                     i, d_size))

    def _setup_global(self, ext_num_vars, ext_num_vars_byset, ext_sizes, ext_sizes_byset):
        """
        Compute total number and total size of variables in systems before / after this system.

        Parameters
        ----------
        ext_num_vars : {'input': (int, int), 'output': (int, int)}
            Total number of allprocs variables in system before/after this one.
        ext_num_vars_byset : {'input': dict of (int, int), 'output': dict of (int, int)}
            Same as above, but by var_set name.
        ext_sizes : {'input': (int, int), 'output': (int, int)}
            Total size of allprocs variables in system before/after this one.
        ext_sizes_byset : {'input': dict of (int, int), 'output': dict of (int, int)}
            Same as above, but by var_set name.
        """
        super(Group, self)._setup_global(
            ext_num_vars, ext_num_vars_byset, ext_sizes, ext_sizes_byset)

        iproc = self.comm.rank

        for subsys in self._subsystems_myproc:
            sub_ext_num_vars = {}
            sub_ext_sizes = {}
            sub_ext_num_vars_byset = {}
            sub_ext_sizes_byset = {}

            for vec_name in subsys._lin_rel_vec_name_list:
                subsystems_var_range = self._subsystems_var_range[vec_name]
                subsystems_var_range_byset = self._subsystems_var_range_byset[vec_name]
                sizes = self._var_sizes[vec_name]
                sizes_byset = self._var_sizes_byset[vec_name]

                sub_ext_num_vars[vec_name] = {}
                sub_ext_sizes[vec_name] = {}
                sub_ext_num_vars_byset[vec_name] = {}
                sub_ext_sizes_byset[vec_name] = {}

                for type_ in ['input', 'output']:
                    num = self._num_var[vec_name][type_]
                    idx1, idx2 = subsystems_var_range[type_][subsys.name]
                    size1 = np.sum(sizes[type_][iproc, :idx1])
                    size2 = np.sum(sizes[type_][iproc, idx2:])

                    sub_ext_num_vars[vec_name][type_] = (
                        ext_num_vars[vec_name][type_][0] + idx1,
                        ext_num_vars[vec_name][type_][1] + num - idx2,
                    )
                    sub_ext_sizes[vec_name][type_] = (
                        ext_sizes[vec_name][type_][0] + size1,
                        ext_sizes[vec_name][type_][1] + size2,
                    )

                    sub_ext_sizes_byset[vec_name][type_] = {}
                    sub_ext_num_vars_byset[vec_name][type_] = {}
                    for set_name, num in iteritems(self._num_var_byset[vec_name][type_]):
                        idx1, idx2 = subsystems_var_range_byset[type_][set_name][subsys.name]
                        size1 = np.sum(sizes_byset[type_][set_name][iproc, :idx1])
                        size2 = np.sum(sizes_byset[type_][set_name][iproc, idx2:])

                        sub_ext_num_vars_byset[vec_name][type_][set_name] = (
                            ext_num_vars_byset[vec_name][type_][set_name][0] + idx1,
                            ext_num_vars_byset[vec_name][type_][set_name][1] + num - idx2,
                        )
                        sub_ext_sizes_byset[vec_name][type_][set_name] = (
                            ext_sizes_byset[vec_name][type_][set_name][0] + size1,
                            ext_sizes_byset[vec_name][type_][set_name][1] + size2,
                        )

            sub_ext_num_vars['nonlinear'] = sub_ext_num_vars['linear']
            sub_ext_sizes['nonlinear'] = sub_ext_sizes['linear']
            sub_ext_num_vars_byset['nonlinear'] = sub_ext_num_vars_byset['linear']
            sub_ext_sizes_byset['nonlinear'] = sub_ext_sizes_byset['linear']

            subsys._setup_global(
                sub_ext_num_vars, sub_ext_num_vars_byset,
                sub_ext_sizes, sub_ext_sizes_byset
            )

    def _setup_transfers(self, recurse=True):
        """
        Compute all transfers that are owned by this system.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(Group, self)._setup_transfers()

        def merge(indices_list):
            if len(indices_list) > 0:
                return np.concatenate(indices_list)
            else:
                return np.array([], int)

        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_transfers(recurse)

        # Pre-compute map from abs_names to the index of the containing subsystem
        abs2isub = {'input': {}, 'output': {}}
        for subsys, isub in zip(self._subsystems_myproc, self._subsystems_myproc_inds):
            for type_ in ['input', 'output']:
                for abs_name in subsys._var_allprocs_abs_names[type_]:
                    abs2isub[type_][abs_name] = isub

        abs2meta = self._var_abs2meta
        allprocs_abs2meta = self._var_allprocs_abs2meta

        transfers = self._transfers
        vectors = self._vectors
        for vec_name in self._lin_rel_vec_name_list:
            relvars, _ = self._relevant[vec_name]['@all']

            # Initialize empty lists for the transfer indices
            nsub_allprocs = len(self._subsystems_allprocs)
            xfer_in = {}
            xfer_out = {}
            fwd_xfer_in = [{} for i in range(nsub_allprocs)]
            fwd_xfer_out = [{} for i in range(nsub_allprocs)]
            rev_xfer_in = [{} for i in range(nsub_allprocs)]
            rev_xfer_out = [{} for i in range(nsub_allprocs)]
            for set_name_in in self._num_var_byset[vec_name]['input']:
                for set_name_out in self._num_var_byset[vec_name]['output']:
                    key = (set_name_in, set_name_out)
                    xfer_in[key] = []
                    xfer_out[key] = []
                    for isub in range(nsub_allprocs):
                        fwd_xfer_in[isub][key] = []
                        fwd_xfer_out[isub][key] = []
                        rev_xfer_in[isub][key] = []
                        rev_xfer_out[isub][key] = []

            allprocs_abs2idx_byset = self._var_allprocs_abs2idx_byset[vec_name]
            sizes_byset_in = self._var_sizes_byset[vec_name]['input']
            sizes_byset_out = self._var_sizes_byset[vec_name]['output']

            # Loop through all explicit / implicit connections owned by this system
            for abs_in, abs_out in iteritems(self._conn_abs_in2out):
                if abs_out not in relvars['output']:
                    continue

                # Only continue if the input exists on this processor
                if abs_in in abs2meta and abs_in in relvars['input']:

                    # Get meta
                    meta_in = abs2meta[abs_in]
                    meta_out = allprocs_abs2meta[abs_out]

                    # Get varset info
                    set_name_in = meta_in['var_set']
                    set_name_out = meta_out['var_set']
                    idx_byset_in = allprocs_abs2idx_byset[abs_in]
                    idx_byset_out = allprocs_abs2idx_byset[abs_out]

                    # Get the sizes (byset) array
                    sizes_in = sizes_byset_in[set_name_in]
                    sizes_out = sizes_byset_out[set_name_out]

                    # Read in and process src_indices
                    shape_in = meta_in['shape']
                    shape_out = meta_out['shape']
                    global_shape_out = meta_out['global_shape']
                    global_size_out = meta_out['global_size']
                    src_indices = meta_in['src_indices']
                    if src_indices is None:
                        src_indices = np.arange(meta_in['size'], dtype=int)
                    elif src_indices.ndim == 1:
                        src_indices = convert_neg(src_indices, global_size_out)
                    else:
                        if len(shape_out) == 1 or shape_in == src_indices.shape:
                            src_indices = src_indices.flatten()
                            src_indices = convert_neg(src_indices, global_size_out)
                        else:
                            # TODO: this duplicates code found
                            # in System._setup_scaling.
                            entries = [list(range(x)) for x in shape_in]
                            cols = np.vstack(src_indices[i] for i in product(*entries))
                            dimidxs = [convert_neg(cols[:, i], global_shape_out[i])
                                       for i in range(cols.shape[1])]
                            src_indices = np.ravel_multi_index(dimidxs, global_shape_out)

                    # 1. Compute the output indices
                    output_inds = np.zeros(src_indices.shape[0], int)
                    ind1 = ind2 = 0
                    for iproc in range(self.comm.size):
                        ind2 += sizes_out[iproc, idx_byset_out]

                        # The part of src on iproc
                        on_iproc = np.logical_and(ind1 <= src_indices, src_indices < ind2)

                        # This converts from iproc-then-ivar to ivar-then-iproc ordering
                        # Subtract off part of previous procs
                        # Then add all variables on previous procs
                        # Then all previous variables on this proc
                        # - np.sum(out_sizes[:iproc, idx_byset_out])
                        # + np.sum(out_sizes[:iproc, :])
                        # + np.sum(out_sizes[iproc, :idx_byset_out])
                        # + inds
                        offset = -ind1
                        offset += np.sum(sizes_out[:iproc, :])
                        offset += np.sum(sizes_out[iproc, :idx_byset_out])
                        output_inds[on_iproc] = src_indices[on_iproc] + offset

                        ind1 += sizes_out[iproc, idx_byset_out]

                    # 2. Compute the input indices
                    iproc = self.comm.rank
                    ind1 = ind2 = np.sum(sizes_in[:iproc, :])
                    delta = np.sum(sizes_in[iproc, :idx_byset_in])
                    ind1 += delta
                    ind2 += (delta + sizes_in[iproc, idx_byset_in])
                    input_inds = np.arange(ind1, ind2)

                    # Now the indices are ready - input_inds, output_inds
                    key = (set_name_in, set_name_out)
                    xfer_in[key].append(input_inds)
                    xfer_out[key].append(output_inds)

                    isub = abs2isub['input'][abs_in]
                    fwd_xfer_in[isub][key].append(input_inds)
                    fwd_xfer_out[isub][key].append(output_inds)
                    if abs_out in abs2isub['output']:
                        isub = abs2isub['output'][abs_out]
                        rev_xfer_in[isub][key].append(input_inds)
                        rev_xfer_out[isub][key].append(output_inds)

            for set_name_in in self._num_var_byset[vec_name]['input']:
                for set_name_out in self._num_var_byset[vec_name]['output']:
                    key = (set_name_in, set_name_out)
                    xfer_in[key] = merge(xfer_in[key])
                    xfer_out[key] = merge(xfer_out[key])
                    for isub in range(nsub_allprocs):
                        fwd_xfer_in[isub][key] = merge(fwd_xfer_in[isub][key])
                        fwd_xfer_out[isub][key] = merge(fwd_xfer_out[isub][key])
                        rev_xfer_in[isub][key] = merge(rev_xfer_in[isub][key])
                        rev_xfer_out[isub][key] = merge(rev_xfer_out[isub][key])

            out_vec = vectors['output'][vec_name]
            transfer_class = out_vec.TRANSFER

            transfers[vec_name] = {}
            xfer_all = transfer_class(vectors['input'][vec_name], out_vec,
                                      xfer_in, xfer_out, self.comm)
            transfers[vec_name]['fwd', None] = xfer_all
            transfers[vec_name]['rev', None] = xfer_all
            for isub in range(nsub_allprocs):
                transfers[vec_name]['fwd', isub] = transfer_class(
                    vectors['input'][vec_name], vectors['output'][vec_name],
                    fwd_xfer_in[isub], fwd_xfer_out[isub], self.comm)
                transfers[vec_name]['rev', isub] = transfer_class(
                    vectors['input'][vec_name], vectors['output'][vec_name],
                    rev_xfer_in[isub], rev_xfer_out[isub], self.comm)

        transfers['nonlinear'] = transfers['linear']

    def add(self, name, subsys, promotes=None):
        """
        Add a subsystem (deprecated version of <Group.add_subsystem>).

        Parameters
        ----------
        name : str
            Name of the subsystem being added
        subsys : System
            An instantiated, but not-yet-set up system object.
        promotes : iter of str, optional
            A list of variable names specifying which subsystem variables
            to 'promote' up to this group. This is for backwards compatibility
            with older versions of OpenMDAO.

        Returns
        -------
        System
            The System that was passed in.
        """
        warn_deprecation("The 'add' method provides backwards compatibility with "
                         "OpenMDAO <= 1.x ; use 'add_subsystem' instead.")

        return self.add_subsystem(name, subsys, promotes=promotes)

    def add_subsystem(self, name, subsys, promotes=None,
                      promotes_inputs=None, promotes_outputs=None,
                      min_procs=1, max_procs=None, proc_weight=1.0):
        """
        Add a subsystem.

        Parameters
        ----------
        name : str
            Name of the subsystem being added
        subsys : <System>
            An instantiated, but not-yet-set up system object.
        promotes : iter of (str or tuple), optional
            A list of variable names specifying which subsystem variables
            to 'promote' up to this group. If an entry is a tuple of the
            form (old_name, new_name), this will rename the variable in
            the parent group.
        promotes_inputs : iter of (str or tuple), optional
            A list of input variable names specifying which subsystem input
            variables to 'promote' up to this group. If an entry is a tuple of
            the form (old_name, new_name), this will rename the variable in
            the parent group.
        promotes_outputs : iter of (str or tuple), optional
            A list of output variable names specifying which subsystem output
            variables to 'promote' up to this group. If an entry is a tuple of
            the form (old_name, new_name), this will rename the variable in
            the parent group.
        min_procs : int
            Minimum number of MPI processes usable by the subsystem. Defaults to 1.
        max_procs : int or None
            Maximum number of MPI processes usable by the subsystem.  A value
            of None (the default) indicates there is no maximum limit.
        proc_weight : float
            Weight given to the subsystem when allocating available MPI processes
            to all subsystems.  Default is 1.0.

        Returns
        -------
        <System>
            the subsystem that was passed in. This is returned to
            enable users to instantiate and add a subsystem at the
            same time, and get the reference back.
        """
        if inspect.isclass(subsys):
            raise TypeError("Subsystem '%s' should be an instance, "
                            "but a class object was found." % name)

        for sub in chain(self._subsystems_allprocs,
                         self._static_subsystems_allprocs):
            if name == sub.name:
                raise RuntimeError("Subsystem name '%s' is already used." %
                                   name)

        if hasattr(self, name) and not isinstance(getattr(self, name), System):
            # replacing a subsystem is ok (e.g. resetup) but no other attribute
            raise RuntimeError("Group '%s' already has an attribute '%s'." %
                               (self.name, name))

        match = namecheck_rgx.match(name)
        if match is None or match.group() != name:
            raise NameError("'%s' is not a valid system name." % name)

        subsys.name = name

        if isinstance(promotes, string_types) or \
           isinstance(promotes_inputs, string_types) or \
           isinstance(promotes_outputs, string_types):
                raise RuntimeError("%s: promotes"
                                   " must be an iterator of strings and/or tuples." %
                                   self.name)
        if promotes:
            subsys._var_promotes['any'] = promotes
        if promotes_inputs:
            subsys._var_promotes['input'] = promotes_inputs
        if promotes_outputs:
            subsys._var_promotes['output'] = promotes_outputs

        if self._static_mode:
            subsystems_allprocs = self._static_subsystems_allprocs
        else:
            subsystems_allprocs = self._subsystems_allprocs

        subsystems_allprocs.append(subsys)

        if not isinstance(min_procs, int) or min_procs < 1:
            raise TypeError("%s: min_procs must be an int > 0 but (%s) was given." %
                            (self.name, min_procs))
        if max_procs is not None and (not isinstance(max_procs, int) or max_procs < min_procs):
            raise TypeError("%s: max_procs must be None or an int >= min_procs but (%s) was given."
                            % (self.name, max_procs))
        if isinstance(proc_weight, Number) and proc_weight < 0:
            raise TypeError("%s: proc_weight must be a float > 0. but (%s) was given." %
                            (self.name, proc_weight))

        self._proc_info[name] = (min_procs, max_procs, proc_weight)

        setattr(self, name, subsys)

        return subsys

    def connect(self, src_name, tgt_name, src_indices=None, flat_src_indices=None):
        """
        Connect source src_name to target tgt_name in this namespace.

        Parameters
        ----------
        src_name : str
            name of the source variable to connect
        tgt_name : str or [str, ... ] or (str, ...)
            name of the target variable(s) to connect
        src_indices : int or list of ints or tuple of ints or int ndarray or Iterable or None
            The global indices of the source variable to transfer data from.
            The shapes of the target and src_indices must match, and form of the
            entries within is determined by the value of 'flat_src_indices'.
        flat_src_indices : bool
            If True, each entry of src_indices is assumed to be an index into the
            flattened source.  Otherwise it must be a tuple or list of size equal
            to the number of dimensions of the source.
        """
        # if src_indices argument is given, it should be valid
        if isinstance(src_indices, string_types):
            if isinstance(tgt_name, string_types):
                tgt_name = [tgt_name]
            tgt_name.append(src_indices)
            raise TypeError("src_indices must be an index array, did you mean"
                            " connect('%s', %s)?" % (src_name, tgt_name))

        if isinstance(src_indices, Iterable):
            src_indices = np.atleast_1d(src_indices)

        if isinstance(src_indices, np.ndarray):
            if not np.issubdtype(src_indices.dtype, np.integer):
                raise TypeError("src_indices must contain integers, but src_indices for "
                                "connection from '%s' to '%s' is %s." %
                                (src_name, tgt_name, src_indices.dtype.type))

        # if multiple targets are given, recursively connect to each
        if not isinstance(tgt_name, string_types) and isinstance(tgt_name, Iterable):
            for name in tgt_name:
                self.connect(src_name, name, src_indices, flat_src_indices=flat_src_indices)
            return

        # target should not already be connected
        for manual_connections in [self._manual_connections, self._static_manual_connections]:
            if tgt_name in manual_connections:
                srcname = manual_connections[tgt_name][0]
                raise RuntimeError("Input '%s' is already connected to '%s'." %
                                   (tgt_name, srcname))

        # source and target should not be in the same system
        if src_name.rsplit('.', 1)[0] == tgt_name.rsplit('.', 1)[0]:
            raise RuntimeError("Output and input are in the same System for " +
                               "connection from '%s' to '%s'." % (src_name, tgt_name))

        if self._static_mode:
            manual_connections = self._static_manual_connections
        else:
            manual_connections = self._manual_connections

        manual_connections[tgt_name] = (src_name, src_indices, flat_src_indices)

    def set_order(self, new_order):
        """
        Specify a new execution order for this system.

        Parameters
        ----------
        new_order : list of str
            List of system names in desired new execution order.
        """
        # Make sure the new_order is valid. It must contain all subsystems
        # in this model.
        newset = set(new_order)
        if self._static_mode:
            subsystems = self._static_subsystems_allprocs
        else:
            subsystems = self._subsystems_allprocs
        olddict = {s.name: s for s in subsystems}
        oldset = set(olddict)

        if oldset != newset:
            msg = []

            missing = oldset - newset
            if missing:
                msg.append("%s: %s expected in subsystem order and not found." %
                           (self.pathname, sorted(missing)))

            extra = newset - oldset
            if extra:
                msg.append("%s: subsystem(s) %s found in subsystem order but don't exist." %
                           (self.pathname, sorted(extra)))

            raise ValueError('\n'.join(msg))

        # Don't allow duplicates either.
        if len(newset) < len(new_order):
            dupes = [key for key, val in iteritems(Counter(new_order)) if val > 1]
            raise ValueError("%s: Duplicate name(s) found in subsystem order list: %s" %
                             (self.pathname, sorted(dupes)))

        subsystems[:] = [olddict[name] for name in new_order]

    def _get_subsystem(self, name):
        """
        Return the system called 'name' in the current namespace.

        Parameters
        ----------
        name : str
            name of the desired system in the current namespace.

        Returns
        -------
        System or None
            System if found else None.
        """
        system = self
        for subname in name.split('.'):
            for sub in chain(system._static_subsystems_allprocs,
                             system._subsystems_allprocs):
                if sub.name == subname:
                    system = sub
                    break
            else:
                return None
        return system

    def _apply_nonlinear(self):
        """
        Compute residuals. The model is assumed to be in a scaled state.
        """
        name = self.pathname if self.pathname else 'root'

        self._transfer('nonlinear', 'fwd')
        # Apply recursion
        with Recording(name + '._apply_nonlinear', self.iter_count, self):
            for subsys in self._subsystems_myproc:
                subsys._apply_nonlinear()

    def _solve_nonlinear(self):
        """
        Compute outputs. The model is assumed to be in a scaled state.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        super(Group, self)._solve_nonlinear()

        name = self.pathname if self.pathname else 'root'

        with Recording(name + '._solve_nonlinear', self.iter_count, self):
            result = self._nonlinear_solver.solve()

        return result

    def _guess_nonlinear(self):
        """
        Provide initial guess for states.
        """
        if self._has_guess:
            for isub, sub in enumerate(self._subsystems_myproc):
                if sub._has_guess:
                    self._transfer('nonlinear', 'fwd', isub)
                    sub._guess_nonlinear()

    def _apply_linear(self, vec_names, rel_systems, mode, scope_out=None, scope_in=None):
        """
        Compute jac-vec product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.
        mode : str
            'fwd' or 'rev'.
        scope_out : set or None
            Set of absolute output names in the scope of this mat-vec product.
            If None, all are in the scope.
        scope_in : set or None
            Set of absolute input names in the scope of this mat-vec product.
            If None, all are in the scope.
        """
        name = self.pathname if self.pathname else 'root'

        vec_names = [v for v in vec_names if v in self._rel_vec_names]

        with Recording(name + '._apply_linear', self.iter_count, self):
            with self.jacobian_context() as J:
                # Use global Jacobian
                if self._owns_assembled_jac or self._views_assembled_jac or self._owns_approx_jac:
                    for vec_name in vec_names:
                        with self._matvec_context(vec_name, scope_out, scope_in, mode) as vecs:
                            d_inputs, d_outputs, d_residuals = vecs
                            J._apply(d_inputs, d_outputs, d_residuals, mode)
                # Apply recursion
                else:
                    if rel_systems is not None:
                        irrelevant_subs = [s for s in self._subsystems_myproc
                                           if s.pathname not in rel_systems]
                    if mode == 'fwd':
                        for vec_name in vec_names:
                            self._transfer(vec_name, mode)
                        if rel_systems is not None:
                            for s in irrelevant_subs:
                                # zero out dvecs of irrelevant subsystems
                                # TODO: it's not completely clear that this is
                                #       necessary in fwd mode.  I wasn't able to
                                #       produce convergence failures during testing
                                #       in fwd mode.
                                s._vectors['residual']['linear'].set_const(0.0)

                    for subsys in self._subsystems_myproc:
                        if rel_systems is None or subsys.pathname in rel_systems:
                            subsys._apply_linear(vec_names, rel_systems, mode, scope_out, scope_in)

                    if mode == 'rev':
                        for vec_name in vec_names:
                            self._transfer(vec_name, mode)
                            if rel_systems is not None:
                                for s in irrelevant_subs:
                                    # zero out dvecs of irrelevant subsystems
                                    s._vectors['output']['linear'].set_const(0.0)

    def _solve_linear(self, vec_names, mode, rel_systems):
        """
        Apply inverse jac product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        name = self.pathname if self.pathname else 'root'

        vec_names = [v for v in vec_names if v in self._rel_vec_names]

        with Recording(name + '._solve_linear', self.iter_count, self):
            result = self._linear_solver.solve(vec_names, mode, rel_systems)

        return result

    def _linearize(self, do_nl=True, do_ln=True):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        do_nl : boolean
            Flag indicating if the nonlinear solver should be linearized.
        do_ln : boolean
            Flag indicating if the linear solver should be linearized.
        """
        with self.jacobian_context() as J:

            sub_do_nl = (self._nonlinear_solver is not None) and \
                        (self._nonlinear_solver._linearize_children())
            sub_do_ln = (self._linear_solver is not None) and \
                        (self._linear_solver._linearize_children())

            # Group finite difference
            if self._owns_approx_jac:
                with self._unscaled_context(outputs=[self._outputs]):
                    for approximation in itervalues(self._approx_schemes):
                        approximation.compute_approximations(self, jac=J, deriv_type='total')

                J._update()

            else:
                # Only linearize subsystems if we aren't approximating the derivs at this level.
                for subsys in self._subsystems_myproc:
                    subsys._linearize(do_nl=sub_do_nl, do_ln=sub_do_ln)

                # Update jacobian
                if self._owns_assembled_jac or self._views_assembled_jac:
                    J._update()

        if self._nonlinear_solver is not None and do_nl:
            self._nonlinear_solver._linearize()

        if self._linear_solver is not None and do_ln:
            self._linear_solver._linearize()

    def approx_totals(self, method='fd', **kwargs):
        """
        Approximate derivatives for a Group using the specified approximation method.

        Parameters
        ----------
        method : str
            The type of approximation that should be used. Valid options include:
            'fd': Finite Difference, 'cs': Complex Step
        **kwargs : dict
            Keyword arguments for controlling the behavior of the approximation.
        """
        self._approx_schemes = OrderedDict()
        supported_methods = {'fd': FiniteDifference,
                             'cs': ComplexStep}

        if method not in supported_methods:
            msg = 'Method "{}" is not supported, method must be one of {}'
            raise ValueError(msg.format(method, supported_methods.keys()))

        if method not in self._approx_schemes:
            self._approx_schemes[method] = supported_methods[method]()

        self._owns_approx_jac = True
        self._owns_approx_jac_meta = dict(kwargs)

    def _setup_jacobians(self, jacobian=None, recurse=True):
        """
        Set and populate jacobians down through the system tree.

        In <Group>, we only need to prepare for Group finite difference. However, to be efficient,
        we need to find the minimum set of inputs and outputs to approximate.

        Parameters
        ----------
        jacobian : <AssembledJacobian> or None
            The global jacobian to populate for this system.
        recurse : bool
            Whether to call this method in subsystems.
        """
        self._jacobian_changed = False

        # Group finite difference or complex step.
        # TODO: Does this work under or over an AssembledJacobian (and does that make sense)
        if self._owns_approx_jac:
            method = list(self._approx_schemes.keys())[0]
            approx = self._approx_schemes[method]
            pro2abs = self._var_allprocs_prom2abs_list

            if self._owns_approx_of:
                of = self._owns_approx_of
            else:
                of = set(var[0] for var in pro2abs['output'].values())

            if self._owns_approx_wrt:
                candidate_wrt = self._owns_approx_wrt
            else:
                candidate_wrt = list(var[0] for var in pro2abs['input'].values())

            from openmdao.core.indepvarcomp import IndepVarComp
            wrt = set()
            ivc = set()
            for var in candidate_wrt:

                # Weed out inputs connected to anything inside our system unless the source is an
                # indepvarcomp.
                if var in self._conn_abs_in2out:
                    src = self._conn_abs_in2out[var]
                    compname = src.rsplit('.', 1)[0]
                    comp = self._get_subsystem(compname)
                    if isinstance(comp, IndepVarComp):
                        wrt.add(src)
                        ivc.add(src)
                else:
                    wrt.add(var)

            with self.jacobian_context() as J:
                for key in product(of, wrt.union(of)):
                    if key in self._subjacs_info:
                        meta = self._subjacs_info[key]
                    else:
                        meta = SUBJAC_META_DEFAULTS.copy()

                    meta['method'] = method
                    if key[0] == key[1]:
                        size = self._var_allprocs_abs2meta[key[0]]['size']
                        meta['rows'] = meta['cols'] = np.arange(size)
                        meta['value'] = np.ones(size)

                    # A group under approximation needs all keys from below, so set dependent to
                    # True.
                    # TODO: Maybe just need a subset of keys (those that go to the boundaries.)
                    meta['dependent'] = True

                    meta.update(self._owns_approx_jac_meta)
                    self._subjacs_info[key] = meta

                    # Create Jacobian stub for every key pair
                    J._set_partials_meta(key, meta)

                    # Create approximations, but only for the ones we need.
                    if meta['dependent']:

                        # Skip indepvarcomp res wrt other srcs
                        if key[0] in ivc:
                            continue

                        # Skip explicit res wrt outputs
                        if key[1] in of and key[1] not in ivc:

                            # Support for specifying a desvar as an obj/con.
                            if key[1] not in wrt or key[0] == key[1]:
                                continue

                        approx.add_approximation(key, meta)

            approx._init_approximations()

            self._jacobian._system = self
            self._jacobian._initialize()

        super(Group, self)._setup_jacobians(jacobian, recurse)

    def compute_sys_graph(self, comps_only=False):
        """
        Compute a dependency graph for subsystems in this group.

        Variable connection information is stored in each edge of
        the system graph.

        Parameters
        ----------
        comps_only : bool (False)
            If True, return a graph of all components within this group
            or any of its descendants. No sub-groups will be included. Otherwise,
            a graph containing only direct children (both Components and Groups)
            of this group will be returned.

        Returns
        -------
        DiGraph
            A directed graph containing names of subsystems and their connections.
        """
        input_srcs = self._conn_global_abs_in2out
        glen = len(self.pathname.split('.')) if self.pathname else 0
        graph = nx.DiGraph()

        # add all systems as nodes in the graph so they'll be there even if
        # unconnected.
        if comps_only:
            graph.add_nodes_from(s.pathname for s in
                                 self.system_iter(recurse=True, typ=Component))
        else:
            graph.add_nodes_from(s.pathname for s in self._subsystems_allprocs)

        edge_data = defaultdict(lambda: defaultdict(list))

        for in_abs, src_abs in iteritems(input_srcs):
            if src_abs is not None:
                if comps_only:
                    src = src_abs.rsplit('.', 1)[0]
                    tgt = in_abs.rsplit('.', 1)[0]
                else:
                    src = src_abs.split('.')[glen]
                    tgt = in_abs.split('.')[glen]

                # store var connection data in each system to system edge for later
                # use in relevance calculation.
                edge_data[(src, tgt)][src_abs].append(in_abs)

        for key in edge_data:
            src_sys, tgt_sys = key
            graph.add_edge(src_sys, tgt_sys, conns=edge_data[key])

        return graph


def get_relevant_vars(connections, desvars, responses, mode):
    """
    Find all relevant vars between desvars and responses.

    Both vars are assumed to be outputs (either design vars or responses).

    Parameters
    ----------
    connections : dict
        Mapping of inputs and their connected sources.
    desvars : list of str
        Names of design variables.
    responses : list of str
        Names of response variables.
    mode : str
        Direction of derivatives, either 'fwd' or 'rev'.

    Returns
    -------
    dict
        Dict of ({'outputs': dep_outputs, 'inputs': dep_inputs, dep_systems)
        keyed by design vars and responses.
    """
    relevant = defaultdict(dict)
    cache = {}
    fwd = mode == 'fwd'

    # Create a hybrid graph with components and all connected vars.  If a var is connected,
    # also connect it to its corresponding component.
    graph = nx.DiGraph()
    for tgt, src in iteritems(connections):
        if src not in graph:
            graph.add_node(src, type_='out')
        graph.add_node(tgt, type_='in')

        src_sys = src.rsplit('.', 1)[0]
        graph.add_edge(src_sys, src)

        tgt_sys = tgt.rsplit('.', 1)[0]
        graph.add_edge(tgt, tgt_sys)

        graph.add_edge(src, tgt)

    for dv in desvars:
        if dv not in graph:
            graph.add_node(dv, type_='out')
            system = dv.rsplit('.', 1)[0]
            graph.add_edge(system, dv)

    for res in responses:
        if res not in graph:
            graph.add_node(res, type_='out')
            system = res.rsplit('.', 1)[0]
            graph.add_edge(system, res)

    nodes = graph.nodes
    grev = graph.reverse(copy=False)

    for desvar in desvars:
        dv = (desvar, 'dv')
        if dv not in cache:
            cache[dv] = set(all_connected_nodes(graph, desvar))

        for response in responses:
            res = (response, 'r')
            if res not in cache:
                cache[res] = set(all_connected_nodes(grev, response))

            common = cache[dv].intersection(cache[res])

            if common:
                input_deps = set()
                output_deps = set()
                sys_deps = set()
                for node in common:
                    if 'type_' in nodes[node]:
                        typ = nodes[node]['type_']
                        if typ == 'in':  # input var
                            input_deps.add(node)
                            system = node.rsplit('.', 1)[0]
                            if system not in sys_deps:
                                sys_deps.update(all_ancestors(system))
                        else:  # output var
                            output_deps.add(node)
                            system = node.rsplit('.', 1)[0]
                            if system not in sys_deps:
                                sys_deps.update(all_ancestors(system))

            elif desvar == response:
                input_deps = set()
                output_deps = set([response])
                sys_deps = set(all_ancestors(desvar.rsplit('.', 1)[0]))

            if common or desvar == response:
                if fwd:
                    relevant[desvar][response] = ({'input': input_deps,
                                                   'output': output_deps}, sys_deps)
                else:  # rev
                    relevant[response][desvar] = ({'input': input_deps,
                                                   'output': output_deps}, sys_deps)

                sys_deps.add('')  # top level Group is always relevant

    if fwd:
        inputs, outputs = desvars, responses
    else:
        inputs, outputs = responses, desvars

    # now calculate dependencies between each VOI and all other VOIs of the
    # other type, e.g for each input VOI wrt all output VOIs.  This is only
    # done for design vars in fwd mode or responses in rev mode.
    for inp in inputs:
        relinp = relevant[inp]
        if relinp:
            total_inps = set()
            total_outs = set()
            total_systems = set()
            for out in outputs:
                if out in relinp:
                    dct, systems = relinp[out]
                    total_inps.update(dct['input'])
                    total_outs.update(dct['output'])
                    total_systems.update(systems)
            relinp['@all'] = ({'input': total_inps, 'output': total_outs},
                              total_systems)
        else:
            relinp['@all'] = ({'input': set(), 'output': set()}, set())

    relevant['linear'] = {'@all': ({'input': ContainsAll(), 'output': ContainsAll()},
                                   ContainsAll())}
    relevant['nonlinear'] = relevant['linear']

    return relevant
