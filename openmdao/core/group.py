"""Define the Group class."""
from __future__ import division

from collections import Iterable, Counter, OrderedDict, defaultdict
from itertools import product, chain
import warnings

from six import iteritems, string_types, itervalues
from six.moves import range

import numpy as np
import networkx as nx

from openmdao.approximation_schemes.complex_step import ComplexStep
from openmdao.approximation_schemes.finite_difference import FiniteDifference
from openmdao.core.system import System
from openmdao.core.component import Component
from openmdao.jacobians.assembled_jacobian import SUBJAC_META_DEFAULTS
from openmdao.proc_allocators.proc_allocator import ProcAllocationError
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.nonlinear.nonlinear_runonce import NonLinearRunOnce
from openmdao.solvers.linear.linear_runonce import LinearRunOnce
from openmdao.utils.array_utils import convert_neg
from openmdao.utils.general_utils import warn_deprecation
from openmdao.utils.units import is_compatible

# regex to check for valid names.
import re
namecheck_rgx = re.compile('[a-zA-Z][_a-zA-Z0-9]*')


class Group(System):
    """
    Class used to group systems together; instantiate or inherit.
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
            self._nonlinear_solver = NonLinearRunOnce()
        if not self._linear_solver:
            self._linear_solver = LinearRunOnce()

    def setup(self):
        """
        Add subsystems to this group.

        Available attributes:
            name
            pathname
            comm
            metadata
        """
        pass

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
        self._subsystems_proc_range = subsystems_proc_range = []

        self._subsystems_allprocs = []
        self._manual_connections = {}
        self._design_vars = {}
        self._responses = {}

        self._static_mode = False
        self._subsystems_allprocs.extend(self._static_subsystems_allprocs)
        self._manual_connections.update(self._static_manual_connections)
        self._design_vars.update(self._static_design_vars)
        self._responses.update(self._static_responses)
        self.setup()
        self._static_mode = True

        req_procs = [s.get_req_procs() for s in self._subsystems_allprocs]
        # Call the load balancing algorithm
        try:
            sub_inds, sub_comm, sub_proc_range = self._mpi_proc_allocator(req_procs, comm)
        except ProcAllocationError as err:
            raise RuntimeError("subsystem %s requested %d processes "
                               "but got %d" %
                               (self._subsystems_allprocs[err.sub_idx].pathname,
                                err.requested, err.remaining))

        # Define local subsystems
        self._subsystems_myproc = [self._subsystems_allprocs[ind]
                                   for ind in sub_inds]
        self._subsystems_myproc_inds = sub_inds

        # Compute _subsystems_proc_range
        for subsys in self._subsystems_myproc:
            subsystems_proc_range.append(sub_proc_range)

        # Perform recursion
        for subsys in self._subsystems_myproc:
            if self.pathname is not '':
                sub_pathname = '.'.join((self.pathname, subsys.name))
            else:
                sub_pathname = subsys.name

            subsys._setup_procs(sub_pathname, sub_comm)

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
        for type_ in ['input', 'output']:
            num_var[type_] = np.sum(
                [subsys._num_var[type_] for subsys in self._subsystems_myproc])

            for subsys in self._subsystems_myproc:
                for set_name, num in iteritems(subsys._num_var_byset[type_]):
                    if set_name in num_var_byset[type_]:
                        num_var_byset[type_][set_name] += num
                    else:
                        num_var_byset[type_][set_name] = num

        # If running in parallel, allgather
        if self.comm.size > 1:
            # Perform a single allgather
            if self._subsystems_myproc[0].comm.rank == 0:
                raw = (num_var, num_var_byset)
            else:
                raw = ({'input': 0, 'output': 0}, {'input': {}, 'output': {}})
            gathered = self.comm.allgather(raw)

            # Empty the dictionaries
            for type_ in ['input', 'output']:
                num_var[type_] = 0
                num_var_byset[type_] = {}

            # Process the gathered data and update the dictionaries
            for myproc_num_var, myproc_num_var_byset in gathered:
                for type_ in ['input', 'output']:
                    num_var[type_] += myproc_num_var[type_]
                    for set_name, num in iteritems(myproc_num_var_byset[type_]):
                        if set_name in num_var_byset[type_]:
                            num_var_byset[type_][set_name] += num
                        else:
                            num_var_byset[type_][set_name] = num

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
        subsystems_var_range = self._subsystems_var_range
        subsystems_var_range_byset = self._subsystems_var_range_byset

        nsub_allprocs = len(self._subsystems_allprocs)

        # Here, we count the number of variables (total and by varset) in each subsystem.
        # We do this so that we can compute the offset when we recurse into each subsystem.
        allprocs_counters = {
            type_: np.zeros(nsub_allprocs, int) for type_ in ['input', 'output']}
        allprocs_counters_byset = {
            type_: np.zeros((nsub_allprocs, len(set2iset[type_])), int)
            for type_ in ['input', 'output']}

        # First compute these on one processor for each subsystem
        for type_ in ['input', 'output']:
            for subsys, isub in zip(self._subsystems_myproc, self._subsystems_myproc_inds):
                if subsys.comm.rank == 0:
                    allprocs_counters[type_][isub] = subsys._num_var[type_]
                    for set_name in subsys._num_var_byset[type_]:
                        iset = set2iset[type_][set_name]
                        allprocs_counters_byset[type_][isub, iset] = \
                            subsys._num_var_byset[type_][set_name]

        # If running in parallel, allgather
        if self.comm.size > 1:
            raw = (allprocs_counters, allprocs_counters_byset)
            gathered = self.comm.allgather(raw)

            allprocs_counters = {
                type_: np.zeros(nsub_allprocs, int) for type_ in ['input', 'output']}
            allprocs_counters_byset = {
                type_: np.zeros((nsub_allprocs, len(set2iset[type_])), int)
                for type_ in ['input', 'output']}

            for myproc_counters, myproc_counters_byset in gathered:
                for type_ in ['input', 'output']:
                    allprocs_counters[type_] += myproc_counters[type_]
                    allprocs_counters_byset[type_] += myproc_counters_byset[type_]

        # Compute _subsystems_var_range, _subsystems_var_range_byset
        for type_ in ['input', 'output']:
            subsystems_var_range[type_] = []
            subsystems_var_range_byset[type_] = {set_name: [] for set_name in set2iset[type_]}

        for subsys, isub in zip(self._subsystems_myproc, self._subsystems_myproc_inds):
            for type_ in ['input', 'output']:
                subsystems_var_range[type_].append((
                    np.sum(allprocs_counters[type_][:isub]),
                    np.sum(allprocs_counters[type_][:isub + 1])))
                for set_name in set2iset[type_]:
                    iset = set2iset[type_][set_name]
                    subsystems_var_range_byset[type_][set_name].append((
                        np.sum(allprocs_counters_byset[type_][:isub, iset]),
                        np.sum(allprocs_counters_byset[type_][:isub + 1, iset])))

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

        for subsys in self._subsystems_myproc:
            var_maps = subsys._get_maps(subsys._var_allprocs_prom2abs_list)

            for type_ in ['input', 'output']:

                # Assemble abs_names and allprocs_abs_names
                allprocs_abs_names[type_].extend(subsys._var_allprocs_abs_names[type_])
                abs_names[type_].extend(subsys._var_abs_names[type_])

                # Assemble allprocs_abs2meta and abs2meta
                allprocs_abs2meta[type_].update(subsys._var_allprocs_abs2meta[type_])
                abs2meta[type_].update(subsys._var_abs2meta[type_])

                # Assemble abs2prom
                for abs_name in subsys._var_abs_names[type_]:
                    sub_prom_name = subsys._var_abs2prom[type_][abs_name]
                    prom_name = var_maps[type_][sub_prom_name]

                    abs2prom[type_][abs_name] = prom_name

                # Assemble allprocs_prom2abs_list
                sub_allprocs_prom2abs_list_t = subsys._var_allprocs_prom2abs_list[type_]
                for sub_prom_name in sub_allprocs_prom2abs_list_t:
                    prom_name = var_maps[type_][sub_prom_name]

                    for abs_name in sub_allprocs_prom2abs_list_t[sub_prom_name]:
                        if prom_name not in allprocs_prom2abs_list[type_]:
                            allprocs_prom2abs_list[type_][prom_name] = [abs_name]
                        else:
                            allprocs_prom2abs_list[type_][prom_name].append(abs_name)

        for prom_name, abs_list in iteritems(allprocs_prom2abs_list['output']):
            if len(abs_list) > 1:
                raise RuntimeError("Output name '%s' refers to "
                                   "multiple outputs: %s." %
                                   (prom_name, sorted(abs_list)))

        # If running in parallel, allgather
        if self.comm.size > 1:
            if self._subsystems_myproc[0].comm.rank == 0:
                raw = (allprocs_abs_names, allprocs_prom2abs_list, allprocs_abs2meta)
            else:
                raw = (
                    {'input': [], 'output': []},
                    {'input': {}, 'output': {}},
                    {'input': {}, 'output': {}})
            gathered = self.comm.allgather(raw)

            for type_ in ['input', 'output']:
                allprocs_abs_names[type_] = []
                allprocs_prom2abs_list[type_] = {}

            for myproc_abs_names, myproc_prom2abs_list, myproc_abs2meta in gathered:

                for type_ in ['input', 'output']:

                    # Assemble in parallel allprocs_abs_names
                    allprocs_abs_names[type_].extend(myproc_abs_names[type_])

                    # Assemble in parallel allprocs_abs2meta
                    allprocs_abs2meta[type_].update(myproc_abs2meta[type_])

                    # Assemble in parallel allprocs_prom2abs_list
                    for prom_name, abs_names_list in iteritems(myproc_prom2abs_list[type_]):
                        if prom_name not in allprocs_prom2abs_list[type_]:
                            allprocs_prom2abs_list[type_][prom_name] = abs_names_list
                        else:
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
        sizes = self._var_sizes
        sizes_byset = self._var_sizes_byset

        iproc = self.comm.rank
        nproc = self.comm.size

        set2iset = self._var_set2iset
        subsystems_proc_range = self._subsystems_proc_range
        subsystems_var_range = self._subsystems_var_range
        subsystems_var_range_byset = self._subsystems_var_range_byset

        # Recursion
        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_var_sizes(recurse)

        # Compute _var_sizes
        for type_ in ['input', 'output']:

            sizes[type_] = np.zeros((nproc, self._num_var[type_]), int)
            for set_name in set2iset[type_]:
                sizes_byset[type_][set_name] = np.zeros(
                    (nproc, self._num_var_byset[type_][set_name]), int)

            for ind, subsys in enumerate(self._subsystems_myproc):
                proc_slice = slice(*subsystems_proc_range[ind])
                var_slice = slice(*subsystems_var_range[type_][ind])
                sizes[type_][proc_slice, var_slice] = subsys._var_sizes[type_]

                for set_name in set2iset[type_]:
                    var_slice = slice(*subsystems_var_range_byset[type_][set_name][ind])
                    sizes_byset[type_][set_name][proc_slice, var_slice] = \
                        subsys._var_sizes_byset[type_][set_name]

        # If parallel, all gather
        if self.comm.size > 1:
            for type_ in ['input', 'output']:
                self.comm.Allgather(sizes[type_][iproc, :], sizes[type_])
                for set_name in set2iset[type_]:
                    self.comm.Allgather(
                        sizes_byset[type_][set_name][iproc, :], sizes_byset[type_][set_name])

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
        abs2meta_in = self._var_abs2meta['input']
        abs2meta_out = self._var_abs2meta['output']
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
        for prom_in, (prom_out, src_indices) in iteritems(self._manual_connections):

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

                if src_indices is not None and abs_in in abs2meta_in:
                    meta = abs2meta_in[abs_in]
                    if meta['src_indices'] is not None:
                        raise RuntimeError("%s: src_indices has been defined "
                                           "in both connect('%s', '%s') "
                                           "and add_input('%s', ...)." %
                                           (self.pathname, prom_out,
                                            prom_in, prom_in))
                    meta['src_indices'] = np.atleast_1d(src_indices)

                abs_in2out[abs_in] = abs_out

                # if connection is contained in a subgroup, add to conns to pass down to subsystems.
                if inparts[:nparts + 1] == outparts[:nparts + 1]:
                    new_conns[inparts[nparts]][abs_in] = abs_out

        # Now that both implicit & explicit connections have been added,
        # check unit/shape compatibility, but only for connections that are
        # either owned by (implicit) or declared by (explicit) this Group.
        # This way, we don't repeat the error checking in multiple groups
        allprocs_abs2meta_out = self._var_allprocs_abs2meta['output']
        allprocs_abs2meta_in = self._var_allprocs_abs2meta['input']

        for abs_in, abs_out in iteritems(abs_in2out):
            # check unit compatibility
            out_units = allprocs_abs2meta_out[abs_out]['units']
            in_units = allprocs_abs2meta_in[abs_in]['units']

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
            if abs_in in abs2meta_in and abs_out in abs2meta_out:
                # get output shape from allprocs meta dict, since it may
                # be distributed (we want global shape)
                out_shape = allprocs_abs2meta_out[abs_out]['global_shape']
                # get input shape and src_indices from the local meta dict
                # (input is always local)
                in_shape = abs2meta_in[abs_in]['shape']
                src_indices = abs2meta_in[abs_in].get('src_indices')
                prom_out = self._var_abs2prom['output'][abs_out]
                prom_in = self._var_abs2prom['input'][abs_in]

                if src_indices is None and out_shape != in_shape:
                    msg = ("The source and target shapes do not match"
                           " for the connection '%s' to '%s' in Group"
                           " '%s'.  Expected %s but got %s.")
                    raise ValueError(msg % (prom_out, prom_in,
                                            self.pathname,
                                            in_shape, out_shape))

                if src_indices is not None:
                    src_indices = np.atleast_1d(src_indices)

                    if len(src_indices.shape) < len(in_shape):
                        raise ValueError("Unexpected indices shape: %s vs %s" %
                                         (src_indices.shape, in_shape))

                    # initial dimensions of indices shape must be same shape as target
                    for idx_d, inp_d in zip(src_indices.shape, in_shape):
                        if idx_d != inp_d:
                            msg = ("The source indices %s do not specify a "
                                   "valid shape for the connection '%s' to "
                                   "'%s' in Group '%s'. The target shape is "
                                   "%s but indices are %s.")
                            raise ValueError(msg % (src_indices, prom_out, prom_in,
                                                    self.pathname,
                                                    in_shape, src_indices.shape))

                    # remaining dimension of indices must match shape of source
                    if len(src_indices.shape) > len(in_shape):
                        source_dimensions = src_indices.shape[len(in_shape)]
                        if source_dimensions != len(out_shape):
                            msg = ("The source indices %s do not specify a "
                                   "valid shape for the connection '%s' to "
                                   "'%s' in Group '%s'. The source has %d "
                                   "dimensions but the indices expect %d.")
                            raise ValueError(msg % (src_indices,
                                                    prom_out, prom_in,
                                                    self.pathname,
                                                    len(out_shape),
                                                    source_dimensions))

                        # check all indices are in range of the source dimensions
                        for d in range(source_dimensions):
                            d_size = out_shape[d]
                            for i in src_indices[..., d].flat:
                                if abs(i) >= d_size:
                                    msg = ("The source indices do not specify "
                                           "a valid index for the connection "
                                           "'%s' to '%s' in Group '%s'. Index "
                                           "'%d' is out of range for source " 
                                           "dimension %d.")
                                    raise ValueError(msg % (prom_out, prom_in,
                                                            self.pathname, i, d_size))

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
        global_abs_in2out.update(abs_in2out)

        for subsys in self._subsystems_myproc:
            global_abs_in2out.update(subsys._conn_global_abs_in2out)

        # If running in parallel, allgather
        if self.comm.size > 1:
            if self._subsystems_myproc[0].comm.rank == 0:
                raw = global_abs_in2out
            else:
                raw = {}
            gathered = self.comm.allgather(raw)

            for myproc_global_abs_in2out in gathered:
                global_abs_in2out.update(myproc_global_abs_in2out)

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

        for abs_in, abs_out in iteritems(global_abs_in2out):
            # First, check that this system owns both the input and output.
            if abs_in[:len(pathname)] == pathname and abs_out[:len(pathname)] == pathname:
                # Second, check that they are in different subsystems of this system.
                out_subsys = abs_out[path_len:].split('.', 1)[0]
                in_subsys = abs_in[path_len:].split('.', 1)[0]
                if out_subsys != in_subsys:
                    abs_in2out[abs_in] = abs_out

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

        subsystems_var_range = self._subsystems_var_range
        subsystems_var_range_byset = self._subsystems_var_range_byset

        for ind, subsys in enumerate(self._subsystems_myproc):
            sub_ext_num_vars = {}
            sub_ext_sizes = {}
            sub_ext_num_vars_byset = {}
            sub_ext_sizes_byset = {}

            for type_ in ['input', 'output']:
                num = self._num_var[type_]
                idx1, idx2 = subsystems_var_range[type_][ind]
                size1 = np.sum(self._var_sizes[type_][iproc, :idx1])
                size2 = np.sum(self._var_sizes[type_][iproc, idx2:])

                sub_ext_num_vars[type_] = (
                    ext_num_vars[type_][0] + idx1,
                    ext_num_vars[type_][1] + num - idx2,
                )
                sub_ext_sizes[type_] = (
                    ext_sizes[type_][0] + size1,
                    ext_sizes[type_][1] + size2,
                )

                sub_ext_sizes_byset[type_] = {}
                sub_ext_num_vars_byset[type_] = {}
                for set_name in self._var_set2iset[type_]:
                    num = self._num_var_byset[type_][set_name]
                    idx1, idx2 = subsystems_var_range_byset[type_][set_name][ind]
                    size1 = np.sum(self._var_sizes_byset[type_][set_name][iproc, :idx1])
                    size2 = np.sum(self._var_sizes_byset[type_][set_name][iproc, idx2:])

                    sub_ext_num_vars_byset[type_][set_name] = (
                        ext_num_vars_byset[type_][set_name][0] + idx1,
                        ext_num_vars_byset[type_][set_name][1] + num - idx2,
                    )
                    sub_ext_sizes_byset[type_][set_name] = (
                        ext_sizes_byset[type_][set_name][0] + size1,
                        ext_sizes_byset[type_][set_name][1] + size2,
                    )

            subsys._setup_global(
                sub_ext_num_vars, sub_ext_num_vars_byset,
                sub_ext_sizes, sub_ext_sizes_byset,
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

        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_transfers(recurse)

        # Pre-compute map from abs_names to the index of the containing subsystem
        abs2isub = {'input': {}, 'output': {}}
        for subsys, isub in zip(self._subsystems_myproc, self._subsystems_myproc_inds):
            for type_ in ['input', 'output']:
                for abs_name in subsys._var_allprocs_abs_names[type_]:
                    abs2isub[type_][abs_name] = isub

        # Initialize empty lists for the transfer indices
        nsub_allprocs = len(self._subsystems_allprocs)
        xfer_in = {}
        xfer_out = {}
        fwd_xfer_in = [{} for i in range(nsub_allprocs)]
        fwd_xfer_out = [{} for i in range(nsub_allprocs)]
        rev_xfer_in = [{} for i in range(nsub_allprocs)]
        rev_xfer_out = [{} for i in range(nsub_allprocs)]
        for set_name_in in self._var_set2iset['input']:
            for set_name_out in self._var_set2iset['output']:
                key = (set_name_in, set_name_out)
                xfer_in[key] = []
                xfer_out[key] = []
                for isub in range(nsub_allprocs):
                    fwd_xfer_in[isub][key] = []
                    fwd_xfer_out[isub][key] = []
                    rev_xfer_in[isub][key] = []
                    rev_xfer_out[isub][key] = []

        abs2meta_in = self._var_abs2meta['input']
        allprocs_abs2meta_out = self._var_allprocs_abs2meta['output']
        allprocs_abs2idx_in = self._var_allprocs_abs2idx['input']
        allprocs_abs2idx_out = self._var_allprocs_abs2idx['output']
        allprocs_abs2idx_byset_in = self._var_allprocs_abs2idx_byset['input']
        allprocs_abs2idx_byset_out = self._var_allprocs_abs2idx_byset['output']
        sizes_byset_in = self._var_sizes_byset['input']
        sizes_byset_out = self._var_sizes_byset['output']
        set2iset_in = self._var_set2iset['input']
        set2iset_out = self._var_set2iset['output']

        # Loop through all explicit / implicit connections owned by this system
        for abs_in, abs_out in iteritems(self._conn_abs_in2out):

            # Only continue if the input exists on this processor
            if abs_in in abs2meta_in:

                # Get meta
                meta_in = abs2meta_in[abs_in]
                meta_out = allprocs_abs2meta_out[abs_out]

                # Get varset info
                set_name_in = meta_in['var_set']
                set_name_out = meta_out['var_set']
                idx_byset_in = allprocs_abs2idx_byset_in[abs_in]
                idx_byset_out = allprocs_abs2idx_byset_out[abs_out]

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
                    src_indices = np.arange(np.prod(shape_in), dtype=int)
                elif src_indices.ndim == 1:
                    src_indices = convert_neg(src_indices, global_size_out)
                else:
                    if len(shape_out) == 1:
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
                ind1 += np.sum(sizes_in[iproc, :idx_byset_in])
                ind2 += np.sum(sizes_in[iproc, :idx_byset_in + 1])
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

        def merge(indices_list):
            if len(indices_list) > 0:
                return np.concatenate(indices_list)
            else:
                return np.array([], int)

        for set_name_in in self._var_set2iset['input']:
            for set_name_out in self._var_set2iset['output']:
                key = (set_name_in, set_name_out)
                xfer_in[key] = merge(xfer_in[key])
                xfer_out[key] = merge(xfer_out[key])
                for isub in range(nsub_allprocs):
                    fwd_xfer_in[isub][key] = merge(fwd_xfer_in[isub][key])
                    fwd_xfer_out[isub][key] = merge(fwd_xfer_out[isub][key])
                    rev_xfer_in[isub][key] = merge(rev_xfer_in[isub][key])
                    rev_xfer_out[isub][key] = merge(rev_xfer_out[isub][key])

        transfers = self._transfers
        vectors = self._vectors
        for vec_name, out_vec in iteritems(vectors['output']):
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
                      promotes_inputs=None, promotes_outputs=None):
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

        Returns
        -------
        <System>
            the subsystem that was passed in. This is returned to
            enable users to instantiate and add a subsystem at the
            same time, and get the reference back.
        """
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

        setattr(self, name, subsys)

        return subsys

    def connect(self, src_name, tgt_name, src_indices=None):
        """
        Connect source src_name to target tgt_name in this namespace.

        Parameters
        ----------
        src_name : str
            name of the source variable to connect
        tgt_name : str or [str, ... ] or (str, ...)
            name of the target variable(s) to connect
        src_indices : collection of int optional
            When an input variable connects to some subset of an array output
            variable, you can specify which global indices of the source to be
            transferred to the input here.  Negative indices are supported.
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
                self.connect(src_name, name, src_indices)
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

        manual_connections[tgt_name] = (src_name, src_indices)

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

    def get_req_procs(self):
        """
        Return the min and max MPI processes usable by this Group.

        Returns
        -------
        tuple : (int, int or None)
            A tuple of the form (min_procs, max_procs), indicating the min
            and max processors usable by this <Group>.  max_procs can be None,
            indicating all available procs can be used.
        """
        # NOTE: this must only be called BEFORE _subsystems_allprocs and
        # _static_subsystems_allprocs have been combined, else we may
        # double count some subsystems and mess up the proc allocation.

        if self._static_subsystems_allprocs or self._subsystems_allprocs:
            if self._mpi_proc_allocator.parallel:
                # for a parallel group, we add up all of the required procs
                min_procs, max_procs = 0, 0

                for sub in chain(self._static_subsystems_allprocs,
                                 self._subsystems_allprocs):
                    sub_min, sub_max = sub.get_req_procs()
                    if sub_min > min_procs:
                        min_procs = sub_min
                    if max_procs is not None:
                        if sub_max is None:
                            max_procs = None
                        else:
                            max_procs += sub_max

                if min_procs == 0:
                    min_procs = 1

                if max_procs == 0:
                    max_procs = 1

                return (min_procs, max_procs)
            else:
                # for a serial group, we take the max required procs
                min_procs, max_procs = 1, 1

                for sub in chain(self._static_subsystems_allprocs,
                                 self._subsystems_allprocs):
                    sub_min, sub_max = sub.get_req_procs()
                    min_procs = max(min_procs, sub_min)
                    if max_procs is not None:
                        if sub_max is None:
                            max_procs = None
                        else:
                            max_procs = max(max_procs, sub_max)

                return (min_procs, max_procs)
        else:
            return super(Group, self).get_req_procs()

    def get_subsystem(self, name):
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

        # Execute guess_nonlinear if specified.
        # We need to call this early enough so that any solver that needs initial guesses has
        # them.
        # TODO: It is pointless to run this ahead of non-iterative solvers.
        for sub in self.system_iter(recurse=True):
            if hasattr(sub, 'guess_nonlinear'):
                with sub._unscaled_context(outputs=[sub._outputs], residuals=[sub._residuals]):
                    sub.guess_nonlinear(sub._inputs, sub._outputs, sub._residuals)

        with Recording(name + '._solve_nonlinear', self.iter_count, self):
            result = self._nonlinear_solver.solve()

        return result

    def _apply_linear(self, vec_names, mode, scope_out=None, scope_in=None):
        """
        Compute jac-vec product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
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
                    if mode == 'fwd':
                        for vec_name in vec_names:
                            self._transfer(vec_name, mode)

                    for subsys in self._subsystems_myproc:
                        subsys._apply_linear(vec_names, mode, scope_out, scope_in)

                    if mode == 'rev':
                        for vec_name in vec_names:
                            self._transfer(vec_name, mode)

    def _solve_linear(self, vec_names, mode):
        """
        Apply inverse jac product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.

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

        with Recording(name + '._solve_linear', self.iter_count, self):
            result = self._linear_solver.solve(vec_names, mode)

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

            for subsys in self._subsystems_myproc:
                subsys._linearize(do_nl=sub_do_nl, do_ln=sub_do_ln)

            # Group finite difference
            if self._owns_approx_jac:
                with self._unscaled_context(outputs=[self._outputs]):
                    for approximation in itervalues(self._approx_schemes):
                        approximation.compute_approximations(self, jac=J, deriv_type='total')

                J._update()

            # Update jacobian
            elif self._owns_assembled_jac or self._views_assembled_jac:
                J._update()

        if self._nonlinear_solver is not None and do_nl:
            self._nonlinear_solver._linearize()

        if self._linear_solver is not None and do_ln:
            self._linear_solver._linearize()

    def approx_total_derivs(self, method='fd', **kwargs):
        """
        Approximate derivatives for a Group using the specified approximation method.

        Parameters
        ----------
        method : str
            The type of approximation that should be used. Valid options include:
                - 'fd': Finite Difference, 'cs': Complex Step
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
            ivc = []
            for var in candidate_wrt:
                src = self._conn_abs_in2out.get(var)

                if src is None:
                    wrt.add(var)

                # Weed out inputs connected to anything inside our system unless the source is an
                # indepvarcomp.
                else:
                    compname = src.rsplit('.', 1)[0]
                    comp = self.get_subsystem(compname)
                    if isinstance(comp, IndepVarComp):
                        wrt.add(src)
                        ivc.append(src)

            with self.jacobian_context() as J:
                for key in product(of, wrt.union(of)):
                    meta_changes = {
                        'method': method,
                    }
                    if key[0] == key[1]:
                        size = self._outputs._views_flat[key[0]].shape[0]
                        meta_changes['rows'] = np.arange(size)
                        meta_changes['cols'] = np.arange(size)
                        meta_changes['value'] = np.ones(size)

                    meta = self._subjacs_info.get(key, SUBJAC_META_DEFAULTS.copy())
                    meta.update(meta_changes)
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
                            continue

                        approx.add_approximation(key, meta)

            approx._init_approximations()

            self._jacobian._system = self
            self._jacobian._initialize()

        super(Group, self)._setup_jacobians(jacobian, recurse)

    def compute_sys_graph(self, comps_only=False, save_vars=False):
        """
        Compute a dependency graph for subsystems in this group.

        Parameters
        ----------
        comps_only : bool (False)
            If True, return a graph of all components within this group
            or any of its descendants. No sub-groups will be included. Otherwise,
            a graph containing only direct children (both Components and Groups)
            of this group will be returned.

        save_vars : bool (False)
            If True, store variable connection information in each edge in
            the system graph.

        Returns
        -------
        DiGraph
            A directed graph containing names of subsystems and their connections.
        """
        input_srcs = self._conn_global_abs_in2out
        glen = len(self.pathname.split('.')) if self.pathname else 0
        graph = nx.DiGraph()

        if comps_only:
            subsystems = list(self.system_iter(recurse=True, typ=Component))
        else:
            subsystems = self._subsystems_allprocs

        if save_vars:
            edge_data = defaultdict(lambda: defaultdict(list))

        for in_abs, src_abs in iteritems(input_srcs):
            if src_abs is not None:
                if comps_only:
                    src = src_abs.rsplit('.', 1)[0]
                    tgt = in_abs.rsplit('.', 1)[0]
                else:
                    src = src_abs.split('.')[glen]
                    tgt = in_abs.split('.')[glen]

                if save_vars:
                    # store var connection data in each system to system edge for later
                    # use in relevance calculation.
                    edge_data[(src, tgt)][src_abs].append(in_abs)
                else:
                    graph.add_edge(src, tgt)

        if save_vars:
            for key in edge_data:
                src_sys, tgt_sys = key
                graph.add_edge(src_sys, tgt_sys, conns=edge_data[key])

        return graph
