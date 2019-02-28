"""Define the Group class."""
from __future__ import division

from collections import Iterable, Counter, OrderedDict, defaultdict
from itertools import product, chain
from numbers import Number
import inspect

from six import iteritems, string_types, itervalues
from six.moves import range

import numpy as np
import networkx as nx

from openmdao.jacobians.dictionary_jacobian import DictionaryJacobian
from openmdao.approximation_schemes.complex_step import ComplexStep, DEFAULT_CS_OPTIONS
from openmdao.approximation_schemes.finite_difference import FiniteDifference, DEFAULT_FD_OPTIONS
from openmdao.core.system import System, INT_DTYPE, get_relevant_vars
from openmdao.core.component import Component, _DictValues
from openmdao.proc_allocators.default_allocator import DefaultAllocator, ProcAllocationError
from openmdao.jacobians.assembled_jacobian import SUBJAC_META_DEFAULTS
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce
from openmdao.solvers.linear.linear_runonce import LinearRunOnce
from openmdao.utils.array_utils import convert_neg, array_connection_compatible
from openmdao.utils.general_utils import warn_deprecation, ContainsAll, all_ancestors, \
    simple_warning
from openmdao.utils.units import is_compatible, get_conversion
from openmdao.utils.mpi import MPI

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
    _local_system_set : set or None
        Set of pathnames of all fully local (not remote or distributed)
        direct or indirect subsystems.
    _subgroups_myproc : list
        List of local subgroups.
    _manual_connections : dict
        Dictionary of input_name: (output_name, src_indices) connections.
    _static_manual_connections : dict
        Dictionary that stores all explicit connections added outside of setup.
    _conn_abs_in2out : {'abs_in': 'abs_out'}
        Dictionary containing all explicit & implicit connections owned
        by this system only. The data is the same across all processors.
    _conn_discrete_in2out : {'abs_in': 'abs_out'}
        Dictionary containing all explicit & implicit discrete var connections owned
        by this system only. The data is the same across all processors.
    _transfers : dict of dict of Transfers
        First key is the vec_name, second key is (mode, isub) where
        mode is 'fwd' or 'rev' and isub is the subsystem index among allprocs subsystems
        or isub can be None for the full, simultaneous transfer.
    _discrete_transfers : dict of discrete transfer metadata
        Key is system pathname or None for the full, simultaneous transfer.
    _loc_subsys_map : dict
        Mapping of local subsystem names to their corresponding System.
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
        self._mpi_proc_allocator = DefaultAllocator()
        self._proc_info = {}

        super(Group, self).__init__(**kwargs)

        self._local_system_set = None
        self._subgroups_myproc = None
        self._manual_connections = {}
        self._static_manual_connections = {}
        self._conn_abs_in2out = {}
        self._conn_discrete_in2out = {}
        self._transfers = {}
        self._discrete_transfers = {}

        # TODO: we cannot set the solvers with property setters at the moment
        # because our lint check thinks that we are defining new attributes
        # called nonlinear_solver and linear_solver without documenting them.
        if not self._nonlinear_solver:
            self._nonlinear_solver = NonlinearRunOnce()
        if not self._linear_solver:
            self._linear_solver = LinearRunOnce()

    def setup(self):
        """
        Build this group.

        This method should be overidden by your Group's method. The reason for using this
        method to add subsystem is to save memory and setup time when using your Group
        while running under MPI.  This avoids the creation of systems that will not be
        used in the current process.

        You may call 'add_subsystem' to add systems to this group. You may also issue connections,
        and set the linear and nonlinear solvers for this group level. You cannot safely change
        anything on children systems; use the 'configure' method instead.

        Available attributes:
            name
            pathname
            comm
            options
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
            options
            system hieararchy with attribute access
        """
        pass

    def _get_scope(self, excl_sub=None):
        """
        Find the input and output variables that are needed for a particular matvec product.

        Parameters
        ----------
        excl_sub : <System>
            A subsystem whose variables should be excluded from the matvec product.

        Returns
        -------
        (set, set)
            Sets of output and input variables.
        """
        try:
            return self._scope_cache[excl_sub]
        except KeyError:
            pass

        if excl_sub is None:
            # All myproc outputs
            scope_out = frozenset(self._var_abs_names['output'])

            # All myproc inputs connected to an output in this system
            scope_in = frozenset(self._conn_global_abs_in2out).intersection(
                self._var_abs_names['input'])

        else:
            # All myproc outputs not in excl_sub
            scope_out = frozenset(self._var_abs_names['output']).difference(
                excl_sub._var_abs_names['output'])

            # All myproc inputs connected to an output in this system but not in excl_sub
            scope_in = set()
            for abs_in in self._var_abs_names['input']:
                if abs_in in self._conn_global_abs_in2out:
                    abs_out = self._conn_global_abs_in2out[abs_in]

                    if abs_out not in excl_sub._var_allprocs_abs2idx['linear']:
                        scope_in.add(abs_in)
            scope_in = frozenset(scope_in)

        self._scope_cache[excl_sub] = (scope_out, scope_in)
        return scope_out, scope_in

    def _compute_root_scale_factors(self):
        """
        Compute scale factors for all variables.

        Returns
        -------
        dict
            Mapping of each absolute var name to its corresponding scaling factor tuple.
        """
        scale_factors = super(Group, self)._compute_root_scale_factors()

        if self._has_input_scaling:
            abs2meta_in = self._var_abs2meta
            allprocs_meta_out = self._var_allprocs_abs2meta
            for abs_in, abs_out in iteritems(self._conn_global_abs_in2out):
                meta_out = allprocs_meta_out[abs_out]
                if abs_in not in abs2meta_in:
                    # we only perform scaling on local arrays, so skip
                    continue

                meta_in = abs2meta_in[abs_in]

                ref = meta_out['ref']
                ref0 = meta_out['ref0']

                src_indices = meta_in['src_indices']

                if src_indices is not None:
                    if not (np.isscalar(ref) and np.isscalar(ref0)):
                        # TODO: if either ref or ref0 are not scalar and the output is
                        # distributed, we need to do a scatter
                        # to obtain the values needed due to global src_indices
                        if meta_out['distributed']:
                            raise RuntimeError("vector scalers with distrib vars "
                                               "not supported yet.")

                        global_shape_out = meta_out['global_shape']
                        if src_indices.ndim != 1:
                            shape_in = meta_in['shape']
                            if len(meta_out['shape']) == 1 or shape_in == src_indices.shape:
                                src_indices = src_indices.flatten()
                                src_indices = convert_neg(src_indices, src_indices.size)
                            else:
                                entries = [list(range(x)) for x in shape_in]
                                cols = np.vstack(src_indices[i] for i in product(*entries))
                                dimidxs = [convert_neg(cols[:, i], global_shape_out[i])
                                           for i in range(cols.shape[1])]
                                src_indices = np.ravel_multi_index(dimidxs, global_shape_out)

                        ref = ref[src_indices]
                        ref0 = ref0[src_indices]

                # Compute scaling arrays for inputs using a0 and a1
                # Example:
                #   Let x, x_src, x_tgt be the dimensionless variable,
                #   variable in source units, and variable in target units, resp.
                #   x_src = a0 + a1 x
                #   x_tgt = b0 + b1 x
                #   x_tgt = g(x_src) = d0 + d1 x_src
                #   b0 + b1 x = d0 + d1 a0 + d1 a1 x
                #   b0 = d0 + d1 a0
                #   b0 = g(a0)
                #   b1 = d0 + d1 a1 - d0
                #   b1 = g(a1) - g(0)

                units_in = meta_in['units']
                units_out = meta_out['units']

                if units_in is None or units_out is None or units_in == units_out:
                    a0 = ref0
                    a1 = ref - ref0
                else:
                    factor, offset = get_conversion(units_out, units_in)
                    a0 = (ref0 + offset) * factor
                    a1 = (ref - ref0) * factor

                scale_factors[abs_in] = {
                    ('input', 'phys'): (a0, a1),
                    ('input', 'norm'): (-a0 / a1, 1.0 / a1)
                }

        return scale_factors

    def _configure(self):
        """
        Configure our model recursively to assign any children settings.

        Highest system's settings take precedence.
        """
        for subsys in self._subsystems_myproc:
            subsys._configure()

            if subsys._has_guess:
                self._has_guess = True
            if subsys.matrix_free:
                self.matrix_free = True

        self.configure()

    def _setup_procs(self, pathname, comm, mode):
        """
        Execute first phase of the setup process.

        Distribute processors, assign pathnames, and call setup on the group. This method recurses
        downward through the model.

        Parameters
        ----------
        pathname : str
            Global name of the system, including the path.
        comm : MPI.Comm or <FakeComm>
            MPI communicator object.
        mode : string
            Derivatives calculation mode, 'fwd' for forward, and 'rev' for
            reverse (adjoint). Default is 'rev'.
        """
        self.pathname = pathname

        if self._num_par_fd > 1:
            if comm.size > 1:
                if self._owns_approx_jac:
                    comm = self._setup_par_fd_procs(comm)
                else:
                    msg = "'%s': num_par_fd = %d but FD is not active." % (self.pathname,
                                                                           self._num_par_fd)
                    raise RuntimeError(msg)
            elif not MPI:
                msg = ("'%s': MPI is not active but num_par_fd = %d. No parallel finite difference "
                       "will be performed." % (self.pathname, self._num_par_fd))
                simple_warning(msg)

        self.comm = comm
        self._mode = mode

        self._subsystems_allprocs = []
        self._manual_connections = {}
        self._design_vars = OrderedDict()
        self._responses = OrderedDict()

        self._static_mode = False
        self._subsystems_allprocs.extend(self._static_subsystems_allprocs)
        self._manual_connections.update(self._static_manual_connections)
        self._design_vars.update(self._static_design_vars)
        self._responses.update(self._static_responses)

        # Call setup function for this group.
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

        self._local_system_set = set()

        # Perform recursion
        for subsys in self._subsystems_myproc:
            subsys._local_vector_class = self._local_vector_class
            subsys._distributed_vector_class = self._distributed_vector_class
            subsys.force_alloc_complex = self.force_alloc_complex
            subsys._use_derivatives = self._use_derivatives
            subsys._solver_info = self._solver_info
            subsys._recording_iter = self._recording_iter

            if self.pathname:
                subsys._setup_procs('.'.join((self.pathname, subsys.name)), sub_comm, mode)
            else:
                subsys._setup_procs(subsys.name, sub_comm, mode)

        # build a list of local subgroups to speed up later loops
        self._subgroups_myproc = [s for s in self._subsystems_myproc if isinstance(s, Group)]

        self._loc_subsys_map = {s.name: s for s in self._subsystems_myproc}

    def _check_reconf_update(self, subsys=None):
        """
        Check if any subsystem has reconfigured and if so, perform the necessary update setup.

        Parameters
        ----------
        subsys : System or None
            If not None, check only if the given subsystem has reconfigured.
        """
        if subsys is None:
            # See if any local subsystem has reconfigured
            for subsys in self._subgroups_myproc:
                if subsys._reconfigured:
                    reconf = 1
                    break
            else:
                reconf = 0
        else:
            reconf = int(subsys._reconfigured) if subsys.name in self._loc_subsys_map else 0

        # See if any subsystem on this or any other processor has configured
        if self.comm.size > 1:
            reconf = self.comm.allreduce(reconf) > 0

        if reconf:
            # Perform an update setup
            with self._unscaled_context_all():
                self.resetup('update')

            # Reset the _reconfigured attribute to False
            for subsys in self._subsystems_myproc:
                subsys._reconfigured = False

            self._reconfigured = True

    def _list_states(self):
        """
        Return list of all local states at and below this system.

        Returns
        -------
        list
            List of all states.
        """
        states = []
        for subsys in self._subsystems_myproc:
            states.extend(subsys._list_states())

        return sorted(states)

    def _list_states_allprocs(self):
        """
        Return list of all states at and below this system across all procs.

        Returns
        -------
        list
            List of all states.
        """
        if MPI:
            all_states = set()
            byproc = self.comm.allgather(self._list_states())
            for proc_states in byproc:
                all_states.update(proc_states)
        else:
            all_states = self._list_states()

        return sorted(all_states)

    def _setup_var_index_ranges(self, recurse=True):
        """
        Compute the division of variables by subsystem.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(Group, self)._setup_var_index_ranges()

        nsub_allprocs = len(self._subsystems_allprocs)

        subsystems_var_range = self._subsystems_var_range = {}

        vec_names = self._lin_rel_vec_name_list if self._use_derivatives else self._vec_names

        # First compute these on one processor for each subsystem
        for vec_name in vec_names:

            # Here, we count the number of variables in each subsystem.
            # We do this so that we can compute the offset when we recurse into each subsystem.
            allprocs_counters = {
                type_: np.zeros(nsub_allprocs, INT_DTYPE) for type_ in ['input', 'output']}

            for type_ in ['input', 'output']:
                for subsys, isub in zip(self._subsystems_myproc, self._subsystems_myproc_inds):
                    comm = subsys.comm if subsys._full_comm is None else subsys._full_comm
                    if comm.rank == 0 and vec_name in subsys._rel_vec_names:
                        allprocs_counters[type_][isub] = \
                            len(subsys._var_allprocs_relevant_names[vec_name][type_])

            # If running in parallel, allgather
            if self.comm.size > 1:
                gathered = self.comm.allgather(allprocs_counters)

                allprocs_counters = {
                    type_: np.zeros(nsub_allprocs, INT_DTYPE) for type_ in ['input', 'output']}
                for myproc_counters in gathered:
                    for type_ in ['input', 'output']:
                        allprocs_counters[type_] += myproc_counters[type_]

            # Compute _subsystems_var_range
            subsystems_var_range[vec_name] = {}

            for type_ in ['input', 'output']:
                subsystems_var_range[vec_name][type_] = {}

                for subsys, isub in zip(self._subsystems_myproc, self._subsystems_myproc_inds):
                    if vec_name not in subsys._rel_vec_names:
                        continue
                    subsystems_var_range[vec_name][type_][subsys.name] = (
                        np.sum(allprocs_counters[type_][:isub]),
                        np.sum(allprocs_counters[type_][:isub + 1]))

        if self._use_derivatives:
            subsystems_var_range['nonlinear'] = subsystems_var_range['linear']

        # Recursion
        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_var_index_ranges(recurse)

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
        allprocs_discrete = self._var_allprocs_discrete
        abs_names = self._var_abs_names
        var_discrete = self._var_discrete
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

            sub_prefix = subsys.name + '.'

            for type_ in ['input', 'output']:

                # Assemble abs_names and allprocs_abs_names
                allprocs_abs_names[type_].extend(subsys._var_allprocs_abs_names[type_])
                allprocs_discrete[type_].update({k: v for k, v in
                                                 iteritems(subsys._var_allprocs_discrete[type_])})
                abs_names[type_].extend(subsys._var_abs_names[type_])
                var_discrete[type_].update({sub_prefix + k: v for k, v in
                                            iteritems(subsys._var_discrete[type_])})

                # Assemble abs2prom
                for abs_name in subsys._var_abs_names[type_]:
                    sub_prom_name = subsys._var_abs2prom[type_][abs_name]
                    abs2prom[type_][abs_name] = var_maps[type_][sub_prom_name]

                # Assemble allprocs_prom2abs_list
                for sub_prom, sub_abs in iteritems(subsys._var_allprocs_prom2abs_list[type_]):
                    prom_name = var_maps[type_][sub_prom]
                    if prom_name not in allprocs_prom2abs_list[type_]:
                        allprocs_prom2abs_list[type_][prom_name] = []
                    allprocs_prom2abs_list[type_][prom_name].extend(sub_abs)

        for prom_name, abs_list in iteritems(allprocs_prom2abs_list['output']):
            if len(abs_list) > 1:
                raise RuntimeError("Output name '%s' refers to "
                                   "multiple outputs: %s." %
                                   (prom_name, sorted(abs_list)))

        # If running in parallel, allgather
        if self.comm.size > 1:
            mysub = self._subsystems_myproc[0] if self._subsystems_myproc else False
            if (mysub and mysub.comm.rank == 0 and (mysub._full_comm is None or
                                                    mysub._full_comm.rank == 0)):
                raw = (allprocs_abs_names, allprocs_discrete, allprocs_prom2abs_list,
                       allprocs_abs2meta, self._has_output_scaling, self._has_resid_scaling)
            else:
                raw = (
                    {'input': [], 'output': []},
                    {'input': {}, 'output': {}},
                    {'input': {}, 'output': {}},
                    {'input': {}, 'output': {}},
                    False,
                    False
                )
            gathered = self.comm.allgather(raw)

            for type_ in ['input', 'output']:
                allprocs_abs_names[type_] = []
                allprocs_prom2abs_list[type_] = OrderedDict()

            for (myproc_abs_names, myproc_discrete, myproc_prom2abs_list, myproc_abs2meta,
                 oscale, rscale) in gathered:
                self._has_output_scaling |= oscale
                self._has_resid_scaling |= rscale

                # Assemble in parallel allprocs_abs2meta
                allprocs_abs2meta.update(myproc_abs2meta)

                for type_ in ['input', 'output']:

                    # Assemble in parallel allprocs_abs_names
                    allprocs_abs_names[type_].extend(myproc_abs_names[type_])
                    allprocs_discrete[type_].update(myproc_discrete[type_])

                    # Assemble in parallel allprocs_prom2abs_list
                    for prom_name, abs_names_list in iteritems(myproc_prom2abs_list[type_]):
                        if prom_name not in allprocs_prom2abs_list[type_]:
                            allprocs_prom2abs_list[type_][prom_name] = []
                        allprocs_prom2abs_list[type_][prom_name].extend(abs_names_list)

        if self._var_discrete['input'] or self._var_discrete['output']:
            self._discrete_inputs = _DictValues(self._var_discrete['input'])
            self._discrete_outputs = _DictValues(self._var_discrete['output'])
        else:
            self._discrete_inputs = self._discrete_outputs = ()

    def _setup_var_sizes(self, recurse=True):
        """
        Compute the arrays of local variable sizes for all variables/procs on this system.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(Group, self)._setup_var_sizes()

        self._var_offsets = None

        iproc = self.comm.rank
        nproc = self.comm.size

        subsystems_proc_range = self._subsystems_proc_range

        # Recursion
        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_var_sizes(recurse)

        sizes = self._var_sizes
        relnames = self._var_allprocs_relevant_names

        vec_names = self._lin_rel_vec_name_list if self._use_derivatives else self._vec_names

        # Compute _var_sizes
        for vec_name in vec_names:
            sizes[vec_name] = {}
            subsystems_var_range = self._subsystems_var_range[vec_name]

            for type_ in ['input', 'output']:
                sizes[vec_name][type_] = np.zeros((nproc, len(relnames[vec_name][type_])),
                                                  INT_DTYPE)

                for ind, subsys in enumerate(self._subsystems_myproc):
                    if vec_name not in subsys._rel_vec_names:
                        continue
                    proc_slice = slice(*subsystems_proc_range[ind])
                    var_slice = slice(*subsystems_var_range[type_][subsys.name])
                    if proc_slice.stop - proc_slice.start > subsys.comm.size:
                        # in this case, we've split the proc for parallel FD, so subsys doesn't
                        # have var_sizes for all the ranks we need. Since each parallel FD comm
                        # has the same size distribution (since all are identical), just 'tile'
                        # the var_sizes from the subsystem to fill in the full rank range we need
                        # at this level.
                        assert (proc_slice.stop - proc_slice.start) % subsys.comm.size == 0, \
                            "%s comm size (%d) is not an exact multiple of %s comm size (%d)" % (
                                self.pathname, self.comm.size, subsys.pathname, subsys.comm.size)
                        proc_i = proc_slice.start
                        while proc_i < proc_slice.stop:
                            sizes[vec_name][type_][proc_i:proc_i + subsys.comm.size, var_slice] = \
                                subsys._var_sizes[vec_name][type_]
                            proc_i += subsys.comm.size
                    else:
                        sizes[vec_name][type_][proc_slice, var_slice] = \
                            subsys._var_sizes[vec_name][type_]

        # If parallel, all gather
        if self.comm.size > 1:
            for vec_name in self._lin_rel_vec_name_list:
                sizes = self._var_sizes[vec_name]
                for type_ in ['input', 'output']:
                    self.comm.Allgather(sizes[type_][iproc, :], sizes[type_])

            # compute owning ranks
            owns = self._owning_rank
            for type_ in ('input', 'output'):
                sizes = self._var_sizes[vec_names[0]][type_]
                for i, name in enumerate(self._var_allprocs_abs_names[type_]):
                    for rank in range(self.comm.size):
                        if sizes[rank, i] > 0:
                            owns[name] = rank
                            break

                if self._var_allprocs_discrete[type_]:
                    local = list(self._var_discrete[type_])
                    for i, names in enumerate(self.comm.allgather(local)):
                        for n in names:
                            if n not in owns:
                                owns[n] = i

        if self._use_derivatives:
            self._var_sizes['nonlinear'] = self._var_sizes['linear']

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
            if (prom_out not in allprocs_prom2abs_list_out and
                    prom_out not in self._var_allprocs_discrete['output']):
                raise NameError(
                    "Output '%s' does not exist for connection in '%s' from '%s' to '%s'." %
                    (prom_out, self.pathname, prom_out, prom_in))

            if (prom_in not in allprocs_prom2abs_list_in and
                    prom_in not in self._var_allprocs_discrete['input']):
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
            for subsys in self._subgroups_myproc:
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

        for subsys in self._subgroups_myproc:
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

    def _setup_connections(self, recurse=True):
        """
        Compute dict of all implicit and explicit connections owned by this Group.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        abs_in2out = self._conn_abs_in2out = {}
        global_abs_in2out = self._conn_global_abs_in2out
        pathname = self.pathname
        allprocs_discrete_in = self._var_allprocs_discrete['input']
        allprocs_discrete_out = self._var_allprocs_discrete['output']

        # Recursion
        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_connections(recurse)

        if MPI:
            # collect set of local (not remote, not distributed) subsystems so we can
            # identify cross-process connections, which require the use of distributed
            # instead of purely local vector and transfer objects.
            self._local_system_set = set()
            for s in self._subsystems_myproc:
                if isinstance(s, Group):
                    self._local_system_set.update(s._local_system_set)
                elif not s.options['distributed']:
                    self._local_system_set.add(s.pathname)

        path_len = len(pathname) + 1 if pathname else 0

        allprocs_abs2meta = self._var_allprocs_abs2meta

        self._vector_class = None

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
                    if abs_in in allprocs_discrete_in:
                        self._conn_discrete_in2out[abs_in] = abs_out
                    elif abs_out in allprocs_discrete_out:
                        raise RuntimeError("Can't connect discrete output '%s' to continuous "
                                           "input '%s'." % (abs_out, abs_in))
                    else:
                        abs_in2out[abs_in] = abs_out

                    if MPI and self._vector_class is None:
                        # check for any cross-process data transfer.  If found, use
                        # self._distributed_vector_class as our vector class.
                        in_path = abs_in.rsplit('.', 1)[0]
                        if in_path not in self._local_system_set:
                            self._vector_class = self._distributed_vector_class
                        else:
                            out_path = abs_out.rsplit('.', 1)[0]
                            if out_path not in self._local_system_set:
                                self._vector_class = self._distributed_vector_class

            # if connected output has scaling then we need input scaling
            if not self._has_input_scaling and not (abs_in in allprocs_discrete_in or
                                                    abs_out in allprocs_discrete_out):
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

        if self._vector_class is None:
            # our vectors are just local vectors.
            self._vector_class = self._local_vector_class

        # check compatability for any discrete connections
        for abs_in, abs_out in iteritems(self._conn_discrete_in2out):
            in_type = self._var_allprocs_discrete['input'][abs_in]['type']
            try:
                out_type = self._var_allprocs_discrete['output'][abs_out]['type']
            except KeyError:
                raise RuntimeError("Can't connect discrete output '%s' to continuous "
                                   "input '%s'." % (abs_out, abs_in))
            if not issubclass(in_type, out_type):
                raise RuntimeError("Type '%s' of output '%s' is"
                                   " incompatible with type '%s' of input '%s'." %
                                   (out_type.__name__, abs_out, in_type.__name__, abs_in))

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
                    simple_warning("Output '%s' with units of '%s' is "
                                   "connected to input '%s' which has no"
                                   " units." % (abs_out, out_units, abs_in))
                elif not is_compatible(in_units, out_units):
                    raise RuntimeError("Output units of '%s' for '%s' are"
                                       " incompatible with input units of "
                                       "'%s' for '%s'." %
                                       (out_units, abs_out, in_units, abs_in))
            elif in_units is not None:
                simple_warning("Input '%s' with units of '%s' is "
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
                            raise ValueError(msg % (abs_out, abs_in, bad_idx, out_size))
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
                                    raise ValueError(msg % (abs_out, abs_in, i, d_size))

    def _transfer(self, vec_name, mode, isub=None):
        """
        Perform a vector transfer.

        Parameters
        ----------
        vec_name : str
            Name of the vector RHS on which to perform a transfer.
        mode : str
            Either 'fwd' or 'rev'
        isub : None or int
            If None, perform a full transfer.
            If int, perform a partial transfer for linear Gauss--Seidel.
        """
        vec_inputs = self._vectors['input'][vec_name]

        if mode == 'fwd':
            if self._has_input_scaling:
                vec_inputs.scale('norm')
                self._transfers[vec_name][mode, isub].transfer(vec_inputs,
                                                               self._vectors['output'][vec_name],
                                                               mode)
                vec_inputs.scale('phys')
            else:
                self._transfers[vec_name][mode, isub].transfer(vec_inputs,
                                                               self._vectors['output'][vec_name],
                                                               mode)
            if self._conn_discrete_in2out and vec_name == 'nonlinear':
                self._discrete_transfer(isub)

        else:  # rev
            if self._has_input_scaling:
                vec_inputs.scale('phys')
                self._transfers[vec_name][mode, isub].transfer(vec_inputs,
                                                               self._vectors['output'][vec_name],
                                                               mode)
                vec_inputs.scale('norm')
            else:
                self._transfers[vec_name][mode, isub].transfer(vec_inputs,
                                                               self._vectors['output'][vec_name],
                                                               mode)

    def _discrete_transfer(self, isub):
        """
        Transfer discrete variables between components.  This only occurs in fwd mode.

        Parameters
        ----------
        isub : None or int
            If None, perform a full transfer.
            If int, perform a partial transfer for linear Gauss--Seidel.
        """
        comm = self.comm
        key = None if isub is None else self._subsystems_allprocs[isub].name

        if comm.size == 1:
            for src_sys_name, src, tgt_sys_name, tgt in self._discrete_transfers[key]:
                tgt_sys = self._loc_subsys_map[tgt_sys_name]
                src_sys = self._loc_subsys_map[src_sys_name]
                # note that we are not copying the discrete value here, so if the
                # discrete value is some mutable object, for example not an int or str,
                # the downstream system will have a reference to the same object
                # as the source, allowing the downstream system to modify the value as
                # seen by the source system.
                tgt_sys._discrete_inputs[tgt] = src_sys._discrete_outputs[src]

        else:  # MPI
            allprocs_recv = self._allprocs_discrete_recv[key]
            discrete_out = self._var_discrete['output']
            if key in self._discrete_transfers:
                xfers, remote_send = self._discrete_transfers[key]
                if allprocs_recv:
                    sendvars = [(n, discrete_out[n]['value']) for n in remote_send]
                    allprocs_send = comm.gather(sendvars, root=0)
                    if comm.rank == 0:
                        allprocs_dict = {}
                        for i in range(comm.size):
                            allprocs_dict.update(allprocs_send[i])
                        recvs = [{} for i in range(comm.size)]
                        for rname, ranks in iteritems(allprocs_recv):
                            val = allprocs_dict[rname]
                            for i in ranks:
                                recvs[i][rname] = val
                        data = comm.scatter(recvs, root=0)
                    else:
                        data = comm.scatter(None, root=0)
                else:
                    data = None

                for src_sys_name, src, tgt_sys_name, tgt in xfers:
                    if tgt_sys_name in self._loc_subsys_map:
                        tgt_sys = self._loc_subsys_map[tgt_sys_name]
                        if tgt in tgt_sys._discrete_inputs:
                            abs_src = '.'.join((src_sys_name, src))
                            if data is not None and abs_src in data:
                                src_val = data[abs_src]
                            else:
                                src_val = self._loc_subsys_map[src_sys_name]._discrete_outputs[src]
                            tgt_sys._discrete_inputs[tgt] = src_val

    def _setup_global(self, ext_num_vars, ext_sizes):
        """
        Compute total number and total size of variables in systems before / after this system.

        Parameters
        ----------
        ext_num_vars : {'input': (int, int), 'output': (int, int)}
            Total number of allprocs variables in system before/after this one.
        ext_sizes : {'input': (int, int), 'output': (int, int)}
            Total size of allprocs variables in system before/after this one.
        """
        super(Group, self)._setup_global(ext_num_vars, ext_sizes)

        iproc = self.comm.rank
        relnames = self._var_allprocs_relevant_names

        for subsys in self._subsystems_myproc:
            sub_ext_num_vars = {}
            sub_ext_sizes = {}

            if subsys._use_derivatives:
                vec_names = subsys._lin_rel_vec_name_list
            else:
                vec_names = subsys._vec_names

            for vec_name in vec_names:
                subsystems_var_range = self._subsystems_var_range[vec_name]
                sizes = self._var_sizes[vec_name]

                sub_ext_num_vars[vec_name] = {}
                sub_ext_sizes[vec_name] = {}

                for type_ in ['input', 'output']:
                    num = len(relnames[vec_name][type_])
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

            if subsys._use_derivatives:
                sub_ext_num_vars['nonlinear'] = sub_ext_num_vars['linear']
                sub_ext_sizes['nonlinear'] = sub_ext_sizes['linear']

            subsys._setup_global(sub_ext_num_vars, sub_ext_sizes)

    def _setup_transfers(self, recurse=True):
        """
        Compute all transfers that are owned by this system.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        self._vector_class.TRANSFER._setup_transfers(self, recurse=recurse)
        if self._conn_discrete_in2out:
            self._vector_class.TRANSFER._setup_discrete_transfers(self, recurse=recurse)

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

        subsys.name = subsys.pathname = name

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
        """
        super(Group, self)._solve_nonlinear()

        name = self.pathname if self.pathname else 'root'

        with Recording(name + '._solve_nonlinear', self.iter_count, self):
            self._nonlinear_solver.solve()

    def _guess_nonlinear(self):
        """
        Provide initial guess for states.
        """
        if self._has_guess:
            for ind, sub in enumerate(self._subsystems_myproc):
                if sub._has_guess:
                    isub = self._subsystems_myproc_inds[ind]
                    self._transfer('nonlinear', 'fwd', isub)
                    sub._guess_nonlinear()

    def _apply_linear(self, jac, vec_names, rel_systems, mode, scope_out=None, scope_in=None):
        """
        Compute jac-vec product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            If None, use local jacobian, else use assembled jacobian jac.
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
        vec_names = [v for v in vec_names if v in self._rel_vec_names]

        if self._owns_approx_jac:
            jac = self._jacobian
        elif jac is None and self._assembled_jac is not None:
            jac = self._assembled_jac
        if jac is not None:
            for vec_name in vec_names:
                with self._matvec_context(vec_name, scope_out, scope_in, mode) as vecs:
                    d_inputs, d_outputs, d_residuals = vecs
                    jac._apply(self, d_inputs, d_outputs, d_residuals, mode)
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
                        s._vectors['residual']['linear'].set_const(0.0)

            for subsys in self._subsystems_myproc:
                if rel_systems is None or subsys.pathname in rel_systems:
                    subsys._apply_linear(jac, vec_names, rel_systems, mode,
                                         scope_out, scope_in)

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
        """
        vec_names = [v for v in vec_names if v in self._rel_vec_names]

        self._linear_solver.solve(vec_names, mode, rel_systems)

    def _linearize(self, jac, sub_do_ln=True):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            If None, use local jacobian, else use assembled jacobian jac.
        sub_do_ln : boolean
            Flag indicating if the children should call linearize on their linear solvers.
        """
        # Group finite difference
        if self._owns_approx_jac:
            jac = self._jacobian
            with self._unscaled_context(outputs=[self._outputs]):
                for approximation in itervalues(self._approx_schemes):
                    approximation.compute_approximations(self, jac=jac, total=True)
        else:
            if self._assembled_jac is not None:
                jac = self._assembled_jac

            # Only linearize subsystems if we aren't approximating the derivs at this level.
            for subsys in self._subsystems_myproc:
                do_ln = sub_do_ln and (subsys._linear_solver is not None and
                                       subsys._linear_solver._linearize_children())
                subsys._linearize(jac, sub_do_ln=do_ln)

            # Update jacobian
            if self._assembled_jac is not None:
                self._assembled_jac._update(self)

            if sub_do_ln:
                for subsys in self._subsystems_myproc:
                    if subsys._linear_solver is not None:
                        subsys._linear_solver._linearize()

        self._check_coloring_update()

    def approx_totals(self, method='fd', step=None, form=None, step_calc=None):
        """
        Approximate derivatives for a Group using the specified approximation method.

        Parameters
        ----------
        method : str
            The type of approximation that should be used. Valid options include:
            'fd': Finite Difference, 'cs': Complex Step
        step : float
            Step size for approximation. Defaults to None, in which case, the approximation
            method provides its default value.
        form : string
            Form for finite difference, can be 'forward', 'backward', or 'central'. Defaults to
            None, in which case, the approximation method provides its default value.
        step_calc : string
            Step type for finite difference, can be 'abs' for absolute', or 'rel' for
            relative. Defaults to None, in which case, the approximation method
            provides its default value.
        """
        self._approx_schemes = OrderedDict()
        supported_methods = {'fd': (FiniteDifference, DEFAULT_FD_OPTIONS),
                             'cs': (ComplexStep, DEFAULT_CS_OPTIONS)}

        if method not in supported_methods:
            msg = 'Method "{}" is not supported, method must be one of {}'
            raise ValueError(msg.format(method, supported_methods.keys()))

        if method not in self._approx_schemes:
            self._approx_schemes[method] = supported_methods[method][0]()

        default_opts = supported_methods[method][1]

        kwargs = {}
        if step:
            if 'step' in default_opts:
                kwargs['step'] = step
            else:
                raise RuntimeError("'step' is not a valid option for '%s'" % method)
        if form:
            if 'form' in default_opts:
                kwargs['form'] = form
            else:
                raise RuntimeError("'form' is not a valid option for '%s'" % method)
        if step_calc:
            if 'step_calc' in default_opts:
                kwargs['step_calc'] = step_calc
            else:
                raise RuntimeError("'step_calc' is not a valid option for '%s'" % method)

        self._owns_approx_jac = True
        self._owns_approx_jac_meta = kwargs

    def _setup_partials(self, recurse=True):
        """
        Call setup_partials in components.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        self._subjacs_info = info = {}

        super(Group, self)._setup_partials(recurse)

        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_partials(recurse)
                info.update(subsys._subjacs_info)

    def _setup_jacobians(self, recurse=True):
        """
        Set and populate jacobians down through the system tree.

        In <Group>, we only need to prepare for Group finite difference. However, to be efficient,
        we need to find the minimum set of inputs and outputs to approximate.

        Parameters
        ----------
        recurse : bool
            If True, setup jacobians in all descendants.
        """
        if not self._use_derivatives:
            return

        # Group finite difference or complex step.
        # TODO: Does this work under or over an AssembledJacobian (and does that make sense)
        if self._owns_approx_jac:
            self._jacobian = DictionaryJacobian(system=self)

            method = list(self._approx_schemes.keys())[0]
            approx = self._approx_schemes[method]
            pro2abs = self._var_allprocs_prom2abs_list
            abs2meta = self._var_allprocs_abs2meta

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

            for key in product(of, wrt.union(of)):
                if key in self._subjacs_info:
                    meta = self._subjacs_info[key]
                else:
                    meta = SUBJAC_META_DEFAULTS.copy()
                    if key[0] == key[1]:
                        size = self._var_allprocs_abs2meta[key[0]]['size']
                        meta['rows'] = meta['cols'] = np.arange(size)
                        # All group approximations are treated as explicit components, so we
                        # have a -1 on the diagonal.
                        meta['value'] = np.full(size, -1.0)

                meta['method'] = method

                # TODO: Maybe just need a subset of keys (those that go to the boundaries.)

                meta.update(self._owns_approx_jac_meta)

                # Create approximations for the ones we need.
                # Skip indepvarcomp res wrt other srcs
                if key[0] in ivc:
                    continue

                # Skip explicit res wrt outputs
                if key[1] in of and key[1] not in ivc:

                    # Support for specifying a desvar as an obj/con.
                    if key[1] not in wrt or key[0] == key[1]:
                        continue

                approx.add_approximation(key, meta)

                if meta['value'] is None:
                    shape = (abs2meta[key[0]]['size'], abs2meta[key[1]]['size'])
                    meta['shape'] = shape
                    meta['value'] = np.zeros(shape)

                self._subjacs_info[key] = meta

        super(Group, self)._setup_jacobians(recurse=recurse)

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
