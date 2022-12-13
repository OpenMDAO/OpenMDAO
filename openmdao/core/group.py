"""Define the Group class."""
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable

from itertools import product, chain
from numbers import Number
import inspect
from fnmatch import fnmatchcase
from difflib import get_close_matches

import numpy as np
import networkx as nx

from openmdao.jacobians.dictionary_jacobian import DictionaryJacobian
from openmdao.core.system import System, collect_errors
from openmdao.core.component import Component, _DictValues
from openmdao.vectors.vector import _full_slice
from openmdao.core.constants import _UNDEFINED, INT_DTYPE
from openmdao.proc_allocators.default_allocator import DefaultAllocator, ProcAllocationError
from openmdao.jacobians.jacobian import SUBJAC_META_DEFAULTS
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce
from openmdao.solvers.linear.linear_runonce import LinearRunOnce
from openmdao.utils.array_utils import array_connection_compatible, _flatten_src_indices, \
    shape_to_len
from openmdao.utils.general_utils import common_subpath, \
    convert_src_inds, ContainsAll, shape2tuple, get_connection_owner
from openmdao.utils.units import is_compatible, unit_conversion, _has_val_mismatch, _find_unit, \
    _is_unitless, simplify_unit
from openmdao.utils.mpi import MPI, check_mpi_exceptions, multi_proc_exception_check
import openmdao.utils.coloring as coloring_mod
from openmdao.utils.indexer import indexer, Indexer
from openmdao.utils.om_warnings import issue_warning, UnitsWarning, UnusedOptionWarning, \
    PromotionWarning, MPIWarning
from openmdao.core.constants import _SetupStatus

# regex to check for valid names.
import re
namecheck_rgx = re.compile('[a-zA-Z][_a-zA-Z0-9]*')


# use a class with slots instead of a namedtuple so that we can
# change index after creation if needed.
class _SysInfo(object):

    __slots__ = ['system', 'index']

    def __init__(self, system, index):
        self.system = system
        self.index = index

    def __iter__(self):
        yield self.system
        yield self.index


class _PromotesInfo(object):
    __slots__ = ['src_indices', 'flat', 'src_shape', 'promoted_from', 'prom']

    def __init__(self, src_indices=None, flat=None, src_shape=None, promoted_from='', prom=None):
        self.flat = flat
        self.src_shape = src_shape
        if src_indices is not None:
            if isinstance(src_indices, Indexer):
                self.src_indices = src_indices
                self.src_indices.set_src_shape(self.src_shape)
            else:
                self.src_indices = indexer(src_indices, src_shape=self.src_shape, flat_src=flat)
        else:
            self.src_indices = None
        self.promoted_from = promoted_from  # pathname of promoting system
        self.prom = prom  # local promoted name of input

    def __iter__(self):
        yield self.src_indices
        yield self.flat
        yield self.src_shape

    def __repr__(self):  # pragma no cover
        return (f"_PromotesInfo(src_indices={self.src_indices}, flat={self.flat}, "
                f"src_shape={self.src_shape}, promoted_from={self.promoted_from}, "
                f"prom={self.prom})")

    def prom_path(self):
        if self.promoted_from is None or self.prom is None:
            return ''
        return '.'.join((self.promoted_from, self.prom)) if self.promoted_from else self.prom

    def copy(self):
        return _PromotesInfo(self.src_indices.copy(), self.flat, self.src_shape, self.promoted_from,
                             self.prom)

    def set_src_shape(self, shape):
        if self.src_indices is not None:
            self.src_indices.set_src_shape(shape)
        self.src_shape = shape

    def compare(self, other):
        """
        Compare attributes in the two objects.

        Two attributes are considered mismatched only if neither is None and their values
        are unequal.

        Returns
        -------
        list
            List of unequal atrribute names.
        """
        mismatches = []

        if self.flat != other.flat:
            if self.flat is not None and other.flat is not None:
                mismatches.append('flat_src_indices')

        if self.src_shape != other.src_shape:
            if self.src_shape is not None and other.src_shape is not None:
                mismatches.append('src_shape')

        self_srcinds = None if self.src_indices is None else self.src_indices.as_array()
        other_srcinds = None if other.src_indices is None else other.src_indices.as_array()

        if isinstance(self_srcinds, np.ndarray) and isinstance(other_srcinds, np.ndarray):
            if (self_srcinds.shape != other_srcinds.shape or
                    not np.all(self_srcinds == other_srcinds)):
                mismatches.append('src_indices')

        return mismatches


class Group(System):
    """
    Class used to group systems together; instantiate or inherit.

    Parameters
    ----------
    **kwargs : dict
        Dict of arguments available here and in all descendants of this Group.

    Attributes
    ----------
    _mpi_proc_allocator : ProcAllocator
        Object used to allocate MPI processes to subsystems.
    _proc_info : dict of subsys_name: (min_procs, max_procs, weight, proc_group)
        Information used to determine MPI process allocation to subsystems.
    _subgroups_myproc : list
        List of local subgroups.
    _manual_connections : dict
        Dictionary of input_name: (output_name, src_indices) connections.
    _group_inputs : dict
        Mapping of promoted names to certain metadata (src_indices, units).
    _static_group_inputs : dict
        Group inputs added outside of setup/configure.
    _pre_config_group_inputs : dict
        Group inputs added inside of setup but before configure.
    _static_manual_connections : dict
        Dictionary that stores all explicit connections added outside of setup.
    _conn_abs_in2out : {'abs_in': 'abs_out'}
        Dictionary containing all explicit & implicit continuous var connections owned
        by this system only. The data is the same across all processors.
    _conn_discrete_in2out : {'abs_in': 'abs_out'}
        Dictionary containing all explicit & implicit discrete var connections owned
        by this system only. The data is the same across all processors.
    _transfers : dict of dict of Transfers
        First key is mode, second is subname where
        mode is 'fwd' or 'rev' and subname is the subsystem name
        or subname can be None for the full, simultaneous transfer.
    _discrete_transfers : dict of discrete transfer metadata
        Key is system pathname or None for the full, simultaneous transfer.
    _setup_procs_finished : bool
        Flag to check if setup_procs is complete
    _contains_parallel_group : bool
        If True, this Group contains a ParallelGroup. Only used to determine if a parallel
        group or distributed component is below a DirectSolver so that we can raise an exception.
    _order_set : bool
        Flag to check if set_order has been called.
    _auto_ivc_warnings : list
        List of Auto IVC warnings to be raised later with simple_warnings.
    _shapes_graph : nx.Graph
        Dynamic shape dependency graph, or None.
    _shape_knowns : set
        Set of shape dependency graph nodes with known (non-dynamic) shapes.
    """

    def __init__(self, **kwargs):
        """
        Set the solvers to nonlinear and linear block Gauss--Seidel by default.
        """
        self._mpi_proc_allocator = DefaultAllocator()
        self._proc_info = {}

        super().__init__(**kwargs)

        self._subgroups_myproc = None
        self._manual_connections = {}
        self._group_inputs = {}
        self._pre_config_group_inputs = {}
        self._static_group_inputs = {}
        self._static_manual_connections = {}
        self._conn_abs_in2out = {}
        self._conn_discrete_in2out = {}
        self._transfers = {}
        self._discrete_transfers = {}
        self._setup_procs_finished = False
        self._contains_parallel_group = False
        self._order_set = False
        self._shapes_graph = None
        self._shape_knowns = None

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

    def set_input_defaults(self, name, val=_UNDEFINED, units=None, src_shape=None):
        """
        Specify metadata to be assumed when multiple inputs are promoted to the same name.

        Parameters
        ----------
        name : str
            Promoted input name.
        val : object
            Value to assume for the promoted input.
        units : str or None
            Units to assume for the promoted input.
        src_shape : int or tuple
            Assumed shape of any connected source or higher level promoted input.
        """
        meta = {'prom': name, 'auto': False}
        if val is _UNDEFINED:
            src_shape = shape2tuple(src_shape)
        else:
            meta['val'] = val
            if src_shape is not None:
                issue_warning("value was set in set_input_defaults, so ignoring "
                              f"value {src_shape} of src_shape.", prefix=self.msginfo,
                              category=PromotionWarning)
            if isinstance(val, np.ndarray):
                src_shape = val.shape
            elif isinstance(val, Number):
                src_shape = (1,)
        if units is not None:
            if not isinstance(units, str):
                raise TypeError('%s: The units argument should be a str or None' % self.msginfo)
            meta['units'] = simplify_unit(units, msginfo=self.msginfo)

        if src_shape is not None:
            meta['src_shape'] = src_shape

        if self._static_mode:
            dct = self._static_group_inputs
        else:
            dct = self._group_inputs

        if name in dct:
            old = dct[name][0]
            overlap = set(old).intersection(meta)
            if overlap:
                issue_warning(f"Setting input defaults for input '{name}' which "
                              f"override previously set defaults for {sorted(overlap)}.",
                              prefix=self.msginfo, category=PromotionWarning)
            old.update(meta)
        else:
            dct[name] = [meta]

    def _get_matvec_scope(self, excl_sub=None):
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
        if excl_sub is None:
            cache_key = None
        else:
            cache_key = excl_sub.pathname

        try:
            iovars, excl = self._scope_cache[cache_key]

            # Make sure they're the same subsystem instance before returning
            if excl is excl_sub:
                return iovars
        except KeyError:
            pass

        if excl_sub is None:
            # A value of None will be interpreted as 'all outputs'.
            scope_out = None

            # All inputs connected to an output in this system
            scope_in = frozenset(self._conn_global_abs_in2out).intersection(
                self._var_allprocs_abs2meta['input'])

        else:
            # Empty for the excl_sub
            scope_out = frozenset()

            # All inputs connected to an output in this system but not in excl_sub
            # allins is used to filter out discrete variables that might be found in
            # self._conn_global_abs_in2out.
            allins = self._var_allprocs_abs2meta['input']
            exvars = excl_sub._var_allprocs_abs2idx
            scope_in = frozenset(abs_in for abs_in, abs_out in self._conn_global_abs_in2out.items()
                                 if abs_out not in exvars and abs_in in allins)

        # Use the pathname as the dict key instead of the object itself. When
        # the object is used as the key, memory leaks result from multiple
        # calls to setup().
        self._scope_cache[cache_key] = ((scope_out, scope_in), excl_sub)
        return scope_out, scope_in

    def _compute_root_scale_factors(self):
        """
        Compute scale factors for all variables.

        Returns
        -------
        dict
            Mapping of each absolute var name to its corresponding scaling factor tuple.
        """
        # The output and residual vectors are handled in system.py.
        scale_factors = super()._compute_root_scale_factors()

        # Input scaling for connected inputs is added here.
        # This is a combined scale factor that includes the scaling of the connected source
        # and the unit conversion between the source output and each target input.
        if self._has_input_scaling:
            abs2meta_in = self._var_abs2meta['input']
            allprocs_meta_out = self._var_allprocs_abs2meta['output']
            for abs_in, abs_out in self._conn_global_abs_in2out.items():
                if abs_in not in abs2meta_in:
                    # we only perform scaling on local, non-discrete arrays, so skip
                    continue

                meta_in = abs2meta_in[abs_in]

                meta_out = allprocs_meta_out[abs_out]
                ref = meta_out['ref']
                ref0 = meta_out['ref0']

                src_indices = meta_in['src_indices']

                if src_indices is not None:
                    if not (np.ndim(ref) == 0 and np.ndim(ref0) == 0):
                        # TODO: if either ref or ref0 are not scalar and the output is
                        # distributed, we need to do a scatter
                        # to obtain the values needed due to global src_indices
                        if meta_out['distributed']:
                            raise RuntimeError("{}: vector scalers with distrib vars "
                                               "not supported yet.".format(self.msginfo))

                        if not src_indices._flat_src:
                            src_indices = _flatten_src_indices(src_indices, meta_in['shape'],
                                                               meta_out['global_shape'],
                                                               meta_out['global_size'])

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

                    # No unit conversion, only scaling. Just send the scale factors.
                    scale_factors[abs_in] = {
                        'input': (a0, a1),
                    }

                else:
                    factor, offset = unit_conversion(units_out, units_in)
                    a0 = ref0
                    a1 = ref - ref0

                    # Send both unit scaling and solver scaling. Linear input vectors need to
                    # treat them differently in reverse mode.
                    scale_factors[abs_in] = {
                        'input': (a0, a1, factor, offset),
                    }

                    # For adder allocation check.
                    a0 = (ref0 + offset) * factor

                # Check whether we need to allocate an adder for the input vector.
                if np.any(np.asarray(a0)):
                    self._has_input_adder = True

        return scale_factors

    def _configure(self):
        """
        Configure our model recursively to assign any children settings.

        Highest system's settings take precedence.
        """
        # reset group_inputs back to what it was just after self.setup() in case _configure
        # is called multiple times.
        self._group_inputs = self._pre_config_group_inputs.copy()
        for n, lst in self._group_inputs.items():
            self._group_inputs[n] = lst.copy()

        for subsys in self._subsystems_myproc:
            subsys._configure()
            subsys._setup_var_data()

            self._has_guess |= subsys._has_guess
            self._has_bounds |= subsys._has_bounds
            self.matrix_free |= subsys.matrix_free

        self._problem_meta['setup_status'] = _SetupStatus.POST_CONFIGURE
        self.configure()

        # if our configure() has added or promoted any variables, we have to call
        # _setup_var_data again on any modified systems and their ancestors (only those that
        # are our descendents).
        self._problem_meta['config_info']._update_modified_systems(self)

    def _setup_procs(self, pathname, comm, mode, prob_meta):
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
        mode : str
            Derivatives calculation mode, 'fwd' for forward, and 'rev' for
            reverse (adjoint). Default is 'rev'.
        prob_meta : dict
            Problem level metadata.
        """
        super()._setup_procs(pathname, comm, mode, prob_meta)
        self._setup_procs_finished = False

        nproc = comm.size

        if self._num_par_fd > 1:
            info = self._coloring_info
            if comm.size > 1:
                # if approx_totals has been declared, or there is an approx coloring, setup par FD
                if self._owns_approx_jac or info['dynamic'] or info['static'] is not None:
                    comm = self._setup_par_fd_procs(comm)
                else:
                    msg = "%s: num_par_fd = %d but FD is not active." % (self.msginfo,
                                                                         self._num_par_fd)
                    raise RuntimeError(msg)
            elif not MPI:
                msg = f"MPI is not active but num_par_fd = {self._num_par_fd}. No parallel " \
                      f"finite difference will be performed."
                issue_warning(msg, prefix=self.msginfo, category=MPIWarning)

        self.comm = comm

        self._subsystems_allprocs = self._static_subsystems_allprocs.copy()
        self._manual_connections = self._static_manual_connections.copy()
        self._group_inputs = self._static_group_inputs.copy()
        # copy doesn't copy the internal list so we have to do it manually (we don't want
        # a full deepcopy either because we want the internal metadata dicts to be shared)
        for n, lst in self._group_inputs.items():
            self._group_inputs[n] = lst.copy()

        # Call setup function for this group.
        self.setup()
        self._setup_check()

        # need to save these because _setup_var_data can be called multiple times
        # during the config process and we don't want to wipe out any group_inputs
        # that were added during self.setup()
        self._pre_config_group_inputs = self._group_inputs.copy()
        for n, lst in self._pre_config_group_inputs.items():
            self._pre_config_group_inputs[n] = lst.copy()

        if MPI:

            allsubs = list(self._subsystems_allprocs.values())
            proc_info = [self._proc_info[s.name] for s, _ in allsubs]

            # Call the load balancing algorithm
            try:
                sub_inds, sub_comm, sub_proc_range = self._mpi_proc_allocator(
                    proc_info, len(allsubs), comm)
            except ProcAllocationError as err:
                if err.sub_inds is None:
                    raise RuntimeError("%s: %s" % (self.msginfo, err.msg))
                else:
                    raise RuntimeError("%s: MPI process allocation failed: %s for the following "
                                       "subsystems: %s" %
                                       (self.msginfo, err.msg,
                                        [allsubs[i].system.name for i in err.sub_inds]))

            self._subsystems_myproc = [allsubs[ind].system for ind in sub_inds]

            # Define local subsystems
            if (self._mpi_proc_allocator.parallel and
                    not (np.sum([minp for minp, _, _, _ in proc_info]) <= comm.size)):
                # reorder the subsystems_allprocs based on which procs they live on. If we don't
                # do this, we can get ordering mismatches in some of our data structures.
                new_allsubs = {}
                seen = set()
                gathered = self.comm.allgather(sub_inds)
                for inds in gathered:
                    for ind in inds:
                        if ind not in seen:
                            sinfo = allsubs[ind]
                            sinfo.index = len(new_allsubs)
                            new_allsubs[sinfo.system.name] = sinfo
                            seen.add(ind)
                self._subsystems_allprocs = new_allsubs
        else:
            sub_comm = comm
            self._subsystems_myproc = [s for s, _ in self._subsystems_allprocs.values()]

        # need to set pathname correctly even for non-local subsystems
        for s, _ in self._subsystems_allprocs.values():
            s.pathname = '.'.join((self.pathname, s.name)) if self.pathname else s.name

        # Perform recursion
        allsubs = self._subsystems_allprocs
        for subsys in self._subsystems_myproc:
            subsys._setup_procs(subsys.pathname, sub_comm, mode, prob_meta)

        # build a list of local subgroups to speed up later loops
        self._subgroups_myproc = [s for s in self._subsystems_myproc if isinstance(s, Group)]

        if nproc > 1 and self._mpi_proc_allocator.parallel:
            self._problem_meta['parallel_groups'].append(self.pathname)

            allpars = self.comm.allgather(self._problem_meta['parallel_groups'])
            full = set()
            for p in allpars:
                full.update(p)
            self._problem_meta['parallel_groups'] = sorted(full)

        if self._problem_meta['parallel_groups']:
            prefix = self.pathname + '.' if self.pathname else ''
            for par in self._problem_meta['parallel_groups']:
                if par.startswith(prefix) and par != prefix:
                    self._contains_parallel_group = True
                    break

        self._setup_procs_finished = True

    def _configure_check(self):
        """
        Do any error checking on i/o and connections.
        """
        for subsys in self._subsystems_myproc:
            subsys._configure_check()

        super()._configure_check()

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
        if MPI and self.comm.size > 1:
            all_states = set()
            byproc = self.comm.allgather(self._list_states())
            for proc_states in byproc:
                all_states.update(proc_states)
            return sorted(all_states)
        else:
            return self._list_states()

    def _get_all_promotes(self):
        """
        Create the top level mapping of all promoted names to absolute names for all local systems.

        This includes all buried promoted names.

        Returns
        -------
        dict
            Mapping of all promoted names to absolute names.
        """
        iotypes = ('input', 'output')
        if self.comm.size > 1:
            prom2abs = {'input': defaultdict(set), 'output': defaultdict(set)}
            rem_prom2abs = {'input': defaultdict(set), 'output': defaultdict(set)}
            myrank = self.comm.rank
            vars_to_gather = self._vars_to_gather

            for s in self.system_iter(recurse=True, include_self=True):
                prefix = s.pathname + '.' if s.pathname else ''
                for typ in iotypes:
                    # use abs2prom to determine locality since prom2abs is for allprocs
                    sys_abs2prom = s._var_abs2prom[typ]
                    t_remprom2abs = rem_prom2abs[typ]
                    t_prom2abs = prom2abs[typ]
                    for prom, alist in s._var_allprocs_prom2abs_list[typ].items():
                        abs_names = [n for n in alist if n in sys_abs2prom]
                        t_prom2abs[prefix + prom].update(abs_names)
                        t_remprom2abs[prefix + prom].update(n for n in abs_names
                                                            if n in vars_to_gather
                                                            and vars_to_gather[n] == myrank)

            all_proms = self.comm.gather(rem_prom2abs, root=0)
            if myrank == 0:
                for typ in iotypes:
                    t_prom2abs = prom2abs[typ]
                    for rankproms in all_proms:
                        for prom, absnames in rankproms[typ].items():
                            t_prom2abs[prom].update(absnames)

                    for prom, absnames in t_prom2abs.items():
                        t_prom2abs[prom] = sorted(absnames)  # sort to keep order same on all procs

                self.comm.bcast(prom2abs, root=0)
            else:
                prom2abs = self.comm.bcast(None, root=0)
        else:  # serial
            prom2abs = {'input': defaultdict(list), 'output': defaultdict(list)}
            for s in self.system_iter(recurse=True, include_self=True):
                prefix = s.pathname + '.' if s.pathname else ''
                for typ in iotypes:
                    t_prom2abs = prom2abs[typ]
                    for prom, abslist in s._var_allprocs_prom2abs_list[typ].items():
                        t_prom2abs[prefix + prom] = abslist

        return prom2abs

    def _top_level_post_connections(self, mode):
        # this is called on the top level group after all connections are known
        self._problem_meta['vars_to_gather'] = self._vars_to_gather
        self._problem_meta['prom2abs'] = self._get_all_promotes()

        self._resolve_group_input_defaults()
        self._setup_auto_ivcs(mode)
        self._check_prom_masking()

    def _check_prom_masking(self):
        """
        Raise exception if any promoted variable name masks an absolute variable name.

        Only called on the top level group.
        """
        prom2abs_in = self._var_allprocs_prom2abs_list['input']
        prom2abs_out = self._var_allprocs_prom2abs_list['output']
        abs2meta = self._var_allprocs_abs2meta

        for absname in chain(abs2meta['input'], abs2meta['output']):
            if absname in prom2abs_in:
                for name in prom2abs_in[absname]:
                    if name != absname:
                        raise RuntimeError(f"{self.msginfo}: Absolute variable name '{absname}'"
                                           " is masked by a matching promoted name. Try"
                                           " promoting to a different name. This can be caused"
                                           " by promoting '*' at group level or promoting using"
                                           " dotted names.")
            elif absname in prom2abs_out:
                if absname != prom2abs_out[absname][0]:
                    raise RuntimeError(f"{self.msginfo}: Absolute variable name '{absname}' is"
                                       " masked by a matching promoted name. Try"
                                       " promoting to a different name. This can be caused"
                                       " by promoting '*' at group level or promoting using"
                                       " dotted names.")

    def _top_level_post_sizes(self):
        # this runs after the variable sizes are known
        self._setup_global_shapes()

        self._resolve_ambiguous_input_meta()

        all_abs2meta_out = self._var_allprocs_abs2meta['output']
        conns = self._conn_global_abs_in2out

        self._resolve_src_indices()

        if self.comm.size > 1:
            abs2idx = self._var_allprocs_abs2idx
            all_abs2meta = self._var_allprocs_abs2meta
            all_abs2meta_in = all_abs2meta['input']

            # the code below is to handle the case where src_indices were not specified
            # for a distributed input or an input connected to a distributed auto_ivc
            # output. This update can't happen until sizes are known.
            dist_ins = (n for n, m in all_abs2meta_in.items() if m['distributed'] or
                        (conns[n].startswith('_auto_ivc.') and
                         all_abs2meta_out[conns[n]]['distributed']))
            dcomp_names = set(d.rpartition('.')[0] for d in dist_ins)
            if dcomp_names:
                added_src_inds = []
                for comp in self.system_iter(recurse=True, typ=Component):
                    if comp.pathname in dcomp_names:
                        added_src_inds.extend(
                            comp._update_dist_src_indices(conns, all_abs2meta, abs2idx,
                                                          self._var_sizes))

                updated = set()
                for alist in self.comm.allgather(added_src_inds):
                    updated.update(alist)

                for a in updated:
                    all_abs2meta_in[a]['has_src_indices'] = True

        abs2meta_in = self._var_abs2meta['input']
        allprocs_abs2meta_in = self._var_allprocs_abs2meta['input']
        allprocs_abs2meta_out = self._var_allprocs_abs2meta['output']

        if self.comm.size > 1:
            for abs_in, abs_out in sorted(conns.items()):
                if abs_out not in allprocs_abs2meta_out:
                    continue  # discrete var

                in_dist = allprocs_abs2meta_in[abs_in]['distributed']
                out_dist = allprocs_abs2meta_out[abs_out]['distributed']

                # check that src_indices match for dist->serial connection
                # FIXME: this transfers src_indices from all ranks to the owning rank so we could
                # run into memory issues if src_indices are large.  Maybe try something like
                # computing a hash in each rank and comparing those?
                if out_dist and not in_dist:
                    # all non-distributed inputs must have src_indices if they connect to a
                    # distributed output.
                    owner = self._owning_rank[abs_in]
                    if abs_in in abs2meta_in:  # input is local
                        src_inds = abs2meta_in[abs_in]['src_indices']
                        if src_inds is not None:
                            shaped = src_inds.shaped_instance()
                            if shaped is None:
                                self._collect_error(f"For connection from '{abs_out}' to '{abs_in}'"
                                                    f", src_indices {src_inds} have no source "
                                                    "shape.", ident=(abs_out, abs_in))
                                continue
                            else:
                                src_inds = shaped
                    else:
                        src_inds = None
                    if self.comm.rank == owner:
                        baseline = None
                        err = 0
                        for sinds in self.comm.gather(src_inds, root=owner):
                            if sinds is not None:
                                if baseline is None:
                                    baseline = sinds.as_array()
                                else:
                                    if not np.all(sinds.as_array() == baseline):
                                        err = 1
                                        break
                        if baseline is None:  # no src_indices were set
                            err = -1
                        self.comm.bcast(err, root=owner)
                    else:
                        self.comm.gather(src_inds, root=owner)
                        err = self.comm.bcast(None, root=owner)
                    if err == 1:
                        self._collect_error(f"{self.msginfo}: Can't connect distributed output "
                                            f"'{abs_out}' to non-distributed input '{abs_in}' "
                                            "because src_indices differ on different ranks.",
                                            ident=(abs_out, abs_in))
                    elif err == -1:
                        self._collect_error(f"{self.msginfo}: Can't connect distributed output "
                                            f"'{abs_out}' to non-distributed input '{abs_in}' "
                                            "without specifying src_indices.",
                                            ident=(abs_out, abs_in))

    @collect_errors
    def _resolve_src_indices(self):
        """
        Populate the promotes info list for each absolute input.

        This is called only at the top level of the system tree.
        """
        all_abs2meta_out = self._var_allprocs_abs2meta['output']
        all_abs2meta_in = self._var_allprocs_abs2meta['input']
        conns = self._conn_global_abs_in2out

        for tgt, plist in self._problem_meta['abs_in2prom_info'].items():
            src = conns[tgt]
            smeta = all_abs2meta_out[src]
            tmeta = all_abs2meta_in[tgt]

            if not smeta['distributed'] and tmeta['distributed']:
                root_shape = self._get_full_dist_shape(src, smeta['shape'])
            else:
                root_shape = smeta['global_shape']

            # plist is a list of (pinfo, shape, use_tgt) tuples, one for each level in the
            # system tree corresponding to an absolute input name, e.g., a plist for the
            # input 'abc.def.ghi.x' would look like [tup0, tup1, tup2, tup3] corresponding to
            # the ['', 'abc', 'abc.def', 'abc.def.ghi'] levels in the tree.

            # After this routine runs, all pinfo entries will have src_indices wrt the root
            # shape.

            # use a _PromotesInfo for the top level even though there really isn't a promote there
            current_pinfo = _PromotesInfo(src_shape=root_shape,
                                          prom=self._var_allprocs_abs2prom['input'][tgt])
            if plist[0] is None:  # no top level pinfo
                plist[0] = current_pinfo

            for i, pinfo in enumerate(plist):
                if pinfo is None:
                    pass
                elif current_pinfo.src_indices is None:
                    try:
                        if pinfo.src_shape is None:
                            pinfo.set_src_shape(root_shape)
                        elif pinfo.src_indices is not None and root_shape != pinfo.src_shape:
                            self._collect_error(f"When connecting '{src}' to "
                                                f"'{pinfo.prom_path()}': Promoted src_shape of "
                                                f"{pinfo.src_shape} for "
                                                f"'{pinfo.prom_path()}' differs from src_shape "
                                                f"{root_shape} for '{current_pinfo.prom_path()}'.",
                                                ident=(src, tgt))
                    except Exception:
                        type_exc, exc, tb = sys.exc_info()
                        self._collect_error(f"When connecting '{src}' to "
                                            f"'{pinfo.prom_path()}': {exc}",
                                            exc_type=type_exc, tback=tb, ident=(src, tgt))
                    current_pinfo = pinfo
                    continue
                elif pinfo.src_indices is None:
                    pinfo.src_indices = current_pinfo.src_indices
                    if pinfo.src_shape is None:
                        pinfo.set_src_shape(current_pinfo.src_shape)
                    current_pinfo = pinfo
                else:  # both have src_indices
                    try:
                        if pinfo.src_shape is None:
                            pinfo.set_src_shape(current_pinfo.src_indices.indexed_src_shape)
                        sinds = convert_src_inds(current_pinfo.src_indices, current_pinfo.src_shape,
                                                 pinfo.src_indices, pinfo.src_shape)
                    except Exception:
                        type_exc, exc, tb = sys.exc_info()
                        self._collect_error(f"When connecting '{conns[tgt]}' to "
                                            f"'{pinfo.prom_path()}': input "
                                            f"'{current_pinfo.prom_path()}' src_indices are "
                                            f"{current_pinfo.src_indices} and indexing into those "
                                            f"failed using src_indices {pinfo.src_indices} from "
                                            f"input '{pinfo.prom_path()}'. Error was: "
                                            f"{exc}", exc_type=type_exc, tback=tb,
                                            ident=(conns[tgt], tgt))
                        continue

                    # final src_indices are wrt original full sized source and are flat,
                    # so use val.shape and flat_src=True
                    # It would be nice if we didn't have to convert these and could just keep
                    # them in their original form and stack them to get the final result. We can
                    # do this when doing a get_val, but it doesn't work when doing a set_val.
                    src_indices = indexer(sinds, src_shape=root_shape, flat_src=True)
                    current_pinfo = _PromotesInfo(src_indices=src_indices, src_shape=root_shape,
                                                  flat=True, promoted_from=pinfo.promoted_from,
                                                  prom=pinfo.prom)
                plist[i] = current_pinfo

        with multi_proc_exception_check(self.comm):
            self._resolve_src_inds()

    def _resolve_src_inds(self):
        abs2prom = self._var_abs2prom['input']
        tree_level = len(self.pathname.split('.')) if self.pathname else 0
        abs_in2prom_info = self._problem_meta['abs_in2prom_info']
        seen = set()

        for tgt in self._var_abs2meta['input']:
            if tgt in abs_in2prom_info:
                prom = abs2prom[tgt]
                if prom in seen:
                    continue
                seen.add(prom)

                plist = abs_in2prom_info[tgt]
                pinfo = plist[tree_level]
                if pinfo is not None:
                    inds, flat, shape = pinfo
                    if inds is not None:
                        self._var_prom2inds[prom] = [shape, inds, flat]

        for s in self._subsystems_myproc:
            s._resolve_src_inds()

    def _setup_var_data(self):
        """
        Compute the list of abs var names, abs/prom name maps, and metadata dictionaries.
        """
        if self._var_allprocs_prom2abs_list is None:
            old_prom2abs = {}
        else:
            old_prom2abs = self._var_allprocs_prom2abs_list['input']

        super()._setup_var_data()

        var_discrete = self._var_discrete
        allprocs_discrete = self._var_allprocs_discrete

        abs2meta = self._var_abs2meta
        abs2prom = self._var_abs2prom

        allprocs_abs2meta = {'input': {}, 'output': {}}

        allprocs_prom2abs_list = self._var_allprocs_prom2abs_list

        for n, lst in self._group_inputs.items():
            lst[0]['path'] = self.pathname  # used for error reporting
            self._group_inputs[n] = lst.copy()  # must copy the list manually

        self._has_distrib_vars = False
        abs_in2prom_info = self._problem_meta['abs_in2prom_info']
        self._promotes_src_indices = {}

        for subsys in self._subsystems_myproc:
            self._has_output_scaling |= subsys._has_output_scaling
            self._has_output_adder |= subsys._has_output_adder
            self._has_resid_scaling |= subsys._has_resid_scaling
            self._has_distrib_vars |= subsys._has_distrib_vars

            var_maps = subsys._get_promotion_maps()

            sub_prefix = subsys.name + '.'

            for io in ['input', 'output']:
                abs2meta[io].update(subsys._var_abs2meta[io])
                allprocs_abs2meta[io].update(subsys._var_allprocs_abs2meta[io])
                subprom2prom = var_maps[io]

                allprocs_discrete[io].update(subsys._var_allprocs_discrete[io])
                var_discrete[io].update({sub_prefix + k: v for k, v in
                                         subsys._var_discrete[io].items()})

                sub_loc_proms = subsys._var_abs2prom[io]
                for sub_prom, sub_abs in subsys._var_allprocs_prom2abs_list[io].items():
                    if sub_prom in subprom2prom:
                        prom_name, _, pinfo, _ = subprom2prom[sub_prom]
                        if pinfo is not None and io == 'input':
                            pinfo = pinfo.copy()
                            pinfo.promoted_from = subsys.pathname
                            pinfo.prom = sub_prom
                            tree_level = len(subsys.pathname.split('.'))
                            for abs_in in sub_abs:
                                if abs_in not in abs_in2prom_info:
                                    # need a level for each system including '', so we don't
                                    # subtract 1 from abs_in.split('.') which includes the var name
                                    abs_in2prom_info[abs_in] = [None for s in abs_in.split('.')]
                                abs_in2prom_info[abs_in][tree_level] = pinfo
                    else:
                        prom_name = sub_prefix + sub_prom
                    if prom_name not in allprocs_prom2abs_list[io]:
                        allprocs_prom2abs_list[io][prom_name] = []
                    allprocs_prom2abs_list[io][prom_name].extend(sub_abs)
                    for abs_name in sub_abs:
                        if abs_name in sub_loc_proms:
                            abs2prom[io][abs_name] = prom_name

            if isinstance(subsys, Group):
                # propagate any subsystem 'set_input_defaults' info up to this Group
                subprom2prom = var_maps['input']
                for sub_prom, metalist in subsys._group_inputs.items():
                    if sub_prom in subprom2prom:
                        key = subprom2prom[sub_prom][0]
                    else:
                        key = sub_prefix + sub_prom
                    if key not in self._group_inputs:
                        self._group_inputs[key] = [{'path': self.pathname, 'prom': key,
                                                    'auto': True}]
                    self._group_inputs[key].extend(metalist)

        # If running in parallel, allgather
        if self.comm.size > 1 and self._mpi_proc_allocator.parallel:
            mysub = self._subsystems_myproc[0] if self._subsystems_myproc else False
            if (mysub and mysub.comm.rank == 0 and (mysub._full_comm is None or
                                                    mysub._full_comm.rank == 0)):
                raw = (allprocs_discrete, allprocs_prom2abs_list, allprocs_abs2meta,
                       self._has_output_scaling, self._has_output_adder,
                       self._has_resid_scaling, self._group_inputs, self._has_distrib_vars)
            else:
                raw = (
                    {'input': {}, 'output': {}},
                    {'input': {}, 'output': {}},
                    {'input': {}, 'output': {}},
                    False,
                    False,
                    False,
                    {},
                    False,
                )

            gathered = self.comm.allgather(raw)

            # start with a fresh dict to keep order the same in all procs
            old_abs2meta = allprocs_abs2meta
            allprocs_abs2meta = {'input': {}, 'output': {}}

            for io in ['input', 'output']:
                allprocs_prom2abs_list[io] = {}

            myrank = self.comm.rank
            for rank, (proc_discrete, proc_prom2abs_list, proc_abs2meta,
                       oscale, oadd, rscale, ginputs, has_dist_vars) in enumerate(gathered):
                self._has_output_scaling |= oscale
                self._has_output_adder |= oadd
                self._has_resid_scaling |= rscale
                self._has_distrib_vars |= has_dist_vars

                if rank != myrank:
                    for p, mlist in ginputs.items():
                        if p not in self._group_inputs:
                            self._group_inputs[p] = []
                        self._group_inputs[p].extend(mlist)

                for io in ['input', 'output']:
                    allprocs_abs2meta[io].update(proc_abs2meta[io])
                    allprocs_discrete[io].update(proc_discrete[io])

                    for prom_name, abs_names_list in proc_prom2abs_list[io].items():
                        if prom_name not in allprocs_prom2abs_list[io]:
                            allprocs_prom2abs_list[io][prom_name] = []
                        allprocs_prom2abs_list[io][prom_name].extend(abs_names_list)

            for io in ('input', 'output'):
                if allprocs_abs2meta[io]:
                    # update new allprocs_abs2meta with our local version (now that we have a
                    # consistent order for our dict), so that the 'size' metadata will
                    # accurately reflect this proc's var size instead of one from some other proc.
                    allprocs_abs2meta[io].update(old_abs2meta[io])

        self._var_allprocs_abs2meta = allprocs_abs2meta

        for prom_name, abs_list in allprocs_prom2abs_list['output'].items():
            if len(abs_list) > 1:
                self._collect_error("{}: Output name '{}' refers to "
                                    "multiple outputs: {}.".format(self.msginfo, prom_name,
                                                                   sorted(abs_list)))

        for io in ('input', 'output'):
            a2p = self._var_allprocs_abs2prom[io]
            for prom, abslist in self._var_allprocs_prom2abs_list[io].items():
                for abs_name in abslist:
                    a2p[abs_name] = prom

        if self._group_inputs:
            p2abs_in = self._var_allprocs_prom2abs_list['input']
            extra = [gin for gin in self._group_inputs if gin not in p2abs_in]
            if extra:
                # make sure that we don't have a leftover group input default entry from a previous
                # execution of _setup_var_data before promoted names were updated.
                ex = set()
                for e in extra:
                    if e in old_prom2abs:
                        del self._group_inputs[e]  # clean up old key using old promoted name
                    else:
                        ex.add(e)
                if ex:
                    self._collect_error(f"{self.msginfo}: The following group inputs, passed to "
                                        f"set_input_defaults(), could not be found: {sorted(ex)}.")

        if self._var_discrete['input'] or self._var_discrete['output']:
            self._discrete_inputs = _DictValues(self._var_discrete['input'])
            self._discrete_outputs = _DictValues(self._var_discrete['output'])
        else:
            self._discrete_inputs = self._discrete_outputs = ()

        self._vars_to_gather = self._find_vars_to_gather()

    def _resolve_group_input_defaults(self, show_warnings=False):
        """
        Resolve any ambiguities in group input defaults throughout the model.

        Only called at the model level.

        Parameters
        ----------
        show_warnings : bool
            Bool to show or hide the auto_ivc warnings.
        """
        skip = set(('path', 'use_tgt', 'prom', 'src_shape', 'src_indices', 'auto'))
        prom2abs_in = self._var_allprocs_prom2abs_list['input']
        abs_in2prom_info = self._problem_meta['abs_in2prom_info']
        abs2meta_in = self._var_allprocs_abs2meta['input']

        self._auto_ivc_warnings = []

        for prom, metalist in self._group_inputs.items():
            try:
                paths = [(i, m['path']) for i, m in enumerate(metalist) if not m['auto']]
                top_origin = paths[0][1]
                top_prom = metalist[paths[0][0]]['prom']
            except KeyError:
                issue_warning("No auto IVCs found", prefix=self.msginfo, category=PromotionWarning)
            allmeta = set()
            for meta in metalist:
                allmeta.update(meta)
            fullmeta = {n: _UNDEFINED for n in allmeta - skip}

            for key in sorted(fullmeta):
                for submeta in metalist:
                    if submeta['auto']:
                        continue
                    if key in submeta:
                        if fullmeta[key] is _UNDEFINED:
                            origin = submeta['path']
                            origin_prom = submeta['prom']
                            val = fullmeta[key] = submeta[key]
                            if origin != top_origin:
                                msg = (f"Group '{top_origin}' did not set a default "
                                       f"'{key}' for input '{top_prom}', so the value of "
                                       f"({val}) from group '{origin}' will be used.")
                                if show_warnings:
                                    issue_warning(msg, category=PromotionWarning)
                                else:
                                    self._auto_ivc_warnings.append(msg)

                        else:
                            eq = submeta[key] == val
                            if isinstance(eq, np.ndarray):
                                eq = np.all(eq)
                            if not eq:
                                # first, see if origin is an ancestor
                                if not origin or submeta['path'].startswith(origin + '.'):
                                    msg = (f"Groups '{origin}' and '{submeta['path']}' "
                                           f"called set_input_defaults for the input "
                                           f"'{origin_prom}' with conflicting '{key}'. "
                                           f"The value ({val}) from '{origin}' will be "
                                           "used.")
                                    if show_warnings:
                                        issue_warning(msg, category=PromotionWarning)
                                    else:
                                        self._auto_ivc_warnings.append(msg)
                                else:  # origin is not an ancestor, so we have an ambiguity
                                    if origin_prom != submeta['prom']:
                                        prm = f"('{origin_prom}' / '{submeta['prom']}')"
                                    else:
                                        prm = f"'{origin_prom}'"
                                    common = common_subpath((origin, submeta['path']))
                                    if common:
                                        sub = self._get_subsystem(common)
                                        if sub is not None:
                                            for a in prom2abs_in[prom]:
                                                if a in sub._var_abs2prom['input']:
                                                    prom = sub._var_abs2prom['input'][a]
                                                    break

                                    gname = f"Group named '{common}'" if common else 'model'
                                    self._collect_error(f"{self.msginfo}: The subsystems {origin} "
                                                        f"and {submeta['path']} called "
                                                        f"set_input_defaults for promoted input "
                                                        f"{prm} with conflicting values for "
                                                        f"'{key}'. Call <group>.set_input_defaults("
                                                        f"'{prom}', {key}=?), where <group> is the "
                                                        f"{gname} to remove the ambiguity.")

            # update all metadata dicts with any missing metadata that was filled in elsewhere
            # and update src_shape and use_tgt in abs_in2prom_info
            for meta in metalist:
                tree_level = len(meta['path'].split('.')) if meta['path'] else 0
                prefix = meta['path'] + '.' if meta['path'] else ''
                src_shape = None
                if 'val' in meta:
                    abs_in = prom2abs_in[prom][0]
                    if abs_in in abs2meta_in:  # it's a continuous variable
                        src_shape = np.asarray(meta['val']).shape
                elif 'src_shape' in meta:
                    src_shape = meta['src_shape']

                if src_shape is not None:
                    # Now update the global promotes info dict
                    for tgt in prom2abs_in[prom]:
                        if tgt in abs_in2prom_info and tgt.startswith(prefix):
                            pinfo = abs_in2prom_info[tgt][tree_level]
                            if pinfo is not None:
                                p2 = abs_in2prom_info[tgt][tree_level + 1]
                                if p2 is not None:
                                    # src_shape from a set_input_defaults call actually
                                    # must match the promoted src_shape from one level
                                    # deeper in the tree.
                                    if p2.src_shape is not None and p2.src_shape != src_shape:
                                        self._collect_error(f"{self.msginfo}: src_shape {src_shape}"
                                                            f" set by set_input_defaults('{prom}', "
                                                            f"...) in group '{meta['path']}' "
                                                            "conflicts with src_shape of "
                                                            f"{pinfo.src_shape} for promoted input "
                                                            f"'{pinfo.prom_path()}")
                                    p2.set_src_shape(src_shape)
                            else:
                                abs_in2prom_info[tgt][tree_level] = \
                                    _PromotesInfo(src_shape=src_shape, prom=prom,
                                                  promoted_from=self.pathname)

                meta.update(fullmeta)

    def _find_vars_to_gather(self):
        """
        Return a mapping of var pathname to owning rank.

        The mapping will contain ONLY systems that are remote on at least one proc.
        Distributed systems are not included.

        Returns
        -------
        dict
            The mapping of variable pathname to owning rank.
        """
        remote_vars = {}

        if self.comm.size > 1:
            myproc = self.comm.rank
            nprocs = self.comm.size

            for io in ('input', 'output'):
                abs2prom = self._var_abs2prom[io]
                abs2meta = self._var_allprocs_abs2meta[io]

                # var order must be same on all procs
                sorted_names = sorted(self._var_allprocs_abs2prom[io])
                locality = np.zeros((nprocs, len(sorted_names)), dtype=bool)
                for i, name in enumerate(sorted_names):
                    if name in abs2prom:
                        locality[myproc, i] = True

                my_loc = locality[myproc, :].copy()
                self.comm.Allgather(my_loc, locality)

                for i, name in enumerate(sorted_names):
                    nzs = np.nonzero(locality[:, i])[0]
                    if name in abs2meta and abs2meta[name]['distributed']:
                        pass
                    elif 0 < nzs.size < nprocs:
                        remote_vars[name] = nzs[0]

        return remote_vars

    @collect_errors
    def _setup_var_sizes(self):
        """
        Compute the arrays of variable sizes for all variables/procs on this system.
        """
        self._var_offsets = None
        abs2idx = self._var_allprocs_abs2idx = {}
        all_abs2meta = self._var_allprocs_abs2meta
        self._var_sizes = {
            'input': np.zeros((self.comm.size, len(all_abs2meta['input'])), dtype=INT_DTYPE),
            'output': np.zeros((self.comm.size, len(all_abs2meta['output'])), dtype=INT_DTYPE),
        }

        for subsys in self._subsystems_myproc:
            subsys._setup_var_sizes()

        iproc = self.comm.rank
        for io, sizes in self._var_sizes.items():
            abs2meta = self._var_abs2meta[io]
            for i, name in enumerate(self._var_allprocs_abs2meta[io]):
                abs2idx[name] = i
                if name in abs2meta:
                    sz = abs2meta[name]['size']
                    sizes[iproc, i] = 0 if sz is None else sz

            if self.comm.size > 1:
                my_sizes = sizes[iproc, :].copy()
                self.comm.Allgather(my_sizes, sizes)

        if self.comm.size > 1:
            if (self._has_distrib_vars or self._contains_parallel_group or
                not np.all(self._var_sizes['output']) or
               not np.all(self._var_sizes['input'])):

                if self._distributed_vector_class is not None:
                    self._vector_class = self._distributed_vector_class
                else:
                    raise RuntimeError("{}: Distributed vectors are required but no distributed "
                                       "vector type has been set.".format(self.msginfo))
        else:
            self._vector_class = self._local_vector_class

        self._compute_owning_ranks()

    def _compute_owning_ranks(self):
        abs2meta = self._var_allprocs_abs2meta
        abs2discrete = self._var_allprocs_discrete

        if self.comm.size > 1:
            owns = self._owning_rank
            self._owned_sizes = self._var_sizes['output'].copy()
            abs2idx = self._var_allprocs_abs2idx
            for io in ('input', 'output'):
                sizes = self._var_sizes[io]
                for name, meta in abs2meta[io].items():
                    i = abs2idx[name]
                    for rank in range(self.comm.size):
                        if sizes[rank, i] > 0:
                            owns[name] = rank
                            if io == 'output' and not meta['distributed']:
                                self._owned_sizes[rank + 1:, i] = 0  # zero out all dups
                            break

                if abs2discrete[io]:
                    prefix = self.pathname + '.' if self.pathname else ''
                    for rank, names in enumerate(self.comm.allgather(self._var_discrete[io])):
                        if prefix:
                            toadd = {prefix + n for n in names}.difference(owns)
                        else:
                            toadd = set(names).difference(owns)
                        for n in toadd:
                            owns[n] = rank
        else:
            self._owned_sizes = self._var_sizes['output']

    def _setup_global_connections(self, conns=None):
        """
        Compute dict of all connections between this system's inputs and outputs.

        Parameters
        ----------
        conns : dict
            Dictionary of connections passed down from parent group.
        """
        global_abs_in2out = self._conn_global_abs_in2out = {}

        allprocs_prom2abs_list_in = self._var_allprocs_prom2abs_list['input']
        allprocs_prom2abs_list_out = self._var_allprocs_prom2abs_list['output']

        allprocs_discrete_in = self._var_allprocs_discrete['input']
        allprocs_discrete_out = self._var_allprocs_discrete['output']

        abs_in2prom_info = self._problem_meta['abs_in2prom_info']

        pathname = self.pathname

        abs_in2out = {}
        new_conns = {}

        if pathname:
            path_len = len(pathname) + 1
            nparts = len(pathname.split('.'))
        else:
            path_len = nparts = 0

        if conns is not None:
            for abs_in, abs_out in conns.items():
                inparts = abs_in.split('.')
                outparts = abs_out.split('.')

                if inparts[:nparts] == outparts[:nparts]:
                    global_abs_in2out[abs_in] = abs_out

                    # if connection is contained in a subgroup, add to conns
                    # to pass down to subsystems.
                    if inparts[nparts] == outparts[nparts]:
                        if inparts[nparts] not in new_conns:
                            new_conns[inparts[nparts]] = {}
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

        src_ind_inputs = set()
        abs2meta = self._var_abs2meta['input']
        allprocs_abs2meta_in = self._var_allprocs_abs2meta['input']

        # Add explicit connections (only ones declared by this group)
        for prom_in, (prom_out, src_indices, flat) in self._manual_connections.items():

            # throw an exception if either output or input doesn't exist
            # (not traceable to a connect statement, so provide context)
            if not (prom_out in allprocs_prom2abs_list_out or prom_out in allprocs_discrete_out):
                if (prom_out in allprocs_prom2abs_list_in or prom_out in allprocs_discrete_in):
                    msg = f"{self.msginfo}: Attempted to connect from '{prom_out}' to " + \
                          f"'{prom_in}', but '{prom_out}' is an input. " + \
                          "All connections must be from an output to an input."
                else:
                    guesses = get_close_matches(prom_out, list(allprocs_prom2abs_list_out.keys()) +
                                                list(allprocs_discrete_out.keys()))
                    msg = f"{self.msginfo}: Attempted to connect from '{prom_out}' to " + \
                          f"'{prom_in}', but '{prom_out}' doesn't exist. Perhaps you meant " + \
                          f"to connect to one of the following outputs: {guesses}."
                self._collect_error(msg)
                continue

            if not (prom_in in allprocs_prom2abs_list_in or prom_in in allprocs_discrete_in):
                if (prom_in in allprocs_prom2abs_list_out or prom_in in allprocs_discrete_out):
                    msg = f"{self.msginfo}: Attempted to connect from '{prom_out}' to " + \
                          f"'{prom_in}', but '{prom_in}' is an output. " + \
                          "All connections must be from an output to an input."
                else:
                    guesses = get_close_matches(prom_in, list(allprocs_prom2abs_list_in.keys()) +
                                                list(allprocs_discrete_in.keys()))
                    msg = f"{self.msginfo}: Attempted to connect from '{prom_out}' to " + \
                          f"'{prom_in}', but '{prom_in}' doesn't exist. Perhaps you meant " + \
                          f"to connect to one of the following inputs: {guesses}."
                self._collect_error(msg)
                continue

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
                    self._collect_error(
                        f"{self.msginfo}: Output and input are in the same System for connection "
                        f"from '{prom_out}' to '{prom_in}'.")
                    continue

                if src_indices is not None:
                    a2m = allprocs_abs2meta_in[abs_in]
                    if (a2m['shape_by_conn'] or a2m['copy_shape']):
                        self._collect_error(
                            f"{self.msginfo}: Setting of 'src_indices' along with 'shape_by_conn' "
                            f"or 'copy_shape' for variable '{abs_in}' is currently unsupported.")
                        continue

                    if abs_in in abs2meta:
                        if abs_in not in abs_in2prom_info:
                            abs_in2prom_info[abs_in] = [None for s in abs_in.split('.')]
                        # place a _PromotesInfo at the top level to handle the src_indices
                        if abs_in2prom_info[abs_in][0] is None:
                            try:
                                abs_in2prom_info[abs_in][0] = _PromotesInfo(src_indices=src_indices,
                                                                            flat=flat, prom=abs_in)
                            except Exception:
                                type_exc, exc, tb = sys.exc_info()
                                self._collect_error(
                                    f"When connecting from '{prom_out}' to '{prom_in}': {exc}",
                                    exc_type=type_exc, tback=tb, ident=(abs_out, abs_in))
                                continue

                        meta = abs2meta[abs_in]
                        if meta['src_indices'] is not None:
                            self._collect_error(
                                f"{self.msginfo}: src_indices has been defined in both "
                                f"connect('{prom_out}', '{prom_in}') and "
                                f"add_input('{prom_in}', ...).")

                        meta['manual_connection'] = True
                        meta['src_indices'] = src_indices
                        meta['flat_src_indices'] = flat

                    src_ind_inputs.add(abs_in)

                if abs_in in abs_in2out:
                    self._collect_error(
                        f"{self.msginfo}: Input '{abs_in}' cannot be connected to '{abs_out}' "
                        f"because it's already connected to '{abs_in2out[abs_in]}'.",
                        ident=(abs_out, abs_in))
                    continue

                abs_in2out[abs_in] = abs_out

                # if connection is contained in a subgroup, add to conns to pass down to subsystems.
                if inparts[:nparts + 1] == outparts[:nparts + 1]:
                    if inparts[nparts] not in new_conns:
                        new_conns[inparts[nparts]] = {}
                    new_conns[inparts[nparts]][abs_in] = abs_out

        # Compute global_abs_in2out by first adding this group's contributions,
        # then adding contributions from systems above/below, then allgathering.
        conn_list = list(global_abs_in2out.items())
        conn_list.extend(abs_in2out.items())
        global_abs_in2out.update(abs_in2out)

        for subgroup in self._subgroups_myproc:
            if subgroup.name in new_conns:
                subgroup._setup_global_connections(conns=new_conns[subgroup.name])
            else:
                subgroup._setup_global_connections()
            global_abs_in2out.update(subgroup._conn_global_abs_in2out)
            conn_list.extend(subgroup._conn_global_abs_in2out.items())

        if len(conn_list) > len(global_abs_in2out):
            dupes = [n for n, val in Counter(tgt for tgt, _ in conn_list).items() if val > 1]
            dup_info = defaultdict(set)
            for tgt, src in conn_list:
                for dup in dupes:
                    if tgt == dup:
                        dup_info[tgt].add(src)
            dup_info = [(n, srcs) for n, srcs in dup_info.items() if len(srcs) > 1]
            if dup_info:
                dup = ["%s from %s" % (tgt, sorted(srcs)) for tgt, srcs in dup_info]
                self._collect_error(f"{self.msginfo}: The following inputs have multiple "
                                    f"connections: {', '.join(dup)}.")

        if self.comm.size > 1 and self._mpi_proc_allocator.parallel:
            # If running in parallel, allgather
            if self._subsystems_myproc and self._subsystems_myproc[0].comm.rank == 0:
                raw = (global_abs_in2out, src_ind_inputs)
            else:
                raw = ({}, ())
            gathered = self.comm.allgather(raw)

            all_src_ind_ins = set()
            for myproc_global_abs_in2out, src_ind_ins in gathered:
                global_abs_in2out.update(myproc_global_abs_in2out)
                all_src_ind_ins.update(src_ind_ins)
            src_ind_inputs = all_src_ind_ins

        for inp in src_ind_inputs:
            allprocs_abs2meta_in[inp]['has_src_indices'] = True

    def _setup_dynamic_shapes(self):
        """
        Add shape/size metadata for variables that were created with shape_by_conn or copy_shape.
        """
        all_abs2meta_out = self._var_allprocs_abs2meta['output']
        all_abs2meta_in = self._var_allprocs_abs2meta['input']

        def get_group_input_shape(prom, gshapes):
            if prom in gshapes:
                return gshapes[prom]

            if prom in self._group_inputs:
                for d in self._group_inputs[prom]:
                    if 'src_shape' in d:
                        return d['src_shape']
                    elif 'val' in d:
                        return np.asarray(d['val']).shape

        def copy_var_meta(from_var, to_var, distrib_sizes, gshapes):
            # copy size/shape info from from_var's metadata to to_var's metadata

            if to_var.startswith('#'):
                return

            nprocs = self.comm.size
            to_io = 'output' if to_var in all_abs2meta_out else 'input'

            if from_var.startswith('#'):
                from_dist = False
                from_shape = gshapes[from_var[1:]]
                from_size = shape_to_len(from_shape)
            else:
                from_io = 'output' if from_var in all_abs2meta_out else 'input'

                # transfer shape/size info from from_var to to_var
                all_from_meta = self._var_allprocs_abs2meta[from_io][from_var]
                from_dist = nprocs > 1 and all_from_meta['distributed']
                from_size = all_from_meta['size']
                from_shape = all_from_meta['shape']

            all_to_meta = self._var_allprocs_abs2meta[to_io][to_var]
            to_meta = self._var_abs2meta[to_io].get(to_var, {})

            to_dist = nprocs > 1 and all_to_meta['distributed']

            # known dist output to/from non-distributed input.  We don't allow this case because
            # non-distributed variables must have the same value on all procs and the only way
            # this is possible is if the src_indices on each proc are identical, but that's not
            # possible if we assume 'always local' transfer (see POEM 46).
            if from_dist and not to_dist:
                if from_io == 'output':
                    self._collect_error(
                        f"{self.msginfo}: dynamic sizing of non-distributed {to_io} '{to_var}' "
                        f"from distributed {from_io} '{from_var}' is not supported.")
                    return
                else:  # serial_out <- dist_in
                    # all input rank sizes must be the same
                    if not np.all(distrib_sizes[from_var] == distrib_sizes[from_var][0]):
                        if from_io == 'output':
                            ident = (from_var, to_var)
                        else:
                            ident = (to_var, from_var)
                        self._collect_error(
                            f"{self.msginfo}: dynamic sizing of non-distributed {to_io} '{to_var}' "
                            f"from distributed {from_io} '{from_var}' is not supported because not "
                            f"all {from_var} ranks are the same size "
                            f"(sizes={distrib_sizes[from_var]}).", ident=ident)
                        return

            all_to_meta['shape'] = from_shape
            all_to_meta['size'] = from_size
            if to_meta:
                to_meta['shape'] = from_shape
                to_meta['size'] = from_size
                to_meta['val'] = np.full(from_shape, to_meta['val'])
            if from_var in distrib_sizes:
                distrib_sizes[to_var] = distrib_sizes[from_var]

        all_abs2prom_in = self._var_allprocs_abs2prom['input']
        all_abs2prom_out = self._var_allprocs_abs2prom['output']
        nprocs = self.comm.size
        conn = self._conn_global_abs_in2out
        rev_conn = None

        def get_rev_conn():
            # build reverse connection dict (src: tgts)
            rev = {}
            for tgt, src in conn.items():
                if src in rev:
                    rev[src].append(tgt)
                else:
                    rev[src] = [tgt]
            return rev

        self._shapes_graph = graph = nx.Graph()
        self._shape_knowns = knowns = set()
        dist_sz = {}  # local distrib sizes
        my_abs2meta_out = self._var_abs2meta['output']
        my_abs2meta_in = self._var_abs2meta['input']
        grp_shapes = {}

        # find all variables that have an unknown shape (across all procs) and connect them
        # to other unknown and known shape variables to form an undirected graph.
        for io in ('input', 'output'):
            for name, meta in self._var_allprocs_abs2meta[io].items():
                if meta['shape_by_conn']:
                    if name in conn:  # it's a connected input
                        abs_from = conn[name]
                        graph.add_edge(name, abs_from)
                        if all_abs2meta_out[abs_from]['shape'] is not None:
                            knowns.add(abs_from)
                    else:
                        if rev_conn is None:
                            rev_conn = get_rev_conn()
                        if name in rev_conn:  # connected output
                            for inp in rev_conn[name]:
                                graph.add_edge(name, inp)
                                if all_abs2meta_in[inp]['shape'] is not None:
                                    knowns.add(inp)
                        elif not meta['copy_shape']:
                            # check to see if we can get shape from _group_inputs
                            fail = True
                            if io == 'input':
                                prom = all_abs2prom_in[name]
                                grp_shape = get_group_input_shape(prom, grp_shapes)
                                if grp_shape is not None:
                                    # use '#' to designate this as an entry that's not a variable
                                    gnode = f"#{prom}"
                                    graph.add_edge(gnode, name)
                                    knowns.add(gnode)
                                    grp_shapes[prom] = grp_shape
                                    fail = False
                                else:  # see if there are any connected inputs with known shape
                                    for n in self._var_allprocs_prom2abs_list['input'][prom]:
                                        if n != name:
                                            m = all_abs2meta_in[n]
                                            if not m['distributed'] and not m['has_src_indices']:
                                                if not m['shape_by_conn'] and not m['copy_shape']:
                                                    fail = False
                                                    knowns.add(n)
                                                    graph.add_edge(n, name)
                                                    break
                            if fail:
                                self._collect_error(
                                    f"{self.msginfo}: 'shape_by_conn' was set for "
                                    f"unconnected variable '{name}'.")

                if meta['copy_shape']:
                    # variable whose shape is being copied must be on the same component, and
                    # name stored in 'copy_shape' entry must be the relative name.
                    abs_from = name.rsplit('.', 1)[0] + '.' + meta['copy_shape']
                    if abs_from in all_abs2prom_in or abs_from in all_abs2prom_out:
                        graph.add_edge(name, abs_from)
                        # this is unlikely, but a user *could* do it, so we'll check
                        a2m = all_abs2meta_in if abs_from in all_abs2meta_in else all_abs2meta_out
                        if a2m[abs_from]['shape'] is not None:
                            knowns.add(abs_from)
                    else:
                        self._collect_error(f"{self.msginfo}: Can't copy shape of variable "
                                            f"'{abs_from}'. Variable doesn't exist.")

                # store known distributed size info needed for computing shapes
                if nprocs > 1:
                    my_abs2meta = my_abs2meta_in if name in my_abs2meta_in else my_abs2meta_out
                    if name in my_abs2meta:
                        sz = my_abs2meta[name]['size']
                        if sz is not None:
                            dist_sz[name] = sz
                    else:
                        dist_sz[name] = 0

        if graph.order() == 0:
            # we don't have any shape_by_conn or copy_shape variables, so we're done
            return

        if nprocs > 1:
            distrib_sizes = defaultdict(lambda: np.zeros(nprocs, dtype=INT_DTYPE))
            for rank, dsz in enumerate(self.comm.allgather(dist_sz)):
                for n, sz in dsz.items():
                    distrib_sizes[n][rank] = sz
        else:
            distrib_sizes = {}

        unresolved = set()
        seen = knowns.copy()

        for comps in nx.connected_components(graph):
            comp_knowns = knowns.intersection(comps)
            if not comp_knowns:
                # we need at least 1 known node to resolve this component, so we fail.
                # store the list of unresolved nodes so we have the total list at the end.
                unresolved.update(comps)
                continue

            # because comps is a connected component, we only need 1 known node to resolve
            # the rest
            stack = [sorted(comp_knowns)[0]]  # sort to keep error messages consistent
            while stack:
                known = stack.pop()
                if known.startswith('#'):  # it's a non-variable node (group default input)
                    known_shape = grp_shapes[known[1:]]
                    known_dist = False
                else:
                    known_a2m = all_abs2meta_in if known in all_abs2meta_in else all_abs2meta_out
                    known_shape = known_a2m[known]['shape']
                    known_dist = known_a2m[known]['distributed']
                for node in graph.neighbors(known):
                    if node in seen:
                        if node.startswith('#'):
                            shape = grp_shapes[node[1:]]
                            dist = False
                        else:
                            a2m = all_abs2meta_in if node in all_abs2meta_in else all_abs2meta_out
                            shape = a2m[node]['shape']
                            dist = a2m[node]['distributed']

                        # check to see if shapes agree
                        # can't compare shapes if one is dist and other is not. The mismatch
                        # will be caught later in setup_connections in that case.
                        if shape != known_shape and not (dist ^ known_dist):
                            self._collect_error(f"{self.msginfo}: Shape mismatch, {shape} vs. "
                                                f"{known_shape} for variable '{node}' during "
                                                "dynamic shape determination.")
                    else:
                        # transfer the known shape info to the unshaped variable
                        copy_var_meta(known, node, distrib_sizes, grp_shapes)
                        seen.add(node)
                        stack.append(node)

        # save graph info for possible later plotting
        self._shapes_graph = graph
        self._shape_knowns = knowns

        if unresolved:
            unresolved = sorted(unresolved)
            self._collect_error(f"{self.msginfo}: Failed to resolve shapes for {unresolved}. "
                                "To see the dynamic shape dependency graph, "
                                "do 'openmdao view_dyn_shapes <your_py_file>'.")

    @collect_errors
    @check_mpi_exceptions
    def _setup_connections(self):
        """
        Compute dict of all connections owned by this Group.

        Also, check shapes of connected variables.
        """
        # clean up promotion maps since we don't need them any more
        self._promotes_src_indices = None

        abs_in2out = self._conn_abs_in2out = {}
        self._conn_discrete_in2out = {}
        global_abs_in2out = self._conn_global_abs_in2out
        pathname = self.pathname
        allprocs_discrete_in = self._var_allprocs_discrete['input']
        allprocs_discrete_out = self._var_allprocs_discrete['output']

        for subsys in self._subsystems_myproc:
            subsys._setup_connections()

        path_dot = pathname + '.' if pathname else ''
        path_len = len(path_dot)

        allprocs_abs2meta_in = self._var_allprocs_abs2meta['input']
        allprocs_abs2meta_out = self._var_allprocs_abs2meta['output']
        abs2meta_in = self._var_abs2meta['input']
        abs2meta_out = self._var_abs2meta['output']

        nproc = self.comm.size

        # Check input/output units here, and set _has_input_scaling
        # to True for this Group if units are defined and different, or if
        # ref or ref0 are defined for the output.
        for abs_in, abs_out in global_abs_in2out.items():
            # Check that they are in different subsystems of this system.
            out_subsys = abs_out[path_len:].split('.', 1)[0]
            in_subsys = abs_in[path_len:].split('.', 1)[0]
            if out_subsys != in_subsys:
                if abs_in in allprocs_discrete_in:
                    self._conn_discrete_in2out[abs_in] = abs_out
                elif abs_out in allprocs_discrete_out:
                    self._collect_error(
                        f"{self.msginfo}: Can't connect discrete output '{abs_out}' "
                        f"to continuous input '{abs_in}'.", ident=(abs_out, abs_in))
                    continue
                else:
                    abs_in2out[abs_in] = abs_out

                if nproc > 1 and self._vector_class is None:
                    # check for any cross-process data transfer.  If found, use
                    # self._problem_meta['distributed_vector_class'] as our vector class.
                    if (abs_in not in abs2meta_in or abs_out not in abs2meta_out or
                            abs2meta_in[abs_in]['distributed'] or
                            abs2meta_out[abs_out]['distributed']):
                        self._vector_class = self._distributed_vector_class

            # if connected output has scaling then we need input scaling
            if not self._has_input_scaling and not (abs_in in allprocs_discrete_in or
                                                    abs_out in allprocs_discrete_out):
                out_units = allprocs_abs2meta_out[abs_out]['units']
                in_units = allprocs_abs2meta_in[abs_in]['units']

                # if units are defined and different, or if a connected output has any scaling,
                # we need input scaling.
                self._has_input_scaling = self._has_output_scaling or self._has_resid_scaling or \
                    (in_units and out_units and in_units != out_units)

        # check compatability for any discrete connections
        for abs_in, abs_out in self._conn_discrete_in2out.items():
            in_type = self._var_allprocs_discrete['input'][abs_in]['type']
            try:
                out_type = self._var_allprocs_discrete['output'][abs_out]['type']
            except KeyError:
                self._collect_error(
                    f"{self.msginfo}: Can't connect continuous output '{abs_out}' "
                    f"to discrete input '{abs_in}'.", ident=(abs_out, abs_in))
                continue

            if not issubclass(in_type, out_type):
                self._collect_error(
                    f"{self.msginfo}: Type '{out_type.__name__}' of output '{abs_out}' is "
                    f"incompatible with type '{in_type.__name__}' of input '{abs_in}'.",
                    ident=(abs_out, abs_in))

        # check unit/shape compatibility, but only for connections that are
        # either owned by (implicit) or declared by (explicit) this Group.
        # This way, we don't repeat the error checking in multiple groups.

        for abs_in, abs_out in abs_in2out.items():
            all_meta_out = allprocs_abs2meta_out[abs_out]
            all_meta_in = allprocs_abs2meta_in[abs_in]

            # check unit compatibility
            out_units = all_meta_out['units']
            in_units = all_meta_in['units']

            if out_units:
                if not in_units:
                    if not _is_unitless(out_units):
                        msg = f"Output '{abs_out}' with units of '{out_units}' " + \
                            f"is connected to input '{abs_in}' which has no units."
                        issue_warning(msg, prefix=self.msginfo, category=UnitsWarning)
                elif not is_compatible(in_units, out_units):
                    self._collect_error(
                        f"{self.msginfo}: Output units of '{out_units}' for '{abs_out}' "
                        f"are incompatible with input units of '{in_units}' for '{abs_in}'.",
                        ident=(abs_out, abs_in))
                    continue
            elif in_units is not None:
                if not _is_unitless(in_units):
                    msg = f"Input '{abs_in}' with units of '{in_units}' is " + \
                        f"connected to output '{abs_out}' which has no units."
                    issue_warning(msg, prefix=self.msginfo, category=UnitsWarning)

            # check shape compatibility
            if abs_in in abs2meta_in:
                meta_in = abs2meta_in[abs_in]

                # get output shape from allprocs meta dict, since it may
                # be distributed (we want global shape)
                out_shape = all_meta_out['global_shape']

                # get input shape and src_indices from the local meta dict
                # (input is always local)
                if meta_in['distributed']:
                    # if output is non-distributed and input is distributed, make output shape the
                    # full distributed shape, i.e., treat it in this regard as a distributed output
                    out_shape = self._get_full_dist_shape(abs_out, all_meta_out['shape'])

                in_shape = meta_in['shape']
                src_indices = meta_in['src_indices']

                if src_indices is None and out_shape != in_shape:
                    # out_shape != in_shape is allowed if there's no ambiguity in storage order
                    if (in_shape is None or out_shape is None or
                            not array_connection_compatible(in_shape, out_shape)):
                        self._collect_error(
                            f"{self.msginfo}: The source and target shapes do not match or "
                            f"are ambiguous for the connection '{abs_out}' to '{abs_in}'. "
                            f"The source shape is {out_shape} "
                            f"but the target shape is {in_shape}.", ident=(abs_out, abs_in))
                        continue

                elif src_indices is not None:

                    try:
                        shp = (out_shape if all_meta_out['distributed'] else
                               all_meta_out['global_shape'])
                        src_indices.set_src_shape(shp, dist_shape=out_shape)
                        src_indices = src_indices.shaped_instance()
                    except Exception:
                        type_exc, exc, tb = sys.exc_info()
                        s, src, tgt = get_connection_owner(self, abs_in)
                        abs_out = self._conn_global_abs_in2out[tgt]
                        self._collect_error(
                            f"{s.msginfo}: When connecting '{src}' to '{tgt}': {exc}",
                            exc_type=type_exc, tback=tb, ident=(abs_out, abs_in))
                        continue

                    if src_indices.indexed_src_size == 0:
                        continue

                    if src_indices.indexed_src_size != shape_to_len(in_shape):
                        # initial dimensions of indices shape must be same shape as target
                        for idx_d, inp_d in zip(src_indices.indexed_src_shape, in_shape):
                            if idx_d != inp_d:
                                self._collect_error(
                                    f"{self.msginfo}: The source indices {meta_in['src_indices']} "
                                    f"do not specify a valid shape for the connection '{abs_out}' "
                                    f"to '{abs_in}'. The target shape is {in_shape} but indices "
                                    f"are shape {src_indices.indexed_src_shape}.",
                                    ident=(abs_out, abs_in))
                                break
                        else:
                            self._collect_error(
                                f"{self.msginfo}: src_indices shape {src_indices.indexed_src_shape}"
                                f" does not match {abs_in} shape {in_shape}.",
                                ident=(abs_out, abs_in))
                        continue

                    # any remaining dimension of indices must match shape of source
                    if not src_indices._flat_src and (len(src_indices.indexed_src_shape) >
                                                      len(out_shape)):
                        self._collect_error(
                            f"{self.msginfo}: The source indices {meta_in['src_indices']} do not "
                            f"specify a valid shape for the connection '{abs_out}' to '{abs_in}'. "
                            f"The source has {len(out_shape)} dimensions but the indices expect at "
                            f"least {len(src_indices.indexed_src_shape)}.",
                            ident=(abs_out, abs_in))

    def _transfer(self, vec_name, mode, sub=None):
        """
        Perform a vector transfer.

        Parameters
        ----------
        vec_name : str
            Name of the vector RHS on which to perform a transfer.
        mode : str
            Either 'fwd' or 'rev'
        sub : None or str
            If None, perform a full transfer.
            If str, perform a partial transfer to named subsystem for linear Gauss--Seidel.
        """
        xfer = self._transfers[mode]
        if sub in xfer:
            xfer = xfer[sub]
        else:
            if mode == 'fwd' and self._conn_discrete_in2out and vec_name == 'nonlinear':
                self._discrete_transfer(sub)
            return

        vec_inputs = self._vectors['input'][vec_name]

        if mode == 'fwd':
            if xfer is not None:
                if self._has_input_scaling:
                    vec_inputs.scale_to_norm()
                    xfer._transfer(vec_inputs, self._vectors['output'][vec_name], mode)
                    vec_inputs.scale_to_phys()
                else:
                    xfer._transfer(vec_inputs, self._vectors['output'][vec_name], mode)
            if self._conn_discrete_in2out and vec_name == 'nonlinear':
                self._discrete_transfer(sub)

        else:  # rev
            if xfer is not None:
                if self._has_input_scaling:
                    vec_inputs.scale_to_norm(mode='rev')
                    xfer._transfer(vec_inputs, self._vectors['output'][vec_name], mode)
                    vec_inputs.scale_to_phys(mode='rev')
                else:
                    xfer._transfer(vec_inputs, self._vectors['output'][vec_name], mode)

    def _discrete_transfer(self, sub):
        """
        Transfer discrete variables between components.  This only occurs in fwd mode.

        Parameters
        ----------
        sub : None or str
            If None, perform a full transfer.
            If not, perform a partial transfer for linear Gauss--Seidel.
        """
        comm = self.comm
        key = None if sub is None else self._subsystems_allprocs[sub].system.name

        if comm.size == 1:
            for src_sys_name, src, tgt_sys_name, tgt in self._discrete_transfers[key]:
                tgt_sys = self._subsystems_allprocs[tgt_sys_name].system
                src_sys = self._subsystems_allprocs[src_sys_name].system
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
                    sendvars = [(n, discrete_out[n]['val']) for n in remote_send]
                    allprocs_send = comm.gather(sendvars, root=0)
                    if comm.rank == 0:
                        allprocs_dict = {}
                        for i in range(comm.size):
                            allprocs_dict.update(allprocs_send[i])
                        recvs = [{} for i in range(comm.size)]
                        for rname, ranks in allprocs_recv.items():
                            val = allprocs_dict[rname]
                            for i in ranks:
                                recvs[i][rname] = val
                        data = comm.scatter(recvs, root=0)
                    else:
                        data = comm.scatter(None, root=0)
                else:
                    data = None

                for src_sys_name, src, tgt_sys_name, tgt in xfers:
                    tgt_sys, _ = self._subsystems_allprocs[tgt_sys_name]
                    if tgt_sys._is_local:
                        if tgt in tgt_sys._discrete_inputs:
                            abs_src = '.'.join((src_sys_name, src))
                            if data is not None and abs_src in data:
                                src_val = data[abs_src]
                            else:
                                src_sys, _ = self._subsystems_allprocs[src_sys_name]
                                src_val = src_sys._discrete_outputs[src]
                            tgt_sys._discrete_inputs[tgt] = src_val

    def _setup_transfers(self):
        """
        Compute all transfers that are owned by this system.
        """
        self._vector_class.TRANSFER._setup_transfers(self)
        if self._conn_discrete_in2out:
            self._vector_class.TRANSFER._setup_discrete_transfers(self)

    @collect_errors
    def promotes(self, subsys_name, any=None, inputs=None, outputs=None,
                 src_indices=None, flat_src_indices=None, src_shape=None):
        """
        Promote a variable in the model tree.

        Parameters
        ----------
        subsys_name : str
            The name of the child subsystem whose inputs/outputs are being promoted.
        any : Sequence of str or tuple
            A Sequence of variable names (or tuples) to be promoted, regardless
            of if they are inputs or outputs. This is equivalent to the items
            passed via the `promotes=` argument to add_subsystem.  If given as a
            tuple, we use the "promote as" standard of "('real name', 'promoted name')*[]:".
        inputs : Sequence of str or tuple
            A Sequence of input names (or tuples) to be promoted. Tuples are
            used for the "promote as" capability.
        outputs : Sequence of str or tuple
            A Sequence of output names (or tuples) to be promoted. Tuples are
            used for the "promote as" capability.
        src_indices : int or list of ints or tuple of ints or int ndarray or Iterable or None
            This argument applies only to promoted inputs.
            The global indices of the source variable to transfer data from.
            A value of None implies this input depends on all entries of source.
            Default is None. The shapes of the target and src_indices must match,
            and form of the entries within is determined by the value of 'flat_src_indices'.
        flat_src_indices : bool
            This argument applies only to promoted inputs.
            If True, each entry of src_indices is assumed to be an index into the
            flattened source.  Otherwise each entry must be a tuple or list of size equal
            to the number of dimensions of the source.
        src_shape : int or tuple
            Assumed shape of any connected source or higher level promoted input.
        """
        if isinstance(any, str):
            self._collect_error(f"{self.msginfo}: Trying to promote any='{any}', "
                                "but an iterator of strings and/or tuples is required.")
            return
        if isinstance(inputs, str):
            self._collect_error(f"{self.msginfo}: Trying to promote inputs='{inputs}', "
                                "but an iterator of strings and/or tuples is required.")
            return
        if isinstance(outputs, str):
            self._collect_error(f"{self.msginfo}: Trying to promote outputs='{outputs}', "
                                "but an iterator of strings and/or tuples is required.")
            return

        src_shape = shape2tuple(src_shape)

        if src_indices is None:
            prominfo = None
            if flat_src_indices is not None or src_shape is not None:
                issue_warning(f"ignored flat_src_indices and/or src_shape because"
                              " src_indices was not specified.", prefix=self.msginfo,
                              category=UnusedOptionWarning)

        else:
            promoted = inputs if inputs else any
            try:
                src_indices = indexer(src_indices, flat_src=flat_src_indices)
            except Exception:
                type_exc, exc, tb = sys.exc_info()
                self._collect_error(f"{self.msginfo}: When promoting {promoted} from "
                                    f"'{subsys_name}': {exc}", exc_type=type_exc, tback=tb,
                                    ident=(self.pathname, tuple(promoted)))

            if outputs:
                self._collect_error(f"{self.msginfo}: Trying to promote outputs {outputs} while "
                                    f"specifying src_indices {src_indices} is not meaningful.")
                return

            try:
                prominfo = _PromotesInfo(src_indices, flat_src_indices, src_shape)
            except Exception as err:
                lst = []
                if any is not None:
                    lst.extend(any)
                if inputs is not None:
                    lst.extend(inputs)
                self._collect_error(f"{self.msginfo}: When promoting {sorted(lst)}: {err}",
                                    ident=(self.pathname, tuple(lst)))
                return

        subsys = getattr(self, subsys_name)
        if any:
            subsys._var_promotes['any'].extend((a, prominfo) for a in any)
        if inputs:
            subsys._var_promotes['input'].extend((i, prominfo) for i in inputs)
        if outputs:
            subsys._var_promotes['output'].extend((o, None) for o in outputs)

        # check for attempt to promote with different alias
        list_comp = [i if isinstance(i, tuple) else (i, i)
                     for i, _ in subsys._var_promotes['input']]

        for original, new in list_comp:
            for original_inside, new_inside in list_comp:
                if original == original_inside and new != new_inside:
                    self._collect_error("%s: Trying to promote '%s' when it has been aliased to "
                                        "'%s'." % (self.msginfo, original_inside, new))
                    continue

        # if this was called during configure(), mark this group as modified
        if self._problem_meta is not None:
            if self._problem_meta['config_info'] is not None:
                self._problem_meta['config_info']._prom_added(self.pathname, any=any,
                                                              inputs=inputs, outputs=outputs)

    def add_subsystem(self, name, subsys, promotes=None,
                      promotes_inputs=None, promotes_outputs=None,
                      min_procs=1, max_procs=None, proc_weight=1.0, proc_group=None):
        """
        Add a subsystem.

        Parameters
        ----------
        name : str
            Name of the subsystem being added.
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
        proc_group : str or None
            Name of a processor group such that any system with that processor group name
            within the same parent group will be allocated on the same mpi process(es).
            If this is not None, then any other systems sharing the same proc_group must
            have identical values of min_procs, max_procs, and proc_weight or an exception
            will be raised.

        Returns
        -------
        <System>
            The subsystem that was passed in. This is returned to
            enable users to instantiate and add a subsystem at the
            same time, and get the reference back.
        """
        if self._setup_procs_finished:
            raise RuntimeError(f"{self.msginfo}: Cannot call add_subsystem in "
                               "the configure method.")

        if inspect.isclass(subsys):
            raise TypeError(f"{self.msginfo}: Subsystem '{name}' should be an instance, but a "
                            f"{subsys.__name__} class object was found.")

        if name in self._subsystems_allprocs or name in self._static_subsystems_allprocs:
            raise RuntimeError(f"{self.msginfo}: Subsystem name '{name}' is already used.")

        if hasattr(self, name) and not isinstance(getattr(self, name), System):
            # replacing a subsystem is ok (e.g. resetup) but no other attribute
            raise RuntimeError(f"{self.msginfo}: Can't add subsystem '{name}' because an attribute "
                               f"with that name already exits.")

        if not isinstance(subsys, System):
            raise TypeError(f"{self.msginfo}: Subsystem '{name}' should be a System instance, but "
                            f"an instance of type {type(subsys).__name__} was found.")

        if proc_group is not None and not isinstance(proc_group, str):
            raise TypeError(f"{self.msginfo}: proc_group must be a str or None, but is of type "
                            f"'{type(proc_group).__name__}'.")

        match = namecheck_rgx.match(name)
        if match is None or match.group() != name:
            raise NameError(f"{self.msginfo}: '{name}' is not a valid sub-system name.")

        subsys.name = subsys.pathname = name

        if isinstance(promotes, str) or \
           isinstance(promotes_inputs, str) or \
           isinstance(promotes_outputs, str):
            raise RuntimeError(f"{self.msginfo}: promotes must be an iterator of strings and/or "
                               "tuples.")

        prominfo = None

        # Note, the declared order in any of these promotes arguments shouldn't matter. However,
        # the order does matter when using system.promotes during configure. There, you are
        # permitted to promote '*' then promote_to an alias afterwards, but not in the reverse.
        # To make this work, we sort the promotes lists for this subsystem to put the wild card
        # entries at the beginning.
        if promotes:
            subsys._var_promotes['any'] = [(p, prominfo) for p in
                                           sorted(promotes, key=lambda x: '*' not in x)]
        if promotes_inputs:
            subsys._var_promotes['input'] = [(p, prominfo) for p in
                                             sorted(promotes_inputs, key=lambda x: '*' not in x)]
        if promotes_outputs:
            subsys._var_promotes['output'] = [(p, prominfo) for p in
                                              sorted(promotes_outputs, key=lambda x: '*' not in x)]

        if self._static_mode:
            subsystems_allprocs = self._static_subsystems_allprocs
        else:
            subsystems_allprocs = self._subsystems_allprocs

        subsystems_allprocs[subsys.name] = _SysInfo(subsys, len(subsystems_allprocs))

        if not isinstance(min_procs, int) or min_procs < 1:
            raise TypeError(f"{self.msginfo}: min_procs must be an int > 0 but ({min_procs}) was "
                            "given.")
        if max_procs is not None and (not isinstance(max_procs, int) or max_procs < min_procs):
            raise TypeError(f"{self.msginfo}: max_procs must be None or an int >= min_procs but "
                            f"({max_procs}) was given.")
        if isinstance(proc_weight, Number) and proc_weight < 0:
            raise TypeError(f"{self.msginfo}: proc_weight must be a float > 0. but ({proc_weight}) "
                            "was given.")

        self._proc_info[name] = (min_procs, max_procs, proc_weight, proc_group)

        setattr(self, name, subsys)

        return subsys

    def connect(self, src_name, tgt_name, src_indices=None, flat_src_indices=None):
        """
        Connect source src_name to target tgt_name in this namespace.

        Parameters
        ----------
        src_name : str
            Name of the source variable to connect.
        tgt_name : str or [str, ... ] or (str, ...)
            Name of the target variable(s) to connect.
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
        if isinstance(src_indices, str):
            if isinstance(tgt_name, str):
                tgt_name = [tgt_name]
            tgt_name.append(src_indices)
            self._collect_error(f"{self.msginfo}: src_indices must be a slice, int, or index array."
                                f" Did you mean connect('{src_name}', '{tgt_name}')?")
            return

        # if multiple targets are given, recursively connect to each
        if not isinstance(tgt_name, str) and isinstance(tgt_name, Iterable):
            for name in tgt_name:
                self.connect(src_name, name, src_indices, flat_src_indices=flat_src_indices)
            return

        if src_indices is not None:
            try:
                src_indices = indexer(src_indices, flat_src=flat_src_indices)
            except Exception:
                type_exc, exc, tb = sys.exc_info()
                self._collect_error(f"{self.msginfo}: When connecting from '{src_name}' to "
                                    f"'{tgt_name}': {exc}", exc_type=type_exc, tback=tb)
                return

        # target should not already be connected
        for manual_connections in [self._manual_connections, self._static_manual_connections]:
            if tgt_name in manual_connections:
                srcname = manual_connections[tgt_name][0]
                self._collect_error(f"{self.msginfo}: Input '{tgt_name}' is already connected to "
                                    f"'{srcname}'.")
                return

        # source and target should not be in the same system
        if src_name.rsplit('.', 1)[0] == tgt_name.rsplit('.', 1)[0]:
            self._collect_error(f"{self.msginfo}: Output and input are in the same System for "
                                f"connection from '{src_name}' to '{tgt_name}'.")
            return

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
        if self._problem_meta is not None and \
                self._problem_meta['setup_status'] == _SetupStatus.POST_CONFIGURE:
            raise RuntimeError(f"{self.msginfo}: Cannot call set_order in the configure method.")

        # Make sure the new_order is valid. It must contain all subsystems
        # in this model.
        newset = set(new_order)
        if self._static_mode:
            olddict = self._static_subsystems_allprocs
        else:
            olddict = self._subsystems_allprocs
        oldset = set(olddict)

        if oldset != newset:
            msg = []

            missing = oldset - newset
            if missing:
                msg.append("%s: %s expected in subsystem order and not found." %
                           (self.msginfo, sorted(missing)))

            extra = newset - oldset
            if extra:
                msg.append("%s: subsystem(s) %s found in subsystem order but don't exist." %
                           (self.msginfo, sorted(extra)))

            raise ValueError('\n'.join(msg))

        # Don't allow duplicates either.
        if len(newset) < len(new_order):
            dupes = [key for key, val in Counter(new_order).items() if val > 1]
            raise ValueError("%s: Duplicate name(s) found in subsystem order list: %s" %
                             (self.msginfo, sorted(dupes)))

        subsystems = {}  # need a fresh one to keep the right order
        if self._static_mode:
            self._static_subsystems_allprocs = subsystems
        else:
            self._subsystems_allprocs = subsystems

        for i, name in enumerate(new_order):
            sinfo = olddict[name]
            subsystems[name] = sinfo
            sinfo.index = i

        self._order_set = True
        if self._problem_meta is not None:
            # order has been changed so we need a new full setup
            self._problem_meta['setup_status'] = _SetupStatus.PRE_SETUP

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
        if name == '':
            return self

        system = self
        for subname in name.split('.'):
            if subname in system._subsystems_allprocs:
                system = system._subsystems_allprocs[subname].system
            elif subname in system._static_subsystems_allprocs:
                system = system._static_subsystems_allprocs[subname].system
            else:
                return None
        return system

    def _apply_nonlinear(self):
        """
        Compute residuals. The model is assumed to be in a scaled state.
        """
        self._transfer('nonlinear', 'fwd')
        # Apply recursion
        for subsys in self._subsystems_myproc:
            subsys._apply_nonlinear()

        self.iter_count_apply += 1

    def _solve_nonlinear(self):
        """
        Compute outputs. The model is assumed to be in a scaled state.
        """
        name = self.pathname if self.pathname else 'root'

        with Recording(name + '._solve_nonlinear', self.iter_count, self):
            self._nonlinear_solver._solve_with_cache_check()

        # Iteration counter is incremented in the Recording context manager at exit.

    def _guess_nonlinear(self):
        """
        Provide initial guess for states.
        """
        # let any lower level systems do their guessing first
        if self._has_guess:
            for sname, sinfo in self._subsystems_allprocs.items():
                sub = sinfo.system
                # TODO: could gather 'has_guess' information during setup and be able to
                # skip transfer for subs that don't have guesses...
                self._transfer('nonlinear', 'fwd', sname)
                if sub._is_local and sub._has_guess:
                    sub._guess_nonlinear()

        # call our own guess_nonlinear method, after the recursion is done to
        # all the lower level systems and the data transfers have happened
        complex_step = self._inputs._under_complex_step

        if complex_step:
            self._inputs.set_complex_step_mode(False)
            self._residuals.set_complex_step_mode(False)
            self._outputs.set_complex_step_mode(False)

        try:
            if self._discrete_inputs or self._discrete_outputs:
                self.guess_nonlinear(self._inputs, self._outputs, self._residuals,
                                     self._discrete_inputs, self._discrete_outputs)
            else:
                self.guess_nonlinear(self._inputs, self._outputs, self._residuals)
        finally:

            if complex_step:
                self._inputs.set_complex_step_mode(True)
                self._residuals.set_complex_step_mode(True)
                self._outputs.set_complex_step_mode(True)

    def guess_nonlinear(self, inputs, outputs, residuals,
                        discrete_inputs=None, discrete_outputs=None):
        """
        Provide initial guess for states.

        Override this method to set the initial guess for states.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        residuals : Vector
            Unscaled, dimensional residuals written to via residuals[key].
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        discrete_outputs : dict or None
            If not None, dict containing discrete output values.
        """
        pass

    def _iter_call_apply_linear(self):
        """
        Return whether to call _apply_linear on this Group from within linear block GS/Jac.

        Linear block solvers call _apply_linear then _solve_linear (fwd) or _solve_linear then
        _apply_linear (rev) during an iteration.  This will tell those solvers whether they
        should call _apply_linear on this group when they're calling _apply_linear on their
        subsystems.  Note that _apply_linear will still be called from within a subsystem's
        _solve_linear.

        Returns
        -------
        bool
            True if _apply_linear should be called from within a parent _apply_linear.
        """
        return (self._owns_approx_jac and self._jacobian is not None) or \
            self._assembled_jac is not None or not self._linear_solver.does_recursive_applies()

    def _apply_linear(self, jac, rel_systems, mode, scope_out=None, scope_in=None):
        """
        Compute jac-vec product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            If None, use local jacobian, else use assembled jacobian jac.
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
        if self._owns_approx_jac:
            jac = self._jacobian
        elif jac is None and self._assembled_jac is not None:
            jac = self._assembled_jac

        if jac is not None:
            with self._matvec_context(scope_out, scope_in, mode) as vecs:
                d_inputs, d_outputs, d_residuals = vecs
                jac._apply(self, d_inputs, d_outputs, d_residuals, mode)
        # Apply recursion
        else:
            if mode == 'fwd':
                self._transfer('linear', mode)
                if rel_systems is not None:
                    for s in self._subsystems_myproc:
                        if s.pathname not in rel_systems:
                            # zero out dvecs of irrelevant subsystems
                            s._dresiduals.set_val(0.0)

            for subsys in self._subsystems_myproc:
                if rel_systems is None or subsys.pathname in rel_systems:
                    subsys._apply_linear(jac, rel_systems, mode, scope_out, scope_in)

            if mode == 'rev':
                self._transfer('linear', mode)
                if rel_systems is not None:
                    for s in self._subsystems_myproc:
                        if s.pathname not in rel_systems:
                            # zero out dvecs of irrelevant subsystems
                            s._doutputs.set_val(0.0)

    def _solve_linear(self, mode, rel_systems, scope_out=_UNDEFINED, scope_in=_UNDEFINED):
        """
        Apply inverse jac product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.
        scope_out : set, None, or _UNDEFINED
            Outputs relevant to possible lower level calls to _apply_linear on Components.
        scope_in : set, None, or _UNDEFINED
            Inputs relevant to possible lower level calls to _apply_linear on Components.
        """
        if self._owns_approx_jac:
            # No subsolves if we are approximating our jacobian. Instead, we behave like an
            # ExplicitComponent and pass on the values in the derivatives vectors.
            d_outputs = self._doutputs
            d_residuals = self._dresiduals

            if mode == 'fwd':
                if self._has_resid_scaling:
                    with self._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
                        d_outputs.set_vec(d_residuals)
                else:
                    d_outputs.set_vec(d_residuals)

                # ExplicitComponent jacobian defined with -1 on diagonal.
                d_outputs *= -1.0

            else:  # rev
                if self._has_resid_scaling:
                    with self._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
                        d_residuals.set_vec(d_outputs)
                else:
                    d_residuals.set_vec(d_outputs)

                # ExplicitComponent jacobian defined with -1 on diagonal.
                d_residuals *= -1.0

        else:
            self._linear_solver._set_matvec_scope(scope_out, scope_in)
            self._linear_solver.solve(mode, rel_systems)

    def _linearize(self, jac, sub_do_ln=True, rel_systems=ContainsAll()):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            If None, use local jacobian, else use assembled jacobian jac.
        sub_do_ln : bool
            Flag indicating if the children should call linearize on their linear solvers.
        rel_systems : set or ContainsAll
            Set of relevant system pathnames passed in to the model during computation of total
            derivatives.
        """
        if self._jacobian is None:
            self._jacobian = DictionaryJacobian(self)

        self._check_first_linearize()

        # Group finite difference
        if self._owns_approx_jac:

            jac = self._jacobian
            if self.pathname == "":
                for approximation in self._approx_schemes.values():
                    approximation.compute_approximations(self, jac=jac)
            else:
                # When an approximation exists in a submodel (instead of in root), the model is
                # in a scaled state.
                with self._unscaled_context(outputs=[self._outputs]):
                    for approximation in self._approx_schemes.values():
                        approximation.compute_approximations(self, jac=jac)

        else:
            if self._assembled_jac is not None:
                jac = self._assembled_jac

            # Only linearize subsystems if we aren't approximating the derivs at this level.
            for subsys in self._subsystems_myproc:
                if subsys.pathname in rel_systems:
                    do_ln = sub_do_ln and (subsys._linear_solver is not None and
                                           subsys._linear_solver._linearize_children())
                    if len(subsys._subsystems_allprocs) > 0:
                        subsys._linearize(jac, sub_do_ln=do_ln, rel_systems=rel_systems)
                    else:
                        subsys._linearize(jac, sub_do_ln=do_ln)

            # Update jacobian
            if self._assembled_jac is not None:
                self._assembled_jac._update(self)

            if sub_do_ln:
                for subsys in self._subsystems_myproc:
                    if subsys._linear_solver is not None and subsys.pathname in rel_systems:
                        subsys._linear_solver._linearize()

    def _check_first_linearize(self):
        if self._first_call_to_linearize:
            self._first_call_to_linearize = False  # only do this once
            coloring = self._get_coloring() if coloring_mod._use_partial_sparsity else None

            if coloring is not None:
                if not self._coloring_info['dynamic']:
                    coloring._check_config_partial(self)
                self._setup_approx_coloring()
            # TODO: for top level FD, call below is unnecessary, but we need this
            # for some tests that just call run_linearize directly without calling
            # compute_totals.
            elif self._approx_schemes:
                self._setup_approx_partials()

    def approx_totals(self, method='fd', step=None, form=None, step_calc=None):
        """
        Approximate derivatives for a Group using the specified approximation method.

        Parameters
        ----------
        method : str
            The type of approximation that should be used. Valid options include:
            'fd': Finite Difference, 'cs': Complex Step.
        step : float
            Step size for approximation. Defaults to None, in which case, the approximation
            method provides its default value.
        form : str
            Form for finite difference, can be 'forward', 'backward', or 'central'. Defaults to
            None, in which case, the approximation method provides its default value.
        step_calc : str
            Step type for computing the size of the finite difference step. It can be 'abs' for
            absolute, 'rel_avg' for a size relative to the absolute value of the vector input, or
            'rel_element' for a size relative to each value in the vector input. In addition, it
            can be 'rel_legacy' for a size relative to the norm of the vector.  For backwards
            compatibilty, it can be 'rel', which currently defaults to 'rel_legacy', but in the
            future will default to 'rel_avg'. Defaults to None, in which case the approximation
            method provides its default value.
        """
        self._has_approx = True
        self._approx_schemes = {}
        approx_scheme = self._get_approx_scheme(method)

        default_opts = approx_scheme.DEFAULT_OPTIONS

        kwargs = {}
        for name, attr in (('step', step), ('form', form), ('step_calc', step_calc)):
            if attr is not None:
                if name in default_opts:
                    kwargs[name] = attr
                else:
                    raise RuntimeError("%s: '%s' is not a valid option for '%s'" % (self.msginfo,
                                                                                    name, method))

        self._owns_approx_jac = True
        self._owns_approx_jac_meta = kwargs

    def _setup_partials(self):
        """
        Call setup_partials in components.
        """
        self._subjacs_info = info = {}

        for subsys in self._subsystems_myproc:
            subsys._setup_partials()
            info.update(subsys._subjacs_info)

        if self._has_distrib_vars and self._owns_approx_jac:
            # We currently cannot approximate across a group with a distributed component if the
            # inputs are distributed via src_indices.
            for iname, meta in self._var_allprocs_abs2meta['input'].items():
                if meta['has_src_indices'] and \
                   meta['distributed'] and \
                   iname not in self._conn_abs_in2out:
                    msg = "{}: Approx_totals is not supported on a group with a distributed "
                    msg += "component whose input '{}' is distributed using src_indices. "
                    raise RuntimeError(msg.format(self.msginfo, iname))

    def _approx_subjac_keys_iter(self):
        pro2abs = self._var_allprocs_prom2abs_list

        if self._owns_approx_wrt and not self.pathname:
            candidate_wrt = self._owns_approx_wrt
        else:
            candidate_wrt = list(var[0] for var in pro2abs['input'].values())

        from openmdao.core.indepvarcomp import IndepVarComp
        wrt = set()
        ivc = set()
        if self.pathname:  # get rid of any old stuff in here
            self._owns_approx_of = self._owns_approx_wrt = None

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

        if self._owns_approx_of:
            of = set(self._owns_approx_of)
        else:
            of = set(var[0] for var in pro2abs['output'].values())
            # Skip indepvarcomp res wrt other srcs
            of -= ivc

        for key in product(of, wrt.union(of)):
            # Create approximations for the ones we need.
            if self._tot_jac is not None:
                yield key  # get all combos if we're doing total derivs
                continue

            # Skip explicit res wrt outputs
            if key[1] in of and key[1] not in ivc:

                # Support for specifying a desvar as an obj/con.
                if key[1] not in wrt or key[0] == key[1]:
                    continue

            yield key

    def _jac_of_iter(self):
        """
        Iterate over (name, start, end, idxs, dist_sizes) for each 'of' (row) var in the jacobian.

        idxs will usually be the var slice into the full variable in the result array,
        except in cases where _owns_approx__idx has a value for that variable, in which case it'll
        be indices into the variable.

        Yields
        ------
        str
            Name of 'of' variable.
        int
            Starting index.
        int
            Ending index.
        slice or ndarray
            A full slice or indices for the 'of' variable.
        ndarray or None
            Distributed sizes if var is distributed else None
        """
        total = self.pathname == ''

        abs2meta = self._var_allprocs_abs2meta['output']
        abs2idx = self._var_allprocs_abs2idx
        sizes = self._var_sizes['output']
        approx_of_idx = self._owns_approx_of_idx
        responses = self.get_responses(recurse=True)

        if self._owns_approx_of:
            szname = 'global_size' if total else 'size'
            # we're computing totals/semi-totals (vars may not be local)
            start = end = 0
            for of in self._owns_approx_of:

                # Support for constraint aliases.
                if of in responses and responses[of]['alias'] is not None:
                    path = responses[of]['source']
                else:
                    path = of

                meta = abs2meta[path]
                if meta['distributed']:
                    dist_sizes = sizes[:, abs2idx[path]]
                else:
                    dist_sizes = None
                if of in approx_of_idx:
                    end += approx_of_idx[of].indexed_src_size
                    yield path, start, end, approx_of_idx[of].shaped_array().ravel(), dist_sizes
                else:
                    end += abs2meta[path][szname]
                    yield path, start, end, _full_slice, dist_sizes

                start = end
        else:
            yield from super()._jac_of_iter()

    def _jac_wrt_iter(self, wrt_matches=None):
        """
        Iterate over (name, offset, end, vec, idxs, dist_sizes) for each column var in the jacobian.

        Parameters
        ----------
        wrt_matches : set or None
            Only include row vars that are contained in this set.  This will determine what
            the actual offsets are, i.e. the offsets will be into a reduced jacobian
            containing only the matching columns.

        Yields
        ------
        str
            Name of 'wrt' variable.
        int
            Starting index.
        int
            Ending index.
        Vector
            Either the _outputs or _inputs vector.
        slice or ndarray
            A full slice or indices for the 'wrt' variable.
        ndarray or None
            Distributed sizes if var is distributed else None
        """
        total = self.pathname == ''

        if self._owns_approx_wrt:
            sizes = self._var_sizes
            toidx = self._var_allprocs_abs2idx
            abs2meta = self._var_allprocs_abs2meta
            approx_wrt_idx = self._owns_approx_wrt_idx
            local_ins = self._var_abs2meta['input']
            local_outs = self._var_abs2meta['output']

            szname = 'global_size' if total else 'size'

            start = end = 0
            if self.pathname:  # doing semitotals, so include output columns
                for of, _start, _end, _, dist_sizes in self._jac_of_iter():
                    if wrt_matches is None or of in wrt_matches:
                        end += (_end - _start)
                        vec = self._outputs if of in local_outs else None
                        yield of, start, end, vec, _full_slice, dist_sizes
                        start = end

            for wrt in self._owns_approx_wrt:
                if wrt_matches is None or wrt in wrt_matches:
                    io = 'input' if wrt in abs2meta['input'] else 'output'
                    meta = abs2meta[io][wrt]
                    if wrt in local_ins:
                        vec = self._inputs
                    elif wrt in local_outs:
                        vec = self._outputs
                    else:
                        vec = None
                    if wrt in approx_wrt_idx:
                        sub_wrt_idx = approx_wrt_idx[wrt]
                        size = sub_wrt_idx.indexed_src_size
                        sub_wrt_idx = sub_wrt_idx.flat()
                    else:
                        sub_wrt_idx = _full_slice
                        size = abs2meta[io][wrt][szname]
                    end += size
                    dist_sizes = sizes[io][:, toidx[wrt]] if meta['distributed'] else None
                    yield wrt, start, end, vec, sub_wrt_idx, dist_sizes
                    start = end
        else:
            yield from super()._jac_wrt_iter(wrt_matches)

    def _update_wrt_matches(self, info):
        """
        Determine the list of wrt variables that match the wildcard(s) given in declare_coloring.

        Parameters
        ----------
        info : dict
            Coloring metadata dict.
        """
        if not (self._owns_approx_of or self.pathname):
            return

        wrt_color_patterns = info['wrt_patterns']

        info['wrt_matches'] = wrt_colors_matched = set()

        abs2prom = self._var_allprocs_abs2prom
        for _, wrt in self._get_approx_subjac_keys():
            if wrt in wrt_colors_matched:
                continue
            if wrt in abs2prom['output']:
                wrtprom = abs2prom['output'][wrt]
            else:
                wrtprom = abs2prom['input'][wrt]

            if wrt_color_patterns is None:
                wrt_colors_matched.add(wrt)
            else:
                for patt in wrt_color_patterns:
                    if patt == '*' or fnmatchcase(wrtprom, patt):
                        wrt_colors_matched.add(wrt)
                        break

        baselen = len(self.pathname) + 1 if self.pathname else 0
        info['wrt_matches_rel'] = [n[baselen:] for n in wrt_colors_matched]

        if info.get('dynamic') and info['coloring'] is None and self._owns_approx_of:
            if not wrt_colors_matched:
                raise ValueError("{}: Invalid 'wrt' variable(s) specified for colored approx "
                                 "partial options: {}.".format(self.msginfo, wrt_color_patterns))

    def _setup_approx_partials(self):
        """
        Add approximations for all approx derivs.
        """
        self._jacobian = DictionaryJacobian(system=self)

        abs2meta = self._var_allprocs_abs2meta
        info = self._coloring_info
        responses = self.get_responses(recurse=True)

        if info['coloring'] is not None and (self._owns_approx_of is None or
                                             self._owns_approx_wrt is None):
            method = info['method']
        else:
            method = list(self._approx_schemes)[0]

        wrt_matches = self._get_static_wrt_matches()

        approx = self._get_approx_scheme(method)
        # reset the approx if necessary
        approx._wrt_meta = {}
        approx._reset()

        approx_keys = self._get_approx_subjac_keys()
        for key in approx_keys:
            left, right = key
            if left in responses and responses[left]['alias'] is not None:
                left = responses[left]['source']
            if right in responses and responses[right]['alias'] is not None:
                right = responses[right]['source']

            if key in self._subjacs_info:
                meta = self._subjacs_info[key]
            else:
                meta = SUBJAC_META_DEFAULTS.copy()

                if left == right:
                    size = abs2meta['output'][left]['size']
                    meta['rows'] = meta['cols'] = np.arange(size)
                    # All group approximations are treated as explicit components, so we
                    # have a -1 on the diagonal.
                    meta['val'] = np.full(size, -1.0)
                self._subjacs_info[key] = meta

            meta['method'] = method

            meta.update(self._owns_approx_jac_meta)

            if wrt_matches is None or right in wrt_matches:
                self._update_approx_coloring_meta(meta)

            if meta['val'] is None:
                if right in abs2meta['input']:
                    sz = abs2meta['input'][right]['size']
                else:
                    sz = abs2meta['output'][right]['size']
                shape = (abs2meta['output'][left]['size'], sz)
                meta['shape'] = shape
                if meta['rows'] is not None:  # subjac is sparse
                    meta['val'] = np.zeros(len(meta['rows']))
                else:
                    meta['val'] = np.zeros(shape)

            approx.add_approximation(key, self, meta)

        if self.pathname:
            abs_outs = self._var_allprocs_abs2meta['output']
            abs_ins = self._var_allprocs_abs2meta['input']
            # we're taking semi-total derivs for this group. Update _owns_approx_of
            # and _owns_approx_wrt so we can use the same approx code for totals and
            # semi-totals.  Also, the order must match order of vars in the output and
            # input vectors.
            wrtset = set([k[1] for k in approx_keys])
            self._owns_approx_of = list(abs_outs)
            self._owns_approx_wrt = [n for n in chain(abs_outs, abs_ins) if n in wrtset]
            self._owns_approx_jac = True

    def _setup_approx_coloring(self):
        """
        Ensure that if coloring is declared, approximations will be set up.
        """
        if self._coloring_info['coloring'] is not None:
            meta = self._coloring_info
            self.approx_totals(meta['method'], meta.get('step'), meta.get('form'))
        self._setup_approx_partials()

    def _setup_check(self):
        """
        Do any error checking on user's setup, before any other recursion happens.
        """
        info = self._coloring_info
        if (info['static'] or info['dynamic']) and self.pathname != '':
            msg = f"{self.msginfo}: semi-total coloring is currently not supported."
            raise RuntimeError(msg)

    def _update_approx_coloring_meta(self, meta):
        """
        Update metadata for a subjac based on coloring metadata.

        Parameters
        ----------
        meta : dict
            Metadata for a subjac.
        """
        info = self._coloring_info
        meta['coloring'] = True
        for name in ('method', 'step', 'form'):
            if name in info:
                meta[name] = info[name]

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
            systems = [s for s in self._ordered_comp_name_iter()]
        else:
            systems = [s.name for s in self._subsystems_myproc]

        if not comps_only and self.comm.size > 1 and self._mpi_proc_allocator.parallel:
            systems = set()
            for slist in self.comm.allgather(systems):
                systems.update(slist)

        graph.add_nodes_from(systems)

        edge_data = defaultdict(lambda: defaultdict(list))

        for in_abs, src_abs in input_srcs.items():
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
            if comps_only or src_sys != tgt_sys:
                graph.add_edge(src_sys, tgt_sys, conns=edge_data[key])

        return graph

    def _get_auto_ivc_out_val(self, tgts, vars_to_gather, abs2meta_in):
        # all tgts are continuous variables
        # only called from top level group
        info = None
        src_idx_found = []
        max_size = -1
        found_dup = False

        abs_in2prom_info = self._problem_meta['abs_in2prom_info']

        start_val = val = None
        val_shape = None
        chosen_tgt = None

        # first, find the auto_ivc output shape
        loc_tgts = [t for t in tgts if t in abs2meta_in]
        full_tgts = [t for t in loc_tgts if t not in abs_in2prom_info]
        if full_tgts:  # full variable connections without any src_indices
            val_shape = abs2meta_in[full_tgts[0]]['shape']
            for tgt in full_tgts:
                if tgt not in vars_to_gather:
                    found_dup = True
                    chosen_tgt = tgt
                    break
        else:
            plist_tgts = [tgt for tgt in loc_tgts if tgt in abs_in2prom_info]
            if plist_tgts:
                plists = [abs_in2prom_info[tgt] for tgt in plist_tgts]
                plens = [len(plist) for plist in plists]
                nlevels = max(plens)
                # find highest specification of src_shape in bfs order to shape the auto_ivc output
                for i in range(nlevels):
                    for tgt, plist, plen in zip(plist_tgts, plists, plens):
                        if i < plen:
                            pinfo = plist[i]
                            if pinfo is not None and pinfo.src_shape is not None:
                                val_shape = pinfo.src_shape
                                break
                    if val_shape is not None:
                        chosen_tgt = tgt
                        break

        if val_shape is not None:
            start_val = val = np.ones(val_shape)

        info = None
        for tgt in tgts:
            # if tgt in vars_to_gather and tgt not in abs2meta_in:
            if tgt in vars_to_gather and self.comm.rank != vars_to_gather[tgt]:
                if info is None or max_size < 0:
                    info = (tgt, np.zeros(0), True)
                    max_size = 0
                continue

            # if we get here, tgt is local
            meta = abs2meta_in[tgt]
            size = meta['size']
            value = meta['val']
            src_indices = None
            if tgt in abs_in2prom_info:
                # traverse down the promotes list, (abs_in2prom_info[tgt]), to get the
                # final src_indices down at the component level so we can set the value of
                # that component input into the appropriate place(s) in the auto_ivc output.
                # If a tgt has no src_indices anywhere, it will not be found in
                # abs_in2prom_info.
                newshape = val_shape
                for pinfo in abs_in2prom_info[tgt]:
                    if pinfo is None:
                        continue
                    inds, _, shape = pinfo
                    if inds is not None:
                        if shape is None:
                            shape = newshape
                            if inds._src_shape is None:
                                try:
                                    inds.set_src_shape(shape)
                                except IndexError:
                                    exc_class, exc, tb = sys.exc_info()
                                    self._collect_error(f"When promoting '{pinfo.prom}' from system"
                                                        f" '{pinfo.promoted_from}' with src_indices"
                                                        f" {inds} and src_shape {shape}: {exc}",
                                                        exc_type=exc_class, tback=tb,
                                                        ident=(pinfo.prom, pinfo.promoted_from))

                        if src_indices is None:
                            src_indices = inds
                        else:
                            sinds = convert_src_inds(src_indices, newshape, inds, shape)
                            # final src_indices are wrt original full sized source and are flat,
                            # so use val_shape and flat_src=True
                            src_indices = indexer(sinds, src_shape=val_shape, flat_src=True)
                        newshape = src_indices.indexed_src_shape

            if src_indices is None:
                # for an auto_ivc connection, src_indices here, if they exist, must be coming from
                # an add_input call (this is a deprecated feature and will be removed eventually)
                src_indices = meta['src_indices']

            if src_indices is not None:
                if val is None:
                    if val_shape is None and not found_dup:
                        src_idx_found.append(tgt)
                    val = value
                else:
                    try:
                        if src_indices._flat_src:
                            val.ravel()[src_indices.flat()] = value.flat
                        else:
                            val[src_indices()] = value
                    except Exception as err:
                        src = self._conn_global_abs_in2out[tgt]
                        msg = f"{self.msginfo}: The source indices " + \
                              f"{src_indices} do not specify a " + \
                              f"valid shape for the connection '{src}' to " + \
                              f"'{tgt}'. (target shape=" + \
                              f"{meta['shape']}, indices_shape=" + \
                              f"{src_indices.indexed_src_shape}): {err}"
                        self._collect_error(msg, ident=(src, tgt))
                        continue
            else:
                if val is None:
                    val = value
                elif np.ndim(value) == 0:
                    if val.size > 1:
                        src = self._conn_global_abs_in2out[tgt]
                        self._collect_error(f"Shape of input '{tgt}', (), doesn't match shape "
                                            f"{val.shape}.", ident=(src, tgt))
                        continue
                elif np.squeeze(val).shape != np.squeeze(value).shape:
                    src = self._conn_global_abs_in2out[tgt]
                    self._collect_error(f"Shape of input '{tgt}', {value.shape}, doesn't match "
                                        f"shape {val.shape}.", ident=(src, tgt))
                    continue

                if val is not value:
                    if val.shape:
                        val = np.reshape(value, newshape=val.shape)
                    else:
                        val = value

                if tgt not in vars_to_gather:
                    found_dup = True

            if tgt == chosen_tgt:
                max_size = size
                info = (tgt, val, False)
            elif chosen_tgt is None and size > max_size:
                max_size = size
                info = (tgt, val, False)

            val = start_val

        if src_idx_found:  # auto_ivc connected to local vars with src_indices
            self._collect_error("Attaching src_indices to inputs requires that the shape of the "
                                "source variable is known, but the source shape for inputs "
                                f"{src_idx_found} is unknown. You can specify the src shape for "
                                "these inputs by setting 'val' or 'src_shape' in a call to "
                                "set_input_defaults, or by adding an IndepVarComp as the source.",
                                ident=(self.pathname, tuple(src_idx_found)))
            return None

        return info

    def _setup_auto_ivcs(self, mode):
        # only happens at top level
        from openmdao.core.indepvarcomp import _AutoIndepVarComp

        if self.comm.size > 1 and self._mpi_proc_allocator.parallel:
            raise RuntimeError("The top level system must not be a ParallelGroup.")

        # create the IndepVarComp that will contain all auto-ivc outputs
        self._auto_ivc = auto_ivc = _AutoIndepVarComp()
        auto_ivc.name = '_auto_ivc'
        auto_ivc.pathname = auto_ivc.name

        prom2auto = {}
        count = 0
        auto2tgt = {}
        abs2prom = self._var_allprocs_abs2prom['input']
        all_abs2meta = self._var_allprocs_abs2meta['input']
        conns = self._conn_global_abs_in2out
        auto_conns = {}

        for tgt in all_abs2meta:
            if tgt in conns:
                continue

            all_meta = all_abs2meta[tgt]
            if all_meta['distributed']:
                # OpenMDAO currently can't create an automatic IndepVarComp for inputs on
                # distributed components.
                raise RuntimeError(f'Distributed component input "{tgt}" requires an IndepVarComp.')

            prom = abs2prom[tgt]
            if prom in prom2auto:
                # multiple connected inputs w/o a src. Connect them to the same IVC
                src = prom2auto[prom][0]
                auto_conns[tgt] = src
            else:
                src = f"_auto_ivc.v{count}"
                count += 1
                prom2auto[prom] = (src, tgt)
                auto_conns[tgt] = src

            if src in auto2tgt:
                auto2tgt[src].append(tgt)
            else:
                auto2tgt[src] = [tgt]

        conns.update(auto_conns)
        auto_ivc.auto2tgt = auto2tgt

        abs2meta_in = self._var_abs2meta['input']
        vars2gather = self._vars_to_gather

        for src, tgts in auto2tgt.items():
            prom = self._var_allprocs_abs2prom['input'][tgts[0]]
            ret = self._get_auto_ivc_out_val(tgts, vars2gather, abs2meta_in)
            if ret is None:  # setup error occurred. Try to continue
                continue
            tgt, val, remote = ret
            prom = abs2prom[tgt]
            if prom not in self._group_inputs:
                self._group_inputs[prom] = [{'use_tgt': tgt, 'auto': True, 'path': self.pathname,
                                             'prom': prom}]
            else:
                self._group_inputs[prom][0]['use_tgt'] = tgt
            gmeta = self._group_inputs[prom][0]

            if 'units' in gmeta:
                units = gmeta['units']
            else:
                units = all_abs2meta[tgt]['units']

            if not remote and 'val' in gmeta:
                val = gmeta['val']

            if self.comm.size > 1:
                tgt_local_procs = set()
                # do a preliminary check to avoid the allgather if we can
                for t in tgts:
                    if t in vars2gather:
                        tgt_local_procs.add(vars2gather[t])
                    else:   # t is duplicated in all procs
                        break
                else:
                    if len(tgt_local_procs) < self.comm.size:  # don't have a local var in each proc
                        tgt_local_procs = set()
                        for t in self.comm.allgather(tgt):
                            if t in vars2gather:
                                tgt_local_procs.add(vars2gather[t])
                        if len(tgt_local_procs) > 1:
                            # the 'local' val can only exist on 1 proc (distrib auto_ivcs not
                            # allowed), so must consolidate onto one proc
                            rank = sorted(tgt_local_procs)[0]
                            if rank != self.comm.rank:
                                val = np.zeros(0)
                                remote = True

            relsrc = src.rsplit('.', 1)[-1]
            auto_ivc.add_output(relsrc, val=np.atleast_1d(val), units=units)
            if remote:
                auto_ivc._add_remote(relsrc)

        # have to sort to keep vars in sync because we may be doing bcasts
        for abs_in in sorted(self._var_allprocs_discrete['input']):
            if abs_in not in conns:  # unconnected, so connect the input to an _auto_ivc output
                prom = abs2prom[abs_in]
                val = _UNDEFINED

                if prom in prom2auto:
                    # multiple connected inputs w/o a src. Connect them to the same IVC
                    # check if they have different metadata, and if they do, there must be
                    # a group input defined that sets the default, else it's an error
                    conns[abs_in] = prom2auto[prom][0]
                else:
                    ivc_name = f"_auto_ivc.v{count}"
                    loc_out_name = ivc_name.rsplit('.', 1)[-1]
                    count += 1
                    prom2auto[prom] = (ivc_name, abs_in)
                    conns[abs_in] = ivc_name

                    if abs_in in self._var_abs2prom['input']:  # var is local
                        val = self._var_discrete['input'][abs_in]['val']
                    else:
                        val = None
                    if abs_in in vars2gather:
                        if vars2gather[abs_in] == self.comm.rank:
                            self.comm.bcast(val, root=vars2gather[abs_in])
                        else:
                            val = self.comm.bcast(None, root=vars2gather[abs_in])
                    auto_ivc.add_discrete_output(loc_out_name, val=val)

                src = conns[abs_in]
                if src in auto_ivc.auto2tgt:
                    auto_ivc.auto2tgt[src].append(abs_in)
                else:
                    auto_ivc.auto2tgt[src] = [abs_in]

        if not prom2auto:
            return auto_ivc

        auto_ivc._setup_procs(auto_ivc.pathname, self.comm, mode, self._problem_meta)
        auto_ivc._configure()
        auto_ivc._configure_check()
        auto_ivc._setup_var_data()

        # now update our own data structures based on the new auto_ivc component variables
        old = self._subsystems_allprocs
        self._subsystems_allprocs = allsubs = {}
        allsubs['_auto_ivc'] = _SysInfo(auto_ivc, 0)
        for i, (name, s) in enumerate(old.items()):
            allsubs[name] = s
            s.index = i + 1

        self._subsystems_myproc = [auto_ivc] + self._subsystems_myproc

        io = 'output'  # auto_ivc has only output vars
        old = self._var_allprocs_prom2abs_list[io]
        p2abs = {}
        for name in auto_ivc._var_allprocs_abs2meta[io]:
            p2abs[name] = [name]
        p2abs.update(old)
        self._var_allprocs_prom2abs_list[io] = p2abs

        # auto_ivc never promotes anything
        self._var_abs2prom[io].update({n: n for n in auto_ivc._var_abs2prom[io]})
        self._var_allprocs_abs2prom[io].update({n: n for n in
                                                auto_ivc._var_allprocs_abs2prom[io]})

        self._var_discrete[io].update({'_auto_ivc.' + k: v for k, v in
                                       auto_ivc._var_discrete[io].items()})
        self._var_allprocs_discrete[io].update(auto_ivc._var_allprocs_discrete[io])

        old = self._var_abs2meta[io]
        self._var_abs2meta[io] = {}
        self._var_abs2meta[io].update(auto_ivc._var_abs2meta[io])
        self._var_abs2meta[io].update(old)

        old = self._var_allprocs_abs2meta[io]
        self._var_allprocs_abs2meta[io] = {}
        self._var_allprocs_abs2meta[io].update(auto_ivc._var_allprocs_abs2meta[io])
        self._var_allprocs_abs2meta[io].update(old)

        self._approx_subjac_keys = None  # this will force re-initialization
        self._setup_procs_finished = True

        return auto_ivc

    @collect_errors
    def _resolve_ambiguous_input_meta(self):
        """
        Resolve ambiguous input units and values for auto_ivcs with multiple targets.

        This should only be called on the top level Group.
        """
        all_abs2meta_in = self._var_allprocs_abs2meta['input']
        all_abs2meta_out = self._var_allprocs_abs2meta['output']

        abs2prom = self._var_allprocs_abs2prom['input']
        abs2meta_in = self._var_abs2meta['input']
        all_discrete_outs = self._var_allprocs_discrete['output']
        all_discrete_ins = self._var_allprocs_discrete['input']

        for src, tgts in self._auto_ivc.auto2tgt.items():
            if len(tgts) < 2:
                continue
            if src not in all_discrete_outs:
                smeta = all_abs2meta_out[src]
                sunits = smeta['units'] if 'units' in smeta else None

            sval = self.get_val(src, kind='output', get_remote=True, from_src=False)
            errs = set()

            prom = abs2prom[tgts[0]]
            if prom in self._group_inputs:
                gmeta = self._group_inputs[prom][0]
            else:
                gmeta = {'path': self.pathname, 'prom': prom, 'auto': True}

            for tgt in tgts:
                tval = self.get_val(tgt, kind='input', get_remote=True, from_src=False)

                if tgt in all_discrete_ins:
                    if 'val' not in gmeta and sval != tval:
                        errs.add('val')
                else:
                    tmeta = all_abs2meta_in[tgt]
                    tunits = tmeta['units'] if 'units' in tmeta else None
                    if 'units' not in gmeta and sunits != tunits:

                        # Detect if either Source or Targe units are None.
                        if sunits is None or tunits is None:
                            errs.add('units')

                        elif _find_unit(sunits) != _find_unit(tunits):
                            errs.add('units')

                    if 'val' not in gmeta:
                        if tval.shape == sval.shape:
                            if _has_val_mismatch(tunits, tval, sunits, sval):
                                errs.add('val')
                        else:
                            if all_abs2meta_in[tgt]['has_src_indices'] and tgt in abs2meta_in:
                                if abs2meta_in[tgt]['flat_src_indices']:
                                    srcpart = sval.ravel()[abs2meta_in[tgt]['src_indices'].flat()]
                                else:
                                    srcpart = sval[abs2meta_in[tgt]['src_indices']()]
                                if _has_val_mismatch(tunits, tval, sunits, srcpart):
                                    errs.add('val')

            if errs:
                self._show_ambiguity_msg(prom, errs, tgts)
            elif src not in all_discrete_outs:
                gmeta['units'] = sunits

    def _show_ambiguity_msg(self, prom, metavars, tgts, metadata=None):
        errs = sorted(metavars)
        if metadata is None:
            meta = errs
        else:
            meta = sorted(metadata)
        inputs = sorted(tgts)
        gpath = common_subpath(tgts)
        if gpath == self.pathname:
            g = self
        else:
            g = self._get_subsystem(gpath)
        gprom = None

        # get promoted name relative to g
        if MPI and self.comm.size > 1:
            if g is not None and not g._is_local:
                g = None
            if self.comm.allreduce(int(g is not None)) < self.comm.size:
                # some procs have remote g
                if g is not None:
                    gprom = g._var_allprocs_abs2prom['input'][inputs[0]]
                proms = self.comm.allgather(gprom)
                for p in proms:
                    if p is not None:
                        gprom = p
                        break
        if gprom is None:
            gprom = g._var_allprocs_abs2prom['input'][inputs[0]]

        gname = f"Group named '{gpath}'" if gpath else 'model'
        args = ', '.join([f'{n}=?' for n in errs])
        self._collect_error(f"{self.msginfo}: The following inputs, {inputs}, promoted "
                            f"to '{prom}', are connected but their metadata entries {meta}"
                            f" differ. Call <group>.set_input_defaults('{gprom}', {args}), "
                            f"where <group> is the {gname} to remove the ambiguity.")

    def _ordered_comp_name_iter(self):
        """
        Yield contained component pathnames in order of execution.

        For components within ParallelGroups, true execution order is unknown so components
        will be ordered by rank within a ParallelGroup.
        """
        if self._mpi_proc_allocator.parallel and self.comm.size > 1:
            names = []
            for s in self._subsystems_myproc:
                if isinstance(s, Group):
                    names.extend(s._ordered_comp_name_iter())
                else:
                    names.append(s.pathname)
            seen = set()
            for ranknames in self.comm.allgather(names):
                for name in ranknames:
                    if name not in seen:
                        yield name
                        seen.add(name)
        else:
            for s in self._subsystems_myproc:
                if isinstance(s, Group):
                    yield from s._ordered_comp_name_iter()
                else:
                    yield s.pathname
