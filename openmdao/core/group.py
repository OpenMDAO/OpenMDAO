"""Define the Group class."""
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable

from itertools import product, chain
from numbers import Number
import inspect

import numpy as np
import networkx as nx

from openmdao.core.configinfo import _ConfigInfo
from openmdao.core.system import System, collect_errors
from openmdao.core.component import Component, _DictValues
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.core.constants import _UNDEFINED, INT_DTYPE, _SetupStatus
from openmdao.vectors.vector import _full_slice
from openmdao.proc_allocators.default_allocator import DefaultAllocator, ProcAllocationError
from openmdao.jacobians.subjac import SUBJAC_META_DEFAULTS
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce
from openmdao.solvers.linear.linear_runonce import LinearRunOnce
from openmdao.solvers.linear.direct import DirectSolver
from openmdao.utils.array_utils import _flatten_src_indices, \
    ValueRepeater
from openmdao.utils.general_utils import shape2tuple, ensure_compatible, \
    meta2src_iter, is_undefined, env_truthy
from openmdao.utils.units import unit_conversion, simplify_unit, _find_unit
from openmdao.utils.graph_utils import get_out_of_order_nodes, get_sccs_topo, \
    get_unresolved_knowns, is_unresolved, get_active_edges, are_connected
from openmdao.utils.mpi import MPI, check_mpi_exceptions  # , translate_ranks
import openmdao.utils.coloring as coloring_mod
from openmdao.utils.indexer import indexer, Indexer
from openmdao.utils.relevance import get_relevance
from openmdao.utils.om_warnings import issue_warning, UnusedOptionWarning, \
    PromotionWarning, MPIWarning, DerivativesWarning
from openmdao.utils.class_util import overrides_method
from openmdao.utils.jax_utils import jax
from openmdao.core.total_jac import _TotalJacInfo
from openmdao.utils.name_maps import LOCAL, CONTINUOUS, DISTRIBUTED
from openmdao.jacobians.dictionary_jacobian import DictionaryJacobian
from openmdao.jacobians.subjac import Subjac
from openmdao.jacobians.jacobian import GroupJacobianUpdateContext
from openmdao.visualization.conn_graph import AllConnGraph
from openmdao.utils.indexer import idx_list_to_index_array


# regex to check for valid names.
import re
namecheck_rgx = re.compile('[a-zA-Z][_a-zA-Z0-9]*')


dump_graphs = env_truthy('DUMP_GRAPHS')

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


def _chk_scale_factor(factor):
    try:
        if factor == (0.0, 1.0):
            return None
    except ValueError:
        pass
    return factor


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
        List of local subgroups, (sorted by name if Problem option allow_post_setup_reorder is
        True).
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
        List of Auto IVC warnings to be raised later.
    _shapes_graph : nx.Graph
        Dynamic shape dependency graph, or None.
    _pre_components : set of str or None
        Set of pathnames of components that are executed prior to the optimization loop.  Empty
        unless the 'group_by_pre_opt_post' option is True in the Problem.
    _post_components : set of str or None
        Set of pathnames of components that are executed after the optimization loop.  Empty
        unless the 'group_by_pre_opt_post' option is True in the Problem.
    _iterated_components : set of str or ContainsAll
        Set of pathnames of components that are executed in the optimization loop if
        'group_by_pre_opt_post' is True in the Problem.
    _fd_rev_xfer_correction_dist : dict
        If this group is using finite difference to compute derivatives,
        this is the set of inputs that are upstream of a distributed response
        within this group, keyed by active response.  These determine if contributions
        from all ranks will be added together to get the correct input values when derivatives
        in the larger model are being solved using reverse mode.
    _auto_ivc_recorders : list
        List of recorders that were added to _auto_ivc before it existed so they can be added
        after _auto_ivc is created.
    _is_explicit : bool or None
        True if neither this Group nor any of its descendants contains implicit systems or cycles.
    _sys_graph_cache : dict
        Cache for the system graph.
    _key_owner : dict
        The owning rank keyed by absolute jacobian key.
    _var_existence : dict or None
        Keeps track of which ranks each variable exists on.
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
        # self._static_conn_graph = AllConnGraph()
        self._conn_graph = None
        self._conn_abs_in2out = {}
        self._conn_discrete_in2out = {}
        self._transfers = {}
        self._discrete_transfers = {}
        self._setup_procs_finished = False
        self._contains_parallel_group = False
        self._order_set = False
        self._shapes_graph = None
        self._pre_components = None
        self._post_components = None
        self._iterated_components = None
        self._fd_rev_xfer_correction_dist = {}
        self._auto_ivc_recorders = []
        self._is_explicit = None
        self._sys_graph_cache = None
        self._key_owner = None
        self._var_existence = None

        # TODO: we cannot set the solvers with property setters at the moment
        # because our lint check thinks that we are defining new attributes
        # called nonlinear_solver and linear_solver without documenting them.
        if not self._nonlinear_solver:
            self._nonlinear_solver = NonlinearRunOnce()
        if not self._linear_solver:
            self._linear_solver = LinearRunOnce()

        self.options.declare('auto_order', types=bool, default=False,
                             desc='If True the order of subsystems is determined automatically '
                             'based on the dependency graph.  It will not break or reorder '
                             'cycles.')

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
        meta = {} # {'prom': name, 'auto': False}
        if is_undefined(val):
            src_shape = shape2tuple(src_shape)
            meta['val'] = None
        else:
            if src_shape is not None:
                # make sure value and src_shape are compatible
                val, src_shape = ensure_compatible(name, val, src_shape)
            elif isinstance(val, np.ndarray):
                src_shape = val.shape
            elif isinstance(val, Number):
                src_shape = (1,)
            meta['val'] = val

        if units is not None:
            if not isinstance(units, str):
                raise TypeError('%s: The units argument should be a str or None' % self.msginfo)
            meta['units'] = simplify_unit(units, msginfo=self.msginfo)
        else:
            meta['units'] = None

        meta['src_shape'] = src_shape

        if self._static_mode:
            dct = self._static_group_inputs
        else:
            dct = self._group_inputs

        if name in dct:
            old = dct[name]
            oldnames = [k for k, v in old.items() if v is not None]
            overlap = set(oldnames).intersection(k for k, v in meta.items() if v is not None)
            if overlap:
                issue_warning(f"Setting input defaults for input '{name}' which "
                              f"override previously set defaults for {sorted(overlap)}.",
                              prefix=self.msginfo, category=PromotionWarning)
            old.update(meta)
        else:
            dct[name] = meta

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

            # All inputs that are connected to outputs within this group.  Use only local inputs
            # because only local inputs exist in our input vector.
            scope_in = frozenset(self._conn_global_abs_in2out).intersection(
                self._var_abs2meta['input'])

        else:
            # Empty for the excl_sub
            scope_out = frozenset()

            # All inputs connected to an output in this Group but not in excl_sub.
            # allins is used to filter out discrete variables that might be found in
            # self._conn_global_abs_in2out.
            allins = self._var_abs2meta['input']
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
        scale_factors = {}
        graph = self._get_conn_graph()

        if self._has_output_scaling or self._has_resid_scaling:
            for abs_name, meta in self._var_allprocs_abs2meta['output'].items():
                ref0 = meta['ref0']
                a0 = ref0
                a1 = meta['ref'] - ref0
                if _chk_scale_factor((a0, a1)) is not None:
                    scale_factors[abs_name] = {'output': (a0, a1, None, None)}

                res_ref = meta['res_ref']
                try:
                    if res_ref == 1.0:
                        res_ref = None
                except ValueError:
                    pass

                if res_ref is not None:
                    if abs_name not in scale_factors:
                        scale_factors[abs_name] = {'residual': (0.0, res_ref, None, None)}
                    else:
                        scale_factors[abs_name]['residual'] = (0.0, res_ref, None, None)

        # Input scaling for connected inputs is added here.
        # This is a combined scale factor that includes the scaling of the connected source
        # and the unit conversion between the source output and each target input.
        if self._has_input_scaling:
            allprocs_meta_out = self._var_allprocs_abs2meta['output']
            for abs_in, meta_in in self._var_abs2meta['input'].items():
                src = self._conn_global_abs_in2out[abs_in]
                in_node_meta = graph.nodes[('i', abs_in)]
                src_node_meta = graph.nodes[('o', src)]

                src_meta = allprocs_meta_out[src]
                ref = src_meta['ref']
                ref0 = src_meta['ref0']

                scalar_ref = np.ndim(ref) == 0
                scalar_ref0 = np.ndim(ref0) == 0

                has_scaling = not scalar_ref or not scalar_ref0 or ref != 1.0 or ref0 != 0.0

                units_in = in_node_meta.units
                units_out = src_node_meta.units

                has_unit_conv = \
                    units_in is not None and units_out is not None and units_in != units_out

                if has_unit_conv:
                    try:
                        factor, offset = unit_conversion(units_out, units_in)
                    except Exception as err:
                        self._collect_error(f"{self.msginfo}: When connecting '{src}' to '{abs_in}': {err}")
                    if factor == 1.0 and offset == 0.0:
                        has_unit_conv = False
                else:
                    factor = 1.0
                    offset = 0.0

                if not has_scaling and not has_unit_conv:
                    continue

                # units_in = meta_in['units']
                # units_out = src_meta['units']

                src_inds_list = meta_in['src_inds_list']

                if src_inds_list is not None:
                    if not (scalar_ref and scalar_ref0):
                        # TODO: if either ref or ref0 are not scalar and the output is
                        # distributed, we need to do a scatter
                        # to obtain the values needed due to global src_indices
                        if src_node_meta.distributed:
                            raise RuntimeError("{}: vector scalers with distrib vars "
                                               "not supported yet.".format(self.msginfo))

                        src_indices = idx_list_to_index_array(src_inds_list)
                        if not np.ndim(src_indices) < 2:
                            src_indices = _flatten_src_indices(src_indices,
                                                               in_node_meta.shape,
                                                               src_node_meta.global_shape,
                                                               src_node_meta.global_size)

                        if not scalar_ref:
                            ref = ref[src_indices]
                        else:  # ref is scalar so ref0 must be an array
                            ref = np.full(ref0.shape, ref)
                        if not scalar_ref0:
                            ref0 = ref0[src_indices]
                        else:  # ref0 is scalar so ref must be an array
                            ref0 = np.full(ref.shape, ref0)

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

                a0 = ref0
                a1 = ref - ref0

                if units_in is None or units_out is None or units_in == units_out:
                    if _chk_scale_factor((a0, a1)) is None:
                        continue
                    # No unit conversion, only scaling. Just send the scale factors.
                    scale_factors[abs_in] = {'input': (a0, a1, None, None)}
                else:
                    factor, offset = unit_conversion(units_out, units_in)

                    # Send both unit scaling and solver scaling. Linear input vectors need to
                    # treat them differently in reverse mode.
                    scale_factors[abs_in] = {'input': (a0, a1, factor, offset)}

                    # For adder allocation check.
                    a0 = (ref0 + offset) * factor

                # Check whether we need to allocate an adder for the input vector.
                self._has_input_adder |= np.any(np.asarray(a0))

        return scale_factors

    def _configure(self):
        """
        Configure our model recursively to assign any children settings.

        Highest system's settings take precedence.
        """
        # reset group_inputs back to what it was just after self.setup() in case _configure
        # is called multiple times.
        self._group_inputs = self._pre_config_group_inputs.copy()

        self.matrix_free = False
        self._has_guess = overrides_method('guess_nonlinear', self, Group)

        for subsys in self._sorted_subsystems_myproc:
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

    def _reset_setup_vars(self):
        """
        Reset all the stuff that gets initialized in setup.
        """
        super()._reset_setup_vars()
        self._setup_procs_finished = False

    def _setup_procs(self, pathname, comm, prob_meta):
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
        prob_meta : dict
            Problem level metadata.
        """
        super()._setup_procs(pathname, comm, prob_meta)

        nproc = comm.size

        if self._num_par_fd > 1:
            info = self._coloring_info
            if comm.size > 1:
                # if approx_totals has been declared, or there is an approx coloring, setup par FD
                if self._owns_approx_jac or info.dynamic or info.static is not None:
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

        if self.pathname == '':
            self._conn_graph = AllConnGraph()

        # self._get_conn_graph().update(self._static_conn_graph)

        # Call setup function for this group.
        self.setup()
        self._setup_check()

        # need to save these because _setup_var_data can be called multiple times
        # during the config process and we don't want to wipe out any group_inputs
        # that were added during self.setup()
        self._pre_config_group_inputs = self._group_inputs.copy()

        if MPI:

            allsubs = list(self._subsystems_allprocs.values())
            proc_info = [self._proc_info[s.name] for s, _ in allsubs]

            # Call the load balancing algorithm
            try:
                sub_inds, sub_comm = self._mpi_proc_allocator(proc_info, len(allsubs), comm)
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

        self._sorted_subsystems_myproc = list(self._sorted_subsystems_myproc_iter())

        # need to set pathname correctly even for non-local subsystems
        for s, _ in self._subsystems_allprocs.values():
            s.pathname = '.'.join((self.pathname, s.name)) if self.pathname else s.name
            s._problem_meta = prob_meta

        # Perform recursion
        for subsys in self._sorted_subsystems_myproc:
            subsys._setup_procs(subsys.pathname, sub_comm, prob_meta)

        # build a list of local subgroups to speed up later loops
        self._subgroups_myproc = [s for s in self._subsystems_myproc if isinstance(s, Group)]
        if prob_meta['allow_post_setup_reorder']:
            self._subgroups_myproc.sort(key=lambda x: x.name)

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

    def is_explicit(self, is_comp=True):
        """
        Return True if this Group contains only explicit systems and has no cycles.

        Parameters
        ----------
        is_comp : bool
            If True, return True if this is an explicit component.
            If False, return True if this is an explicit component or group.

        Returns
        -------
        bool
            True if this is an explicit component.
        """
        if is_comp:
            return False

        if self._is_explicit is None:
            self._is_explicit = True
            for subsys in self._subsystems_myproc:
                if not subsys.is_explicit(is_comp=False):
                    self._is_explicit = False
                    break

            # check for cycles
            if self._is_explicit:
                self._is_explicit = nx.is_directed_acyclic_graph(self.compute_sys_graph())

        return self._is_explicit

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
        for subsys in self._sorted_subsystems_myproc:
            states.extend(subsys._list_states())

        return states

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
            return all_states
        else:
            return self._list_states()

    def _setup(self, comm, prob_meta):
        """
        Perform setup for this system and its descendant systems.

        This is only called on the top-level model.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm> or None
            The global communicator.
        prob_meta : dict
            Problem level metadata dictionary.
        """
        # save a ref to the problem level options.
        self._problem_meta = prob_meta
        # self._initial_condition_cache = {}
        self._auto_ivc_recorders = []
        self._sys_graph_cache = None
        self._remote_sets = []
        self._conn_graph = None

        # reset any coloring if a Coloring object was not set explicitly
        if self._coloring_info.dynamic or self._coloring_info.static is not None:
            self._coloring_info.coloring = None

        self.pathname = ''
        self.comm = comm

        self._pre_components = None
        self._post_components = None

        # Besides setting up the processors, this method also builds the model hierarchy.
        self._setup_procs(self.pathname, comm, prob_meta)

        prob_meta['config_info'] = _ConfigInfo()

        try:
            # Recurse model from the bottom to the top for configuring.
            self._configure()
        finally:
            prob_meta['config_info'] = None
            prob_meta['setup_status'] = _SetupStatus.POST_CONFIGURE

        self._configure_check()

        self._setup_var_data()

        # have to do this again because we are passed the point in _setup_var_data when this happens
        self._has_output_scaling = False
        self._has_output_adder = False
        self._has_resid_scaling = False
        self._has_bounds = False

        _has_applied_options = set()
        for grp in self.system_iter(include_self=True, recurse=True, depth_first=True, typ=Group):
            for subsys in grp.system_iter(include_self=True, recurse=False):
                if subsys.pathname not in _has_applied_options:
                    subsys._apply_output_solver_options()
                    _has_applied_options.add(subsys.pathname)

                grp._has_output_scaling |= subsys._has_output_scaling
                grp._has_output_adder |= subsys._has_output_adder
                grp._has_resid_scaling |= subsys._has_resid_scaling
                grp._has_bounds |= subsys._has_bounds

        # promoted names must be known to determine implicit connections so this must be
        # called after _setup_var_data, and _setup_var_data will have to be partially redone
        # after auto_ivcs have been added, but auto_ivcs can't be added until after we know all of
        # the connections.
        self._get_conn_graph().setup_global_connections(self)

    def _get_dataflow_graph(self):
        """
        Return a graph of all variables and components in the model.

        Each component is connected to each of its input and output variables, and those variables
        are connected to other variables based on the connections in the model.

        This results in a smaller graph (fewer edges) than would be the case for a pure variable
        graph where all inputs to a particular component would have to be connected to all outputs
        from that component.

        This should only be called on the top level Group.

        Returns
        -------
        networkx.DiGraph
            Graph of all variables and components in the model.
        """
        assert self.pathname == '', "call _get_dataflow_graph on the top level Group only."

        graph = nx.DiGraph()
        empty_comps = set()

        # locate any components that don't have any inputs or outputs and add them to the graph
        for subsys in self.system_iter(recurse=True, typ=Component):
            if subsys._resolver.num_abs(local=True) == 0:
                graph.add_node(subsys.pathname, local=True)
                empty_comps.add(subsys.pathname)

        if self.comm.size > 1:
            allemptycomps = self.comm.allgather(empty_comps)
            for compset in allemptycomps:
                for comp in compset:
                    if comp not in empty_comps:
                        graph.add_node(comp, local=False)
                        empty_comps.add(comp)

        resolver = self._resolver
        for direction in ('input', 'output'):
            isout = direction == 'output'
            for vname, flags in resolver.flags_iter(direction):
                graph.add_node(vname, type_=direction, local=bool(flags & LOCAL))

                comp = vname.rpartition('.')[0]
                if comp not in empty_comps:
                    graph.add_node(comp, local=bool(flags & LOCAL))
                    empty_comps.add(comp)

                if isout:
                    graph.add_edge(comp, vname)
                else:
                    graph.add_edge(vname, comp)

        for tgt, src in self._conn_global_abs_in2out.items():
            # connect the variables src and tgt
            graph.add_edge(src, tgt)

        return graph

    def _check_alias_overlaps(self, responses):
        """
        Check for overlapping indices in aliased responses.

        If the responses contain aliases, the returned response dict will
        be a copy with the alias keys removed and any missing alias sources
        added.

        This may only be called on the top level Group.

        Parameters
        ----------
        responses : dict
            Dictionary of response metadata. Keys don't matter.

        Returns
        -------
        dict
            Dictionary of response metadata with alias keys removed.
        """
        assert self.pathname == '', "call _check_alias_overlaps on the top level System only."

        aliases = set()
        srcdict = {}
        discrete_outs = self._var_allprocs_discrete['output']

        # group all aliases by source so we can compute overlaps for each source individually
        for meta in responses.values():
            src = meta['source']
            if src not in discrete_outs:
                if meta['alias']:
                    aliases.add(meta['alias'])
                if src in srcdict:
                    srcdict[src].append(meta)
                else:
                    srcdict[src] = [meta]

        graph = self._get_conn_graph()

        # loop over any sources having multiple aliases to ensure no overlap of indices
        for src, metalist in srcdict.items():
            if len(metalist) == 1:
                continue

            src_meta = graph.nodes[('o', src)]

            size = src_meta.global_size
            shape = src_meta.global_shape
            mat = np.zeros(size, dtype=np.ushort)

            for meta in metalist:
                indices = meta['indices']
                if indices is None:
                    mat[:] += 1
                else:
                    indices.set_src_shape(shape)
                    mat[indices.flat()] += 1

            if np.any(mat > 1):
                matching_aliases = sorted(m['alias'] for m in metalist if m['alias'])
                raise RuntimeError(f"{self.msginfo}: Indices for aliases {matching_aliases} are "
                                   f"overlapping constraint/objective '{src}'.")

        return responses

    def _get_var_offsets(self):
        """
        Compute global offsets for variables.

        These are the offsets into the global arrays that combine all variables of a certain
        iotype across all ranks.

        Returns
        -------
        dict
            Arrays of global offsets keyed by vec_name and deriv direction.
        """
        if self._var_offsets is None:
            offsets = self._var_offsets = {}
            for io in ['input', 'output']:
                vsizes = self._var_sizes[io]
                if vsizes.size > 0:
                    csum = np.empty(vsizes.size, dtype=INT_DTYPE)
                    csum[0] = 0
                    csum[1:] = np.cumsum(vsizes)[:-1]
                    offsets[io] = csum.reshape(vsizes.shape)
                else:
                    offsets[io] = np.zeros(0, dtype=INT_DTYPE).reshape((1, 0))

        #DBG
        # om_dump(f"{self.msginfo}: var_offsets")
        # om_dump_pprint(self._var_offsets)

        return self._var_offsets

    def _get_jac_col_scatter(self):
        """
        Return source and target indices for a scatter from output vector to total jacobian column.

        If the transfer involves remote or distributed variables, the indices will be global.
        Otherwise they will be converted to local.

        This is only called on the top level system.

        Returns
        -------
        ndarray
            Source indices.
        ndarray
            Target indices.
        int
            Size of jacobian column.
        bool
            True if remote or distributed vars are present.
        """
        myrank = self.comm.rank
        nranks = self.comm.size
        owns = self._owning_rank
        abs2idx = self._var_allprocs_abs2idx
        abs2meta = self._var_abs2meta['output']
        sizes = self._var_sizes['output']
        global_offsets = self._get_var_offsets()['output']
        oflist = self._get_jac_ofs()
        tsize = oflist[-1][2]
        toffset = myrank * tsize
        has_dist_data = False

        sinds = []
        tinds = []

        for name, tstart, tend, jinds, dist_sizes in oflist:
            vind = abs2idx[name]
            if dist_sizes is None:
                if name in abs2meta:
                    owner = myrank
                else:
                    owner = owns[name]
                    has_dist_data |= nranks > 1

                voff = global_offsets[owner, vind]
                if jinds is _full_slice:
                    vsize = sizes[owner, vind]
                    sinds.append(range(voff, voff + vsize))
                else:
                    sinds.append(jinds + voff)
                tinds.append(range(tstart + toffset, tend + toffset))
                assert len(sinds[-1]) == len(tinds[-1])
            else:  # 'name' refers to a distributed variable
                has_dist_data |= nranks > 1
                dtstart = dtend = tstart
                dsstart = dsend = 0
                for rnk, sz in enumerate(dist_sizes):
                    dsend += sz
                    if sz > 0:
                        voff = global_offsets[rnk, vind]
                        if jinds is _full_slice:
                            dtend += sz
                            sinds.append(range(voff, voff + sz))
                            tinds.append(range(toffset + dtstart, toffset + dtend))
                        elif jinds.size > 0:  # jinds is a flat array
                            subinds = jinds[jinds >= dsstart]
                            subinds = subinds[subinds < dsend]
                            if subinds.size > 0:
                                dtend += subinds.size
                                sinds.append(subinds + (voff - dsstart))
                                tinds.append(range(toffset + dtstart, toffset + dtend))
                        dtstart = dtend
                    dsstart = dsend
                assert (len(sinds) == 0 and len(tinds) == 0) or len(sinds[-1]) == len(tinds[-1])

        sarr = np.array(list(chain(*sinds)), dtype=INT_DTYPE)
        tarr = np.array(list(chain(*tinds)), dtype=INT_DTYPE)

        if nranks > 1:
            # do an allreduce to see if any procs have distrib/remote vars
            has_dist_data = bool(self.comm.allreduce(int(has_dist_data)))

        if not has_dist_data:
            # convert global indices back to local so we can use them to transfer between two
            # local arrays
            sysoffset = np.sum(sizes[:myrank, :])
            sarr -= sysoffset
            tarr -= toffset

        return sarr, tarr, tsize, has_dist_data

    def _setup_dynamic_properties(self):
        # called on the top level Group only
        graph = self._get_conn_graph()
        if graph._has_dynamic_shapes:
            self._setup_dynamic_property('shape')
        if graph._has_dynamic_units:
            self._setup_dynamic_property('units')

    def _setup_part2(self):
        """
        Complete setup of connections, sizes, and residuals.

        This part of setup is called automatically at the start of run_model or run_driver.
        This method is called only on the top level Group.
        """
        self._setup_dynamic_properties()

        graph = self._get_conn_graph()
        graph.update_all_node_meta(self)

        if dump_graphs:
            self.display_conn_graph(outfile=f'conn_graph_{self.comm.rank}.svg')
            self.display_dataflow_graph(outfile=f'dataflow_graph_{self.comm.rank}.svg')

        self._check_order()
        self._resolver._check_dup_prom_outs()

        self._setup_var_sizes()

        self._top_level_post_sizes()

        self._check_connections()

        graph.check(self)

        nodes = graph.nodes

        # now make sure all graph data is in sync with the variable metadata
        for io in ['input', 'output']:
            for name in self._var_allprocs_abs2meta[io]:
                node = (io[0], name)
                nodes[node].update_model_meta()

        # setup of residuals must occur before setup of vectors and partials
        self._setup_residuals()

        for recorder in self._auto_ivc_recorders:
            self._auto_ivc.add_recorder(recorder)
        self._auto_ivc_recorders = []

        self._problem_meta['setup_status'] = _SetupStatus.POST_SETUP2

    def _final_setup(self):
        """
        Perform final setup for this system and its descendant systems.

        This part of setup is called automatically at the start of run_model or run_driver.
        This method is called only on the top level Group.
        """
        if self._use_derivatives:
            self._setup_partials()

        self._setup_vectors(None)

        self._setup_jax()

        self._fd_rev_xfer_correction_dist = {}

        desvars = self.get_design_vars(get_sizes=False)
        responses = self._check_alias_overlaps(self.get_responses(get_sizes=False,
                                                                  use_prom_ivc=True))

        self._dataflow_graph = self._get_dataflow_graph()
        self._problem_meta['dataflow_graph'] = self._dataflow_graph

        # figure out if we can remove any edges based on zero partials we find
        # in components.  By default all component connected outputs
        # are also connected to all connected inputs from the same component.
        self._missing_partials = {}
        if not self._owns_approx_jac:  # don't check for missing partials when doing FD
            self._get_missing_partials(self._missing_partials)
            if self._missing_partials:
                self._update_dataflow_graph(responses)

        self._problem_meta['relevance'] = get_relevance(self, responses, desvars)

        # Transfers have to be set up after the vector setup.
        self._setup_transfers()

        self._setup_solvers()
        self._setup_solver_print()

        self._setup_recording()

        self.set_initial_values()

    def _setup_vectors(self, parent_vectors):
        """
        Compute all vectors for all vec names and assign excluded variables lists.

        Parameters
        ----------
        parent_vectors : dict or None
            Parent vectors.  Same structure as root_vectors.
        """
        super()._setup_vectors(parent_vectors)

        for subsys in self._sorted_subsystems_myproc:
            subsys._scale_factors = self._scale_factors
            subsys._setup_vectors(self._vectors)

    def _update_dataflow_graph(self, responses):
        """
        Update the dataflow graph based on missing partials.

        Parameters
        ----------
        responses : dict
            Dictionary of response metadata.
        """
        resps = set(meta2src_iter(responses.values()))

        missing_responses = set()
        for pathname, missing in self._missing_partials.items():
            inputs = [n for n, _ in self._dataflow_graph.in_edges(pathname)]
            outputs = [n for _, n in self._dataflow_graph.out_edges(pathname)]

            self._dataflow_graph.remove_node(pathname)

            for output in outputs:
                found = False
                for inp in inputs:
                    if (output, inp) not in missing:
                        self._dataflow_graph.add_edge(inp, output)
                        found = True

                if not found and output in resps:
                    missing_responses.add(output)

        if missing_responses:
            msg = (f"Constraints or objectives [{', '.join(sorted(missing_responses))}] cannot"
                   " be impacted by the design variables of the problem because no partials "
                   "were defined for them in their parent component(s).")
            if self._problem_meta['singular_jac_behavior'] == 'error':
                raise RuntimeError(msg)
            else:
                issue_warning(msg, category=DerivativesWarning)

    def set_initial_values(self):
        """
        Set all input and output variables to their declared initial values.

        This should only be called on the top level group.
        """
        assert self.pathname == '', "set_initial_values must be called on the top level group"

        if self._problem_meta['setup_status'] < _SetupStatus.POST_FINAL_SETUP:
            self._problem_meta['setup_status'] = _SetupStatus.POST_FINAL_SETUP

        graph = self._get_conn_graph()
        for name in chain(self._var_abs2meta['output'], self._var_discrete['output']):
            node = ('o', name)
            node_meta = graph.nodes[node]
            if node_meta.val is not None:
                graph.set_tree_val(self, node, node_meta.val)  # force updates of input values
                if node_meta.discrete:
                    self._discrete_outputs[name] = node_meta.val
                else:
                    self._outputs.set_var(name, node_meta.val)

        for abs_name, meta in self._var_abs2meta['input'].items():
            self._inputs.set_var(abs_name, graph.nodes[('i', abs_name)].val)

        for name in self._discrete_inputs:
            self._discrete_inputs[name] = graph.nodes[('i', name)].val

        # for abs_name, meta in self._var_abs2meta['output'].items():
        #     self._outputs.set_var(abs_name, graph.nodes[('o', abs_name)].val)

        # for name in self._discrete_outputs:
        #     self._discrete_outputs[name] = graph.nodes[('o', name)].val

    def _get_root_vectors(self):
        """
        Get the root vectors for the nonlinear and linear vectors for the model.

        Returns
        -------
        dict of dict of Vector
            Root vectors: first key is 'input', 'output', or 'residual'; second key is vec_name.
        """
        force_alloc_complex = self._problem_meta['force_alloc_complex']

        # Check for complex step to set vectors up appropriately.
        # If any subsystem needs complex step, then we need to allocate it everywhere.
        nl_alloc_complex = force_alloc_complex
        if not nl_alloc_complex:
            for sub in self.system_iter(include_self=True, recurse=True):
                nl_alloc_complex |= 'cs' in sub._approx_schemes
                if nl_alloc_complex:
                    break

        # Linear vectors allocated complex only if subsolvers require derivatives.
        if nl_alloc_complex and self._use_derivatives:
            from openmdao.error_checking.check_config import check_allocate_complex_ln
            ln_alloc_complex = check_allocate_complex_ln(self, force_alloc_complex)
        else:
            ln_alloc_complex = False

        # If any proc's local systems need a complex vector, then all procs need it.
        if self.comm.size > 1:
            need_alloc_complex = np.array([nl_alloc_complex, ln_alloc_complex], dtype=int)
            result = np.zeros(2, dtype=int)
            self.comm.Allreduce(need_alloc_complex, result)
            nl_alloc_complex = bool(result[0])
            ln_alloc_complex = bool(result[1])

        if self._has_input_scaling or self._has_output_scaling or self._has_resid_scaling:
            self._scale_factors = self._compute_root_scale_factors()
        else:
            self._scale_factors = None

        if self._vector_class is None:
            self._vector_class = self._local_vector_class

        do_scaling = {
            'input': self._has_input_scaling,
            'output': self._has_output_scaling,
            'residual': self._has_resid_scaling
        }
        do_adder = {
            'input': self._has_input_adder,
            'output': self._has_output_adder,
            'residual': self._has_resid_scaling
        }

        # save root vecs as an attribute so that we can reuse the nonlinear scaling vecs in the
        # linear root vec
        root_vectors = {'input': {}, 'output': {}, 'residual': {}}

        for kind in ['input', 'output', 'residual']:
            rvec = root_vectors[kind]
            rvec['nonlinear'] = nlvec = self._vector_class('nonlinear', kind, self, None,
                                                           alloc_complex=nl_alloc_complex)

            if self._use_derivatives:
                rvec['linear'] = self._vector_class('linear', kind, self, None,
                                                    alloc_complex=ln_alloc_complex)

            if do_scaling[kind] or do_adder[kind]:
                nlvec._set_scaling(self, do_adder[kind])
                if self._use_derivatives:
                    rvec['linear']._set_scaling(self, do_adder[kind], nlvec)

        return root_vectors

    def _check_order(self, reorder=True, recurse=True, out_of_order=None):
        """
        Check if auto ordering is needed, optionally reordering subsystems if appropriate.

        Parameters
        ----------
        reorder : bool
            If True, reorder the subsystems based on the computed order.  Otherwise
            just return the out-of-order connections.
        recurse : bool
            If True, call this method on all subgroups.
        out_of_order : dict or None
            Lists of out-of-order connections keyed by group pathname. Out of order connections
            are keyed by target system name and have values that are lists of source system names.
            If incoming value of out_of_order is None, then a new dict is created and returned.

        Returns
        -------
        dict
            Lists of out-of-order connections keyed by group pathname.
        """
        if out_of_order is None:
            out_of_order = {}

        if self.options['auto_order'] or not reorder:
            G = self.compute_sys_graph()
            orders = {name: i for i, name in enumerate(self._subsystems_allprocs)}
            strongcomps, new_out_of_order = get_out_of_order_nodes(G, orders)

            if new_out_of_order:
                # group targets with all of their sources
                tgts = {}
                for u, v in new_out_of_order:
                    if v not in tgts:
                        tgts[v] = []
                    tgts[v].append(u)

                for t in tgts:
                    tgts[t] = sorted(tgts[t])

                out_of_order[self.pathname] = tgts
                if reorder:
                    self._set_auto_order(strongcomps, orders)

        if recurse:
            for s in self._subgroups_myproc:
                s._check_order(reorder, recurse, out_of_order)

        return out_of_order

    def _set_auto_order(self, strongcomps, orders):
        """
        Set the order of the subsystems based on the dependency graph.

        Parameters
        ----------
        strongcomps : list of list of str
            List of sets of subsystem names. Each list contains subsystems that are strongly
            connected.  Sets containing 2 or more subsystems indicate a cycle.
        orders : dict
            Dictionary mapping subsystem names to their index in the current ordering.
        """
        new_order = []
        for strongcomp in strongcomps:
            if len(strongcomp) > 1:
                # never change the internal order in a cycle
                order_list = [(name, orders[name]) for name in strongcomp]
                new_order.extend([name for name, _ in sorted(order_list, key=lambda x: x[1])])
            else:
                for s in strongcomp:
                    new_order.append(s)

        if self._problem_meta['allow_post_setup_reorder']:
            self.set_order(new_order)
        else:
            if '_auto_ivc' in new_order:
                new_order.remove('_auto_ivc')
            issue_warning(f"{self.msginfo}: A new execution order {new_order} is recommended, but "
                          "auto ordering has been disabled because the Problem option "
                          "'allow_post_setup_reorder' is False. It is recommended to either set "
                          "`allow_post_setup_reorder` to True or to manually set the execution "
                          "order to the recommended order using `set_order`.")

    def _check_nondist_sizes(self):
        # verify that nondistributed variables have same size across all procs
        abs2idx = self._var_allprocs_abs2idx
        for io in ('input', 'output'):
            sizes = self._var_sizes[io]
            for abs_name in self._resolver.abs_iter(io, continuous=True, distributed=False):
                vsizes = sizes[:, abs2idx[abs_name]]
                unique = set(vsizes)
                unique.discard(0)
                if len(unique) > 1:
                    # sizes differ, now find which procs don't agree
                    rnklist = []
                    for sz in unique:
                        rnklist.append((sz, [i for i, s in enumerate(vsizes) if s == sz]))
                    msg = ', '.join([f"rank(s) {r} have size {s}" for s, r in rnklist])
                    self._collect_error(f"{self.msginfo}: Size of {io} '{abs_name}' "
                                        f"differs between processes ({msg}).",
                                        ident=('size', abs_name))

    def _top_level_post_sizes(self):
        # # this runs after the variable sizes are known

        self._check_nondist_sizes()

    def _setup_var_data(self):
        """
        Compute the list of abs var names, abs/prom name maps, and metadata dictionaries.
        """
        self._var_existence = None

        super()._setup_var_data()

        resolver = self._resolver
        self._cross_keys = set()

        var_discrete = self._var_discrete = {'input': {}, 'output': {}}
        allprocs_discrete = self._var_allprocs_discrete

        abs2meta = self._var_abs2meta

        allprocs_abs2meta = self._var_allprocs_abs2meta = {'input': {}, 'output': {}}

        self._has_distrib_vars = False
        self._has_fd_group = self._owns_approx_jac
        rank = self.comm.rank
        conn_graph = self._get_conn_graph()

        # sort the subsystems alphabetically in order to make the ordering
        # of vars in vectors and other data structures independent of the
        # execution order.
        for subsys in self._sorted_subsystems_myproc:
            self._has_output_scaling |= subsys._has_output_scaling
            self._has_output_adder |= subsys._has_output_adder
            self._has_resid_scaling |= subsys._has_resid_scaling
            self._has_distrib_vars |= subsys._has_distrib_vars
            if len(subsys._subsystems_allprocs) > 0:
                self._has_fd_group |= subsys._has_fd_group

            var_maps = subsys._get_promotion_maps()

            sub_prefix = subsys.name + '.'

            for io in ['input', 'output']:
                abs2meta[io].update(subsys._var_abs2meta[io])
                allprocs_abs2meta[io].update(subsys._var_allprocs_abs2meta[io])
                subprom2prom = var_maps[io]

                allprocs_discrete[io].update(subsys._var_allprocs_discrete[io])
                var_discrete[io].update({sub_prefix + k: v for k, v in
                                         subsys._var_discrete[io].items()})

                for sub_prom, sub_abs in subsys._resolver.prom2abs_iter(io):
                    if sub_prom in subprom2prom:
                        prom_name, _, pinfo, _ = subprom2prom[sub_prom]
                    else:
                        prom_name = sub_prefix + sub_prom

                    for abs_name in sub_abs:
                        flags = subsys._resolver.flags(abs_name, io)
                        resolver.add_mapping(abs_name, prom_name, io,
                                             local=flags & LOCAL,
                                             continuous=flags & CONTINUOUS,
                                             distributed=flags & DISTRIBUTED)

                    if sub_prom in subprom2prom:
                        conn_graph.add_promotion(io, self, prom_name, subsys, sub_prom, pinfo)

        # If running in parallel, allgather
        if self.comm.size > 1 and self._mpi_proc_allocator.parallel:
            resolver.reset_prom_maps()  # no use sending prom2abs to other procs
            if self._gather_full_data():
                raw = (allprocs_discrete, resolver, allprocs_abs2meta,
                       self._has_output_scaling, self._has_output_adder,
                       self._has_resid_scaling, self._group_inputs, self._has_distrib_vars,
                       self._has_fd_group)
            else:
                raw = (
                    {'input': {}, 'output': {}},
                    None,
                    {'input': {}, 'output': {}},
                    False,
                    False,
                    False,
                    {},
                    False,
                    False,
                )

            gathered = self.comm.allgather(raw)

            # start with a fresh dict to keep order the same in all procs
            old_abs2meta = allprocs_abs2meta
            allprocs_abs2meta = {'input': {}, 'output': {}}

            proc_resolvers = [None] * self.comm.size
            myrank = self.comm.rank
            for rank, (proc_discrete, proc_resolver, proc_abs2meta,
                       oscale, oadd, rscale, ginputs, has_dist_vars,
                       has_fd_group) in enumerate(gathered):
                self._has_output_scaling |= oscale
                self._has_output_adder |= oadd
                self._has_resid_scaling |= rscale
                self._has_distrib_vars |= has_dist_vars
                self._has_fd_group |= has_fd_group

                if rank != myrank:
                    for p, mta in ginputs.items():
                        if p not in self._group_inputs:
                            self._group_inputs[p] = mta

                proc_resolvers[rank] = proc_resolver

                for io in ['input', 'output']:
                    allprocs_abs2meta[io].update(proc_abs2meta[io])
                    allprocs_discrete[io].update(proc_discrete[io])

            self._resolver.update_from_ranks(myrank, proc_resolvers)

            for io in ('input', 'output'):
                if old_abs2meta[io]:
                    # update new allprocs_abs2meta with our local version (now that we have a
                    # consistent order for our dict), so that the 'size' metadata will
                    # accurately reflect this proc's var size instead of one from some other proc.
                    allprocs_abs2meta[io].update(old_abs2meta[io])

        for io in ('input', 'output'):
            self._var_allprocs_abs2idx.update(
                {n: i for i, n in enumerate(allprocs_abs2meta[io])})

        self._var_allprocs_abs2meta = allprocs_abs2meta

        if self._var_discrete['input'] or self._var_discrete['output']:
            self._discrete_inputs = _DictValues(self._var_discrete['input'])
            self._discrete_outputs = _DictValues(self._var_discrete['output'])
        else:
            self._discrete_inputs = self._discrete_outputs = {}

        self._vars_to_gather = self._find_vars_to_gather()
        if self.pathname == '':
            self._problem_meta['vars_to_gather'] = self._vars_to_gather

    def _resolve_group_input_defaults(self, show_warnings=False):
        """
        Resolve any ambiguities in group input defaults throughout the model.

        Only called at the model level.

        Parameters
        ----------
        show_warnings : bool
            Bool to show or hide the auto_ivc warnings.
        """
        self._auto_ivc_warnings = []

    def _find_vars_to_gather(self):
        """
        Return a mapping of var pathname to owning rank.

        The mapping will contain ONLY systems that are remote on at least one proc.
        Discrete variables ARE included.

        Returns
        -------
        dict
            The mapping of variable pathname to the lowest rank where it's local.
        """
        remote_vars = {}

        if self.comm.size > 1:
            myproc = self.comm.rank
            nprocs = self.comm.size

            resflags = self._resolver.flags
            for io in ('input', 'output'):

                # var order must be same on all procs
                sorted_names = sorted(self._resolver.abs_iter(io))
                locality = np.zeros((nprocs, len(sorted_names)), dtype=bool)
                for i, name in enumerate(sorted_names):
                    if resflags(name, io) & LOCAL:
                        locality[myproc, i] = True

                my_loc = locality[myproc, :].copy()
                self.comm.Allgather(my_loc, locality)

                for i, name in enumerate(sorted_names):
                    nzs = np.nonzero(locality[:, i])[0]
                    if 0 < nzs.size < nprocs:
                        remote_vars[name] = nzs[0]

        return remote_vars

    def _setup_vector_class(self):
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

    @collect_errors
    def _setup_var_sizes(self):
        """
        Compute the arrays of variable sizes for all variables/procs in the model.

        This should only be called at the top level of the system tree.
        """
        self._var_offsets = None
        all_abs2meta = self._var_allprocs_abs2meta

        # only allocate these arrays the first time through
        self._var_sizes = {
            'input': np.zeros((self.comm.size, len(all_abs2meta['input'])), dtype=INT_DTYPE),
            'output': np.zeros((self.comm.size, len(all_abs2meta['output'])), dtype=INT_DTYPE),
        }

        for subsys in self._sorted_subsystems_myproc:
            subsys._setup_var_sizes()

        iproc = self.comm.rank
        nprocs = self.comm.size
        for io, sizes in self._var_sizes.items():
            abs2meta = self._var_abs2meta[io]
            for i, name in enumerate(self._var_allprocs_abs2meta[io]):
                if name in abs2meta:  # local var
                    sz = abs2meta[name]['size']
                    if sz is not None:
                        sizes[iproc, i] = sz

            if nprocs > 1:
                my_sizes = sizes[iproc, :].copy()
                self.comm.Allgather(my_sizes, sizes)

        # TODO: see about moving vector class setup elsewhere
        self._setup_vector_class()

        self._compute_owning_ranks()

    def _compute_owning_ranks(self):
        abs2meta = self._var_allprocs_abs2meta
        abs2discrete = self._var_allprocs_discrete

        if self.comm.size > 1:
            owns = self._owning_rank
            self._owned_output_sizes = self._var_sizes['output'].copy()
            abs2idx = self._var_allprocs_abs2idx
            for io in ('input', 'output'):
                sizes = self._var_sizes[io]
                for name, meta in abs2meta[io].items():
                    dist = meta['distributed']
                    i = abs2idx[name]
                    for rank in range(self.comm.size):
                        if sizes[rank, i] > 0:
                            owns[name] = rank
                            if not dist and io == 'output':
                                self._owned_output_sizes[rank + 1:, i] = 0  # zero out all dups
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
            self._owned_output_sizes = self._var_sizes['output']

        if self.pathname == '':
            graph = self._get_conn_graph()
            graph._owned_output_sizes = self._owned_output_sizes
            graph._var_allprocs_abs2idx = self._var_allprocs_abs2idx
            graph._owning_rank = self._owning_rank
            graph._var_sizes = self._var_sizes

    def _owned_size(self, abs_name):
        """
        Return the size of the variable on the owning rank.

        Parameters
        ----------
        abs_name : str
            The absolute name of the variable.

        Returns
        -------
        int
            The size of the variable on this rank, 0 if this is not the owning rank.
        """
        return self._owned_output_sizes[self.comm.rank, self._var_allprocs_abs2idx[abs_name]]

    def _get_implicit_connections(self):
        """
        Return a set of promoted variable names where an output and an input match.
        """
        if self.pathname:
            implicit_connections = {self.pathname + '.' + n
                                    for n in self._resolver.get_implicit_conns()}
        else:
            implicit_connections = self._resolver.get_implicit_conns()
        for subsys in self._sorted_subsystems_myproc:
            if isinstance(subsys, Group):
                implicit_connections.update(subsys._get_implicit_connections())

        return implicit_connections

    def get_indep_vars(self, local, include_discrete=False):
        """
        Return a dict of independant variables contained in this group or any of its subgroups.

        Parameters
        ----------
        local : bool
            If True, return only the variables local to the current process.
        include_discrete : bool
            If True, include discrete variables in the returned dict.

        Returns
        -------
        dict
            Dict of all independant variables in this group and their corresponding metadata.
        """
        abs2meta = self._var_abs2meta['output'] if local else self._var_allprocs_abs2meta['output']
        ivcs = {n: meta for n, meta in abs2meta.items() if 'openmdao:indep_var' in meta['tags']}
        if include_discrete:
            if local:
                # local discrete vars use relative names
                prefix = self.pathname + '.' if self.pathname else ''
                ivcs.update({prefix + n: meta for n, meta in self._var_discrete['output'].items()
                             if 'openmdao:indep_var' in meta['tags']})
            else:
                ivcs.update({n: meta for n, meta in self._var_allprocs_discrete['output'].items()
                             if 'openmdao:indep_var' in meta['tags']})
        return ivcs

    def _setup_jax(self):
        if jax is None:
            return

        # recurse down the tree and setup
        # jax anywhere below where it's active, then return.
        for subsys in self._subsystems_myproc:
            if isinstance(subsys, Group) or subsys.options['derivs_method'] == 'jax':
                subsys._setup_jax()

    def _setup_dynamic_property(self, prop):
        """
        Dynamically add property metadata for variables.

        This only happens if the user has set shape_by_conn, units_by_conn, copy_shape, copy_units,
        compute_shape, or compute_units for a variable.

        Parameters
        ----------
        prop : str
            Name of the property to compute.
        """
        knowns = set()
        all_abs2meta_out = self._var_allprocs_abs2meta['output']
        conn_graph = self._get_conn_graph()
        conn_nodes = conn_graph.nodes

        if prop == 'shape':
            dist_shp = {}  # local distrib sizes
            self._shapes_graph = graph = nx.DiGraph()
        elif prop == 'units':
            self._units_graph = graph = nx.DiGraph()
        else:
            raise ValueError(f"Invalid property name: {prop}")

        nprocs = self.comm.size
        do_dist = nprocs > 1 and prop == 'shape'

        def compute_var_property(to_var, prop_dict, func):
            """
            Compute info for the given variable using the given function.

            Parameters
            ----------
            to_var : str
                Name of variable to compute property for.
            prop_dict : dict
                Mapping of variable name to property.
            func : function
                Function to use to compute the property.

            Returns
            -------
            tuple or None
                If the property of the variable is known, return it.
                Otherwise, return None.
            """
            try:
                from_prop = func(prop_dict)
            except KeyError:
                self._collect_error(f"{self.msginfo}: Can't compute {prop} of variable "
                                    f"'{to_var}': variable '{to_var}' doesn't exist.")
                return
            except Exception as err:
                self._collect_error(f"{self.msginfo}: Error occurred while computing the "
                                    f"{prop} of variable '{to_var}': {err}")
                return

            return from_prop

        def copy_var_units(from_var, to_var):
            from_units = conn_nodes[from_var].units
            conn_to = conn_nodes[to_var]
            conn_to.units = from_units
            return from_units

        def connect_nodes(src_node, tgt_node, multi=False):
            if src_node not in graph:
                graph.add_node(src_node, conn_meta=conn_nodes[src_node])
            if tgt_node not in graph:
                graph.add_node(tgt_node, conn_meta=conn_nodes[tgt_node])

            graph.add_edge(src_node, tgt_node, multi=multi)

        copy_var_property = conn_graph.copy_var_shape if prop == 'shape' else copy_var_units

        prop_by_conn = f"{prop}_by_conn"
        copy_prop = f"copy_{prop}"
        compute_prop = f"compute_{prop}"

        component_io = {}

        my_abs2meta = self._var_abs2meta

        # determine all component inputs and outputs
        for io in ['input', 'output']:
            a2m = my_abs2meta[io]
            for abs_name in self._var_allprocs_abs2meta[io]:
                comp_path = abs_name.rpartition('.')[0]
                if comp_path not in component_io:
                    component_io[comp_path] = {'input': [], 'output': []}
                component_io[comp_path][io].append(abs_name)

                if do_dist and abs_name in a2m and a2m[abs_name]['distributed']:
                    dist_shp[abs_name] = a2m[abs_name]['shape']

        opposite = {'input': 'output', 'output': 'input'}
        for comp_path, io_dict in component_io.items():
            for io in ['input', 'output']:
                for abs_name in io_dict[io]:
                    node = (io[0], abs_name)
                    node_meta = conn_nodes[node]
                    copy_var = getattr(node_meta, copy_prop)
                    if copy_var:
                        cpy_var_path = comp_path + '.' + copy_var
                        if (opposite[io][0], cpy_var_path) in conn_graph:
                            cpy_node = (opposite[io][0], cpy_var_path)
                        elif (io[0], cpy_var_path) in conn_graph:
                            cpy_node = (io[0], cpy_var_path)
                        else:
                            self._collect_error(f"{comp_path}: copy_shape variable '{copy_var}' "
                                                f"not found.")
                            continue
                        connect_nodes(cpy_node, node)
                    elif getattr(node_meta, compute_prop):
                        oi = opposite[io]
                        for vabs_name in io_dict[oi]:
                            connect_nodes((oi[0], vabs_name), node, multi=True)

        # find all variables that have an unknown property (across all procs) and connect them
        # to other unknown and known property variables to form a directed graph.
        for out_name, out_meta in all_abs2meta_out.items():
            src_node = ('o', out_name)
            src_meta = conn_nodes[src_node]

            src_prop = src_meta[prop]
            src_by_conn = src_meta[prop_by_conn] if src_prop is None else False

            for tgt_node in conn_graph.leaf_input_iter(src_node):
                tgt_meta = conn_nodes[tgt_node]
                tgt_prop = tgt_meta[prop]
                tgt_prop_by_conn = tgt_meta.get(prop_by_conn) if tgt_prop is None else False

                if tgt_prop_by_conn:
                    connect_nodes(src_node, tgt_node)

                if src_by_conn:
                    connect_nodes(tgt_node, src_node)

        if graph.order() == 0:
            # we don't have any {prop}_by_conn or copy_{prop} or compute_{prop} variables,
            # so we're done
            return

        knowns = {n for n in graph.nodes() if getattr(conn_nodes[n], prop) is not None}
        all_knowns = knowns.copy()

        do_sort = self.comm.size > 1

        # connected_components needs an undirected graph
        for comps in nx.weakly_connected_components(graph):

            # treat all knowns initially as unresolved
            unresolved_knowns = all_knowns.intersection(comps)
            if not unresolved_knowns:
                # no knowns in this component, so we fail.
                continue

            progress = True
            while progress:
                progress = False
                unresolved_knowns = get_unresolved_knowns(graph, prop, unresolved_knowns)

                active_single_edges, computed_nodes = get_active_edges(graph, unresolved_knowns,
                                                                       prop)
                if do_sort:
                    active_single_edges = sorted(active_single_edges)

                for k, u in active_single_edges:
                    copied_prop = copy_var_property(k, u)
                    if copied_prop is not None:
                        if is_unresolved(graph, u, prop):
                            unresolved_knowns.add(u)

                        all_knowns.add(u)
                        progress = True

                for mnode in computed_nodes:
                    for k, _, data in graph.in_edges(mnode, data=True):
                        if data['multi'] and getattr(conn_nodes[k], prop) is None:
                            # if any preds are unknown, we can't compute the property yet
                            break
                    else:
                        # all compute_prop preds are known so compute the property
                        if prop == 'shape':
                            props = {
                                n[1].rpartition('.')[-1]: getattr(conn_nodes[n], prop)
                                for n in graph.predecessors(mnode)
                            }
                        elif prop == 'units':
                            props = {
                                n[1].rpartition('.')[-1]: _find_unit(getattr(conn_nodes[n], prop))
                                for n in graph.predecessors(mnode)
                            }

                        mnode_meta = conn_nodes[mnode]
                        val = compute_var_property(mnode, props, getattr(mnode_meta, compute_prop))
                        if val is not None:
                            if prop == 'units':
                                val = val.name()
                            setattr(mnode_meta, prop, val)
                            if is_unresolved(graph, mnode, prop):
                                unresolved_knowns.add(mnode)
                            all_knowns.add(mnode)
                            progress = True

        unresolved = set(graph.nodes()) - all_knowns
        if unresolved:
            if prop == 'units':
                # throw out any that have units of None but are not dynamic
                unresolved = {node for node in unresolved if conn_nodes[node].dyn_units}
            if unresolved:
                unresolved = sorted([u[1] for u in unresolved])
                propstr = 'shapes' if prop == 'shape' else 'units'
                self._collect_error(f"{self.msginfo}: Failed to resolve {propstr} for {unresolved}."
                                    f" To see the dynamic {propstr} dependency graph, "
                                    f"do 'openmdao view_dyn_{propstr} <your_py_file>'.")
                # if prop=='shape' and self.comm.rank == 0:
                #     from openmdao.visualization.dyn_shape_plot import view_dyn_shapes
                #     view_dyn_shapes(self, outfile=f'shape_dep_graph_{self.comm.rank}.html')

    @collect_errors
    @check_mpi_exceptions
    def _check_connections(self):
        """
        Compute dict of all connections owned by this Group.

        Also, check shapes of connected variables.
        """
        conn_graph = self._get_conn_graph()
        conn_nodes = conn_graph.nodes

        abs_in2out = self._conn_abs_in2out
        self._conn_discrete_in2out = {}
        allprocs_discrete_in = self._var_allprocs_discrete['input']
        allprocs_discrete_out = self._var_allprocs_discrete['output']

        for subsys in self._subsystems_myproc:
            subsys._check_connections()
            if subsys._has_input_scaling:
                self._has_input_scaling = True

        allprocs_abs2meta_in = self._var_allprocs_abs2meta['input']
        allprocs_abs2meta_out = self._var_allprocs_abs2meta['output']
        abs2meta_in = self._var_abs2meta['input']
        abs2meta_out = self._var_abs2meta['output']

        nproc = self.comm.size

        # Check input/output units here, and set _has_input_scaling
        # to True for this Group if units are defined and different, or if
        # ref or ref0 are defined for the output.
        for abs_in, abs_out in abs_in2out.items():
            inode = ('i', abs_in)
            if conn_nodes[inode].discrete:
                self._conn_discrete_in2out[abs_in] = abs_out

            if True:
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
                self._has_input_scaling = self._has_output_scaling or \
                    (in_units and out_units and in_units != out_units)

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
                if xfer._has_input_scaling:
                    vec_inputs.scale_to_norm()
                    xfer._transfer(vec_inputs, self._vectors['output'][vec_name], mode)
                    vec_inputs.scale_to_phys()
                else:
                    xfer._transfer(vec_inputs, self._vectors['output'][vec_name], mode)

            if self._conn_discrete_in2out and vec_name == 'nonlinear':
                self._discrete_transfer(sub)

        else:  # rev
            if xfer is not None:
                if xfer._has_input_scaling:
                    vec_inputs.scale_to_norm(mode='rev')

                xfer._transfer(vec_inputs, self._vectors['output'][vec_name], mode)

                if self._problem_meta['parallel_deriv_color'] is None:
                    key = (sub, '@nocolor')
                    if key in self._transfers['rev']:
                        xfer = self._transfers['rev'][key]
                        xfer._transfer(vec_inputs, self._vectors['output'][vec_name], mode)

                if xfer._has_input_scaling:
                    vec_inputs.scale_to_phys(mode='rev')

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
        for subsys in self._subgroups_myproc:
            subsys._setup_transfers()

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
        try:
            subsys = getattr(self, subsys_name)
        except AttributeError:
            raise AttributeError(f"{self.msginfo}: subsystem '{subsys_name}' does not exist.")

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
                issue_warning("ignored flat_src_indices and/or src_shape because"
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
                prominfo = _PromotesInfo(src_indices, flat_src_indices, src_shape,
                                         promoted_from=subsys.pathname)
            except Exception as err:
                lst = []
                if any is not None:
                    lst.extend(any)
                if inputs is not None:
                    lst.extend(inputs)
                self._collect_error(f"{self.msginfo}: When promoting {sorted(lst)}: {err}",
                                    ident=(self.pathname, tuple(lst)))
                return

        if any:
            subsys._var_promotes['any'].extend((a, prominfo) for a in any)
        if inputs:
            subsys._var_promotes['input'].extend((i, prominfo) for i in inputs)
        if outputs:
            subsys._var_promotes['output'].extend((o, None) for o in outputs)

        # # check for attempt to promote with different alias
        # list_comp = [i if isinstance(i, tuple) else (i, i)
        #              for i, _ in subsys._var_promotes['input']]

        # for original, new in list_comp:
        #     for original_inside, new_inside in list_comp:
        #         if original == original_inside and new != new_inside:
        #             self._collect_error("%s: Trying to promote '%s' when it has been aliased to "
        #                                 "'%s'." % (self.msginfo, original_inside, new))
        #             continue

        # if this was called during configure(), mark this group as modified
        if self._problem_meta is not None and self._problem_meta['config_info'] is not None:
            self._problem_meta['config_info']._prom_added(self.pathname)

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

        if subsys is self:
            raise RuntimeError(f"{self.msginfo}: System '{name}' can't be added to itself.")

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

        # Note, the declared order in any of these promotes arguments shouldn't matter. However,
        # the order does matter when using system.promotes during configure. There, you are
        # permitted to promote '*' then promote_to an alias afterwards, but not in the reverse.
        # To make this work, we sort the promotes lists for this subsystem to put the wild card
        # entries at the beginning.
        if promotes:
            subsys._var_promotes['any'] = [(p, None) for p in
                                           sorted(promotes, key=lambda x: '*' not in x)]
        if promotes_inputs:
            subsys._var_promotes['input'] = [(p, None) for p in
                                             sorted(promotes_inputs, key=lambda x: '*' not in x)]
        if promotes_outputs:
            subsys._var_promotes['output'] = [(p, None) for p in
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

    def add_recorder(self, recorder, recurse=False):
        """
        Add a recorder to the system.

        Parameters
        ----------
        recorder : <CaseRecorder>
           A recorder instance.
        recurse : bool
            Flag indicating if the recorder should be added to all the subsystems.
        """
        if MPI:
            raise RuntimeError(self.msginfo + ": Recording of Systems when running parallel "
                                              "code is not supported yet")

        self._rec_mgr.append(recorder)

        if recurse:
            for s in self.system_iter(include_self=False, recurse=recurse):
                s._rec_mgr.append(recorder)

            if self.pathname == '':  # top level group
                # too early in setup for _auto_ivc to exist, so we'll add recorder to it later
                if '_auto_ivc' not in self._subsystems_allprocs:
                    if recorder not in self._auto_ivc_recorders:
                        self._auto_ivc_recorders.append(recorder)

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

        if self._static_mode:
            manual_connections = self._static_manual_connections
        else:
            manual_connections = self._manual_connections

        manual_connections[tgt_name] = (src_name, src_indices, flat_src_indices)

    def set_order(self, new_order):
        """
        Specify a new execution order for subsystems in this group.

        Parameters
        ----------
        new_order : list of str
            List of system names in desired new execution order.
        """
        if self._problem_meta is not None and not self._problem_meta['allow_post_setup_reorder'] \
                and self._problem_meta['setup_status'] == _SetupStatus.POST_CONFIGURE:
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

        if not self._static_mode:
            self._subsystems_myproc = [s for s, _ in self._subsystems_allprocs.values()]
            self._sorted_subsystems_myproc = list(self._sorted_subsystems_myproc_iter())

        self._order_set = True
        if self._problem_meta is not None and not self._problem_meta['allow_post_setup_reorder']:
            # order has been changed so we need a new full setup
            self._problem_meta['setup_status'] = _SetupStatus.PRE_SETUP

    def _resolve_remote_sets(self):
        assert self.pathname == ''  # only call this on the top level model

        if self.comm.allreduce(len(self._remote_sets), op=MPI.SUM) > 0:
            abs2meta_out = self._var_abs2meta['output']
            abs2meta_in = self._var_abs2meta['input']
            for setinfo in self.comm.allgather(self._remote_sets):
                for abs_in, src, val in setinfo:
                    if src in abs2meta_out and abs_in not in abs2meta_in:
                        self.set_val(src, val)

        self._remote_sets = []

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
            try:
                system = system._subsystems_allprocs[subname].system
            except KeyError:
                try:
                    system = system._static_subsystems_allprocs[subname].system
                except KeyError:
                    if name == '':
                        return self
                    return None
        return system

    def run_linearize(self, sub_do_ln=True, driver=None):
        """
        Compute jacobian / factorization.

        This calls _linearize, but with the model assumed to be in an unscaled state.

        Parameters
        ----------
        sub_do_ln : bool
            Flag indicating if the children should call linearize on their linear solvers.
        driver : Driver or None
            If this system is the top level system and approx derivatives have not been
            initialized, the driver for this model must be supplied in order to properly
            initialize the approximations.
        """
        if driver is not None and self.pathname == '' and self._owns_approx_jac:
            self._tot_jac = _TotalJacInfo(driver._problem(), None, None, 'flat_dict', approx=True)

        try:
            super().run_linearize(sub_do_ln=sub_do_ln)
        finally:
            self._tot_jac = None

    def _apply_nonlinear(self):
        """
        Compute residuals. The model is assumed to be in a scaled state.
        """
        self._transfer('nonlinear', 'fwd')
        # Apply recursion
        for subsys in self._relevance.filter(self._subsystems_myproc):
            subsys._apply_nonlinear()

        self.iter_count_apply += 1

    def _solve_nonlinear(self):
        """
        Compute outputs. The model is assumed to be in a scaled state.
        """
        name = self.pathname if self.pathname else 'root'

        with Recording(name + '._solve_nonlinear', self.iter_count, self):
            with self._relevance.active(self._nonlinear_solver.use_relevance()):
                self._nonlinear_solver._solve_with_cache_check()

        # Iteration counter is incremented in the Recording context manager at exit.

    def _guess_nonlinear(self):
        """
        Provide initial guess for states.
        """
        # let any lower level systems do their guessing first
        if self._has_guess:
            # call our own guess_nonlinear method, after the recursion is done to
            # all the lower level systems and the data transfers have happened
            complex_step = self._inputs._under_complex_step

            if complex_step:
                self._inputs.set_complex_step_mode(False)
                self._residuals.set_complex_step_mode(False)
                self._outputs.set_complex_step_mode(False)

            try:
                for sname, sinfo in self._subsystems_allprocs.items():
                    sub = sinfo.system
                    # TODO: could gather 'has_guess' information during setup and be able to
                    # skip transfer for subs that don't have guesses...
                    self._transfer('nonlinear', 'fwd', sname)
                    if sub._is_local and sub._has_guess:
                        sub._guess_nonlinear()

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

    def _apply_linear(self, mode, scope_out=None, scope_in=None):
        """
        Compute jac-vec product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        mode : str
            'fwd' or 'rev'.
        scope_out : set or None
            Set of absolute output names in the scope of this mat-vec product.
            If None, all are in the scope.
        scope_in : set or None
            Set of absolute input names in the scope of this mat-vec product.
            If None, all are in the scope.
        """
        jac = self._get_jacobian()

        if jac is not None and jac is not self._tot_jac:
            with self._matvec_context(scope_out, scope_in, mode) as vecs:
                d_inputs, d_outputs, d_residuals = vecs

                jac._apply(self, d_inputs, d_outputs, d_residuals, mode)

                if self._fd_rev_xfer_correction_dist and mode == 'rev':
                    self._apply_fd_rev_xfer_correction()

        # Apply recursion
        else:
            if mode == 'fwd':
                self._transfer('linear', mode)
                for s in self._relevance.filter(self._subsystems_myproc, relevant=False):
                    # zero out dvecs of irrelevant subsystems
                    s._dresiduals.set_val(0.0)

            for s in self._relevance.filter(self._subsystems_myproc, relevant=True):
                s._apply_linear(mode, scope_out, scope_in)

            if mode == 'rev':
                self._transfer('linear', mode)
                for s in self._relevance.filter(self._subsystems_myproc, relevant=False):
                    # zero out dvecs of irrelevant subsystems
                    s._doutputs.set_val(0.0)

    def _apply_fd_rev_xfer_correction(self):
        """
        Apply the fd reverse transfer correction.

        FD reverse transfer correction is used to correct for the fact that we don't
        do reverse transfers internal to an FD group.  Reverse transfers
        are constructed such that derivative values are correct when transferred into
        system doutput variables, taking into account distributed inputs.
        Since the transfers are not correcting for those issues, we need to do it here.

        If we have a distributed constraint/obj within the FD group and that con/obj is
        active, we perform essentially an allreduce on the d_inputs vars that connect to
        outside systems so they'll include the contribution from all procs.
        """
        seed_vars = self._problem_meta['seed_vars']
        if seed_vars is not None:
            seed_vars = [n for n in seed_vars if n in self._fd_rev_xfer_correction_dist]
            dinvec = self._dinputs
            inarr = dinvec.asarray()
            data = {}
            for seed_var in seed_vars:
                for inp in self._fd_rev_xfer_correction_dist[seed_var]:
                    if inp not in data:
                        if dinvec._contains_abs(inp):  # inp is a local input
                            start, stop = dinvec.get_range(inp)
                            arr = inarr[start:stop]
                            if np.any(arr):
                                data[inp] = arr
                            else:
                                data[inp] = None  # don't send an array of zeros
                        else:
                            data[inp] = None  # prevent possible MPI hangs

            if data:
                myrank = self.comm.rank
                for rank, d in enumerate(self.comm.allgather(data)):
                    if rank != myrank:
                        for n, val in d.items():
                            if val is not None and dinvec._contains_abs(n):
                                start, stop = dinvec.get_range(n)
                                inarr[start:stop] += val

    def _solve_linear(self, mode, scope_out=_UNDEFINED, scope_in=_UNDEFINED):
        """
        Apply inverse jac product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        mode : str
            'fwd' or 'rev'.
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
            with self._relevance.active(self._linear_solver.use_relevance()):
                self._linear_solver.solve(mode, None)

    def _linearize(self, sub_do_ln=True):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        sub_do_ln : bool
            Flag indicating if the children should call linearize on their linear solvers.
        """
        self._check_first_linearize()

        # Group finite difference
        if self._owns_approx_jac:

            with GroupJacobianUpdateContext(self) as jac:
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

            relevance = self._relevance
            with relevance.active(self._linear_solver.use_relevance()):
                subs = list(relevance.filter(self._subsystems_myproc))

                with GroupJacobianUpdateContext(self) as jac:
                    # Only linearize subsystems if we aren't approximating the derivs at this level.
                    for subsys in subs:
                        do_ln = sub_do_ln and (subsys._linear_solver is not None and
                                               subsys._linear_solver._linearize_children())
                        subsys._linearize(sub_do_ln=do_ln)
                        if sub_do_ln and subsys._linear_solver is not None:
                            subsys._linear_solver._linearize()

    def _check_first_linearize(self):
        if self._first_call_to_linearize:
            self._first_call_to_linearize = False  # only do this once
            coloring = self._get_coloring() if coloring_mod._use_partial_sparsity else None

            if coloring is None:
                if self._approx_schemes:
                    # TODO: for top level FD, call below is unnecessary, but we need this
                    # for some tests that just call run_linearize directly without calling
                    # compute_totals.
                    self._setup_approx_derivs()
            else:
                self._setup_approx_coloring()

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
            compatibilty, it can be 'rel', which is now equivalent to 'rel_avg'. Defaults to None,
            in which case the approximation method provides its default value.
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
        graph = self._get_conn_graph()
        nodes = graph.nodes()

        for subsys in self._sorted_subsystems_myproc:
            subsys._setup_partials()
            info.update(subsys._subjacs_info)

        if self._has_distrib_vars and self._owns_approx_jac:
            # We currently cannot approximate across a group with a distributed component if the
            # inputs are distributed via src_indices.
            for iname in self._var_allprocs_abs2meta['input']:
                node = nodes[('i', iname)]
                if node.distributed and node.src_inds_list and iname not in self._conn_abs_in2out:
                    msg = "{}: Approx_totals is not supported on a group with a distributed "
                    msg += "component whose input '{}' is distributed using src_indices. "
                    raise RuntimeError(msg.format(self.msginfo, iname))

    def _setup_residuals(self):
        """
        Call setup_residuals in components.
        """
        for subsys in self._sorted_subsystems_myproc:
            subsys._setup_residuals()

    def _declared_partials_iter(self):
        """
        Iterate over all declared partials.

        Yields
        ------
        key : tuple (of, wrt)
            Subjacobian key.
        """
        for subsys in self._sorted_subsystems_myproc:
            yield from subsys._declared_partials_iter()

    def _get_missing_partials(self, missing):
        """
        Provide (of, wrt) tuples for which derivatives have not been declared in the system.

        Parameters
        ----------
        missing : dict
            Dictionary containing list of missing derivatives keyed by system pathname.
        """
        if self._has_approx:
            return
        for subsys in self._sorted_subsystems_myproc:
            subsys._get_missing_partials(missing)

    def _get_jacobian(self):
        """
        Initialize the jacobian if it is not already initialized.

        Override this in a subclass to use a different jacobian type.

        Returns
        -------
        Jacobian
            The initialized jacobian.
        """
        if self._relevance_changed():
            self._jacobian = None

        if self._jacobian is None:
            if self._owns_approx_jac:
                # Group needs a jacobian if it's approximating derivatives.
                self._jacobian = DictionaryJacobian(system=self)
                self._get_static_wrt_matches()

            if self._jacobian is None:
                self._jacobian = self._get_assembled_jac()

        return self._jacobian

    def _approx_subjac_keys_iter(self):
        """
        Iterate over all approx subjacobian keys needed for this Group.

        These keys include all declared at the Component level below this group, plus any added
        at this group level for approximations, etc.

        Relevance is not considered for these, but will be used later when using these to
        construct a Jacobian.

        Yields
        ------
        tuple
            A tuple of the form (of_name, wrt_name).  Names are absolute (no aliases).
        """
        if self.pathname == '':  # totals
            yield from self._approx_total_subjac_keys_iter()
        else:
            yield from self._approx_semitotal_subjac_keys_iter()

    def _approx_total_subjac_keys_iter(self):
        """
        Iterate over all approx total subjacobian keys needed for this Group.
        """
        if self._owns_approx_of is None:
            return

        ivc = self.get_indep_vars(local=False)

        # When computing totals, weed out inputs connected to anything inside our system unless
        # the source is an indepvarcomp.
        wrt = []
        for name, m in self._owns_approx_wrt.items():
            src = m['source']
            if src in ivc:
                wrt.append(src)
            else:
                raise RuntimeError("When computing total derivatives for the model, the "
                                   f"wrt variable '{name}' is not an independent variable "
                                   "or does not have an independant variable as a source.")

        of = [m['source'] for m in self._owns_approx_of.values()]

        yield from product(of, wrt)

    def _approx_semitotal_subjac_keys_iter(self):
        wrt = []

        for _, abs_inps in self._resolver.prom2abs_iter('input'):
            if abs_inps[0] not in self._conn_abs_in2out:
                # If connection is inside of this Group, perturbation of all implicitly
                # connected inputs will be handled properly via internal transfers.
                # Otherwise, we need to add all implicitly connected inputs separately.
                wrt.extend(abs_inps)

        # get rid of any old stuff in here
        self._owns_approx_of = self._owns_approx_wrt = None

        ivcs = self.get_indep_vars(local=False)
        of = [name for name in self._var_allprocs_abs2meta['output'] if name not in ivcs]

        yield from product(of, wrt)

    def _jac_of_iter(self):
        """
        Iterate over (name, start, end, idxs, dist_sizes) for each 'of' (row) var in the jacobian.

        idxs will usually be the var slice into the full variable in the result array,
        except in cases where _owns_approx__idx has a value for that variable, in which case it'll
        be indices into the variable.

        Yields
        ------
        str
            Absolute name of 'of' variable source.
        int
            Starting index.
        int
            Ending index.
        slice or ndarray
            A full slice or indices for the 'of' variable.
        ndarray or None
            Distributed sizes if var is distributed else None
        """
        if self._owns_approx_of:
            total = self.pathname == ''

            abs2idx = self._var_allprocs_abs2idx
            sizes = self._var_sizes['output']
            graph = self._get_conn_graph()

            # szname = 'global_size' if total else 'size'
            # we're computing totals/semi-totals (vars may not be local)
            start = end = 0
            for name, ofmeta in self._owns_approx_of.items():
                if total:
                    src = ofmeta['source']
                else:
                    src = name

                # for semi-totals, only include outputs that are local
                if not total and src not in self._var_abs2meta['output']:
                    continue

                # meta = abs2meta[src]
                meta = graph.nodes[('o', src)]
                if meta.distributed:
                    dist_sizes = sizes[:, abs2idx[src]]
                else:
                    dist_sizes = None

                indices = ofmeta['indices']
                if indices is not None:  # of in approx_of_idx:
                    end += indices.indexed_src_size
                    yield src, start, end, indices.shaped_array().ravel(), dist_sizes
                else:
                    end += meta.global_size if total else meta.size
                    yield src, start, end, _full_slice, dist_sizes

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
            A full slice or indices for the corresponding design variable when computing total
            derivatives.
        ndarray or None
            Distributed sizes if var is distributed else None
        """
        total = self.pathname == ''

        if self._owns_approx_wrt:
            sizes = self._var_sizes
            toidx = self._var_allprocs_abs2idx
            abs2meta = self._var_allprocs_abs2meta
            local_ins = self._var_abs2meta['input']
            local_outs = self._var_abs2meta['output']
            graph = self._get_conn_graph()

            # szname = 'global_size' if total else 'size'

            seen = set()
            start = end = 0

            for wrt, approx_meta in self._owns_approx_wrt.items():
                if total:
                    wrt = approx_meta['source']
                    if approx_meta['remote']:
                        vec = None
                    else:
                        vec = self._outputs
                else:
                    if wrt in local_ins:
                        vec = self._inputs
                    elif wrt in local_outs:
                        vec = self._outputs
                    else:
                        vec = None  # remote wrt

                if (wrt_matches is None or wrt in wrt_matches) and wrt not in seen:
                    io = 'input' if wrt in abs2meta['input'] else 'output'
                    # meta = abs2meta[io][wrt]
                    meta = graph.nodes[(io[0], wrt)]
                    if total and approx_meta['indices'] is not None:
                        sub_wrt_idx = approx_meta['indices'].as_array()
                        size = sub_wrt_idx.size
                    else:
                        sub_wrt_idx = _full_slice
                        size = meta.global_size if total else meta.size
                    if vec is None:
                        sub_wrt_idx = ValueRepeater(None, size)
                    end += size
                    dist_sizes = sizes[io][:, toidx[wrt]] if meta.distributed else None
                    yield wrt, start, end, vec, sub_wrt_idx, dist_sizes
                    start = end
        else:
            yield from super()._get_jac_wrts(wrt_matches)

    def _promoted_wrt_iter(self):
        if not (self._owns_approx_of or self.pathname):
            return

        resolver = self._resolver
        seen = set()
        for _, wrt in self._get_approx_subjac_keys():
            if wrt not in seen:
                seen.add(wrt)
                prom = resolver.abs2prom(wrt)
                if prom not in seen:
                    seen.add(prom)
                    yield prom

    def _setup_approx_derivs(self):
        """
        Add approximations for all approx derivs.
        """
        total = self.pathname == ''
        abs2meta_out = self._var_allprocs_abs2meta['output']
        abs2meta_in = self._var_allprocs_abs2meta['input']

        if self._coloring_info.coloring is not None and (self._owns_approx_of is None or
                                                         self._owns_approx_wrt is None):
            method = self._coloring_info.method
        else:
            method = list(self._approx_schemes)[0]

        wrt_matches = self._get_static_wrt_matches()

        approx = self._get_approx_scheme(method)
        # reset the approx if necessary
        approx._wrt_meta = {}
        approx._reset()

        sizes_out = self._var_sizes['output']
        sizes_in = self._var_sizes['input']
        abs2idx = self._var_allprocs_abs2idx

        approx_keys = self._get_approx_subjac_keys(initialize=True)
        wrt_seen = set()
        compute_cross_keys = not total and self.comm.size > 1 and self._has_fd_group
        graph = self._problem_meta['dataflow_graph']

        for key in approx_keys:
            of, wrt = key

            if key in self._subjacs_info:
                meta = self._subjacs_info[key]
            else:
                # check connection between of and wrt before creating a subjac
                if not are_connected(graph, wrt, of):
                    continue

                meta = SUBJAC_META_DEFAULTS.copy()
                osize = abs2meta_out[of]['size']
                if wrt in abs2meta_in:
                    isize = abs2meta_in[wrt]['size']
                else:
                    isize = abs2meta_out[wrt]['size']

                if of == wrt:
                    meta['diagonal'] = True
                    # All group approximations are treated as explicit components, so we
                    # have a -1 on the diagonal.
                    meta['val'] = np.full(osize, -1.0)

                self._subjacs_info[key] = meta = Subjac.get_instance_metadata(meta, None,
                                                                              (osize, isize),
                                                                              self, key)

            if compute_cross_keys:
                sout = sizes_out[:, abs2idx[of]]
                sin = sizes_in[:, abs2idx[wrt]]
                if np.count_nonzero(sout[sin == 0]) > 0 and np.count_nonzero(sin[sout == 0]) > 0:
                    self._cross_keys.add(key)

            meta['method'] = method

            meta.update(self._owns_approx_jac_meta)

            if wrt_matches is None or wrt in wrt_matches:
                self._update_approx_coloring_meta(meta)

            if wrt not in wrt_seen:
                wrt_seen.add(wrt)
                approx.add_approximation(wrt, self, meta)

        if not total:
            # we're taking semi-total derivs for this group. Update _owns_approx_of
            # and _owns_approx_wrt so we can use the same approx code for totals and
            # semi-totals.  Also, the order must match order of vars in the output and
            # input vectors.
            resolver = self._resolver

            self._owns_approx_of = {}
            for n, m in self._var_allprocs_abs2meta['output'].items():
                self._owns_approx_of[n] = dct = m.copy()
                dct['name'] = resolver.abs2prom(n, 'output')
                dct['source'] = n
                dct['indices'] = None

            wrtset = set([wrt for _, wrt in approx_keys])

            self._owns_approx_wrt = {}
            for n, m in self._var_allprocs_abs2meta['input'].items():
                if n in wrtset:
                    self._owns_approx_wrt[n] = {'distributed': m['distributed']}

            self._owns_approx_jac = True

    def _setup_approx_coloring(self):
        """
        Ensure that if coloring is declared, approximations will be set up.
        """
        if self._coloring_info.coloring is not None:
            self.approx_totals(self._coloring_info.method,
                               self._coloring_info.get('step'),
                               self._coloring_info.get('form'))
        self._setup_approx_derivs()

    def _setup_check(self):
        """
        Do any error checking on user's setup, before any other recursion happens.
        """
        if (self._coloring_info.static or self._coloring_info.dynamic) and self.pathname != '':
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

    def compute_sys_graph(self, comps_only=False, add_edge_info=True):
        """
        Compute a dependency graph for subsystems in this group.

        Variable connection information is stored in each edge of
        the system graph if comps_only is True and add_edge_info is True.

        Parameters
        ----------
        comps_only : bool (False)
            If True, return a graph of all components within this group
            or any of its descendants. No sub-groups will be included. Otherwise,
            a graph containing only direct children (both Components and Groups)
            of this group will be returned.
        add_edge_info : bool (True)
            If True and comps_only is also True, store variable connection information in each
            edge of the system graph.

        Returns
        -------
        DiGraph
            A directed graph containing names of subsystems and their connections.
        """
        graph = nx.DiGraph()

        if comps_only:
            # add all components as nodes in the graph so they'll be there even if unconnected.
            comps = set(v.rpartition('.')[0] for v in self._resolver.abs_iter())
            graph.add_nodes_from(comps)

            if add_edge_info:
                edge_data = defaultdict(lambda: defaultdict(list))
                for in_abs, src_abs in self._conn_global_abs_in2out.items():
                    src_sys = src_abs.rpartition('.')[0]
                    tgt_sys = in_abs.rpartition('.')[0]

                    # store var connection data in each system to system edge.
                    edge_data[(src_sys, tgt_sys)][src_abs].append(in_abs)

                for (src_sys, tgt_sys), data in edge_data.items():
                    graph.add_edge(src_sys, tgt_sys, conns=data)
            else:
                for in_abs, src_abs in self._conn_global_abs_in2out.items():
                    graph.add_edge(src_abs.rpartition('.')[0], in_abs.rpartition('.')[0])
        else:
            if self._sys_graph_cache is None:
                # add all systems as nodes in the graph so they'll be there even if unconnected.
                graph.add_nodes_from(self._subsystems_allprocs)

                glen = self.pathname.count('.') + 1 if self.pathname else 0
                for in_abs, src_abs in self._conn_global_abs_in2out.items():
                    src_sys = src_abs.split('.', glen + 1)[glen]
                    tgt_sys = in_abs.split('.', glen + 1)[glen]
                    if src_sys != tgt_sys:
                        graph.add_edge(src_sys, tgt_sys)
                self._sys_graph_cache = graph
            else:
                graph = self._sys_graph_cache

        return graph

    def _setup_auto_ivcs(self):
        # only happens at top level
        from openmdao.core.indepvarcomp import _AutoIndepVarComp

        # create the IndepVarComp that will contain all auto-ivc outputs
        self._auto_ivc = auto_ivc = _AutoIndepVarComp()
        auto_ivc.name = '_auto_ivc'
        auto_ivc.pathname = auto_ivc.name

        graph = self._get_conn_graph()

        auto2tgt = {}
        for srcnode in graph.nodes():
            io, src = srcnode
            if io == 'o' and src.startswith('_auto_ivc.'):
                auto2tgt[src] = [n for _, n in graph.leaf_input_iter(srcnode)]

        auto_ivc.auto2tgt = auto2tgt

        for src, tgts in auto2tgt.items():
            abs_tgt_node = graph.nodes[('i', tgts[0])]
            # need to query discrete from a leaf node because internal nodes have not resolved yet
            discrete = abs_tgt_node['discrete']
            remote = False  # TODO: fix this

            relsrc = src.rsplit('.', 1)[-1]
            if discrete:
                auto_ivc.add_discrete_output(relsrc, val=abs_tgt_node['val'])
            else:
                auto_ivc.add_output(relsrc)
            if remote:
                auto_ivc._add_remote(relsrc)

        if not auto2tgt:
            return auto_ivc

        auto_ivc._setup_procs(auto_ivc.pathname, self.comm, self._problem_meta)
        auto_ivc._configure()
        auto_ivc._configure_check()
        auto_ivc._setup_var_data()

        # bind auto_ivc conn graph nodes to their variable metadata
        locmeta = auto_ivc._var_abs2meta['output']
        for name, meta in auto_ivc._var_allprocs_abs2meta['output'].items():
            graph.set_model_meta(self, ('o', name), meta, locmeta.get(name))

        locmeta = auto_ivc._var_discrete['output']
        for name, meta in auto_ivc._var_allprocs_discrete['output'].items():
            node_meta = graph.nodes[('o', name)]
            node_meta.discrete = True
            graph.set_model_meta(self, ('o', name), meta, locmeta.get(name.rpartition('.')[-1]))

        # now update our own data structures based on the new auto_ivc component variables
        old = self._subsystems_allprocs
        self._subsystems_allprocs = allsubs = {}
        allsubs['_auto_ivc'] = _SysInfo(auto_ivc, 0)
        for i, (name, s) in enumerate(old.items()):
            allsubs[name] = s
            s.index = i + 1

        self._subsystems_myproc = [auto_ivc] + self._subsystems_myproc
        self._sorted_subsystems_myproc = list(self._sorted_subsystems_myproc_iter())

        self._resolver._auto_ivc_update(auto_ivc._resolver, auto2tgt)

        io = 'output'  # auto_ivc has only output vars
        self._var_discrete[io].update({'_auto_ivc.' + k: v for k, v in
                                       auto_ivc._var_discrete[io].items()})
        self._var_allprocs_discrete[io].update(auto_ivc._var_allprocs_discrete[io])

        self._var_abs2meta[io] = {}
        self._var_allprocs_abs2meta[io] = {}
        # rebuild _var_abs2meta and _var_allprocs_abs2meta in the correct order
        for s in self._sorted_subsystems_myproc:
            self._var_abs2meta[io].update(s._var_abs2meta[io])
            self._var_allprocs_abs2meta[io].update(s._var_allprocs_abs2meta[io])

        self._var_allprocs_abs2idx.update(
            {n: i for i, n in enumerate(self._var_allprocs_abs2meta[io])})

        self._approx_subjac_keys = None  # this will force re-initialization
        self._setup_procs_finished = True

        return auto_ivc

    def _ordered_comp_name_iter(self):
        """
        Yield contained component pathnames in order of execution.

        For components within ParallelGroups, true execution order is unknown so components
        will be ordered by rank within a ParallelGroup.
        """
        for s in self._subsystems_myproc:
            if isinstance(s, Group):
                yield from s._ordered_comp_name_iter()
            else:
                yield s.pathname

    def _sorted_subsystems_myproc_iter(self):
        """
        Yield subsystems in sorted order if Problem option allow_post_setup_reorder is True.

        Otherwise, yield subsystems in the order they were added to their parent group.

        Yields
        ------
        System
            A subsystem.
        """
        if self._problem_meta['allow_post_setup_reorder']:
            for s in sorted(self._subsystems_myproc, key=lambda s: s.name):
                yield s
        else:
            yield from self._subsystems_myproc

    def _sorted_subsystems_allprocs_iter(self):
        """
        Yield subsystem names in sorted order if Problem option allow_post_setup_reorder is True.

        Otherwise, yield subsystem names in the order they were added to their parent group.

        Yields
        ------
        System
            A subsystem.
        """
        if self._problem_meta['allow_post_setup_reorder']:
            for s in sorted(self._subsystems_allprocs):
                yield s
        else:
            yield from self._subsystems_allprocs

    def _all_subsystem_iter(self):
        """
        Iterate over all subsystems, local and nonlocal.

        Yields
        ------
        System
            A subsystem.
        """
        for s, _ in self._subsystems_allprocs.values():
            yield s

    def _get_relevance_modifiers(self, grad_groups, always_opt_comps):
        """
        Collect information from the model that will modify the relevance graph of the model.

        Parameters
        ----------
        grad_groups : set
            Set of groups having nonlinear solvers that use gradients.
        always_opt_comps : set
            Set of components that are to be included in every iteration of the optimization,
            even if they aren't relevant in terms of data flow.
        """
        if self.nonlinear_solver is not None and self.nonlinear_solver.supports['gradients']:
            grad_groups.add(self.pathname)
        elif self.linear_solver is not None and isinstance(self.linear_solver, DirectSolver):
            grad_groups.add(self.pathname)

        for s in self._subsystems_myproc:
            if isinstance(s, Group):
                s._get_relevance_modifiers(grad_groups, always_opt_comps)
            elif s.options['always_opt']:
                always_opt_comps.add(s.pathname)

    @property
    def model_options(self):
        """
        Get the model options from self._problem_meta.

        The user may change the contents of model_options to impact values sent
        to subsystems of this Group.

        Returns
        -------
        dict
            The model options metadata provided by the associated Problem object.
        """
        return self._problem_meta['model_options']

    def _gather_full_data(self):
        """
        Return True if this system should contribute full data to a collective MPI call.

        This prevents sending a lot of unnecessary data across the network when
        the data is duplicated across multiple processes.

        Returns
        -------
        bool
            True if this system should contribute its full data. Otherwise it
            should contribute only an 'empty' version of the data.  What
            'empty' means depends on the structure of the data being gathered.
        """
        if self._mpi_proc_allocator.parallel:
            if self._subsystems_myproc and self._subsystems_myproc[0].comm.rank == 0:
                return self._subsystems_myproc[0]._full_comm is None or \
                    self._subsystems_myproc[0]._full_comm.rank == 0

        return False

    def get_design_vars(self, recurse=True, get_sizes=True, use_prom_ivc=True):
        """
        Get the DesignVariable settings from this system.

        Retrieve all design variable settings from the system and, if recurse
        is True, all of its subsystems.

        Parameters
        ----------
        recurse : bool
            If True, recurse through the subsystems and return the path of
            all design vars relative to the this Group.
        get_sizes : bool, optional
            If True, compute the size of each design variable.
        use_prom_ivc : bool
            Use promoted names for inputs, else convert to absolute source names.

        Returns
        -------
        dict
            The design variables defined in the current system and, if
            recurse=True, its subsystems.
        """
        out = super().get_design_vars(recurse=recurse, get_sizes=get_sizes,
                                      use_prom_ivc=use_prom_ivc)
        if recurse:
            resolver = self._resolver
            if (self.comm.size > 1 and self._mpi_proc_allocator.parallel):

                # For parallel groups, we need to make sure that the design variable dictionary is
                # assembled in the same order under mpi as for serial runs.
                out_by_sys = {}

                for subsys in self._sorted_subsystems_myproc:
                    sub_out = {}
                    name = subsys.name
                    dvs = subsys.get_design_vars(recurse=recurse, get_sizes=get_sizes,
                                                 use_prom_ivc=use_prom_ivc)
                    if use_prom_ivc:
                        # have to promote subsystem prom name to this level's prom name
                        subres = subsys._resolver
                        for dv, meta in dvs.items():
                            if subres.is_prom(dv, 'input'):
                                abs_dv = subres.absnames(dv, 'input')[0]
                                sub_out[resolver.abs2prom(abs_dv, 'input')] = meta
                            elif subres.is_prom(dv, 'output'):
                                abs_dv = subres.absnames(dv, 'output')[0]
                                sub_out[resolver.abs2prom(abs_dv, 'output')] = meta
                            else:
                                sub_out[dv] = meta
                    else:
                        sub_out.update(dvs)

                    out_by_sys[name] = sub_out

                out_by_sys_by_rank = self.comm.allgather(out_by_sys)
                all_outs_by_sys = {}
                for outs in out_by_sys_by_rank:
                    for name, meta in outs.items():
                        all_outs_by_sys[name] = meta

                for subsys_name in self._sorted_subsystems_allprocs_iter():
                    for name, meta in all_outs_by_sys[subsys_name].items():
                        if name not in out:
                            out[name] = meta

            else:

                for subsys in self._sorted_subsystems_myproc:
                    dvs = subsys.get_design_vars(recurse=recurse, get_sizes=get_sizes,
                                                 use_prom_ivc=use_prom_ivc)
                    if use_prom_ivc:
                        # have to promote subsystem prom name to this level
                        subres = subsys._resolver
                        for dv, meta in dvs.items():
                            if subres.is_prom(dv, 'input'):
                                out[resolver.prom2prom(dv, subsys._resolver, 'input')] = meta
                            elif subres.is_prom(dv, 'output'):
                                out[resolver.prom2prom(dv, subsys._resolver, 'output')] = meta
                            else:
                                out[dv] = meta
                    else:
                        out.update(dvs)

        if self.pathname == '':  # do this at the top level only
            abs2meta_out = self._var_allprocs_abs2meta['output']
            for outmeta in out.values():
                src = outmeta['source']
                if src in abs2meta_out and "openmdao:allow_desvar" not in abs2meta_out[src]['tags']:
                    prom_src, prom_tgt = outmeta['orig']
                    if prom_src is None:
                        self._collect_error(f"Design variable '{prom_tgt}' is connected to '{src}',"
                                            f" but '{src}' is not an IndepVarComp or ImplicitComp "
                                            "output.")
                    else:
                        self._collect_error(f"Design variable '{prom_src}' is not an IndepVarComp "
                                            "or ImplicitComp output.")

        return out

    def get_responses(self, recurse=True, get_sizes=True, use_prom_ivc=False):
        """
        Get the response variable settings from this system.

        Retrieve all response variable settings from the system as a dict,
        keyed by either absolute variable name, promoted name, or alias name,
        depending on the value of use_prom_ivc and whether the original key was
        a promoted output, promoted input, or an alias.

        Parameters
        ----------
        recurse : bool, optional
            If True, recurse through the subsystems and return the path of
            all responses relative to the this system.
        get_sizes : bool, optional
            If True, compute the size of each response.
        use_prom_ivc : bool
            Translate ivc names to their promoted input names.

        Returns
        -------
        dict
            The responses defined in the current system and, if
            recurse=True, its subsystems.
        """
        out = super().get_responses(recurse=recurse, get_sizes=get_sizes, use_prom_ivc=use_prom_ivc)
        if recurse:
            prom2prom = self._resolver.prom2prom
            if self.comm.size > 1 and self._mpi_proc_allocator.parallel:
                # For parallel groups, we need to make sure that the response dictionary is
                # assembled in the same order under mpi as for serial runs.
                out_by_sys = {}

                for subsys in self._sorted_subsystems_myproc:
                    name = subsys.name
                    sub_out = {}

                    subresps = subsys.get_responses(recurse=recurse, get_sizes=get_sizes,
                                                    use_prom_ivc=use_prom_ivc)
                    if use_prom_ivc:
                        subresolver = subsys._resolver
                        # have to promote subsystem prom name to this level
                        for res, meta in subresps.items():
                            sub_out[prom2prom(res, subresolver, 'output')] = meta
                    else:
                        for rkey, rmeta in subresps.items():
                            if rkey in out:
                                tdict = {'con': 'constraint', 'obj': 'objective'}
                                rpath = rmeta['parent']
                                rname = '.'.join((rpath, rmeta['name'])) if rpath else rkey
                                rtype = tdict[rmeta['type']]
                                ometa = sub_out[rkey]
                                opath = ometa['parent']
                                oname = '.'.join((opath, ometa['name'])) if opath else ometa['name']
                                otype = tdict[ometa['type']]
                                raise NameError(f"The same response alias, '{rkey}' was declared"
                                                f" for {rtype} '{rname}' and {otype} '{oname}'.")
                            sub_out[rkey] = rmeta

                    out_by_sys[name] = sub_out

                out_by_sys_by_rank = self.comm.allgather(out_by_sys)
                all_outs_by_sys = {}
                for outs in out_by_sys_by_rank:
                    for name, meta in outs.items():
                        all_outs_by_sys[name] = meta

                for subsys_name in self._sorted_subsystems_allprocs_iter():
                    for name, meta in all_outs_by_sys[subsys_name].items():
                        out[name] = meta

            else:
                for subsys in self._sorted_subsystems_myproc:
                    subresps = subsys.get_responses(recurse=recurse, get_sizes=get_sizes,
                                                    use_prom_ivc=use_prom_ivc)
                    if use_prom_ivc:
                        subresolver = subsys._resolver
                        # have to promote subsystem prom name to this level
                        for res, meta in subresps.items():
                            out[prom2prom(res, subresolver, 'output')] = meta
                    else:
                        for rkey, rmeta in subresps.items():
                            if rkey in out:
                                tdict = {'con': 'constraint', 'obj': 'objective'}
                                rpath = rmeta['parent']
                                rname = '.'.join((rpath, rmeta['name'])) if rpath else rkey
                                rtype = tdict[rmeta['type']]
                                ometa = out[rkey]
                                opath = ometa['parent']
                                oname = '.'.join((opath, ometa['name'])) if opath else ometa['name']
                                otype = tdict[ometa['type']]
                                raise NameError(f"The same response alias, '{rkey}' was declared"
                                                f" for {rtype} '{rname}' and {otype} '{oname}'.")
                            out[rkey] = rmeta

        return out

    def _get_totals_metadata(self, driver=None, of=None, wrt=None):
        if isinstance(of, str):
            of = [of]
        if isinstance(wrt, str):
            wrt = [wrt]

        if not driver:
            if of is None or wrt is None:
                raise RuntimeError("driver must be specified if of and wrt variables are not "
                                   "provided.")

            if driver is False:  # force to not use any existing desvar or response metadata
                return self._active_responses(of, responses=False), \
                    self._active_desvars(wrt, designvars=False), True

            return self._active_responses(of), self._active_desvars(wrt), True

        has_custom_derivs = False
        list_wrt = list(wrt) if wrt is not None else []
        list_of = list(of) if of is not None else []

        lincons = [d for d, meta in driver._cons.items() if meta['linear']]
        if lincons and list_of == lincons:  # this is for a linear jacobian
            driver_wrt = list(driver._get_lin_dvs())
        else:
            driver_wrt = list(driver._get_nl_dvs())

        if wrt is None:
            wrt = driver_wrt
            if not wrt:
                raise RuntimeError("No design variables were passed to compute_totals and "
                                   "the driver is not providing any.")
        else:
            if list_wrt != driver_wrt:
                wrt_src_names = [driver._designvars[n]['source'] for n in driver_wrt]
                if list_wrt != wrt_src_names:
                    has_custom_derivs = True

        driver_ordered_nl_resp_names = driver._get_ordered_nl_responses()
        if of is None:
            of = driver_ordered_nl_resp_names
            if not of:
                raise RuntimeError("No response variables were passed to compute_totals and "
                                   "the driver is not providing any.")
        else:
            of_src_names = [m['source'] for n, m in driver._responses.items()
                            if n in driver_ordered_nl_resp_names]
            of = list(of)
            if of != driver_ordered_nl_resp_names and of != of_src_names:
                has_custom_derivs = True

        return self._active_responses(of, driver._responses), \
            self._active_desvars(wrt, driver._designvars), has_custom_derivs

    def _active_desvars(self, user_dv_names, designvars=None):
        """
        Return a design variable dictionary.

        Whatever names match the names of design variables in this system will use the metadata
        from the design variable.  For other variables that have not been registered as design
        variables, metadata will be constructed based on variable metadata.

        Parameters
        ----------
        user_dv_names : iter of str
            Iterator of user facing design variable names.
        designvars : dict, None, or False
            Dictionary of design variables.  If None, get_design_vars will be called. If False,
            no design vars will be used.

        Returns
        -------
        dict
            Dictionary of design variables.
        """
        # do this to keep ordering the same as in the user list
        active_dvs = {n: None for n in user_dv_names}

        if designvars is None:
            designvars = self.get_design_vars(recurse=True, get_sizes=True, use_prom_ivc=True)

        if designvars:  # use any matching metadata from existing design vars
            for name, meta in designvars.items():
                if name in active_dvs:
                    active_dvs[name] = meta.copy()
                elif meta['name'] in active_dvs:
                    active_dvs[meta['name']] = meta.copy()
                elif meta['source'] in active_dvs:
                    active_dvs[meta['source']] = meta.copy()

        is_prom = self._resolver.is_prom
        is_local = self._resolver.is_local

        for name, meta in active_dvs.items():
            if meta is None:
                meta = {
                    'parallel_deriv_color': None,
                    'indices': None,
                    'name': name,
                    'cache_linear_solution': False,
                }
                self._update_dv_meta(meta, get_size=True)

                if is_prom(name, 'input'):
                    meta['ivc_print_name'] = name
                else:
                    meta['ivc_print_name'] = None

                active_dvs[name] = meta

            meta['remote'] = not is_local(meta['source'], 'output')

        return active_dvs

    def _active_responses(self, user_response_names, responses=None):
        """
        Return a response dictionary containing the given variables.

        Whatever names match the names of responses in this system, use the metadata
        from the response.  For other variables that have not been registered as responses,
        construct metadata based on variable metadata.

        Parameters
        ----------
        user_response_names : iter of str
            Iterator of user facing response names.  Aliases are allowed.
        responses : dict, None, or False.
            Dictionary of responses.  If None, get_responses will be called. If False,
            no responses will be used.

        Returns
        -------
        dict
            Dictionary of responses.
        """
        # do this to keep ordering the same as in the user list
        active_resps = {n: None for n in user_response_names}

        if responses is None:
            responses = self.get_responses(recurse=True, get_sizes=True, use_prom_ivc=True)

        if responses:
            for name, meta in responses.items():
                if name in active_resps:
                    active_resps[name] = meta.copy()

        is_local = self._resolver.is_local

        for name, meta in active_resps.items():
            if meta is None:
                # no response exists for this name, so create metadata with default values and
                # update size, etc. based on the variable metadata.
                meta = {
                    'parallel_deriv_color': None,
                    'indices': None,
                    'alias': None,
                    'name': name,
                    'cache_linear_solution': False,
                    'linear': False,
                }
                self._update_response_meta(meta, get_size=True)
                active_resps[name] = meta

            meta['remote'] = not is_local(meta['source'], 'output')

        return active_resps

    def _get_graph_node_meta(self):
        """
        Return metadata to add to this system's graph node.

        Returns
        -------
        dict
            Metadata for this system's graph node.
        """
        meta = super()._get_graph_node_meta()
        # TODO: maybe set 'implicit' based on whether there are any implicit comps anywhere
        # inside of the group or its children.
        meta['base'] = 'Group'
        return meta

    def display_dataflow_graph(self, recurse=True, show_vars=True, show_boundary=False, exclude=(),
                               outfile=None):
        from openmdao.visualization.graph_viewer import GraphViewer

        GraphViewer(self).write_graph(gtype='dataflow', recurse=recurse,
                                      show_vars=show_vars, display=True,
                                      exclude=exclude, show_boundary=show_boundary,
                                      outfile=outfile)

    def display_conn_graph(self, varname=None, outfile=None):
        self._get_conn_graph().display(self.pathname, varname=varname, outfile=outfile)

    def _column_iotypes(self):
        """
        Return a tuple of the iotypes that make up columns of the jacobian.

        Returns
        -------
        tuple of the form ('input',)
            The iotypes that make up columns of the jacobian.
        """
        return ('input',)

    def _get_subjac_owners(self):
        """
        Return a dictionary of owning rank keyed by subjac key.

        This only applies to groups approximating a jacobian using FD or CS, and is empty unless
        there are of or wrt variables that are not local in at least one rank.

        It is also empty in the top level group, because when computing total jacobians, the
        subjacobians exist locally on each process.

        The return value is cached in self._key_owner.

        Returns
        -------
        dict
            Dictionary of owning rank keyed by subjac key.
        """
        if self._key_owner is None:
            if self.pathname and self.comm.size > 1 and self._owns_approx_jac:
                allouts = self._var_allprocs_abs2meta['output']
                local_out = self._var_abs2meta['output']
                local_in = self._var_abs2meta['input']
                remote_keys = []
                for key in self._jacobian._get_ordered_subjac_keys(self):
                    of, wrt = key
                    if of not in local_out or (wrt not in local_in and wrt not in local_out):
                        remote_keys.append(key)

                abs2idx = self._var_allprocs_abs2idx
                sizes_out = self._var_sizes['output']
                sizes_in = self._var_sizes['input']
                owner_dict = {}
                for keys in self.comm.allgather(remote_keys):
                    for key in keys:
                        if key not in owner_dict:
                            of, wrt = key
                            ofsizes = sizes_out[:, abs2idx[of]]
                            if wrt in allouts:
                                wrtsizes = sizes_out[:, abs2idx[wrt]]
                            else:
                                wrtsizes = sizes_in[:, abs2idx[wrt]]
                            for rank, (ofsz, wrtsz) in enumerate(zip(ofsizes, wrtsizes)):
                                # find first rank where both of and wrt are local
                                if ofsz and wrtsz:
                                    owner_dict[key] = rank
                                    break
                            else:  # no rank was found where both were local. Use 'of' local rank
                                owner_dict[key] = np.min(np.nonzero(ofsizes)[0])

                self._key_owner = owner_dict
            else:
                self._key_owner = {}

        return self._key_owner

    def debug_dump(self):
        print('subsystems:', [s.pathname for s in self._subsystems_myproc])
        print('sorted_subsystems:', [s.pathname for s in self._sorted_subsystems_myproc])
        print('global_inputs:', list(self._var_allprocs_abs2meta['input']))
        print('local_inputs:', list(self._var_abs2meta['input']))
        print('inputs:', list(self._inputs))
        print('invec:', self._inputs.asarray())
        print('global_outputs:', list(self._var_allprocs_abs2meta['output']))
        print('local_outputs:', list(self._var_abs2meta['output']))
        print('outputs:', list(self._outputs))
        print('outvec:', self._outputs.asarray())
        import pprint
        print('var_sizes:')
        pprint.pprint(self._var_sizes)
        print('abs2idx:')
        pprint.pprint(self._var_allprocs_abs2idx)
        print('', flush=True)


def iter_solver_info(system):
    """
    Return solver information for this System.

    Parameters
    ----------
    system : System
        Return solver information for this System.

    Returns
    -------
    str
        System pathname.
    str
        Class name.
    list of sets of str
        Strongly connected components in this Group's subsystem graph.  If not a Group, this will
        be an empty list.
    str
        Linear solver class name.
    str
        Nonlinear solver class name.
    int
        Linear solver max iterations.
    int
        Nonlinear solver max iterations.
    int
        Number of subsystems that are not part of any strongly connected component. If this
        number is greater than 0 and strongly connected components exist in this group, it
        indicates that this group contains subcycles and that it may be more efficient to
        separate those subcyles into their own groups and apply iterative solvers to them.
    bool
        True if this is a Group, False if it is an ImplicitComponent.
    bool
        True if the linear solver found for this System can solve a cycle or implicit component.
    bool
        True if the nonlinear solver found for this System can solve a cycle or implicit component.
    """
    sccs = []
    missing = 0
    lnmaxiter = nlmaxiter = 1
    nl_can_solve = lin_can_solve = False
    nlslvname = lnslvname = None

    if isinstance(system, Group):
        isgrp = True
        for s in get_sccs_topo(system.compute_sys_graph()):
            if len(s) > 1:
                sccs.append(s)
            else:
                missing += 1
    elif isinstance(system, ImplicitComponent):
        isgrp = False
    else:
        return (system.pathname, system.__class__.__name__, sccs, None, None, 0, 0, 0, False,
                False, False)

    if system.nonlinear_solver:
        if isgrp:
            nl_can_solve = system.nonlinear_solver.can_solve_cycle()
        else:
            nl_can_solve = system.nonlinear_solver.supports['implicit_components']
        nlslvname = system.nonlinear_solver.__class__.__name__
        if 'maxiter' in system.nonlinear_solver.options:
            nlmaxiter = system.nonlinear_solver.options['maxiter']

    if system.linear_solver:
        if isgrp:
            lin_can_solve = system.linear_solver.can_solve_cycle()
        else:
            lin_can_solve = system.linear_solver.supports['implicit_components']
        lnslvname = system.linear_solver.__class__.__name__

    if lnslvname and 'maxiter' in system.linear_solver.options:
        lnmaxiter = system.linear_solver.options['maxiter']

    if not isgrp:
        if lnslvname is None and system._has_solve_linear:
            lnslvname = 'solve_linear'
            lin_can_solve = True
        if nlslvname is None and system._has_solve_nl:
            nlslvname = 'solve_nonlinear'
            nl_can_solve = True

    return (system.pathname, system.__class__.__name__, sccs, lnslvname, nlslvname, lnmaxiter,
            nlmaxiter, missing, isgrp, nl_can_solve, lin_can_solve)
