"""Define the base System class."""
import sys
import os
import time

from contextlib import contextmanager
from collections import OrderedDict, defaultdict
from collections.abc import Iterable
from itertools import chain
from enum import IntEnum

import re
from fnmatch import fnmatchcase

from numbers import Integral

import numpy as np
import networkx as nx

import openmdao
from openmdao.core.configinfo import _ConfigInfo
from openmdao.core.constants import _DEFAULT_OUT_STREAM, _UNDEFINED, INT_DTYPE
from openmdao.jacobians.assembled_jacobian import DenseJacobian, CSCJacobian
from openmdao.recorders.recording_manager import RecordingManager
from openmdao.vectors.vector import _full_slice
from openmdao.utils.mpi import MPI
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.record_util import create_local_meta, check_path
from openmdao.utils.units import is_compatible, unit_conversion, valid_units, simplify_unit
from openmdao.utils.variable_table import write_var_table
from openmdao.utils.array_utils import evenly_distrib_idxs, _flatten_src_indices
from openmdao.utils.graph_utils import all_connected_nodes
from openmdao.utils.name_maps import name2abs_name, name2abs_names
from openmdao.utils.coloring import _compute_coloring, Coloring, \
    _STD_COLORING_FNAME, _DEF_COMP_SPARSITY_ARGS
import openmdao.utils.coloring as coloring_mod
from openmdao.utils.general_utils import determine_adder_scaler, \
    format_as_float_or_array, ContainsAll, all_ancestors, _slice_indices, \
    simple_warning, make_set, match_prom_or_abs, _is_slicer_op, shape_from_idx
from openmdao.approximation_schemes.complex_step import ComplexStep
from openmdao.approximation_schemes.finite_difference import FiniteDifference
from openmdao.utils.units import unit_conversion


_empty_frozen_set = frozenset()

_asm_jac_types = {
    'csc': CSCJacobian,
    'dense': DenseJacobian,
}

# Suppored methods for derivatives
_supported_methods = {
    'fd': FiniteDifference,
    'cs': ComplexStep,
    'exact': None
}

_DEFAULT_COLORING_META = {
    'wrt_patterns': ('*',),  # patterns used to match wrt variables
    'method': 'fd',          # finite differencing method  ('fd' or 'cs')
    'wrt_matches': None,     # where matched wrt names are stored
    'per_instance': True,    # assume each instance can have a different coloring
    'coloring': None,        # this will contain the actual Coloring object
    'dynamic': False,        # True if dynamic coloring is being used
    'static': None,          # either _STD_COLORING_FNAME, a filename, or a Coloring object
                             # if use_fixed_coloring was called
}

_DEFAULT_COLORING_META.update(_DEF_COMP_SPARSITY_ARGS)

_recordable_funcs = frozenset(['_apply_linear', '_apply_nonlinear', '_solve_linear',
                               '_solve_nonlinear'])

# the following are local metadata that will also be accessible for vars on all procs
global_meta_names = {
    'input': ('units', 'shape', 'size', 'distributed', 'tags', 'desc', 'shape_by_conn',
              'copy_shape'),
    'output': ('units', 'shape', 'size', 'desc',
               'ref', 'ref0', 'res_ref', 'distributed', 'lower', 'upper', 'tags', 'shape_by_conn',
               'copy_shape'),
}

allowed_meta_names = {
    'value',
    'global_shape',
    'global_size',
    'src_indices',
    'src_slice',
    'flat_src_indices',
    'type',
    'res_units',
}
allowed_meta_names.update(global_meta_names['input'])
allowed_meta_names.update(global_meta_names['output'])


class _MatchType(IntEnum):
    """
    Class used to define different types of promoted name matches.

    Attributes
    ----------
    NAME : int
        Literal name match.
    RENAME : int
        Rename match.
    PATTERN : int
        Glob pattern match.
    """

    NAME = 0
    RENAME = 1
    PATTERN = 2


class System(object):
    """
    Base class for all systems in OpenMDAO.

    Never instantiated; subclassed by <Group> or <Component>.
    All subclasses have their attributes defined here.

    In attribute names:
        abs / abs_name : absolute, unpromoted variable name, seen from root (unique).
        rel / rel_name : relative, unpromoted variable name, seen from current system (unique).
        prom / prom_name : relative, promoted variable name, seen from current system (non-unique).
        idx : global variable index among variables on all procs (I/O indices separate).
        my_idx : index among variables in this system, on this processor (I/O indices separate).
        io : indicates explicitly that input and output variables are combined in the same dict.

    Attributes
    ----------
    name : str
        Name of the system, must be different from siblings.
    pathname : str
        Global name of the system, including the path.
    comm : MPI.Comm or <FakeComm>
        MPI communicator object.
    options : OptionsDictionary
        options dictionary
    recording_options : OptionsDictionary
        Recording options dictionary
    _problem_meta : dict
        Problem level metadata.
    under_complex_step : bool
        When True, this system is undergoing complex step.
    under_approx : bool
        When True, this system is undergoing approximation.
    iter_count : int
        Counts the number of times this system has called _solve_nonlinear. This also
        corresponds to the number of times that the system's outputs are recorded if a recorder
        is present.
    iter_count_apply : int
        Counts the number of times the system has called _apply_nonlinear. For ExplicitComponent,
        calls to apply_nonlinear also call compute, so number of executions can be found by adding
        this and iter_count together. Recorders do no record calls to apply_nonlinear.
    iter_count_without_approx : int
        Counts the number of times the system has iterated but excludes any that occur during
        approximation of derivatives.
    cite : str
        Listing of relevant citations that should be referenced when
        publishing work that uses this class.
    _full_comm : MPI.Comm or None
        MPI communicator object used when System's comm is split for parallel FD.
    _solver_print_cache : list
        Allows solver iprints to be set to requested values after setup calls.
    _subsystems_allprocs : OrderedDict
        Dict mapping subsystem name to SysInfo(system, index) for children of this system.
    _subsystems_myproc : [<System>, ...]
        List of local subsystems that exist on this proc.
    _var_promotes : { 'any': [], 'input': [], 'output': [] }
        Dictionary of lists of variable names/wildcards specifying promotion
        (used to calculate promoted names)
    _var_prom2inds : dict
        Maps promoted name to src_indices in scope of system.
    _var_allprocs_prom2abs_list : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to list of all absolute names.
        For outputs, the list will have length one since promoted output names are unique.
    _var_abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names, on current proc.
    _var_allprocs_abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names, on all procs.
    _var_allprocs_abs2meta : dict
        Dictionary mapping absolute names to metadata dictionaries for allprocs variables.
        The keys are
        ('units', 'shape', 'size') for inputs and
        ('units', 'shape', 'size', 'ref', 'ref0', 'res_ref', 'distributed') for outputs.
    _var_abs2meta : dict
        Dictionary mapping absolute names to metadata dictionaries for myproc variables.
    _var_discrete : dict
        Dictionary of discrete var metadata and values local to this process.
    _var_allprocs_discrete : dict
        Dictionary of discrete var metadata and values for all processes.
    _discrete_inputs : dict-like or None
        Storage for discrete input values.
    _discrete_outputs : dict-like or None
        Storage for discrete output values.
    _var_allprocs_abs2idx : dict
        Dictionary mapping absolute names to their indices among this system's allprocs variables.
        Therefore, the indices range from 0 to the total number of this system's variables.
    _var_sizes : {<vecname>: {'input': ndarray, 'output': ndarray}, ...}
        Array of local sizes of this system's allprocs variables.
        The array has size nproc x num_var where nproc is the number of processors
        owned by this system and num_var is the number of allprocs variables.
    _owned_sizes : ndarray
        Array of local sizes for 'owned' or distributed vars only.
    _var_offsets : {<vecname>: {'input': dict of ndarray, 'output': dict of ndarray}, ...} or None
        Dict of distributed offsets, keyed by var name.  Offsets are stored in an array
        of size nproc x num_var where nproc is the number of processors
        in this System's communicator and num_var is the number of allprocs variables
        in the given system.  This is only defined in a Group that owns one or more interprocess
        connections or a top level Group or System that is used to compute total derivatives
        across multiple processes.
    _vars_to_gather : dict
        Contains names of non-distributed variables that are remote on at least one proc in the comm
    _dist_var_locality : dict
        Contains names of distrib vars mapped to the ranks in the comm where they are local.
    _conn_global_abs_in2out : {'abs_in': 'abs_out'}
        Dictionary containing all explicit & implicit connections owned by this system
        or any descendant system. The data is the same across all processors.
    _vectors : {'input': dict, 'output': dict, 'residual': dict}
        Dictionaries of vectors keyed by vec_name.
    _inputs : <Vector>
        The inputs vector; points to _vectors['input']['nonlinear'].
    _outputs : <Vector>
        The outputs vector; points to _vectors['output']['nonlinear'].
    _residuals : <Vector>
        The residuals vector; points to _vectors['residual']['nonlinear'].
    _nonlinear_solver : <NonlinearSolver>
        Nonlinear solver to be used for solve_nonlinear.
    _linear_solver : <LinearSolver>
        Linear solver to be used for solve_linear; not the Newton system.
    _approx_schemes : OrderedDict
        A mapping of approximation types to the associated ApproximationScheme.
    _jacobian : <Jacobian>
        <Jacobian> object to be used in apply_linear.
    _owns_approx_jac : bool
        If True, this system approximated its Jacobian
    _owns_approx_jac_meta : dict
        Stores approximation metadata (e.g., step_size) from calls to approx_totals
    _owns_approx_of : list or None
        Overrides aproximation outputs. This is set when calculating system derivatives, and serves
        as a way to communicate the driver's output quantities to the approximation objects so that
        we only take derivatives of variables that the driver needs.
    _owns_approx_of_idx : dict
        Index for override 'of' approximations if declared. When the user calls  `add_objective`
        or `add_constraint`, they may optionally specify an "indices" argument. This argument must
        also be communicated to the approximations when they are set up so that 1) the Jacobian is
        the correct size, and 2) we don't perform any extra unnecessary calculations.
    _owns_approx_wrt : list or None
        Overrides aproximation inputs. This is set when calculating system derivatives, and serves
        as a way to communicate the driver's input quantities to the approximation objects so that
        we only take derivatives with respect to variables that the driver needs.
    _owns_approx_wrt_idx : dict
        Index for override 'wrt' approximations if declared. When the user calls  `add_designvar`
        they may optionally specify an "indices" argument. This argument must also be communicated
        to the approximations when they are set up so that 1) the Jacobian is the correct size, and
        2) we don't perform any extra unnecessary calculations.
    _subjacs_info : dict of dict
        Sub-jacobian metadata for each (output, input) pair added using
        declare_partials. Members of each pair may be glob patterns.
    _design_vars : dict of dict
        dict of all driver design vars added to the system.
    _responses : dict of dict
        dict of all driver responses added to the system.
    _rec_mgr : <RecordingManager>
        object that manages all recorders added to this system.
    _static_subsystems_allprocs : OrderedDict
        Dict of SysInfo(subsys, index) that stores all subsystems added outside of setup.
    _static_design_vars : dict of dict
        Driver design variables added outside of setup.
    _static_responses : dict of dict
        Driver responses added outside of setup.
    supports_multivecs : bool
        If True, this system overrides compute_multi_jacvec_product (if an ExplicitComponent),
        or solve_multi_linear/apply_multi_linear (if an ImplicitComponent).
    matrix_free : Bool
        This is set to True if the component overrides the appropriate function with a user-defined
        matrix vector product with the Jacobian or any of its subsystems do.
    _relevant : dict
        Mapping of a VOI to a tuple containing dependent inputs, dependent outputs,
        and dependent systems.
    _vois : dict
        Either design vars or responses metadata, depending on the direction of
        derivatives.
    _mode : str
        Indicates derivative direction for the model, either 'fwd' or 'rev'.
    _scope_cache : dict
        Cache for variables in the scope of various mat-vec products.
    _has_guess : bool
        True if this system has or contains a system with a `guess_nonlinear` method defined.
    _has_output_scaling : bool
        True if this system has output scaling.
    _has_resid_scaling : bool
        True if this system has resid scaling.
    _has_input_scaling : bool
        True if this system has input scaling.
    _has_bounds: bool
        True if this system has upper or lower bounds on outputs.
    _owning_rank : dict
        Dict mapping var name to the lowest rank where that variable is local.
    _filtered_vars_to_record: Dict
        Dict of list of var names to record
    _vector_class : class
        Class to use for data vectors.  After setup will contain the value of either
        _problem_meta['distributed_vector_class'] or _problem_meta['local_vector_class'].
    _assembled_jac : AssembledJacobian or None
        If not None, this is the AssembledJacobian owned by this system's linear_solver.
    _num_par_fd : int
        If FD is active, and the value is > 1, turns on parallel FD and specifies the number of
        concurrent FD solves.
    _par_fd_id : int
        ID used to determine which columns in the jacobian will be computed when using parallel FD.
    _has_approx : bool
        If True, this system or its descendent has declared approximated partial or semi-total
        derivatives.
    _coloring_info : tuple
        Metadata that defines how to perform coloring of this System's approx jacobian. Not
        used if this System does no partial or semi-total coloring.
    _first_call_to_linearize : bool
        If True, this is the first call to _linearize.
    _is_local : bool
        If True, this system is local to this mpi process.
    """

    def __init__(self, num_par_fd=1, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        num_par_fd : int
            If FD is active, number of concurrent FD solves.
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the System options.
        """
        self.name = ''
        self.pathname = None
        self.comm = None
        self._is_local = False

        # System options
        self.options = OptionsDictionary(parent_name=type(self).__name__)

        self.options.declare('assembled_jac_type', values=['csc', 'dense'], default='csc',
                             desc='Linear solver(s) in this group, if using an assembled '
                                  'jacobian, will use this type.')

        # Case recording options
        self.recording_options = OptionsDictionary(parent_name=type(self).__name__)
        self.recording_options.declare('record_inputs', types=bool, default=True,
                                       desc='Set to True to record inputs at the system level')
        self.recording_options.declare('record_outputs', types=bool, default=True,
                                       desc='Set to True to record outputs at the system level')
        self.recording_options.declare('record_residuals', types=bool, default=True,
                                       desc='Set to True to record residuals at the system level')
        self.recording_options.declare('record_metadata', types=bool,
                                       desc='Deprecated. Recording of metadata will always be done',
                                       default=True,
                                       deprecation="The recording option, record_metadata, "
                                       "on System is "
                                       "deprecated. Recording of metadata will always be done")
        self.recording_options.declare('record_model_metadata', types=bool,
                                       desc='Deprecated. Recording of model metadata will always '
                                       'be done',
                                       deprecation="The recording option, record_model_metadata, "
                                       "on System is deprecated. Recording of model metadata will "
                                       "always be done",
                                       default=True)
        self.recording_options.declare('includes', types=list, default=['*'],
                                       desc='Patterns for variables to include in recording. \
                                       Uses fnmatch wildcards')
        self.recording_options.declare('excludes', types=list, default=[],
                                       desc='Patterns for vars to exclude in recording '
                                       '(processed post-includes). Uses fnmatch wildcards')
        self.recording_options.declare('options_excludes', types=list, default=[],
                                       desc='User-defined metadata to exclude in recording')

        self._problem_meta = None

        # Counting iterations.
        self.iter_count = 0
        self.iter_count_apply = 0
        self.iter_count_without_approx = 0

        self.cite = ""

        self._solver_print_cache = []

        self._subsystems_allprocs = {}
        self._subsystems_myproc = []
        self._vars_to_gather = {}
        self._dist_var_locality = {}

        self._var_promotes = {'input': [], 'output': [], 'any': []}

        self._var_allprocs_prom2abs_list = None
        self._var_prom2inds = {}
        self._var_abs2prom = {'input': {}, 'output': {}}
        self._var_allprocs_abs2prom = {'input': {}, 'output': {}}
        self._var_allprocs_abs2meta = {'input': {}, 'output': {}}
        self._var_abs2meta = {'input': {}, 'output': {}}
        self._var_discrete = {'input': {}, 'output': {}}
        self._var_allprocs_discrete = {'input': {}, 'output': {}}

        self._var_allprocs_abs2idx = {}

        self._var_sizes = None
        self._owned_sizes = None
        self._var_offsets = None

        self._full_comm = None

        self._vectors = {'input': {}, 'output': {}, 'residual': {}}

        self._inputs = None
        self._outputs = None
        self._residuals = None
        self._discrete_inputs = None
        self._discrete_outputs = None

        self._nonlinear_solver = None
        self._linear_solver = None

        self._jacobian = None
        self._approx_schemes = OrderedDict()
        self._subjacs_info = {}
        self.matrix_free = False

        self.under_approx = False
        self._owns_approx_jac = False
        self._owns_approx_jac_meta = {}
        self._owns_approx_wrt = None
        self._owns_approx_of = None
        self._owns_approx_wrt_idx = {}
        self._owns_approx_of_idx = {}

        self.under_complex_step = False

        self._design_vars = OrderedDict()
        self._responses = OrderedDict()
        self._rec_mgr = RecordingManager()

        self._conn_global_abs_in2out = {}

        self._static_subsystems_allprocs = {}
        self._static_design_vars = OrderedDict()
        self._static_responses = OrderedDict()

        self.supports_multivecs = False

        self._relevant = None
        self._mode = None

        self._scope_cache = {}

        self._num_par_fd = num_par_fd

        self._declare_options()
        self.initialize()

        self.options.update(kwargs)

        self._has_guess = False
        self._has_output_scaling = False
        self._has_resid_scaling = False
        self._has_input_scaling = False
        self._has_bounds = False

        self._vector_class = None
        self._has_approx = False

        self._assembled_jac = None

        self._par_fd_id = 0

        self._filtered_vars_to_record = {}
        self._owning_rank = None
        self._coloring_info = _DEFAULT_COLORING_META.copy()
        self._first_call_to_linearize = True   # will check in first call to _linearize

    @property
    def msginfo(self):
        """
        Our instance pathname, if available, or our class name.  For use in error messages.

        Returns
        -------
        str
            Either our instance pathname or class name.
        """
        if self.pathname is not None:
            if self.pathname == '':
                return f"<model> <class {type(self).__name__}>"
            return f"'{self.pathname}' <class {type(self).__name__}>"
        if self.name:
            return f"'{self.name}' <class {type(self).__name__}>"
        return f"<class {type(self).__name__}>"

    def _get_inst_id(self):
        return self.pathname

    def abs_name_iter(self, iotype, local=True, cont=True, discrete=False):
        """
        Iterate over absolute variable names for this System.

        By setting appropriate values for 'cont' and 'discrete', yielded variable
        names can be continuous only, discrete only, or both.

        Parameters
        ----------
        iotype : str
            Either 'input' or 'output'.
        local : bool
            If True, include only names of local variables. Default is True.
        cont : bool
            If True, include names of continuous variables.  Default is True.
        discrete : bool
            If True, include names of discrete variables.  Default is False.
        """
        if cont:
            if local:
                yield from self._var_abs2meta[iotype]
            else:
                yield from self._var_allprocs_abs2meta[iotype]

        if discrete:
            if local:
                prefix = self.pathname + '.' if self.pathname else ''
                for name in self._var_discrete[iotype]:
                    yield prefix + name
            else:
                yield from self._var_allprocs_discrete[iotype]

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.

        This is optionally implemented by subclasses of Component or Group
        that themselves are intended to be subclassed by the end user. The
        options of the intermediate class are declared here leaving the
        `initialize` method available for user-defined options.
        """
        pass

    def initialize(self):
        """
        Perform any one-time initialization run at instantiation.
        """
        pass

    def _configure(self):
        """
        Configure this system to assign children settings.
        """
        pass

    def _get_root_vectors(self):
        """
        Get the root vectors for the nonlinear and linear vectors for the model.

        Returns
        -------
        dict of dict of Vector
            Root vectors: first key is 'input', 'output', or 'residual'; second key is vec_name.
        """
        # save root vecs as an attribute so that we can reuse the nonlinear scaling vecs in the
        # linear root vec
        self._root_vecs = root_vectors = {'input': OrderedDict(),
                                          'output': OrderedDict(),
                                          'residual': OrderedDict()}

        relevant = self._relevant
        vec_names = self._rel_vec_name_list if self._use_derivatives else self._vec_names
        vectorized_vois = self._problem_meta['vectorized_vois']
        force_alloc_complex = self._problem_meta['force_alloc_complex']
        abs2idx = self._var_allprocs_abs2idx

        # Check for complex step to set vectors up appropriately.
        # If any subsystem needs complex step, then we need to allocate it everywhere.
        nl_alloc_complex = force_alloc_complex
        for sub in self.system_iter(include_self=True, recurse=True):
            nl_alloc_complex |= 'cs' in sub._approx_schemes
            if nl_alloc_complex:
                break

        # Linear vectors allocated complex only if subsolvers require derivatives.
        if nl_alloc_complex:
            from openmdao.error_checking.check_config import check_allocate_complex_ln
            ln_alloc_complex = check_allocate_complex_ln(self, force_alloc_complex)
        else:
            ln_alloc_complex = False

        if self._has_input_scaling or self._has_output_scaling or self._has_resid_scaling:
            self._scale_factors = self._compute_root_scale_factors()
        else:
            self._scale_factors = {}

        if self._vector_class is None:
            self._vector_class = self._local_vector_class

        for vec_name in vec_names:
            sizes = self._var_sizes[vec_name]['output']
            ncol = 1
            if vec_name == 'nonlinear':
                alloc_complex = nl_alloc_complex
            else:
                alloc_complex = ln_alloc_complex

                if vec_name != 'linear':
                    if vec_name in vectorized_vois:
                        voi = vectorized_vois[vec_name]
                        if 'size' in voi:
                            ncol = voi['size']
                        else:
                            owner = self._owning_rank[vec_name]
                            ncol = sizes[owner, abs2idx[vec_name][vec_name]]

            for key in ['input', 'output', 'residual']:
                root_vectors[key][vec_name] = self._vector_class(vec_name, key, self,
                                                                 alloc_complex=alloc_complex,
                                                                 ncol=ncol)
        return root_vectors

    def _get_approx_scheme(self, method):
        """
        Return the approximation scheme associated with the given method, creating one if needed.

        Parameters
        ----------
        method : str
            Name of the type of approxmation scheme.

        Returns
        -------
        ApproximationScheme
            The ApproximationScheme associated with the given method.
        """
        if method == 'exact':
            return None
        if method not in _supported_methods:
            msg = '{}: Method "{}" is not supported, method must be one of {}'
            raise ValueError(msg.format(self.msginfo, method,
                             [m for m in _supported_methods if m != 'exact']))
        if method not in self._approx_schemes:
            self._approx_schemes[method] = _supported_methods[method]()
        return self._approx_schemes[method]

    def get_source(self, name):
        """
        Return the source variable connected to the given named variable.

        The name can be a promoted name or an absolute name.
        If the given variable is an input, the absolute name of the connected source will
        be returned.  If the given variable itself is a source, its own absolute name will
        be returned.

        Parameters
        ----------
        name : str
            Absolute or promoted name of the variable.

        Returns
        -------
        str
            The absolute name of the source variable.
        """
        if self._problem_meta is None or 'prom2abs' not in self._problem_meta:
            raise RuntimeError(f"{self.msginfo}: get_source cannot be called for variable {name} "
                               "before Problem.setup is complete.")

        model = self._problem_meta['model_ref']()
        prom2abs = self._problem_meta['prom2abs']
        if name in prom2abs['input']:
            name = prom2abs['input'][name][0]
        elif name in prom2abs['output']:
            return prom2abs['output'][name][0]

        if name in model._conn_global_abs_in2out:
            return model._conn_global_abs_in2out[name]

        return name

    def _setup(self, comm, mode, prob_meta):
        """
        Perform setup for this system and its descendant systems.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm> or None
            The global communicator.
        mode : str
            Derivative direction, either 'fwd', or 'rev', or 'auto'
        prob_meta : dict
            Problem level metadata dictionary.
        """
        # save a ref to the problem level options.
        self._problem_meta = prob_meta

        # reset any coloring if a Coloring object was not set explicitly
        if self._coloring_info['dynamic'] or self._coloring_info['static'] is not None:
            self._coloring_info['coloring'] = None

        self.pathname = ''
        self.comm = comm
        self._relevant = None
        self._mode = mode

        # Besides setting up the processors, this method also builds the model hierarchy.
        self._setup_procs(self.pathname, comm, mode, self._problem_meta)

        prob_meta['config_info'] = _ConfigInfo()

        try:
            # Recurse model from the bottom to the top for configuring.
            self._configure()
        finally:
            prob_meta['config_info'] = None

        self._configure_check()

        self._setup_var_data()

        self._setup_vec_names(mode)

        # promoted names must be known to determine implicit connections so this must be
        # called after _setup_var_data, and _setup_var_data will have to be partially redone
        # after auto_ivcs have been added, but auto_ivcs can't be added until after we know all of
        # the connections.
        self._setup_global_connections()
        self._setup_dynamic_shapes()

        self._top_level_post_connections(mode)

        # Now that connections are setup, we need to convert relevant vector names into their
        # auto_ivc source where applicable.
        conns = self._conn_global_abs_in2out
        new_names = [conns[v] if v in conns else v for v in self._vec_names]
        self._problem_meta['vec_names'] = new_names
        self._problem_meta['lin_vec_names'] = new_names[1:]

        self._setup_relevance(mode)
        self._setup_var_sizes()

        self._top_level_post_sizes()

        # determine which connections are managed by which group, and check validity of connections
        self._setup_connections()

    def _top_level_post_connections(self, mode):
        # this runs after all connections are known
        pass

    def _top_level_post_sizes(self):
        # this runs after the variable sizes are known
        self._setup_global_shapes()

    def _configure_check(self):
        """
        Do any error checking on i/o and connections.
        """
        pass

    def _setup_dynamic_shapes(self):
        pass

    def _final_setup(self, comm):
        """
        Perform final setup for this system and its descendant systems.

        This part of setup is called automatically at the start of run_model or run_driver.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm> or None
            The global communicator.
        """
        if self._use_derivatives:
            # must call this before vector setup because it determines if we need to alloc commplex
            self._setup_partials()

        self._setup_vectors(self._get_root_vectors())

        # Transfers do not require recursion, but they have to be set up after the vector setup.
        self._setup_transfers()

        # Same situation with solvers, partials, and Jacobians.
        # If we're updating, we just need to re-run setup on these, but no recursion necessary.
        self._setup_solvers()
        self._setup_solver_print()
        if self._use_derivatives:
            self._setup_jacobians()

        self._setup_recording()

        self.set_initial_values()

    def use_fixed_coloring(self, coloring=_STD_COLORING_FNAME, recurse=True):
        """
        Use a precomputed coloring for this System.

        Parameters
        ----------
        coloring : str
            A coloring filename.  If no arg is passed, filename will be determined
            automatically.
        recurse : bool
            If True, set fixed coloring in all subsystems that declare a coloring. Ignored
            if a specific coloring is passed in.
        """
        if coloring_mod._force_dyn_coloring and coloring is _STD_COLORING_FNAME:
            self._coloring_info['dynamic'] = True
            return  # don't use static this time

        self._coloring_info['static'] = coloring
        self._coloring_info['dynamic'] = False

        if coloring is not _STD_COLORING_FNAME:
            if recurse:
                simple_warning("%s: recurse was passed to use_fixed_coloring but a specific "
                               "coloring was set, so recurse was ignored." % self.pathname)
            if isinstance(coloring, Coloring):
                approx = self._get_approx_scheme(coloring._meta['method'])
                # force regen of approx groups on next call to compute_approximations
                approx._reset()
            return

        if recurse:
            for s in self._subsystems_myproc:
                s.use_fixed_coloring(coloring, recurse)

    def declare_coloring(self,
                         wrt=_DEFAULT_COLORING_META['wrt_patterns'],
                         method=_DEFAULT_COLORING_META['method'],
                         form=None,
                         step=None,
                         per_instance=_DEFAULT_COLORING_META['per_instance'],
                         num_full_jacs=_DEFAULT_COLORING_META['num_full_jacs'],
                         tol=_DEFAULT_COLORING_META['tol'],
                         orders=_DEFAULT_COLORING_META['orders'],
                         perturb_size=_DEFAULT_COLORING_META['perturb_size'],
                         min_improve_pct=_DEFAULT_COLORING_META['min_improve_pct'],
                         show_summary=_DEFAULT_COLORING_META['show_summary'],
                         show_sparsity=_DEFAULT_COLORING_META['show_sparsity']):
        """
        Set options for deriv coloring of a set of wrt vars matching the given pattern(s).

        Parameters
        ----------
        wrt : str or list of str
            The name or names of the variables that derivatives are taken with respect to.
            This can contain input names, output names, or glob patterns.
        method : str
            Method used to compute derivative: "fd" for finite difference, "cs" for complex step.
        form : str
            Finite difference form, can be "forward", "central", or "backward". Leave
            undeclared to keep unchanged from previous or default value.
        step : float
            Step size for finite difference. Leave undeclared to keep unchanged from previous
            or default value.
        per_instance : bool
            If True, a separate coloring will be generated for each instance of a given class.
            Otherwise, only one coloring for a given class will be generated and all instances
            of that class will use it.
        num_full_jacs : int
            Number of times to repeat partial jacobian computation when computing sparsity.
        tol : float
            Tolerance used to determine if an array entry is nonzero during sparsity determination.
        orders : int
            Number of orders above and below the tolerance to check during the tolerance sweep.
        perturb_size : float
            Size of input/output perturbation during generation of sparsity.
        min_improve_pct : float
            If coloring does not improve (decrease) the number of solves more than the given
            percentage, coloring will not be used.
        show_summary : bool
            If True, display summary information after generating coloring.
        show_sparsity : bool
            If True, display sparsity with coloring info after generating coloring.
        """
        if method not in ('fd', 'cs'):
            raise RuntimeError("{}: method must be one of ['fd', 'cs'].".format(self.msginfo))

        self._has_approx = True
        approx = self._get_approx_scheme(method)

        # start with defaults
        options = _DEFAULT_COLORING_META.copy()
        options.update(approx.DEFAULT_OPTIONS)

        if self._coloring_info['static'] is None:
            options['dynamic'] = True
        else:
            options['dynamic'] = False
            options['static'] = self._coloring_info['static']

        options['wrt_patterns'] = [wrt] if isinstance(wrt, str) else wrt
        options['method'] = method
        options['per_instance'] = per_instance
        options['repeat'] = num_full_jacs
        options['tol'] = tol
        options['orders'] = orders
        options['perturb_size'] = perturb_size
        options['min_improve_pct'] = min_improve_pct
        options['show_summary'] = show_summary
        options['show_sparsity'] = show_sparsity
        options['coloring'] = self._coloring_info['coloring']
        if form is not None:
            options['form'] = form
        if step is not None:
            options['step'] = step

        self._coloring_info = options

    def _compute_approx_coloring(self, recurse=False, **overrides):
        """
        Compute a coloring of the approximated derivatives.

        This assumes that the current System is in a proper state for computing approximated
        derivatives.

        Parameters
        ----------
        recurse : bool
            If True, recurse from this system down the system hierarchy.  Whenever a group
            is encountered that has specified its coloring metadata, we don't recurse below
            that group unless that group has a subsystem that has a nonlinear solver that uses
            gradients.
        **overrides : dict
            Any args that will override either default coloring settings or coloring settings
            resulting from an earlier call to declare_coloring.

        Returns
        -------
        list of Coloring
            The computed colorings.
        """
        if recurse:
            colorings = []
            my_coloring = self._coloring_info['coloring']
            grad_systems = self._get_gradient_nl_solver_systems()
            for s in self.system_iter(include_self=True, recurse=True):
                if my_coloring is None or s in grad_systems:
                    if s._coloring_info['coloring'] is not None:
                        coloring = s._compute_approx_coloring(recurse=False, **overrides)[0]
                        colorings.append(coloring)
                        if coloring is not None:
                            coloring._meta['pathname'] = s.pathname
                            coloring._meta['class'] = type(s).__name__
            return [c for c in colorings if c is not None] or [None]

        # don't override metadata if it's already declared
        info = self._coloring_info

        info.update(**overrides)
        if isinstance(info['wrt_patterns'], str):
            info['wrt_patterns'] = [info['wrt_patterns']]

        if info['method'] is None and self._approx_schemes:
            info['method'] = list(self._approx_schemes)[0]

        if self._coloring_info['coloring'] is None:
            # check to see if any approx derivs have been declared
            for meta in self._subjacs_info.values():
                if 'method' in meta and meta['method']:
                    break
            else:  # no approx derivs found
                simple_warning("%s: No approx partials found but coloring was requested.  "
                               "Declaring ALL partials as approx (method='%s')" %
                               (self.msginfo, self._coloring_info['method']))
                try:
                    self.declare_partials('*', '*', method=self._coloring_info['method'])
                except AttributeError:  # this system must be a group
                    from openmdao.core.component import Component
                    for s in self.system_iter(recurse=True, typ=Component):
                        s.declare_partials('*', '*', method=self._coloring_info['method'])
                self._setup_partials()

        approx_scheme = self._get_approx_scheme(self._coloring_info['method'])

        if self._coloring_info['coloring'] is None and self._coloring_info['static'] is None:
            self._coloring_info['dynamic'] = True

        coloring_fname = self.get_approx_coloring_fname()

        # if we find a previously computed class coloring for our class, just use that
        # instead of regenerating a coloring.
        if not info['per_instance'] and coloring_fname in coloring_mod._CLASS_COLORINGS:
            info['coloring'] = coloring = coloring_mod._CLASS_COLORINGS[coloring_fname]
            if coloring is None:
                print("\nClass coloring for class '{}' wasn't good enough, "
                      "so skipping for '{}'".format(type(self).__name__, self.pathname))
                info['static'] = None
            else:
                print("\n{} using class coloring for class '{}'".format(self.pathname,
                                                                        type(self).__name__))
                info.update(coloring._meta)
                # force regen of approx groups during next compute_approximations
                approx_scheme._reset()
            return [coloring]

        from openmdao.core.group import Group
        is_total = isinstance(self, Group)

        # compute perturbations
        starting_inputs = self._inputs.asarray(copy=True)
        in_offsets = starting_inputs.copy()
        in_offsets[in_offsets == 0.0] = 1.0
        in_offsets *= info['perturb_size']

        starting_outputs = self._outputs.asarray(copy=True)
        out_offsets = starting_outputs.copy()
        out_offsets[out_offsets == 0.0] = 1.0
        out_offsets *= info['perturb_size']

        starting_resids = self._residuals.asarray(copy=True)

        # for groups, this does some setup of approximations
        self._setup_approx_coloring()

        save_first_call = self._first_call_to_linearize
        self._first_call_to_linearize = False
        sparsity_start_time = time.time()

        for i in range(info['num_full_jacs']):
            # randomize inputs (and outputs if implicit)
            if i > 0:
                self._inputs.set_val(starting_inputs +
                                     in_offsets * np.random.random(in_offsets.size))
                self._outputs.set_val(starting_outputs +
                                      out_offsets * np.random.random(out_offsets.size))
                if is_total:
                    self._solve_nonlinear()
                else:
                    self._apply_nonlinear()

                for scheme in self._approx_schemes.values():
                    scheme._reset()  # force a re-initialization of approx

            self.run_linearize()
            self._jacobian._save_sparsity(self)

        sparsity_time = time.time() - sparsity_start_time

        self._update_wrt_matches(info)

        ordered_of_info = list(self._jacobian_of_iter())
        ordered_wrt_info = list(self._jacobian_wrt_iter(info['wrt_matches']))
        sparsity, sp_info = self._jacobian._compute_sparsity(ordered_of_info, ordered_wrt_info,
                                                             num_full_jacs=info['num_full_jacs'],
                                                             tol=info['tol'],
                                                             orders=info['orders'])
        sp_info['sparsity_time'] = sparsity_time
        sp_info['pathname'] = self.pathname
        sp_info['class'] = type(self).__name__
        sp_info['type'] = 'semi-total' if self._subsystems_allprocs else 'partial'

        info = self._coloring_info

        self._jacobian._jac_summ = None  # reclaim the memory
        if self.pathname:
            ordered_of_info = self._jac_var_info_abs2prom(ordered_of_info)
            ordered_wrt_info = self._jac_var_info_abs2prom(ordered_wrt_info)

        coloring = _compute_coloring(sparsity, 'fwd')

        # if the improvement wasn't large enough, don't use coloring
        pct = coloring._solves_info()[-1]
        if info['min_improve_pct'] > pct:
            info['coloring'] = info['static'] = None
            simple_warning("%s: Coloring was deactivated.  Improvement of %.1f%% was less than min "
                           "allowed (%.1f%%)." % (self.msginfo, pct, info['min_improve_pct']))
            if not info['per_instance']:
                coloring_mod._CLASS_COLORINGS[coloring_fname] = None
            return [None]

        coloring._row_vars = [t[0] for t in ordered_of_info]
        coloring._col_vars = [t[0] for t in ordered_wrt_info]
        coloring._row_var_sizes = [t[2] - t[1] for t in ordered_of_info]
        coloring._col_var_sizes = [t[2] - t[1] for t in ordered_wrt_info]

        coloring._meta.update(info)  # save metadata we used to create the coloring
        del coloring._meta['coloring']
        coloring._meta.update(sp_info)

        info['coloring'] = coloring

        approx = self._get_approx_scheme(coloring._meta['method'])
        # force regen of approx groups during next compute_approximations
        approx._reset()

        if info['show_sparsity'] or info['show_summary']:
            print("\nApprox coloring for '%s' (class %s)" % (self.pathname, type(self).__name__))

        if info['show_sparsity']:
            coloring.display_txt()
        if info['show_summary']:
            coloring.summary()

        self._save_coloring(coloring)

        if not info['per_instance']:
            # save the class coloring for other instances of this class to use
            coloring_mod._CLASS_COLORINGS[coloring_fname] = coloring

        # restore original inputs/outputs
        self._inputs.set_val(starting_inputs)
        self._outputs.set_val(starting_outputs)
        self._residuals.set_val(starting_resids)

        self._first_call_to_linearize = save_first_call

        return [coloring]

    def _setup_approx_coloring(self):
        pass

    def _jacobian_of_iter(self):
        """
        Iterate over (name, offset, end, idxs) for each row var in the systems's jacobian.
        """
        abs2meta = self._var_allprocs_abs2meta
        offset = end = 0
        for of, meta in self._var_allprocs_abs2meta['output'].items():
            end += meta['size']
            yield of, offset, end, _full_slice
            offset = end

    def _jacobian_wrt_iter(self, wrt_matches=None):
        """
        Iterate over (name, offset, end, idxs) for each column var in the systems's jacobian.

        Parameters
        ----------
        wrt_matches : set or None
            Only include row vars that are contained in this set.  This will determine what
            the actual offsets are, i.e. the offsets will be into a reduced jacobian
            containing only the matching columns.
        """
        if wrt_matches is None:
            wrt_matches = ContainsAll()
        abs2meta = self._var_allprocs_abs2meta
        offset = end = 0
        for of, _offset, _end, sub_of_idx in self._jacobian_of_iter():
            if of in wrt_matches:
                end += (_end - _offset)
                yield of, offset, end, sub_of_idx
                offset = end

        for wrt, meta in self._var_allprocs_abs2meta['input'].items():
            if wrt in wrt_matches:
                end += meta['size']
                yield wrt, offset, end, _full_slice
                offset = end

    def get_approx_coloring_fname(self):
        """
        Return the full pathname to a coloring file.

        Parameters
        ----------
        system : System
            The System having its coloring saved or loaded.

        Returns
        -------
        str
            Full pathname of the coloring file.
        """
        directory = self._problem_meta['coloring_dir']
        if not self.pathname:
            # total coloring
            return os.path.join(directory, 'total_coloring.pkl')

        if self._coloring_info.get('per_instance'):
            # base the name on the instance pathname
            fname = 'coloring_' + self.pathname.replace('.', '_') + '.pkl'
        else:
            # base the name on the class name
            fname = 'coloring_' + '_'.join(
                [self.__class__.__module__.replace('.', '_'), self.__class__.__name__]) + '.pkl'

        return os.path.join(directory, fname)

    def _save_coloring(self, coloring):
        """
        Save the coloring to a file based on this system's class or pathname.

        Parameters
        ----------
        coloring : Coloring
            See Coloring class docstring.
        """
        # under MPI, only save on proc 0
        if ((self._full_comm is not None and self._full_comm.rank == 0) or
                (self._full_comm is None and self.comm.rank == 0)):
            coloring.save(self.get_approx_coloring_fname())

    def _get_static_coloring(self):
        """
        Get the Coloring for this system.

        If necessary, load the Coloring from a file.

        Returns
        -------
        Coloring or None
            Coloring object, possible loaded from a file, or None
        """
        info = self._coloring_info
        coloring = info['coloring']
        if coloring is not None:
            return coloring

        static = info['static']
        if static is _STD_COLORING_FNAME or isinstance(static, str):
            if static is _STD_COLORING_FNAME:
                fname = self.get_approx_coloring_fname()
            else:
                fname = static
            print("%s: loading coloring from file %s" % (self.msginfo, fname))
            info['coloring'] = coloring = Coloring.load(fname)
            if info['wrt_patterns'] != coloring._meta['wrt_patterns']:
                raise RuntimeError("%s: Loaded coloring has different wrt_patterns (%s) than "
                                   "declared ones (%s)." %
                                   (self.msginfo, coloring._meta['wrt_patterns'],
                                    info['wrt_patterns']))
            info.update(info['coloring']._meta)
            approx = self._get_approx_scheme(info['method'])
            # force regen of approx groups during next compute_approximations
            approx._reset()
        elif isinstance(static, coloring_mod.Coloring):
            info['coloring'] = coloring = static

        if coloring is not None:
            info['dynamic'] = False

        info['static'] = coloring

        return coloring

    def _get_coloring(self):
        """
        Get the Coloring for this system.

        If necessary, load the Coloring from a file or dynamically generate it.

        Returns
        -------
        Coloring or None
            Coloring object, possible loaded from a file or dynamically generated, or None
        """
        coloring = self._get_static_coloring()
        if coloring is None and self._coloring_info['dynamic']:
            self._coloring_info['coloring'] = coloring = self._compute_approx_coloring()[0]
            if coloring is not None:
                self._coloring_info.update(coloring._meta)

        return coloring

    def _setup_par_fd_procs(self, comm):
        """
        Split up the comm for use in parallel FD.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm>
            MPI communicator object.

        Returns
        -------
        MPI.Comm or <FakeComm>
            MPI communicator object.
        """
        num_par_fd = self._num_par_fd
        if comm.size < num_par_fd:
            raise ValueError("%s: num_par_fd must be <= communicator size (%d)" %
                             (self.msginfo, comm.size))

        self._full_comm = comm

        if num_par_fd > 1:
            sizes, offsets = evenly_distrib_idxs(num_par_fd, comm.size)

            # a 'color' is assigned to each subsystem, with
            # an entry for each processor it will be given
            # e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
            color = np.empty(comm.size, dtype=int)
            for i in range(num_par_fd):
                color[offsets[i]:offsets[i] + sizes[i]] = i

            self._par_fd_id = color[comm.rank]

            comm = self._full_comm.Split(self._par_fd_id)

        return comm

    def _setup_recording(self):
        if self._rec_mgr._recorders:
            myinputs = myoutputs = myresiduals = []

            options = self.recording_options
            incl = options['includes']
            excl = options['excludes']

            # includes and excludes for inputs are specified using _absolute_ names
            # vectors are keyed on absolute name, discretes on relative/promoted name
            if options['record_inputs']:
                myinputs = sorted([n for n in self._var_abs2prom['input']
                                   if check_path(n, incl, excl)])

            # includes and excludes for outputs are specified using _promoted_ names
            # vectors are keyed on absolute name, discretes on relative/promoted name
            if options['record_outputs']:
                myoutputs = sorted([n for n, prom in self._var_abs2prom['output'].items()
                                    if check_path(prom, incl, excl)])

                if self._var_discrete['output']:
                    # if we have discrete outputs then residual name set doesn't match output one
                    if options['record_residuals']:
                        contains = self._residuals._contains_abs
                        myresiduals = [n for n in myoutputs if contains(n)]
                elif options['record_residuals']:
                    myresiduals = myoutputs

            elif options['record_residuals']:
                abs2prom = self._var_abs2prom['output']
                myresiduals = [n for n in self._residuals._abs_iter()
                               if check_path(abs2prom[n], incl, excl)]

            self._filtered_vars_to_record = {
                'input': myinputs,
                'output': myoutputs,
                'residual': myresiduals
            }

            self._rec_mgr.startup(self)

        for subsys in self._subsystems_myproc:
            subsys._setup_recording()

    def _setup_procs(self, pathname, comm, mode, prob_meta):
        """
        Execute first phase of the setup process.

        Distribute processors, assign pathnames, and call setup on the component.
        Also reset internal data structures.

        Parameters
        ----------
        pathname : str
            Global name of the system, including the path.
        comm : MPI.Comm or <FakeComm>
            MPI communicator object.
        mode : string
            Derivatives calculation mode, 'fwd' for forward, and 'rev' for
            reverse (adjoint). Default is 'rev'.
        prob_meta : dict
            Problem level options.
        """
        self.pathname = pathname
        self._problem_meta = prob_meta
        self._first_call_to_linearize = True
        self._is_local = True
        self._vectors = {}
        self._full_comm = None

        self.options._parent_name = self.msginfo
        self.recording_options._parent_name = self.msginfo
        self._mode = mode
        self._design_vars = OrderedDict()
        self._responses = OrderedDict()
        self._design_vars.update(self._static_design_vars)
        self._responses.update(self._static_responses)

    def _setup_var_data(self):
        """
        Compute the list of abs var names, abs/prom name maps, and metadata dictionaries.
        """
        self._var_prom2inds = {}
        self._var_allprocs_prom2abs_list = {'input': OrderedDict(), 'output': OrderedDict()}
        self._var_abs2prom = {'input': {}, 'output': {}}
        self._var_allprocs_abs2prom = {'input': {}, 'output': {}}
        self._var_allprocs_abs2meta = {'input': {}, 'output': {}}
        self._var_abs2meta = {'input': {}, 'output': {}}
        self._var_allprocs_discrete = {'input': {}, 'output': {}}
        self._var_allprocs_abs2idx = {}
        self._owning_rank = defaultdict(int)
        self._var_sizes = {'nonlinear': {}}
        self._owned_sizes = None
        self._var_allprocs_relevant_names = defaultdict(lambda: {'input': [], 'output': []})
        self._var_relevant_names = defaultdict(lambda: {'input': [], 'output': []})

    def _setup_var_index_maps(self, vec_name):
        """
        Compute maps from abs var names to their index among allprocs variables in this system.

        Parameters
        ----------
        vec_name : str
            Name of vector.
        """
        abs2idx = self._var_allprocs_abs2idx[vec_name] = {}
        for io in ['input', 'output']:
            for i, abs_name in enumerate(self._var_allprocs_relevant_names[vec_name][io]):
                abs2idx[abs_name] = i

    def _setup_global_shapes(self):
        """
        Compute the global size and shape of all variables on this system.
        """
        loc_meta = self._var_abs2meta

        for io in ('input', 'output'):
            # now set global sizes and shapes into metadata for distributed variables
            sizes = self._var_sizes['nonlinear'][io]
            for idx, (abs_name, mymeta) in enumerate(self._var_allprocs_abs2meta[io].items()):
                local_shape = mymeta['shape']
                if mymeta['distributed']:
                    global_size = np.sum(sizes[:, idx])
                    mymeta['global_size'] = global_size

                    # assume that all but the first dimension of the shape of a
                    # distributed variable is the same on all procs
                    high_dims = local_shape[1:]
                    if high_dims:
                        high_size = np.prod(high_dims)
                        dim1 = global_size // high_size
                        if global_size % high_size != 0:
                            raise RuntimeError("%s: Global size of output '%s' (%s) does not agree "
                                               "with local shape %s" % (self.msginfo, abs_name,
                                                                        global_size, local_shape))
                        mymeta['global_shape'] = tuple([dim1] + list(high_dims))
                    else:
                        mymeta['global_shape'] = (global_size,)

                else:
                    # not distributed, just use local shape and size
                    mymeta['global_size'] = mymeta['size']
                    mymeta['global_shape'] = local_shape

                if abs_name in loc_meta[io]:
                    loc_meta[io][abs_name]['global_shape'] = mymeta['global_shape']
                    loc_meta[io][abs_name]['global_size'] = mymeta['global_size']

    def _setup_global_connections(self, conns=None):
        """
        Compute dict of all connections between this system's inputs and outputs.

        The connections come from 4 sources:
        1. Implicit connections owned by the current system
        2. Explicit connections declared by the current system
        3. Explicit connections declared by parent systems
        4. Implicit / explicit from subsystems

        Parameters
        ----------
        conns : dict
            Dictionary of connections passed down from parent group.
        """
        pass

    def _setup_vec_names(self, mode):
        """
        Compute the list of vec_names and the vois dict.

        This is only called on the top level System during initial setup.

        Parameters
        ----------
        mode : str
            Derivative direction, either 'fwd' or 'rev'.
        """
        vois = set()
        vectorized_vois = {}

        if self._use_derivatives:
            vec_names = ['nonlinear', 'linear']
            # Now that connections are setup, we need to convert relevant vector names into their
            # auto_ivc source where applicable.
            for system in self.system_iter(include_self=True, recurse=True):
                for name, meta in system._get_vec_names_from_vois(mode):
                    vois.add(system.get_source(name))
                    if meta['vectorize_derivs']:
                        vectorized_vois[name] = meta

            vec_names.extend(sorted(vois))
        else:
            vec_names = ['nonlinear']

        self._problem_meta['vec_names'] = vec_names
        self._problem_meta['lin_vec_names'] = vec_names[1:]
        self._problem_meta['vectorized_vois'] = vectorized_vois

    def _get_vec_names_from_vois(self, mode):
        """
        Compute the list of vec_names and the vois dict.

        This is only called on the top level System during initial setup.

        Parameters
        ----------
        mode : str
            Derivative direction, either 'fwd' or 'rev'.
        """
        vois = self._design_vars if mode == 'fwd' else self._responses

        pro2abs = self._var_allprocs_prom2abs_list
        try:
            for prom_name, data in vois.items():
                if data['parallel_deriv_color'] is not None or data['vectorize_derivs']:
                    if prom_name in pro2abs['output']:
                        yield pro2abs['output'][prom_name][0], data
                    else:
                        yield pro2abs['input'][prom_name][0], data

        except KeyError as err:
            typ = 'design variable' if mode == 'fwd' else 'response'
            raise RuntimeError(f"{self.msginfo}: Output not found for {typ} {str(err)}.")

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
        if self._use_derivatives:
            desvars = self.get_design_vars(recurse=True, get_sizes=False, use_prom_ivc=False)
            responses = self.get_responses(recurse=True, get_sizes=False, use_prom_ivc=False)
            return self.get_relevant_vars(desvars, responses, mode)
        else:
            relevant = defaultdict(dict)
            relevant['nonlinear'] = {'@all': ({'input': ContainsAll(), 'output': ContainsAll()},
                                              ContainsAll())}
            return relevant

    def _setup_driver_units(self):
        """
        Compute unit conversions for driver variables.
        """
        abs2meta = self._var_abs2meta['output']
        pro2abs = self._var_allprocs_prom2abs_list['output']
        pro2abs_in = self._var_allprocs_prom2abs_list['input']

        dv = self._design_vars
        for name, meta in dv.items():

            units = meta['units']
            dv[name]['total_adder'] = dv[name]['adder']
            dv[name]['total_scaler'] = dv[name]['scaler']

            if units is not None:
                # If derivatives are not being calculated, then you reach here before ivc_source
                # is placed in the meta.
                try:
                    units_src = meta['ivc_source']
                except KeyError:
                    units_src = self.get_source(name)

                var_units = abs2meta[units_src]['units']

                if var_units == units:
                    continue

                if var_units is None:
                    msg = "{}: Target for design variable {} has no units, but '{}' units " + \
                          "were specified."
                    raise RuntimeError(msg.format(self.msginfo, name, units))

                if not is_compatible(var_units, units):
                    msg = "{}: Target for design variable {} has '{}' units, but '{}' units " + \
                          "were specified."
                    raise RuntimeError(msg.format(self.msginfo, name, var_units, units))

                factor, offset = unit_conversion(var_units, units)
                base_adder, base_scaler = determine_adder_scaler(None, None,
                                                                 dv[name]['adder'],
                                                                 dv[name]['scaler'])

                dv[name]['total_adder'] = offset + base_adder / factor
                dv[name]['total_scaler'] = base_scaler * factor

        resp = self._responses
        type_dict = {'con': 'constraint', 'obj': 'objective'}
        for name, meta in resp.items():

            units = meta['units']
            resp[name]['total_scaler'] = resp[name]['scaler']
            resp[name]['total_adder'] = resp[name]['adder']

            if units is not None:
                # If derivatives are not being calculated, then you reach here before ivc_source
                # is placed in the meta.
                try:
                    units_src = meta['ivc_source']
                except KeyError:
                    units_src = self.get_source(name)

                var_units = abs2meta[units_src]['units']

                if var_units == units:
                    continue

                if var_units is None:
                    msg = "{}: Target for {} {} has no units, but '{}' units " + \
                          "were specified."
                    raise RuntimeError(msg.format(self.msginfo, type_dict[meta['type']],
                                                  name, units))

                if not is_compatible(var_units, units):
                    msg = "{}: Target for {} {} has '{}' units, but '{}' units " + \
                          "were specified."
                    raise RuntimeError(msg.format(self.msginfo, type_dict[meta['type']],
                                                  name, var_units, units))

                factor, offset = unit_conversion(var_units, units)
                base_adder, base_scaler = determine_adder_scaler(None, None,
                                                                 resp[name]['adder'],
                                                                 resp[name]['scaler'])

                resp[name]['total_scaler'] = base_scaler * factor
                resp[name]['total_adder'] = offset + base_adder / factor

        for s in self._subsystems_myproc:
            s._setup_driver_units()

    def _setup_relevance(self, mode, relevant=None):
        """
        Set up the relevance dictionary.

        Parameters
        ----------
        mode : str
            Derivative direction, either 'fwd' or 'rev'.
        relevant : dict or None
            Dictionary mapping VOI name to all variables necessary for computing
            derivatives between the VOI and all other VOIs.
        """
        if relevant is None:  # should only occur at top level on full setup
            self._relevant = relevant = self._init_relevance(mode)
        else:
            self._relevant = relevant

        self._rel_vec_name_list = ['nonlinear', 'linear']
        for vec_name in self._vec_names[2:]:
            rel, relsys = relevant[vec_name]['@all']
            if self.pathname in relsys:
                self._rel_vec_name_list.append(vec_name)
            for io in ('input', 'output'):
                relio = rel[io]
                self._var_allprocs_relevant_names[vec_name][io].extend(
                    v for v in self._var_allprocs_abs2meta[io] if v in relio)
                self._var_relevant_names[vec_name][io].extend(
                    v for v in self._var_abs2meta[io] if v in relio)

        self._rel_vec_names = frozenset(self._rel_vec_name_list)
        self._lin_rel_vec_name_list = self._rel_vec_name_list[1:]

        for s in self._subsystems_myproc:
            s._setup_relevance(mode, relevant)

    def _setup_connections(self):
        """
        Compute dict of all connections owned by this system.
        """
        pass

    def _setup_vectors(self, root_vectors, alloc_complex=False):
        """
        Compute all vectors for all vec names and assign excluded variables lists.

        Parameters
        ----------
        root_vectors : dict of dict of Vector
            Root vectors: first key is 'input', 'output', or 'residual'; second key is vec_name.
        alloc_complex : bool
            Whether to allocate any imaginary storage to perform complex step. Default is False.
        """
        self._vectors = vectors = {'input': OrderedDict(),
                                   'output': OrderedDict(),
                                   'residual': OrderedDict()}

        # Allocate complex if root vector was allocated complex.
        alloc_complex = root_vectors['output']['nonlinear']._alloc_complex

        # This happens if you reconfigure and switch to 'cs' without forcing the vectors to be
        # initially allocated as complex.
        if not alloc_complex and 'cs' in self._approx_schemes:
            raise RuntimeError("{}: In order to activate complex step during reconfiguration, "
                               "you need to set 'force_alloc_complex' to True during setup. e.g. "
                               "'problem.setup(force_alloc_complex=True)'".format(self.msginfo))

        if self._vector_class is None:
            self._vector_class = self._local_vector_class

        vector_class = self._vector_class

        vec_names = self._rel_vec_name_list if self._use_derivatives else self._vec_names

        for vec_name in vec_names:

            # Only allocate complex in the vectors we need.
            vec_alloc_complex = root_vectors['output'][vec_name]._alloc_complex

            for kind in ['input', 'output', 'residual']:
                rootvec = root_vectors[kind][vec_name]
                vectors[kind][vec_name] = vector_class(
                    vec_name, kind, self, rootvec,
                    alloc_complex=vec_alloc_complex, ncol=rootvec._ncol)

        self._inputs = vectors['input']['nonlinear']
        self._outputs = vectors['output']['nonlinear']
        self._residuals = vectors['residual']['nonlinear']

        for subsys in self._subsystems_myproc:
            subsys._scale_factors = self._scale_factors
            subsys._setup_vectors(root_vectors)

    def _compute_root_scale_factors(self):
        """
        Compute scale factors for all variables.

        Returns
        -------
        dict
            Mapping of each absoute var name to its corresponding scaling factor tuple.
        """
        # make this a defaultdict to handle the case of access using unconnected inputs
        scale_factors = defaultdict(lambda: {
            ('input', 'phys'): (0.0, 1.0),
            ('input', 'norm'): (0.0, 1.0)
        })

        for abs_name, meta in self._var_allprocs_abs2meta['output'].items():
            ref0 = meta['ref0']
            res_ref = meta['res_ref']
            a0 = ref0
            a1 = meta['ref'] - ref0
            scale_factors[abs_name] = {
                ('output', 'phys'): (a0, a1),
                ('output', 'norm'): (-a0 / a1, 1.0 / a1),
                ('residual', 'phys'): (0.0, res_ref),
                ('residual', 'norm'): (0.0, 1.0 / res_ref),
            }
        return scale_factors

    def _setup_transfers(self):
        """
        Compute all transfers that are owned by this system.
        """
        pass

    def _setup_solvers(self):
        """
        Perform setup in all solvers.
        """
        # remove old solver error files if they exist
        if self.pathname == '':
            rank = MPI.COMM_WORLD.rank if MPI is not None else 0
            if rank == 0:
                for f in os.listdir('.'):
                    if fnmatchcase(f, 'solver_errors.*.out'):
                        os.remove(f)

        if self._nonlinear_solver is not None:
            self._nonlinear_solver._setup_solvers(self, 0)
        if self._linear_solver is not None:
            self._linear_solver._setup_solvers(self, 0)

        for subsys in self._subsystems_myproc:
            subsys._setup_solvers()

    def _setup_jacobians(self, recurse=True):
        """
        Set and populate jacobians down through the system tree.

        Parameters
        ----------
        recurse : bool
            If True, setup jacobians in all descendants.
        """
        asm_jac_solvers = set()
        if self._linear_solver is not None:
            asm_jac_solvers.update(self._linear_solver._assembled_jac_solver_iter())

        nl_asm_jac_solvers = set()
        if self.nonlinear_solver is not None:
            nl_asm_jac_solvers.update(self.nonlinear_solver._assembled_jac_solver_iter())

        asm_jac = None
        if asm_jac_solvers:
            asm_jac = _asm_jac_types[self.options['assembled_jac_type']](system=self)
            self._assembled_jac = asm_jac
            for s in asm_jac_solvers:
                s._assembled_jac = asm_jac

        if nl_asm_jac_solvers:
            if asm_jac is None:
                asm_jac = _asm_jac_types[self.options['assembled_jac_type']](system=self)
            for s in nl_asm_jac_solvers:
                s._assembled_jac = asm_jac

        if self._has_approx:
            self._set_approx_partials_meta()

        # At present, we don't support a AssembledJacobian in a group
        # if any subcomponents are matrix-free.
        if asm_jac is not None:
            if self.matrix_free:
                raise RuntimeError("%s: AssembledJacobian not supported for matrix-free "
                                   "subcomponent." % self.msginfo)

        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_jacobians()

    def set_initial_values(self):
        """
        Set all input and output variables to their declared initial values.
        """
        for abs_name, meta in self._var_abs2meta['input'].items():
            self._inputs.set_var(abs_name, meta['value'])

        for abs_name, meta in self._var_abs2meta['output'].items():
            self._outputs.set_var(abs_name, meta['value'])

    def _get_promotion_maps(self):
        """
        Define variable maps based on promotes lists.

        Returns
        -------
        dict of {'input': {str:(str, info), ...}, 'output': {str:(str, info), ...}}
            dictionary mapping input/output variable names
            to (promoted name, promotion_info) tuple.
        """
        prom_names = self._var_allprocs_prom2abs_list
        gname = self.name + '.' if self.name else ''

        def split_list(lst):
            """
            Yield match type, name/pattern/tuple info, and src_indices info.

            Parameters
            ----------
            lst : list
                List of names, patterns and/or tuples specifying promotes.

            Yields
            ------
            Enum
                match type
            str
                name or pattern string
            (str, _PromotesInfo)
                name/rename/pattern, promotion info (src_indices, etc.)
            """
            for entry in lst:
                key, pinfo = entry
                if isinstance(key, str):
                    # note, conditional here is faster than using precompiled regex
                    if '*' in key or '?' in key or '[' in key:
                        yield _MatchType.PATTERN, key, entry
                    else:
                        yield _MatchType.NAME, key, entry
                elif isinstance(key, tuple) and len(key) == 2:
                    yield _MatchType.RENAME, key[0], (key[1], pinfo)
                else:
                    raise TypeError(f"when adding subsystem '{self.pathname}', entry '{key}'"
                                    " is not a string or tuple of size 2.")

        def _dup(io, matches, match_type, name, tup):
            """
            Report error or warning when attempting to promote a variable twice.

            Parameters
            ----------
            matches : dict {'input': ..., 'output': ...}
                Dict of promoted names and associated info.
            match_type : IntEnum
                Indicates whether match is an explicit name, rename, or pattern match.
            name : str
                Name of promoted variable that is specified multiple times.
            tup : tuple (?, _PromotesInfo)
                First entry can be name, rename, or pattern depending on the match type.

            Returns
            -------
            bool
                If True, ignore the new match, else replace the old with the new.
            """
            old_name, old_key, old_info, old_match_type = matches[io][name]
            _, info = tup
            if old_match_type == _MatchType.RENAME:
                old_key = (old_name, old_key)
            else:
                old_using = f"'{old_key}'"
            if match_type == _MatchType.RENAME:
                new_using = (name, tup[0])
            else:
                new_using = f"'{tup[0]}'"

            mismatch = info.compare(old_info) if info is not None else ()
            if mismatch:
                raise RuntimeError(f"{self.msginfo}: {io} variable '{name}', promoted using "
                                   f"{new_using}, was already promoted using {old_using} with "
                                   f"different values for {mismatch}.")

            if old_match_type != _MatchType.PATTERN:
                if old_key != tup[0]:
                    raise RuntimeError(f"{self.msginfo}: Can't alias promoted {io} '{name}' to "
                                       f"'{tup[0]}' because '{name}' has already been promoted as "
                                       f"'{old_key}'.")

            if old_key != '*':
                simple_warning(f"{self.msginfo}: {io} variable '{name}', promoted using "
                               f"{new_using}, was already promoted using {old_using}.")

            return match_type == _MatchType.PATTERN

        def resolve(to_match, io_types, matches, proms):
            """
            Determine the mapping of promoted names to the parent scope for a promotion type.

            This is called once for promotes or separately for promotes_inputs and promotes_outputs.
            """
            if not to_match:
                return

            # always add '*' and so we won't report if it matches nothing (in the case where the
            # system has no variables of that io type)
            found = set(('*',))

            for match_type, key, tup in split_list(to_match):
                s, pinfo = tup
                if match_type == _MatchType.PATTERN:
                    for io in io_types:
                        if io == 'output':
                            pinfo = None
                        if key == '*' and not matches[io]:  # special case. add everything
                            matches[io] = pmap = {n: (n, key, pinfo, match_type) for n in proms[io]}
                        else:
                            pmap = matches[io]
                            nmatch = len(pmap)
                            for n in proms[io]:
                                if fnmatchcase(n, key):
                                    if not (n in pmap and _dup(io, matches, match_type, n, tup)):
                                        pmap[n] = (n, key, pinfo, match_type)
                            if len(pmap) > nmatch:
                                found.add(key)
                else:  # NAME or RENAME
                    for io in io_types:
                        if io == 'output':
                            pinfo = None
                        pmap = matches[io]
                        if key in proms[io]:
                            if key in pmap:
                                _dup(io, matches, match_type, key, tup)
                            pmap[key] = (s, key, pinfo, match_type)
                            if match_type == _MatchType.NAME:
                                found.add(key)
                            else:
                                found.add((key, s))

            not_found = set(n for n, _ in to_match) - found
            if not_found:
                if (not self._var_abs2meta['input'] and not self._var_abs2meta['output'] and
                        isinstance(self, openmdao.core.group.Group)):
                    empty_group_msg = ' Group contains no variables.'
                else:
                    empty_group_msg = ''
                if len(io_types) == 2:
                    call = 'promotes'
                else:
                    call = 'promotes_%ss' % io_types[0]

                not_found = sorted(not_found, key=lambda x: x if isinstance(x, str) else x[0])
                raise RuntimeError(f"{self.msginfo}: '{call}' failed to find any matches for the "
                                   f"following names or patterns: {not_found}.{empty_group_msg}")

        maps = {'input': {}, 'output': {}}

        if self._var_promotes['input'] or self._var_promotes['output']:
            if self._var_promotes['any']:
                raise RuntimeError("%s: 'promotes' cannot be used at the same time as "
                                   "'promotes_inputs' or 'promotes_outputs'." % self.msginfo)
            resolve(self._var_promotes['input'], ('input',), maps, prom_names)
            resolve(self._var_promotes['output'], ('output',), maps, prom_names)
        else:
            resolve(self._var_promotes['any'], ('input', 'output'), maps, prom_names)

        return maps

    def _get_scope(self):
        """
        Find the input and output variables that are needed for a particular matvec product.

        Returns
        -------
        (set, set)
            Sets of output and input variables.
        """
        try:
            return self._scope_cache[None]
        except KeyError:
            self._scope_cache[None] = (frozenset(self._var_abs2meta['output']), _empty_frozen_set)
            return self._scope_cache[None]

    def _get_potential_partials_lists(self, include_wrt_outputs=True):
        """
        Return full lists of possible 'of' and 'wrt' variables.

        Filters out any discrete variables.

        Parameters
        ----------
        include_wrt_outputs : bool
            If True, include outputs in the wrt list.

        Returns
        -------
        list
            List of 'of' variable names.
        list
            List of 'wrt' variable names.
        """
        of_list = list(self._var_allprocs_prom2abs_list['output'])
        wrt_list = list(self._var_allprocs_prom2abs_list['input'])

        # filter out any discrete inputs or outputs
        if self._discrete_outputs:
            of_list = [n for n in of_list if n not in self._discrete_outputs]
        if self._discrete_inputs:
            wrt_list = [n for n in wrt_list if n not in self._discrete_inputs]

        if include_wrt_outputs:
            wrt_list = of_list + wrt_list

        return of_list, wrt_list

    @contextmanager
    def _unscaled_context(self, outputs=(), residuals=()):
        """
        Context manager for units and scaling for vectors.

        Temporarily puts vectors in a physical and unscaled state, because
        internally, vectors are nominally in a dimensionless and scaled state.

        Parameters
        ----------
        outputs : list of output <Vector> objects
            List of output vectors to apply the unit and scaling conversions.
        residuals : list of residual <Vector> objects
            List of residual vectors to apply the unit and scaling conversions.
        """
        if self._has_output_scaling:
            for vec in outputs:
                vec.scale('phys')
        if self._has_resid_scaling:
            for vec in residuals:
                vec.scale('phys')

        yield

        if self._has_output_scaling:
            for vec in outputs:
                vec.scale('norm')

        if self._has_resid_scaling:
            for vec in residuals:
                vec.scale('norm')

    @contextmanager
    def _scaled_context_all(self):
        """
        Context manager that temporarily puts all vectors in a scaled state.
        """
        if self._has_output_scaling:
            for vec in self._vectors['output'].values():
                vec.scale('norm')
        if self._has_resid_scaling:
            for vec in self._vectors['residual'].values():
                vec.scale('norm')

        yield

        if self._has_output_scaling:
            for vec in self._vectors['output'].values():
                vec.scale('phys')
        if self._has_resid_scaling:
            for vec in self._vectors['residual'].values():
                vec.scale('phys')

    @contextmanager
    def _matvec_context(self, vec_name, scope_out, scope_in, mode, clear=True):
        """
        Context manager for vectors.

        For the given vec_name, return vectors that use a set of
        internal variables that are relevant to the current matrix-vector
        product.  This is called only from _apply_linear.

        Parameters
        ----------
        vec_name : str
            Name of the vector to use.
        scope_out : frozenset or None
            Set of absolute output names in the scope of this mat-vec product.
            If None, all are in the scope.
        scope_in : frozenset or None
            Set of absolute input names in the scope of this mat-vec product.
            If None, all are in the scope.
        mode : str
            Key for specifying derivative direction. Values are 'fwd'
            or 'rev'.
        clear : bool(True)
            If True, zero out residuals (in fwd mode) or inputs and outputs
            (in rev mode).

        Yields
        ------
        (d_inputs, d_outputs, d_residuals) : tuple of Vectors
            Yields the three Vectors configured internally to deal only
            with variables relevant to the current matrix vector product.

        """
        d_inputs = self._vectors['input'][vec_name]
        d_outputs = self._vectors['output'][vec_name]
        d_residuals = self._vectors['residual'][vec_name]

        if clear:
            if mode == 'fwd':
                d_residuals.set_val(0.0)
            else:  # rev
                d_inputs.set_val(0.0)
                d_outputs.set_val(0.0)

        if scope_out is None and scope_in is None:
            yield d_inputs, d_outputs, d_residuals
        else:
            old_ins = d_inputs._names
            old_outs = d_outputs._names

            if scope_out is not None:
                d_outputs._names = scope_out.intersection(d_outputs._abs_iter())
            if scope_in is not None:
                d_inputs._names = scope_in.intersection(d_inputs._abs_iter())

            yield d_inputs, d_outputs, d_residuals

            # reset _names so users will see full vector contents
            d_inputs._names = old_ins
            d_outputs._names = old_outs

    @contextmanager
    def _call_user_function(self, fname, protect_inputs=True,
                            protect_outputs=False, protect_residuals=False):
        """
        Context manager that wraps a call to a user defined function.

        Protect any vectors that should not be modified to help prevent user error
        and add information about the system to any errors that don't have it already.

        Parameters
        ----------
        fname : str
            Name of the user defined function.
        protect_inputs : bool
            If True, then set the inputs vector to be read only
        protect_outputs : bool
            If True, then set the outputs vector to be read only
        protect_residuals : bool
            If True, then set the residuals vector to be read only
        """
        self._inputs.read_only = protect_inputs
        self._outputs.read_only = protect_outputs
        self._residuals.read_only = protect_residuals

        try:
            yield
        except Exception:
            err_type, err, trace = sys.exc_info()
            if str(err).startswith(self.msginfo):
                raise err
            else:
                raise err_type(f"{self.msginfo}: Error calling {fname}(), {err}")
        finally:
            self._inputs.read_only = False
            self._outputs.read_only = False
            self._residuals.read_only = False

    def get_nonlinear_vectors(self):
        """
        Return the inputs, outputs, and residuals vectors.

        Returns
        -------
        (inputs, outputs, residuals) : tuple of <Vector> instances
            Yields the inputs, outputs, and residuals nonlinear vectors.
        """
        if self._inputs is None:
            raise RuntimeError("{}: Cannot get vectors because setup has not yet been "
                               "called.".format(self.msginfo))

        return self._inputs, self._outputs, self._residuals

    def get_linear_vectors(self, vec_name='linear'):
        """
        Return the linear inputs, outputs, and residuals vectors.

        Parameters
        ----------
        vec_name : str
            Name of the linear right-hand-side vector. The default is 'linear'.

        Returns
        -------
        (inputs, outputs, residuals) : tuple of <Vector> instances
            Yields the inputs, outputs, and residuals linear vectors for vec_name.
        """
        if self._inputs is None:
            raise RuntimeError("{}: Cannot get vectors because setup has not yet been "
                               "called.".format(self.msginfo))

        if vec_name not in self._vectors['input']:
            raise ValueError("%s: There is no linear vector named %s" % (self.msginfo, vec_name))

        return (self._vectors['input'][vec_name],
                self._vectors['output'][vec_name],
                self._vectors['residual'][vec_name])

    def _get_var_offsets(self):
        """
        Compute global offsets for variables.

        Returns
        -------
        dict
            Arrays of global offsets keyed by vec_name and deriv direction.
        """
        if self._var_offsets is None:
            offsets = self._var_offsets = {}
            vec_names = self._lin_rel_vec_name_list if self._use_derivatives else self._vec_names

            for vec_name in vec_names:
                offsets[vec_name] = off_vn = {}
                for type_ in ['input', 'output']:
                    vsizes = self._var_sizes[vec_name][type_]
                    if vsizes.size > 0:
                        csum = np.empty(vsizes.size, dtype=int)
                        csum[0] = 0
                        csum[1:] = np.cumsum(vsizes)[:-1]
                        off_vn[type_] = csum.reshape(vsizes.shape)
                    else:
                        off_vn[type_] = np.zeros(0, dtype=int).reshape((1, 0))

            if self._use_derivatives:
                offsets['nonlinear'] = offsets['linear']

        return self._var_offsets

    @property
    def nonlinear_solver(self):
        """
        Get the nonlinear solver for this system.
        """
        return self._nonlinear_solver

    @nonlinear_solver.setter
    def nonlinear_solver(self, solver):
        """
        Set this system's nonlinear solver.
        """
        self._nonlinear_solver = solver

    @property
    def linear_solver(self):
        """
        Get the linear solver for this system.
        """
        return self._linear_solver

    @linear_solver.setter
    def linear_solver(self, solver):
        """
        Set this system's linear solver.
        """
        self._linear_solver = solver

    @property
    def _force_alloc_complex(self):
        return self._problem_meta['force_alloc_complex']

    @property
    def _use_derivatives(self):
        return self._problem_meta['use_derivatives']

    @property
    def _local_vector_class(self):
        return self._problem_meta['local_vector_class']

    @property
    def _distributed_vector_class(self):
        return self._problem_meta['distributed_vector_class']

    @property
    def _vec_names(self):
        return self._problem_meta['vec_names']

    @property
    def _lin_vec_names(self):
        return self._problem_meta['lin_vec_names']

    @property
    def _recording_iter(self):
        return self._problem_meta['recording_iter']

    @property
    def _static_mode(self):
        """
        Return True if we are outside of setup.

        In this case, add_input, add_output, and add_subsystem all add to the
        '_static' versions of the respective data structures.
        These data structures are never reset during setup.

        Returns
        -------
        True if outside of setup.
        """
        return self._problem_meta is None or self._problem_meta['static_mode']

    def _set_solver_print(self, level=2, depth=1e99, type_='all'):
        """
        Apply the given print settings to the internal solvers, recursively.

        Parameters
        ----------
        level : int
            iprint level. Set to 2 to print residuals each iteration; set to 1
            to print just the iteration totals; set to 0 to disable all printing
            except for failures, and set to -1 to disable all printing including failures.
        depth : int
            How deep to recurse. For example, you can set this to 0 if you only want
            to print the top level linear and nonlinear solver messages. Default
            prints everything.
        type_ : str
            Type of solver to set: 'LN' for linear, 'NL' for nonlinear, or 'all' for all.
        """
        if self._linear_solver is not None and type_ != 'NL':
            self._linear_solver._set_solver_print(level=level, type_=type_)
        if self.nonlinear_solver is not None and type_ != 'LN':
            self.nonlinear_solver._set_solver_print(level=level, type_=type_)

        for subsys, _ in self._subsystems_allprocs.values():

            current_depth = subsys.pathname.count('.')
            if current_depth >= depth:
                continue

            subsys._set_solver_print(level=level, depth=depth - current_depth, type_=type_)

            if subsys._linear_solver is not None and type_ != 'NL':
                subsys._linear_solver._set_solver_print(level=level, type_=type_)
            if subsys.nonlinear_solver is not None and type_ != 'LN':
                subsys.nonlinear_solver._set_solver_print(level=level, type_=type_)

    def _setup_solver_print(self, recurse=True):
        """
        Apply the cached solver print settings during setup.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        for level, depth, type_ in self._solver_print_cache:
            self._set_solver_print(level, depth, type_)

        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_solver_print(recurse=recurse)

    def set_solver_print(self, level=2, depth=1e99, type_='all'):
        """
        Control printing for solvers and subsolvers in the model.

        Parameters
        ----------
        level : int
            iprint level. Set to 2 to print residuals each iteration; set to 1
            to print just the iteration totals; set to 0 to disable all printing
            except for failures, and set to -1 to disable all printing including failures.
        depth : int
            How deep to recurse. For example, you can set this to 0 if you only want
            to print the top level linear and nonlinear solver messages. Default
            prints everything.
        type_ : str
            Type of solver to set: 'LN' for linear, 'NL' for nonlinear, or 'all' for all.
        """
        if (level, depth, type_) not in self._solver_print_cache:
            self._solver_print_cache.append((level, depth, type_))

    def _set_approx_partials_meta(self):
        # this will load a static coloring (if any) and will populate wrt_matches if
        # there is any coloring (static or dynamic).
        self._get_static_wrt_matches()

    def _get_static_wrt_matches(self):
        """
        Return wrt_matches for static coloring if there is one.

        Returns
        -------
        list of str or ()
            List of wrt_matches for a static coloring or () if there isn't one.
        """
        if (self._coloring_info['coloring'] is not None and
                self._coloring_info['wrt_matches'] is None):
            self._update_wrt_matches(self._coloring_info)

        # if coloring has been specified, we don't want to have multiple
        # approximations for the same subjac, so don't register any new
        # approximations when the wrt matches those used in the coloring.
        if self._get_static_coloring() is not None:  # static coloring has been specified
            return self._coloring_info['wrt_matches']

        return ()  # for dynamic coloring or no coloring

    def system_iter(self, include_self=False, recurse=True, typ=None):
        """
        Yield a generator of local subsystems of this system.

        Parameters
        ----------
        include_self : bool
            If True, include this system in the iteration.
        recurse : bool
            If True, iterate over the whole tree under this system.
        typ : type
            If not None, only yield Systems that match that are instances of the
            given type.
        """
        if include_self and (typ is None or isinstance(self, typ)):
            yield self

        for s in self._subsystems_myproc:
            if typ is None or isinstance(s, typ):
                yield s
            if recurse:
                for sub in s.system_iter(recurse=True, typ=typ):
                    yield sub

    def add_design_var(self, name, lower=None, upper=None, ref=None, ref0=None, indices=None,
                       adder=None, scaler=None, units=None,
                       parallel_deriv_color=None, vectorize_derivs=False,
                       cache_linear_solution=False):
        r"""
        Add a design variable to this system.

        Parameters
        ----------
        name : string
            Name of the design variable in the system.
        lower : float or ndarray, optional
            Lower boundary for the input
        upper : upper or ndarray, optional
            Upper boundary for the input
        ref : float or ndarray, optional
            Value of design var that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of design var that scales to 0.0 in the driver.
        indices : iter of int, optional
            If an input is an array, these indicate which entries are of
            interest for this particular design variable.  These may be
            positive or negative integers.
        units : str, optional
            Units to convert to before applying scaling.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value for the driver. adder
            is first in precedence.  adder and scaler are an alterantive to using ref
            and ref0.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value for the driver. scaler
            is second in precedence. adder and scaler are an alterantive to using ref
            and ref0.
        parallel_deriv_color : string
            If specified, this design var will be grouped for parallel derivative
            calculations with other variables sharing the same parallel_deriv_color.
        vectorize_derivs : bool
            If True, vectorize derivative calculations.
        cache_linear_solution : bool
            If True, store the linear solution vectors for this variable so they can
            be used to start the next linear solution with an initial guess equal to the
            solution from the previous linear solve.

        Notes
        -----
        The response can be scaled using ref and ref0.
        The argument :code:`ref0` represents the physical value when the scaled value is 0.
        The argument :code:`ref` represents the physical value when the scaled value is 1.
        """
        if name in self._design_vars or name in self._static_design_vars:
            msg = "{}: Design Variable '{}' already exists."
            raise RuntimeError(msg.format(self.msginfo, name))

        # Name must be a string
        if not isinstance(name, str):
            raise TypeError('{}: The name argument should be a string, got {}'.format(self.msginfo,
                                                                                      name))

        if units is not None:
            if not isinstance(units, str):
                raise TypeError(f"{self.msginfo}: The units argument should be a str or None for "
                                f"design_var '{name}'.")

            if not valid_units(units):
                raise ValueError(f"{self.msginfo}: The units '{units}' are invalid for "
                                 f"design_var '{name}'.")
            units = simplify_unit(units)

        # Convert ref/ref0 to ndarray/float as necessary
        ref = format_as_float_or_array('ref', ref, val_if_none=None, flatten=True)
        ref0 = format_as_float_or_array('ref0', ref0, val_if_none=None, flatten=True)

        # determine adder and scaler based on args
        adder, scaler = determine_adder_scaler(ref0, ref, adder, scaler)

        if lower is None:
            # if not set, set lower to -INF_BOUND and don't apply adder/scaler
            lower = -openmdao.INF_BOUND
        else:
            # Convert lower to ndarray/float as necessary
            lower = format_as_float_or_array('lower', lower, flatten=True)
            # Apply scaler/adder
            lower = (lower + adder) * scaler

        if upper is None:
            # if not set, set upper to INF_BOUND and don't apply adder/scaler
            upper = openmdao.INF_BOUND
        else:
            # Convert upper to ndarray/float as necessary
            upper = format_as_float_or_array('upper', upper, flatten=True)
            # Apply scaler/adder
            upper = (upper + adder) * scaler

        if self._static_mode:
            design_vars = self._static_design_vars
        else:
            design_vars = self._design_vars

        dvs = OrderedDict()

        if isinstance(scaler, np.ndarray):
            if np.all(scaler == 1.0):
                scaler = None
        elif scaler == 1.0:
            scaler = None
        dvs['scaler'] = scaler

        if isinstance(adder, np.ndarray):
            if not np.any(adder):
                adder = None
        elif adder == 0.0:
            adder = None
        dvs['adder'] = adder

        dvs['name'] = name
        dvs['upper'] = upper
        dvs['lower'] = lower
        dvs['ref'] = ref
        dvs['ref0'] = ref0
        dvs['units'] = units
        dvs['cache_linear_solution'] = cache_linear_solution

        if indices is not None:

            if _is_slicer_op(indices):
                pass
            # If given, indices must be a sequence
            elif not (isinstance(indices, Iterable) and
                      all([isinstance(i, Integral) for i in indices])):
                raise ValueError("{}: If specified, design var indices must be a sequence of "
                                 "integers.".format(self.msginfo))
            else:
                indices = np.atleast_1d(indices)
                dvs['size'] = size = len(indices)

            # All refs: check the shape if necessary
            for item, item_name in zip([ref, ref0, scaler, adder, upper, lower],
                                       ['ref', 'ref0', 'scaler', 'adder', 'upper', 'lower']):
                if isinstance(item, np.ndarray):
                    if item.size != size:
                        raise ValueError("%s: When adding design var '%s', %s should have size "
                                         "%d but instead has size %d." % (self.msginfo, name,
                                                                          item_name, size,
                                                                          item.size))

        dvs['indices'] = indices
        dvs['parallel_deriv_color'] = parallel_deriv_color
        dvs['vectorize_derivs'] = vectorize_derivs

        design_vars[name] = dvs

    def add_response(self, name, type_, lower=None, upper=None, equals=None,
                     ref=None, ref0=None, indices=None, index=None, units=None,
                     adder=None, scaler=None, linear=False, parallel_deriv_color=None,
                     vectorize_derivs=False, cache_linear_solution=False):
        r"""
        Add a response variable to this system.

        The response can be scaled using ref and ref0.
        The argument :code:`ref0` represents the physical value when the scaled value is 0.
        The argument :code:`ref` represents the physical value when the scaled value is 1.

        Parameters
        ----------
        name : string
            Name of the response variable in the system.
        type_ : string
            The type of response. Supported values are 'con' and 'obj'
        lower : float or ndarray, optional
            Lower boundary for the variable
        upper : upper or ndarray, optional
            Upper boundary for the variable
        equals : equals or ndarray, optional
            Equality constraint value for the variable
        ref : float or ndarray, optional
            Value of response variable that scales to 1.0 in the driver.
        ref0 : upper or ndarray, optional
            Value of response variable that scales to 0.0 in the driver.
        indices : sequence of int, optional
            If variable is an array, these indicate which entries are of
            interest for this particular response.
        index : int, optional
            If variable is an array, this indicates which entry is of
            interest for this particular response.
        units : str, optional
            Units to convert to before applying scaling.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value for the driver. adder
            is first in precedence.  adder and scaler are an alterantive to using ref
            and ref0.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value for the driver. scaler
            is second in precedence. adder and scaler are an alterantive to using ref
            and ref0.
        linear : bool
            Set to True if constraint is linear. Default is False.
        parallel_deriv_color : string
            If specified, this design var will be grouped for parallel derivative
            calculations with other variables sharing the same parallel_deriv_color.
        vectorize_derivs : bool
            If True, vectorize derivative calculations.
        cache_linear_solution : bool
            If True, store the linear solution vectors for this variable so they can
            be used to start the next linear solution with an initial guess equal to the
            solution from the previous linear solve.
        """
        # Name must be a string
        if not isinstance(name, str):
            raise TypeError('{}: The name argument should be a string, '
                            'got {}'.format(self.msginfo, name))

        # Type must be a string and one of 'con' or 'obj'
        if not isinstance(type_, str):
            raise TypeError('{}: The type argument should be a string'.format(self.msginfo))
        elif type_ not in ('con', 'obj'):
            raise ValueError('{}: The type must be one of \'con\' or \'obj\': '
                             'Got \'{}\' instead'.format(self.msginfo, name))

        if units is not None:
            if not isinstance(units, str):
                raise TypeError(f"{self.msginfo}: The units argument should be a str or None for "
                                f"response '{name}'.")

            if not valid_units(units):
                raise ValueError(f"{self.msginfo}: The units '{units}' are invalid for "
                                 f"response '{name}'.")

            units = simplify_unit(units)

        if name in self._responses or name in self._static_responses:
            typemap = {'con': 'Constraint', 'obj': 'Objective'}
            msg = "{}: {} '{}' already exists.".format(self.msginfo, typemap[type_], name)
            raise RuntimeError(msg.format(name))

        # Convert ref/ref0 to ndarray/float as necessary
        ref = format_as_float_or_array('ref', ref, val_if_none=None, flatten=True)
        ref0 = format_as_float_or_array('ref0', ref0, val_if_none=None, flatten=True)

        # determine adder and scaler based on args
        adder, scaler = determine_adder_scaler(ref0, ref, adder, scaler)

        # A constraint cannot be an equality and inequality constraint
        if equals is not None and (lower is not None or upper is not None):
            msg = "{}: Constraint '{}' cannot be both equality and inequality."
            raise ValueError(msg.format(self.msginfo, name))

        if _is_slicer_op(indices):
            pass
        # If given, indices must be a sequence
        elif (indices is not None and not (
                isinstance(indices, Iterable) and all([isinstance(i, Integral) for i in indices]))):
            raise ValueError("{}: If specified, response indices must be a sequence of "
                             "integers.".format(self.msginfo))

        if self._static_mode:
            responses = self._static_responses
        else:
            responses = self._responses

        resp = OrderedDict()

        if type_ == 'con':

            # Convert lower to ndarray/float as necessary
            try:
                if lower is None:
                    # don't apply adder/scaler if lower not set
                    lower = -openmdao.INF_BOUND
                else:
                    lower = format_as_float_or_array('lower', lower, flatten=True)
                    lower = (lower + adder) * scaler
            except (TypeError, ValueError):
                raise TypeError("Argument 'lower' can not be a string ('{}' given). You can not "
                                "specify a variable as lower bound. You can only provide constant "
                                "float values".format(lower))

            # Convert upper to ndarray/float as necessary
            try:
                if upper is None:
                    # don't apply adder/scaler if upper not set
                    upper = openmdao.INF_BOUND
                else:
                    upper = format_as_float_or_array('upper', upper, flatten=True)
                    upper = (upper + adder) * scaler
            except (TypeError, ValueError):
                raise TypeError("Argument 'upper' can not be a string ('{}' given). You can not "
                                "specify a variable as upper bound. You can only provide constant "
                                "float values".format(upper))
            # Convert equals to ndarray/float as necessary
            if equals is not None:
                try:
                    equals = format_as_float_or_array('equals', equals, flatten=True)
                except (TypeError, ValueError):
                    raise TypeError("Argument 'equals' can not be a string ('{}' given). You can "
                                    "not specify a variable as equals bound. You can only provide "
                                    "constant float values".format(equals))
                equals = (equals + adder) * scaler

            resp['lower'] = lower
            resp['upper'] = upper
            resp['equals'] = equals
            resp['linear'] = linear
            if indices is not None:
                indices = np.atleast_1d(indices)
                resp['size'] = len(indices)
            resp['indices'] = indices
        else:  # 'obj'
            if index is not None:
                resp['size'] = 1
                index = np.array([index], dtype=INT_DTYPE)
            resp['indices'] = index

        if isinstance(scaler, np.ndarray):
            if np.all(scaler == 1.0):
                scaler = None
        elif scaler == 1.0:
            scaler = None
        resp['scaler'] = scaler

        if isinstance(adder, np.ndarray):
            if not np.any(adder):
                adder = None
        elif adder == 0.0:
            adder = None
        resp['adder'] = adder

        if resp['indices'] is not None:
            size = resp['indices'].size
            vlist = [ref, ref0, scaler, adder]
            nlist = ['ref', 'ref0', 'scaler', 'adder']
            if type_ == 'con':
                tname = 'constraint'
                vlist.extend([upper, lower, equals])
                nlist.extend(['upper', 'lower', 'equals'])
            else:
                tname = 'objective'

            # All refs: check the shape if necessary
            for item, item_name in zip(vlist, nlist):
                if isinstance(item, np.ndarray):
                    if item.size != size:
                        raise ValueError("%s: When adding %s '%s', %s should have size "
                                         "%d but instead has size %d." % (self.msginfo, tname,
                                                                          name, item_name, size,
                                                                          item.size))
        resp['name'] = name
        resp['ref'] = ref
        resp['ref0'] = ref0
        resp['type'] = type_
        resp['units'] = units
        resp['cache_linear_solution'] = cache_linear_solution

        resp['parallel_deriv_color'] = parallel_deriv_color
        resp['vectorize_derivs'] = vectorize_derivs

        responses[name] = resp

    def add_constraint(self, name, lower=None, upper=None, equals=None,
                       ref=None, ref0=None, adder=None, scaler=None, units=None,
                       indices=None, linear=False, parallel_deriv_color=None,
                       vectorize_derivs=False, cache_linear_solution=False):
        r"""
        Add a constraint variable to this system.

        Parameters
        ----------
        name : string
            Name of the response variable in the system.
        lower : float or ndarray, optional
            Lower boundary for the variable
        upper : float or ndarray, optional
            Upper boundary for the variable
        equals : float or ndarray, optional
            Equality constraint value for the variable
        ref : float or ndarray, optional
            Value of response variable that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of response variable that scales to 0.0 in the driver.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value for the driver. adder
            is first in precedence.  adder and scaler are an alterantive to using ref
            and ref0.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value for the driver. scaler
            is second in precedence. adder and scaler are an alterantive to using ref
            and ref0.
        units : str, optional
            Units to convert to before applying scaling.
        indices : sequence of int, optional
            If variable is an array, these indicate which entries are of
            interest for this particular response.  These may be positive or
            negative integers.
        linear : bool
            Set to True if constraint is linear. Default is False.
        parallel_deriv_color : string
            If specified, this design var will be grouped for parallel derivative
            calculations with other variables sharing the same parallel_deriv_color.
        vectorize_derivs : bool
            If True, vectorize derivative calculations.
        cache_linear_solution : bool
            If True, store the linear solution vectors for this variable so they can
            be used to start the next linear solution with an initial guess equal to the
            solution from the previous linear solve.

        Notes
        -----
        The response can be scaled using ref and ref0.
        The argument :code:`ref0` represents the physical value when the scaled value is 0.
        The argument :code:`ref` represents the physical value when the scaled value is 1.
        The arguments (:code:`lower`, :code:`upper`, :code:`equals`) can not be strings or variable
        names.
        """
        self.add_response(name=name, type_='con', lower=lower, upper=upper,
                          equals=equals, scaler=scaler, adder=adder, ref=ref,
                          ref0=ref0, indices=indices, linear=linear, units=units,
                          parallel_deriv_color=parallel_deriv_color,
                          vectorize_derivs=vectorize_derivs,
                          cache_linear_solution=cache_linear_solution)

    def add_objective(self, name, ref=None, ref0=None, index=None, units=None,
                      adder=None, scaler=None, parallel_deriv_color=None,
                      vectorize_derivs=False, cache_linear_solution=False):
        r"""
        Add a response variable to this system.

        Parameters
        ----------
        name : string
            Name of the response variable in the system.
        ref : float or ndarray, optional
            Value of response variable that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of response variable that scales to 0.0 in the driver.
        index : int, optional
            If variable is an array, this indicates which entry is of
            interest for this particular response. This may be a positive
            or negative integer.
        units : str, optional
            Units to convert to before applying scaling.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value for the driver. adder
            is first in precedence.  adder and scaler are an alterantive to using ref
            and ref0.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value for the driver. scaler
            is second in precedence. adder and scaler are an alterantive to using ref
            and ref0.
        parallel_deriv_color : string
            If specified, this design var will be grouped for parallel derivative
            calculations with other variables sharing the same parallel_deriv_color.
        vectorize_derivs : bool
            If True, vectorize derivative calculations.
        cache_linear_solution : bool
            If True, store the linear solution vectors for this variable so they can
            be used to start the next linear solution with an initial guess equal to the
            solution from the previous linear solve.

        Notes
        -----
        The objective can be scaled using scaler and adder, where

        .. math::

            x_{scaled} = scaler(x + adder)

        or through the use of ref/ref0, which map to scaler and adder through
        the equations:

        .. math::

            0 = scaler(ref_0 + adder)

            1 = scaler(ref + adder)

        which results in:

        .. math::

            adder = -ref_0

            scaler = \frac{1}{ref + adder}
        """
        if index is not None and not isinstance(index, int):
            raise TypeError('{}: If specified, objective index must be '
                            'an int.'.format(self.msginfo))
        self.add_response(name, type_='obj', scaler=scaler, adder=adder,
                          ref=ref, ref0=ref0, index=index, units=units,
                          parallel_deriv_color=parallel_deriv_color,
                          vectorize_derivs=vectorize_derivs,
                          cache_linear_solution=cache_linear_solution)

    def get_design_vars(self, recurse=True, get_sizes=True, use_prom_ivc=True):
        """
        Get the DesignVariable settings from this system.

        Retrieve all design variable settings from the system and, if recurse
        is True, all of its subsystems.

        Parameters
        ----------
        recurse : bool
            If True, recurse through the subsystems and return the path of
            all design vars relative to the this system.
        get_sizes : bool, optional
            If True, compute the size of each design variable.
        use_prom_ivc : bool
            Translate auto_ivc_names to their promoted input names.

        Returns
        -------
        dict
            The design variables defined in the current system and, if
            recurse=True, its subsystems.

        """
        pro2abs_out = self._var_allprocs_prom2abs_list['output']
        pro2abs_in = self._var_allprocs_prom2abs_list['input']
        model = self._problem_meta['model_ref']()
        conns = model._conn_global_abs_in2out
        abs2meta_out = model._var_allprocs_abs2meta['output']

        # Human readable error message during Driver setup.
        out = OrderedDict()
        try:
            for name, data in self._design_vars.items():
                if name in pro2abs_out:

                    # This is an output name, most likely a manual indepvarcomp.
                    abs_name = pro2abs_out[name][0]
                    out[abs_name] = data
                    out[abs_name]['ivc_source'] = abs_name
                    out[abs_name]['distributed'] = \
                        abs_name in abs2meta_out and abs2meta_out[abs_name]['distributed']

                else:  # assume an input name else KeyError

                    # Design variable on an auto_ivc input, so use connected output name.
                    in_abs = pro2abs_in[name][0]
                    ivc_path = conns[in_abs]
                    distrib = ivc_path in abs2meta_out and abs2meta_out[ivc_path]['distributed']
                    if use_prom_ivc:
                        out[name] = data
                        out[name]['ivc_source'] = ivc_path
                        out[name]['distributed'] = distrib
                    else:
                        out[ivc_path] = data
                        out[ivc_path]['ivc_source'] = ivc_path
                        out[ivc_path]['distributed'] = distrib

        except KeyError as err:
            msg = "{}: Output not found for design variable {}."
            raise RuntimeError(msg.format(self.msginfo, str(err)))

        if get_sizes:
            # Size them all
            sizes = model._var_sizes['nonlinear']['output']
            abs2idx = model._var_allprocs_abs2idx['nonlinear']
            owning_rank = model._owning_rank

            for name, meta in out.items():

                src_name = name
                if meta['ivc_source'] is not None:
                    src_name = meta['ivc_source']

                if 'size' not in meta:
                    if src_name in abs2idx:
                        if meta['distributed']:
                            meta['size'] = sizes[model.comm.rank, abs2idx[src_name]]
                        else:
                            meta['size'] = sizes[owning_rank[src_name], abs2idx[src_name]]
                    else:
                        meta['size'] = 0  # discrete var, don't know size
                meta['size'] = int(meta['size'])  # make default int so will be json serializable

                if src_name in abs2idx:
                    meta = abs2meta_out[src_name]
                    out[name]['distributed'] = meta['distributed']
                    out[name]['global_size'] = meta['global_size']
                else:
                    out[name]['global_size'] = 0  # discrete var

        if recurse:
            abs2prom_in = self._var_allprocs_abs2prom['input']
            for subsys in self._subsystems_myproc:
                dvs = subsys.get_design_vars(recurse=recurse, get_sizes=get_sizes,
                                             use_prom_ivc=use_prom_ivc)
                if use_prom_ivc:
                    # have to promote subsystem prom name to this level
                    sub_pro2abs_in = subsys._var_allprocs_prom2abs_list['input']
                    for dv, meta in dvs.items():
                        if dv in sub_pro2abs_in:
                            abs_dv = sub_pro2abs_in[dv][0]
                            out[abs2prom_in[abs_dv]] = meta
                        else:
                            out[dv] = meta
                else:
                    out.update(dvs)

            if self.comm.size > 1 and self._subsystems_allprocs:
                my_out = out
                allouts = self.comm.allgather(out)
                out = OrderedDict()
                for rank, all_out in enumerate(allouts):
                    for name, meta in all_out.items():
                        if name not in out:
                            if name in my_out:
                                out[name] = my_out[name]
                            else:
                                out[name] = meta

        return out

    def get_responses(self, recurse=True, get_sizes=True, use_prom_ivc=False):
        """
        Get the response variable settings from this system.

        Retrieve all response variable settings from the system as a dict,
        keyed by variable name.

        Parameters
        ----------
        recurse : bool, optional
            If True, recurse through the subsystems and return the path of
            all responses relative to the this system.
        get_sizes : bool, optional
            If True, compute the size of each response.
        use_prom_ivc : bool
            Translate auto_ivc_names to their promoted input names.

        Returns
        -------
        dict
            The responses defined in the current system and, if
            recurse=True, its subsystems.

        """
        prom2abs = self._var_allprocs_prom2abs_list['output']
        prom2abs_in = self._var_allprocs_prom2abs_list['input']
        model = self._problem_meta['model_ref']()
        conns = model._conn_global_abs_in2out
        abs2meta_out = model._var_allprocs_abs2meta['output']

        # Human readable error message during Driver setup.
        try:
            out = {}
            for name, data in self._responses.items():
                if name in prom2abs:
                    abs_name = prom2abs[name][0]
                    out[abs_name] = data
                    out[abs_name]['ivc_source'] = abs_name
                    out[abs_name]['distributed'] = \
                        abs_name in abs2meta_out and abs2meta_out[abs_name]['distributed']

                else:
                    # A constraint can actaully be on an auto_ivc input, so use connected
                    # output name.
                    in_abs = prom2abs_in[name][0]
                    ivc_path = conns[in_abs]
                    distrib = ivc_path in abs2meta_out and abs2meta_out[ivc_path]['distributed']
                    if use_prom_ivc:
                        out[name] = data
                        out[name]['ivc_source'] = ivc_path
                        out[name]['distributed'] = distrib
                    else:
                        out[ivc_path] = data
                        out[ivc_path]['ivc_source'] = ivc_path
                        out[ivc_path]['distributed'] = distrib

        except KeyError as err:
            msg = "{}: Output not found for response {}."
            raise RuntimeError(msg.format(self.msginfo, str(err)))

        if get_sizes:
            # Size them all
            sizes = model._var_sizes['nonlinear']['output']
            abs2idx = model._var_allprocs_abs2idx['nonlinear']
            owning_rank = model._owning_rank
            for prom_name, response in out.items():
                name = response['ivc_source']

                # Discrete vars
                if name not in abs2idx:
                    response['size'] = response['global_size'] = 0  # discrete var, don't know size
                    continue

                meta = abs2meta_out[name]
                response['distributed'] = meta['distributed']

                if response['indices'] is not None:
                    # Index defined in this response.
                    response['global_size'] = len(response['indices']) if meta['distributed'] \
                        else meta['global_size']

                else:
                    response['size'] = sizes[owning_rank[name], abs2idx[name]]
                    response['global_size'] = meta['global_size']

        if recurse:
            abs2prom_in = self._var_allprocs_abs2prom['input']
            for subsys in self._subsystems_myproc:
                resps = subsys.get_responses(recurse=recurse, get_sizes=get_sizes,
                                             use_prom_ivc=use_prom_ivc)
                if use_prom_ivc:
                    # have to promote subsystem prom name to this level
                    sub_pro2abs_in = subsys._var_allprocs_prom2abs_list['input']
                    for dv, meta in resps.items():
                        if dv in sub_pro2abs_in:
                            abs_resp = sub_pro2abs_in[dv][0]
                            out[abs2prom_in[abs_resp]] = meta
                        else:
                            out[dv] = meta
                else:
                    out.update(resps)

            if self.comm.size > 1 and self._subsystems_allprocs:
                all_outs = self.comm.allgather(out)
                out = OrderedDict()
                for rank, all_out in enumerate(all_outs):
                    out.update(all_out)

        return out

    def get_constraints(self, recurse=True):
        """
        Get the Constraint settings from this system.

        Retrieve the constraint settings for the current system as a dict,
        keyed by variable name.

        Parameters
        ----------
        recurse : bool, optional
            If True, recurse through the subsystems and return the path of
            all constraints relative to the this system.

        Returns
        -------
        dict
            The constraints defined in the current system.

        """
        return OrderedDict((key, response) for (key, response) in
                           self.get_responses(recurse=recurse).items()
                           if response['type'] == 'con')

    def get_objectives(self, recurse=True):
        """
        Get the Objective settings from this system.

        Retrieve all objectives settings from the system as a dict, keyed
        by variable name.

        Parameters
        ----------
        recurse : bool, optional
            If True, recurse through the subsystems and return the path of
            all objective relative to the this system.

        Returns
        -------
        dict
            The objectives defined in the current system.

        """
        return OrderedDict((key, response) for (key, response) in
                           self.get_responses(recurse=recurse).items()
                           if response['type'] == 'obj')

    def run_apply_nonlinear(self):
        """
        Compute residuals.

        This calls _apply_nonlinear, but with the model assumed to be in an unscaled state.
        """
        with self._scaled_context_all():
            self._apply_nonlinear()

    def get_io_metadata(self, iotypes=('input', 'output'), metadata_keys=None,
                        includes=None, excludes=None, tags=(), get_remote=False, rank=None,
                        return_rel_names=True):
        """
        Retrieve metdata for a filtered list of variables.

        Parameters
        ----------
        iotypes : str or iter of str
            Will contain either 'input', 'output', or both.  Defaults to both.
        metadata_keys : iter of str or None
            Names of metadata entries to be retrieved or None, meaning retrieve all
            available 'allprocs' metadata.  If 'values' or 'src_indices' are required,
            their keys must be provided explicitly since they are not found in the 'allprocs'
            metadata and must be retrieved from local metadata located in each process.
        includes : str, iter of str or None
            Collection of glob patterns for pathnames of variables to include. Default is None,
            which includes all variables.
        excludes : str, iter of str or None
            Collection of glob patterns for pathnames of variables to exclude. Default is None.
        tags : str or iter of strs
            User defined tags that can be used to filter what gets listed. Only inputs with the
            given tags will be listed.
            Default is None, which means there will be no filtering based on tags.
        get_remote : bool
            If True, retrieve variables from other MPI processes as well.
        rank : int or None
            If None, and get_remote is True, retrieve values from all MPI process to all other
            MPI processes.  Otherwise, if get_remote is True, retrieve values from all MPI
            processes only to the specified rank.
        return_rel_names : bool
            If True, the names returned will be relative to the scope of this System. Otherwise
            they will be absolute names.

        Returns
        -------
        dict
            A dict of metadata keyed on name, where name is either absolute or relative
            based on the value of the `return_rel_names` arg, and metadata is a dict containing
            entries based on the value of the metadata_keys arg.  Every metadata dict will
            always contain two entries, 'promoted_name' and 'discrete', to indicate a given
            variable's promoted name and whether or not it is discrete.
        """
        prefix = self.pathname + '.' if self.pathname else ''
        rel_idx = len(prefix)

        if isinstance(iotypes, str):
            iotypes = (iotypes,)
        if isinstance(includes, str):
            includes = (includes,)
        if isinstance(excludes, str):
            excludes = (excludes,)

        loc2meta = self._var_abs2meta
        all2meta = self._var_allprocs_abs2meta

        dynset = set(('shape', 'size', 'value'))
        gather_keys = {'value', 'src_indices'}
        need_gather = get_remote and self.comm.size > 1
        if metadata_keys is not None:
            keyset = set(metadata_keys)
            diff = keyset - allowed_meta_names
            if diff:
                raise RuntimeError(f"{self.msginfo}: {sorted(diff)} are not valid metadata entry "
                                   "names.")
        need_local_meta = metadata_keys is not None and len(gather_keys.intersection(keyset)) > 0
        nodyn = metadata_keys is None or keyset.intersection(dynset)

        if need_local_meta:
            metadict = loc2meta
            disc_metadict = self._var_discrete
        else:
            metadict = all2meta
            disc_metadict = self._var_allprocs_discrete
            need_gather = False  # we can get everything from 'allprocs' dict without gathering

        if tags:
            tagset = make_set(tags)

        result = {}

        it = self._var_allprocs_abs2prom if get_remote else self._var_abs2prom

        for iotype in iotypes:
            cont2meta = metadict[iotype]
            disc2meta = disc_metadict[iotype]

            for abs_name, prom in it[iotype].items():
                if not match_prom_or_abs(abs_name, prom, includes, excludes):
                    continue

                rel_name = abs_name[rel_idx:]

                if abs_name in all2meta[iotype]:  # continuous
                    meta = cont2meta[abs_name] if abs_name in cont2meta else None
                    distrib = all2meta[iotype][abs_name]['distributed']
                    if nodyn:
                        a2m = all2meta[iotype][abs_name]
                        if a2m['shape'] is None and (a2m['shape_by_conn'] or a2m['copy_shape']):
                            raise RuntimeError(f"{self.msginfo}: Can't retrieve shape, size, or "
                                               f"value for dynamically sized variable '{prom}' "
                                               "because they aren't known yet.")
                else:  # discrete
                    if need_local_meta:  # use relative name for discretes
                        meta = disc2meta[rel_name] if rel_name in disc2meta else None
                    else:
                        meta = disc2meta[abs_name]
                    distrib = False

                if meta is None:
                    ret_meta = None
                else:
                    if metadata_keys is None:
                        ret_meta = meta.copy()
                    else:
                        ret_meta = {}
                        for key in metadata_keys:
                            try:
                                ret_meta[key] = meta[key]
                            except KeyError:
                                ret_meta[key] = 'Unavailable'

                if need_gather:
                    if distrib or abs_name in self._vars_to_gather:
                        if rank is None:
                            allproc_metas = self.comm.allgather(ret_meta)
                        else:
                            allproc_metas = self.comm.gather(ret_meta, root=rank)

                        if rank is None or self.comm.rank == rank:
                            if not ret_meta:
                                ret_meta = {}
                            if distrib:
                                if 'value' in metadata_keys:
                                    # assemble the full distributed value
                                    dist_vals = [m['value'] for m in allproc_metas
                                                 if m is not None and m['value'].size > 0]
                                    if dist_vals:
                                        ret_meta['value'] = np.concatenate(dist_vals)
                                    else:
                                        ret_meta['value'] = np.zeros(0)
                                if 'src_indices' in metadata_keys:
                                    # assemble full src_indices
                                    dist_src_inds = [m['src_indices'] for m in allproc_metas
                                                     if m is not None and m['src_indices'].size > 0]
                                    if dist_src_inds:
                                        ret_meta['src_indices'] = np.concatenate(dist_src_inds)
                                    else:
                                        ret_meta['src_indices'] = np.zeros(0, dtype=INT_DTYPE)

                            elif abs_name in self._vars_to_gather:
                                for m in allproc_metas:
                                    if m is not None:
                                        ret_meta = m
                                        break
                        else:
                            ret_meta = None

                if ret_meta is not None:
                    ret_meta['prom_name'] = prom
                    ret_meta['discrete'] = abs_name not in all2meta

                    vname = rel_name if return_rel_names else abs_name

                    if tags and not tagset & ret_meta['tags']:
                        continue

                    result[vname] = ret_meta

        return result

    def list_inputs(self,
                    values=True,
                    prom_name=False,
                    units=False,
                    shape=False,
                    global_shape=False,
                    desc=False,
                    hierarchical=True,
                    print_arrays=False,
                    tags=None,
                    includes=None,
                    excludes=None,
                    all_procs=False,
                    out_stream=_DEFAULT_OUT_STREAM):
        """
        Write a list of input names and other optional information to a specified stream.

        Parameters
        ----------
        values : bool, optional
            When True, display/return input values. Default is True.
        prom_name : bool, optional
            When True, display/return the promoted name of the variable.
            Default is False.
        units : bool, optional
            When True, display/return units. Default is False.
        shape : bool, optional
            When True, display/return the shape of the value. Default is False.
        global_shape : bool, optional
            When True, display/return the global shape of the value. Default is False.
        desc : bool, optional
            When True, display/return description. Default is False.
        hierarchical : bool, optional
            When True, human readable output shows variables in hierarchical format.
        print_arrays : bool, optional
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format is affected
            by the values set with numpy.set_printoptions
            Default is False.
        tags : str or list of strs
            User defined tags that can be used to filter what gets listed. Only inputs with the
            given tags will be listed.
            Default is None, which means there will be no filtering based on tags.
        includes : None or iter of str
            Collection of glob patterns for pathnames of variables to include. Default is None,
            which includes all input variables.
        excludes : None or iter of str
            Collection of glob patterns for pathnames of variables to exclude. Default is None.
        all_procs : bool, optional
            When True, display output on all ranks. Default is False, which will display
            output only from rank 0.
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        list of (name, metadata)
            List of input names and other optional information about those inputs.
        """
        metavalues = values and self._inputs is None
        keynames = ['value', 'units', 'shape', 'global_shape', 'desc', 'tags']
        keyvals = [metavalues, units, shape, global_shape, desc, tags is not None]
        keys = [n for i, n in enumerate(keynames) if keyvals[i]]

        inputs = self.get_io_metadata(('input',), keys, includes, excludes, tags,
                                      get_remote=True,
                                      rank=None if all_procs or values else 0,
                                      return_rel_names=False)

        if inputs:
            to_remove = ['discrete']
            if tags:
                to_remove.append('tags')
            if not prom_name:
                to_remove.append('prom_name')

            for _, meta in inputs.items():
                for key in to_remove:
                    del meta[key]

        if values and self._inputs is not None:
            # we want value from the input vector, not from the metadata
            for n, meta in inputs.items():
                meta['value'] = self._abs_get_val(n, get_remote=True,
                                                  rank=None if all_procs else 0, kind='input')

        if not inputs or (not all_procs and self.comm.rank != 0):
            return []

        if out_stream is _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        if out_stream:
            self._write_table('input', inputs, hierarchical, print_arrays, all_procs, out_stream)

        if self.pathname:
            # convert to relative names
            rel_idx = len(self.pathname) + 1
            inputs = [(n[rel_idx:], meta) for n, meta in inputs.items()]
        else:
            inputs = list(inputs.items())

        return inputs

    def list_outputs(self,
                     explicit=True, implicit=True,
                     values=True,
                     prom_name=False,
                     residuals=False,
                     residuals_tol=None,
                     units=False,
                     shape=False,
                     global_shape=False,
                     bounds=False,
                     scaling=False,
                     desc=False,
                     hierarchical=True,
                     print_arrays=False,
                     tags=None,
                     includes=None,
                     excludes=None,
                     all_procs=False,
                     list_autoivcs=False,
                     out_stream=_DEFAULT_OUT_STREAM):
        """
        Write a list of output names and other optional information to a specified stream.

        Parameters
        ----------
        explicit : bool, optional
            include outputs from explicit components. Default is True.
        implicit : bool, optional
            include outputs from implicit components. Default is True.
        values : bool, optional
            When True, display output values. Default is True.
        prom_name : bool, optional
            When True, display the promoted name of the variable.
            Default is False.
        residuals : bool, optional
            When True, display residual values. Default is False.
        residuals_tol : float, optional
            If set, limits the output of list_outputs to only variables where
            the norm of the resids array is greater than the given 'residuals_tol'.
            Default is None.
        units : bool, optional
            When True, display units. Default is False.
        shape : bool, optional
            When True, display/return the shape of the value. Default is False.
        global_shape : bool, optional
            When True, display/return the global shape of the value. Default is False.
        bounds : bool, optional
            When True, display/return bounds (lower and upper). Default is False.
        scaling : bool, optional
            When True, display/return scaling (ref, ref0, and res_ref). Default is False.
        desc : bool, optional
            When True, display/return description. Default is False.
        hierarchical : bool, optional
            When True, human readable output shows variables in hierarchical format.
        print_arrays : bool, optional
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format  is affected
            by the values set with numpy.set_printoptions
            Default is False.
        tags : str or list of strs
            User defined tags that can be used to filter what gets listed. Only outputs with the
            given tags will be listed.
            Default is None, which means there will be no filtering based on tags.
        includes : None or iter of str
            Collection of glob patterns for pathnames of variables to include. Default is None,
            which includes all output variables.
        excludes : None or iter of str
            Collection of glob patterns for pathnames of variables to exclude. Default is None.
        all_procs : bool, optional
            When True, display output on all processors. Default is False.
        list_autoivcs : bool
            If True, include auto_ivc outputs in the listing.  Defaults to False.
        out_stream : file-like
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        list of (name, metadata)
            List of output names and other optional information about those outputs.
        """
        keynames = np.array(['value', 'units', 'shape', 'global_shape', 'desc', 'tags'])
        keys = [str(n) for n in keynames[np.array([values, units, shape, global_shape, desc, tags],
                                                  dtype=bool)]]
        if bounds:
            keys.extend(('lower', 'upper'))
        if scaling:
            keys.extend(('ref', 'ref0', 'res_ref'))

        outputs = self.get_io_metadata(('output',), keys, includes, excludes, tags,
                                       get_remote=True,
                                       rank=None if all_procs or values or residuals else 0,
                                       return_rel_names=False)

        if outputs:
            if not list_autoivcs:
                outputs = {n: m for n, m in outputs.items() if not n.startswith('_auto_ivc.')}

            to_remove = ['discrete']
            if tags:
                to_remove.append('tags')
            if not prom_name:
                to_remove.append('prom_name')

            for _, meta in outputs.items():
                for key in to_remove:
                    del meta[key]

        if self._outputs is not None and (values or residuals):
            # we want value from the input vector, not from the metadata
            for n, meta in outputs.items():
                if values:
                    meta['value'] = self._abs_get_val(n, get_remote=True,
                                                      rank=None if all_procs else 0, kind='output')
                if residuals:
                    meta['resids'] = self._abs_get_val(n, get_remote=True,
                                                       rank=None if all_procs else 0,
                                                       kind='residual')

        if not outputs or (not all_procs and self.comm.rank != 0):
            return []

        if out_stream is _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        rel_idx = len(self.pathname) + 1 if self.pathname else 0

        states = set(self._list_states())

        if explicit:
            expl_outputs = {n: m for n, m in outputs.items() if n not in states}
            if out_stream:
                self._write_table('explicit', expl_outputs, hierarchical, print_arrays,
                                  all_procs, out_stream)
            if self.name:  # convert to relative name
                expl_outputs = [(n[rel_idx:], meta) for n, meta in expl_outputs.items()]
            else:
                expl_outputs = list(expl_outputs.items())

        if implicit:
            impl_outputs = {}
            if residuals_tol:
                for n, m in outputs.items():
                    if "resids" in m and n in states:
                        if not np.isscalar(m['resids']) and len(m['resids']) > 1:
                            for i in m['resids']:
                                if i > residuals_tol:
                                    impl_outputs[n] = m
                                    break
                        elif m['resids'] > residuals_tol:
                            impl_outputs[n] = m
            else:
                impl_outputs = {n: m for n, m in outputs.items() if n in states}
            if out_stream:
                self._write_table('implicit', impl_outputs, hierarchical, print_arrays,
                                  all_procs, out_stream)
            if self.name:  # convert to relative name
                impl_outputs = [(n[rel_idx:], meta) for n, meta in impl_outputs.items()]
            else:
                impl_outputs = list(impl_outputs.items())

        if explicit:
            if implicit:
                return expl_outputs + impl_outputs
            return expl_outputs
        elif implicit:
            return impl_outputs
        else:
            raise RuntimeError(self.msginfo +
                               ': You have excluded both Explicit and Implicit components.')

    def _write_table(self, var_type, var_data, hierarchical, print_arrays, all_procs, out_stream):
        """
        Write table of variable names, values, residuals, and metadata to out_stream.

        Parameters
        ----------
        var_type : 'input', 'explicit' or 'implicit'
            Indicates type of variables, input or explicit/implicit output.
        var_data : dict
            dict of name and metadata.
        hierarchical : bool
            When True, human readable output shows variables in hierarchical format.
        print_arrays : bool
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format  is affected
            by the values set with numpy.set_printoptions
            Default is False.
        all_procs : bool, optional
            When True, display output on all processors.
        out_stream : file-like object
            Where to send human readable output.
            Set to None to suppress.
        """
        if out_stream is None:
            return

        if self._outputs is None:
            var_list = var_data.keys()
            top_name = self.name
        else:
            inputs = var_type == 'input'
            outputs = not inputs
            var_list = self._get_vars_exec_order(inputs=inputs, outputs=outputs, variables=var_data)
            top_name = self.name if self.name else 'model'

        if all_procs or self.comm.rank == 0:
            write_var_table(self.pathname, var_list, var_type, var_data,
                            hierarchical, top_name, print_arrays, out_stream)

    def _get_vars_exec_order(self, inputs=False, outputs=False, variables=None):
        """
        Get list of variable names in execution order, based on the order subsystems were setup.

        Parameters
        ----------
        outputs : bool, optional
            Get names of output variables. Default is False.
        inputs : bool, optional
            Get names of input variables. Default is False.
        variables : Collection (list or dict)
            Absolute path names of the subset of variables to include.
            If None then all variables will be included. Default is None.

        Returns
        -------
        list
            list of variable names in execution order
        """
        var_list = []

        real_vars = self._var_allprocs_abs2meta
        disc_vars = self._var_allprocs_discrete

        in_or_out = []
        if inputs:
            in_or_out.append('input')
        if outputs:
            in_or_out.append('output')

        if self._subsystems_allprocs:
            for subsys, _ in self._subsystems_allprocs.values():
                prefix = subsys.pathname + '.'
                for io in in_or_out:
                    for var_name in chain(real_vars[io], disc_vars[io]):
                        if variables is None or var_name in variables:
                            if var_name.startswith(prefix):
                                var_list.append(var_name)
        else:
            # For components with no children, self._subsystems_allprocs is empty.
            for io in in_or_out:
                for var_name in chain(real_vars[io], disc_vars[io]):
                    if not variables or var_name in variables:
                        var_list.append(var_name)

        return var_list

    def run_solve_nonlinear(self):
        """
        Compute outputs.

        This calls _solve_nonlinear, but with the model assumed to be in an unscaled state.

        """
        with self._scaled_context_all():
            self._solve_nonlinear()

    def run_apply_linear(self, vec_names, mode, scope_out=None, scope_in=None):
        """
        Compute jac-vec product.

        This calls _apply_linear, but with the model assumed to be in an unscaled state.

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
        with self._scaled_context_all():
            self._apply_linear(None, vec_names, ContainsAll(), mode, scope_out, scope_in)

    def run_solve_linear(self, vec_names, mode):
        """
        Apply inverse jac product.

        This calls _solve_linear, but with the model assumed to be in an unscaled state.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        """
        with self._scaled_context_all():
            self._solve_linear(vec_names, mode, ContainsAll())

    def run_linearize(self, sub_do_ln=True):
        """
        Compute jacobian / factorization.

        This calls _linearize, but with the model assumed to be in an unscaled state.

        Parameters
        ----------
        sub_do_ln : boolean
            Flag indicating if the children should call linearize on their linear solvers.
        """
        with self._scaled_context_all():
            do_ln = self._linear_solver is not None and self._linear_solver._linearize_children()
            self._linearize(self._assembled_jac, sub_do_ln=do_ln)
            if self._linear_solver is not None:
                self._linear_solver._linearize()

    def _apply_nonlinear(self):
        """
        Compute residuals. The model is assumed to be in a scaled state.
        """
        pass

    def check_config(self, logger):
        """
        Perform optional error checks.

        Parameters
        ----------
        logger : object
            The object that manages logging output.
        """
        pass

    def _apply_linear(self, jac, vec_names, rel_systems, mode, scope_in=None, scope_out=None):
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
        raise NotImplementedError(self.msginfo + ": _apply_linear has not been overridden")

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
        pass

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
        pass

    def _list_states(self):
        """
        Return list of all states at and below this system.

        Returns
        -------
        list
            List of all states.
        """
        return []

    def _list_states_allprocs(self):
        """
        Return list of all states at and below this system across all procs.

        Returns
        -------
        list
            List of all states.
        """
        return []

    def add_recorder(self, recorder, recurse=False):
        """
        Add a recorder to the system.

        Parameters
        ----------
        recorder : <CaseRecorder>
           A recorder instance.
        recurse : boolean
            Flag indicating if the recorder should be added to all the subsystems.
        """
        if MPI:
            raise RuntimeError(self.msginfo + ": Recording of Systems when running parallel "
                               "code is not supported yet")

        self._rec_mgr.append(recorder)

        if recurse:
            for s in self.system_iter(include_self=False, recurse=recurse):
                s._rec_mgr.append(recorder)

    def record_iteration(self):
        """
        Record an iteration of the current System.
        """
        global _recordable_funcs

        if self._rec_mgr._recorders:
            parallel = self._rec_mgr._check_parallel() if self.comm.size > 1 else False
            options = self.recording_options
            metadata = create_local_meta(self.pathname)

            # Get the data to record
            stack_top = self._recording_iter.stack[-1][0]
            method = stack_top.rsplit('.', 1)[-1]

            if method not in _recordable_funcs:
                raise ValueError("{}: {} must be one of: {}".format(self.msginfo, method,
                                                                    sorted(_recordable_funcs)))

            if 'nonlinear' in method:
                inputs, outputs, residuals = self.get_nonlinear_vectors()
                vec_name = 'nonlinear'
            else:
                inputs, outputs, residuals = self.get_linear_vectors()
                vec_name = 'linear'

            discrete_inputs = self._discrete_inputs
            discrete_outputs = self._discrete_outputs
            filt = self._filtered_vars_to_record

            data = {'input': {}, 'output': {}, 'residual': {}}
            if options['record_inputs'] and (inputs._names or len(discrete_inputs) > 0):
                data['input'] = self._retrieve_data_of_kind(filt, 'input', vec_name, parallel)

            if options['record_outputs'] and (outputs._names or len(discrete_outputs) > 0):
                data['output'] = self._retrieve_data_of_kind(filt, 'output', vec_name, parallel)

            if options['record_residuals'] and residuals._names:
                data['residual'] = self._retrieve_data_of_kind(filt, 'residual', vec_name, parallel)

            self._rec_mgr.record_iteration(self, data, metadata)

        # All calls to _solve_nonlinear are recorded, The counter is incremented after recording.
        self.iter_count += 1
        if not self.under_approx:
            self.iter_count_without_approx += 1

    def is_active(self):
        """
        Determine if the system is active on this rank.

        Returns
        -------
        bool
            If running under MPI, returns True if this `System` has a valid
            communicator. Always returns True if not running under MPI.
        """
        return MPI is None or not (self.comm is None or
                                   self.comm == MPI.COMM_NULL)

    def _clear_iprint(self):
        """
        Clear out the iprint stack from the solvers.
        """
        self.nonlinear_solver._solver_info.clear()

    def _reset_iter_counts(self):
        """
        Recursively reset iteration counter for all systems and solvers.
        """
        for s in self.system_iter(include_self=True, recurse=True):
            s.iter_count = 0
            s.iter_count_apply = 0
            s.iter_count_without_approx = 0

            if s._linear_solver:
                s._linear_solver._iter_count = 0
            if s._nonlinear_solver:
                nl = s._nonlinear_solver
                nl._iter_count = 0
                if hasattr(nl, 'linesearch') and nl.linesearch:
                    nl.linesearch._iter_count = 0

    def _set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        Recurses to turn on or off complex stepping mode in all subsystems and their vectors.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        for sub in self.system_iter(include_self=True, recurse=True):
            sub.under_complex_step = active
            sub._inputs.set_complex_step_mode(active)
            sub._outputs.set_complex_step_mode(active)
            sub._residuals.set_complex_step_mode(active)

            if sub._vectors['output']['linear']._alloc_complex:
                sub._vectors['output']['linear'].set_complex_step_mode(active)
                sub._vectors['input']['linear'].set_complex_step_mode(active)
                sub._vectors['residual']['linear'].set_complex_step_mode(active)

                if sub.linear_solver:
                    sub.linear_solver._set_complex_step_mode(active)

                if sub.nonlinear_solver:
                    sub.nonlinear_solver._set_complex_step_mode(active)

                if sub._owns_approx_jac:
                    sub._jacobian.set_complex_step_mode(active)

                if sub._assembled_jac:
                    sub._assembled_jac.set_complex_step_mode(active)

    def _set_approx_mode(self, active):
        """
        Turn on or off approx mode flag.

        Recurses to turn on or off approx mode flag in all subsystems.

        Parameters
        ----------
        active : bool
            Approx mode flag; set to True prior to commencing approximation.
        """
        for sub in self.system_iter(include_self=True, recurse=True):
            sub.under_approx = active

    def cleanup(self):
        """
        Clean up resources prior to exit.
        """
        # shut down all recorders
        self._rec_mgr.shutdown()

        # do any required cleanup on solvers
        if self._nonlinear_solver:
            self._nonlinear_solver.cleanup()
        if self._linear_solver:
            self._linear_solver.cleanup()

    def _get_partials_varlists(self):
        """
        Get lists of 'of' and 'wrt' variables that form the partial jacobian.

        Returns
        -------
        tuple(list, list)
            'of' and 'wrt' variable lists.
        """
        of = list(self._var_allprocs_prom2abs_list['output'])
        wrt = list(self._var_allprocs_prom2abs_list['input'])

        # wrt should include implicit states
        return of, of + wrt

    def _get_partials_var_sizes(self):
        """
        Get sizes of 'of' and 'wrt' variables that form the partial jacobian.

        Returns
        -------
        tuple(ndarray, ndarray, is_implicit)
            'of' and 'wrt' variable sizes.
        """
        iproc = self.comm.rank
        out_sizes = self._var_sizes['nonlinear']['output'][iproc]
        in_sizes = self._var_sizes['nonlinear']['input'][iproc]
        return out_sizes, np.hstack((out_sizes, in_sizes))

    def _get_gradient_nl_solver_systems(self):
        """
        Return a set of all Systems, including this one, that have a gradient nonlinear solver.

        Returns
        -------
        set
            Set of Systems containing nonlinear solvers that compute gradients.
        """
        return set(s for s in self.system_iter(include_self=True, recurse=True)
                   if s.nonlinear_solver and s.nonlinear_solver.supports['gradients'])

    def _jac_var_info_abs2prom(self, var_info):
        """
        Return a new list with tuples' [0] entry converted from absolute to promoted names.

        Parameters
        ----------
        var_info : list of (name, offset, end, idxs)
            The list that uses absolute names.

        Returns
        -------
        list
            The new list with promoted names.
        """
        new_list = []
        abs2prom_in = self._var_allprocs_abs2prom['input']
        abs2prom_out = self._var_allprocs_abs2prom['output']
        for abs_name, offset, end, idxs in var_info:
            if abs_name in abs2prom_out:
                new_list.append((abs2prom_out[abs_name], offset, end, idxs))
            else:
                new_list.append((abs2prom_in[abs_name], offset, end, idxs))
        return new_list

    def _abs_get_val(self, abs_name, get_remote=False, rank=None, vec_name=None, kind=None,
                     flat=False, from_root=False):
        """
        Return the value of the variable specified by the given absolute name.

        Parameters
        ----------
        abs_name : str
            The absolute name of the variable.
        get_remote : bool or None
            If True, return the value even if the variable is remote. NOTE: This function must be
            called in all procs in the Problem's MPI communicator.
            If False, only retrieve the value if it is on the current process, or only the part
            of the value that's on the current process for a distributed variable.
            If None and the variable is remote or distributed, a RuntimeError will be raised.
        rank : int or None
            If not None, specifies that the value is to be gathered to the given rank only.
            Otherwise, if get_remote is specified, the value will be broadcast to all procs
            in the MPI communicator.
        vec_name : str
            Name of the vector to use.
        kind : str or None
            Kind of variable ('input', 'output', or 'residual').  If None, returned value
            will be either an input or output.
        flat : bool
            If True, return the flattened version of the value.
        from_root : bool
            If True, resolve variables from top level scope.

        Returns
        -------
        object or None
            The value of the requested output/input/resid variable.  None if variable is not found.
        """
        discrete = distrib = False
        val = _UNDEFINED
        if from_root:
            all_meta = self._problem_meta['model_ref']()._var_allprocs_abs2meta
            my_meta = self._problem_meta['model_ref']()._var_abs2meta
            io = 'output' if abs_name in all_meta['output'] else 'input'
            all_meta = all_meta[io]
            my_meta = my_meta[io]
        else:
            io = 'output' if abs_name in self._var_allprocs_abs2meta['output'] else 'input'
            all_meta = self._var_allprocs_abs2meta[io]
            my_meta = self._var_abs2meta[io]

        # if abs_name is non-discrete it should be found in all_meta
        if abs_name in all_meta:
            if get_remote:
                meta = all_meta[abs_name]
                distrib = meta['distributed']
            elif self.comm.size > 1:
                vars_to_gather = self._problem_meta['vars_to_gather']
                if abs_name in vars_to_gather and vars_to_gather[abs_name] != self.comm.rank:
                    raise RuntimeError(f"{self.msginfo}: Variable '{abs_name}' is not local to "
                                       f"rank {self.comm.rank}. You can retrieve values from "
                                       "other processes using `get_val(<name>, get_remote=True)`.")

                meta = my_meta[abs_name]
                distrib = meta['distributed']
                if distrib and get_remote is None:
                    raise RuntimeError(f"{self.msginfo}: Variable '{abs_name}' is a distributed "
                                       "variable. You can retrieve values from all processes "
                                       "using `get_val(<name>, get_remote=True)` or from the "
                                       "local process using `get_val(<name>, get_remote=False)`.")
        else:
            discrete = True
            relname = abs_name[len(self.pathname) + 1:] if self.pathname else abs_name
            if relname in self._discrete_outputs:
                val = self._discrete_outputs[relname]
            elif relname in self._discrete_inputs:
                val = self._discrete_inputs[relname]
            elif abs_name in self._var_allprocs_discrete['output']:
                pass  # non-local discrete output
            elif abs_name in self._var_allprocs_discrete['input']:
                pass  # non-local discrete input
            elif get_remote:
                raise ValueError(f"{self.msginfo}: Can't find variable named '{abs_name}'.")
            else:
                return _UNDEFINED

        typ = 'output' if abs_name in self._var_allprocs_abs2prom['output'] else 'input'
        if kind is None:
            kind = typ
        if vec_name is None:
            vec_name = 'nonlinear'

        if not discrete:
            try:
                vec = self._vectors[kind][vec_name]
            except KeyError:
                if abs_name in my_meta:
                    if vec_name != 'nonlinear':
                        raise ValueError(f"{self.msginfo}: Can't get variable named '{abs_name}' "
                                         "because linear vectors are not available before "
                                         "final_setup.")
                    val = my_meta[abs_name]['value']
            else:
                if from_root:
                    vec = vec._root_vector
                if vec._contains_abs(abs_name):
                    val = vec._abs_get_val(abs_name, flat)

        if get_remote and self.comm.size > 1:
            owner = self._owning_rank[abs_name]
            myrank = self.comm.rank
            if rank is None:   # bcast
                if distrib:
                    idx = self._var_allprocs_abs2idx[vec_name][abs_name]
                    sizes = self._var_sizes[vec_name][typ][:, idx]
                    # TODO: could cache these offsets
                    offsets = np.zeros(sizes.size, dtype=INT_DTYPE)
                    offsets[1:] = np.cumsum(sizes[:-1])
                    loc_val = val if val is not _UNDEFINED else np.zeros(sizes[myrank])
                    val = np.zeros(np.sum(sizes))
                    self.comm.Allgatherv(loc_val, [val, sizes, offsets, MPI.DOUBLE])
                    if not flat:
                        val.shape = meta['global_shape'] if get_remote else meta['shape']
                else:
                    if owner != self.comm.rank:
                        val = None
                    # TODO: use Bcast if not discrete for speed
                    new_val = self.comm.bcast(val, root=owner)
                    val = new_val
            else:   # retrieve to rank
                if distrib:
                    idx = self._var_allprocs_abs2idx[vec_name][abs_name]
                    sizes = self._var_sizes[vec_name][typ][:, idx]
                    # TODO: could cache these offsets
                    offsets = np.zeros(sizes.size, dtype=INT_DTYPE)
                    offsets[1:] = np.cumsum(sizes[:-1])
                    loc_val = val if val is not _UNDEFINED else np.zeros(sizes[idx])
                    val = np.zeros(np.sum(sizes))
                    self.comm.Gatherv(loc_val, [val, sizes, offsets, MPI.DOUBLE], root=rank)
                    if not flat:
                        val.shape = meta['global_shape'] if get_remote else meta['shape']
                else:
                    if rank != owner:
                        tag = self._var_allprocs_abs2idx[vec_name][abs_name]
                        # avoid tag collisions between inputs, outputs, and resids
                        if kind != 'output':
                            tag += len(self._var_allprocs_abs2meta['output'])
                            if kind == 'residual':
                                tag += len(self._var_allprocs_abs2meta['input'])
                        if self.comm.rank == owner:
                            self.comm.send(val, dest=rank, tag=tag)
                        elif self.comm.rank == rank:
                            val = self.comm.recv(source=owner, tag=tag)

        return val

    def get_val(self, name, units=None, indices=None, get_remote=False, rank=None,
                vec_name='nonlinear', kind=None, flat=False, from_src=True):
        """
        Get an output/input/residual variable.

        Function is used if you want to specify display units.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the root system's namespace.
        units : str, optional
            Units to convert to before return.
        indices : int or list of ints or tuple of ints or int ndarray or Iterable or None, optional
            Indices or slice to return.
        get_remote : bool or None
            If True, retrieve the value even if it is on a remote process.  Note that if the
            variable is remote on ANY process, this function must be called on EVERY process
            in the Problem's MPI communicator.
            If False, only retrieve the value if it is on the current process, or only the part
            of the value that's on the current process for a distributed variable.
            If None and the variable is remote or distributed, a RuntimeError will be raised.
        rank : int or None
            If not None, only gather the value to this rank.
        vec_name : str
            Name of the vector to use.   Defaults to 'nonlinear'.
        kind : str or None
            Kind of variable ('input', 'output', or 'residual').  If None, returned value
            will be either an input or output.
        flat : bool
            If True, return the flattened version of the value.
        from_src : bool
            If True, retrieve value of an input variable from its connected source.

        Returns
        -------
        object
            The value of the requested output/input variable.
        """
        abs_names = name2abs_names(self, name)
        if not abs_names:
            raise KeyError('{}: Variable "{}" not found.'.format(self.msginfo, name))

        conns = self._problem_meta['model_ref']()._conn_global_abs_in2out
        if from_src and abs_names[0] in conns:  # pull input from source
            src = conns[abs_names[0]]
            if src in self._var_allprocs_abs2prom['output']:
                caller = self
            else:
                # src is outside of this system so get the value from the model
                caller = self._problem_meta['model_ref']()
            return caller._get_input_from_src(name, abs_names, conns, units=units, indices=indices,
                                              get_remote=get_remote, rank=rank,
                                              vec_name='nonlinear', flat=flat, scope_sys=self)
        else:
            val = self._abs_get_val(abs_names[0], get_remote, rank, vec_name, kind, flat)

            if indices is not None:
                val = val[indices]

            if units is not None:
                val = self.convert2units(abs_names[0], val, units)

        return val

    def _get_input_from_src(self, name, abs_ins, conns, units=None, indices=None,
                            get_remote=False, rank=None, vec_name='nonlinear', flat=False,
                            scope_sys=None):
        """
        Given an input name, retrieve the value from its source output.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the root system's namespace.
        abs_ins : list of str
            List of absolute input names.
        conns : dict
            Mapping of absolute names of each input to its connected output across the whole model.
        units : str, optional
            Units to convert to before return.
        indices : int or list of ints or tuple of ints or int ndarray or Iterable or None, optional
            Indices or slice to return.
        get_remote : bool
            If True, retrieve the value even if it is on a remote process.  Note that if the
            variable is remote on ANY process, this function must be called on EVERY process
            in the Problem's MPI communicator.
            If False, only retrieve the value if it is on the current process, or only the part
            of the value that's on the current process for a distributed variable.
            If None and the variable is remote or distributed, a RuntimeError will be raised.
        rank : int or None
            If not None, only gather the value to this rank.
        vec_name : str
            Name of the vector to use.   Defaults to 'nonlinear'.
        flat : bool
            If True, return the flattened version of the value.
        scope_sys : <System> or None
            If not None, the System where the original get_val was called.  This situation
            happens when get_val is called on an input, and the source connected to that input
            resides in a different scope.

        Returns
        -------
        object
            The value of the requested variable.
        """
        abs_name = abs_ins[0]
        src = conns[abs_name]
        if src in self._var_allprocs_discrete['output']:
            return self._abs_get_val(src, get_remote, rank, vec_name, 'output', flat,
                                     from_root=True)

        if scope_sys is None:
            scope_sys = self

        # if we have multiple promoted inputs that are explicitly connected to an output and units
        # have not been specified, look for group input to disambiguate
        if units is None and len(abs_ins) > 1:
            if abs_name not in self._var_allprocs_discrete['input']:
                # can't get here unless Group because len(abs_ins) always == 1 for comp
                try:
                    units = scope_sys._group_inputs[name][0]['units']
                except (KeyError, IndexError):
                    unit0 = self._var_allprocs_abs2meta['input'][abs_ins[0]]['units']
                    for n in abs_ins[1:]:
                        if unit0 != self._var_allprocs_abs2meta['input'][n]['units']:
                            self._show_ambiguity_msg(name, ('units',), abs_ins)
                            break

        if abs_name in self._var_abs2meta['input']:  # input is local
            vmeta = self._var_abs2meta['input'][abs_name]
            src_indices = vmeta['src_indices']
        else:
            vmeta = self._var_allprocs_abs2meta['input'][abs_name]
            src_indices = None  # FIXME: remote var could have src_indices

        distrib = vmeta['distributed']
        vshape = vmeta['shape']
        has_src_indices = any(self._var_allprocs_abs2meta['input'][n]['has_src_indices']
                              for n in abs_ins)

        # see if we have any 'intermediate' level src_indices when using a promoted name
        if name in scope_sys._var_prom2inds:
            src_shape, inds, flat = scope_sys._var_prom2inds[name]
            if inds is None:
                if len(abs_ins) > 1 or name != abs_ins[0]:  # using a promoted lookup
                    src_indices = None
                    vshape = None
                    has_src_indices = False
                is_slice = _is_slicer_op(src_indices)
            else:
                is_slice = _is_slicer_op(inds)
                shp = shape_from_idx(src_shape, inds, flat)
                if not flat and not _is_slicer_op(inds):
                    inds = _flatten_src_indices(inds, shp,
                                                src_shape, np.product(src_shape))
                src_indices = inds
                has_src_indices = True
                if len(abs_ins) > 1 or name != abs_name:
                    vshape = shp
        else:
            is_slice = _is_slicer_op(src_indices)
            shpname = 'global_shape' if get_remote else 'shape'
            src_shape = self._var_allprocs_abs2meta['output'][src][shpname]

        model_ref = self._problem_meta['model_ref']()
        smeta = model_ref._var_allprocs_abs2meta['output'][src]
        sdistrib = smeta['distributed']
        slocal = src in model_ref._var_abs2meta['output']

        if self.comm.size > 1:
            if distrib and get_remote is None:
                raise RuntimeError(f"{self.msginfo}: Variable '{abs_name}' is a distributed "
                                   "variable. You can retrieve values from all processes "
                                   "using `get_val(<name>, get_remote=True)` or from the "
                                   "local process using `get_val(<name>, get_remote=False)`.")

            if sdistrib and not distrib and not get_remote:
                raise RuntimeError(f"{self.msginfo}: Non-distributed variable '{abs_name}' has "
                                   f"a distributed source, '{src}', so you must retrieve its value "
                                   "using 'get_remote=True'.")

        # get value of the source
        val = self._abs_get_val(src, get_remote, rank, vec_name, 'output', flat, from_root=True)

        if has_src_indices:
            if src_indices is None:  # input is remote
                val = np.zeros(0)
            else:
                if is_slice:
                    val.shape = src_shape
                    val = val[tuple(src_indices)].ravel()
                elif distrib and (sdistrib or not slocal) and not get_remote:
                    var_idx = self._var_allprocs_abs2idx[vec_name][src]
                    # sizes for src var in each proc
                    sizes = self._var_sizes[vec_name]['output'][:, var_idx]
                    start = np.sum(sizes[:self.comm.rank])
                    end = start + sizes[self.comm.rank]
                    if np.all(np.logical_and(src_indices >= start, src_indices < end)):
                        if src_indices.size > 0:
                            src_indices = src_indices - np.min(src_indices)
                        val = val.ravel()[src_indices]
                        fail = 0
                    else:
                        fail = 1
                    if self.comm.allreduce(fail) > 0:
                        raise RuntimeError(f"{self.msginfo}: Can't retrieve distributed variable "
                                           f"'{abs_name}' because its src_indices reference "
                                           "entries from other processes. You can retrieve values "
                                           "from all processes using "
                                           "`get_val(<name>, get_remote=True)`.")
                else:
                    val = val.ravel()[src_indices]

            if get_remote and self.comm.size > 1:
                if distrib:
                    if rank is None:
                        parts = self.comm.allgather(val)
                        parts = [p for p in parts if p.size > 0]
                        val = np.hstack(parts)
                    else:
                        parts = self.comm.gather(val, root=rank)
                        if rank == self.comm.rank:
                            parts = [p for p in parts if p.size > 0]
                            val = np.hstack(parts)
                        else:
                            val = None
                else:  # non-distrib input
                    if self.comm.rank == self._owning_rank[abs_name]:
                        self.comm.bcast(val, root=self.comm.rank)
                    else:
                        val = self.comm.bcast(None, root=self._owning_rank[abs_name])

            if distrib and get_remote:
                val.shape = self._var_allprocs_abs2meta['input'][abs_name]['global_shape']
            elif not flat and val.size > 0:
                val.shape = vshape
        elif vshape is not None:
            val = val.reshape(vshape)

        if indices is not None:
            val = val[indices]

        if units is not None:
            if smeta['units'] is not None:
                try:
                    val = self.convert2units(src, val, units)
                except TypeError:  # just call this to get the right error message
                    self.convert2units(abs_name, val, units)
            else:
                val = self.convert2units(abs_name, val, units)
        elif (vmeta['units'] is not None and smeta['units'] is not None and
                vmeta['units'] != smeta['units']):
            val = self.convert2units(src, val, vmeta['units'])

        return val

    def _get_src_inds_array(self, varname):
        """
        Return src_indices, if any, for absolute input 'varname', converting from slice if needed.

        Parameters
        ----------
        varname : str
            Absolute name of the input variable.

        Returns
        -------
        ndarray or None
            The value of src_indices for the given input variable.
        """
        meta = self._var_abs2meta['input'][varname]
        src_indices = meta['src_indices']
        if src_indices is not None:
            src_slice = meta['src_slice']
            # if src_indices is still a slice, update it to an array
            if src_slice is src_indices:
                model = self._problem_meta['model_ref']()
                src = model._conn_global_abs_in2out[varname]
                try:
                    global_size = model._var_allprocs_abs2meta['output'][src]['global_size']
                    global_shape = model._var_allprocs_abs2meta['output'][src]['global_shape']
                except KeyError:
                    raise RuntimeError(f"{self.msginfo}: Can't compute src_indices array from "
                                       f"src_slice for input '{varname}' because we don't know "
                                       "the global shape of its source yet.")
                src_indices = _slice_indices(src_slice, global_size, global_shape)

                meta['src_indices'] = src_indices  # store converted value

        return src_indices

    def _retrieve_data_of_kind(self, filtered_vars, kind, vec_name, parallel=False):
        """
        Retrieve variables, either local or remote, in the filtered_vars list.

        Parameters
        ----------
        filtered_vars : dict
            Dictionary containing entries for 'input', 'output', and/or 'residual'.
        kind : str
            Either 'input', 'output', or 'residual'.
        vec_name : str
            Either 'nonlinear' or 'linear'.
        parallel : bool
            If True, recorders are parallel, so only local values should be saved in each proc.

        Returns
        -------
        dict
            Variable values keyed on absolute name.
        """
        prom2abs_in = self._var_allprocs_prom2abs_list['input']
        conns = self._problem_meta['model_ref']()._conn_global_abs_in2out
        vdict = {}
        variables = filtered_vars.get(kind)
        if variables:
            vec = self._vectors[kind][vec_name]
            rank = self.comm.rank
            discrete_vec = () if kind == 'residual' else self._var_discrete[kind]
            offset = len(self.pathname) + 1 if self.pathname else 0

            if self.comm.size == 1:
                get = vec._abs_get_val
                srcget = self._vectors['output'][vec_name]._abs_get_val
                vdict = {}
                if discrete_vec:
                    for n in variables:
                        if vec._contains_abs(n):
                            vdict[n] = get(n, False)
                        elif n[offset:] in discrete_vec:
                            vdict[n] = discrete_vec[n[offset:]]['value']
                        else:
                            ivc_path = conns[prom2abs_in[n][0]]
                            if vec._contains_abs(ivc_path):
                                vdict[ivc_path] = srcget(ivc_path, False)
                            elif ivc_path[offset:] in discrete_vec:
                                vdict[ivc_path] = discrete_vec[ivc_path[offset:]]['value']
                else:
                    for name in variables:
                        if vec._contains_abs(name):
                            vdict[name] = get(name, False)
                        else:
                            ivc_path = conns[prom2abs_in[name][0]]
                            vdict[ivc_path] = srcget(ivc_path, False)
            elif parallel:
                get = self._abs_get_val
                vdict = {}
                if discrete_vec:
                    for name in variables:
                        if vec._contains_abs(name):
                            vdict[name] = get(name, get_remote=True, rank=0,
                                              vec_name=vec_name, kind=kind)
                        elif name[offset:] in discrete_vec and self._owning_rank[name] == rank:
                            vdict[name] = discrete_vec[name[offset:]]['value']
                else:
                    for name in variables:
                        if vec._contains_abs(name):
                            vdict[name] = get(name, get_remote=True, rank=0,
                                              vec_name=vec_name, kind=kind)
                        else:
                            ivc_path = conns[prom2abs_in[name][0]]
                            vdict[name] = get(ivc_path, get_remote=True, rank=0,
                                              vec_name=vec_name, kind='output')
            else:
                io = 'input' if kind == 'input' else 'output'
                meta = self._var_allprocs_abs2meta[io]
                for name in variables:
                    if self._owning_rank[name] == 0 and not meta[name]['distributed']:
                        # if using a serial recorder and rank 0 owns the variable,
                        # use local value on rank 0 and do nothing on other ranks.
                        if rank == 0:
                            if vec._contains_abs(name):
                                vdict[name] = vec._abs_get_val(name, flat=False)
                            elif name[offset:] in discrete_vec:
                                vdict[name] = discrete_vec[name[offset:]]['value']
                    else:
                        vdict[name] = self.get_val(name, get_remote=True, rank=0,
                                                   vec_name=vec_name, kind=kind, from_src=False)

        return vdict

    def convert2units(self, name, val, units):
        """
        Convert the given value to the specified units.

        Parameters
        ----------
        name : str
            Name of the variable.
        val : float or ndarray of float
            The value of the variable.
        units : str
            The units to convert to.

        Returns
        -------
        float or ndarray of float
            The value converted to the specified units.
        """
        base_units = self._get_var_meta(name, 'units')

        if base_units == units:
            return val

        try:
            scale, offset = unit_conversion(base_units, units)
        except Exception:
            msg = "{}: Can't express variable '{}' with units of '{}' in units of '{}'."
            raise TypeError(msg.format(self.msginfo, name, base_units, units))

        return (val + offset) * scale

    def convert_from_units(self, name, val, units):
        """
        Convert the given value from the specified units to those of the named variable.

        Parameters
        ----------
        name : str
            Name of the variable.
        val : float or ndarray of float
            The value of the variable.
        units : str
            The units to convert to.

        Returns
        -------
        float or ndarray of float
            The value converted to the specified units.
        """
        base_units = self._get_var_meta(name, 'units')

        if base_units == units:
            return val

        try:
            scale, offset = unit_conversion(units, base_units)
        except Exception:
            msg = "{}: Can't express variable '{}' with units of '{}' in units of '{}'."
            raise TypeError(msg.format(self.msginfo, name, base_units, units))

        return (val + offset) * scale

    def convert_units(self, name, val, units_from, units_to):
        """
        Wrap the utility convert_units and give a good error message.

        Parameters
        ----------
        name : str
            Name of the variable.
        val : float or ndarray of float
            The value of the variable.
        units_from : str
            The units to convert from.
        units_to : str
            The units to convert to.

        Returns
        -------
        float or ndarray of float
            The value converted to the specified units.
        """
        if units_from == units_to:
            return val

        try:
            scale, offset = unit_conversion(units_from, units_to)
        except Exception:
            raise TypeError(f"{self.msginfo}: Can't set variable '{name}' with units "
                            f"'{units_from}' to value with units '{units_to}'.")

        return (val + offset) * scale

    def _get_var_meta(self, name, key):
        """
        Get metadata for a variable.

        Parameters
        ----------
        name : str
            Variable name (promoted, relative, or absolute) in the root system's namespace.
        key : str
            Key into the metadata dict for the given variable.

        Returns
        -------
        object
            The value stored under key in the metadata dictionary for the named variable.
        """
        if self._problem_meta is not None:
            meta_all = self._problem_meta['model_ref']()._var_allprocs_abs2meta
            meta_loc = self._problem_meta['model_ref']()._var_abs2meta
        else:
            meta_all = self._var_allprocs_abs2meta
            meta_loc = self._var_abs2meta

        meta = None
        if name in meta_all['output']:
            abs_name = name
            meta = meta_all['output'][name]
        elif name in meta_all['input']:
            abs_name = name
            meta = meta_all['input'][name]

        if meta is None:
            abs_name = name2abs_name(self, name)
            if abs_name is not None:
                if abs_name in meta_all['output']:
                    meta = meta_all['output'][abs_name]
                elif abs_name in meta_all['input']:
                    meta = meta_all['input'][abs_name]

        if meta:
            if key in meta:
                return meta[key]
            else:
                # key is either bogus or a key into the local metadata dict
                # (like 'value' or 'src_indices'). If MPI is active, this val may be remote
                # on some procs
                if self.comm.size > 1 and abs_name in self._vars_to_gather:
                    # TODO: fix this
                    # cause a failure in all procs to avoid a hang
                    raise RuntimeError(f"{self.msgifo}: No support yet for retrieving local "
                                       f"metadata key '{key}' from a remote proc.")
                elif abs_name in meta_loc['output']:
                    try:
                        return meta_loc['output'][abs_name][key]
                    except KeyError:
                        raise KeyError(f"{self.msginfo}: Metadata key '{key}' not found for "
                                       f"variable '{name}'.")
                elif abs_name in meta_loc['input']:
                    try:
                        return meta_loc['input'][abs_name][key]
                    except KeyError:
                        raise KeyError(f"{self.msginfo}: Metadata key '{key}' not found for "
                                       f"variable '{name}'.")

        if abs_name is not None:
            if abs_name in self._var_allprocs_discrete['output']:
                meta = self._var_allprocs_discrete['output'][abs_name]
            elif abs_name in self._var_allprocs_discrete['input']:
                meta = self._var_allprocs_discrete['input'][abs_name]

            if meta and key in meta:
                return meta[key]

            rel_idx = len(self.pathname) + 1 if self.pathname else 0
            relname = abs_name[rel_idx:]
            if relname in self._var_discrete['output']:
                meta = self._var_discrete['output'][relname]
            elif relname in self._var_discrete['input']:
                meta = self._var_discrete['input'][relname]

            if meta:
                try:
                    return meta[key]
                except KeyError:
                    raise KeyError(f"{self.msginfo}: Metadata key '{key}' not found for "
                                   f"variable '{name}'.")

        raise KeyError(f"{self.msginfo}: Metadata for variable '{name}' not found.")

    def _resolve_ambiguous_input_meta(self):
        pass

    def get_relevant_vars(self, desvars, responses, mode):
        """
        Find all relevant vars between desvars and responses.

        Both vars are assumed to be outputs (either design vars or responses).

        Parameters
        ----------
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
        conns = self._conn_global_abs_in2out
        relevant = defaultdict(dict)

        # Create a hybrid graph with components and all connected vars.  If a var is connected,
        # also connect it to its corresponding component.
        graph = nx.DiGraph()
        for tgt, src in conns.items():
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
                parts = dv.rsplit('.', 1)
                if len(parts) == 1:
                    system = ''  # this happens when a component is the model
                    graph.add_edge(dv, system)
                else:
                    system = parts[0]
                    graph.add_edge(system, dv)

        for res in responses:
            if res not in graph:
                graph.add_node(res, type_='out')
                parts = res.rsplit('.', 1)
                if len(parts) == 1:
                    system = ''  # this happens when a component is the model
                else:
                    system = parts[0]
                graph.add_edge(system, res)

        nodes = graph.nodes
        grev = graph.reverse(copy=False)
        dvcache = {}
        rescache = {}

        for desvar in desvars:
            if desvar not in dvcache:
                dvcache[desvar] = set(all_connected_nodes(graph, desvar))

            for response in responses:
                if response not in rescache:
                    rescache[response] = set(all_connected_nodes(grev, response))

                common = dvcache[desvar].intersection(rescache[response])

                if common:
                    input_deps = set()
                    output_deps = set()
                    sys_deps = set()
                    for node in common:
                        if 'type_' in nodes[node]:
                            typ = nodes[node]['type_']
                            parts = node.rsplit('.', 1)
                            if len(parts) == 1:
                                system = ''
                            else:
                                system = parts[0]
                            if typ == 'in':  # input var
                                input_deps.add(node)
                                if system not in sys_deps:
                                    sys_deps.update(all_ancestors(system))
                            else:  # output var
                                output_deps.add(node)
                                if system not in sys_deps:
                                    sys_deps.update(all_ancestors(system))

                elif desvar == response:
                    input_deps = set()
                    output_deps = set([response])
                    parts = desvar.rsplit('.', 1)
                    if len(parts) == 1:
                        s = ''
                    else:
                        s = parts[0]
                    sys_deps = set(all_ancestors(s))

                if common or desvar == response:
                    if desvar in conns:
                        desvar = conns[desvar]
                    if response in conns:
                        response = conns[response]
                    if mode != 'rev':  # fwd or auto
                        relevant[desvar][response] = ({'input': input_deps,
                                                       'output': output_deps}, sys_deps)
                    if mode != 'fwd':  # rev or auto
                        relevant[response][desvar] = ({'input': input_deps,
                                                       'output': output_deps}, sys_deps)

                    sys_deps.add('')  # top level Group is always relevant

        voi_lists = []
        if mode != 'rev':
            voi_lists.append((desvars, responses))
        if mode != 'fwd':
            voi_lists.append((responses, desvars))

        # now calculate dependencies between each VOI and all other VOIs of the
        # other type, e.g for each input VOI wrt all output VOIs.  This is only
        # done for design vars in fwd mode or responses in rev mode. In auto mode,
        # we combine the results for fwd and rev modes.
        for inputs, outputs in voi_lists:
            for inp in inputs:
                if inp in conns:
                    inp = conns[inp]
                relinp = relevant[inp]
                if relinp:
                    if '@all' in relinp:
                        dct, total_systems = relinp['@all']
                        total_inps = dct['input']
                        total_outs = dct['output']
                    else:
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
