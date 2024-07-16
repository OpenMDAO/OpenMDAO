"""Define the base System class."""
import sys
import os
import hashlib
import time
import functools

from contextlib import contextmanager
from collections import defaultdict
from itertools import chain
from enum import IntEnum

from fnmatch import fnmatchcase

from numbers import Integral

import numpy as np

from openmdao.core.constants import _DEFAULT_OUT_STREAM, _UNDEFINED, INT_DTYPE, INF_BOUND, \
    _SetupStatus
from openmdao.jacobians.jacobian import Jacobian
from openmdao.jacobians.assembled_jacobian import DenseJacobian, CSCJacobian
from openmdao.recorders.recording_manager import RecordingManager
from openmdao.vectors.vector import _full_slice
from openmdao.utils.mpi import MPI, multi_proc_exception_check
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.record_util import create_local_meta, check_path, has_match
from openmdao.utils.units import is_compatible, unit_conversion, simplify_unit
from openmdao.utils.variable_table import write_var_table, NA
from openmdao.utils.array_utils import evenly_distrib_idxs, shape_to_len
from openmdao.utils.name_maps import name2abs_name, name2abs_names
from openmdao.utils.coloring import _compute_coloring, Coloring, \
    _STD_COLORING_FNAME, _DEF_COMP_SPARSITY_ARGS, _ColSparsityJac
import openmdao.utils.coloring as coloring_mod
from openmdao.utils.indexer import indexer
from openmdao.utils.om_warnings import issue_warning, \
    DerivativesWarning, PromotionWarning, UnusedOptionWarning, UnitsWarning, warn_deprecation
from openmdao.utils.general_utils import determine_adder_scaler, \
    format_as_float_or_array, all_ancestors, match_prom_or_abs, \
    ensure_compatible, env_truthy, make_traceback, _is_slicer_op
from openmdao.approximation_schemes.complex_step import ComplexStep
from openmdao.approximation_schemes.finite_difference import FiniteDifference

_empty_frozen_set = frozenset()

_asm_jac_types = {
    'csc': CSCJacobian,
    'dense': DenseJacobian,
}

# Suppored methods for derivatives
_supported_methods = {
    'fd': FiniteDifference,
    'cs': ComplexStep,
    'exact': None,
    'jax': None
}

_DEFAULT_COLORING_META = {
    'wrt_patterns': ('*',),  # patterns used to match wrt variables
    'method': 'fd',  # finite differencing method  ('fd' or 'cs')
    'wrt_matches': None,  # where matched wrt names are stored
    'per_instance': True,  # assume each instance can have a different coloring
    'coloring': None,  # this will contain the actual Coloring object
    'dynamic': False,  # True if dynamic coloring is being used
    'static': None,  # either _STD_COLORING_FNAME, a filename, or a Coloring object
    # if use_fixed_coloring was called
}

_DEFAULT_COLORING_META.update(_DEF_COMP_SPARSITY_ARGS)

_recordable_funcs = frozenset(['_apply_linear', '_apply_nonlinear', '_solve_linear',
                               '_solve_nonlinear'])

# the following are local metadata that will also be accessible for vars on all procs
global_meta_names = {
    'input': ('units', 'shape', 'size', 'distributed', 'tags', 'desc', 'shape_by_conn',
              'compute_shape', 'copy_shape'),
    'output': ('units', 'shape', 'size', 'desc',
               'ref', 'ref0', 'res_ref', 'distributed', 'lower', 'upper', 'tags', 'shape_by_conn',
               'compute_shape', 'copy_shape'),
}

allowed_meta_names = {
    'val',
    'global_shape',
    'global_size',
    'src_indices',
    'flat_src_indices',
    'type',
    'res_units',
}
allowed_meta_names.update(global_meta_names['input'])
allowed_meta_names.update(global_meta_names['output'])

resp_size_checks = {
    'con': ['ref', 'ref0', 'scaler', 'adder', 'upper', 'lower', 'equals'],
    'obj': ['ref', 'ref0', 'scaler', 'adder']
}
resp_types = {'con': 'constraint', 'obj': 'objective'}


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


class _OptStatus(IntEnum):
    """
    Class used to define different states during the optimization process.

    Attributes
    ----------
    PRE : int
        Before the optimization.
    OPTIMIZING : int
        During the optimization.
    POST : int
        After the optimization.
    """

    PRE = 0
    OPTIMIZING = 1
    POST = 2


def collect_errors(method):
    """
    Decorate a method so that it will collect any exceptions for later display.

    Parameters
    ----------
    method : method
        The method to be decorated.

    Returns
    -------
    method
        The wrapped method.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            if env_truthy('OPENMDAO_FAIL_FAST'):
                raise

            type_exc, exc, tb = sys.exc_info()
            if isinstance(exc, KeyError) and self._get_saved_errors():
                # it's likely the result of an earlier error, so ignore it
                return

            self._collect_error(str(exc), exc_type=type_exc, tback=tb)

    return wrapper


class System(object):
    """
    Base class for all systems in OpenMDAO.

    Never instantiated; subclassed by <Group> or <Component>.

    In attribute names:
        abs: absolute, unpromoted variable name, seen from root (unique).
        rel: relative, unpromoted variable name, seen from current system (unique).
        prom: relative, promoted variable name, seen from current system (non-unique for inputs).

    Parameters
    ----------
    num_par_fd : int
        If FD is active, number of concurrent FD solves.
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the System options.

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
    under_finite_difference : bool
        When True, this system is undergoing finite differencing.
    iter_count : int
        Counts the number of times this system has called _solve_nonlinear. This also
        corresponds to the number of times that the system's outputs are recorded if a recorder
        is present.
    iter_count_apply : int
        Counts the number of times the system has called _apply_nonlinear. For ExplicitComponent,
        calls to apply_nonlinear also call compute, so number of executions can be found by adding
        this and iter_count together. Recorders do not record calls to apply_nonlinear.
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
    _subsystems_allprocs : dict
        Dict mapping subsystem name to SysInfo(system, index) for children of this system.
    _subsystems_myproc : [<System>, ...]
        List of local subsystems that exist on this proc.
    _var_promotes : { 'any': [], 'input': [], 'output': [] }
        Dictionary of lists of variable names/wildcards specifying promotion
        (used to calculate promoted names)
    _var_prom2inds : dict
        Maps promoted name to src_indices in scope of system.
    _var_allprocs_prom2abs_list : {'input': dict, 'output': dict}
        Dictionary mapping promoted names (continuous and discrete) to list of all absolute names.
        For outputs, the list will have length one since promoted output names are unique.
    _var_abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names, on current proc. Contains continuous
        and discrete variables.
    _var_allprocs_abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names, on all procs.  Contains continuous
        and discrete variables.
    _var_allprocs_abs2meta : dict
        Dictionary mapping absolute names to metadata dictionaries for allprocs continuous
        variables.
    _var_abs2meta : dict
        Dictionary mapping absolute names to metadata dictionaries for myproc continuous variables.
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
    _var_sizes : {'input': ndarray, 'output': ndarray}
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
        connections or a top level Group that is used to compute total derivatives
        across multiple processes.
    _vars_to_gather : dict
        Contains names of non-distributed variables that are remote on at least one proc in the comm
    _conn_global_abs_in2out : {'abs_in': 'abs_out'}
        Dictionary containing all explicit & implicit connections (continuous and discrete)
        owned by this system or any descendant system. The data is the same across all processors.
    _vectors : {'input': dict, 'output': dict, 'residual': dict}
        Dictionaries of vectors keyed by vec_name.
    _inputs : <Vector>
        The nonlinear inputs vector.
    _outputs : <Vector>
        The nonlinear outputs vector.
    _residuals : <Vector>
        The nonlinear residuals vector.
    _dinputs : <Vector>
        The linear inputs vector.
    _doutputs : <Vector>
        The linear outputs vector.
    _dresiduals : <Vector>
        The linear residuals vector.
    _nonlinear_solver : <NonlinearSolver>
        Nonlinear solver to be used for solve_nonlinear.
    _linear_solver : <LinearSolver>
        Linear solver to be used for solve_linear; not the Newton system.
    _approx_schemes : dict
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
    _owns_approx_wrt : list or None
        Overrides aproximation inputs. This is set when calculating system derivatives, and serves
        as a way to communicate the driver's input quantities to the approximation objects so that
        we only take derivatives with respect to variables that the driver needs.
    _subjacs_info : dict of dict
        Sub-jacobian metadata for each (output, input) pair added using
        declare_partials. Members of each pair may be glob patterns.
    _approx_subjac_keys : list
        List of subjacobian keys used for approximated derivatives.
    _design_vars : dict of dict
        dict of all driver design vars added to the system.
    _responses : dict of dict
        dict of all driver responses added to the system.
    _rec_mgr : <RecordingManager>
        object that manages all recorders added to this system.
    _static_subsystems_allprocs : dict
        Dict of SysInfo(subsys, index) that stores all subsystems added outside of setup.
    _static_design_vars : dict of dict
        Driver design variables added outside of setup.
    _static_responses : dict of dict
        Driver responses added outside of setup.
    matrix_free : bool
        This is set to True if the component overrides the appropriate function with a user-defined
        matrix vector product with the Jacobian or any of its subsystems do. Note that the framework
        will not set the matrix_free flag correctly for Component instances having a matrix vector
        product function that is added dynamically (not declared as part of the class) and in that
        case the matrix_free flag must be set manually to True.
    _mode : str
        Indicates derivative direction for the model, either 'fwd' or 'rev'.
    _scope_cache : dict
        Cache for variables in the scope of various mat-vec products.
    _has_guess : bool
        True if this system has or contains a system with a `guess_nonlinear` method defined.
    _has_output_scaling : bool
        True if this system has output scaling.
    _has_output_adder : bool
        True if this system has scaling that includes an adder term.
    _has_resid_scaling : bool
        True if this system has resid scaling.
    _has_input_scaling : bool
        True if this system has input scaling.
    _has_input_adder : bool
        True if this system has scaling that includes an adder term.
    _has_bounds : bool
        True if this system has upper or lower bounds on outputs.
    _has_distrib_vars : bool
        If True, this System contains at least one distributed variable. Used to determine if a
        parallel group or distributed component is below a DirectSolver so that we can raise an
        exception.
    _owning_rank : dict
        Dict mapping var name to the lowest rank where that variable is local.
    _filtered_vars_to_record : Dict
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
    _tot_jac : __TotalJacInfo or None
        If a total jacobian is being computed and this is the top level System, this will
        be a reference to the _TotalJacInfo object.
    _saved_errors : list
        Temporary storage for any saved errors that occur before this System is assigned
        a parent Problem.
    _output_solver_options : dict or None
        Solver output options if set_output_solver_options has been called.
    _promotion_tree : dict
        Mapping of system path to promotion info indicating all subsystems where variables
        were promoted.
    _during_sparsity : bool
        If True, we're doing a sparsity computation and uncolored approxs need to be restricted
        to only colored columns.
    """

    def __init__(self, num_par_fd=1, **kwargs):
        """
        Initialize all attributes.
        """
        self.name = ''
        self.pathname = None
        self.comm = None
        self._is_local = False

        # System options
        self.options = OptionsDictionary(parent_name=type(self).__name__)

        self.options.declare('assembled_jac_type', values=['csc', 'dense'], default='csc',
                             desc='Linear solver(s) in this group or implicit component, '
                                  'if using an assembled jacobian, will use this type.')

        # Case recording options
        self.recording_options = OptionsDictionary(parent_name=type(self).__name__)
        self.recording_options.declare('record_inputs', types=bool, default=True,
                                       desc='Set to True to record inputs at the system level')
        self.recording_options.declare('record_outputs', types=bool, default=True,
                                       desc='Set to True to record outputs at the system level')
        self.recording_options.declare('record_residuals', types=bool, default=True,
                                       desc='Set to True to record residuals at the system level')
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

        self._vectors = {}

        self._inputs = None
        self._outputs = None
        self._residuals = None
        self._dinputs = None
        self._doutputs = None
        self._dresiduals = None
        self._discrete_inputs = None
        self._discrete_outputs = None

        self._nonlinear_solver = None
        self._linear_solver = None

        self._jacobian = None
        self._approx_schemes = {}
        self._subjacs_info = {}
        self._approx_subjac_keys = None
        self.matrix_free = _UNDEFINED

        self._owns_approx_jac = False
        self._owns_approx_jac_meta = {}
        self._owns_approx_wrt = None
        self._owns_approx_of = None

        self.under_complex_step = False
        self.under_finite_difference = False

        self._design_vars = {}
        self._responses = {}
        self._rec_mgr = RecordingManager()

        self._conn_global_abs_in2out = {}

        self._static_subsystems_allprocs = {}
        self._static_design_vars = {}
        self._static_responses = {}

        self._scope_cache = {}

        self._num_par_fd = num_par_fd

        self._declare_options()
        self.initialize()

        self.options.update(kwargs)

        self._has_guess = False
        self._has_output_scaling = False
        self._has_output_adder = False
        self._has_resid_scaling = False
        self._has_input_scaling = False
        self._has_input_adder = False
        self._has_bounds = False
        self._has_distrib_vars = False
        self._has_approx = False

        self._vector_class = None

        self._assembled_jac = None

        self._par_fd_id = 0

        self._filtered_vars_to_record = {}
        self._owning_rank = None
        self._coloring_info = coloring_mod.Partial_ColoringMeta()
        self._first_call_to_linearize = True  # will check in first call to _linearize
        self._tot_jac = None
        self._saved_errors = None if env_truthy('OPENMDAO_FAIL_FAST') else []

        self._output_solver_options = {}
        self._promotion_tree = None

        self._during_sparsity = False

    @property
    def under_approx(self):
        """
        Return True if under complex step or finite difference.

        Returns
        -------
        bool
            True if under CS or FD.
        """
        return self.under_complex_step or self.under_finite_difference

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
        return self.pathname if self.pathname is not None else ''

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

        Yields
        ------
        str
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

    def _jac_of_iter(self):
        """
        Iterate over (name, offset, end, slice, dist_sizes) for each 'of' (row) var in the jacobian.

        The slice is internal to the given variable in the result, and this is always a full
        slice except when indices are defined for the 'of' variable.

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
        toidx = self._var_allprocs_abs2idx
        sizes = self._var_sizes['output']
        total = self.pathname == ''
        szname = 'global_size' if total else 'size'
        start = end = 0
        for of, meta in self._var_abs2meta['output'].items():
            end += meta[szname]
            yield of, start, end, _full_slice, sizes[:, toidx[of]] if meta['distributed'] else None
            start = end

    def _jac_wrt_iter(self, wrt_matches=None):
        """
        Iterate over (name, offset, end, vec, slc, dist_sizes) for each column var in the jacobian.

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
        Vector or None
            Either the _outputs or _inputs vector if var is local else None.
        slice
            A full slice.
        ndarray or None
            Distributed sizes if var is distributed else None
        """
        toidx = self._var_allprocs_abs2idx
        sizes_in = self._var_sizes['input']

        tometa_in = self._var_allprocs_abs2meta['input']

        local_ins = self._var_abs2meta['input']
        local_outs = self._var_abs2meta['output']

        total = self.pathname == ''
        szname = 'global_size' if total else 'size'

        start = end = 0
        for of, _start, _end, _, dist_sizes in self._jac_of_iter():
            if wrt_matches is None or of in wrt_matches:
                end += (_end - _start)
                vec = self._outputs if of in local_outs else None
                yield of, start, end, vec, _full_slice, dist_sizes
                start = end

        for wrt, meta in self._var_abs2meta['input'].items():
            if wrt_matches is None or wrt in wrt_matches:
                end += meta[szname]
                vec = self._inputs if wrt in local_ins else None
                dist_sizes = sizes_in[:, toidx[wrt]] if tometa_in[wrt]['distributed'] else None
                yield wrt, start, end, vec, _full_slice, dist_sizes
                start = end

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.

        This is optionally implemented by subclasses of Component or Group
        that themselves are intended to be subclassed by the end user. The
        options of the intermediate class are declared here leaving the
        `initialize` method available for user-defined options.
        """
        pass

    def _have_output_solver_options_been_applied(self):
        """
        Check to see if the cached output solver options were applied.
        """
        for subsys in self.system_iter(include_self=True, recurse=True):
            if subsys._output_solver_options:  # If options dict not empty, has not been applied
                return False  # No need to look for more
        return True

    def set_output_solver_options(self, name, lower=_UNDEFINED, upper=_UNDEFINED,
                                  ref=_UNDEFINED, ref0=_UNDEFINED, res_ref=_UNDEFINED):
        """
        Set solver output options.

        Allows the user to set output solver options after the output has been defined and
        metadata set using the add_ouput method.

        Parameters
        ----------
        name : str
            Name of the variable in this system's namespace.
        lower : float or list or tuple or ndarray or None
            Lower bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no lower bound.
            Default is None.
        upper : float or list or tuple or ndarray or None
            Upper bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no upper bound.
            Default is None.
        ref : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 1. Default is 1.
        ref0 : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 0. Default is 0.
        res_ref : float
            Scaling parameter. The value in the user-defined res_units of this output's residual
            when the scaled value is 1. Default is None, which means residual scaling matches
            output scaling.
        """
        # Cache the solver options for use later in the setup process.
        # Since this can be called before setup, there is no way to update the
        # self._var_allprocs_abs2meta['output'] values since those have not been setup yet.
        # These values are applied in the System._apply_output_solver_options method
        # which is called in System._setup. That method is only called by the top model.

        output_solver_options = {}
        if lower is not _UNDEFINED:
            output_solver_options['lower'] = lower
        if upper is not _UNDEFINED:
            output_solver_options['upper'] = upper
        if ref is not _UNDEFINED:
            output_solver_options['ref'] = ref
        if ref0 is not _UNDEFINED:
            output_solver_options['ref0'] = ref0
        if res_ref is not _UNDEFINED:
            output_solver_options['res_ref'] = res_ref
        self._output_solver_options[name] = output_solver_options
        return

    def _apply_output_solver_options(self):
        """
        Apply the cached output solver options.

        Solver options can be set using the System.set_output_solver_options method.
        These cannot be set immediately when that method is called because not
        all the variables have been setup at the time a user could potentially want to call it.
        So they are cached so that they can be applied later in the setup process.
        They are applied in System._setup using this method.
        """
        # Loop through the output solver options that have been set on this System
        prefix = self.pathname + '.' if self.pathname else ''
        for name, options in self._output_solver_options.items():
            subsys_path = name.rpartition('.')[0]
            subsys = self._get_subsystem(subsys_path) if subsys_path else self

            abs_name = prefix + name

            # Will need to set both of these dicts to keep them both up-to-date
            # _var_allprocs_abs2meta is a partial copy of _var_abs2meta
            abs2meta = subsys._var_abs2meta['output']
            allprocs_abs2meta = subsys._var_allprocs_abs2meta['output']

            if abs_name not in abs2meta:
                raise RuntimeError(
                    f"Output solver options set using System.set_output_solver_options for "
                    f"non-existent variable '{abs_name}' in System '{self.pathname}'.")

            metadatadict_abs2meta = abs2meta[abs_name]
            metadatadict_allprocs_abs2meta = allprocs_abs2meta[abs_name]

            # Update the metadata that was set
            for meta_key in options:
                if options[meta_key] is None:
                    val_as_float_or_array_or_none = None
                else:
                    shape = metadatadict_abs2meta['shape']
                    val = ensure_compatible(name, options[meta_key], shape)[0]
                    val_as_float_or_array_or_none = format_as_float_or_array(meta_key, val,
                                                                             flatten=True)

                # Setting both here because the copying of _var_abs2meta to
                #   _var_allprocs_abs2meta happens before this. Need to keep both up to date
                metadatadict_abs2meta.update({
                    meta_key: val_as_float_or_array_or_none,
                })
                metadatadict_allprocs_abs2meta.update({
                    meta_key: val_as_float_or_array_or_none,
                })

        # recalculate the _has scaling and bounds vars (_has_output_scaling, _has_output_adder,
        # _has_resid_scaling, _has_bounds ) across all outputs.
        # Since you are allowed to reference multiple subsystems from set_output_solver_options,
        #    need to loop over all of the ones that got modified by those calls.
        # Loop over all the options set. Each one of these could be referencing a different
        #    subsystem since the name could be a path
        for name, options in self._output_solver_options.items():
            subsys_path = name.rpartition('.')[0]
            subsys = self._get_subsystem(subsys_path) if subsys_path else self

            # Now that we know which subsystem was affected. We have to recalculate
            #   _has_output_scaling, _has_output_adder, _has_resid_scaling, _has_bounds
            #   across all the outputs of that subsystem, since the changes might have
            #   affected their values
            subsys._has_output_scaling = False
            subsys._has_output_adder = False
            subsys._has_resid_scaling = False
            subsys._has_bounds = False

            abs2meta = subsys._var_abs2meta['output']
            for abs_name, metadata in abs2meta.items():  # Loop over all outputs for that subsystem
                ref = metadata['ref']
                if np.isscalar(ref):
                    subsys._has_output_scaling |= ref != 1.0
                else:
                    subsys._has_output_scaling |= np.any(ref != 1.0)

                ref0 = metadata['ref0']
                if np.isscalar(ref0):
                    subsys._has_output_scaling |= ref0 != 0.0
                    subsys._has_output_adder |= ref0 != 0.0
                else:
                    subsys._has_output_scaling |= np.any(ref0)
                    subsys._has_output_adder |= np.any(ref0)

                res_ref = metadata['res_ref']
                if np.isscalar(res_ref):
                    subsys._has_resid_scaling |= res_ref != 1.0
                else:
                    subsys._has_resid_scaling |= np.any(res_ref != 1.0)

                if metadata['lower'] is not None or metadata['upper'] is not None:
                    subsys._has_bounds = True

        # Clear the cached to indicate that the cached values have been applied
        self._output_solver_options = {}

    def set_design_var_options(self, name,
                               lower=_UNDEFINED, upper=_UNDEFINED,
                               scaler=_UNDEFINED, adder=_UNDEFINED,
                               ref=_UNDEFINED, ref0=_UNDEFINED):
        """
        Set options for design vars in the model.

        Can be used to set the options outside of setting them when calling add_design_var

        Parameters
        ----------
        name : str
            Name of the variable in this system's namespace.
        lower : float or ndarray, optional
            Lower boundary for the input.
        upper : upper or ndarray, optional
            Upper boundary for the input.
        scaler : float or ndarray, optional
            Value to multiply the model value to get the scaled value for the driver. scaler
            is second in precedence. adder and scaler are an alterantive to using ref
            and ref0.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value for the driver. adder
            is first in precedence.  adder and scaler are an alterantive to using ref
            and ref0.
        ref : float or ndarray, optional
            Value of design var that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of design var that scales to 0.0 in the driver.
        """
        # Check inputs

        # Name must be a string
        if not isinstance(name, str):
            raise TypeError('{}: The name argument should be a string, got {}'.format(self.msginfo,
                                                                                      name))
        are_new_bounds = lower is not _UNDEFINED or upper is not _UNDEFINED
        are_new_scaling = scaler is not _UNDEFINED or adder is not _UNDEFINED or ref is not \
            _UNDEFINED or ref0 is not _UNDEFINED

        # Must set at least one argument for this function to do something
        if not are_new_scaling and not are_new_bounds:
            raise RuntimeError(
                'Must set a value for at least one argument in call to set_design_var_options.')

        if self._static_mode:
            design_vars = self._static_design_vars
        else:
            design_vars = self._design_vars

        if name not in design_vars:
            msg = "{}: set_design_var_options called with design variable '{}' that does not exist."
            raise RuntimeError(msg.format(self.msginfo, name))

        existing_dv_meta = design_vars[name]

        are_existing_scaling = existing_dv_meta['scaler'] is not None or \
            existing_dv_meta['adder'] is not None or \
            existing_dv_meta['ref'] is not None or \
            existing_dv_meta['ref0'] is not None
        are_existing_bounds = existing_dv_meta['lower'] is not None or \
            existing_dv_meta['upper'] is not None

        # figure out the bounds (lower, upper) based on what is passed to this
        #   method and what were the existing bounds
        if are_new_bounds:
            # wipe out all the bounds and only use what is set by the arguments to this call
            if lower is _UNDEFINED:
                lower = None
            if upper is _UNDEFINED:
                upper = None
        else:
            lower = existing_dv_meta['lower']
            upper = existing_dv_meta['upper']

        if are_new_scaling and are_existing_scaling and are_existing_bounds and not are_new_bounds:
            # need to unscale bounds using the existing scaling so the new scaling can
            # be applied. But if no new bounds, no need to
            if lower is not None:
                lower = lower / existing_dv_meta['scaler'] - existing_dv_meta['adder']
            if upper is not None:
                upper = upper / existing_dv_meta['scaler'] - existing_dv_meta['adder']

        # Now figure out scaling
        if are_new_scaling:
            if scaler is _UNDEFINED:
                scaler = None
            if adder is _UNDEFINED:
                adder = None
            if ref is _UNDEFINED:
                ref = None
            if ref0 is _UNDEFINED:
                ref0 = None
        else:
            scaler = existing_dv_meta['scaler']
            adder = existing_dv_meta['adder']
            ref = existing_dv_meta['ref']
            ref0 = existing_dv_meta['ref0']

        # Convert ref/ref0 to ndarray/float as necessary
        ref = format_as_float_or_array('ref', ref, val_if_none=None, flatten=True)
        ref0 = format_as_float_or_array('ref0', ref0, val_if_none=None, flatten=True)

        # determine adder and scaler based on args
        adder, scaler = determine_adder_scaler(ref0, ref, adder, scaler)

        if lower is None:
            # if not set, set lower to -INF_BOUND and don't apply adder/scaler
            lower = -INF_BOUND
        else:
            # Convert lower to ndarray/float as necessary
            lower = format_as_float_or_array('lower', lower, flatten=True)
            # Apply scaler/adder
            lower = (lower + adder) * scaler

        if upper is None:
            # if not set, set upper to INF_BOUND and don't apply adder/scaler
            upper = INF_BOUND
        else:
            # Convert upper to ndarray/float as necessary
            upper = format_as_float_or_array('upper', upper, flatten=True)
            # Apply scaler/adder
            upper = (upper + adder) * scaler

        if isinstance(scaler, np.ndarray):
            if np.all(scaler == 1.0):
                scaler = None
        elif scaler == 1.0:
            scaler = None

        if isinstance(adder, np.ndarray):
            if not np.any(adder):
                adder = None
        elif adder == 0.0:
            adder = None

        # Put together a dict of the new values so they can be used to update the metadata for
        #   this var
        new_desvar_metadata = {
            'scaler': scaler,
            'total_scaler': scaler,
            'adder': adder,
            'total_adder': adder,
            'upper': upper,
            'lower': lower,
            'ref': ref,
            'ref0': ref0,
        }

        design_vars[name].update(new_desvar_metadata)

    def set_constraint_options(self, name, ref=_UNDEFINED, ref0=_UNDEFINED,
                               equals=_UNDEFINED, lower=_UNDEFINED, upper=_UNDEFINED,
                               adder=_UNDEFINED, scaler=_UNDEFINED, alias=_UNDEFINED):
        """
        Set options for constraints in the model.

        Can be used to set options that were set using add_constraint.

        Parameters
        ----------
        name : str
            Name of the response variable in the system, or alias if given.
        ref : float or ndarray, optional
            Value of response variable that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of response variable that scales to 0.0 in the driver.
        equals : float or ndarray, optional
            Equality constraint value for the variable.
        lower : float or ndarray, optional
            Lower boundary for the variable.
        upper : float or ndarray, optional
            Upper boundary for the variable.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value for the driver. adder
            is first in precedence.  adder and scaler are an alterantive to using ref
            and ref0.
        scaler : float or ndarray, optional
            Value to multiply the model value to get the scaled value for the driver. scaler
            is second in precedence. adder and scaler are an alterantive to using ref
            and ref0.
        alias : str, optional
            Alias for this response. Necessary when adding multiple constraints on different
            indices or slices of a single variable.
        """
        # Check inputs
        if not isinstance(name, str):
            raise TypeError('{}: The name argument should be a string, '
                            'got {}'.format(self.msginfo, name))

        if alias is not _UNDEFINED:
            warn_deprecation("Option 'alias' of set_constraint_options is deprecated. "
                             "If the constraint has an alias, provide that as the "
                             "'name' argument to set_constraint_options.")
            name = alias

        are_new_bounds = equals is not _UNDEFINED or lower is not _UNDEFINED or upper is not \
            _UNDEFINED
        are_new_scaling = scaler is not _UNDEFINED or adder is not _UNDEFINED or \
            ref is not _UNDEFINED or ref0 is not _UNDEFINED

        # At least one of the scaling or bounds parameters must be set or function won't do anything
        if not are_new_scaling and not are_new_bounds:
            raise RuntimeError(
                'Must set a value for at least one argument in call to set_constraint_options.')

        # A constraint cannot be an equality and inequality constraint
        if equals is not _UNDEFINED and (lower is not _UNDEFINED or upper is not _UNDEFINED):
            msg = "{}: Constraint '{}' cannot be both equality and inequality."
            raise ValueError(msg.format(self.msginfo, name))

        if self._static_mode and self._static_responses:
            responses = self._static_responses
        else:
            responses = self._responses

        if name not in responses:
            msg = f"{self.msginfo}: set_constraint_options called with " \
                f"constraint '{name}' that does not exist. If the constraint was provided " \
                f"an alias, use that in place of its name for set_constraint_options."
            raise RuntimeError(msg)

        existing_cons_meta = responses[name]
        are_existing_scaling = existing_cons_meta['scaler'] is not None or \
            existing_cons_meta['adder'] is not None or \
            existing_cons_meta['ref'] is not None or \
            existing_cons_meta['ref0'] is not None
        are_existing_bounds = existing_cons_meta['equals'] is not None or \
            existing_cons_meta['lower'] is not None or \
            existing_cons_meta['upper'] is not None

        # figure out the bounds (equals, lower, upper) based on what is passed to this
        #   method and what were the existing bounds
        if are_new_bounds:
            # wipe the slate clean and only use what is set by the arguments to this call
            if equals is _UNDEFINED:
                equals = None
            if lower is _UNDEFINED:
                lower = None
            if upper is _UNDEFINED:
                upper = None
        else:
            equals = existing_cons_meta['equals']
            lower = existing_cons_meta['lower']
            upper = existing_cons_meta['upper']

        if are_new_scaling and are_existing_scaling and are_existing_bounds and not are_new_bounds:
            # need to unscale bounds using the existing scaling so the new scaling can
            # be applied
            if lower is not None:
                lower = lower / existing_cons_meta['scaler'] - existing_cons_meta['adder']
            if upper is not None:
                upper = upper / existing_cons_meta['scaler'] - existing_cons_meta['adder']
            if equals is not None:
                equals = equals / existing_cons_meta['scaler'] - existing_cons_meta['adder']

        # Now figure out scaling
        if are_new_scaling:
            if scaler is _UNDEFINED:
                scaler = None
            if adder is _UNDEFINED:
                adder = None
            if ref is _UNDEFINED:
                ref = None
            if ref0 is _UNDEFINED:
                ref0 = None
        else:
            scaler = existing_cons_meta['scaler']
            adder = existing_cons_meta['adder']
            ref = existing_cons_meta['ref']
            ref0 = existing_cons_meta['ref0']

        # Convert ref/ref0 to ndarray/float as necessary
        ref = format_as_float_or_array('ref', ref, val_if_none=None, flatten=True)
        ref0 = format_as_float_or_array('ref0', ref0, val_if_none=None, flatten=True)

        # determine adder and scaler based on args
        adder, scaler = determine_adder_scaler(ref0, ref, adder, scaler)

        # Convert lower to ndarray/float as necessary
        try:
            if lower is None:
                # don't apply adder/scaler if lower not set
                lower = -INF_BOUND
            else:
                lower = format_as_float_or_array('lower', lower, flatten=True)
                if lower != - INF_BOUND:
                    lower = (lower + adder) * scaler
        except (TypeError, ValueError):
            raise TypeError("Argument 'lower' can not be a string ('{}' given). You can not "
                            "specify a variable as lower bound. You can only provide constant "
                            "float values".format(lower))

        # Convert upper to ndarray/float as necessary
        try:
            if upper is None:
                # don't apply adder/scaler if upper not set
                upper = INF_BOUND
            else:
                upper = format_as_float_or_array('upper', upper, flatten=True)
                if upper != INF_BOUND:
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

        if isinstance(scaler, np.ndarray):
            if np.all(scaler == 1.0):
                scaler = None
        elif scaler == 1.0:
            scaler = None

        if isinstance(adder, np.ndarray):
            if not np.any(adder):
                adder = None
        elif adder == 0.0:
            adder = None

        new_cons_metadata = {
            'ref': ref,
            'ref0': ref0,
            'equals': equals,
            'lower': lower,
            'upper': upper,
            'adder': adder,
            'total_adder': adder,
            'scaler': scaler,
            'total_scaler': scaler,
        }

        responses[name].update(new_cons_metadata)

    def set_objective_options(self, name, ref=_UNDEFINED, ref0=_UNDEFINED,
                              adder=_UNDEFINED, scaler=_UNDEFINED, alias=_UNDEFINED):
        """
        Set options for objectives in the model.

        Can be used to set options after they have been set by add_objective.

        Parameters
        ----------
        name : str
            Name of the response variable in the system, or alias if given.
        ref : float or ndarray, optional
            Value of response variable that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of response variable that scales to 0.0 in the driver.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value for the driver. adder
            is first in precedence.  adder and scaler are an alterantive to using ref
            and ref0.
        scaler : float or ndarray, optional
            Value to multiply the model value to get the scaled value for the driver. scaler
            is second in precedence. adder and scaler are an alterantive to using ref
            and ref0.
        alias : str
            Alias for this response. Used to disambiguate variable names when adding
            multiple objectives on different indices or slices of a single variable. Deprecated.
        """
        # Check inputs
        # Name must be a string
        if not isinstance(name, str):
            raise TypeError(f'{self.msginfo}: The name argument should be a string, got {name}')

        if alias is not _UNDEFINED:
            warn_deprecation("Option 'alias' of set_objective_options is deprecated. "
                             "If the objective has an alias, provide that as the 'name' "
                             "argument to set_objective_options.")
            name = alias

        # At least one of the scaling parameters must be set or function does nothing
        if scaler is _UNDEFINED and adder is _UNDEFINED and ref is _UNDEFINED and ref0 == \
                _UNDEFINED:
            raise RuntimeError(
                'Must set a value for at least one argument in call to set_objective_options.')

        if self._static_mode and self._static_responses:
            responses = self._static_responses
        else:
            responses = self._responses

        # If the name is not in responses, which are keyed by alias, then it was given
        # as the actual variable name but the variable has a different alias.
        if name not in responses:
            msg = f"{self.msginfo}: set_objective_options called with " \
                f"objective '{name}' that does not exist. If the objective was provided " \
                f"an alias, use that in place of its name for set_objective_options."
            raise RuntimeError(msg)

        # Since one or more of these are being set by the incoming arguments, the
        #   ones that are not being set should be set to None since they will be re-computed below
        if scaler is _UNDEFINED:
            scaler = None
        if adder is _UNDEFINED:
            adder = None
        if ref is _UNDEFINED:
            ref = None
        if ref0 is _UNDEFINED:
            ref0 = None

        # Convert ref/ref0 to ndarray/float as necessary
        ref = format_as_float_or_array('ref', ref, val_if_none=None, flatten=True)
        ref0 = format_as_float_or_array('ref0', ref0, val_if_none=None, flatten=True)

        # determine adder and scaler based on args
        adder, scaler = determine_adder_scaler(ref0, ref, adder, scaler)

        if isinstance(scaler, np.ndarray):
            if np.all(scaler == 1.0):
                scaler = None
        elif scaler == 1.0:
            scaler = None

        if isinstance(adder, np.ndarray):
            if not np.any(adder):
                adder = None
        elif adder == 0.0:
            adder = None

        new_obj_metadata = {
            'ref': ref,
            'ref0': ref0,
            'adder': adder,
            'total_adder': adder,
            'scaler': scaler,
            'total_scaler': scaler,
        }

        responses[name].update(new_obj_metadata)

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
        try:
            prom2abs = self._problem_meta['prom2abs']
        except Exception:
            raise RuntimeError(f"{self.msginfo}: get_source cannot be called for variable {name} "
                               "before Problem.setup has been called.")

        if name in prom2abs['output']:
            return prom2abs['output'][name][0]

        if name in prom2abs['input']:
            name = prom2abs['input'][name][0]

        model = self._problem_meta['model_ref']()
        if name in model._conn_global_abs_in2out:
            return model._conn_global_abs_in2out[name]

        raise KeyError(f"{self.msginfo}: source for '{name}' not found.")

    def _get_graph_node_meta(self):
        """
        Return metadata to add to this system's graph node.

        Returns
        -------
        dict
            Metadata for this system's graph node.
        """
        return {
            'classname': type(self).__name__,
            'implicit': not self.is_explicit(),
        }

    def _setup_check(self):
        """
        Do any error checking on user's setup, before any other recursion happens.
        """
        pass

    def _configure_check(self):
        """
        Do any error checking on i/o and connections.
        """
        pass

    def _get_approx_subjac_keys(self):
        """
        Return a list of (of, wrt) keys needed for approx derivs for this system.

        All keys are absolute names. If this system is the top level Group, the keys will be source
        names.  If not, they will be absolute input and output names.

        Returns
        -------
        list
            List of approx derivative subjacobian keys.
        """
        if self._approx_subjac_keys is None:
            self._approx_subjac_keys = list(self._approx_subjac_keys_iter())

        return self._approx_subjac_keys

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
            self._coloring_info.dynamic = True
            return  # don't use static this time

        self._coloring_info.static = coloring
        self._coloring_info.dynamic = False

        if coloring is not _STD_COLORING_FNAME:
            if recurse:
                issue_warning('recurse was passed to use_fixed_coloring but a specific coloring '
                              'was set, so recurse was ignored.',
                              prefix=self.pathname,
                              category=UnusedOptionWarning)
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
            If True, plot sparsity with coloring info after generating coloring.
        """
        if method not in ('fd', 'cs', 'jax'):
            raise RuntimeError(
                "{}: method must be one of ['fd', 'cs', 'jax'].".format(self.msginfo))

        self._has_approx = True

        # start with defaults
        options = coloring_mod.Partial_ColoringMeta()

        if method != 'jax':
            approx = self._get_approx_scheme(method)
            options.update({k: v for k, v in approx.DEFAULT_OPTIONS.items()
                            if k in ('step', 'form')})

        if self._coloring_info.static is None:
            options.dynamic = True
        else:
            options.dynamic = False
            options.static = self._coloring_info.static

        options.coloring = self._coloring_info.coloring

        if isinstance(wrt, str):
            options.wrt_patterns = (wrt, )
        else:
            options.wrt_patterns = tuple(wrt)
        options.method = method
        options.per_instance = per_instance
        options.num_full_jacs = num_full_jacs
        options.tol = tol
        options.orders = orders
        options.perturb_size = perturb_size
        options.min_improve_pct = min_improve_pct
        options.show_summary = show_summary
        options.show_sparsity = show_sparsity
        if form is not None:
            options.form = form
        if step is not None:
            options.step = step

        self._coloring_info = options

    def _finalize_coloring(self, coloring, info, sp_info, sparsity_time):
        # if the improvement wasn't large enough, don't use coloring
        info.set_coloring(coloring, msginfo=self.msginfo)
        if info._failed:
            if not info.per_instance:
                # save the class coloring for so resources won't be wasted computing
                # a bad coloring
                coloring_mod._CLASS_COLORINGS[self.get_coloring_fname()] = None
            return False

        sp_info['sparsity_time'] = sparsity_time
        sp_info['pathname'] = self.pathname
        sp_info['class'] = type(self).__name__
        sp_info['type'] = 'semi-total' if self._subsystems_allprocs else 'partial'

        ordered_wrt_info = list(self._jac_wrt_iter(info.wrt_matches))
        ordered_of_info = list(self._jac_of_iter())

        if self.pathname:
            ordered_of_info = self._jac_var_info_abs2prom(ordered_of_info)
            ordered_wrt_info = self._jac_var_info_abs2prom(ordered_wrt_info)

        coloring._row_vars = [t[0] for t in ordered_of_info]
        coloring._col_vars = [t[0] for t in ordered_wrt_info]
        coloring._row_var_sizes = [t[2] - t[1] for t in ordered_of_info]
        coloring._col_var_sizes = [t[2] - t[1] for t in ordered_wrt_info]

        coloring._meta.update(info)  # save metadata we used to create the coloring
        coloring._meta.update(sp_info)

        if info.show_sparsity or info.show_summary:
            print("\nColoring for '%s' (class %s)" % (self.pathname, type(self).__name__))

        info.display()

        self._save_coloring(coloring)

        if not info.per_instance:
            # save the class coloring for other instances of this class to use
            coloring_mod._CLASS_COLORINGS[self.get_coloring_fname()] = coloring

        return True

    def _compute_coloring(self, recurse=False, **overrides):
        """
        Compute a coloring of the partial jacobian.

        This assumes that the current System is in a proper state for computing derivatives.

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
            my_coloring = self._coloring_info.coloring
            grad_systems = self._get_gradient_nl_solver_systems()
            for s in self.system_iter(include_self=True, recurse=True):
                if my_coloring is None or s in grad_systems:
                    if s._coloring_info.coloring is not None:
                        coloring = s._compute_coloring(recurse=False, **overrides)[0]
                        colorings.append(coloring)
                        if coloring is not None:
                            coloring._meta['pathname'] = s.pathname
                            coloring._meta['class'] = type(s).__name__
            return [c for c in colorings if c is not None] or [None]

        info = self._coloring_info

        use_jax = False
        try:
            if self.options['use_jax']:
                info['method'] = 'jax'
                use_jax = True
        except KeyError:
            pass

        info.update(overrides)

        if info['method'] is None and self._approx_schemes:
            info['method'] = list(self._approx_schemes)[0]

        if info.coloring is None:
            # check to see if any approx or jax derivs have been declared
            for meta in self._subjacs_info.values():
                if 'method' in meta and meta['method']:
                    break
            else:  # no approx derivs found
                if not (self._owns_approx_of or self._owns_approx_wrt):
                    issue_warning("No partials found but coloring was requested.  "
                                  "Declaring ALL partials as dense "
                                  "(method='{}')".format(info['method']),
                                  prefix=self.msginfo, category=DerivativesWarning)
                    try:
                        self.declare_partials('*', '*', method=info['method'])
                    except AttributeError:  # assume system is a group
                        from openmdao.core.component import Component
                        from openmdao.core.indepvarcomp import IndepVarComp
                        from openmdao.components.exec_comp import ExecComp
                        for s in self.system_iter(recurse=True, typ=Component):
                            if not isinstance(s, ExecComp) and not isinstance(s, IndepVarComp):
                                s.declare_partials('*', '*', method=info['method'])
                    self._setup_partials()

        if not use_jax:
            approx_scheme = self._get_approx_scheme(info['method'])

        if info.coloring is None and info.static is None:
            info.dynamic = True

        coloring_fname = self.get_coloring_fname()

        # if we find a previously computed class coloring for our class, just use that
        # instead of regenerating a coloring.
        if not info.per_instance and coloring_fname in coloring_mod._CLASS_COLORINGS:
            info.coloring = coloring = coloring_mod._CLASS_COLORINGS[coloring_fname]
            if coloring is None:
                print("\nClass coloring for class '{}' wasn't good enough, "
                      "so skipping for '{}'".format(type(self).__name__, self.pathname))
                info.static = None
            else:
                print("\n{} using class coloring for class '{}'".format(self.pathname,
                                                                        type(self).__name__))
                info.update(coloring._meta)
                # force regen of approx groups during next compute_approximations
                if not use_jax:
                    approx_scheme._reset()
            return [coloring]

        save_first_call = self._first_call_to_linearize
        self._first_call_to_linearize = False
        sparsity_start_time = time.perf_counter()

        # for groups, this does some setup of approximations
        self._setup_approx_coloring()

        # tell approx scheme to limit itself to only colored columns
        if not use_jax:
            approx_scheme._reset()
            self._during_sparsity = True

        info._update_wrt_matches(self)

        save_jac = self._jacobian

        # use special sparse jacobian to collect sparsity info
        self._jacobian = _ColSparsityJac(self, info)

        from openmdao.core.group import Group
        is_total = isinstance(self, Group)
        is_explicit = self.is_explicit()

        # compute perturbations
        starting_inputs = self._inputs.asarray(copy=True)
        in_offsets = starting_inputs.copy()
        in_offsets[in_offsets == 0.0] = 1.0
        in_offsets *= info['perturb_size']

        starting_outputs = self._outputs.asarray(copy=True)

        if not is_explicit:
            out_offsets = starting_outputs.copy()
            out_offsets[out_offsets == 0.0] = 1.0
            out_offsets *= info['perturb_size']

        starting_resids = self._residuals.asarray(copy=True)

        for i in range(info['num_full_jacs']):
            # randomize inputs (and outputs if implicit)
            if i > 0:
                self._inputs.set_val(starting_inputs +
                                     in_offsets * np.random.random(in_offsets.size))
                if not is_explicit:
                    self._outputs.set_val(starting_outputs +
                                          out_offsets * np.random.random(out_offsets.size))
                if is_total:
                    with self._relevance.nonlinear_active('iter'):
                        self._solve_nonlinear()
                else:
                    self._apply_nonlinear()

                if not use_jax:
                    for scheme in self._approx_schemes.values():
                        scheme._reset()  # force a re-initialization of approx

            if use_jax:
                self._jax_linearize()
            else:
                self.run_linearize(sub_do_ln=False)

        sparsity, sp_info = self._jacobian.get_sparsity(self)

        self._jacobian = save_jac

        if not use_jax:
            self._during_sparsity = False

            # revert uncolored approx back to normal
            for scheme in self._approx_schemes.values():
                scheme._reset()

        if use_jax:
            direction = self._mode
        else:
            direction = 'fwd'

        sparsity_time = time.perf_counter() - sparsity_start_time

        coloring = _compute_coloring(sparsity, direction)

        # restore original inputs/outputs
        self._inputs.set_val(starting_inputs)
        self._outputs.set_val(starting_outputs)
        self._residuals.set_val(starting_resids)

        if not self._finalize_coloring(coloring, info, sp_info, sparsity_time):
            return [None]

        self._first_call_to_linearize = save_first_call

        if not use_jax:
            approx = self._get_approx_scheme(coloring._meta['method'])
            # force regen of approx groups during next compute_approximations
            approx._reset()

        return [coloring]

    def _setup_approx_coloring(self):
        pass

    def get_coloring_fname(self):
        """
        Return the full pathname to a coloring file.

        Returns
        -------
        str
            Full pathname of the coloring file.
        """
        directory = self._problem_meta['coloring_dir']
        if not self.pathname:
            # total coloring
            return os.path.join(directory, 'total_coloring.pkl')

        if self._coloring_info.per_instance:
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
            coloring.save(self.get_coloring_fname())

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
        coloring = info.coloring
        if coloring is not None:
            return coloring

        static = info.static
        if static is _STD_COLORING_FNAME or isinstance(static, str):
            if static is _STD_COLORING_FNAME:
                fname = self.get_coloring_fname()
            else:
                fname = static
            print("%s: loading coloring from file %s" % (self.msginfo, fname))
            info.coloring = coloring = Coloring.load(fname)
            if info.wrt_patterns != coloring._meta['wrt_patterns']:
                raise RuntimeError("%s: Loaded coloring has different wrt_patterns (%s) than "
                                   "declared ones (%s)." %
                                   (self.msginfo, coloring._meta['wrt_patterns'],
                                    info.wrt_patterns))
            info.update(info.coloring._meta)
            approx = self._get_approx_scheme(info['method'])
            # force regen of approx groups during next compute_approximations
            approx._reset()
        elif isinstance(static, coloring_mod.Coloring):
            info.coloring = coloring = static

        if coloring is not None:
            info.dynamic = False

        info.static = coloring

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
        if coloring is None:
            if self._coloring_info.dynamic:
                self._coloring_info.coloring = coloring = self._compute_coloring()[0]
                if coloring is not None:
                    self._coloring_info.update(coloring._meta)
        else:
            if not self._coloring_info.dynamic:
                coloring._check_config_partial(self)

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
            color = np.empty(comm.size, dtype=INT_DTYPE)
            for i in range(num_par_fd):
                color[offsets[i]:offsets[i] + sizes[i]] = i

            self._par_fd_id = color[comm.rank]

            comm = self._full_comm.Split(self._par_fd_id)

        return comm

    def _setup_recording(self):
        """
        Set up case recording.
        """
        if self._rec_mgr._recorders:
            myinputs = myoutputs = myresiduals = []

            options = self.recording_options
            incl = options['includes']
            excl = options['excludes']

            # includes and excludes for outputs are specified using promoted names
            # includes and excludes for inputs are specified using _absolute_ names
            abs2prom_output = self._var_allprocs_abs2prom['output']
            abs2prom_inputs = self._var_allprocs_abs2prom['input']

            # set of promoted output names and absolute input and residual names
            # used for matching includes/excludes
            match_names = set()

            # includes and excludes for inputs are specified using _absolute_ names
            # vectors are keyed on absolute name, discretes on relative/promoted name
            if options['record_inputs']:
                match_names.update(abs2prom_inputs.keys())
                myinputs = sorted([n for n in abs2prom_inputs
                                   if check_path(n, incl, excl)])

            # includes and excludes for outputs are specified using _promoted_ names
            # vectors are keyed on absolute name, discretes on relative/promoted name
            if options['record_outputs']:
                match_names.update(abs2prom_output.values())
                myoutputs = sorted([n for n, prom in abs2prom_output.items()
                                    if check_path(prom, incl, excl)])

                if self._var_discrete['output']:
                    # if we have discrete outputs then residual name set doesn't match output one
                    if options['record_residuals']:
                        contains = self._residuals._contains_abs
                        myresiduals = [n for n in myoutputs if contains(n)]
                elif options['record_residuals']:
                    myresiduals = myoutputs

            elif options['record_residuals']:
                match_names.update(self._residuals.keys())
                myresiduals = [n for n in self._residuals._abs_iter()
                               if check_path(abs2prom_output[n], incl, excl)]

            # check that all exclude/include globs have at least one matching output or input name
            for pattern in excl:
                if not has_match(pattern, match_names):
                    issue_warning(f"{self.msginfo}: No matches for pattern '{pattern}' in "
                                  "recording_options['excludes'].")
            for pattern in incl:
                if not has_match(pattern, match_names):
                    issue_warning(f"{self.msginfo}: No matches for pattern '{pattern}' in "
                                  "recording_options['includes'].")

            self._filtered_vars_to_record = {
                'input': myinputs,
                'output': myoutputs,
                'residual': myresiduals
            }

            self._rec_mgr.startup(self, self._problem_meta['comm'])

        for subsys in self._subsystems_myproc:
            subsys._setup_recording()

    def _reset_setup_vars(self):
        """
        Reset all the stuff that gets initialized in setup.
        """
        self._first_call_to_linearize = True
        self._is_local = True
        self._vectors = {}
        self._full_comm = None
        self._approx_subjac_keys = None

        self.options._parent_name = self.msginfo
        self.recording_options._parent_name = self.msginfo
        self._design_vars = {}
        self._responses = {}
        self._design_vars.update(self._static_design_vars)
        self._responses.update(self._static_responses)

    def _setup_procs(self, pathname, comm, prob_meta):
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
        prob_meta : dict
            Problem level options.
        """
        self._reset_setup_vars()

        self.pathname = pathname
        self._set_problem_meta(prob_meta)
        self.load_model_options()

    def _setup_var_data(self):
        """
        Compute the list of abs var names, abs/prom name maps, and metadata dictionaries.
        """
        self._var_prom2inds = {}
        self._var_allprocs_prom2abs_list = {'input': {}, 'output': {}}
        self._var_abs2prom = {'input': {}, 'output': {}}
        self._var_allprocs_abs2prom = {'input': {}, 'output': {}}
        self._var_allprocs_abs2meta = {'input': {}, 'output': {}}
        self._var_abs2meta = {'input': {}, 'output': {}}
        self._var_allprocs_discrete = {'input': {}, 'output': {}}
        self._var_allprocs_abs2idx = {}
        self._owning_rank = defaultdict(int)
        self._var_sizes = {}
        self._owned_sizes = None

        cfginfo = self._problem_meta['config_info']
        if cfginfo and self.pathname in cfginfo._modified_systems:
            cfginfo._modified_systems.remove(self.pathname)

    def _setup_global_shapes(self):
        """
        Compute the global size and shape of all variables on this system.
        """
        loc_meta = self._var_abs2meta

        for io in ('input', 'output'):
            # now set global sizes and shapes into metadata for distributed variables
            sizes = self._var_sizes[io]
            for idx, (abs_name, mymeta) in enumerate(self._var_allprocs_abs2meta[io].items()):
                local_shape = mymeta['shape']
                if mymeta['distributed']:
                    global_size = np.sum(sizes[:, idx])
                    mymeta['global_size'] = global_size

                    # assume that all but the first dimension of the shape of a
                    # distributed variable is the same on all procs
                    mymeta['global_shape'] = self._get_full_dist_shape(abs_name, local_shape)
                else:
                    # not distributed, just use local shape and size
                    mymeta['global_size'] = mymeta['size']
                    mymeta['global_shape'] = local_shape

                if abs_name in loc_meta[io]:
                    loc_meta[io][abs_name]['global_shape'] = mymeta['global_shape']
                    loc_meta[io][abs_name]['global_size'] = mymeta['global_size']

    def _setup_driver_units(self, abs2meta=None):
        """
        Compute unit conversions for driver variables.
        """
        if abs2meta is None:
            abs2meta = self._var_allprocs_abs2meta['output']

        has_scaling = False

        for name, meta in self._design_vars.items():

            units = meta['units']
            meta['total_adder'] = meta['adder']
            meta['total_scaler'] = meta['scaler']

            if units is not None:
                # If derivatives are not being calculated, then you reach here before source
                # is placed in the meta.
                try:
                    units_src = meta['source']
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

                # Derivation of the total scaler and total adder for design variables:
                # Given based design variable value y
                # First we apply the desired unit conversion
                # y_in_desired_units = unit_scaler * (y + unit_adder)
                # Then we apply the user-declared scaling
                # y_opt = declared_scaler * (y_in_desired_units + declared_adder)
                # Thus
                # y_opt = declared_scaler * (unit_scaler * (y + unit_adder) + declared_adder)
                # And collecting terms
                # y_opt = [declared_scaler * unit_scaler]
                #         * (y + unit_adder + declared_adder/unit_scaler)
                # So the total_scaler and total_adder for the optimizer are:
                # total_scaler = declared_scaler * unit_scaler
                # total_adder = unit_adder + declared_adder / unit_scaler

                unit_scaler, unit_adder = unit_conversion(var_units, units)
                declared_adder, declared_scaler = determine_adder_scaler(None, None,
                                                                         meta['adder'],
                                                                         meta['scaler'])

                meta['total_adder'] = unit_adder + declared_adder / unit_scaler
                meta['total_scaler'] = declared_scaler * unit_scaler

            if meta['total_scaler'] is not None:
                has_scaling = True

        resp = self._responses
        type_dict = {'con': 'constraint', 'obj': 'objective'}
        for name, meta in resp.items():

            units = meta['units']
            meta['total_scaler'] = meta['scaler']
            meta['total_adder'] = meta['adder']

            if units is not None:
                # If derivatives are not being calculated, then you reach here before source
                # is placed in the meta.
                try:
                    units_src = meta['source']
                except KeyError:
                    units_src = self.get_source(meta['name'])

                src_units = abs2meta[units_src]['units']

                if src_units == units:
                    continue

                if src_units is None:
                    msg = "{}: Target for {} {} has no units, but '{}' units " + \
                          "were specified."
                    raise RuntimeError(msg.format(self.msginfo, type_dict[meta['type']],
                                                  name, units))

                if not is_compatible(src_units, units):
                    msg = "{}: Target for {} {} has '{}' units, but '{}' units " + \
                          "were specified."
                    raise RuntimeError(msg.format(self.msginfo, type_dict[meta['type']],
                                                  name, src_units, units))

                unit_scaler, unit_adder = unit_conversion(src_units, units)
                declared_adder, declared_scaler =\
                    determine_adder_scaler(None, None, meta['adder'], meta['scaler'])

                meta['total_scaler'] = declared_scaler * unit_scaler
                meta['total_adder'] = unit_adder + declared_adder / unit_scaler

            if meta['total_scaler'] is not None:
                has_scaling = True

        for s in self._subsystems_myproc:
            has_scaling |= s._setup_driver_units(abs2meta)

        if (self.comm.size > 1 and self._subsystems_allprocs and
                self._mpi_proc_allocator.parallel):
            has_scaling = bool(self.comm.allreduce(int(has_scaling)))

        return has_scaling

    def _setup_connections(self):
        """
        Compute dict of all connections owned by this system.
        """
        pass

    def _setup_vectors(self, root_vectors):
        """
        Compute all vectors for all vec names and assign excluded variables lists.

        Parameters
        ----------
        root_vectors : dict of dict of Vector
            Root vectors: first key is 'input', 'output', or 'residual'; second key is vec_name.
        """
        self._vectors = vectors = {'input': {}, 'output': {}, 'residual': {}}

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
        vectypes = ('nonlinear', 'linear') if self._use_derivatives else ('nonlinear',)

        for vec_name in vectypes:

            # Only allocate complex in the vectors we need.
            vec_alloc_complex = root_vectors['output'][vec_name]._alloc_complex

            for kind in ['input', 'output', 'residual']:
                rootvec = root_vectors[kind][vec_name]
                vectors[kind][vec_name] = vector_class(
                    vec_name, kind, self, rootvec, alloc_complex=vec_alloc_complex)

        if self._use_derivatives:
            vectors['input']['linear']._scaling_nl_vec = vectors['input']['nonlinear']._scaling

        self._inputs = vectors['input']['nonlinear']
        self._outputs = vectors['output']['nonlinear']
        self._residuals = vectors['residual']['nonlinear']

        if self._use_derivatives:
            self._dinputs = vectors['input']['linear']
            self._doutputs = vectors['output']['linear']
            self._dresiduals = vectors['residual']['linear']

        for subsys in self._sorted_sys_iter():
            subsys._scale_factors = self._scale_factors
            subsys._setup_vectors(root_vectors)

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

    def _get_promotion_maps(self):
        """
        Define variable maps based on promotes lists.

        Returns
        -------
        dict of {'input': {str:(str, info), ...}, 'output': {str:(str, info), ...}}
            dictionary mapping input/output variable names
            to (promoted name, promotion_info) tuple.
        """
        from openmdao.core.group import Group

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

        def _check_dup(io, matches, match_type, name, tup):
            """
            Report error or warning when attempting to promote a variable twice.

            Parameters
            ----------
            io : str
                One of 'input' or 'output'.
            matches : dict {'input': ..., 'output': ...}
                Dict of promoted names and associated info.
            match_type : intEnum
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
            try:
                old_name, old_key, old_info, old_match_type = matches[io][name]
                _, info = tup

                if old_match_type == _MatchType.RENAME:
                    old_key = old_using = (old_name, old_key)
                else:
                    old_using = f"'{old_key}'"
                if match_type == _MatchType.RENAME:
                    new_using = (name, tup[0])
                else:
                    new_using = f"'{tup[0]}'"

                diff = info.compare(old_info) if info is not None and old_info is not None else ()
                if diff:
                    raise RuntimeError(f"{self.msginfo}: {io} variable '{name}', promoted using "
                                       f"'{new_using}', was already promoted using '{old_using}' "
                                       f"with different values for {diff}.")

                if old_match_type != _MatchType.PATTERN:
                    if old_key != tup[0]:
                        raise RuntimeError(f"{self.msginfo}: Can't alias promoted {io} '{name}' to "
                                           f"'{tup[0]}' because '{name}' has already been promoted "
                                           f"as '{old_key}'.")

                if old_using != "'*'" and new_using != "'*'":
                    msg = f"{io} variable '{name}', promoted using {new_using}, " \
                          f"was already promoted using {old_using}."
                    issue_warning(msg, prefix=self.msginfo, category=PromotionWarning)
            except Exception:
                type_exc, exc, tb = sys.exc_info()
                self._collect_error(str(exc), exc_type=type_exc, tback=tb)
                return False

            return match_type == _MatchType.PATTERN

        def resolve(to_match, io_types, matches, proms):
            """
            Determine the mapping of promoted names to the parent scope for a promotion type.

            This is called once for promotes or separately for promotes_inputs and promotes_outputs.
            """
            if not to_match:
                return

            # always add '*' so we won't report if it matches nothing (in the case where the
            # system has no variables of that io type)
            found = {'*'}

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
                                    if not (n in pmap and _check_dup(io, matches, match_type, n,
                                                                     tup)):
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
                                _check_dup(io, matches, match_type, key, tup)
                            pmap[key] = (s, key, pinfo, match_type)
                            if match_type == _MatchType.NAME:
                                found.add(key)
                            else:
                                found.add((key, s))

            not_found = set(n for n, _ in to_match) - found
            if not_found:
                if (not self._var_abs2meta['input'] and not self._var_abs2meta['output'] and
                        isinstance(self, Group)):
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

        prom2abs_list = self._var_allprocs_prom2abs_list
        maps = {'input': {}, 'output': {}}

        if self._var_promotes['input'] or self._var_promotes['output']:
            if self._var_promotes['any']:
                raise RuntimeError("%s: 'promotes' cannot be used at the same time as "
                                   "'promotes_inputs' or 'promotes_outputs'." % self.msginfo)
            resolve(self._var_promotes['input'], ('input',), maps, prom2abs_list)
            resolve(self._var_promotes['output'], ('output',), maps, prom2abs_list)
        else:
            resolve(self._var_promotes['any'], ('input', 'output'), maps, prom2abs_list)

        return maps

    def _get_matvec_scope(self):
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
            self._scope_cache[None] = (None, _empty_frozen_set)
            return self._scope_cache[None]

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
                vec.scale_to_phys()
        if self._has_resid_scaling:
            for vec in residuals:
                vec.scale_to_phys()

        try:

            yield

        finally:

            if self._has_output_scaling:
                for vec in outputs:
                    vec.scale_to_norm()

            if self._has_resid_scaling:
                for vec in residuals:
                    vec.scale_to_norm()

    @contextmanager
    def _scaled_context_all(self):
        """
        Context manager that temporarily puts all vectors in a scaled state.
        """
        if self._has_output_scaling:
            for vec in self._vectors['output'].values():
                vec.scale_to_norm()
        if self._has_resid_scaling:
            for vec in self._vectors['residual'].values():
                vec.scale_to_norm()

        try:

            yield

        finally:

            if self._has_output_scaling:
                for vec in self._vectors['output'].values():
                    vec.scale_to_phys()
            if self._has_resid_scaling:
                for vec in self._vectors['residual'].values():
                    vec.scale_to_phys()

    @contextmanager
    def _matvec_context(self, scope_out, scope_in, mode, clear=True):
        """
        Context manager for vectors.

        Return vectors that use a set of internal variables that are relevant to the current
        matrix-vector product.  This is called only from _apply_linear.

        Parameters
        ----------
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
        d_inputs = self._dinputs
        d_outputs = self._doutputs
        d_residuals = self._dresiduals

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
                d_outputs._names = scope_out.intersection(old_outs)
            if scope_in is not None:
                d_inputs._names = scope_in.intersection(old_ins)

            try:
                yield d_inputs, d_outputs, d_residuals
            finally:
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
                raise
            else:
                raise err_type(
                    f"{self.msginfo}: Error calling {fname}(), {err}").with_traceback(trace)
        finally:
            self._inputs.read_only = False
            self._outputs.read_only = False
            self._residuals.read_only = False

    def get_nonlinear_vectors(self):
        """
        Return the inputs, outputs, and residuals vectors.

        Returns
        -------
        (inputs, outputs, residuals)
            Yields the inputs, outputs, and residuals nonlinear vectors.
        """
        if self._inputs is None:
            raise RuntimeError("{}: Cannot get vectors because setup has not yet been "
                               "called.".format(self.msginfo))

        return self._inputs, self._outputs, self._residuals

    def get_linear_vectors(self):
        """
        Return the linear inputs, outputs, and residuals vectors.

        Returns
        -------
        (inputs, outputs, residuals): tuple of <Vector> instances
            Yields the linear inputs, outputs, and residuals vectors.
        """
        if self._inputs is None:
            raise RuntimeError("{}: Cannot get vectors because setup has not yet been "
                               "called.".format(self.msginfo))

        return (self._dinputs, self._doutputs, self._dresiduals)

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
    def _recording_iter(self):
        return self._problem_meta['recording_iter']

    @property
    def _relevance(self):
        return self._problem_meta['relevance']

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

    @property
    def _mode(self):
        """
        Return the current system mode.

        Returns
        -------
        str
            The current system mode, 'fwd' or 'rev'.
        """
        return self._problem_meta['mode']

    @property
    def _orig_mode(self):
        """
        Return the user specified system mode.

        Returns
        -------
        str
            The system mode specified during setup, 'fwd', 'rev', or 'auto'.
        """
        return self._problem_meta['orig_mode']

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

        if self.pathname.count('.') + 1 >= depth:
            return

        for subsys, _ in self._subsystems_allprocs.values():
            subsys._set_solver_print(level=level, depth=depth, type_=type_)

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
            Iprint level. Set to 2 to print residuals each iteration; set to 1
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
        if (self._coloring_info.coloring is not None and
                self._coloring_info.wrt_matches is None):
            self._coloring_info._update_wrt_matches(self)

        # if coloring has been specified, we don't want to have multiple
        # approximations for the same subjac, so don't register any new
        # approximations when the wrt matches those used in the coloring.
        if self._get_static_coloring() is not None:  # static coloring has been specified
            return self._coloring_info.wrt_matches

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

        Yields
        ------
        type or None
        """
        if include_self and (typ is None or isinstance(self, typ)):
            yield self

        for s in self._subsystems_myproc:
            if typ is None or isinstance(s, typ):
                yield s
            if recurse:
                for sub in s.system_iter(recurse=True, typ=typ):
                    yield sub

    def _all_subsystem_iter(self):
        """
        Do nothing.

        Returns
        -------
        tuple
            An empty tuple.
        """
        return ()

    def _create_indexer(self, indices, typename, vname, flat_src=False):
        """
        Return an Indexer instance and it's size if possible.

        Parameters
        ----------
        indices : ndarray or sequence of ints
            The indices used to create the Indexer.
        typename : str
            Type name of the variable.  Could be 'design var', 'objective' or 'constraint'.
        vname : str
            Name of the variable.
        flat_src : bool
            If True, indices index into a flat array.

        Returns
        -------
        Indexer
            The newly created Indexer
        int or None
            The size of the indices, if known.
        """
        try:
            idxer = indexer(indices, flat_src=flat_src)
        except Exception as err:
            raise err.__class__(f"{self.msginfo}: Invalid indices {indices} for {typename} "
                                f"'{vname}'.")

        # size may not be available at this point, but get it if we can in order to allow
        # some earlier error checking
        try:
            size = idxer.indexed_src_size
        except Exception:
            size = None

        return idxer, size

    def add_design_var(self, name, lower=None, upper=None, ref=None, ref0=None, indices=None,
                       adder=None, scaler=None, units=None, parallel_deriv_color=None,
                       cache_linear_solution=False, flat_indices=False):
        r"""
        Add a design variable to this system.

        Parameters
        ----------
        name : str
            Promoted name of the design variable in the system.
        lower : float or ndarray, optional
            Lower boundary for the input.
        upper : upper or ndarray, optional
            Upper boundary for the input.
        ref : float or ndarray, optional
            Value of design var that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of design var that scales to 0.0 in the driver.
        indices : iter of int, optional
            If an input is an array, these indicate which entries are of
            interest for this particular design variable.  These may be
            positive or negative integers.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value for the driver. adder
            is first in precedence.  adder and scaler are an alterantive to using ref
            and ref0.
        scaler : float or ndarray, optional
            Value to multiply the model value to get the scaled value for the driver. scaler
            is second in precedence. adder and scaler are an alterantive to using ref
            and ref0.
        units : str, optional
            Units to convert to before applying scaling.
        parallel_deriv_color : str
            If specified, this design var will be grouped for parallel derivative
            calculations with other variables sharing the same parallel_deriv_color.
        cache_linear_solution : bool
            If True, store the linear solution vectors for this variable so they can
            be used to start the next linear solution with an initial guess equal to the
            solution from the previous linear solve.
        flat_indices : bool
            If True, interpret specified indices as being indices into a flat source array.

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
            try:
                units = simplify_unit(units, msginfo=self.msginfo)
            except ValueError as e:
                raise ValueError(f"{str(e)[:-1]} for design_var '{name}'.")

        # Convert ref/ref0 to ndarray/float as necessary
        ref = format_as_float_or_array('ref', ref, val_if_none=None, flatten=True)
        ref0 = format_as_float_or_array('ref0', ref0, val_if_none=None, flatten=True)

        # determine adder and scaler based on args
        adder, scaler = determine_adder_scaler(ref0, ref, adder, scaler)

        if lower is None:
            # if not set, set lower to -INF_BOUND and don't apply adder/scaler
            lower = -INF_BOUND
        else:
            # Convert lower to ndarray/float as necessary
            lower = format_as_float_or_array('lower', lower, flatten=True)
            # Apply scaler/adder
            lower = (lower + adder) * scaler

        if upper is None:
            # if not set, set upper to INF_BOUND and don't apply adder/scaler
            upper = INF_BOUND
        else:
            # Convert upper to ndarray/float as necessary
            upper = format_as_float_or_array('upper', upper, flatten=True)
            # Apply scaler/adder
            upper = (upper + adder) * scaler

        if self._static_mode:
            design_vars = self._static_design_vars
        else:
            design_vars = self._design_vars

        if isinstance(scaler, np.ndarray):
            if np.all(scaler == 1.0):
                scaler = None
        elif scaler == 1.0:
            scaler = None

        if isinstance(adder, np.ndarray):
            if not np.any(adder):
                adder = None
        elif adder == 0.0:
            adder = None

        if indices is not None:
            indices, size = self._create_indexer(indices, 'design var', name,
                                                 flat_src=flat_indices)
        else:
            size = None

        design_vars[name] = {
            'adder': adder,
            'scaler': scaler,
            'name': name,
            'upper': upper,
            'lower': lower,
            'ref': ref,
            'ref0': ref0,
            'units': units,
            'cache_linear_solution': cache_linear_solution,
            'total_scaler': scaler,
            'total_adder': adder,
            'indices': indices,
            'flat_indices': flat_indices,
            'parallel_deriv_color': parallel_deriv_color,
            'size': size,
        }

    def add_response(self, name, type_, lower=None, upper=None, equals=None,
                     ref=None, ref0=None, indices=None, index=None, units=None,
                     adder=None, scaler=None, linear=False, parallel_deriv_color=None,
                     cache_linear_solution=False, flat_indices=None, alias=None):
        r"""
        Add a response variable to this system.

        The response can be scaled using ref and ref0.
        The argument :code:`ref0` represents the physical value when the scaled value is 0.
        The argument :code:`ref` represents the physical value when the scaled value is 1.

        Parameters
        ----------
        name : str
            Promoted name of the response variable in the system.
        type_ : str
            The type of response. Supported values are 'con' and 'obj'.
        lower : float or ndarray, optional
            Lower boundary for the variable.
        upper : upper or ndarray, optional
            Upper boundary for the variable.
        equals : equals or ndarray, optional
            Equality constraint value for the variable.
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
            Value to multiply the model value to get the scaled value for the driver. scaler
            is second in precedence. adder and scaler are an alterantive to using ref
            and ref0.
        linear : bool
            Set to True if constraint is linear. Default is False.
        parallel_deriv_color : str
            If specified, this design var will be grouped for parallel derivative
            calculations with other variables sharing the same parallel_deriv_color.
        cache_linear_solution : bool
            If True, store the linear solution vectors for this variable so they can
            be used to start the next linear solution with an initial guess equal to the
            solution from the previous linear solve.
        flat_indices : bool
            If True, interpret specified indices as being indices into a flat source array.
        alias : str or None
            Alias for this response. Necessary when adding multiple responses on different
            indices of the same variable.
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
            try:
                units = simplify_unit(units, msginfo=self.msginfo)
            except ValueError as e:
                raise ValueError(f"{str(e)[:-1]} for response '{name}'.")

        resp = {}

        if (name in self._responses or name in self._static_responses) and alias is None:
            typemap = {'con': 'Constraint', 'obj': 'Objective'}
            msg = ("{}: {} '{}' already exists. Use the 'alias' argument to apply a second "
                   "constraint".format(self.msginfo, typemap[type_], name))
            raise RuntimeError(msg.format(name))

        resp['name'] = name
        resp['alias'] = alias

        # Convert ref/ref0 to ndarray/float as necessary
        ref = format_as_float_or_array('ref', ref, val_if_none=None, flatten=True)
        ref0 = format_as_float_or_array('ref0', ref0, val_if_none=None, flatten=True)

        # determine adder and scaler based on args
        adder, scaler = determine_adder_scaler(ref0, ref, adder, scaler)

        # A constraint cannot be an equality and inequality constraint
        if equals is not None and (lower is not None or upper is not None):
            msg = "{}: Constraint '{}' cannot be both equality and inequality."
            if alias is not None:
                namestr = f"'{name}' (alias '{alias}')"
            else:
                namestr = name
            raise ValueError(msg.format(self.msginfo, namestr))

        if type_ == 'con':

            # Convert lower to ndarray/float as necessary
            try:
                if lower is None:
                    # don't apply adder/scaler if lower not set
                    lower = -INF_BOUND
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
                    upper = INF_BOUND
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
                indices, size = self._create_indexer(indices, resp_types[type_], name,
                                                     flat_src=flat_indices)
                if size is not None:
                    resp['size'] = size
            resp['indices'] = indices
        else:  # 'obj'
            if index is not None:
                if not isinstance(index, Integral):
                    raise TypeError(f"{self.msginfo}: index must be of integral type, but type is "
                                    f"{type(index).__name__}")
                index = indexer(index, flat_src=flat_indices)
                resp['size'] = 1
            resp['indices'] = index

        if isinstance(scaler, np.ndarray):
            if np.all(scaler == 1.0):
                scaler = None
        elif scaler == 1.0:
            scaler = None
        resp['scaler'] = scaler
        resp['total_scaler'] = scaler

        if isinstance(adder, np.ndarray):
            if not np.any(adder):
                adder = None
        elif adder == 0.0:
            adder = None
        resp['adder'] = adder
        resp['total_adder'] = adder

        resp['ref'] = ref
        resp['ref0'] = ref0
        resp['type'] = type_
        resp['units'] = units
        resp['cache_linear_solution'] = cache_linear_solution
        resp['parallel_deriv_color'] = parallel_deriv_color
        resp['flat_indices'] = flat_indices

        if self._static_mode:
            responses = self._static_responses
        else:
            responses = self._responses

        if alias in responses:
            raise TypeError(f"{self.msginfo}: Constraint alias '{alias}' is a duplicate of an "
                            "existing alias or variable name.")

        if alias is not None:
            responses[alias] = resp
        else:
            responses[name] = resp

    def add_constraint(self, name, lower=None, upper=None, equals=None,
                       ref=None, ref0=None, adder=None, scaler=None, units=None,
                       indices=None, linear=False, parallel_deriv_color=None,
                       cache_linear_solution=False, flat_indices=False, alias=None):
        r"""
        Add a constraint variable to this system.

        Parameters
        ----------
        name : str
            Name of the response variable in the system.
        lower : float or ndarray, optional
            Lower boundary for the variable.
        upper : float or ndarray, optional
            Upper boundary for the variable.
        equals : float or ndarray, optional
            Equality constraint value for the variable.
        ref : float or ndarray, optional
            Value of response variable that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of response variable that scales to 0.0 in the driver.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value for the driver. adder
            is first in precedence.  adder and scaler are an alterantive to using ref
            and ref0.
        scaler : float or ndarray, optional
            Value to multiply the model value to get the scaled value for the driver. scaler
            is second in precedence. adder and scaler are an alternative to using ref
            and ref0.
        units : str, optional
            Units to convert to before applying scaling.
        indices : sequence of int, optional
            If variable is an array, these indicate which entries are of
            interest for this particular response.  These may be positive or
            negative integers.
        linear : bool
            Set to True if constraint is linear. Default is False.
        parallel_deriv_color : str
            If specified, this design var will be grouped for parallel derivative
            calculations with other variables sharing the same parallel_deriv_color.
        cache_linear_solution : bool
            If True, store the linear solution vectors for this variable so they can
            be used to start the next linear solution with an initial guess equal to the
            solution from the previous linear solve.
        flat_indices : bool
            If True, interpret specified indices as being indices into a flat source array.
        alias : str
            Alias for this response. Necessary when adding multiple constraints on different
            indices or slices of a single variable.

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
                          cache_linear_solution=cache_linear_solution,
                          flat_indices=flat_indices, alias=alias)

    def add_objective(self, name, ref=None, ref0=None, index=None, units=None,
                      adder=None, scaler=None, parallel_deriv_color=None,
                      cache_linear_solution=False, flat_indices=False, alias=None):
        r"""
        Add a response variable to this system.

        Parameters
        ----------
        name : str
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
            Value to multiply the model value to get the scaled value for the driver. scaler
            is second in precedence. adder and scaler are an alterantive to using ref
            and ref0.
        parallel_deriv_color : str
            If specified, this design var will be grouped for parallel derivative
            calculations with other variables sharing the same parallel_deriv_color.
        cache_linear_solution : bool
            If True, store the linear solution vectors for this variable so they can
            be used to start the next linear solution with an initial guess equal to the
            solution from the previous linear solve.
        flat_indices : bool
            If True, interpret specified indices as being indices into a flat source array.
        alias : str
            Alias for this response. Necessary when adding multiple objectives on different
            indices or slices of a single variable.

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
                          cache_linear_solution=cache_linear_solution,
                          flat_indices=flat_indices, alias=alias)

    def _update_dv_meta(self, meta, get_size=False, use_prom_ivc=False):
        """
        Update the design variable metadata.

        Parameters
        ----------
        meta : dict
            Metadata dictionary that is populated by this method.
        get_size : bool
            If True, compute the size and store it in the metadata.
        use_prom_ivc : bool
            Determines whether return key is promoted name or source name.
        """
        model = self._problem_meta['model_ref']()
        pro2abs_out = self._var_allprocs_prom2abs_list['output']
        abs2meta_out = model._var_allprocs_abs2meta['output']
        prom_name = meta['name']

        if prom_name in pro2abs_out:  # promoted output
            src_name = pro2abs_out[prom_name][0]
            meta['orig'] = (prom_name, None)

        else:  # Design variable on an input connected to an ivc.
            pro2abs_in = self._var_allprocs_prom2abs_list['input']
            src_name = model._conn_global_abs_in2out[pro2abs_in[prom_name][0]]
            meta['orig'] = (None, prom_name)

        key = prom_name if use_prom_ivc else src_name

        meta['source'] = src_name
        meta['distributed'] = \
            src_name in abs2meta_out and abs2meta_out[src_name]['distributed']

        if get_size:
            if 'indices' not in meta:
                meta['indices'] = None
            abs2idx = model._var_allprocs_abs2idx
            sizes = model._var_sizes['output']

            if src_name in abs2idx:  # var is continuous
                vmeta = abs2meta_out[src_name]
                indices = meta['indices']
                if indices is not None:
                    # Index defined in this design var.
                    # update src shapes for Indexer objects
                    indices.set_src_shape(vmeta['global_shape'])
                    indices = indices.shaped_instance()
                    meta['size'] = meta['global_size'] = indices.indexed_src_size
                else:
                    if meta['distributed']:
                        meta['size'] = sizes[model.comm.rank, abs2idx[src_name]]
                    else:
                        meta['size'] = sizes[model._owning_rank[src_name], abs2idx[src_name]]
                    meta['global_size'] = vmeta['global_size']
            else:
                meta['global_size'] = meta['size'] = 0  # discrete var

        return key

    def _check_voi_meta_sizes(self, typename, name, meta, names):
        """
        Check that sizes of named metadata agree with meta['size'].

        Parameters
        ----------
        typename : str
            'design var', 'objective', or 'constraint'
        name : str
            The name of the variable.  May be an alias.
        meta : dict
            Metadata dictionary.
        names : list of str
            The metadata entries at each of these names must match meta['size'].
        """
        if 'size' in meta and meta['size'] is not None:
            size = meta['size']
            for mname in names:
                val = meta[mname]
                if isinstance(val, np.ndarray) and size != val.size:
                    raise ValueError(f"{self.msginfo}: When adding {typename} '{name}',"
                                     f" {mname} should have size {size} but instead has size "
                                     f"{val.size}.")

    def get_design_vars(self, recurse=True, get_sizes=True, use_prom_ivc=True):
        """
        Get the DesignVariable settings from this system.

        Retrieve all design variable settings from the system and, if recurse
        is True, all of its subsystems.

        Parameters
        ----------
        recurse : bool
            If True, recurse through the subsystems of a group and return the path of
            all design vars relative to the this system.
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
        out = {}
        try:
            for name, data in self._design_vars.items():
                if 'parallel_deriv_color' in data and data['parallel_deriv_color'] is not None:
                    self._problem_meta['has_par_deriv_color'] = True

                key = self._update_dv_meta(data, get_size=get_sizes,
                                           use_prom_ivc=use_prom_ivc)
                if get_sizes and data['source'] in self._var_allprocs_abs2idx:
                    self._check_voi_meta_sizes(
                        'design var', name, data,
                        ['ref', 'ref0', 'scaler', 'adder', 'upper', 'lower'])

                out[key] = data

        except KeyError as err:
            raise RuntimeError(f"{self.msginfo}: Output not found for design variable {err}.")

        return out

    def _update_response_meta(self, meta, get_size=False, use_prom_ivc=False):
        """
        Update the design variable metadata.

        Parameters
        ----------
        meta : dict
            Metadata dictionary.
        get_size : bool
            If True, compute the size of each design variable.
        use_prom_ivc : bool
            Use promoted names for inputs, else convert to absolute source names.
        """
        prom2abs_out = self._var_allprocs_prom2abs_list['output']
        prom2abs_in = self._var_allprocs_prom2abs_list['input']
        model = self._problem_meta['model_ref']()
        conns = model._conn_global_abs_in2out
        abs2meta_out = model._var_allprocs_abs2meta['output']

        alias = meta['alias']
        prom = meta['name']  # 'usually' a promoted name, but can be absolute
        if alias is not None:
            if alias in prom2abs_out or alias in prom2abs_in:
                # Constraint alias should never be the same as any openmdao variable.
                path = prom2abs_out[prom][0] if prom in prom2abs_out else prom
                raise RuntimeError(f"{self.msginfo}: Constraint alias '{alias}' on '{path}'"
                                   " is the same name as an existing variable.")
        meta['parent'] = self.pathname

        if prom in prom2abs_out:  # promoted output
            src_name = prom2abs_out[prom][0]
        elif prom in abs2meta_out:
            src_name = prom
        elif prom in prom2abs_in:
            src_name = conns[prom2abs_in[prom][0]]
        else:  # abs input
            src_name = conns[prom][0]

        if alias:
            key = alias
        elif use_prom_ivc:
            key = prom
        else:
            key = src_name

        meta['source'] = src_name
        meta['distributed'] = dist = \
            src_name in abs2meta_out and abs2meta_out[src_name]['distributed']

        if get_size:
            sizes = model._var_sizes['output']
            abs2idx = model._var_allprocs_abs2idx
            owning_rank = model._owning_rank

            if src_name in abs2idx:
                out_meta = abs2meta_out[src_name]

                if 'indices' in meta and meta['indices'] is not None:
                    indices = meta['indices']
                    indices.set_src_shape(out_meta['global_shape'])
                    indices = indices.shaped_instance()
                    meta['size'] = meta['global_size'] = indices.indexed_src_size
                else:
                    if dist:
                        meta['size'] = sizes[self.comm.rank, abs2idx[src_name]]
                    else:
                        meta['size'] = sizes[owning_rank[src_name], abs2idx[src_name]]
                    meta['global_size'] = out_meta['global_size']
            else:
                meta['size'] = meta['global_size'] = 0  # discrete var, don't know size

        return key

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
        out = {}
        try:
            # keys of self._responses are the alias or the promoted name
            for name, meta in self._responses.items():
                if 'parallel_deriv_color' in meta and meta['parallel_deriv_color'] is not None:
                    self._problem_meta['has_par_deriv_color'] = True

                key = self._update_response_meta(meta, get_size=get_sizes,
                                                 use_prom_ivc=use_prom_ivc)
                if get_sizes:
                    self._check_voi_meta_sizes(
                        resp_types[meta['type']], name, meta, resp_size_checks[meta['type']])

                out[key] = meta

        except KeyError as err:
            raise RuntimeError(f"{self.msginfo}: Output not found for response {err}.")

        return out

    def get_constraints(self, recurse=True, get_sizes=True, use_prom_ivc=True):
        """
        Get the Constraint settings from this system.

        Retrieve the constraint settings for the current system as a dict,
        keyed by variable name.

        Parameters
        ----------
        recurse : bool, optional
            If True, recurse through the subsystems and return the path of
            all constraints relative to the this system.
        get_sizes : bool, optional
            If True, compute the size of each constraint.
        use_prom_ivc : bool
            Translate ivc names to their promoted input names.

        Returns
        -------
        dict
            The constraints defined in the current system.
        """
        return {
            key: response for key, response in self.get_responses(recurse=recurse,
                                                                  get_sizes=get_sizes,
                                                                  use_prom_ivc=use_prom_ivc).items()
            if response['type'] == 'con'
        }

    def get_objectives(self, recurse=True, get_sizes=True, use_prom_ivc=True):
        """
        Get the Objective settings from this system.

        Retrieve all objectives settings from the system as a dict, keyed
        by variable name.

        Parameters
        ----------
        recurse : bool, optional
            If True, recurse through the subsystems and return the path of
            all objective relative to the this system.
        get_sizes : bool, optional
            If True, compute the size of each objective.
        use_prom_ivc : bool
            Translate ivc names to their promoted input names.

        Returns
        -------
        dict
            The objectives defined in the current system.
        """
        return {
            key: response for key, response in self.get_responses(recurse=recurse,
                                                                  get_sizes=get_sizes,
                                                                  use_prom_ivc=use_prom_ivc).items()
            if response['type'] == 'obj'
        }

    def run_apply_nonlinear(self):
        """
        Compute residuals.

        This calls _apply_nonlinear, but with the model assumed to be in an unscaled state.
        """
        with self._scaled_context_all():
            self._apply_nonlinear()

    def get_io_metadata(self, iotypes=('input', 'output'), metadata_keys=None,
                        includes=None, excludes=None, is_indep_var=None, is_design_var=None,
                        tags=None, get_remote=False, rank=None,
                        return_rel_names=True):
        """
        Retrieve metadata for a filtered list of variables.

        Parameters
        ----------
        iotypes : str or iter of str
            Will contain either 'input', 'output', or both.  Defaults to both.
        metadata_keys : iter of str or None
            Names of metadata entries to be retrieved or None, meaning retrieve all
            available 'allprocs' metadata.  If 'val' or 'src_indices' are required,
            their keys must be provided explicitly since they are not found in the 'allprocs'
            metadata and must be retrieved from local metadata located in each process.
        includes : str, iter of str or None
            Collection of glob patterns for pathnames of variables to include. Default is None,
            which includes all variables.
        excludes : str, iter of str or None
            Collection of glob patterns for pathnames of variables to exclude. Default is None.
        is_indep_var : bool or None
            If None (the default), do no additional filtering of the inputs.
            If True, list only inputs connected to an output tagged `openmdao:indep_var`.
            If False, list only inputs _not_ connected to outputs tagged `openmdao:indep_var`.
        is_design_var : bool or None
            If None (the default), do no additional filtering of the inputs.
            If True, list only inputs connected to outputs that are driver design variables.
            If False, list only inputs _not_ connected to outputs that are driver design variables.
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
            always contain two entries, 'prom_name' and 'discrete', to indicate a given
            variable's promoted name and whether or not it is discrete.
        """
        prefix = self.pathname + '.' if self.pathname else ''
        rel_idx = len(prefix)

        # Setup any modified subsystems so the metadata dicts are up-to-date.
        if self._problem_meta:
            conf_info = self._problem_meta['config_info']
            if conf_info:
                if self._subsystems_allprocs:
                    conf_info._update_modified_systems(self)
                if self.pathname in conf_info._modified_systems:
                    self._setup_var_data()

        if isinstance(iotypes, str):
            iotypes = (iotypes,)
        if isinstance(includes, str):
            includes = (includes,)
        if isinstance(excludes, str):
            excludes = (excludes,)
        if isinstance(tags, str):
            tags = {tags}

        gather_keys = {'val', 'src_indices'}
        need_gather = get_remote and self.comm is not None and self.comm.size > 1
        if metadata_keys is not None:
            keyset = set(metadata_keys)
            diff = keyset - allowed_meta_names
            if diff:
                raise RuntimeError(f"{self.msginfo}: {sorted(diff)} are not valid metadata entry "
                                   "names.")
        need_local_meta = metadata_keys is not None and len(gather_keys.intersection(keyset)) > 0

        all2meta = self._var_allprocs_abs2meta
        if need_local_meta:
            metadict = self._var_abs2meta
            disc_metadict = self._var_discrete
        else:
            metadict = all2meta
            disc_metadict = self._var_allprocs_discrete
            need_gather = False  # we can get everything from 'allprocs' dict without gathering

        result = {}

        it = self._var_allprocs_abs2prom if get_remote else self._var_abs2prom

        if is_design_var is not None:
            des_vars = self.get_design_vars(get_sizes=False, use_prom_ivc=False)

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
                        ret_meta = dict(meta)
                    else:
                        ret_meta = {}
                        for key in keyset:
                            try:
                                ret_meta[key] = meta[key]
                            except KeyError:
                                ret_meta[key] = NA

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
                                if 'val' in keyset:
                                    # assemble the full distributed value
                                    dist_vals = [m['val'] for m in allproc_metas
                                                 if m is not None and m['val'].size > 0]
                                    if dist_vals:
                                        ret_meta['val'] = np.concatenate(dist_vals)
                                    else:
                                        ret_meta['val'] = np.zeros(0)
                                if 'src_indices' in keyset:
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
                    # handle is_indep_var
                    if is_indep_var is not None:
                        if iotype == 'output':
                            out_meta = meta
                        else:
                            src_name = self.get_source(abs_name)
                            try:
                                out_meta = metadict['output'][src_name]
                            except KeyError:
                                out_meta = disc_metadict['output'][src_name]

                        src_tags = out_meta['tags'] if 'tags' in out_meta else {}
                        if is_indep_var:
                            if 'openmdao:indep_var' not in src_tags:
                                continue
                        elif 'openmdao:indep_var' in src_tags:
                            continue

                    # handle is_design_var
                    if is_design_var is not None:
                        if iotype == 'output':
                            out_name = abs_name
                        else:
                            out_name = self.get_source(abs_name)
                        if is_design_var:
                            if out_name not in des_vars:
                                continue
                        elif out_name in des_vars:
                            continue

                    # handle tags
                    if tags:
                        meta_tags = ret_meta.get('tags', {})
                        match_tag = False
                        for tag in tags:
                            for meta_tag in meta_tags:
                                if fnmatchcase(meta_tag, tag):
                                    match_tag = True
                                    break
                        if not match_tag:
                            continue

                    ret_meta['prom_name'] = prom
                    ret_meta['discrete'] = abs_name not in all2meta[iotype]

                    if return_rel_names:
                        result[rel_name] = ret_meta
                    else:
                        result[abs_name] = ret_meta

        return result

    def list_vars(self,
                  val=True,
                  prom_name=True,
                  residuals=False,
                  residuals_tol=None,
                  units=False,
                  shape=False,
                  global_shape=False,
                  bounds=False,
                  scaling=False,
                  desc=False,
                  print_arrays=False,
                  tags=None,
                  print_tags=False,
                  includes=None,
                  excludes=None,
                  is_indep_var=None,
                  is_design_var=None,
                  all_procs=False,
                  list_autoivcs=False,
                  out_stream=_DEFAULT_OUT_STREAM,
                  print_min=False,
                  print_max=False,
                  return_format='list'):
        """
        Write a list of inputs and outputs sorted by component in execution order.

        Parameters
        ----------
        val : bool, optional
            When True, display output values. Default is True.
        prom_name : bool, optional
            When True, display the promoted name of the variable.
            Default is True.
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
        print_tags : bool
            When true, display tags in the columnar display.
        includes : None, str, or iter of str
            Collection of glob patterns for pathnames of variables to include. Default is None,
            which includes all output variables.
        excludes : None, str, or iter of str
            Collection of glob patterns for pathnames of variables to exclude. Default is None.
        is_indep_var : bool or None
            If None (the default), do no additional filtering of the inputs.
            If True, list only outputs tagged `openmdao:indep_var`.
            If False, list only outputs that are _not_ tagged `openmdao:indep_var`.
        is_design_var : bool or None
            If None (the default), do no additional filtering of the inputs.
            If True, list only inputs connected to outputs that are driver design variables.
            If False, list only inputs _not_ connected to outputs that are driver design variables.
        all_procs : bool, optional
            When True, display output on all processors. Default is False.
        list_autoivcs : bool
            If True, include auto_ivc outputs in the listing.  Defaults to False.
        out_stream : file-like
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.
        print_min : bool
            When true, if the output value is an array, print its smallest value.
        print_max : bool
            When true, if the output value is an array, print its largest value.
        return_format : str
            Indicates the desired format of the return value. Can have value of 'list' or 'dict'.
            If 'list', the return value is a list of (name, metadata) tuples.
            if 'dict', the return value is a dictionary mapping {name: metadata}.

        Returns
        -------
        list of (name, metadata) or dict of {name: metadata}
            List or dict of output names and other optional information about those outputs.
        """
        if (self._problem_meta is None or
                self._problem_meta['setup_status'] < _SetupStatus.POST_FINAL_SETUP) and val:
            issue_warning("Calling `list_vars` before `final_setup` will only "
                          "display the default values of variables and will not show the result of "
                          "any `set_val` calls.")

        if return_format not in ('list', 'dict'):
            badarg = f"'{return_format}'" if isinstance(return_format, str) else f"{return_format}"
            raise ValueError(f"Invalid value ({badarg}) for return_format, "
                             "must be a string value of 'list' or 'dict'")

        keynames = ['val', 'units', 'shape', 'global_shape', 'desc', 'tags']
        keyflags = [val, units, shape, global_shape, desc, tags or print_tags]

        keys = [name for i, name in enumerate(keynames) if keyflags[i]]

        if bounds:
            keys.extend(('lower', 'upper'))
        if scaling:
            keys.extend(('ref', 'ref0', 'res_ref'))

        outputs = self.get_io_metadata(('output',), keys, includes, excludes,
                                       is_indep_var, is_design_var, tags,
                                       get_remote=True,
                                       rank=None if all_procs or val or residuals else 0,
                                       return_rel_names=False)

        metavalues = val and self._inputs is None

        keyvals = [metavalues, units, shape, global_shape, desc, tags or print_tags]
        keys = [n for i, n in enumerate(keynames) if keyvals[i]]

        inputs = self.get_io_metadata(('input',), keys, includes, excludes,
                                      is_indep_var, is_design_var, tags,
                                      get_remote=True,
                                      rank=None if all_procs or val else 0,
                                      return_rel_names=False)

        # filter auto_ivcs if requested
        if outputs and not list_autoivcs:
            outputs = {n: m for n, m in outputs.items() if not n.startswith('_auto_ivc.')}

        # get values & resids
        if self._outputs is not None and (val or residuals or residuals_tol):
            to_remove = []
            print_options = np.get_printoptions()
            np_precision = print_options['precision']

            for name, meta in outputs.items():
                if val:
                    # we want value from the input vector, not from the metadata
                    meta['val'] = self._abs_get_val(name, get_remote=True,
                                                    rank=None if all_procs else 0, kind='output')

                    if isinstance(meta['val'], np.ndarray):
                        if print_min:
                            meta['min'] = np.round(np.min(meta['val']), np_precision)

                        if print_max:
                            meta['max'] = np.round(np.max(meta['val']), np_precision)

                if residuals or residuals_tol:
                    resids = self._abs_get_val(name, get_remote=True,
                                               rank=None if all_procs else 0,
                                               kind='residual')
                    if residuals_tol and np.linalg.norm(resids) < residuals_tol:
                        to_remove.append(name)
                    elif residuals:
                        meta['resids'] = resids

            # remove any outputs that don't pass the residuals_tol filter
            for name in to_remove:
                del outputs[name]

        if val and self._inputs is not None:
            # we want value from the input vector, not from the metadata
            print_options = np.get_printoptions()
            np_precision = print_options['precision']

            for n, meta in inputs.items():
                meta['val'] = self._abs_get_val(n, get_remote=True,
                                                rank=None if all_procs else 0, kind='input')
                if isinstance(meta['val'], np.ndarray):
                    if print_min:
                        meta['min'] = np.round(np.min(meta['val']), np_precision)

                    if print_max:
                        meta['max'] = np.round(np.max(meta['val']), np_precision)

        # NOTE: calls to _abs_get_val() above are collective calls and must be done on all procs
        if not (outputs or inputs) or (not all_procs and self.comm.rank != 0):
            return {} if return_format == 'dict' else []

        # remove metadata we don't want to show/return
        to_remove = ['discrete']
        if not print_tags:
            to_remove.append('tags')
        if not prom_name:
            to_remove.append('prom_name')
        for _, meta in chain(outputs.items(), inputs.items()):
            for key in to_remove:
                try:
                    del meta[key]
                except KeyError:
                    pass

        variables = set(outputs.keys()).union(set(inputs.keys()))
        var_list = []
        var_dict = {}

        var_list = self._get_vars_exec_order(inputs=True, outputs=True,
                                             variables=variables, local=True)
        for var_name in var_list:
            if var_name in outputs:
                var_dict[var_name] = outputs[var_name]
                var_dict[var_name]['io'] = 'output'
            else:
                var_dict[var_name] = inputs[var_name]
                var_dict[var_name]['io'] = 'input'

        if all_procs or self.comm.rank == 0:
            write_var_table(self.pathname, var_list, 'all', var_dict,
                            True, print_arrays, out_stream)

        return var_dict if return_format == 'dict' else list(var_dict.items())

    def list_inputs(self,
                    val=True,
                    prom_name=True,
                    units=False,
                    shape=False,
                    global_shape=False,
                    desc=False,
                    hierarchical=True,
                    print_arrays=False,
                    tags=None,
                    print_tags=False,
                    includes=None,
                    excludes=None,
                    is_indep_var=None,
                    is_design_var=None,
                    all_procs=False,
                    out_stream=_DEFAULT_OUT_STREAM,
                    print_min=False,
                    print_max=False,
                    return_format='list'):
        """
        Write a list of input names and other optional information to a specified stream.

        Parameters
        ----------
        val : bool, optional
            When True, display/return input values. Default is True.
        prom_name : bool, optional
            When True, display/return the promoted name of the variable.
            Default is True.
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
        print_tags : bool
            When true, display tags in the columnar display.
        includes : None, str, or iter of str
            Collection of glob patterns for pathnames of variables to include. Default is None,
            which includes all input variables.
        excludes : None, str, or iter of str
            Collection of glob patterns for pathnames of variables to exclude. Default is None.
        is_indep_var : bool or None
            If None (the default), do no additional filtering of the inputs.
            If True, list only inputs connected to an output tagged `openmdao:indep_var`.
            If False, list only inputs _not_ connected to outputs tagged `openmdao:indep_var`.
        is_design_var : bool or None
            If None (the default), do no additional filtering of the inputs.
            If True, list only inputs connected to outputs that are driver design variables.
            If False, list only inputs _not_ connected to outputs that are driver design variables.
        all_procs : bool, optional
            When True, display output on all ranks. Default is False, which will display
            output only from rank 0.
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.
        print_min : bool
            When true, if the input value is an array, print its smallest value.
        print_max : bool
            When true, if the input value is an array, print its largest value.
        return_format : str
            Indicates the desired format of the return value. Can have value of 'list' or 'dict'.
            If 'list', the return value is a list of (name, metadata) tuples.
            if 'dict', the return value is a dictionary mapping {name: metadata}.

        Returns
        -------
        list of (name, metadata) or dict of {name: metadata}
            List or dict of input names and other optional information about those inputs.
        """
        if (self._problem_meta is None or
                self._problem_meta['setup_status'] < _SetupStatus.POST_FINAL_SETUP) and val:
            issue_warning("Calling `list_inputs` before `final_setup` will only "
                          "display the default values of variables and will not show the result of "
                          "any `set_val` calls.")

        if return_format not in ('list', 'dict'):
            raise ValueError(f"Invalid value ({return_format}) for return_format, "
                             "must be a string value of 'list' or 'dict'")

        metavalues = val and self._inputs is None

        keynames = ['val', 'units', 'shape', 'global_shape', 'desc', 'tags']
        keyvals = [metavalues, units, shape, global_shape, desc, tags or print_tags]
        keys = [n for i, n in enumerate(keynames) if keyvals[i]]

        inputs = self.get_io_metadata(('input',), keys, includes, excludes,
                                      is_indep_var, is_design_var, tags,
                                      get_remote=True,
                                      rank=None if all_procs or val else 0,
                                      return_rel_names=False)

        if val and self._inputs is not None:
            # we want value from the input vector, not from the metadata
            print_options = np.get_printoptions()
            np_precision = print_options['precision']

            for n, meta in inputs.items():
                meta['val'] = self._abs_get_val(n, get_remote=True,
                                                rank=None if all_procs else 0, kind='input')
                if isinstance(meta['val'], np.ndarray):
                    if print_min:
                        meta['min'] = np.round(np.min(meta['val']), np_precision)

                    if print_max:
                        meta['max'] = np.round(np.max(meta['val']), np_precision)

        # NOTE: calls to _abs_get_val() above are collective calls and must be done on all procs
        if not inputs or (not all_procs and self.comm.rank != 0):
            return {} if return_format == 'dict' else []

        to_remove = ['discrete']
        if not print_tags:
            to_remove.append('tags')
        if not prom_name:
            to_remove.append('prom_name')
        for _, meta in inputs.items():
            for key in to_remove:
                try:
                    del meta[key]
                except KeyError:
                    pass

        if out_stream:
            self._write_table('input', inputs, hierarchical, print_arrays, all_procs,
                              out_stream)

        if self.pathname:
            # convert to relative names
            rel_idx = len(self.pathname) + 1
            inputs = [(n[rel_idx:], meta) for n, meta in inputs.items()]
        else:
            inputs = list(inputs.items())

        return dict(inputs) if return_format == 'dict' else inputs

    def list_outputs(self,
                     explicit=True, implicit=True,
                     val=True,
                     prom_name=True,
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
                     print_tags=False,
                     includes=None,
                     excludes=None,
                     is_indep_var=None,
                     is_design_var=None,
                     all_procs=False,
                     list_autoivcs=False,
                     out_stream=_DEFAULT_OUT_STREAM,
                     print_min=False,
                     print_max=False,
                     return_format='list'):
        """
        Write a list of output names and other optional information to a specified stream.

        Parameters
        ----------
        explicit : bool, optional
            Include outputs from explicit components. Default is True.
        implicit : bool, optional
            Include outputs from implicit components. Default is True.
        val : bool, optional
            When True, display output values. Default is True.
        prom_name : bool, optional
            When True, display the promoted name of the variable.
            Default is True.
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
        print_tags : bool
            When true, display tags in the columnar display.
        includes : None, str, or iter of str
            Collection of glob patterns for pathnames of variables to include. Default is None,
            which includes all output variables.
        excludes : None, str, or iter of str
            Collection of glob patterns for pathnames of variables to exclude. Default is None.
        is_indep_var : bool or None
            If None (the default), do no additional filtering of the inputs.
            If True, list only outputs tagged `openmdao:indep_var`.
            If False, list only outputs that are _not_ tagged `openmdao:indep_var`.
        is_design_var : bool or None
            If None (the default), do no additional filtering of the inputs.
            If True, list only inputs connected to outputs that are driver design variables.
            If False, list only inputs _not_ connected to outputs that are driver design variables.
        all_procs : bool, optional
            When True, display output on all processors. Default is False.
        list_autoivcs : bool
            If True, include auto_ivc outputs in the listing.  Defaults to False.
        out_stream : file-like
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.
        print_min : bool
            When true, if the output value is an array, print its smallest value.
        print_max : bool
            When true, if the output value is an array, print its largest value.
        return_format : str
            Indicates the desired format of the return value. Can have value of 'list' or 'dict'.
            If 'list', the return value is a list of (name, metadata) tuples.
            if 'dict', the return value is a dictionary mapping {name: metadata}.

        Returns
        -------
        list of (name, metadata) or dict of {name: metadata}
            List or dict of output names and other optional information about those outputs.
        """
        if return_format not in ('list', 'dict'):
            badarg = f"'{return_format}'" if isinstance(return_format, str) else f"{return_format}"
            raise ValueError(f"Invalid value ({badarg}) for return_format, "
                             "must be a string value of 'list' or 'dict'")

        keynames = ['val', 'units', 'shape', 'global_shape', 'desc', 'tags']
        keyflags = [val, units, shape, global_shape, desc, tags or print_tags]

        keys = [name for i, name in enumerate(keynames) if keyflags[i]]

        if bounds:
            keys.extend(('lower', 'upper'))
        if scaling:
            keys.extend(('ref', 'ref0', 'res_ref'))

        outputs = self.get_io_metadata(('output',), keys, includes, excludes,
                                       is_indep_var, is_design_var, tags,
                                       get_remote=True,
                                       rank=None if all_procs or val or residuals else 0,
                                       return_rel_names=False)

        # filter auto_ivcs if requested
        if outputs and not list_autoivcs:
            outputs = {n: m for n, m in outputs.items() if not n.startswith('_auto_ivc.')}

        # get values & resids
        if self._outputs is not None and (val or residuals or residuals_tol):
            to_remove = []
            print_options = np.get_printoptions()
            np_precision = print_options['precision']

            for name, meta in outputs.items():
                if val:
                    # we want value from the input vector, not from the metadata
                    meta['val'] = self._abs_get_val(name, get_remote=True,
                                                    rank=None if all_procs else 0, kind='output')

                    if isinstance(meta['val'], np.ndarray):
                        if print_min:
                            meta['min'] = np.round(np.min(meta['val']), np_precision)

                        if print_max:
                            meta['max'] = np.round(np.max(meta['val']), np_precision)

                if residuals or residuals_tol:
                    resids = self._abs_get_val(name, get_remote=True,
                                               rank=None if all_procs else 0,
                                               kind='residual')
                    if residuals_tol and np.linalg.norm(resids) < residuals_tol:
                        to_remove.append(name)
                    elif residuals:
                        meta['resids'] = resids

            # remove any outputs that don't pass the residuals_tol filter
            for name in to_remove:
                del outputs[name]

        # NOTE: calls to _abs_get_val() above are collective calls and must be done on all procs
        if not outputs or (not all_procs and self.comm.rank != 0):
            return {} if return_format == 'dict' else []

        # remove metadata we don't want to show/return
        to_remove = ['discrete']
        if not print_tags:
            to_remove.append('tags')
        if not prom_name:
            to_remove.append('prom_name')
        for _, meta in outputs.items():
            for key in to_remove:
                try:
                    del meta[key]
                except KeyError:
                    pass

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
                    if n in states:
                        if residuals_tol and 'resids' in m:
                            if np.linalg.norm(m['resids']) >= residuals_tol:
                                impl_outputs[n] = m
                        else:
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

        if explicit and implicit:
            outputs = expl_outputs + impl_outputs
        elif explicit:
            outputs = expl_outputs
        elif implicit:
            outputs = impl_outputs
        else:
            raise RuntimeError('You have excluded both Explicit and Implicit components.')

        return dict(outputs) if return_format == 'dict' else outputs

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
        else:
            inputs = var_type == 'input'
            outputs = not inputs
            var_list = self._get_vars_exec_order(inputs=inputs, outputs=outputs, variables=var_data)

        if all_procs or self.comm.rank == 0:
            write_var_table(self.pathname, var_list, var_type, var_data,
                            hierarchical, print_arrays, out_stream)

    def _get_vars_exec_order(self, inputs=False, outputs=False, variables=None, local=False):
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
        local : bool, optional
            Get variables from local subsystems only. Default is False.

        Returns
        -------
        list
            list of variable names in execution order
        """
        var_list = []

        real_vars = self._var_allprocs_abs2meta
        disc_vars = self._var_allprocs_discrete

        # variable order: real inputs, real outputs, discrete inputs, discrete outputs
        var_dicts = []
        if inputs:
            var_dicts.append(real_vars['input'])
        if outputs:
            var_dicts.append(real_vars['output'])
        if inputs:
            var_dicts.append(disc_vars['input'])
        if outputs:
            var_dicts.append(disc_vars['output'])

        # For components with no children, self._subsystems_allprocs is empty.
        if self._subsystems_allprocs:
            if local:
                from openmdao.core.component import Component
                it = self.system_iter(recurse=True, typ=Component)
            else:
                it = iter(subsys for subsys, _ in self._subsystems_allprocs.values())

            for subsys in it:
                prefix = subsys.pathname + '.'
                for var_name in chain(*var_dicts):
                    if not variables or var_name in variables:
                        if var_name.startswith(prefix):
                            var_list.append(var_name)
        else:
            for var_name in chain(*var_dicts):
                if variables is None or var_name in variables:
                    var_list.append(var_name)

        return var_list

    def run_solve_nonlinear(self):
        """
        Compute outputs.

        This calls _solve_nonlinear, but with the model assumed to be in an unscaled state.
        """
        with self._scaled_context_all():
            self._solve_nonlinear()

    def run_apply_linear(self, mode, scope_out=None, scope_in=None):
        """
        Compute jac-vec product.

        This calls _apply_linear, but with the model assumed to be in an unscaled state.

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
        with self._scaled_context_all():
            self._apply_linear(None, mode, scope_out, scope_in)

    def run_solve_linear(self, mode):
        """
        Apply inverse jac product.

        This calls _solve_linear, but with the model assumed to be in an unscaled state.

        Parameters
        ----------
        mode : str
            'fwd' or 'rev'.
        """
        with self._scaled_context_all():
            self._solve_linear(mode)

    def run_linearize(self, sub_do_ln=True):
        """
        Compute jacobian / factorization.

        This calls _linearize, but with the model assumed to be in an unscaled state.

        Parameters
        ----------
        sub_do_ln : bool
            Flag indicating if the children should call linearize on their linear solvers.
        """
        with self._scaled_context_all():
            self._linearize(self._assembled_jac, sub_do_ln=self._linear_solver is not None and
                            self._linear_solver._linearize_children())
            if self._linear_solver is not None and sub_do_ln:
                self._linear_solver._linearize()

    def _apply_nonlinear(self):
        """
        Compute residuals. The model is assumed to be in a scaled state.
        """
        raise NotImplementedError(self.msginfo + ": _apply_nonlinear has not been overridden")

    def check_config(self, logger):
        """
        Perform optional error checks.

        Parameters
        ----------
        logger : object
            The object that manages logging output.
        """
        pass

    def _iter_call_apply_linear(self):
        """
        Return whether to call _apply_linear on this System from within parent _apply_linear.

        Returns
        -------
        bool
            True if _apply_linear should be called from within a parent _apply_linear.
        """
        return True

    def _apply_linear(self, jac, mode, scope_in=None, scope_out=None):
        """
        Compute jac-vec product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            If None, use local jacobian, else use assembled jacobian jac.
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
        pass

    def _linearize(self, jac, sub_do_ln=True):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            If None, use local jacobian, else use assembled jacobian jac.
        sub_do_ln : bool
            Flag indicating if the children should call linearize on their linear solvers.
        """
        raise NotImplementedError(self.msginfo + ": _linearize has not been overridden")

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

    def load_model_options(self):
        """
        Load the relevant model options from `Problem._metadata['model_options']`.

        This method examines each path filter and corresponding options in
        self._problem_meta['model_options']. If this System's pathname matches
        the given path filter, it will assume the value for each given option
        which it possesses.
        """
        model_options = self._problem_meta['model_options']
        for path_filter, path_options in model_options.items():
            if fnmatchcase(self.pathname, path_filter):
                for option, val in path_options.items():
                    if option in self.options:
                        self.options[option] = val

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
            if self._recording_iter.stack:
                stack_top = self._recording_iter.stack[-1][0]
                method = stack_top.rsplit('.', 1)[-1]

                if method not in _recordable_funcs:
                    raise ValueError(f"{self.msginfo}: {method} must be one of: "
                                     f"{sorted(_recordable_funcs)}")

                if 'nonlinear' in method:
                    inputs, outputs, residuals = self.get_nonlinear_vectors()
                    vec_name = 'nonlinear'
                else:
                    inputs, outputs, residuals = self.get_linear_vectors()
                    vec_name = 'linear'
            else:
                # outside of a run, just record nonlinear vectors
                inputs, outputs, residuals = self.get_nonlinear_vectors()
                vec_name = 'nonlinear'

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
                if nl.linesearch:
                    nl.linesearch._iter_count = 0

    def get_reports_dir(self):
        """
        Get the path to the directory where the report files should go.

        If it doesn't exist, it will be created.

        Returns
        -------
        str
            The path to the directory where reports should be written.
        """
        return self._problem_meta['reports_dir']

    def _set_finite_difference_mode(self, active):
        """
        Turn on or off finite difference mode.

        Recurses to turn on or off finite difference mode in all subsystems.

        Parameters
        ----------
        active : bool
            Finite difference flag; set to True prior to commencing finite difference.
        """
        for sub in self.system_iter(include_self=True, recurse=True):
            sub.under_finite_difference = active

    def _set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        Recurses to turn on or off complex stepping mode in all subsystems and their vectors.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        self.under_complex_step = active
        self._inputs.set_complex_step_mode(active)
        self._outputs.set_complex_step_mode(active)
        self._residuals.set_complex_step_mode(active)

        if self._doutputs._alloc_complex:
            self._doutputs.set_complex_step_mode(active)
            self._dinputs.set_complex_step_mode(active)
            self._dresiduals.set_complex_step_mode(active)
            if self.nonlinear_solver:
                self.nonlinear_solver._set_complex_step_mode(active)

            if self.linear_solver:
                self.linear_solver._set_complex_step_mode(active)

            if isinstance(self._jacobian, Jacobian):
                self._jacobian.set_complex_step_mode(active)

            if self._assembled_jac:
                self._assembled_jac.set_complex_step_mode(active)

        for sub in self._subsystems_myproc:
            sub._set_complex_step_mode(active)

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
        for tup in var_info:
            lst = list(tup)
            if tup[0] in abs2prom_out:
                lst[0] = abs2prom_out[tup[0]]
            else:
                lst[0] = abs2prom_in[tup[0]]
            new_list.append(lst)
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

        vars_to_gather = self._problem_meta['vars_to_gather']

        # if abs_name is non-discrete it should be found in all_meta
        if abs_name in all_meta:
            if get_remote:
                meta = all_meta[abs_name]
                distrib = meta['distributed']
            elif self.comm.size > 1:
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
                    val = my_meta[abs_name]['val']
            else:
                if from_root:
                    vec = vec._root_vector
                if vec._contains_abs(abs_name):
                    val = vec._abs_get_val(abs_name, flat)

        if get_remote and (distrib or abs_name in vars_to_gather) and self.comm.size > 1:
            owner = self._owning_rank[abs_name]
            myrank = self.comm.rank
            if rank is None:  # bcast
                if distrib:
                    idx = self._var_allprocs_abs2idx[abs_name]
                    sizes = self._var_sizes[typ][:, idx]
                    # TODO: could cache these offsets
                    offsets = np.zeros(sizes.size, dtype=INT_DTYPE)
                    offsets[1:] = np.cumsum(sizes[:-1])
                    if val is _UNDEFINED:
                        loc_val = np.zeros(sizes[myrank])
                    else:
                        loc_val = np.ascontiguousarray(val)
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
            else:  # retrieve to rank
                if distrib:
                    idx = self._var_allprocs_abs2idx[abs_name]
                    sizes = self._var_sizes[typ][:, idx]
                    # TODO: could cache these offsets
                    offsets = np.zeros(sizes.size, dtype=INT_DTYPE)
                    offsets[1:] = np.cumsum(sizes[:-1])
                    if val is _UNDEFINED:
                        loc_val = np.zeros(sizes[idx])
                    else:
                        loc_val = np.ascontiguousarray(val)
                    val = np.zeros(np.sum(sizes))
                    self.comm.Gatherv(loc_val, [val, sizes, offsets, MPI.DOUBLE], root=rank)
                    if not flat:
                        val.shape = meta['global_shape'] if get_remote else meta['shape']
                else:
                    if rank != owner:
                        tag = self._var_allprocs_abs2idx[abs_name]
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
        simp_units = simplify_unit(units)

        if from_src:
            conns = self._problem_meta['model_ref']()._conn_global_abs_in2out
        else:
            conns = []
        if from_src and abs_names[0] in conns:  # pull input from source
            src = conns[abs_names[0]]
            if src in self._var_allprocs_abs2prom['output']:
                caller = self
            else:
                # src is outside of this system so get the value from the model
                caller = self._problem_meta['model_ref']()
            return caller._get_input_from_src(name, abs_names, conns, units=simp_units,
                                              indices=indices, get_remote=get_remote, rank=rank,
                                              vec_name='nonlinear', flat=flat, scope_sys=self)
        else:
            val = self._abs_get_val(abs_names[0], get_remote, rank, vec_name, kind, flat)

            if indices is not None:
                val = val[indices]

            if units is not None:
                val = self.convert2units(abs_names[0], val, simp_units)

        return val

    def _get_cached_val(self, name, abs_names, get_remote=False):
        # We have set and cached already
        for abs_name in abs_names:
            if abs_name in self._initial_condition_cache:
                return self._initial_condition_cache[abs_name][0]

        # Vector not setup, so we need to pull values from saved metadata request.
        model = self._problem_meta['model_ref']()

        try:
            conns = model._conn_abs_in2out
        except AttributeError:
            conns = {}

        abs_name = abs_names[0]
        vars_to_gather = self._problem_meta['vars_to_gather']
        units = None

        meta = model._var_abs2meta
        io = 'output' if abs_name in meta['output'] else 'input'
        if abs_name in meta[io]:
            if abs_name in conns:
                smeta = meta['output'][conns[abs_name]]
                val = smeta['val']  # output
                units = smeta['units']
            else:
                vmeta = meta[io][abs_name]
                val = vmeta['val']
                units = vmeta['units']
        else:
            # not found in real outputs or inputs, try discretes
            meta = model._var_discrete
            io = 'output' if abs_name in meta['output'] else 'input'
            if abs_name in meta[io]:
                if abs_name in conns:
                    val = meta['output'][conns[abs_name]]['val']
                else:
                    val = meta[io][abs_name]['val']

        if get_remote and abs_name in vars_to_gather:
            owner = vars_to_gather[abs_name]
            if model.comm.rank == owner:
                model.comm.bcast(val, root=owner)
            else:
                val = model.comm.bcast(None, root=owner)

        if val is not _UNDEFINED:
            # Need to cache the "get" in case the user calls in-place numpy operations.
            self._initial_condition_cache[abs_name] = (val, units, self.pathname, name)

        return val

    def set_val(self, name, val, units=None, indices=None):
        """
        Set an input or output variable.

        Parameters
        ----------
        name : str
            Promoted or relative variable name in the system's namespace.
        val : object
            Value to assign to this variable.
        units : str, optional
            Units of the value.
        indices : int or list of ints or tuple of ints or int ndarray or Iterable or None, optional
            Indices or slice to set.
        """
        post_setup = self._problem_meta is not None and \
            self._problem_meta['setup_status'] >= _SetupStatus.POST_SETUP
        if post_setup:
            abs_names = name2abs_names(self, name)
        else:
            raise RuntimeError(f"{self.msginfo}: Called set_val({name}, ...) before setup "
                               "completes.")

        has_vectors = self._problem_meta['setup_status'] >= _SetupStatus.POST_FINAL_SETUP
        value = val

        model = self._problem_meta['model_ref']()
        conns = model._conn_global_abs_in2out

        all_meta = model._var_allprocs_abs2meta
        loc_meta = model._var_abs2meta
        n_proms = 0  # if nonzero, name given was promoted input name w/o a matching prom output

        try:
            ginputs = self._group_inputs
        except AttributeError:
            ginputs = {}  # could happen if this system is not a Group

        if abs_names:
            n_proms = len(abs_names)  # for output this will never be > 1
            if n_proms > 1 and name in ginputs:
                abs_name = ginputs[name][0].get('use_tgt', abs_names[0])
            else:
                abs_name = abs_names[0]
        else:
            raise KeyError(f'{model.msginfo}: Variable "{name}" not found.')

        set_units = None

        if abs_name in conns:  # we're setting an input
            src = conns[abs_name]
            if abs_name not in model._var_allprocs_discrete['input']:  # input is continuous
                value = np.asarray(value)
                tmeta = all_meta['input'][abs_name]
                tunits = tmeta['units']
                sunits = all_meta['output'][src]['units']
                if abs_name in loc_meta['input']:
                    tlocmeta = loc_meta['input'][abs_name]
                else:
                    tlocmeta = None

                gunits = ginputs[name][0].get('units') if name in ginputs else None
                if n_proms > 1:  # promoted input name was used
                    if gunits is None:
                        tunit_list = [all_meta['input'][n]['units'] for n in abs_names]
                        tu0 = tunit_list[0]
                        for tu in tunit_list:
                            if tu != tu0:
                                model._show_ambiguity_msg(name, ('units',), abs_names)

                if units is None:
                    # avoids double unit conversion
                    ivalue = value
                    if sunits is not None:
                        if gunits is not None and gunits != tunits:
                            value = model.convert_from_units(src, value, gunits)
                        elif tunits is not None:
                            value = model.convert_from_units(src, value, tunits)
                        else:
                            msg = f"A value with no units has been specified for input " + \
                                  f"'{name}', but the source ('{src}') has units '{sunits}'. " + \
                                  f"No unit checking can be done."
                            issue_warning(msg, prefix=self.msginfo, category=UnitsWarning)
                else:
                    if gunits is None:
                        ivalue = model.convert_from_units(abs_name, value, units)
                    else:
                        ivalue = model.convert_units(name, value, units, gunits)
                    value = model.convert_from_units(src, value, units)
                set_units = sunits
        else:
            src = abs_name
            if units is not None:
                value = model.convert_from_units(abs_name, value, units)
                try:
                    set_units = all_meta['output'][abs_name]['units']
                except KeyError:  # this can happen if a component is the top level System
                    set_units = all_meta['input'][abs_name]['units']

        # Caching only needed if vectors aren't allocated yet.
        if not has_vectors:
            ic_cache = model._initial_condition_cache
            if indices is not None:
                self._get_cached_val(name, abs_names)
                try:
                    cval = ic_cache[abs_name][0]
                    if _is_slicer_op(indices):
                        try:
                            ic_cache[abs_name] = (value[indices], set_units, self.pathname, name)
                        except IndexError:
                            cval[indices] = value
                            ic_cache[abs_name] = (cval, set_units, self.pathname, name)
                    else:
                        cval[indices] = value
                        ic_cache[abs_name] = (cval, set_units, self.pathname, name)
                except Exception as err:
                    raise RuntimeError(f"Failed to set value of '{name}': {str(err)}.")
            else:
                ic_cache[abs_name] = (value, set_units, self.pathname, name)
        else:
            myrank = model.comm.rank

            if indices is None:
                indices = _full_slice

            if model._outputs._contains_abs(abs_name):
                distrib = all_meta['output'][abs_name]['distributed']
                if (distrib and indices is _full_slice and
                        value.size == all_meta['output'][abs_name]['global_size']):
                    # assume user is setting using full distributed value
                    sizes = model._var_sizes['output'][:, model._var_allprocs_abs2idx[abs_name]]
                    start = np.sum(sizes[:myrank])
                    end = start + sizes[myrank]
                    model._outputs.set_var(abs_name, value[start:end], indices)
                else:
                    model._outputs.set_var(abs_name, value, indices)
            elif abs_name in conns:  # input name given. Set value into output
                src_is_auto_ivc = src.startswith('_auto_ivc.')
                # when setting auto_ivc output, error messages should refer
                # to the promoted name used in the set_val call
                var_name = name if src_is_auto_ivc else src
                if model._outputs._contains_abs(src):  # src is local
                    if (model._outputs._abs_get_val(src).size == 0 and
                            src_is_auto_ivc and
                            all_meta['output'][src]['distributed']):
                        pass  # special case, auto_ivc dist var with 0 local size
                    elif tmeta['has_src_indices']:
                        if tlocmeta:  # target is local
                            flat = False
                            if name in model._var_prom2inds:
                                sshape, inds, flat = model._var_prom2inds[name]
                                src_indices = inds
                            elif (tlocmeta.get('manual_connection') or
                                  model._inputs._contains_abs(name)):
                                src_indices = tlocmeta['src_indices']
                            else:
                                src_indices = None

                            if src_indices is None:
                                model._outputs.set_var(src, value, _full_slice, flat,
                                                       var_name=var_name)
                            else:
                                flat = src_indices._flat_src

                                if tmeta['distributed']:
                                    src_indices = src_indices.shaped_array()
                                    ssizes = model._var_sizes['output']
                                    sidx = model._var_allprocs_abs2idx[src]
                                    ssize = ssizes[myrank, sidx]
                                    start = np.sum(ssizes[:myrank, sidx])
                                    end = start + ssize
                                    if np.any(src_indices < start) or np.any(src_indices >= end):
                                        raise RuntimeError(f"{model.msginfo}: Can't set {name}: "
                                                           "src_indices refer "
                                                           "to out-of-process array entries.")
                                    if start > 0:
                                        src_indices = src_indices - start
                                    src_indices = indexer(src_indices)
                                if indices is _full_slice:
                                    model._outputs.set_var(src, value, src_indices, flat,
                                                           var_name=var_name)
                                else:
                                    model._outputs.set_var(src, value, src_indices.apply(indices),
                                                           True, var_name=var_name)
                        else:
                            issue_warning(f"{model.msginfo}: Cannot set the value of '{abs_name}':"
                                          " Setting the value of a remote connected input with"
                                          " src_indices is currently not supported, you must call"
                                          " `run_model()` to have the outputs populate their"
                                          " corresponding inputs.")
                    else:
                        value = np.asarray(value)
                        if indices is not _full_slice:
                            indices = indexer(indices)
                        model._outputs.set_var(src, value, indices, var_name=var_name)
                elif src in model._discrete_outputs:
                    model._discrete_outputs[src] = value
                # also set the input
                # TODO: maybe remove this if inputs are removed from case recording
                if n_proms < 2:
                    if model._inputs._contains_abs(abs_name):
                        model._inputs.set_var(abs_name, ivalue, indices)
                    elif abs_name in model._discrete_inputs:
                        model._discrete_inputs[abs_name] = value
                    else:
                        # must be a remote var. so, just do nothing on this proc. We can't get here
                        # unless abs_name is found in connections, so the variable must exist.
                        if abs_name in model._var_allprocs_abs2meta:
                            print(f"Variable '{name}' is remote on rank {self.comm.rank}.  "
                                  "Local assignment ignored.")
            elif abs_name in model._discrete_outputs:
                model._discrete_outputs[abs_name] = value
            elif model._inputs._contains_abs(abs_name):   # could happen if model is a component
                model._inputs.set_var(abs_name, value, indices)
            elif abs_name in model._discrete_inputs:   # could happen if model is a component
                model._discrete_inputs[abs_name] = value

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

        is_prom = len(abs_ins) > 1 or name != abs_ins[0]

        if scope_sys is None:
            scope_sys = self

        abs2meta_all_ins = self._var_allprocs_abs2meta['input']

        # if we have multiple promoted inputs that are explicitly connected to an output and units
        # have not been specified, look for group input to disambiguate
        if units is None and len(abs_ins) > 1:
            if abs_name not in self._var_allprocs_discrete['input']:
                # can't get here unless Group because len(abs_ins) always == 1 for comp
                try:
                    units = scope_sys._group_inputs[name][0]['units']
                except (KeyError, IndexError):
                    unit0 = abs2meta_all_ins[abs_ins[0]]['units']
                    for n in abs_ins[1:]:
                        if unit0 != abs2meta_all_ins[n]['units']:
                            self._show_ambiguity_msg(name, ('units',), abs_ins)
                            break

        is_local = abs_name in self._var_abs2meta['input']
        src_indices = vshape = None
        if is_local:  # input is local
            vmeta = self._var_abs2meta['input'][abs_name]
            if vmeta.get('manual_connection') or not is_prom:
                src_indices = vmeta['src_indices']
                vshape = vmeta['shape']
        else:
            vmeta = abs2meta_all_ins[abs_name]

        distrib = vmeta['distributed']
        vdynshape = vmeta['shape_by_conn']
        for n in abs_ins:
            if abs2meta_all_ins[n]['has_src_indices']:
                has_src_indices = True
                break
        else:
            has_src_indices = False

        if is_prom:
            # see if we have any 'intermediate' level src_indices when using a promoted name
            n = name
            scope = scope_sys
            while n:
                if n in scope._var_prom2inds:
                    _, inds, _ = scope._var_prom2inds[n]
                    if inds is None:
                        if is_prom:  # using a promoted lookup
                            src_indices = None
                            vshape = None
                            has_src_indices = False
                    else:
                        shp = inds.indexed_src_shape
                        src_indices = inds
                        has_src_indices = True
                        if is_prom:
                            vshape = shp
                    break

                parent, _, child = n.partition('.')
                if child:
                    s = scope._get_subsystem(parent)
                    if s is not None:
                        scope = s
                n = child

        if self.comm.size > 1 and get_remote:
            if self.comm.rank == self._owning_rank[abs_name]:
                self.comm.bcast(has_src_indices, root=self.comm.rank)
            else:
                has_src_indices = self.comm.bcast(None, root=self._owning_rank[abs_name])

        model_ref = self._problem_meta['model_ref']()
        smeta = model_ref._var_allprocs_abs2meta['output'][src]
        sdistrib = smeta['distributed']
        dynshape = vdynshape or smeta['shape_by_conn']
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
            if not is_local:
                val = np.zeros(0)
            elif src_indices is None:
                if vshape is not None:
                    val = val.reshape(vshape)
            else:
                var_idx = self._var_allprocs_abs2idx[src]
                sizes = self._var_sizes['output'][:, var_idx]
                if distrib and (sdistrib or dynshape or not slocal) and not get_remote:
                    # sizes for src var in each proc
                    start = np.sum(sizes[:self.comm.rank])
                    end = start + sizes[self.comm.rank]
                    src_indices = src_indices.shaped_array(copy=True)
                    if np.all(np.logical_and(src_indices >= start, src_indices < end)):
                        if src_indices.size > 0:
                            src_indices = src_indices - start
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
                    if src_indices._flat_src:
                        val = val.ravel()[src_indices.flat()]
                        # if at component level, just keep shape of the target and don't flatten
                        if not flat and not is_prom:
                            shp = vmeta['shape']
                            val.shape = shp
                    else:
                        val = val[src_indices()]
                        if vshape is not None and val.shape != vshape:
                            val.shape = vshape
                        elif not is_prom and vmeta is not None and val.shape != vmeta['shape']:
                            val.shape = vmeta['shape']

            if get_remote and self.comm.size > 1:
                if distrib:
                    if rank is None:
                        parts = self.comm.allgather(val)
                        parts = [p for p in parts if p.size > 0]
                        val = np.concatenate(parts, axis=0)
                    else:
                        parts = self.comm.gather(val, root=rank)
                        if rank == self.comm.rank:
                            parts = [p for p in parts if p.size > 0]
                            val = np.concatenate(parts, axis=0)
                        else:
                            val = None
                else:  # non-distrib input
                    if self.comm.rank == self._owning_rank[abs_name]:
                        self.comm.bcast(val, root=self.comm.rank)
                    else:
                        val = self.comm.bcast(None, root=self._owning_rank[abs_name])

            if distrib and get_remote:
                val.shape = abs2meta_all_ins[abs_name]['global_shape']
            elif not flat and val.size > 0 and vshape is not None:
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
                            vdict[n] = discrete_vec[n[offset:]]['val']
                        else:
                            ivc_path = conns[prom2abs_in[n][0]]
                            if vec._contains_abs(ivc_path):
                                vdict[ivc_path] = srcget(ivc_path, False)
                            elif ivc_path[offset:] in discrete_vec:
                                vdict[ivc_path] = discrete_vec[ivc_path[offset:]]['val']
                else:
                    for name in variables:
                        if name in self._responses and self._responses[name]['alias'] is not None:
                            name = self._responses[name]['source']
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
                        if name in self._responses and self._responses[name]['alias'] is not None:
                            name = self._responses[name]['source']
                        if vec._contains_abs(name):
                            vdict[name] = get(name, get_remote=True, rank=0,
                                              vec_name=vec_name, kind=kind)
                        elif name[offset:] in discrete_vec and self._owning_rank[name] == rank:
                            vdict[name] = discrete_vec[name[offset:]]['val']
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
                                vdict[name] = discrete_vec[name[offset:]]['val']
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
            model_ref = self._problem_meta['model_ref']()
        else:
            model_ref = None

        if model_ref is not None:
            meta_all = model_ref._var_allprocs_abs2meta
            meta_loc = model_ref._var_abs2meta
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
                # (like 'val' or 'src_indices'). If MPI is active, this val may be remote
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

    def _generate_md5_hash(self):
        """
        Generate an md5 hash for the data structure of this model.

        The hash is generated from an encoded string containing the physical model hiearchy,
        including all component and variable names, and all connection information.

        The hash is used by the n2 viewer to determine if a saved view can be reused. It is not
        intended to accurately track whether a model has been changed, so no options/settings are
        tracked.

        Returns
        -------
        str
            The md5 hash string for the model.
        """
        data = []

        # Model Hierarchy.
        for system in self.system_iter(include_self=True, recurse=True):

            # System name and depth.
            pathname = system.pathname
            if pathname:
                name_parts = pathname.split('.')
                data.append((name_parts[-1], len(name_parts)))

            else:
                data.append(('model', 0))

            # Local (relative) names for Component inputs and outputs.
            try:
                data.append(sorted(system._var_rel_names['input']))
                data.append(sorted(system._var_rel_names['output']))
            except AttributeError:
                continue

        # All Connections.
        # Note: dictionary can be in any order, so we have to sort.
        for key in sorted(self._conn_global_abs_in2out):
            data.append(self._conn_global_abs_in2out[key])

        try:
            hash = hashlib.md5(str(data).encode(),
                               usedforsecurity=False).hexdigest()  # nosec: content not sensitive
        except TypeError:
            hash = hashlib.md5(str(data).encode()).hexdigest()  # nosec: content not sensitive

        return hash

    def _get_full_dist_shape(self, abs_name, local_shape):
        """
        Get the full 'distributed' shape for a variable.

        Variable name is absolute and variable is assumed to be continuous.

        Parameters
        ----------
        abs_name : str
            Absolute name of the variable.

        local_shape : tuple
            Local shape of the variable, used in error reporting.

        Returns
        -------
        tuple
            The distributed shape for the given variable.
        """
        if abs_name in self._var_allprocs_abs2meta['output']:
            io = 'output'
            scope = self
        elif abs_name in self._problem_meta['model_ref']()._var_allprocs_abs2meta['output']:
            io = 'output'
            scope = self._problem_meta['model_ref']()
        else:
            io = 'input'
            scope = self

        meta = scope._var_allprocs_abs2meta[io][abs_name]
        var_idx = scope._var_allprocs_abs2idx[abs_name]
        global_size = np.sum(scope._var_sizes[io][:, var_idx])

        # assume that all but the first dimension of the shape of a
        # distributed variable is the same on all procs
        shape = meta['shape']
        if shape is None and self._get_saved_errors():
            # a setup error has occurred earlier that caused shape to be None.  Just return (0,)
            # to avoid a confusing KeyError
            return (0,)
        high_dims = shape[1:]
        sz = shape_to_len(shape)
        with multi_proc_exception_check(self.comm):
            if high_dims:
                high_size = shape_to_len(high_dims)

                dim_size_match = bool(global_size % high_size == 0)
                if dim_size_match is False and sz > 0:
                    raise RuntimeError(f"{self.msginfo}: All but the first dimension of the "
                                       "shape's local parts in a distributed variable must match "
                                       f"across processes. For output '{abs_name}', local shape "
                                       f"{local_shape} in MPI rank {self.comm.rank} has a "
                                       "higher dimension that differs in another rank.")

                dim1 = global_size // high_size
                return tuple([dim1] + list(high_dims))

        return (global_size,)

    def _has_fast_rel_lookup(self):
        """
        Return True if this System should have fast relative variable name lookup in vectors.

        Returns
        -------
        bool
            True if this System should have fast relative variable name lookup in vectors.
        """
        return False

    def _collect_error(self, msg, exc_type=None, tback=None, ident=None):
        """
        Save an error message to raise as an exception later.

        Parameters
        ----------
        msg : str
            The connection error message to be saved.
        exc_type : class or None
            The type of exception to be raised if this error is the only one collected.
        tback : traceback or None
            The traceback of a caught exception.
        ident : int
            Identifier of the object responsible for issuing the error.
        """
        if exc_type is None:
            exc_type = RuntimeError

        if tback is None:
            tback = make_traceback()

        if self.msginfo not in msg:
            msg = f"{self.msginfo}: {msg}"

        saved_errors = self._get_saved_errors()

        # if saved_errors is None it means we have already finished setup and all errors should
        # be raised as exceptions immediately.
        if saved_errors is None or env_truthy('OPENMDAO_FAIL_FAST'):
            raise exc_type(msg).with_traceback(tback)

        saved_errors.append((ident, msg, exc_type, tback))

    def _get_saved_errors(self):
        if self._problem_meta is None:
            return self._saved_errors
        return self._problem_meta['saved_errors']

    def _set_problem_meta(self, prob_meta):
        self._problem_meta = prob_meta
        # transfer any temporarily stored error msgs to the Problem
        if self._saved_errors and prob_meta['saved_errors'] is not None:
            prob_meta['saved_errors'].extend(self._saved_errors)
        self._saved_errors = None if env_truthy('OPENMDAO_FAIL_FAST') else []

    def _get_inconsistent_keys(self):
        keys = set()
        if self.comm.size > 1:
            from openmdao.core.component import Component
            if isinstance(self, Component):
                keys.update(self._inconsistent_keys)
            else:
                for comp in self.system_iter(recurse=True, include_self=True, typ=Component):
                    keys.update(comp._inconsistent_keys)
            myrank = self.comm.rank

            for rank, proc_keys in enumerate(self.comm.allgather(keys)):
                if rank != myrank:
                    keys.update(proc_keys)
        return keys

    def is_explicit(self):
        """
        Return True if this is an explicit component.

        Returns
        -------
        bool
            True if this is an explicit component.
        """
        return False

    def _get_sys_promotion_tree(self, tree=None):
        """
        Return a dict of all subsystems and their promoted inputs/outputs.

        Parameters
        ----------
        tree : dict
            A dict of subsystems and their promoted inputs/outputs.

        Returns
        -------
        dict
            A dict of subsystems and their promoted inputs/outputs.
        """
        if tree is None:
            tree = {}

        tree[self.pathname] = {'proms_out': defaultdict(set), 'proms_in': defaultdict(set)}

        parent = self.pathname.rpartition('.')[0] if self.pathname else None
        if parent in tree:
            parent_node = tree[parent]
            out_promotions = parent_node['proms_out']
            in_promotions = parent_node['proms_in']
            maps = self._get_promotion_maps()
            for prom_out, tup in maps['output'].items():
                out_promotions[tup[0]].add(self.name + '.' + prom_out)
            for prom_in, tup in maps['input'].items():
                in_promotions[tup[0]].add(self.name + '.' + prom_in)

        for subsys in self._subsystems_myproc:
            subsys._get_sys_promotion_tree(tree=tree)

        return tree

    def _get_promote_lists(self, tree, abs_vnames, io):
        """
        Return a list of (sysname, varpath, prom_name) tuples.

        Parameters
        ----------
        tree : dict
            A dict of subsystems and their promoted inputs/outputs.
        abs_vnames : list of str
            List of absolute variable names.
        io : str
            'in' or 'out'

        Returns
        -------
        list of (str, str, str)
            List of (sysname, varpath, prom_name) tuples.
        """
        promotion_list = []
        if abs_vnames:
            node_proms = f'proms_{io}'
            for abs_vname in abs_vnames:
                # starting vname is just the local
                vname = abs_vname.rpartition('.')[2]
                # systems from the bottom up
                slist = list(all_ancestors(abs_vname))[1:]
                if not self.pathname:
                    slist += ['']
                for spath in slist:
                    sname = spath.rpartition('.')[2]
                    node = tree[spath]
                    proms = node[node_proms]
                    for pname, subs in proms.items():
                        if vname in subs:
                            promotion_list.append((spath, vname, pname))
                            vname = sname + '.' + pname
                            break
                    else:
                        vname = sname + '.' + vname

        return promotion_list

    def get_promotions(self, inprom=None, outprom=None):
        """
        Return all promotions for the given promoted variable(s).

        In other words, how and where did promotions occur to convert absolute variable names into
        the given promoted name(s) at the current System level.

        Parameters
        ----------
        inprom : str or None
            The promoted input variable name.
        outprom : str or None
            The promoted output variable name.

        Returns
        -------
        dict
            Dictionary keyed on system pathname containing input and/or output promotion lists for
            each System where promotions occurred to produce the given promoted variable(s).
        """
        if inprom is None and outprom is None:
            raise RuntimeError(f"{self.msginfo}: At least one of (inprom, outprom) must be set "
                               "when calling get_promotions.")

        if self._promotion_tree is None:
            self._promotion_tree = self._get_sys_promotion_tree()
        tree = self._promotion_tree

        plist_ins = plist_outs = None
        if outprom is None and inprom in self._var_allprocs_prom2abs_list['output']:
            outprom = inprom

        if outprom is not None:
            try:
                abs_outs = self._var_allprocs_prom2abs_list['output'][outprom]
            except KeyError:
                raise KeyError(f"{self.msginfo}: Promoted output variable '{outprom}' was not "
                               "found.")

            plist_outs = self._get_promote_lists(tree, abs_outs, 'out')

        if inprom is not None:
            try:
                abs_ins = self._var_allprocs_prom2abs_list['input'][inprom]
            except KeyError:
                raise KeyError(f"{self.msginfo}: Promoted input variable '{inprom}' was not "
                               "found.")

            plist_ins = self._get_promote_lists(tree, abs_ins, 'in')

        # create a dict that interleaves all of the input and output promotions, keeping the
        # number of rows consistent when adding a promotion tree info to a table.
        sys_prom_map = defaultdict(lambda: [None, set(), None, set()])
        if plist_outs:
            for spath, sub, theprom in plist_outs:
                sys_prom_map[spath][2] = theprom
                sys_prom_map[spath][3].add(sub)

        if plist_ins:
            for spath, sub, theprom in plist_ins:
                sys_prom_map[spath][0] = theprom
                sys_prom_map[spath][1].add(sub)

        # return regular dict sorted by system pathname
        return {spath: data for spath, data in sorted(sys_prom_map.items(), key=lambda x: x[0])}

    def _sorted_sys_iter(self):
        yield from ()

    def load_case(self, case):
        """
        Pull all input and output variables from a Case into this System.

        Override this method if the System requires special handling when loading a case.

        Parameters
        ----------
        case : Case or dict
            A Case from a CaseReader, or a dictionary with key 'inputs' mapped to the
            output of problem.model.list_inputs and key 'outputs' mapped to the output
            of prob.model.list_outputs. Both list_inputs and list_outputs should be called
            with `prom_name=True` and `return_format='dict'`.
        """
        pass

    def comm_info_iter(self):
        """
        Yield comm size for this system and all subsystems.

        Yields
        ------
        tuple
            A tuple of the form (abs_name, comm_size).
        """
        if MPI:
            yield (self.pathname, self.comm.size, self.comm.rank, MPI.COMM_WORLD.rank)

            for s in self._subsystems_myproc:
                yield from s.comm_info_iter()

    def dist_size_iter(self, io, top_comm):
        """
        Yield names and distributed ranges of all local and remote variables in this system.

        Parameters
        ----------
        io : str
            Either 'input' or 'output'.
        top_comm : MPI.Comm or None
            The top-level MPI communicator.

        Yields
        ------
        tuple
            A tuple of the form ((abs_name, rank), start, end).
        """
        sizes = self._var_sizes
        vmeta = self._var_allprocs_abs2meta

        topranks = np.arange(top_comm.size)

        myrank = self.comm.rank
        toprank = top_comm.rank

        mytopranks = topranks[toprank - myrank: toprank - myrank + self.comm.size]

        for rank in range(self.comm.size):
            for ivar, vname in enumerate(vmeta[io]):
                sz = sizes[io][rank, ivar]
                if sz > 0:
                    yield (vname, mytopranks[rank]), sz

    def local_range_iter(self, io):
        """
        Yield names and local ranges of all local variables in this system.

        Parameters
        ----------
        io : str
            Either 'input' or 'output'.

        Yields
        ------
        tuple
            A tuple of the form (abs_name, start, end).
        """
        vmeta = self._var_allprocs_abs2meta

        offset = 0
        for vname, size in zip(vmeta[io], self._var_sizes[io][self.comm.rank]):
            if size > 0:
                yield vname, offset, offset + size
            offset += size

    def get_var_dup_info(self, name, io):
        """
        Return information about how the given variable is duplicated across MPI processes.

        Parameters
        ----------
        name : str
            Name of the variable.
        io : str
            Either 'input' or 'output'.

        Returns
        -------
        tuple
            A tuple of the form (is_duplicated, num_zeros, is_distributed).
        """
        nz = np.count_nonzero(self._var_sizes[io][:, self._var_allprocs_abs2idx[name]])

        if self._var_allprocs_abs2meta[io][name]['distributed']:
            return False, self._var_sizes[io].shape[0] - nz, True  # distributed vars are never dups

        return nz > 1, self._var_sizes[io].shape[0] - nz, False

    def get_var_sizes(self, name, io):
        """
        Return the sizes of the given variable on all procs.

        Parameters
        ----------
        name : str
            Name of the variable.
        io : str
            Either 'input' or 'output'.

        Returns
        -------
        ndarray
            Array of sizes of the variable on all procs.
        """
        return self._var_sizes[io][:, self._var_allprocs_abs2idx[name]]
