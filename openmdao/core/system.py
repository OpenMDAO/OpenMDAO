"""Define the base System class."""
from __future__ import division

import sys
import os
from contextlib import contextmanager
from collections import OrderedDict, defaultdict
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from fnmatch import fnmatchcase
import sys
import os
import time
from numbers import Integral
import itertools

from six import iteritems, itervalues, string_types

import numpy as np
import networkx as nx

import openmdao
from openmdao.jacobians.assembled_jacobian import DenseJacobian, CSCJacobian
from openmdao.jacobians.dictionary_jacobian import DictionaryJacobian
from openmdao.recorders.recording_manager import RecordingManager
from openmdao.vectors.vector import INT_DTYPE
from openmdao.utils.mpi import MPI
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.record_util import create_local_meta, check_path
from openmdao.utils.variable_table import write_var_table
from openmdao.utils.array_utils import evenly_distrib_idxs, sizes2offsets
from openmdao.utils.general_utils import make_set
from openmdao.utils.graph_utils import all_connected_nodes
from openmdao.utils.name_maps import rel_name2abs_name
from openmdao.utils.coloring import _compute_coloring, Coloring, \
    _STD_COLORING_FNAME, _DYN_COLORING, _DEF_COMP_SPARSITY_ARGS
import openmdao.utils.coloring as coloring_mod
from openmdao.utils.general_utils import determine_adder_scaler, find_matches, \
    format_as_float_or_array, warn_deprecation, ContainsAll, all_ancestors, \
    simple_warning
from openmdao.approximation_schemes.complex_step import ComplexStep
from openmdao.approximation_schemes.finite_difference import FiniteDifference

# Use this as a special value to be able to tell if the caller set a value for the optional
#   out_stream argument. We run into problems running testflo if we use a default of sys.stdout.
_DEFAULT_OUT_STREAM = object()
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
    'wrt_patterns': ('*',),
    'method': 'fd',
    'wrt_matches': None,
    'per_instance': False,
    'coloring': None,
}

_DEFAULT_COLORING_META.update(_DEF_COMP_SPARSITY_ARGS)

_full_slice = slice(None)


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
    _problem_options : OptionsDictionary
        Problem level options.
    under_complex_step : bool
        When True, this system is undergoing complex step.
    force_alloc_complex : bool
        When True, the vectors have been allocated for checking with complex step.
    iter_count : int
        Int that holds the number of times this system has iterated
        in a recording run.
    cite : str
        Listing of relevant citations that should be referenced when
        publishing work that uses this class.
    _full_comm : MPI.Comm or None
        MPI communicator object used when System's comm is split for parallel FD.
    _subsystems_allprocs : [<System>, ...]
        List of all subsystems (children of this system).
    _subsystems_myproc : [<System>, ...]
        List of local subsystems that exist on this proc.
    _subsystems_myproc_inds : [int, ...]
        List of indices of subsystems on this proc among all of this system's subsystems
        (i.e. among _subsystems_allprocs).
    _subsystems_proc_range : (int, int)
        List of ranges of each myproc subsystem's processors relative to those of this system.
    _var_promotes : { 'any': [], 'input': [], 'output': [] }
        Dictionary of lists of variable names/wildcards specifying promotion
        (used to calculate promoted names)
    _var_allprocs_abs_names : {'input': [str, ...], 'output': [str, ...]}
        List of absolute names of this system's variables on all procs.
    _var_abs_names : {'input': [str, ...], 'output': [str, ...]}
        List of absolute names of this system's variables existing on current proc.
    _var_allprocs_abs_names_discrete : {'input': [str, ...], 'output': [str, ...]}
        List of absolute names of this system's discrete variables on all procs.
    _var_abs_names_discrete : {'input': [str, ...], 'output': [str, ...]}
        List of absolute names of this system's discrete variables existing on current proc.
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
    _owned_var_sizes : ndarray
        Array of local sizes for 'owned' or distributed vars only.
    _nodup_out_ranges : dict
        Range of each output/resid in the global non-duplicated array.
    _nodup2local_out_inds : ndarray
        Indices that map values from the global non-duplicated array into the local output/resids.
    _var_offsets : {<vecname>: {'input': dict of ndarray, 'output': dict of ndarray}, ...} or None
        Dict of distributed offsets, keyed by var name.  Offsets are stored in an array
        of size nproc x num_var where nproc is the number of processors
        in this System's communicator and num_var is the number of allprocs variables
        in the given system.  This is only defined in a Group that owns one or more interprocess
        connections or a top level Group or System that is used to compute total derivatives
        across multiple processes.
    _conn_global_abs_in2out : {'abs_in': 'abs_out'}
        Dictionary containing all explicit & implicit connections owned by this system
        or any descendant system. The data is the same across all processors.
    _ext_num_vars : {'input': (int, int), 'output': (int, int)}
        Total number of allprocs variables in system before/after this one.
    _ext_sizes : {'input': (int, int), 'output': (int, int)}
        Total size of allprocs variables in system before/after this one.
    _vec_names : [str, ...]
        List of names of all vectors, including the nonlinear vector.
    _lin_vec_names : [str, ...]
        List of names of the linear vectors (i.e., the right-hand sides).
    _vectors : {'input': dict, 'output': dict, 'residual': dict}
        Dictionaries of vectors keyed by vec_name.
    _inputs : <Vector>
        The inputs vector; points to _vectors['input']['nonlinear'].
    _outputs : <Vector>
        The outputs vector; points to _vectors['output']['nonlinear'].
    _residuals : <Vector>
        The residuals vector; points to _vectors['residual']['nonlinear'].
    _lower_bounds : <Vector>
        Vector of lower bounds, scaled and dimensionless.
    _upper_bounds : <Vector>
        Vector of upper bounds, scaled and dimensionless.
    _nonlinear_solver : <NonlinearSolver>
        Nonlinear solver to be used for solve_nonlinear.
    _linear_solver : <LinearSolver>
        Linear solver to be used for solve_linear; not the Newton system.
    _solver_info : SolverInfo
        A stack-like object shared by all Solvers in the model.
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
    _static_mode : bool
        If true, we are outside of setup.
        In this case, add_input, add_output, and add_subsystem all add to the
        '_static' versions of the respective data structures.
        These data structures are never reset during reconfiguration.
    _static_subsystems_allprocs : [<System>, ...]
        List of subsystems that stores all subsystems added outside of setup.
    _static_design_vars : dict of dict
        Driver design variables added outside of setup.
    _static_responses : dict of dict
        Driver responses added outside of setup.
    _reconfigured : bool
        If True, this system has reconfigured, and the immediate parent should update.
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
        _distributed_vector_class or _local_vector_class.
    _distributed_vector_class : class
        Class to use for distributed data vectors.
    _local_vector_class : class
        Class to use for local data vectors.
    _assembled_jac : AssembledJacobian or None
        If not None, this is the AssembledJacobian owned by this system's linear_solver.
    _num_par_fd : int
        If FD is active, and the value is > 1, turns on parallel FD and specifies the number of
        concurrent FD solves.
    _par_fd_id : int
        ID used to determine which columns in the jacobian will be computed when using parallel FD.
    _use_derivatives : bool
        If True, perform any memory allocations necessary for derivative computation.
    _has_approx : bool
        If True, this system or its descendent has declared approximated partial or semi-total
        derivatives.
    _coloring_info : tuple
        Metadata that defines how to perform coloring of this System's approx jacobian. Not
        used if this System does no partial or semi-total coloring.
    _first_call_to_linearize : bool
        If True, this is the first call to _linearize.
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
                                       desc='Record metadata for this system', default=True)
        self.recording_options.declare('record_model_metadata', types=bool,
                                       desc='Record metadata for all sub systems in the model',
                                       default=True)
        self.recording_options.declare('includes', types=list, default=['*'],
                                       desc='Patterns for variables to include in recording')
        self.recording_options.declare('excludes', types=list, default=[],
                                       desc='Patterns for vars to exclude in recording '
                                       '(processed post-includes)')
        self.recording_options.declare('options_excludes', types=list, default=[],
                                       desc='User-defined metadata to exclude in recording')

        self._problem_options = None

        # Case recording related
        self.iter_count = 0

        self.cite = ""

        self._subsystems_allprocs = []
        self._subsystems_myproc = []
        self._subsystems_myproc_inds = []
        self._subsystems_proc_range = []

        self._var_promotes = {'input': [], 'output': [], 'any': []}
        self._var_allprocs_abs_names = {'input': [], 'output': []}
        self._var_abs_names = {'input': [], 'output': []}
        self._var_allprocs_abs_names_discrete = {'input': [], 'output': []}
        self._var_abs_names_discrete = {'input': [], 'output': []}
        self._var_allprocs_prom2abs_list = None
        self._var_abs2prom = {'input': {}, 'output': {}}
        self._var_allprocs_abs2prom = {'input': {}, 'output': {}}
        self._var_allprocs_abs2meta = {}
        self._var_abs2meta = {}
        self._var_discrete = {'input': {}, 'output': {}}
        self._var_allprocs_discrete = {'input': {}, 'output': {}}

        self._var_allprocs_abs2idx = {}

        self._var_sizes = None
        self._owned_var_sizes = None
        self._var_offsets = None
        self._nodup_out_ranges = None
        self._nodup2local_out_inds = None

        self._full_comm = None

        self._ext_num_vars = {'input': (0, 0), 'output': (0, 0)}
        self._ext_sizes = {'input': (0, 0), 'output': (0, 0)}

        self._vectors = {'input': {}, 'output': {}, 'residual': {}}

        self._inputs = None
        self._outputs = None
        self._residuals = None
        self._discrete_inputs = None
        self._discrete_outputs = None

        self._lower_bounds = None
        self._upper_bounds = None

        self._nonlinear_solver = None
        self._linear_solver = None

        self._jacobian = None
        self._approx_schemes = OrderedDict()
        self._subjacs_info = {}
        self.matrix_free = False

        self._owns_approx_jac = False
        self._owns_approx_jac_meta = {}
        self._owns_approx_wrt = None
        self._owns_approx_of = None
        self._owns_approx_wrt_idx = {}
        self._owns_approx_of_idx = {}

        self.under_complex_step = False
        self.force_alloc_complex = False

        self._design_vars = OrderedDict()
        self._responses = OrderedDict()
        self._rec_mgr = RecordingManager()

        self._conn_global_abs_in2out = {}

        self._static_mode = True
        self._static_subsystems_allprocs = []
        self._static_design_vars = OrderedDict()
        self._static_responses = OrderedDict()

        self._reconfigured = False
        self.supports_multivecs = False

        self._relevant = None
        self._vec_names = None
        self._vois = None
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
        self._local_vector_class = None
        self._distributed_vector_class = None
        self._use_derivatives = True
        self._has_approx = False

        self._assembled_jac = None

        self._par_fd_id = 0

        self._filtered_vars_to_record = {}
        self._owning_rank = None
        self._lin_vec_names = []
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
        if self.pathname == '':
            return '{} (<model>)'.format(type(self).__name__)
        if self.pathname is not None:
            return '{} ({})'.format(type(self).__name__, self.pathname)
        if self.name:
            return '{} ({})'.format(type(self).__name__, self.name)
        return type(self).__name__

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.

        This is optionally implemented by subclasses of Component or Group
        that themselves are intended to be subclassed by the end user. The
        options of the intermediate class are declared here leaving the
        `initialize` method available for user-defined options.
        """
        pass

    def _check_self_reconf(self):
        """
        Check if this systems wants to reconfigure and if so, perform the reconfiguration.
        """
        if self.reconfigure():
            with self._unscaled_context_all():
                # Backup input values
                old_in = self._inputs
                old_out = self._outputs

                # Perform reconfiguration
                self.resetup('reconf')

                new_in = self._inputs
                new_out = self._outputs

                # Reload input and output values where possible
                for vold, vnew in [(old_in, new_in), (old_out, new_out)]:
                    for abs_name, old_view in iteritems(vold._views_flat):
                        if abs_name in vnew._views_flat:
                            new_view = vnew._views_flat[abs_name]

                            if len(old_view) == len(new_view):
                                new_view[:] = old_view

            self._reconfigured = True

    def _check_child_reconf(self, subsys=None):
        """
        Check if any subsystem has reconfigured and if so, perform the necessary update setup.

        Parameters
        ----------
        subsys : System or None
            ignored
        """
        self._reconfigured = False

    def reconfigure(self):
        """
        Perform reconfiguration.

        Returns
        -------
        bool
            If True, reconfiguration is to be performed.
        """
        return False

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

    def _get_initial_global(self, initial):
        """
        Get initial values for _ext_num_vars, _ext_sizes.

        Parameters
        ----------
        initial : bool
            Whether we are reconfiguring - i.e., the model has been previously setup.

        Returns
        -------
        _ext_num_vars : {'input': (int, int), 'output': (int, int)}
            Total number of allprocs variables in system before/after this one.
        _ext_sizes : {'input': (int, int), 'output': (int, int)}
            Total size of allprocs variables in system before/after this one.
        """
        if not initial:
            return (self._ext_num_vars, self._ext_sizes)
        else:
            ext_num_vars = {}
            ext_sizes = {}

            vec_names = self._lin_rel_vec_name_list if self._use_derivatives else self._vec_names

            for vec_name in vec_names:
                ext_num_vars[vec_name] = {}
                ext_sizes[vec_name] = {}
                for type_ in ['input', 'output']:
                    ext_num_vars[vec_name][type_] = (0, 0)
                    ext_sizes[vec_name][type_] = (0, 0)

            if self._use_derivatives:
                ext_num_vars['nonlinear'] = ext_num_vars['linear']
                ext_sizes['nonlinear'] = ext_sizes['linear']

            return ext_num_vars, ext_sizes

    def _get_root_vectors(self, initial, force_alloc_complex=False):
        """
        Get the root vectors for the nonlinear and linear vectors for the model.

        Parameters
        ----------
        initial : bool
            Whether we are reconfiguring - i.e., whether the model has been previously setup.
        force_alloc_complex : bool
            Force allocation of imaginary part in nonlinear vectors. OpenMDAO can generally
            detect when you need to do this, but in some cases (e.g., complex step is used
            after a reconfiguration) you may need to set this to True.

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

        if initial:
            relevant = self._relevant
            vec_names = self._rel_vec_name_list if self._use_derivatives else self._vec_names
            vois = self._vois
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

            vector_class = self._vector_class

            for vec_name in vec_names:
                sizes = self._var_sizes[vec_name]['output']
                ncol = 1
                rel = None
                if vec_name == 'nonlinear':
                    alloc_complex = nl_alloc_complex
                else:
                    alloc_complex = ln_alloc_complex

                    if vec_name != 'linear':
                        voi = vois[vec_name]
                        if voi['vectorize_derivs']:
                            if 'size' in voi:
                                ncol = voi['size']
                            else:
                                owner = self._owning_rank[vec_name]
                                ncol = sizes[owner, abs2idx[vec_name][vec_name]]
                        rdct, _ = relevant[vec_name]['@all']
                        rel = rdct['output']

                for key in ['input', 'output', 'residual']:
                    root_vectors[key][vec_name] = vector_class(vec_name, key, self,
                                                               alloc_complex=alloc_complex,
                                                               ncol=ncol, relevant=rel)
        else:

            for key, vardict in iteritems(self._vectors):
                for vec_name, vec in iteritems(vardict):
                    root_vectors[key][vec_name] = vec._root_vector

        lower, upper = self._get_bounds_root_vectors(self._local_vector_class, initial)
        root_vectors['lower'] = lower
        root_vectors['upper'] = upper

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

    def _get_bounds_root_vectors(self, vector_class, initial):
        """
        Get the root vectors for the lower and upper bounds vectors.

        Parameters
        ----------
        vector_class : Vector
            The Vector class used to instantiate the root vectors.
        initial : bool
            Whether we are reconfiguring - i.e., whether the model has been previously setup.

        Returns
        -------
        Vector
            Root vector for the lower bounds vector.
        Vector
            Root vector for the upper bounds vector.
        """
        if not initial:
            return self._lower_bounds._root_vector, self._upper_bounds._root_vector
        else:
            lower = vector_class('nonlinear', 'output', self)
            upper = vector_class('nonlinear', 'output', self)
            lower._data[:] = -np.inf
            upper._data[:] = np.inf
            return lower, upper

    def resetup(self, setup_mode='full'):
        """
        Public wrapper for _setup that reconfigures after an initial setup has been performed.

        Parameters
        ----------
        setup_mode : str
            Must be one of 'full', 'reconf', or 'update'.
        """
        self._setup(self.comm, setup_mode=setup_mode, mode=self._mode,
                    distributed_vector_class=self._distributed_vector_class,
                    local_vector_class=self._local_vector_class,
                    use_derivatives=self._use_derivatives,
                    prob_options=self._problem_options)
        self._final_setup(self.comm, setup_mode=setup_mode,
                          force_alloc_complex=self._outputs._alloc_complex)

    def _setup(self, comm, setup_mode, mode, distributed_vector_class, local_vector_class,
               use_derivatives, prob_options=None):
        """
        Perform setup for this system and its descendant systems.

        There are three modes of setup:
        1. 'full': wipe everything and setup this and all descendant systems from scratch
        2. 'reconf': don't wipe everything, but reconfigure this and all descendant systems
        3. 'update': update after one or more immediate systems has done a 'reconf' or 'update'

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm> or None
            The global communicator.
        setup_mode : str
            Must be one of 'full', 'reconf', or 'update'.
        mode : str
            Derivative direction, either 'fwd', or 'rev', or 'auto'
        distributed_vector_class : type
            Reference to the <Vector> class or factory function used to instantiate vectors
            and associated transfers involved in interprocess communication.
        local_vector_class : type
            Reference to the <Vector> class or factory function used to instantiate vectors
            and associated transfers involved in intraprocess communication.
        use_derivatives : bool
            If True, perform any memory allocations necessary for derivative computation.
        prob_options : OptionsDictionary
            Problem level options dictionary.
        """
        # save a ref to the problem level options.
        self._problem_options = prob_options

        # 1. Full setup that must be called in the root system.
        if setup_mode == 'full':
            recurse = True

            self.pathname = ''
            self.comm = comm
            self._relevant = None
            self._distributed_vector_class = distributed_vector_class
            self._local_vector_class = local_vector_class
            self._use_derivatives = use_derivatives
        # 2. Partial setup called in the system initiating the reconfiguration.
        elif setup_mode == 'reconf':
            recurse = True
        # 3. Update-mode setup called in all ancestors of the system initiating the reconf.
        elif setup_mode == 'update':
            recurse = False

        self._mode = mode

        # If we're only updating and not recursing, processors don't need to be redistributed.
        if recurse:
            # Besides setting up the processors, this method also builds the model hierarchy.
            self._setup_procs(self.pathname, comm, mode, self._problem_options)

        # Recurse model from the bottom to the top for configuring.
        self._configure()

        # For updating variable and connection data, setup needs to be performed only
        # in the current system, by gathering data from immediate subsystems,
        # and no recursion is necessary.
        self._setup_var_data(recurse=recurse)
        self._setup_vec_names(mode, self._vec_names, self._vois)
        self._setup_global_connections(recurse=recurse)
        self._setup_relevance(mode, self._relevant)
        self._setup_var_index_ranges(recurse=recurse)
        self._setup_var_sizes(recurse=recurse)
        self._setup_connections(recurse=recurse)

    def _final_setup(self, comm, setup_mode, force_alloc_complex=False):
        """
        Perform final setup for this system and its descendant systems.

        This part of setup is called automatically at the start of run_model or run_driver.

        There are three modes of setup:
        1. 'full': wipe everything and setup this and all descendant systems from scratch
        2. 'reconf': don't wipe everything, but reconfigure this and all descendant systems
        3. 'update': update after one or more immediate systems has done a 'reconf' or 'update'

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm> or None
            The global communicator.
        setup_mode : str
            Must be one of 'full', 'reconf', or 'update'.
        force_alloc_complex : bool
            Force allocation of imaginary part in nonlinear vectors. OpenMDAO can generally
            detect when you need to do this, but in some cases (e.g., complex step is used
            after a reconfiguration) you may need to set this to True.
        """
        # 1. Full setup that must be called in the root system.
        if setup_mode == 'full':
            initial = True
            recurse = True
            resize = False
        # 2. Partial setup called in the system initiating the reconfiguration.
        elif setup_mode == 'reconf':
            initial = False
            recurse = True
            resize = True
        # 3. Update-mode setup called in all ancestors of the system initiating the reconf.
        elif setup_mode == 'update':
            initial = False
            recurse = False
            resize = False

        # For vector-related, setup, recursion is always necessary, even for updating.
        # For reconfiguration setup, we resize the vectors once, only in the current system.
        ext_num_vars, ext_sizes = self._get_initial_global(initial)
        self._setup_global(ext_num_vars, ext_sizes)
        root_vectors = self._get_root_vectors(initial, force_alloc_complex=force_alloc_complex)
        self._setup_vectors(root_vectors, resize=resize)

        # Transfers do not require recursion, but they have to be set up after the vector setup.
        self._setup_transfers(recurse=recurse)

        # Same situation with solvers, partials, and Jacobians.
        # If we're updating, we just need to re-run setup on these, but no recursion necessary.
        self._setup_solvers(recurse=recurse)
        if self._use_derivatives:
            self._setup_partials(recurse=recurse)
            self._setup_jacobians(recurse=recurse)

        self._setup_recording(recurse=recurse)

        # If full or reconf setup, reset this system's variables to initial values.
        if setup_mode in ('full', 'reconf'):
            self.set_initial_values()

        rec_model_meta = self.recording_options['record_model_metadata']

        # Tell all subsystems to record their metadata if they have recorders attached
        for sub in self.system_iter(recurse=True, include_self=True):
            if sub.recording_options['record_metadata']:
                sub._rec_mgr.record_metadata(sub)

            # Also, optionally, record to the recorders attached to this System,
            #   the system metadata for all the subsystems
            if rec_model_meta:
                self._rec_mgr.record_metadata(sub)

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
        if coloring not in (_STD_COLORING_FNAME, _DYN_COLORING):
            if recurse:
                simple_warning("%s: recurse was passed to use_fixed_coloring but a specific "
                               "coloring was set, so recurse was ignored." % self.pathname)
            self._coloring_info['coloring'] = coloring
            if isinstance(coloring, Coloring):
                approx = self._get_approx_scheme(coloring._meta['method'])
                # force regen of approx groups on next call to compute_approximations
                approx._colored_approx_groups = None
                approx._approx_groups = None
            return

        if coloring_mod._force_dyn_coloring and coloring is _STD_COLORING_FNAME:
            # force the generation of a dynamic coloring this time
            coloring = _DYN_COLORING

        if self._coloring_info['coloring'] is None:
            simple_warning('%s: use_fixed_coloring() ignored because no coloring was declared.' %
                           self.pathname)
        else:
            self._coloring_info['coloring'] = coloring

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

        if self._coloring_info['coloring'] is None:
            # calling declare_coloring turns on dynamic coloring.  Calling use_fixed_coloring
            # will switch it to use a static coloring.
            options['coloring'] = _DYN_COLORING
        else:
            # this will handle cases where use_fixed_coloring was called before declare_coloring
            options['coloring'] = self._coloring_info['coloring']

        options['wrt_patterns'] = [wrt] if isinstance(wrt, string_types) else wrt
        options['method'] = method
        options['per_instance'] = per_instance
        options['repeat'] = num_full_jacs
        options['tol'] = tol
        options['orders'] = orders
        options['perturb_size'] = perturb_size
        options['show_summary'] = show_summary
        options['show_sparsity'] = show_sparsity
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
                        coloring._meta['pathname'] = s.pathname
                        coloring._meta['class'] = type(s).__name__
            return colorings

        # don't override metadata if it's already declared
        info = self._coloring_info
        info.update(**overrides)
        if isinstance(info['wrt_patterns'], string_types):
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
                self._setup_partials(recurse=True)

        approx_scheme = self._get_approx_scheme(self._coloring_info['method'])

        from openmdao.core.group import Group
        is_total = isinstance(self, Group)

        # compute perturbations
        starting_inputs = self._inputs._data.copy()
        in_offsets = starting_inputs.copy()
        in_offsets[in_offsets == 0.0] = 1.0
        in_offsets *= info['perturb_size']

        starting_outputs = self._outputs._data.copy()
        out_offsets = starting_outputs.copy()
        out_offsets[out_offsets == 0.0] = 1.0
        out_offsets *= info['perturb_size']

        starting_resids = self._residuals._data.copy()

        if self._coloring_info['coloring'] is None:
            self._coloring_info['coloring'] = coloring_mod._DYN_COLORING

        # for groups, this does some setup of approximations
        self._setup_approx_coloring()

        save_first_call = self._first_call_to_linearize
        self._first_call_to_linearize = False
        sparsity_start_time = time.time()

        for i in range(info['num_full_jacs']):
            # randomize inputs (and outputs if implicit)
            if i > 0:
                self._inputs._data[:] = \
                    starting_inputs + in_offsets * np.random.random(in_offsets.size)
                self._outputs._data[:] = \
                    starting_outputs + out_offsets * np.random.random(out_offsets.size)

                if is_total:
                    self._solve_nonlinear()
                else:
                    self._apply_nonlinear()

                for scheme in self._approx_schemes.values():
                    scheme._approx_groups = None  # force a re-initialization of approx

            self.run_linearize()
            self._jacobian._save_sparsity(self)

        sparsity_time = time.time() - sparsity_start_time

        self._update_wrt_matches(self._coloring_info)

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
        approx._colored_approx_groups = None
        approx._approx_groups = None

        if info['show_sparsity'] or info['show_summary']:
            print("\nApprox coloring for '%s' (class %s)\n" % (self.pathname, type(self).__name__))

        if info['show_sparsity']:
            coloring.display_txt()
        if info['show_summary']:
            coloring.summary()

        self._save_coloring(coloring)

        # restore original inputs/outputs
        self._inputs._data[:] = starting_inputs
        self._outputs._data[:] = starting_outputs
        self._residuals._data[:] = starting_resids

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
        for of in self._var_allprocs_abs_names['output']:
            end += abs2meta[of]['size']
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

        for wrt in self._var_allprocs_abs_names['input']:
            if wrt in wrt_matches:
                end += abs2meta[wrt]['size']
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
        directory = self._problem_options['coloring_dir']
        if not self.pathname:
            # total coloring
            return os.path.join(directory, 'total_coloring.pkl')

        per_instance = self._coloring_info.get('per_instance')

        if per_instance:
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

        if coloring is _DYN_COLORING:
            return None

        if coloring is _STD_COLORING_FNAME or isinstance(coloring, string_types):
            if coloring is _STD_COLORING_FNAME:
                fname = self.get_approx_coloring_fname()
            else:
                fname = coloring
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
            approx._colored_approx_groups = None
            approx._approx_groups = None

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
        if coloring is None and self._coloring_info['coloring'] is _DYN_COLORING:
            self._coloring_info['coloring'] = coloring = self._compute_approx_coloring()[0]
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

    def _setup_recording(self, recurse=True):
        myinputs = myoutputs = myresiduals = set()

        options = self.recording_options
        incl = options['includes']
        excl = options['excludes']

        # includes and excludes for inputs are specified using _absolute_ names
        # vectors are keyed on absolute name, discretes on relative/promoted name
        abs2prom = self._var_abs2prom['input']

        if options['record_inputs']:
            myinputs = set()
            if self._inputs:
                myinputs.update({n for n in self._inputs._names
                                 if check_path(n, incl, excl)})

            if len(self._var_discrete['input']) > 0:
                for n in self._var_discrete['input']:
                    abs_name = self.pathname + '.' + n if self.pathname else n
                    if check_path(abs_name, incl, excl):
                        myinputs.add(n)

        # includes and excludes for outputs are specified using _promoted_ names
        # vectors are keyed on absolute name, discretes on relative/promoted name
        abs2prom = self._var_abs2prom['output']

        if options['record_outputs']:
            myoutputs = set()
            if self._outputs:
                myoutputs.update({n for n in self._outputs._names
                                  if n in abs2prom and check_path(abs2prom[n], incl, excl)})

            # residuals have the same names as the continuous outputs
            if options['record_residuals']:
                myresiduals = myoutputs.copy()

            if len(self._var_discrete['output']) > 0:
                myoutputs.update({n for n in self._var_discrete['output']
                                  if check_path(n, incl, excl)})

        elif options['record_residuals']:
            if self._residuals:
                myresiduals = {n for n in self._residuals._names
                               if n in abs2prom and check_path(abs2prom[n], incl, excl)}

        self._filtered_vars_to_record = {
            'i': myinputs,
            'o': myoutputs,
            'r': myresiduals
        }

        self._rec_mgr.startup(self)

        # Recursion
        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_recording(recurse)

    def _setup_var_index_ranges(self, recurse=True):
        """
        Compute the division of variables by subsystem.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems (ignored).
        """
        self._setup_var_index_maps(recurse=recurse)

    def _setup_var_data(self, recurse=True):
        """
        Compute the list of abs var names, abs/prom name maps, and metadata dictionaries.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        self._var_allprocs_abs_names = {'input': [], 'output': []}
        self._var_abs_names = {'input': [], 'output': []}
        self._var_allprocs_prom2abs_list = {'input': OrderedDict(), 'output': OrderedDict()}
        self._var_abs2prom = {'input': {}, 'output': {}}
        self._var_allprocs_abs2meta = {}
        self._var_abs2meta = {}

    def _setup_var_index_maps(self, recurse=True):
        """
        Compute maps from abs var names to their index among allprocs variables in this system.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        self._var_allprocs_abs2idx = abs2idx = {}

        vec_names = self._lin_rel_vec_name_list if self._use_derivatives else self._vec_names

        for vec_name in vec_names:
            abs2idx[vec_name] = abs2idx_t = {}
            for type_ in ['input', 'output']:
                for i, abs_name in enumerate(self._var_allprocs_relevant_names[vec_name][type_]):
                    abs2idx_t[abs_name] = i

        if self._use_derivatives:
            abs2idx['nonlinear'] = abs2idx['linear']

        # Recursion
        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_var_index_maps(recurse)

    def _setup_var_sizes(self, recurse=True):
        """
        Compute the arrays of local variable sizes for all variables/procs on this system.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        self._var_sizes = {}
        self._owned_var_sizes = None
        self._nodup_out_ranges = None
        self._nodup2local_out_inds = None
        self._owning_rank = defaultdict(int)

    def _setup_global_shapes(self):
        """
        Compute the global size and shape of all variables on this system.
        """
        meta = self._var_allprocs_abs2meta

        # now set global sizes and shapes into metadata for distributed outputs
        sizes = self._var_sizes['nonlinear']['output']
        for idx, abs_name in enumerate(self._var_allprocs_abs_names['output']):
            mymeta = meta[abs_name]
            local_shape = mymeta['shape']
            if not mymeta['distributed']:
                # not distributed, just use local shape and size
                mymeta['global_size'] = mymeta['size']
                mymeta['global_shape'] = local_shape
                continue

            global_size = np.sum(sizes[:, idx])
            mymeta['global_size'] = global_size

            # assume that all but the first dimension of the shape of a
            # distributed output is the same on all procs
            high_dims = local_shape[1:]
            if high_dims:
                high_size = np.prod(high_dims)
                dim1 = global_size // high_size
                if global_size % high_size != 0:
                    raise RuntimeError("%s: Global size of output '%s' (%s) does not agree "
                                       "with local shape %s" % (self.msginfo, abs_name,
                                                                global_size, local_shape))
                global_shape = tuple([dim1] + list(high_dims))
            else:
                high_size = 1
                global_shape = (global_size,)
            mymeta['global_shape'] = global_shape

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
        pass

    def _setup_vec_names(self, mode, vec_names=None, vois=None):
        """
        Return the list of vec_names and the vois dict.

        Parameters
        ----------
        mode : str
            Derivative direction, either 'fwd' or 'rev'.
        vec_names : list of str or None
            The list of names of vectors. Depends on the value of mode.
        vois : dict
            Dictionary of either design vars or responses, depending on the value
            of mode.

        """
        self._vois = vois
        if vec_names is None:  # should only occur at top level on full setup
            if self._use_derivatives:
                vec_names = ['nonlinear', 'linear']
                if mode == 'fwd':
                    self._vois = vois = self.get_design_vars(recurse=True, get_sizes=False)
                else:  # rev
                    self._vois = vois = self.get_responses(recurse=True, get_sizes=False)
                vec_names.extend(sorted(set(voi for voi, data in iteritems(vois)
                                            if data['parallel_deriv_color'] is not None
                                            or data['vectorize_derivs'])))
            else:
                vec_names = ['nonlinear']
                self._vois = {}

        self._vec_names = vec_names
        self._lin_vec_names = vec_names[1:]  # only linear vec names

        for s in self.system_iter():
            s._vec_names = vec_names
            s._lin_vec_names = self._lin_vec_names

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
            desvars = self.get_design_vars(recurse=True, get_sizes=False)
            responses = self.get_responses(recurse=True, get_sizes=False)
            return get_relevant_vars(self._conn_global_abs_in2out, desvars, responses,
                                     mode)
        else:
            relevant = defaultdict(dict)
            relevant['nonlinear'] = {'@all': ({'input': ContainsAll(), 'output': ContainsAll()},
                                              ContainsAll())}
            return relevant

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

        self._var_allprocs_relevant_names = defaultdict(lambda: {'input': [], 'output': []})
        self._var_relevant_names = defaultdict(lambda: {'input': [], 'output': []})

        self._rel_vec_name_list = []
        for vec_name in self._vec_names:
            rel, relsys = relevant[vec_name]['@all']
            if self.pathname in relsys:
                self._rel_vec_name_list.append(vec_name)
            for type_ in ('input', 'output'):
                self._var_allprocs_relevant_names[vec_name][type_].extend(
                    v for v in self._var_allprocs_abs_names[type_] if v in rel[type_])
                self._var_relevant_names[vec_name][type_].extend(
                    v for v in self._var_abs_names[type_] if v in rel[type_])

        self._rel_vec_names = frozenset(self._rel_vec_name_list)
        self._lin_rel_vec_name_list = self._rel_vec_name_list[1:]

        for s in self._subsystems_myproc:
            s._setup_relevance(mode, relevant)

    def _setup_connections(self, recurse=True):
        """
        Compute dict of all connections owned by this system.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        pass

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
        self._ext_num_vars = ext_num_vars
        self._ext_sizes = ext_sizes

    def _setup_vectors(self, root_vectors, resize=False, alloc_complex=False):
        """
        Compute all vectors for all vec names and assign excluded variables lists.

        Parameters
        ----------
        root_vectors : dict of dict of Vector
            Root vectors: first key is 'input', 'output', or 'residual'; second key is vec_name.
        resize : bool
            Whether to resize the root vectors - i.e, because this system is initiating a reconf.
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

        for vec_name in self._rel_vec_name_list:

            # Only allocate complex in the vectors we need.
            vec_alloc_complex = root_vectors['output'][vec_name]._alloc_complex

            for kind in ['input', 'output', 'residual']:
                rootvec = root_vectors[kind][vec_name]
                vectors[kind][vec_name] = vector_class(
                    vec_name, kind, self, rootvec, resize=resize,
                    alloc_complex=vec_alloc_complex, ncol=rootvec._ncol)

        self._inputs = vectors['input']['nonlinear']
        self._outputs = vectors['output']['nonlinear']
        self._residuals = vectors['residual']['nonlinear']

        self._setup_bounds(root_vectors['lower'], root_vectors['upper'], resize=resize)

        for subsys in self._subsystems_myproc:
            subsys._scale_factors = self._scale_factors
            subsys._setup_vectors(root_vectors, alloc_complex=alloc_complex)

    def _setup_bounds(self, root_lower, root_upper, resize=False):
        """
        Compute the lower and upper bounds vectors and set their values.

        Parameters
        ----------
        root_lower : Vector
            Root vector for the lower bounds vector.
        root_upper : Vector
            Root vector for the upper bounds vector.
        resize : bool
            Whether to resize the root vectors - i.e, because this system is initiating a reconf.
        """
        vector_class = root_lower.__class__
        self._lower_bounds = lower = vector_class(
            'nonlinear', 'output', self, root_lower, resize=resize)

        self._upper_bounds = upper = vector_class(
            'nonlinear', 'output', self, root_upper, resize=resize)

        if self._has_bounds:
            abs2meta = self._var_abs2meta
            for abs_name in self._var_abs_names['output']:
                meta = abs2meta[abs_name]
                var_lower = meta['lower']
                var_upper = meta['upper']
                ref0 = meta['ref0']
                ref = meta['ref']

                if not np.isscalar(ref0):
                    ref0 = ref0.reshape(meta['shape'])
                if not np.isscalar(ref):
                    ref = ref.reshape(meta['shape'])

                if var_lower is not None:
                    lower._views[abs_name][:] = (var_lower - ref0) / (ref - ref0)

                if var_upper is not None:
                    upper._views[abs_name][:] = (var_upper - ref0) / (ref - ref0)

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

        allprocs_meta_out = self._var_allprocs_abs2meta

        for abs_name in self._var_allprocs_abs_names['output']:
            meta = allprocs_meta_out[abs_name]
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

    def _setup_transfers(self, recurse=True):
        """
        Compute all transfers that are owned by this system.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        pass

    def _setup_solvers(self, recurse=True):
        """
        Perform setup in all solvers.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
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

        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_solvers(recurse=recurse)

    def _setup_jacobians(self, recurse=True):
        """
        Set and populate jacobians down through the system tree.

        Parameters
        ----------
        recurse : bool
            If True, setup jacobians in all descendants.
        """
        if not self._use_derivatives:
            return

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
        abs2meta = self._var_abs2meta
        for abs_name in self._var_abs_names['input']:
            self._inputs._views[abs_name][:] = abs2meta[abs_name]['value']

        for abs_name in self._var_abs_names['output']:
            self._outputs._views[abs_name][:] = abs2meta[abs_name]['value']

    def _get_maps(self, prom_names):
        """
        Define variable maps based on promotes lists.

        Parameters
        ----------
        prom_names : {'input': [], 'output': []}
            Lists of promoted input and output names.

        Returns
        -------
        dict of {'input': {str:str, ...}, 'output': {str:str, ...}}
            dictionary mapping input/output variable names
            to promoted variable names.
        """
        gname = self.name + '.' if self.name else ''

        def split_list(lst):
            """
            Return names, patterns, and renames found in lst.
            """
            names = []
            patterns = []
            renames = {}
            for entry in lst:
                if isinstance(entry, string_types):
                    if '*' in entry or '?' in entry or '[' in entry:
                        patterns.append(entry)
                    else:
                        names.append(entry)
                elif isinstance(entry, tuple) and len(entry) == 2:
                    renames[entry[0]] = entry[1]
                else:
                    raise TypeError("when adding subsystem '%s', entry '%s'"
                                    " is not a string or tuple of size 2" %
                                    (self.pathname, entry))
            return names, patterns, renames

        def resolve(to_match, io_types, matches, proms):
            """
            Determine the mapping of promoted names to the parent scope for a promotion type.

            This is called once for promotes or separately for promotes_inputs and promotes_outputs.
            """
            if not to_match:
                for typ in io_types:
                    if gname:
                        matches[typ] = {name: gname + name for name in proms[typ]}
                    else:
                        matches[typ] = {name: name for name in proms[typ]}
                return True

            found = set()
            names, patterns, renames = split_list(to_match)
            for typ in io_types:
                pmap = matches[typ]
                for name in proms[typ]:
                    if name in names:
                        pmap[name] = name
                        found.add(name)
                    elif name in renames:
                        pmap[name] = renames[name]
                        found.add(name)
                    else:
                        for pattern in patterns:
                            # if name matches, promote that variable to parent
                            if pattern == '*' or fnmatchcase(name, pattern):
                                pmap[name] = name
                                found.add(pattern)
                                break
                        else:
                            # Default: prepend the parent system's name
                            pmap[name] = gname + name if gname else name

            not_found = (set(names).union(renames).union(patterns)) - found
            if not_found:
                if len(io_types) == 2:
                    call = 'promotes'
                else:
                    call = 'promotes_%ss' % io_types[0]
                raise RuntimeError("%s: '%s' failed to find any matches for the following "
                                   "names or patterns: %s." %
                                   (self.msginfo, call, sorted(not_found)))

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
            self._scope_cache[None] = (frozenset(self._var_abs_names['output']), _empty_frozen_set)
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

    @property
    def metadata(self):
        """
        Get the options for this System.
        """
        warn_deprecation("The 'metadata' attribute provides backwards compatibility "
                         "with earlier version of OpenMDAO; use 'options' instead.")
        return self.options

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
    def _unscaled_context_all(self):
        """
        Context manager that temporarily puts all vectors in an unscaled state.
        """
        if self._has_output_scaling:
            for vec in self._vectors['output'].values():
                vec.scale('phys')
        if self._has_resid_scaling:
            for vec in self._vectors['residual'].values():
                vec.scale('phys')

        yield

        if self._has_output_scaling:
            for vec in self._vectors['output'].values():
                vec.scale('norm')
        if self._has_resid_scaling:
            for vec in self._vectors['residual'].values():
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
                d_residuals.set_const(0.0)
            else:  # rev
                d_inputs.set_const(0.0)
                d_outputs.set_const(0.0)

        if scope_out is None and scope_in is None:
            yield d_inputs, d_outputs, d_residuals
        else:
            old_ins = d_inputs._names
            old_outs = d_outputs._names

            if scope_out is not None:
                d_outputs._names = scope_out.intersection(d_outputs._views)
            if scope_in is not None:
                d_inputs._names = scope_in.intersection(d_inputs._views)

            yield d_inputs, d_outputs, d_residuals

            # reset _names so users will see full vector contents
            d_inputs._names = old_ins
            d_outputs._names = old_outs

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
    def nl_solver(self):
        """
        Get the nonlinear solver for this system.
        """
        warn_deprecation("The 'nl_solver' attribute provides backwards compatibility "
                         "with OpenMDAO 1.x ; use 'nonlinear_solver' instead.")
        return self._nonlinear_solver

    @nl_solver.setter
    def nl_solver(self, solver):
        """
        Set this system's nonlinear solver.
        """
        warn_deprecation("The 'nl_solver' attribute provides backwards compatibility "
                         "with OpenMDAO 1.x ; use 'nonlinear_solver' instead.")
        self._nonlinear_solver = solver

    @property
    def ln_solver(self):
        """
        Get the linear solver for this system.
        """
        warn_deprecation("The 'ln_solver' attribute provides backwards compatibility "
                         "with OpenMDAO 1.x ; use 'linear_solver' instead.")
        return self._linear_solver

    @ln_solver.setter
    def ln_solver(self, solver):
        """
        Set this system's linear solver.
        """
        warn_deprecation("The 'ln_solver' attribute provides backwards compatibility "
                         "with OpenMDAO 1.x ; use 'linear_solver' instead.")
        self._linear_solver = solver

    def _set_solver_print(self, level=2, depth=1e99, type_='all'):
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
        if self._linear_solver is not None and type_ != 'NL':
            self._linear_solver._set_solver_print(level=level, type_=type_)
        if self.nonlinear_solver is not None and type_ != 'LN':
            self.nonlinear_solver._set_solver_print(level=level, type_=type_)

        for subsys in self._subsystems_allprocs:

            current_depth = subsys.pathname.count('.')
            if current_depth >= depth:
                continue

            subsys._set_solver_print(level=level, depth=depth - current_depth, type_=type_)

            if subsys._linear_solver is not None and type_ != 'NL':
                subsys._linear_solver._set_solver_print(level=level, type_=type_)
            if subsys.nonlinear_solver is not None and type_ != 'LN':
                subsys.nonlinear_solver._set_solver_print(level=level, type_=type_)

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

    def _all_subsystem_iter(self):
        """
        Yield a generator of subsystems along with their local status.

        Yields
        ------
        System
            Current subsystem.
        bool
            True if current subsystem is local.
        """
        for isub, subsys in enumerate(self._subsystems_allprocs):
            yield subsys, subsys.name in self._loc_subsys_map

    def add_design_var(self, name, lower=None, upper=None, ref=None,
                       ref0=None, indices=None, adder=None, scaler=None,
                       parallel_deriv_color=None, vectorize_derivs=False,
                       cache_linear_solution=False):
        r"""
        Add a design variable to this system.

        Parameters
        ----------
        name : string
            Name of the design variable in the system.
        lower : float or ndarray, optional
            Lower boundary for the param
        upper : upper or ndarray, optional
            Upper boundary for the param
        ref : float or ndarray, optional
            Value of design var that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of design var that scales to 0.0 in the driver.
        indices : iter of int, optional
            If a param is an array, these indicate which entries are of
            interest for this particular design variable.  These may be
            positive or negative integers.
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
        if not isinstance(name, string_types):
            raise TypeError('{}: The name argument should be a string, got {}'.format(self.msginfo,
                                                                                      name))

        # Convert ref/ref0 to ndarray/float as necessary
        ref = format_as_float_or_array('ref', ref, val_if_none=None, flatten=True)
        ref0 = format_as_float_or_array('ref0', ref0, val_if_none=None, flatten=True)

        # determine adder and scaler based on args
        adder, scaler = determine_adder_scaler(ref0, ref, adder, scaler)

        # Convert lower to ndarray/float as necessary
        lower = format_as_float_or_array('lower', lower, val_if_none=-openmdao.INF_BOUND,
                                         flatten=True)

        # Convert upper to ndarray/float as necessary
        upper = format_as_float_or_array('upper', upper, val_if_none=openmdao.INF_BOUND,
                                         flatten=True)

        # Apply scaler/adder to lower and upper
        lower = (lower + adder) * scaler
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
        dvs['cache_linear_solution'] = cache_linear_solution

        if indices is not None:
            # If given, indices must be a sequence
            if not (isinstance(indices, Iterable) and
                    all([isinstance(i, Integral) for i in indices])):
                raise ValueError("{}: If specified, design var indices must be a sequence of "
                                 "integers.".format(self.msginfo))

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
                     ref=None, ref0=None, indices=None, index=None,
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
        if not isinstance(name, string_types):
            raise TypeError('{}: The name argument should be a string, '
                            'got {}'.format(self.msginfo, name))

        # Type must be a string and one of 'con' or 'obj'
        if not isinstance(type_, string_types):
            raise TypeError('{}: The type argument should be a string'.format(self.msginfo))
        elif type_ not in ('con', 'obj'):
            raise ValueError('{}: The type must be one of \'con\' or \'obj\': '
                             'Got \'{}\' instead'.format(self.msginfo, name))

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

        # If given, indices must be a sequence
        if (indices is not None and not (
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
            lower = format_as_float_or_array('lower', lower, val_if_none=-openmdao.INF_BOUND,
                                             flatten=True)

            # Convert upper to ndarray/float as necessary
            upper = format_as_float_or_array('upper', upper, val_if_none=openmdao.INF_BOUND,
                                             flatten=True)

            # Convert equals to ndarray/float as necessary
            if equals is not None:
                equals = format_as_float_or_array('equals', equals, flatten=True)

            # Scale the bounds
            if lower is not None:
                lower = (lower + adder) * scaler

            if upper is not None:
                upper = (upper + adder) * scaler

            if equals is not None:
                equals = (equals + adder) * scaler

            resp['lower'] = lower
            resp['upper'] = upper
            resp['equals'] = equals
            resp['linear'] = linear
            if indices is not None:
                resp['size'] = len(indices)
                indices = np.atleast_1d(indices)
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
        resp['cache_linear_solution'] = cache_linear_solution

        resp['parallel_deriv_color'] = parallel_deriv_color
        resp['vectorize_derivs'] = vectorize_derivs

        responses[name] = resp

    def add_constraint(self, name, lower=None, upper=None, equals=None,
                       ref=None, ref0=None, adder=None, scaler=None,
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
        """
        self.add_response(name=name, type_='con', lower=lower, upper=upper,
                          equals=equals, scaler=scaler, adder=adder, ref=ref,
                          ref0=ref0, indices=indices, linear=linear,
                          parallel_deriv_color=parallel_deriv_color,
                          vectorize_derivs=vectorize_derivs,
                          cache_linear_solution=cache_linear_solution)

    def add_objective(self, name, ref=None, ref0=None, index=None,
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
                          ref=ref, ref0=ref0, index=index,
                          parallel_deriv_color=parallel_deriv_color,
                          vectorize_derivs=vectorize_derivs,
                          cache_linear_solution=cache_linear_solution)

    def get_design_vars(self, recurse=True, get_sizes=True):
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

        Returns
        -------
        dict
            The design variables defined in the current system and, if
            recurse=True, its subsystems.

        """
        pro2abs = self._var_allprocs_prom2abs_list['output']

        # Human readable error message during Driver setup.
        try:
            out = OrderedDict((pro2abs[name][0], data) for name, data in
                              iteritems(self._design_vars))
        except KeyError as err:
            msg = "{}: Output not found for design variable {}."
            raise RuntimeError(msg.format(self.msginfo, str(err)))

        if get_sizes:
            # Size them all
            sizes = self._var_sizes['nonlinear']['output']
            abs2idx = self._var_allprocs_abs2idx['nonlinear']
            for name in out:
                if 'size' not in out[name]:
                    if name in abs2idx:
                        out[name]['size'] = sizes[self._owning_rank[name], abs2idx[name]]
                    else:
                        out[name]['size'] = 0  # discrete var, don't know size

        if recurse:
            for subsys in self._subsystems_myproc:
                out.update(subsys.get_design_vars(recurse=recurse, get_sizes=get_sizes))

            if self.comm.size > 1 and self._subsystems_allprocs:
                allouts = self.comm.allgather(out)
                out = OrderedDict()
                for all_out in allouts:
                    out.update(all_out)

        return out

    def get_responses(self, recurse=True, get_sizes=True):
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

        Returns
        -------
        dict
            The responses defined in the current system and, if
            recurse=True, its subsystems.

        """
        prom2abs = self._var_allprocs_prom2abs_list['output']

        # Human readable error message during Driver setup.
        try:
            out = OrderedDict((prom2abs[name][0], data) for name, data in
                              iteritems(self._responses))
        except KeyError as err:
            msg = "{}: Output not found for response {}."
            raise RuntimeError(msg.format(self.msginfo, str(err)))

        if get_sizes:
            # Size them all
            sizes = self._var_sizes['nonlinear']['output']
            abs2idx = self._var_allprocs_abs2idx['nonlinear']
            for name in out:
                if 'size' not in out[name]:
                    if name in abs2idx:
                        out[name]['size'] = sizes[self._owning_rank[name], abs2idx[name]]
                    else:
                        out[name]['size'] = 0  # discrete var, we don't know the size

        if recurse:
            for subsys in self._subsystems_myproc:
                out.update(subsys.get_responses(recurse=recurse, get_sizes=get_sizes))

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

    def list_inputs(self,
                    values=True,
                    prom_name=False,
                    units=False,
                    shape=False,
                    hierarchical=True,
                    print_arrays=False,
                    tags=None,
                    out_stream=_DEFAULT_OUT_STREAM):
        """
        Return and optionally log a list of input names and other optional information.

        If the model is parallel, only the local variables are returned to the process.
        Also optionally logs the information to a user defined output stream. If the model is
        parallel, the rank 0 process logs information about all variables across all processes.

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
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        list
            list of input names and other optional information about those inputs
        """
        if self._inputs is None:
            raise RuntimeError("{}: Unable to list inputs until model has "
                               "been run.".format(self.msginfo))

        meta = self._var_abs2meta
        inputs = []

        for var_name, val in iteritems(self._inputs._views):  # This is only over the locals
            # Filter based on tags
            if tags and not (make_set(tags) & meta[var_name]['tags']):
                continue

            var_meta = {}
            if values:
                var_meta['value'] = val
            if prom_name:
                var_meta['prom_name'] = self._var_abs2prom['input'][var_name]
            if units:
                var_meta['units'] = meta[var_name]['units']
            if shape:
                var_meta['shape'] = val.shape

            inputs.append((var_name, var_meta))

        if self._discrete_inputs:
            disc_meta = self._discrete_inputs._dict

            for var_name, val in iteritems(self._discrete_inputs):
                # Filter based on tags
                if tags and not (make_set(tags) & disc_meta[var_name]['tags']):
                    continue

                var_meta = {}
                if values:
                    var_meta['value'] = val
                if prom_name:
                    var_meta['prom_name'] = self._var_abs2prom['input'][var_name]
                # remaining items do not apply for discrete vars
                if units:
                    var_meta['units'] = ''
                if shape:
                    var_meta['shape'] = ''

                abs_name = self.pathname + '.' + var_name if self.pathname else var_name

                inputs.append((abs_name, var_meta))

        if out_stream is _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        if out_stream:
            self._write_table('input', inputs, hierarchical, print_arrays, out_stream)

        return inputs

    def list_outputs(self,
                     explicit=True, implicit=True,
                     values=True,
                     prom_name=False,
                     residuals=False,
                     residuals_tol=None,
                     units=False,
                     shape=False,
                     bounds=False,
                     scaling=False,
                     hierarchical=True,
                     print_arrays=False,
                     tags=None,
                     out_stream=_DEFAULT_OUT_STREAM):
        """
        Return and optionally log a list of output names and other optional information.

        If the model is parallel, only the local variables are returned to the process.
        Also optionally logs the information to a user defined output stream. If the model is
        parallel, the rank 0 process logs information about all variables across all processes.

        Parameters
        ----------
        explicit : bool, optional
            include outputs from explicit components. Default is True.
        implicit : bool, optional
            include outputs from implicit components. Default is True.
        values : bool, optional
            When True, display/return output values. Default is True.
        prom_name : bool, optional
            When True, display/return the promoted name of the variable.
            Default is False.
        residuals : bool, optional
            When True, display/return residual values. Default is False.
        residuals_tol : float, optional
            If set, limits the output of list_outputs to only variables where
            the norm of the resids array is greater than the given 'residuals_tol'.
            Default is None.
        units : bool, optional
            When True, display/return units. Default is False.
        shape : bool, optional
            When True, display/return the shape of the value. Default is False.
        bounds : bool, optional
            When True, display/return bounds (lower and upper). Default is False.
        scaling : bool, optional
            When True, display/return scaling (ref, ref0, and res_ref). Default is False.
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
        out_stream : file-like
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        list
            list of output names and other optional information about those outputs
        """
        if self._outputs is None:
            raise RuntimeError("{}: Unable to list outputs until model has "
                               "been run.".format(self.msginfo))

        # Only gathering up values and metadata from this proc, if MPI
        meta = self._var_abs2meta  # This only includes metadata for this process.
        states = self._list_states()

        # Go though the hierarchy. Printing Systems
        # If the System owns an output directly, show its output
        expl_outputs = []
        impl_outputs = []
        for var_name, val in iteritems(self._outputs._views):
            # Filter based on tags
            if tags and not (make_set(tags) & meta[var_name]['tags']):
                continue

            if residuals_tol and np.linalg.norm(self._residuals._views[var_name]) < residuals_tol:
                continue

            var_meta = {}
            if values:
                var_meta['value'] = val
            if prom_name:
                var_meta['prom_name'] = self._var_abs2prom['output'][var_name]
            if residuals:
                var_meta['resids'] = self._residuals._views[var_name]
            if units:
                var_meta['units'] = meta[var_name]['units']
            if shape:
                var_meta['shape'] = val.shape
            if bounds:
                var_meta['lower'] = meta[var_name]['lower']
                var_meta['upper'] = meta[var_name]['upper']
            if scaling:
                var_meta['ref'] = meta[var_name]['ref']
                var_meta['ref0'] = meta[var_name]['ref0']
                var_meta['res_ref'] = meta[var_name]['res_ref']

            if var_name in states:
                impl_outputs.append((var_name, var_meta))
            else:
                expl_outputs.append((var_name, var_meta))

        if self._discrete_outputs and not residuals_tol:
            disc_meta = self._discrete_outputs._dict

            for var_name, val in iteritems(self._discrete_outputs):
                # Filter based on tags
                if tags and not (make_set(tags) & disc_meta[var_name]['tags']):
                    continue

                var_meta = {}
                if values:
                    var_meta['value'] = val
                if prom_name:
                    var_meta['prom_name'] = self._var_abs2prom['output'][var_name]
                # remaining items do not apply for discrete vars
                if residuals:
                    var_meta['resids'] = ''
                if units:
                    var_meta['units'] = ''
                if shape:
                    var_meta['shape'] = ''
                if bounds:
                    var_meta['lower'] = ''
                    var_meta['upper'] = ''
                if scaling:
                    var_meta['ref'] = ''
                    var_meta['ref0'] = ''
                    var_meta['res_ref'] = ''

                abs_name = self.pathname + '.' + var_name if self.pathname else var_name

                if var_name in states:
                    impl_outputs.append((abs_name, var_meta))
                else:
                    expl_outputs.append((abs_name, var_meta))

        if out_stream is _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        if out_stream:
            if explicit:
                self._write_table('explicit', expl_outputs, hierarchical, print_arrays, out_stream)
            if implicit:
                self._write_table('implicit', impl_outputs, hierarchical, print_arrays, out_stream)

        if explicit and implicit:
            return expl_outputs + impl_outputs
        elif explicit:
            return expl_outputs
        elif implicit:
            return impl_outputs
        else:
            raise RuntimeError(self.msginfo +
                               ': You have excluded both Explicit and Implicit components.')

    def _write_table(self, var_type, var_data, hierarchical, print_arrays, out_stream):
        """
        Write table of variable names, values, residuals, and metadata to out_stream.

        Parameters
        ----------
        var_type : 'input', 'explicit' or 'implicit'
            Indicates type of variables, input or explicit/implicit output.
        var_data : list
            List of (name, dict of vals and metadata) tuples.
        hierarchical : bool
            When True, human readable output shows variables in hierarchical format.
        print_arrays : bool
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format  is affected
            by the values set with numpy.set_printoptions
            Default is False.
        out_stream : file-like object
            Where to send human readable output.
            Set to None to suppress.
        """
        if out_stream is None:
            return

        # Make a dict of variables. Makes it easier to work with in this method
        var_dict = OrderedDict()
        for name, vals in var_data:
            var_dict[name] = vals

        # If parallel, gather up the vars.
        if MPI:
            # All procs must call this. Returns a list, one per proc.
            all_var_dicts = self.comm.gather(var_dict, root=0)

            if MPI.COMM_WORLD.rank > 0:  # only the root process should print
                return

            # rest of this only done on rank 0
            meta = self._var_abs2meta

            var_dict = all_var_dicts[0]  # start with rank 0

            for proc_vars in all_var_dicts[1:]:  # In rank order go through rest of the procs
                for name, vals in iteritems(proc_vars):
                    if name not in var_dict:     # If not in the merged dict, add it
                        var_dict[name] = proc_vars[name]
                    else:
                        # In there already, only need to deal with it if it is a distributed array
                        # Checking to see if distributed depends on if it is an input or output
                        if var_type == 'input':
                            is_distributed = meta[name]['src_indices'] is not None
                        else:
                            is_distributed = meta[name]['distributed']
                        if is_distributed:
                            # TODO no support for > 1D arrays
                            #   meta.src_indices has the info we need to piece together arrays
                            if 'value' in var_dict[name]:
                                var_dict[name]['value'] = \
                                    np.append(var_dict[name]['value'],
                                              proc_vars[name]['value'])
                            if 'shape' in var_dict[name]:
                                # TODO might want to use allprocs_abs2meta_out[name]['global_shape']
                                var_dict[name]['shape'] = \
                                    var_dict[name]['value'].shape
                            if 'resids' in var_dict[name]:
                                var_dict[name]['resids'] = \
                                    np.append(var_dict[name]['resids'],
                                              proc_vars[name]['resids'])

        inputs = var_type is 'input'
        outputs = not inputs
        var_list = self._get_vars_exec_order(inputs=inputs, outputs=outputs, variables=var_dict)

        write_var_table(self.pathname, var_list, var_type, var_dict,
                        hierarchical, print_arrays, out_stream)

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
            Absolute path names of the variables to include.
            If None then all varables will be included. Default is None.

        Returns
        -------
        list
            list of variable names in execution order
        """
        var_list = []

        real_vars = self._var_allprocs_abs_names
        disc_vars = self._var_allprocs_discrete

        in_or_out = []
        if inputs:
            in_or_out.append('input')
        if outputs:
            in_or_out.append('output')

        if self._subsystems_allprocs:
            for subsys in self._subsystems_allprocs:
                # subsys.pathname will only be defined properly if a subsystem is local,
                # but subsys.name will be properly defined.
                path = '.'.join((self.pathname, subsys.name)) if self.pathname else subsys.name
                path += '.'
                for var_type in in_or_out:
                    for var_name in real_vars[var_type]:
                        if (not variables or var_name in variables) and var_name.startswith(path):
                            var_list.append(var_name)
                    for var_name in disc_vars[var_type]:
                        if (not variables or var_name in variables) and var_name.startswith(path):
                            var_list.append(var_name)
        else:
            # For components with no children, self._subsystems_allprocs is empty.
            for var_type in in_or_out:
                for var_name in real_vars[var_type]:
                    if not variables or var_name in variables:
                        var_list.append(var_name)
                for var_name in disc_vars[var_type]:
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

    def _solve_nonlinear(self):
        """
        Compute outputs. The model is assumed to be in a scaled state.

        """
        # Reconfigure if needed.
        self._check_self_reconf()

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
        if self._rec_mgr._recorders:
            options = self.recording_options

            metadata = create_local_meta(self.pathname)

            # Get the data to record
            stack_top = self._recording_iter.stack[-1][0]
            method = stack_top.split('.')[-1]

            if method not in ['_apply_linear', '_apply_nonlinear', '_solve_linear',
                              '_solve_nonlinear']:
                raise ValueError("{}: {} must be one of: '_apply_linear, _apply_nonlinear, "
                                 "_solve_linear, _solve_nonlinear'".format(self.msginfo, method))

            if 'nonlinear' in method:
                inputs, outputs, residuals = self.get_nonlinear_vectors()
            else:
                inputs, outputs, residuals = self.get_linear_vectors()

            discrete_inputs = self._discrete_inputs
            discrete_outputs = self._discrete_outputs

            data = {}
            if options['record_inputs'] and (inputs._names or len(discrete_inputs) > 0):
                data['i'] = {}
                if 'i' in self._filtered_vars_to_record:
                    # use filtered inputs
                    for inp in self._filtered_vars_to_record['i']:
                        if inp in inputs._names:
                            data['i'][inp] = inputs._views[inp]
                        elif inp in discrete_inputs:
                            abs_name = self.pathname + '.' + inp if self.pathname else inp
                            data['i'][abs_name] = discrete_inputs[inp]
                else:
                    # use all the inputs
                    if len(discrete_inputs) > 0:
                        for inp in inputs:
                            data['i'][inp] = inputs._views[inp]
                        for inp in discrete_inputs:
                            abs_name = self.pathname + '.' + inp if self.pathname else inp
                            data['i'][abs_name] = discrete_inputs[inp]
                    else:
                        data['i'] = inputs._names

            else:
                data['i'] = None

            if options['record_outputs'] and (outputs._names or len(discrete_outputs) > 0):
                data['o'] = {}
                if 'o' in self._filtered_vars_to_record:
                    # use outputs from filtered list.
                    for out in self._filtered_vars_to_record['o']:
                        if out in outputs._names:
                            data['o'][out] = outputs._views[out]
                        elif out in discrete_outputs:
                            abs_name = self.pathname + '.' + out if self.pathname else out
                            data['o'][abs_name] = discrete_outputs[out]
                else:
                    # use all the outputs
                    if len(discrete_outputs) > 0:
                        for out in outputs:
                            data['o'][out] = outputs._views[out]
                        for out in discrete_outputs:
                            abs_name = self.pathname + '.' + out if self.pathname else out
                            data['o'][abs_name] = discrete_outputs[out]
                    else:
                        data['o'] = outputs._names
            else:
                data['o'] = None

            if options['record_residuals'] and residuals._names:
                data['r'] = {}

                if 'r' in self._filtered_vars_to_record:
                    # use filtered residuals
                    for res in self._filtered_vars_to_record['r']:
                        if res in residuals._names:
                            data['r'][res] = residuals._views[res]
                else:
                    # use all the residuals
                    data['r'] = residuals._names
            else:
                data['r'] = None

            self._rec_mgr.record_iteration(self, data, metadata)

        self.iter_count += 1

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

    def _get_nodup_out_ranges(self):
        """
        Compute necessary ranges/indices for working with non-dup global outputs array.

        Returns
        -------
        OrderedDict
            Tuples of the form (start, end) keyed on variable name.
        ndarray
            Index array mapping global non-dup outputs/resids to local outputs/resids.
        ndarray
            Index array mapping local outputs/resids to owned local outputs/resids.
        ndarray
            Index array mapping global stacked (rank order) array to global array where
            distrib vars are contiguous and all vars appear in global execution order.
            Execution order is meaningless for systems in ParallelGroups, but for purposes
            of global ordering, the declared execution order, which is the same across all
            ranks, is used.
        """
        if self._nodup_out_ranges is None:
            iproc = self.comm.rank
            abs2meta = self._var_allprocs_abs2meta
            sizes = self._var_sizes['linear']['output']
            owned_sizes = self._owned_sizes

            ranges = OrderedDict()
            out_views = self._outputs._views

            # compute offsets into the full non-dup output/resid array by summing down columns
            # of owned_sizes array. This results in the distributed vars being contiguous, and
            # having the same offsets in every proc so that overlapping indices, etc. will be
            # properly handled.
            contig_offsets = sizes2offsets(np.sum(owned_sizes, axis=0))

            # compute offsets into the full non-dup output/resid array where distrib vars are not
            # contiguous
            offsets = sizes2offsets(owned_sizes)

            # order ranks with our rank first
            ordered_ranks = [iproc] + [r for r in range(self.comm.size) if r != iproc]

            has_distribs = False
            contig_inds = []
            non_contig_inds = []
            # compute ranges/indices into the full non-duplicated output/resid arrays
            for i, name in enumerate(self._var_allprocs_abs_names['output']):
                distrib = abs2meta[name]['distributed']
                has_distribs |= distrib
                found = False
                # check each rank (this rank first) for the first nonzero size
                for irank in ordered_ranks:
                    size = owned_sizes[irank, i]

                    if size > 0:
                        if not found:
                            found = True
                            dsize = np.sum(owned_sizes[:, i]) if distrib else size

                            contig_start = contig_offsets[i]
                            ranges[name] = (contig_start, contig_start + dsize)
                            if name in out_views:
                                if distrib:
                                    # need offset into the dist var
                                    dstart = contig_start + np.sum(owned_sizes[:irank, i])
                                    contig_inds.append(np.arange(dstart, dstart + size, dtype=int))
                                else:
                                    contig_inds.append(np.arange(*ranges[name], dtype=int))

                        non_contig_start = offsets[irank, i]
                        non_contig_inds.append(np.arange(non_contig_start, non_contig_start + size))

            self._nodup_out_ranges = ranges
            self._nodup2local_out_inds = _arraylist2array(contig_inds)

            # get indices to pull out only the 'owned' values from the local array
            local2owned_inds = []
            start = end = 0
            for owned_sz, sz in zip(owned_sizes[iproc], sizes[iproc]):
                if sz == 0:
                    continue
                end += sz
                if owned_sz > 0:
                    local2owned_inds.append(np.arange(start, end, dtype=int))
                start = end

            self._local2owned_inds = _arraylist2array(local2owned_inds)

            # compute inds to map gathered nodup order to nodup ordered by ownership
            self._noncontig_dis_inds = _arraylist2array(non_contig_inds)

        return (self._nodup_out_ranges, self._nodup2local_out_inds, self._local2owned_inds,
                self._noncontig_dis_inds)


def _arraylist2array(lst, dtype=int):
    """
    Given a list of arrays, return a stacked array of the specified dtype.

    Parameters
    ----------
    lst : list
        List of arrays.
    dtype : type
        Specified dtype for the return array.

    Returns
    -------
    ndarray
        The stacked array.
    """
    if len(lst) > 1:
        return np.hstack(lst)
    elif lst:
        return lst[0]

    return np.zeros(0, dtype=dtype)


def get_relevant_vars(connections, desvars, responses, mode):
    """
    Find all relevant vars between desvars and responses.

    Both vars are assumed to be outputs (either design vars or responses).

    Parameters
    ----------
    connections : dict
        Mapping of targets to their sources.
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
                if mode == 'fwd' or mode == 'auto':
                    relevant[desvar][response] = ({'input': input_deps,
                                                   'output': output_deps}, sys_deps)
                if mode == 'rev' or mode == 'auto':
                    relevant[response][desvar] = ({'input': input_deps,
                                                   'output': output_deps}, sys_deps)

                sys_deps.add('')  # top level Group is always relevant

    voi_lists = []
    if mode == 'fwd' or mode == 'auto':
        voi_lists.append((desvars, responses))
    if mode == 'rev' or mode == 'auto':
        voi_lists.append((responses, desvars))

    # now calculate dependencies between each VOI and all other VOIs of the
    # other type, e.g for each input VOI wrt all output VOIs.  This is only
    # done for design vars in fwd mode or responses in rev mode. In auto mode,
    # we combine the results for fwd and rev modes.
    for inputs, outputs in voi_lists:
        for inp in inputs:
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
