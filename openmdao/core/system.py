"""Define the base System class."""
from __future__ import division

from contextlib import contextmanager
from collections import OrderedDict, Iterable, defaultdict
from fnmatch import fnmatchcase
import sys
from itertools import product
from numbers import Integral

from six import iteritems, string_types
from six.moves import range

import numpy as np

from openmdao.jacobians.dictionary_jacobian import DictionaryJacobian
from openmdao.jacobians.assembled_jacobian import AssembledJacobian, DenseJacobian
from openmdao.utils.general_utils import determine_adder_scaler, \
    format_as_float_or_array, warn_deprecation, ContainsAll
from openmdao.recorders.recording_manager import RecordingManager
from openmdao.recorders.recording_iteration_stack import recording_iteration, \
    get_formatted_iteration_coordinate
from openmdao.utils.mpi import MPI
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.units import get_conversion
from openmdao.utils.array_utils import convert_neg
from openmdao.utils.record_util import create_local_meta, check_path

# Use this as a special value to be able to tell if the caller set a value for the optional
#   out_stream argument. We run into problems running testflo if we use a default of sys.stdout.
_DEFAULT_OUT_STREAM = object()


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
    metadata : <OptionsDictionary>
        Dictionary of user-defined arguments.
    iter_count : int
        Int that holds the number of times this system has iterated
        in a recording run.
    cite : str
        Listing of relevant citataions that should be referenced when
        publishing work that uses this class.
    _subsystems_allprocs : [<System>, ...]
        List of all subsystems (children of this system).
    _subsystems_myproc : [<System>, ...]
        List of local subsystems that exist on this proc.
    _subsystems_myproc_inds : [int, ...]
        List of indices of subsystems on this proc among all of this system's subsystems
        (i.e. among _subsystems_allprocs).
    _subsystems_proc_range : (int, int)
        List of ranges of each myproc subsystem's processors relative to those of this system.
    _subsystems_var_range : {'input': list of (int, int), 'output': list of (int, int)}
        List of ranges of each myproc subsystem's allprocs variables relative to this system.
    _subsystems_var_range_byset : {<vec_name>: {'input': list of dict, 'output': list of dict}, ...}
        Same as above, but by var_set name.
    _num_var : {<vec_name>: {'input': int, 'output': int}, ...}
        Number of allprocs variables owned by this system.
    _num_var_byset : {<vec_name>: {'input': dict of int, 'output': dict of int}, ...}
        Same as above, but by var_set name.
    _var_set2iset : {'input': dict, 'output': dict}
        Dictionary mapping the var_set name to the var_set index.
    _var_promotes : { 'any': [], 'input': [], 'output': [] }
        Dictionary of lists of variable names/wildcards specifying promotion
        (used to calculate promoted names)
    _var_allprocs_abs_names : {'input': [str, ...], 'output': [str, ...]}
        List of absolute names of this system's variables on all procs.
    _var_abs_names : {'input': [str, ...], 'output': [str, ...]}
        List of absolute names of this system's variables existing on current proc.
    _var_allprocs_prom2abs_list : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to list of all absolute names.
        For outputs, the list will have length one since promoted output names are unique.
    _var_abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names, on current proc.
    _var_allprocs_abs2meta : dict
        Dictionary mapping absolute names to metadata dictionaries for allprocs variables.
        The keys are
        ('units', 'shape', 'size', 'var_set') for inputs and
        ('units', 'shape', 'size', 'var_set', 'ref', 'ref0', 'res_ref', 'distributed') for outputs.
    _var_abs2meta : dict
        Dictionary mapping absolute names to metadata dictionaries for myproc variables.
    _var_allprocs_abs2idx : dict
        Dictionary mapping absolute names to their indices among this system's allprocs variables.
        Therefore, the indices range from 0 to the total number of this system's variables.
    _var_allprocs_abs2idx_byset : {<vec_name>: {dict of dict}, ...}
        Same as above, but by var_set name.
    _var_sizes : {'input': ndarray, 'output': ndarray}
        Array of local sizes of this system's allprocs variables.
        The array has size nproc x num_var where nproc is the number of processors
        owned by this system and num_var is the number of allprocs variables.
    _var_sizes_byset : {'input': dict of ndarray, 'output': dict of ndarray}
        Same as above, but by var_set name.
    _manual_connections : dict
        Dictionary of input_name: (output_name, src_indices) connections.
    _conn_global_abs_in2out : {'abs_in': 'abs_out'}
        Dictionary containing all explicit & implicit connections owned by this system
        or any descendant system. The data is the same across all processors.
    _conn_abs_in2out : {'abs_in': 'abs_out'}
        Dictionary containing all explicit & implicit connections owned
        by this system only. The data is the same across all processors.
    _ext_num_vars : {'input': (int, int), 'output': (int, int)}
        Total number of allprocs variables in system before/after this one.
    _ext_num_vars_byset : {'input': dict of (int, int), 'output': dict of (int, int)}
        Same as above, but by var_set name.
    _ext_sizes : {'input': (int, int), 'output': (int, int)}
        Total size of allprocs variables in system before/after this one.
    _ext_sizes_byset : {'input': dict of (int, int), 'output': dict of (int, int)}
        Same as above, but by var_set name.
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
    _transfers : dict of dict of Transfers
        First key is the vec_name, second key is (mode, isub) where
        mode is 'fwd' or 'rev' and isub is the subsystem index among allprocs subsystems
        or isub can be None for the full, simultaneous transfer.
    _lower_bounds : <Vector>
        Vector of lower bounds, scaled and dimensionless.
    _upper_bounds : <Vector>
        Vector of upper bounds, scaled and dimensionless.
    _nonlinear_solver : <NonlinearSolver>
        Nonlinear solver to be used for solve_nonlinear.
    _linear_solver : <LinearSolver>
        Linear solver to be used for solve_linear; not the Newton system.
    _approx_schemes : OrderedDict
        A mapping of approximation types to the associated ApproximationScheme.
    _jacobian : <Jacobian>
        <Jacobian> object to be used in apply_linear.
    _jacobian_changed : bool
        If True, the jacobian has changed since the last call to setup.
    _owns_assembled_jac : bool
        If True, we are owners of the AssembledJacobian in self._jacobian.
    _owns_approx_jac : bool
        If True, this system approximated its Jacobian
    _owns_approx_jac_meta : dict
        Stores approximation metadata (e.g., step_size) from calls to approx_totals
    _owns_approx_of : set or None
        Overrides aproximation outputs. This is set when calculating system derivatives, and serves
        as a way to communicate the driver's output quantities to the approximation objects so that
        we only take derivatives of variables that the driver needs.
    _owns_approx_of_idx : dict
        Index for override 'of' approximations if declared. When the user calls  `add_objective`
        or `add_constraint`, they may optionally specify an "indices" argument. This argument must
        also be communicated to the approximations when they are set up so that 1) the Jacobian is
        the correct size, and 2) we don't perform any extra unnecessary calculations.
    _owns_approx_wrt : set or None
        Overrides aproximation inputs. This is set when calculating system derivatives, and serves
        as a way to communicate the driver's input quantities to the approximation objects so that
        we only take derivatives with respect to variables that the driver needs.
    _owns_approx_wrt_idx : dict
        Index for override 'wrt' approximations if declared. When the user calls  `add_designvar`
        they may optionally specify an "indices" argument. This argument must also be communicated
        to the approximations when they are set up so that 1) the Jacobian is the correct size, and
        2) we don't perform any extra unnecessary calculations.
    _subjacs_info : OrderedDict of dict
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
    _static_manual_connections : dict
        Dictionary that stores all explicit connections added outside of setup.
    _static_design_vars : dict of dict
        Driver design variables added outside of setup.
    _static_responses : dict of dict
        Driver responses added outside of setup.
    _reconfigured : bool
        If True, this system has reconfigured, and the immediate parent should update.
    supports_multivecs : bool
        If True, this system overrides compute_multi_jacvec_product (if an ExplicitComponent),
        or solve_multi_linear/apply_multi_linear (if an ImplicitComponent).
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
    _owning_rank : dict
        Dict mapping var name to the lowest rank where that variable is local.
    options : OptionsDictionary
        options dictionary
    recording_options : OptionsDictionary
        Recording options dictionary
    _filtered_vars_to_record: Dict
        Dict of list of var names to record
    _norm0: float
        Normalization factor
    _column_widths : dict
        widths of the columns
    _align : str
        The Python formatting alignment used when writing values into columns
    _column_spacing: int
        Number of spaces between columns
    _indent_inc: int
        Number of spaces indented in levels of the hierarchy
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            available here and in all descendants of this system.
        """
        self.name = ''
        self.pathname = ''
        self.comm = None
        self.metadata = OptionsDictionary()

        # System options
        self.options = OptionsDictionary()
        self.recording_options = OptionsDictionary()
        self.recording_options.declare('record_inputs', types=bool, default=True,
                                       desc='Set to True to record inputs at the system level')
        self.recording_options.declare('record_outputs', types=bool, default=True,
                                       desc='Set to True to record outputs at the system level')
        self.recording_options.declare('record_residuals', types=bool, default=True,
                                       desc='Set to True to record residuals at the system level')
        self.recording_options.declare('record_metadata', types=bool, desc='Record metadata',
                                       default=True)
        self.recording_options.declare('includes', types=list, default=['*'],
                                       desc='Patterns for variables to include in recording')
        self.recording_options.declare('excludes', types=list, default=[],
                                       desc='Patterns for vars to exclude in recording '
                                       '(processed post-includes)')

        # Case recording related
        self.iter_count = 0

        self.cite = ""

        self._subsystems_allprocs = []
        self._subsystems_myproc = []
        self._subsystems_myproc_inds = []
        self._subsystems_proc_range = []

        self._num_var = {'input': 0, 'output': 0}
        self._num_var_byset = None
        self._var_set2iset = OrderedDict([('input', {}), ('output', {})])

        self._var_promotes = {'input': [], 'output': [], 'any': []}
        self._var_allprocs_abs_names = {'input': [], 'output': []}
        self._var_abs_names = {'input': [], 'output': []}
        self._var_allprocs_prom2abs_list = None
        self._var_abs2prom = {'input': {}, 'output': {}}
        self._var_allprocs_abs2meta = {}
        self._var_abs2meta = {}

        self._var_allprocs_abs2idx = {}
        self._var_allprocs_abs2idx_byset = None

        self._var_sizes = None
        self._var_sizes_byset = None

        self._manual_connections = {}
        self._conn_global_abs_in2out = {}
        self._conn_abs_in2out = {}

        self._ext_num_vars = {'input': (0, 0), 'output': (0, 0)}
        self._ext_num_vars_byset = {'input': {}, 'output': {}}
        self._ext_sizes = {'input': (0, 0), 'output': (0, 0)}
        self._ext_sizes_byset = {'input': {}, 'output': {}}

        self._vectors = {'input': {}, 'output': {}, 'residual': {}}

        self._inputs = None
        self._outputs = None
        self._residuals = None
        self._transfers = {}

        self._lower_bounds = None
        self._upper_bounds = None

        self._nonlinear_solver = None
        self._linear_solver = None

        self._jacobian = DictionaryJacobian()
        self._jacobian._system = self
        self._jacobian_changed = True
        self._approx_schemes = OrderedDict()
        self._owns_assembled_jac = False
        self._subjacs_info = {}

        self._owns_approx_jac = False
        self._owns_approx_jac_meta = {}
        self._owns_approx_wrt = None
        self._owns_approx_of = None
        self._owns_approx_wrt_idx = {}
        self._owns_approx_of_idx = {}

        self._design_vars = OrderedDict()
        self._responses = OrderedDict()
        self._rec_mgr = RecordingManager()

        self._static_mode = True
        self._static_subsystems_allprocs = []
        self._static_manual_connections = {}
        self._static_design_vars = OrderedDict()
        self._static_responses = OrderedDict()

        self._reconfigured = False
        self.supports_multivecs = False

        self._relevant = None
        self._vec_names = None
        self._vois = None
        self._mode = None

        self._scope_cache = {}

        self.initialize()
        self.metadata.update(kwargs)

        self._has_guess = False
        self._has_output_scaling = False
        self._has_resid_scaling = False
        self._has_input_scaling = False

        # Formatting parameters for _write_outputs and _write_output_row methods
        #     that are used by list_inputs and list_outputs methods
        self._column_widths = {
            'value': 20,
            'resids': 20,
            'units': 10,
            'shape': 10,
            'lower': 20,
            'upper': 20,
            'ref': 20,
            'ref0': 20,
            'res_ref': 20,
        }
        self._align = ''
        self._column_spacing = 2
        self._indent_inc = 2

    def _check_reconf(self):
        """
        Check if this systems wants to reconfigure and if so, perform the reconfiguration.
        """
        reconf = self.reconfigure()

        if reconf:
            with self._unscaled_context_all():
                # Backup input values
                old = {'input': self._inputs, 'output': self._outputs}

                # Perform reconfiguration
                self.resetup('reconf')

                new = {'input': self._inputs, 'output': self._outputs}

                # Reload input and output values where possible
                for type_ in ['input', 'output']:
                    for abs_name, old_view in iteritems(old[type_]._views_flat):
                        if abs_name in new[type_]._views_flat:
                            new_view = new[type_]._views_flat[abs_name]

                            if len(old_view) == len(new_view):
                                new_view[:] = old_view

            self._reconfigured = True

    def _check_reconf_update(self):
        """
        Check if any subsystem has reconfigured and if so, perform the necessary update setup.
        """
        # See if any local subsystem has reconfigured
        reconf = np.any([subsys._reconfigured for subsys in self._subsystems_myproc])

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

    def _get_initial_var_indices(self, initial):
        """
        Get initial values for _var_set2iset.

        Parameters
        ----------
        initial : bool
            Whether we are reconfiguring - i.e., whether the model has been previously setup.

        Returns
        -------
        {'input': dict, 'output': dict}
            Dictionary mapping the var_set name to the var_set index.
        """
        if not initial:
            return self._var_set2iset
        else:
            set2iset = {}
            for type_ in ['input', 'output']:
                set2iset[type_] = {}
                for iset, set_name in enumerate(self._num_var_byset['nonlinear'][type_]):
                    set2iset[type_][set_name] = iset

            return set2iset

    def _get_initial_global(self, initial):
        """
        Get initial values for _ext_num_vars, _ext_num_vars_byset, _ext_sizes, _ext_sizes_byset.

        Parameters
        ----------
        initial : bool
            Whether we are reconfiguring - i.e., the model has been previously setup.

        Returns
        -------
        _ext_num_vars : {'input': (int, int), 'output': (int, int)}
            Total number of allprocs variables in system before/after this one.
        _ext_num_vars_byset : {'input': dict of (int, int), 'output': dict of (int, int)}
            Same as above, but by var_set name.
        _ext_sizes : {'input': (int, int), 'output': (int, int)}
            Total size of allprocs variables in system before/after this one.
        _ext_sizes_byset : {'input': dict of (int, int), 'output': dict of (int, int)}
            Same as above, but by var_set name.
        """
        if not initial:
            return (self._ext_num_vars, self._ext_num_vars_byset,
                    self._ext_sizes, self._ext_sizes_byset)
        else:
            ext_num_vars = {}
            ext_sizes = {}
            ext_num_vars_byset = {}
            ext_sizes_byset = {}

            for vec_name in self._lin_rel_vec_name_list:
                ext_num_vars[vec_name] = {}
                ext_num_vars_byset[vec_name] = {}
                ext_sizes[vec_name] = {}
                ext_sizes_byset[vec_name] = {}
                for type_ in ['input', 'output']:
                    ext_num_vars[vec_name][type_] = (0, 0)
                    ext_sizes[vec_name][type_] = (0, 0)
                    ext_num_vars_byset[vec_name][type_] = {
                        set_name: (0, 0) for set_name in self._var_set2iset[type_]
                    }
                    ext_sizes_byset[vec_name][type_] = {
                        set_name: (0, 0) for set_name in self._var_set2iset[type_]
                    }

            ext_num_vars['nonlinear'] = ext_num_vars['linear']
            ext_num_vars_byset['nonlinear'] = ext_num_vars_byset['linear']
            ext_sizes['nonlinear'] = ext_sizes['linear']
            ext_sizes_byset['nonlinear'] = ext_sizes_byset['linear']

            return ext_num_vars, ext_num_vars_byset, ext_sizes, ext_sizes_byset

    def _get_root_vectors(self, vector_class, initial, force_alloc_complex=False):
        """
        Get the root vectors for the nonlinear and linear vectors for the model.

        Parameters
        ----------
        vector_class : Vector
            The Vector class used to instantiate the root vectors.
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
            vec_names = self._rel_vec_name_list
            vois = self._vois
            iproc = self.comm.rank
            abs2idx = self._var_allprocs_abs2idx

            # Check for complex step to set vectors up appropriately.
            # If any subsystem needs complex step, then we need to allocate it everywhere.
            nl_alloc_complex = force_alloc_complex
            for sub in self.system_iter(include_self=True, recurse=True):
                nl_alloc_complex |= 'cs' in sub._approx_schemes
                if nl_alloc_complex:
                    break

            if self._has_input_scaling or self._has_output_scaling or self._has_resid_scaling:
                self._scale_factors = self._compute_root_scale_factors()
            else:
                self._scale_factors = {}

            for vec_name in vec_names:
                sizes = self._var_sizes[vec_name]['output']
                ncol = 1
                rel = None
                if vec_name == 'nonlinear':
                    alloc_complex = nl_alloc_complex
                else:
                    alloc_complex = force_alloc_complex

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

        return root_vectors

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
            lower = self._lower_bounds._root_vector
            upper = self._upper_bounds._root_vector
        else:
            lower = vector_class('nonlinear', 'output', self)
            upper = vector_class('nonlinear', 'output', self)

        return lower, upper

    def resetup(self, setup_mode='full'):
        """
        Public wrapper for _setup that reconfigures after an initial setup has been performed.

        Parameters
        ----------
        setup_mode : str
            Must be one of 'full', 'reconf', or 'update'.
        """
        self._setup(self.comm, setup_mode=setup_mode, mode=self._mode)
        self._final_setup(self.comm, self._outputs.__class__, setup_mode=setup_mode)

    def _setup(self, comm, setup_mode, mode):
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
        mode : str or None
            Derivative direction, either 'fwd', or 'rev', or None
        """
        # 1. Full setup that must be called in the root system.
        if setup_mode == 'full':
            initial = True
            recurse = True
            resize = False

            self.pathname = ''
            self.comm = comm
            self._relevant = None
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

        self._mode = mode

        # If we're only updating and not recursing, processors don't need to be redistributed
        if recurse:
            # Besides setting up the processors, this method also builds the model hierarchy.
            self._setup_procs(self.pathname, comm)

        # Recurse model from the bottom to the top for configuring.
        self._configure()

        # For updating variable and connection data, setup needs to be performed only
        # in the current system, by gathering data from immediate subsystems,
        # and no recursion is necessary.
        self._setup_var_data(recurse=recurse)
        self._setup_vec_names(mode, self._vec_names, self._vois)
        self._setup_global_connections(recurse=recurse)
        self._setup_relevance(mode, self._relevant)
        self._setup_vars(recurse=recurse)
        self._setup_var_index_ranges(self._get_initial_var_indices(initial), recurse=recurse)
        self._setup_var_index_maps(recurse=recurse)
        self._setup_var_sizes(recurse=recurse)
        self._setup_connections(recurse=recurse)

    def _setup_case_recording(self, recurse=True):
        myinputs = myoutputs = myresiduals = set()
        incl = self.recording_options['includes']
        excl = self.recording_options['excludes']

        if self.recording_options['record_inputs']:
            if self._inputs:
                myinputs = {n for n in self._inputs._names
                            if check_path(n, incl, excl)}
        if self.recording_options['record_outputs']:
            if self._outputs:
                myoutputs = {n for n in self._outputs._names
                             if check_path(n, incl, excl)}
            if self.recording_options['record_residuals']:
                myresiduals = myoutputs  # outputs and residuals have same names
        elif self.recording_options['record_residuals']:
            if self._residuals:
                myresiduals = {n for n in self._residuals._names
                               if check_path(n, incl, excl)}

        self._filtered_vars_to_record = {
            'i': myinputs,
            'o': myoutputs,
            'r': myresiduals
        }

        self._rec_mgr.startup(self)

        # Recursion
        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_case_recording(recurse)

    def _final_setup(self, comm, vector_class, setup_mode, force_alloc_complex=False):
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
        vector_class : type
            reference to an actual <Vector> class; not an instance.
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
        ext_num_vars, ext_num_vars_byset, ext_sizes, ext_sizes_byset = \
            self._get_initial_global(initial)
        self._setup_global(ext_num_vars, ext_num_vars_byset, ext_sizes, ext_sizes_byset)
        root_vectors = self._get_root_vectors(vector_class, initial,
                                              force_alloc_complex=force_alloc_complex)
        self._setup_vectors(root_vectors, resize=resize)
        self._setup_bounds(*self._get_bounds_root_vectors(vector_class, initial), resize=resize)

        # Transfers do not require recursion, but they have to be set up after the vector setup.
        self._setup_transfers(recurse=recurse)

        # Same situation with solvers, partials, and Jacobians.
        # If we're updating, we just need to re-run setup on these, but no recursion necessary.
        self._setup_solvers(recurse=recurse)
        self._setup_partials(recurse=recurse)
        self._setup_jacobians(recurse=recurse)

        self._setup_case_recording(recurse=recurse)

        # If full or reconf setup, reset this system's variables to initial values.
        if setup_mode in ('full', 'reconf'):
            self.set_initial_values()

        for sub in self.system_iter(recurse=True, include_self=True):
            sub._rec_mgr.record_metadata(sub)

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
        self._subsystems_proc_range = []

        # TODO: This version only runs for Components, because it is overriden in Group, so
        # maybe we should move this to Component?

        # Clear out old variable information so that we can call setup on the component.
        self._var_rel_names = {'input': [], 'output': []}
        self._var_rel2data_io = {}
        self._design_vars = OrderedDict()
        self._responses = OrderedDict()

        self._static_mode = False
        self._var_rel2data_io.update(self._static_var_rel2data_io)
        for type_ in ['input', 'output']:
            self._var_rel_names[type_].extend(self._static_var_rel_names[type_])
        self._design_vars.update(self._static_design_vars)
        self._responses.update(self._static_responses)
        self.setup()
        self._static_mode = True

    def _setup_vars(self, recurse=True):
        """
        Call setup in components and count variables, total and by var_set.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        self._num_var = {}
        self._num_var_byset = {}

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
        self._var_set2iset = set2iset

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
        self._var_allprocs_prom2abs_list = {'input': {}, 'output': {}}
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
        self._var_allprocs_abs2idx_byset = abs2idx_byset = {}
        abs2meta = self._var_allprocs_abs2meta

        for vec_name in self._lin_rel_vec_name_list:
            abs2idx[vec_name] = abs2idx_t = {}
            abs2idx_byset[vec_name] = abs2idx_byset_t = {}
            for type_ in ['input', 'output']:
                counter = defaultdict(int)

                for i, abs_name in enumerate(self._var_allprocs_relevant_names[vec_name][type_]):
                    abs2idx_t[abs_name] = i
                    set_name = abs2meta[abs_name]['var_set']
                    abs2idx_byset_t[abs_name] = counter[set_name]
                    counter[set_name] += 1

        abs2idx['nonlinear'] = abs2idx['linear']
        abs2idx_byset['nonlinear'] = abs2idx_byset['linear']

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
        self._var_sizes_byset = {}
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
                    raise RuntimeError("Global size of output '%s' (%s) does not agree "
                                       "with local shape %s" % (abs_name, global_size,
                                                                local_shape))
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
        self._conn_global_abs_in2out = {}

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
        def _filter_names(voi_dict):
            return set(voi for voi, data in iteritems(voi_dict)
                       if data['parallel_deriv_color'] is not None
                       or data['vectorize_derivs'])

        self._vois = vois
        if vec_names is None:  # should only occur at top level on full setup
            vec_names = ['nonlinear', 'linear']
            if mode == 'fwd':
                desvars = self.get_design_vars(recurse=True, get_sizes=False)
                vec_names.extend(sorted(_filter_names(desvars)))
                self._vois = vois = desvars
            else:  # rev
                responses = self.get_responses(recurse=True, get_sizes=False)
                vec_names.extend(sorted(_filter_names(responses)))
                self._vois = vois = responses

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
        self._relevant = relevant = {}
        relevant['nonlinear'] = {'@all': ({'input': ContainsAll(),
                                           'output': ContainsAll()},
                                          ContainsAll())}
        relevant['linear'] = relevant['nonlinear']
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
        Compute dict of all implicit and explicit connections owned by this system.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        self._conn_abs_in2out = {}

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
        self._ext_num_vars = ext_num_vars
        self._ext_num_vars_byset = ext_num_vars_byset
        self._ext_sizes = ext_sizes
        self._ext_sizes_byset = ext_sizes_byset

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
            msg = 'In order to activate complex step during reconfiguration, you need to set ' + \
                '"force_alloc_complex" to True during setup.'
            raise RuntimeError(msg)

        for vec_name in self._rel_vec_name_list:
            vector_class = root_vectors['output'][vec_name].__class__

            for kind in ['input', 'output', 'residual']:
                rootvec = root_vectors[kind][vec_name]
                vectors[kind][vec_name] = vector_class(
                    vec_name, kind, self, rootvec, resize=resize,
                    alloc_complex=alloc_complex and vec_name == 'nonlinear', ncol=rootvec._ncol)

        self._inputs = vectors['input']['nonlinear']
        self._outputs = vectors['output']['nonlinear']
        self._residuals = vectors['residual']['nonlinear']

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

        abs2meta = self._var_abs2meta
        for abs_name in self._var_abs_names['output']:
            meta = abs2meta[abs_name]
            shape = meta['shape']
            ref0 = meta['ref0']
            ref = meta['ref']
            var_lower = meta['lower']
            var_upper = meta['upper']

            if not np.isscalar(ref0):
                ref0 = ref0.reshape(shape)
            if not np.isscalar(ref):
                ref = ref.reshape(shape)

            if var_lower is None:
                lower._views[abs_name][:] = -np.inf
            else:
                lower._views[abs_name][:] = (var_lower - ref0) / (ref - ref0)

            if var_upper is None:
                upper._views[abs_name][:] = np.inf
            else:
                upper._views[abs_name][:] = (var_upper - ref0) / (ref - ref0)

        for subsys in self._subsystems_myproc:
            subsys._setup_bounds(root_lower, root_upper)

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

        abs2meta_in = self._var_abs2meta
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

        if self._has_input_scaling:
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

                        # TODO: if either ref or ref0 are not scalar and the output is
                        # distributed, we need to do a scatter
                        # to obtain the values needed due to global src_indices
                        if meta_out['distributed']:
                            raise RuntimeError("vector scalers with distrib vars "
                                               "not supported yet.")

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

    def _setup_transfers(self, recurse=True):
        """
        Compute all transfers that are owned by this system.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        self._transfers = {}

    def _setup_solvers(self, recurse=True):
        """
        Perform setup in all solvers.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        if self._nonlinear_solver is not None:
            self._nonlinear_solver._setup_solvers(self, 0)
        if self._linear_solver is not None:
            self._linear_solver._setup_solvers(self, 0)

        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_solvers(recurse=recurse)

    def _setup_partials(self, recurse=True):
        """
        Call setup_partials in components.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        self._subjacs_info = {}

        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_partials(recurse)

    def _setup_jacobians(self, jacobian=None, recurse=True):
        """
        Set and populate jacobians down through the system tree.

        Parameters
        ----------
        jacobian : <AssembledJacobian> or None
            The global jacobian to populate for this system.
        recurse : bool
            Whether to call this method in subsystems.
        """
        self._jacobian_changed = False
        self._views_assembled_jac = False

        if jacobian is not None:
            if self._nonlinear_solver is not None and self._nonlinear_solver.supports['gradients']:
                self._views_assembled_jac = True

        if self._owns_assembled_jac:

            # if we have an assembled jac at this level and an assembled jac above us, then
            # our jacs (and any of our children's assembled jacs) will share their internal
            # subjac dicts.  Each will maintain its own internal Matrix objects though.
            if jacobian is not None:
                jacobian._subjacs.update(self._jacobian._subjacs)
                jacobian._subjacs_info.update(self._jacobian._subjacs_info)
                self._jacobian._subjacs = jacobian._subjacs
                self._jacobian._subjacs_info = jacobian._subjacs_info
                self._jacobian._keymap = jacobian._keymap

            # At present, we don't support a AssembledJacobian in a group
            # if any subcomponents are matrix-free.
            for subsys in self.system_iter():

                try:
                    if subsys.matrix_free:
                        msg = "AssembledJacobian not supported if any subcomponent is matrix-free."
                        raise RuntimeError(msg)

                # Groups don't have `matrix_free`
                # Note, we could put this attribute on Group, but this would be True for a
                # default Group, and thus we would need an isinstance on Component, which is the
                # reason for the try block anyway.
                except AttributeError:
                    continue

            jacobian = self._jacobian

        elif jacobian is not None:
            self._jacobian = jacobian

        self._set_partials_meta()

        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_jacobians(jacobian, recurse)

        if self._owns_assembled_jac:
            self._jacobian._system = self
            self._jacobian._initialize()

            for s in self.system_iter(recurse=True):
                if s._jacobian is self._jacobian and s._views_assembled_jac:
                    self._jacobian._init_view(s)

    def set_initial_values(self):
        """
        Set all input and output variables to their declared initial values.
        """
        abs2meta = self._var_abs2meta
        for abs_name in self._var_abs_names['input']:
            self._inputs._views[abs_name][:] = abs2meta[abs_name]['value']

        for abs_name in self._var_abs_names['output']:
            self._outputs._views[abs_name][:] = abs2meta[abs_name]['value']

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
                                   (self.pathname, call, sorted(not_found)))

        maps = {'input': {}, 'output': {}}

        if self._var_promotes['input'] or self._var_promotes['output']:
            if self._var_promotes['any']:
                raise RuntimeError("%s: 'promotes' cannot be used at the same time as "
                                   "'promotes_inputs' or 'promotes_outputs'." % self.pathname)
            resolve(self._var_promotes['input'], ('input',), maps, prom_names)
            resolve(self._var_promotes['output'], ('output',), maps, prom_names)
        else:
            resolve(self._var_promotes['any'], ('input', 'output',), maps, prom_names)

        return maps

    def _get_scope(self, excl_sub=None):
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

    @property
    def jacobian(self):
        """
        Get the Jacobian object assigned to this system (or None if unassigned).
        """
        return self._jacobian

    @jacobian.setter
    def jacobian(self, jacobian):
        """
        Set the Jacobian.
        """
        self._owns_assembled_jac = isinstance(jacobian, AssembledJacobian)
        self._jacobian = jacobian
        self._jacobian_changed = True

    @contextmanager
    def _unscaled_context(self, outputs=[], residuals=[]):
        """
        Context manager for units and scaling for vectors and Jacobians.

        Temporarily puts vectors in a physical and unscaled state, because
        internally, vectors are nominally in a dimensionless and scaled state.
        The same applies (optionally) for Jacobians.

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

        for vec in outputs:

            # Process any complex views if under complex step.
            if vec._vector_info._under_complex_step:
                vec._remove_complex_views()

            if self._has_output_scaling:
                vec.scale('norm')

        for vec in residuals:

            # Process any complex views if under complex step.
            if vec._vector_info._under_complex_step:
                vec._remove_complex_views()

            if self._has_resid_scaling:
                vec.scale('norm')

    @contextmanager
    def _unscaled_context_all(self):
        """
        Context manager that temporarily puts all vectors and Jacobians in an unscaled state.
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
        Context manager that temporarily puts all vectors and Jacobians in a scaled state.
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
            raise RuntimeError("Cannot get vectors because setup has not yet been called.")

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
            raise RuntimeError("Cannot get vectors because setup has not yet been called.")

        if vec_name not in self._vectors['input']:
            raise ValueError("There is no linear vector named %s" % vec_name)

        return (self._vectors['input'][vec_name],
                self._vectors['output'][vec_name],
                self._vectors['residual'][vec_name])

    @contextmanager
    def jacobian_context(self):
        """
        Context manager that yields the Jacobian assigned to this system in this system's context.

        Yields
        ------
        <Jacobian>
            The current system's jacobian with its _system set to self.
        """
        if self._jacobian_changed:
            raise RuntimeError("%s: jacobian has changed and setup was not "
                               "called." % self.pathname)
        oldsys = self._jacobian._system
        self._jacobian._system = self
        yield self._jacobian
        self._jacobian._system = oldsys

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
        if self.linear_solver is not None and type_ != 'NL':
            self.linear_solver._set_solver_print(level=level, type_=type_)
        if self.nonlinear_solver is not None and type_ != 'LN':
            self.nonlinear_solver._set_solver_print(level=level, type_=type_)

        for subsys in self._subsystems_allprocs:

            current_depth = subsys.pathname.count('.')
            if current_depth >= depth:
                continue

            subsys._set_solver_print(level=level, depth=depth - current_depth, type_=type_)

            if subsys.linear_solver is not None and type_ != 'NL':
                subsys.linear_solver._set_solver_print(level=level, type_=type_)
            if subsys.nonlinear_solver is not None and type_ != 'LN':
                subsys.nonlinear_solver._set_solver_print(level=level, type_=type_)

    def _set_partials_meta(self):
        """
        Set subjacobian info into our jacobian.

        Overridden in <Component>.
        """
        pass

    def system_iter(self, include_self=False, recurse=True,
                    typ=None):
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

    def add_design_var(self, name, lower=None, upper=None, ref=None,
                       ref0=None, indices=None, adder=None, scaler=None,
                       parallel_deriv_color=None, vectorize_derivs=False,
                       simul_coloring=None, cache_linear_solution=False):
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
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        parallel_deriv_color : string
            If specified, this design var will be grouped for parallel derivative
            calculations with other variables sharing the same parallel_deriv_color.
        vectorize_derivs : bool
            If True, vectorize derivative calculations.
        simul_coloring : ndarray or list of int
            An array or list of integer color values.  Must match the size of the
            design variable.
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
            msg = "Design Variable '{}' already exists."
            raise RuntimeError(msg.format(name))

        # Name must be a string
        if not isinstance(name, string_types):
            raise TypeError('The name argument should be a string, got {0}'.format(name))

        # Convert ref/ref0 to ndarray/float as necessary
        ref = format_as_float_or_array('ref', ref, val_if_none=None, flatten=True)
        ref0 = format_as_float_or_array('ref0', ref0, val_if_none=None, flatten=True)

        # determine adder and scaler based on args
        adder, scaler = determine_adder_scaler(ref0, ref, adder, scaler)

        # Convert lower to ndarray/float as necessary
        lower = format_as_float_or_array('lower', lower, val_if_none=-sys.float_info.max,
                                         flatten=True)

        # Convert upper to ndarray/float as necessary
        upper = format_as_float_or_array('upper', upper, val_if_none=sys.float_info.max,
                                         flatten=True)

        # Apply scaler/adder to lower and upper
        lower = (lower + adder) * scaler
        upper = (upper + adder) * scaler

        if self._static_mode:
            design_vars = self._static_design_vars
        else:
            design_vars = self._design_vars

        design_vars[name] = dvs = OrderedDict()

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
                raise ValueError("If specified, indices must be a sequence of integers.")

            indices = np.atleast_1d(indices)
            dvs['size'] = len(indices)
        dvs['indices'] = indices
        dvs['parallel_deriv_color'] = parallel_deriv_color
        dvs['vectorize_derivs'] = vectorize_derivs
        dvs['simul_deriv_color'] = simul_coloring

    def add_response(self, name, type_, lower=None, upper=None, equals=None,
                     ref=None, ref0=None, indices=None, index=None,
                     adder=None, scaler=None, linear=False, parallel_deriv_color=None,
                     vectorize_derivs=False, simul_coloring=None,
                     simul_map=None, cache_linear_solution=False):
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
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        linear : bool
            Set to True if constraint is linear. Default is False.
        parallel_deriv_color : string
            If specified, this design var will be grouped for parallel derivative
            calculations with other variables sharing the same parallel_deriv_color.
        vectorize_derivs : bool
            If True, vectorize derivative calculations.
        simul_coloring : ndarray or list of int
            An array or list of integer color values.  Must match the size of the
            response variable.
        simul_map : dict
            Mapping of this response to each design variable where simultaneous derivs will
            be used.  Each design variable entry is another dict keyed on color, and the values
            in the color dict are tuples of the form (resp_idxs, color_idxs).
        cache_linear_solution : bool
            If True, store the linear solution vectors for this variable so they can
            be used to start the next linear solution with an initial guess equal to the
            solution from the previous linear solve.
        """
        # Name must be a string
        if not isinstance(name, string_types):
            raise TypeError('The name argument should be a string, '
                            'got {0}'.format(name))

        # Type must be a string and one of 'con' or 'obj'
        if not isinstance(type_, string_types):
            raise TypeError('The type argument should be a string')
        elif type_ not in ('con', 'obj'):
            raise ValueError('The type must be one of \'con\' or \'obj\': '
                             'Got \'{0}\' instead'.format(name))

        if name in self._responses or name in self._static_responses:
            typemap = {'con': 'Constraint', 'obj': 'Objective'}
            msg = '{0} \'{1}\' already exists.'.format(typemap[type_], name)
            raise RuntimeError(msg.format(name))

        # Convert ref/ref0 to ndarray/float as necessary
        ref = format_as_float_or_array('ref', ref, val_if_none=None, flatten=True)
        ref0 = format_as_float_or_array('ref0', ref0, val_if_none=None, flatten=True)

        # determine adder and scaler based on args
        adder, scaler = determine_adder_scaler(ref0, ref, adder, scaler)

        # A constraint cannot be an equality and inequality constraint
        if equals is not None and (lower is not None or upper is not None):
            msg = "Constraint '{}' cannot be both equality and inequality."
            raise ValueError(msg.format(name))

        # If given, indices must be a sequence
        if (indices is not None and not (
                isinstance(indices, Iterable) and all([isinstance(i, Integral) for i in indices]))):
            raise ValueError("If specified, indices must be a sequence of integers.")

        if self._static_mode:
            responses = self._static_responses
        else:
            responses = self._responses

        resp = OrderedDict()

        if type_ == 'con':
            # Convert lower to ndarray/float as necessary
            lower = format_as_float_or_array('lower', lower, val_if_none=-sys.float_info.max,
                                             flatten=True)

            # Convert upper to ndarray/float as necessary
            upper = format_as_float_or_array('upper', upper, val_if_none=sys.float_info.max,
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
                index = np.array([index], dtype=int)
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

        resp['name'] = name
        resp['ref'] = ref
        resp['ref0'] = ref0
        resp['type'] = type_
        resp['cache_linear_solution'] = cache_linear_solution

        resp['parallel_deriv_color'] = parallel_deriv_color
        resp['vectorize_derivs'] = vectorize_derivs
        resp['simul_deriv_color'] = simul_coloring
        resp['simul_map'] = simul_map

        responses[name] = resp

    def add_constraint(self, name, lower=None, upper=None, equals=None,
                       ref=None, ref0=None, adder=None, scaler=None,
                       indices=None, linear=False, parallel_deriv_color=None,
                       vectorize_derivs=False, simul_coloring=None, simul_map=None,
                       cache_linear_solution=False):
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
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
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
        simul_coloring : ndarray or list of int
            An array or list of integer color values.  Must match the size of the
            constraint variable.
        simul_map : dict
            Mapping of this response to each design variable where simultaneous derivs will
            be used.  Each design variable entry is another dict keyed on color, and the values
            in the color dict are tuples of the form (resp_idxs, color_idxs).
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
                          simul_coloring=simul_coloring, simul_map=simul_map,
                          cache_linear_solution=cache_linear_solution)

    def add_objective(self, name, ref=None, ref0=None, index=None,
                      adder=None, scaler=None, parallel_deriv_color=None,
                      vectorize_derivs=False, simul_coloring=None, simul_map=None,
                      cache_linear_solution=False):
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
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        parallel_deriv_color : string
            If specified, this design var will be grouped for parallel derivative
            calculations with other variables sharing the same parallel_deriv_color.
        vectorize_derivs : bool
            If True, vectorize derivative calculations.
        simul_coloring : ndarray or list of int
            An array or list of integer color values.  Must match the size of the
            objective variable.
        simul_map : dict
            Mapping of this response to each design variable where simultaneous derivs will
            be used.  Each design variable entry is another dict keyed on color, and the values
            in the color dict are tuples of the form (resp_idxs, color_idxs).
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
            raise TypeError('If specified, index must be an int.')
        self.add_response(name, type_='obj', scaler=scaler, adder=adder,
                          ref=ref, ref0=ref0, index=index,
                          parallel_deriv_color=parallel_deriv_color,
                          vectorize_derivs=vectorize_derivs,
                          simul_coloring=simul_coloring, simul_map=simul_map,
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
            If True, compute the size of each response.

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
            msg = "Output not found for design variable {0} in system '{1}'."
            raise RuntimeError(msg.format(str(err), self.pathname))

        if get_sizes:
            # Size them all
            sizes = self._var_sizes['nonlinear']['output']
            abs2idx = self._var_allprocs_abs2idx['nonlinear']
            for name in out:
                if 'size' not in out[name]:
                    out[name]['size'] = sizes[self._owning_rank[name], abs2idx[name]]

        if recurse:
            for subsys in self._subsystems_myproc:
                out.update(subsys.get_design_vars(recurse=recurse, get_sizes=get_sizes))

            if self.comm.size > 1 and self._subsystems_allprocs:
                allouts = self.comm.allgather(out)
                out = OrderedDict()
                for rank, all_out in enumerate(allouts):
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
            msg = "Output not found for response {0} in system '{1}'."
            raise RuntimeError(msg.format(str(err), self.pathname))

        if get_sizes:
            # Size them all
            sizes = self._var_sizes['nonlinear']['output']
            abs2idx = self._var_allprocs_abs2idx['nonlinear']
            for name in out:
                if 'size' not in out[name]:
                    out[name]['size'] = sizes[self._owning_rank[name], abs2idx[name]]

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
                    units=False,
                    hierarchical=True,
                    print_arrays=False,
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
        units : bool, optional
            When True, display/return units. Default is False.
        hierarchical : bool, optional
            When True, human readable output shows variables in hierarchical format.
        print_arrays : bool, optional
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format is affected
            by the values set with numpy.set_printoptions
            Default is False.
        out_stream : file-like object
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        list
            list of input names and other optional information about those inputs
        """
        if self._inputs is None:
            raise RuntimeError("Unable to list inputs until model has been run.")

        meta = self._var_abs2meta
        inputs = []

        for name, val in iteritems(self._inputs._views):  # This is only over the locals
            outs = {}
            if values:
                outs['value'] = val
            if units:
                outs['units'] = meta[name]['units']
            inputs.append((name, outs))

        if out_stream is _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        if out_stream:
            self._write_outputs('input', None, inputs, hierarchical, print_arrays, out_stream)

        return inputs

    def list_outputs(self,
                     explicit=True, implicit=True,
                     values=True,
                     residuals=False,
                     residuals_tol=None,
                     units=False,
                     shape=False,
                     bounds=False,
                     scaling=False,
                     hierarchical=True,
                     print_arrays=False,
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
        out_stream : file-like
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        list
            list of output names and other optional information about those outputs
        """
        if self._outputs is None:
            raise RuntimeError("Unable to list outputs until model has been run.")

        # Only gathering up values and metadata from this proc, if MPI
        meta = self._var_abs2meta  # This only includes metadata for this process.
        states = self._list_states()

        # Go though the hierarchy. Printing Systems
        # If the System owns an output directly, show its output
        expl_outputs = []
        impl_outputs = []
        for name, val in iteritems(self._outputs._views):
            if residuals_tol and np.linalg.norm(self._residuals._views[name]) < residuals_tol:
                continue
            outs = {}
            if values:
                outs['value'] = val
            if residuals:
                outs['resids'] = self._residuals._views[name]
            if units:
                outs['units'] = meta[name]['units']
            if shape:
                outs['shape'] = val.shape
            if bounds:
                outs['lower'] = meta[name]['lower']
                outs['upper'] = meta[name]['upper']
            if scaling:
                outs['ref'] = meta[name]['ref']
                outs['ref0'] = meta[name]['ref0']
                outs['res_ref'] = meta[name]['res_ref']
            if name in states:
                impl_outputs.append((name, outs))
            else:
                expl_outputs.append((name, outs))

        if out_stream is _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        if out_stream:
            if explicit:
                self._write_outputs('output', 'Explicit', expl_outputs, hierarchical, print_arrays,
                                    out_stream)
            if implicit:
                self._write_outputs('output', 'Implicit', impl_outputs, hierarchical, print_arrays,
                                    out_stream)

        if explicit and implicit:
            return expl_outputs + impl_outputs
        elif explicit:
            return expl_outputs
        elif implicit:
            return impl_outputs
        else:
            raise RuntimeError('You have excluded both Explicit and Implicit components.')

    def _write_outputs(self, in_or_out, comp_type, outputs, hierarchical, print_arrays,
                       out_stream):
        """
        Write table of variable names, values, residuals, and metadata to out_stream.

        The output values could actually represent input variables.
        In this context, outputs refers to the data that is being logged to an output stream.

        Parameters
        ----------
        in_or_out : str, 'input' or 'output'
            indicates whether the values passed in are from inputs or output variables.
        comp_type : str, 'Explicit' or 'Implicit'
            the type of component with the output values.
        outputs : list
            list of (name, dict of vals and metadata) tuples.
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

        # Only local metadata but the most complete
        meta = self._var_abs2meta

        # Make a dict of outputs. Makes it easier to work with in this method
        dict_of_outputs = OrderedDict()
        for name, vals in outputs:
            dict_of_outputs[name] = vals

        # If parallel, gather up the outputs. All procs must call this
        if MPI:
            # returns a list, one per proc
            all_dict_of_outputs = self.comm.gather(dict_of_outputs, root=0)

        if MPI and MPI.COMM_WORLD.rank > 0:  # If MPI, only the root process should print
            return

        # If MPI, and on rank 0, need to gather up all the variables
        if MPI:  # rest of this only done on rank 0
            dict_of_outputs = all_dict_of_outputs[0]  # start with rank 0
            for proc_outputs in all_dict_of_outputs[1:]:  # In rank order go thru rest of the procs
                for name, vals in iteritems(proc_outputs):
                    if name not in dict_of_outputs:  # If not in the merged dict, add it
                        dict_of_outputs[name] = proc_outputs[name]
                    else:  # If in there already, only need to deal with it if it is a
                        # distributed array.
                        # Checking to see if  distributed depends on if it is an input or output
                        if in_or_out == 'input':
                            is_distributed = meta[name]['src_indices'] is not None
                        else:
                            is_distributed = meta[name]['distributed']
                        if is_distributed:
                            # TODO no support for > 1D arrays
                            #   meta.src_indices has the info we need to piece together arrays
                            if 'value' in dict_of_outputs[name]:
                                dict_of_outputs[name]['value'] = \
                                    np.append(dict_of_outputs[name]['value'],
                                              proc_outputs[name]['value'])
                            if 'shape' in dict_of_outputs[name]:
                                # TODO might want to use allprocs_abs2meta_out[name]['global_shape']
                                dict_of_outputs[name]['shape'] = \
                                    dict_of_outputs[name]['value'].shape
                            if 'resids' in dict_of_outputs[name]:
                                dict_of_outputs[name]['resids'] = \
                                    np.append(dict_of_outputs[name]['resids'],
                                              proc_outputs[name]['resids'])

        count = len(dict_of_outputs)

        # Write header
        pathname = self.pathname if self.pathname else 'model'
        header_name = 'Input' if in_or_out == 'input' else 'Output'
        if in_or_out == 'input':
            header = "%d %s(s) in '%s'" % (count, header_name, pathname)
        else:
            header = "%d %s %s(s) in '%s'" % (count, comp_type, header_name, pathname)
        out_stream.write(header + '\n')
        out_stream.write('-' * len(header) + '\n' + '\n')

        if not count:
            return

        # Need an ordered list of possible output values for the two cases: inputs and outputs
        #  so that we do the column output in the correct order
        if in_or_out == 'input':
            out_types = ('value', 'units',)
        else:
            out_types = ('value', 'resids', 'units', 'shape', 'lower', 'upper', 'ref',
                         'ref0', 'res_ref')
        # Figure out which columns will be displayed
        # Look at any one of the outputs, they should all be the same
        outputs = dict_of_outputs[list(dict_of_outputs)[0]]
        column_names = []
        for out_type in out_types:
            if out_type in outputs:
                column_names.append(out_type)

        top_level_system_name = 'top'

        # Find with width of the first column in the table
        #    Need to look through all the possible varnames to find the max width
        max_varname_len = max(len(top_level_system_name), len('varname'))
        if hierarchical:
            for name, outs in iteritems(dict_of_outputs):
                for i, name_part in enumerate(name.split('.')):
                    total_len = (i + 1) * self._indent_inc + len(name_part)
                    max_varname_len = max(max_varname_len, total_len)
        else:
            for name, outs in iteritems(dict_of_outputs):
                max_varname_len = max(max_varname_len, len(name))

        # Determine the column widths of the data fields by finding the max width for all rows
        for column_name in column_names:
            self._column_widths[column_name] = len(column_name)  # has to be able to display name!
        for name in self._var_allprocs_abs_names[in_or_out]:
            if name in dict_of_outputs:
                for column_name in column_names:
                    if isinstance(dict_of_outputs[name][column_name], np.ndarray) and \
                            dict_of_outputs[name][column_name].size > 1:
                        out = '|{}|'.format(str(np.linalg.norm(dict_of_outputs[name][column_name])))
                    else:
                        out = str(dict_of_outputs[name][column_name])
                    self._column_widths[column_name] = max(self._column_widths[column_name],
                                                           len(str(out)))

        # Write out the column headers
        column_header = '{:{align}{width}}'.format('varname', align=self._align,
                                                   width=max_varname_len)
        column_dashes = max_varname_len * '-'
        for column_name in column_names:
            column_header += self._column_spacing * ' '
            column_header += '{:{align}{width}}'.format(column_name, align=self._align,
                                                        width=self._column_widths[column_name])
            column_dashes += self._column_spacing * ' ' + self._column_widths[column_name] * '-'
        out_stream.write(column_header + '\n')
        out_stream.write(column_dashes + '\n')

        # Write out the variable names and optional values and metadata
        if hierarchical:
            out_stream.write(top_level_system_name + '\n')

            cur_sys_names = []
            # _var_allprocs_abs_names has all the vars across all procs in execution order
            #   But not all the values need to be written since, at least for output vars,
            #      the output var lists are divided into explicit and implicit
            for varname in self._var_allprocs_abs_names[in_or_out]:
                if varname not in dict_of_outputs:
                    continue

                # For hierarchical, need to display system levels in the rows above the
                #   actual row containing the var name and values. Want to make use
                #   of the hierarchies that have been written about this.
                existing_sys_names = []
                varname_sys_names = varname.split('.')[:-1]
                for i, sys_name in enumerate(varname_sys_names):
                    if varname_sys_names[:i + 1] != cur_sys_names[:i + 1]:
                        break
                    else:
                        existing_sys_names = cur_sys_names[:i + 1]

                # What parts of the hierarchy for this varname need to be written that
                #   were not already written above this
                remaining_sys_path_parts = varname_sys_names[len(existing_sys_names):]

                # Write the Systems in the var name path
                indent = len(existing_sys_names) * self._indent_inc
                for i, sys_name in enumerate(remaining_sys_path_parts):
                    indent += self._indent_inc
                    out_stream.write(indent * ' ' + sys_name + '\n')
                cur_sys_names = varname_sys_names

                indent += self._indent_inc
                row = '{:{align}{width}}'.format(indent * ' ' + varname.split('.')[-1],
                                                 align=self._align, width=max_varname_len)
                self._write_outputs_rows(out_stream, row, column_names, dict_of_outputs[varname],
                                         print_arrays)
        else:
            for name in self._var_allprocs_abs_names[in_or_out]:
                if name in dict_of_outputs:
                    row = '{:{align}{width}}'.format(name, align=self._align, width=max_varname_len)
                    self._write_outputs_rows(out_stream, row, column_names, dict_of_outputs[name],
                                             print_arrays)
        out_stream.write(2 * '\n')

    def _write_outputs_rows(self, out_stream, row, column_names, dict_of_outputs, print_arrays):
        """
        For one variable, write name, values, residuals, and metadata to out_stream.

        Parameters
        ----------
        out_stream : file-like object
            Where to send human readable output.
            Set to None to suppress.
        row : str
            The string containing the contents of the beginning of this row output.
            Contains the name of the System or varname, possibley indented to show hierarchy.

        column_names : list of str
            Indicates which columns will be written in this row.

        dict_of_outputs : dict
            Contains the values to be written in this row. Keys are columns names.

        print_arrays : bool
            When False, in the columnar display, just display norm of any ndarrays with size > 1.
            The norm is surrounded by vertical bars to indicate that it is a norm.
            When True, also display full values of the ndarray below the row. Format  is affected
            by the values set with numpy.set_printoptions
            Default is False.

        """
        if out_stream is None:
            return
        left_column_width = len(row)
        have_array_values = []  # keep track of which values are arrays
        for column_name in column_names:
            row += self._column_spacing * ' '
            if isinstance(dict_of_outputs[column_name], np.ndarray) and \
                    dict_of_outputs[column_name].size > 1:
                have_array_values.append(column_name)
                out = '|{}|'.format(str(np.linalg.norm(dict_of_outputs[column_name])))
            else:
                out = str(dict_of_outputs[column_name])
            row += '{:{align}{width}}'.format(out, align=self._align,
                                              width=self._column_widths[column_name])
        out_stream.write(row + '\n')
        if print_arrays:
            for column_name in have_array_values:
                out_stream.write("{}  {}:\n".format(left_column_width * ' ', column_name))
                out_str = str(dict_of_outputs[column_name])
                indented_lines = [(left_column_width + self._indent_inc) * ' ' +
                                  s for s in out_str.splitlines()]
                out_stream.write('\n'.join(indented_lines) + '\n')

    def run_solve_nonlinear(self):
        """
        Compute outputs.

        This calls _solve_nonlinear, but with the model assumed to be in an unscaled state.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        with self._scaled_context_all():
            result = self._solve_nonlinear()

        return result

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
            self._apply_linear(vec_names, ContainsAll(), mode, scope_out, scope_in)

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

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        with self._scaled_context_all():
            result = self._solve_linear(vec_names, mode, ContainsAll())

        return result

    def run_linearize(self, do_nl=True, do_ln=True):
        """
        Compute jacobian / factorization.

        This calls _linearize, but with the model assumed to be in an unscaled state.

        Parameters
        ----------
        do_nl : boolean
            Flag indicating if the nonlinear solver should be linearized.
        do_ln : boolean
            Flag indicating if the linear solver should be linearized.

        """
        with self._scaled_context_all():
            self._linearize(do_nl, do_ln)

    def _apply_nonlinear(self):
        """
        Compute residuals. The model is assumed to be in a scaled state.
        """
        pass

    def _solve_nonlinear(self):
        """
        Compute outputs. The model is assumed to be in a scaled state.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            Relative error.
        float
            Absolute error.
        """
        # Reconfigure if needed.
        self._check_reconf()

        return False, 0., 0.

    def check_config(self, logger):
        """
        Perform optional error checks.

        Parameters
        ----------
        logger : object
            The object that manages logging output.
        """
        pass

    def _apply_linear(self, vec_names, rel_systems, mode, var_inds=None):
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
        var_inds : [int, int, int, int] or None
            ranges of variable IDs involved in this matrix-vector product.
            The ordering is [lb1, ub1, lb2, ub2].
        """
        raise NotImplementedError("_apply_linear has not been overridden")

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
        pass

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
        pass

    def _list_states(self):
        """
        Return list of all states at and below this system.

        Returns
        -------
        list
            List of all states.
        """
        states = []
        for subsys in self._subsystems_myproc:
            states.extend(subsys._list_states())

        return states

    def add_recorder(self, recorder, recurse=False):
        """
        Add a recorder to the driver.

        Parameters
        ----------
        recorder : <BaseRecorder>
           A recorder instance.
        recurse : boolean
            Flag indicating if the recorder should be added to all the subsystems.
        """
        if MPI:
            raise RuntimeError(
                "Recording of Systems when running parallel code is not supported yet")
        for s in self.system_iter(include_self=True, recurse=recurse):
            s._rec_mgr.append(recorder)

    def record_iteration(self):
        """
        Record an iteration of the current System.
        """
        if self._rec_mgr._recorders:
            metadata = create_local_meta(self.pathname)

            # Get the data to record
            stack_top = recording_iteration.stack[-1][0]
            method = stack_top.split('.')[-1]

            if method not in ['_apply_linear', '_apply_nonlinear', '_solve_linear',
                              '_solve_nonlinear']:
                raise ValueError("method must be one of: '_apply_linear, "
                                 "_apply_nonlinear, _solve_linear, _solve_nonlinear'")

            if 'nonlinear' in method:
                inputs, outputs, residuals = self.get_nonlinear_vectors()
            else:
                inputs, outputs, residuals = self.get_linear_vectors()

            data = {}
            if self.recording_options['record_inputs'] and inputs._names:
                data['i'] = {}
                if 'i' in self._filtered_vars_to_record:
                    # use filtered inputs
                    for inp in self._filtered_vars_to_record['i']:
                        if inp in inputs._names:
                            data['i'][inp] = inputs._views[inp]
                else:
                    # use all the inputs
                    data['i'] = inputs._names
            else:
                data['i'] = None

            if self.recording_options['record_outputs'] and outputs._names:
                data['o'] = {}

                if 'o' in self._filtered_vars_to_record:
                    # use outputs from filtered list.
                    for out in self._filtered_vars_to_record['o']:
                        if out in outputs._names:
                            data['o'][out] = outputs._views[out]
                else:
                    # use all the outputs
                    data['o'] = outputs._names
            else:
                data['o'] = None

            if self.recording_options['record_residuals'] and residuals._names:
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
