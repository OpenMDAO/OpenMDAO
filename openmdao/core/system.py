"""Define the base System class."""
from __future__ import division

from contextlib import contextmanager
from collections import OrderedDict, Iterable
from fnmatch import fnmatchcase
import sys
from itertools import product

from six import iteritems, string_types
from six.moves import range

import numpy as np

from openmdao.jacobians.dictionary_jacobian import DictionaryJacobian
from openmdao.jacobians.assembled_jacobian import AssembledJacobian, DenseJacobian
from openmdao.proc_allocators.default_allocator import DefaultAllocator

from openmdao.utils.general_utils import \
    determine_adder_scaler, format_as_float_or_array, warn_deprecation
from openmdao.utils.mpi import MPI
from openmdao.utils.options_dictionary import OptionsDictionary
from openmdao.utils.units import convert_units
from openmdao.utils.array_utils import convert_neg


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
    #
    _mpi_proc_allocator : <ProcAllocator>
        Object that distributes procs among subsystems.
    #
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
    _subsystems_var_range_byset : {'input': list of dict, 'output': list of dict}
        Same as above, but by var_set name.
    #
    _num_var : {'input': int, 'output': int}
        Number of allprocs variables owned by this system.
    _num_var_byset : {'input': dict of int, 'output': dict of int}
        Same as above, but by var_set name.
    _var_set2iset : {'input': dict, 'output': dict}
        Dictionary mapping the var_set name to the var_set index.
    #
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
    _var_allprocs_abs2meta : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to metadata dictionaries for allprocs variables.
        The keys are
        ('units', 'shape', 'var_set') for inputs and
        ('units', 'shape', 'var_set', 'ref', 'ref0') for outputs.
    _var_abs2meta : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to metadata dictionaries for myproc variables.
    #
    _var_allprocs_abs2idx : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to their indices among this system's allprocs variables.
        Therefore, the indices range from 0 to the total number of this system's variables.
    _var_allprocs_abs2idx_byset : {'input': dict of dict, 'output': dict of dict}
        Same as above, but by var_set name.
    #
    _var_sizes : {'input': ndarray, 'output': ndarray}
        Array of local sizes of this system's allprocs variables.
        The array has size nproc x num_var where nproc is the number of processors
        owned by this system and num_var is the number of allprocs variables.
    _var_sizes_byset : {'input': dict of ndarray, 'output': dict of ndarray}
        Same as above, but by var_set name.
    #
    _manual_connections : dict
        Dictionary of input_name: (output_name, src_indices) connections.
    _conn_global_abs_in2out : {'abs_in': 'abs_out'}
        Dictionary containing all explicit & implicit connections owned by this system
        or any descendant system. The data is the same across all processors.
    _conn_parents_abs_in2out : {'abs_in': 'abs_out'}
        Dictionary containing all explicit & implicit connections from systems above.
    _conn_abs_in2out : {'abs_in': 'abs_out'}
        Dictionary containing all explicit & implicit connections owned
        by this system only. The data is the same across all processors.
    #
    _ext_num_vars : {'input': (int, int), 'output': (int, int)}
        Total number of allprocs variables in system before/after this one.
    _ext_num_vars_byset : {'input': dict of (int, int), 'output': dict of (int, int)}
        Same as above, but by var_set name.
    _ext_sizes : {'input': (int, int), 'output': (int, int)}
        Total size of allprocs variables in system before/after this one.
    _ext_sizes_byset : {'input': dict of (int, int), 'output': dict of (int, int)}
        Same as above, but by var_set name.
    #
    _vec_names : [str, ...]
        List of names of the vectors (i.e., the right-hand sides).
    _vectors : {'input': dict, 'output': dict, 'residual': dict}
        Dictionaries of vectors keyed by vec_name.
    _excluded_vars_out : dict of set
        Set of output variable absolute names not relevant for each vec_name.
    _excluded_vars_in : dict of set
        Set of input variable absolute names not relevant for each vec_name.
    #
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
    #
    _lower_bounds : <Vector>
        Vector of lower bounds, scaled and dimensionless.
    _upper_bounds : <Vector>
        Vector of upper bounds, scaled and dimensionless.
    #
    _scaling_vecs : dict of dict of Vectors
        First key is indicates vector type and coefficient, second key is vec_name.
    #
    _nonlinear_solver : <NonlinearSolver>
        Nonlinear solver to be used for solve_nonlinear.
    _linear_solver : <LinearSolver>
        Linear solver to be used for solve_linear; not the Newton system.
    #
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
        Stores approximation metadata (e.g., step_size) from calls to approx_total_derivs
    _owns_approx_wrt : set or None
        Overrides aproximation inputs.
    _owns_approx_of : set or None
        Overrides aproximation outputs.
    _subjacs_info : OrderedDict of dict
        Sub-jacobian metadata for each (output, input) pair added using
        declare_partials. Members of each pair may be glob patterns.
    #
    _design_vars : dict of dict
        dict of all driver design vars added to the system.
    _responses : dict of dict
        dict of all driver responses added to the system.
    #
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
    #
    _reconfigured : bool
        If True, this system has reconfigured, and the immediate parent should update.
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

        self._mpi_proc_allocator = DefaultAllocator()

        self._subsystems_allprocs = []
        self._subsystems_myproc = []
        self._subsystems_myproc_inds = []
        self._subsystems_proc_range = []
        self._subsystems_var_range = {'input': [], 'output': []}
        self._subsystems_var_range_byset = {'input': [], 'output': []}

        self._num_var = {'input': 0, 'output': 0}
        self._num_var_byset = {'input': {}, 'output': {}}
        self._var_set2iset = {'input': {}, 'output': {}}

        self._var_promotes = {'input': [], 'output': [], 'any': []}
        self._var_allprocs_abs_names = {'input': [], 'output': []}
        self._var_abs_names = {'input': [], 'output': []}
        self._var_allprocs_prom2abs_list = {'input': {}, 'output': {}}
        self._var_abs2prom = {'input': {}, 'output': {}}
        self._var_allprocs_abs2meta = {'input': {}, 'output': {}}
        self._var_abs2meta = {'input': {}, 'output': {}}

        self._var_allprocs_abs2idx = {'input': {}, 'output': {}}
        self._var_allprocs_abs2idx_byset = {'input': {}, 'output': {}}

        self._var_sizes = {'input': None, 'output': None}
        self._var_sizes_byset = {'input': {}, 'output': {}}

        self._manual_connections = {}
        self._conn_global_abs_in2out = {}
        self._conn_parents_abs_in2out = {}
        self._conn_abs_in2out = {}

        self._ext_num_vars = {'input': (0, 0), 'output': (0, 0)}
        self._ext_num_vars_byset = {'input': {}, 'output': {}}
        self._ext_sizes = {'input': (0, 0), 'output': (0, 0)}
        self._ext_sizes_byset = {'input': {}, 'output': {}}

        self._vec_names = ['nonlinear', 'linear']
        self._vectors = {'input': {}, 'output': {}, 'residual': {}}
        self._excluded_vars_out = set()
        self._excluded_vars_in = set()

        self._inputs = None
        self._outputs = None
        self._residuals = None
        self._transfers = {}

        self._lower_bounds = None
        self._upper_bounds = None

        self._scaling_vecs = {
            ('input', 'phys0'): {}, ('input', 'phys1'): {},
            ('input', 'norm0'): {}, ('input', 'norm1'): {},
            ('output', 'phys0'): {}, ('output', 'phys1'): {},
            ('output', 'norm0'): {}, ('output', 'norm1'): {},
            ('residual', 'phys0'): {}, ('residual', 'phys1'): {},
            ('residual', 'norm0'): {}, ('residual', 'norm1'): {},
        }

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

        self._design_vars = {}
        self._responses = {}

        self._static_mode = True
        self._static_subsystems_allprocs = []
        self._static_manual_connections = {}
        self._static_design_vars = {}
        self._static_responses = {}

        self._reconfigured = False

        self.initialize()
        self.metadata.update(kwargs)

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

    def _get_initial_procs(self, comm, initial):
        """
        Get initial values for pathname and comm.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm>
            The MPI communicator.
        initial : bool
            Whether we are reconfiguring - i.e., whether the model has been previously setup.

        Returns
        -------
        str
            Global name of the system, including the path.
        MPI.Comm or <FakeComm>
            The MPI communicator.
        """
        if not initial:
            return self.pathname, self.comm
        else:
            return '', comm

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
                for iset, set_name in enumerate(self._num_var_byset[type_]):
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
            return (
                self._ext_num_vars, self._ext_num_vars_byset,
                self._ext_sizes, self._ext_sizes_byset)
        else:
            ext_num_vars = {'input': (0, 0), 'output': (0, 0)}
            ext_sizes = {'input': (0, 0), 'output': (0, 0)}
            ext_num_vars_byset = {
                'input': {set_name: (0, 0) for set_name in self._var_set2iset['input']},
                'output': {set_name: (0, 0) for set_name in self._var_set2iset['output']},
            }
            ext_sizes_byset = {
                'input': {set_name: (0, 0) for set_name in self._var_set2iset['input']},
                'output': {set_name: (0, 0) for set_name in self._var_set2iset['output']},
            }
            return ext_num_vars, ext_num_vars_byset, ext_sizes, ext_sizes_byset

    def _get_root_vectors(self, vector_class, initial):
        """
        Get the root vectors for the nonlinear and linear vectors for the model.

        Parameters
        ----------
        vector_class : Vector
            The Vector class used to instantiate the root vectors.
        initial : bool
            Whether we are reconfiguring - i.e., whether the model has been previously setup.

        Returns
        -------
        dict of dict of Vector
            Root vectors: first key is 'input', 'output', or 'residual'; second key is vec_name.
        dict of set
            Dictionary of sets of excluded output variable absolute names, keyed by vec_name.
        dict of set
            Dictionary of sets of excluded input variable absolute names, keyed by vec_name.
        """
        root_vectors = {'input': {}, 'output': {}, 'residual': {}}

        for key in ['input', 'output', 'residual']:
            type_ = 'output' if key is 'residual' else key
            for vec_name in self._vec_names:
                if not initial:
                    root_vectors[key][vec_name] = self._vectors[key][vec_name]._root_vector
                else:
                    root_vectors[key][vec_name] = vector_class(vec_name, type_, self)

        if not initial:
            excl_out = self._excluded_vars_out
            excl_in = self._excluded_vars_in
        else:
            excl_out = {vec_name: set() for vec_name in self._vec_names}
            excl_in = {vec_name: set() for vec_name in self._vec_names}

        return root_vectors, excl_out, excl_in

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
            lower = vector_class('lower', 'output', self)
            upper = vector_class('upper', 'output', self)

        return lower, upper

    def _get_scaling_root_vectors(self, vector_class, initial):
        """
        Get the root vectors for the scaling vectors.

        Parameters
        ----------
        vector_class : Vector
            The Vector class used to instantiate the root vectors.
        initial : bool
            Whether we are reconfiguring - i.e., whether the model has been previously setup.

        Returns
        -------
        dict of dict of Vector
            Root vectors: first key is 'input', 'output', or 'residual'; second key is vec_name.
        """
        root_vectors = {
            ('input', 'phys0'): {}, ('input', 'phys1'): {},
            ('input', 'norm0'): {}, ('input', 'norm1'): {},
            ('output', 'phys0'): {}, ('output', 'phys1'): {},
            ('output', 'norm0'): {}, ('output', 'norm1'): {},
            ('residual', 'phys0'): {}, ('residual', 'phys1'): {},
            ('residual', 'norm0'): {}, ('residual', 'norm1'): {},
        }

        for key in root_vectors:
            vec_key, coeff_key = key
            type_ = 'output' if vec_key == 'residual' else vec_key

            for vec_name in self._vec_names:
                if not initial:
                    root_vectors[key][vec_name] = self._scaling_vecs[key][vec_name]._root_vector
                else:
                    root_vectors[key][vec_name] = vector_class(vec_name, type_, self)

                    if coeff_key[-1] != '0':
                        root_vectors[key][vec_name].set_const(1.0)

        return root_vectors

    def resetup(self, setup_mode='full'):
        """
        Public wrapper for _setup that reconfigures after an initial setup has been performed.

        Parameters
        ----------
        setup_mode : str
            Must be one of 'full', 'reconf', or 'update'.
        """
        self._setup(self.comm, self._outputs.__class__, setup_mode=setup_mode)

    def _setup(self, comm, vector_class, setup_mode):
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
        vector_class : type
            reference to an actual <Vector> class; not an instance.
        setup_mode : str
            Must be one of 'full', 'reconf', or 'update'.
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

        # If we're only updating and not recursing, processors don't need to be redistributed
        if recurse:
            self._setup_procs(*self._get_initial_procs(comm, initial))

        # For updating variable and connection data, setup needs to be performed only
        # in the current system, by gathering data from immediate subsystems,
        # and no recursion is necessary.
        self._setup_vars(recurse=recurse)
        self._setup_var_index_ranges(self._get_initial_var_indices(initial), recurse=recurse)
        self._setup_var_data(recurse=recurse)
        self._setup_var_index_maps(recurse=recurse)
        self._setup_var_sizes(recurse=recurse)
        self._setup_global_connections(recurse=recurse)
        self._setup_connections(recurse=recurse)

        # For vector-related, setup, recursion is always necessary, even for updating.
        # For reconfiguration setup, we resize the vectors once, only in the current system.
        self._setup_global(*self._get_initial_global(initial))
        self._setup_vectors(*self._get_root_vectors(vector_class, initial), resize=resize)
        self._setup_bounds(*self._get_bounds_root_vectors(vector_class, initial), resize=resize)
        self._setup_scaling(self._get_scaling_root_vectors(vector_class, initial), resize=resize)

        # Transfers do not require recursion, but they have to be set up after the vector setup.
        self._setup_transfers(recurse=recurse)

        # Same situation with solvers, partials, and Jacobians.
        # If we're updating, we just need to re-run setup on these, but no recursion necessary.
        self._setup_solvers(recurse=recurse)
        self._setup_partials(recurse=recurse)
        self._setup_jacobians(recurse=recurse)

        # If full or reconf setup, reset this system's variables to initial values.
        if setup_mode in ('full', 'reconf'):
            self.set_initial_values()

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

        minp, maxp = self.get_req_procs()
        if MPI and comm is not None and comm != MPI.COMM_NULL and comm.size < minp:
            raise RuntimeError("%s needs %d MPI processes, but was given only %d." %
                               (self.pathname, minp, comm.size))

    def _setup_vars(self, recurse=True):
        """
        Call setup in components and count variables, total and by var_set.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        self._num_var = {'input': 0, 'output': 0}
        self._num_var_byset = {'input': {}, 'output': {}}

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
        self._subsystems_var_range = {'input': [], 'output': []}
        self._subsystems_var_range_byset = {'input': [], 'output': []}

        num_var_byset = self._num_var_byset
        for type_ in ['input', 'output']:
            for set_name in self._var_set2iset[type_]:
                if set_name not in num_var_byset[type_]:
                    num_var_byset[type_][set_name] = 0

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
        self._var_allprocs_abs2meta = {'input': {}, 'output': {}}
        self._var_abs2meta = {'input': {}, 'output': {}}

    def _setup_var_index_maps(self, recurse=True):
        """
        Compute maps from abs var names to their index among allprocs variables in this system.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        self._var_allprocs_abs2idx = allprocs_abs2idx = {'input': {}, 'output': {}}
        self._var_allprocs_abs2idx_byset = allprocs_abs2idx_byset = {'input': {}, 'output': {}}

        for type_ in ['input', 'output']:
            allprocs_abs2meta_t = self._var_allprocs_abs2meta[type_]
            allprocs_abs2idx_t = allprocs_abs2idx[type_]
            allprocs_abs2idx_byset_t = allprocs_abs2idx_byset[type_]

            counter = {set_name: 0 for set_name in self._var_set2iset[type_]}
            for idx, abs_name in enumerate(self._var_allprocs_abs_names[type_]):
                allprocs_abs2idx_t[abs_name] = idx

                set_name = allprocs_abs2meta_t[abs_name]['var_set']
                allprocs_abs2idx_byset_t[abs_name] = counter[set_name]
                counter[set_name] += 1

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
        self._var_sizes = {'input': None, 'output': None}
        self._var_sizes_byset = {'input': {}, 'output': {}}

    def _setup_global_shapes(self):
        """
        Compute the global size and shape of all variables on this system.
        """
        meta = self._var_allprocs_abs2meta['output']

        # now set global sizes and shapes into metadata for distributed outputs
        sizes = self._var_sizes['output']
        for idx, abs_name in enumerate(self._var_allprocs_abs_names['output']):
            mymeta = meta[abs_name]
            local_shape = mymeta['shape']
            if not mymeta['distributed']:
                # not distributed, just use local shape and size
                mymeta['global_size'] = np.prod(local_shape)
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

    def _setup_global_connections(self, recurse=True):
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
        """
        self._conn_global_abs_in2out = {}

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

    def _setup_vectors(self, root_vectors, excl_out, excl_in, resize=False):
        """
        Compute all vectors for all vec names and assign excluded variables lists.

        Parameters
        ----------
        root_vectors : dict of dict of Vector
            Root vectors: first key is 'input', 'output', or 'residual'; second key is vec_name.
        excl_out : dict of set
            Dictionary of sets of excluded output variable absolute names, keyed by vec_name.
        excl_in : dict of set
            Dictionary of sets of excluded input variable absolute names, keyed by vec_name.
        resize : bool
            Whether to resize the root vectors - i.e, because this system is initiating a reconf.
        """
        self._vectors = vectors = {'input': {}, 'output': {}, 'residual': {}}
        self._excluded_vars_out = excl_out
        self._excluded_vars_in = excl_in

        for vec_name in self._vec_names:
            vector_class = root_vectors['output'][vec_name].__class__

            for key in ['input', 'output', 'residual']:
                type_ = 'output' if key is 'residual' else key

                vectors[key][vec_name] = vector_class(
                    vec_name, type_, self, root_vectors[key][vec_name], resize=resize)

        self._inputs = vectors['input']['nonlinear']
        self._outputs = vectors['output']['nonlinear']
        self._residuals = vectors['residual']['nonlinear']

        for subsys in self._subsystems_myproc:
            subsys._setup_vectors(root_vectors, excl_out, excl_in)

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
            'lower', 'output', self, root_lower, resize=resize)
        self._upper_bounds = upper = vector_class(
            'upper', 'output', self, root_upper, resize=resize)

        for abs_name, meta in iteritems(self._var_abs2meta['output']):
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

    def _setup_scaling(self, root_vectors, resize=False):
        """
        Compute all scaling vectors for all vec names.

        Parameters
        ----------
        root_vectors : dict of dict of Vector
            Root vectors: first key is scaling direction; second key is vec_name.
        resize : bool
            Whether to resize the root vectors - i.e, because this system is initiating a reconf.
        """
        self._scaling_vecs = vecs = {
            ('input', 'phys0'): {}, ('input', 'phys1'): {},
            ('input', 'norm0'): {}, ('input', 'norm1'): {},
            ('output', 'phys0'): {}, ('output', 'phys1'): {},
            ('output', 'norm0'): {}, ('output', 'norm1'): {},
            ('residual', 'phys0'): {}, ('residual', 'phys1'): {},
            ('residual', 'norm0'): {}, ('residual', 'norm1'): {},
        }

        allprocs_abs2meta_out = self._var_allprocs_abs2meta['output']
        abs2meta_in = self._var_abs2meta['input']

        for vec_name in self._vec_names:
            vector_class = root_vectors['residual', 'phys0'][vec_name].__class__

            for key in vecs:
                type_ = 'output' if key[0] == 'residual' else key[0]
                vecs[key][vec_name] = vector_class(
                    vec_name, type_, self, root_vectors[key][vec_name], resize=resize)

                # This is necessary because scaling will not be set for inputs
                # whose source is outside of this system. The units and scaling
                # for those sources are not available, so those components of the
                # scaling vectors will just be 0. That will zero out input values
                # during transfers, so the multiplier must be 1 by default.
                if resize:
                    if '1' in key[1]:
                        vecs[key][vec_name].set_const(1.)

            for abs_name, meta in iteritems(self._var_abs2meta['output']):
                shape = meta['shape']
                ref = meta['ref']
                ref0 = meta['ref0']
                res_ref = meta['res_ref']
                if not np.isscalar(ref):
                    ref = ref.reshape(shape)
                if not np.isscalar(ref0):
                    ref0 = ref0.reshape(shape)
                if not np.isscalar(res_ref):
                    res_ref = res_ref.reshape(shape)

                a0 = ref0
                a1 = ref - ref0
                vecs['output', 'phys0'][vec_name]._views[abs_name][:] = a0
                vecs['output', 'phys1'][vec_name]._views[abs_name][:] = a1
                vecs['output', 'norm0'][vec_name]._views[abs_name][:] = -a0 / a1
                vecs['output', 'norm1'][vec_name]._views[abs_name][:] = 1.0 / a1

                vecs['residual', 'phys0'][vec_name]._views[abs_name][:] = 0.0
                vecs['residual', 'phys1'][vec_name]._views[abs_name][:] = res_ref
                vecs['residual', 'norm0'][vec_name]._views[abs_name][:] = 0.0
                vecs['residual', 'norm1'][vec_name]._views[abs_name][:] = 1.0 / res_ref

            for abs_in, abs_out in iteritems(self._conn_abs_in2out):
                if abs_in not in abs2meta_in:
                    continue

                meta_out = allprocs_abs2meta_out[abs_out]
                meta_in = abs2meta_in[abs_in]

                shape_out = meta_out['shape']
                units_out = meta_out['units']
                distrib_out = meta_out['distributed']
                shape_in = meta_in['shape']
                units_in = meta_in['units']

                ref = meta_out['ref']
                ref0 = meta_out['ref0']

                src_indices = meta_in['src_indices']

                if src_indices is not None:
                    if not (np.isscalar(ref) and np.isscalar(ref0)):
                        global_shape_out = meta_out['global_shape']
                        if src_indices.ndim != 1:
                            if len(shape_out) == 1:
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
                        if distrib_out:
                            raise RuntimeError("vector scalers with distrib vars "
                                               "not supported yet.")

                        if not np.isscalar(ref):
                            ref = ref[src_indices]
                        if not np.isscalar(ref0):
                            ref0 = ref0[src_indices]
                else:
                    if not np.isscalar(ref):
                        ref = ref.reshape(shape_out)
                    if not np.isscalar(ref0):
                        ref0 = ref0.reshape(shape_out)

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

                a0 = convert_units(ref0, units_out, units_in)
                a1 = convert_units(ref - ref0, units_out, units_in) \
                    - convert_units(0., units_out, units_in)
                vecs['input', 'phys0'][vec_name]._views[abs_in][:] = a0
                vecs['input', 'phys1'][vec_name]._views[abs_in][:] = a1
                vecs['input', 'norm0'][vec_name]._views[abs_in][:] = -a0 / a1
                vecs['input', 'norm1'][vec_name]._views[abs_in][:] = 1.0 / a1

        for subsys in self._subsystems_myproc:
            subsys._setup_scaling(root_vectors)

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
            # this means that somewhere above us is an AssembledJacobian. If
            # we have a nonlinear solver that uses derivatives, this is
            # currently an error if the AssembledJacobian is not a DenseJacobian.
            # In a future story we'll add support for sparse AssembledJacobians.
            if self._nonlinear_solver is not None and self._nonlinear_solver.supports['gradients']:
                if not isinstance(jacobian, DenseJacobian):
                    raise RuntimeError("System '%s' has a solver of type '%s'"
                                       "but a sparse AssembledJacobian has been set in a "
                                       "higher level system." %
                                       (self.pathname,
                                        self._nonlinear_solver.__class__.__name__))
                self._views_assembled_jac = True
            self._owns_assembled_jac = False

        if self._owns_assembled_jac:

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

            for s in self.system_iter(local=True, recurse=True):
                if s._views_assembled_jac:
                    self._jacobian._init_view(s)

    def set_initial_values(self):
        """
        Set all input and output variables to their declared initial values.
        """
        for abs_name, meta in iteritems(self._var_abs2meta['input']):
            self._inputs._views[abs_name][:] = meta['value']

        for abs_name, meta in iteritems(self._var_abs2meta['output']):
            self._outputs._views[abs_name][:] = meta['value']

    def _scale_vec(self, vec, key, scale_to):
        scal_vecs = self._scaling_vecs
        vec_name = vec._name

        vec.elem_mult(scal_vecs[key, scale_to + '1'][vec_name])
        if vec_name == 'nonlinear':
            vec += scal_vecs[key, scale_to + '0'][vec_name]

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
        vec_outputs = self._vectors['output'][vec_name]

        if mode == 'fwd':
            direction = ('norm', 'phys')
        elif mode == 'rev':
            direction = ('phys', 'norm')

        self._scale_vec(vec_inputs, 'input', direction[0])
        self._transfers[vec_name][mode, isub](vec_inputs, vec_outputs, mode)
        self._scale_vec(vec_inputs, 'input', direction[1])

    def get_req_procs(self):
        """
        Return the min and max MPI processes usable by this System.

        This should be overridden by Components that require more than
        1 process.

        Returns
        -------
        tuple : (int, int or None)
            A tuple of the form (min_procs, max_procs), indicating the min
            and max processors usable by this `System`.  max_procs can be None,
            indicating all available procs can be used.
        """
        # by default, systems only require 1 proc
        return (1, 1)

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

            found = False
            names, patterns, renames = split_list(to_match)
            for typ in io_types:
                pmap = matches[typ]
                for name in proms[typ]:
                    if name in names:
                        pmap[name] = name
                        found = True
                    elif name in renames:
                        pmap[name] = renames[name]
                        found = True
                    else:
                        for pattern in patterns:
                            # if name matches, promote that variable to parent
                            if pattern == '*' or fnmatchcase(name, pattern):
                                pmap[name] = name
                                found = True
                                break
                        else:
                            # Default: prepend the parent system's name
                            pmap[name] = gname + name if gname else name

            return found

        def error(type_):
            names = {'any': 'promotes', 'input': 'promotes_inputs', 'output': 'promotes_outputs'}
            raise RuntimeError("%s: no variables were promoted based on %s=%s" %
                               (self.pathname, names[type_], list(self._var_promotes[type_])))

        maps = {'input': {}, 'output': {}}

        if self._var_promotes['input'] or self._var_promotes['output']:
            if self._var_promotes['any']:
                raise RuntimeError("%s: 'promotes' cannot be used at the same time as "
                                   "'promotes_inputs' or 'promotes_outputs'." % self.pathname)
            if not resolve(self._var_promotes['input'], ('input',), maps, prom_names):
                error('input')
            if not resolve(self._var_promotes['output'], ('output',), maps, prom_names):
                error('output')
        else:
            if not resolve(self._var_promotes['any'], ('input', 'output',), maps, prom_names):
                error('any')

        return maps

    def _get_scope(self, excl_sub=None):
        if excl_sub is None:
            # All myproc outputs
            scope_out = set(self._var_abs_names['output'])

            # All myproc inputs connected to an output in this system
            scope_in = set(self._conn_global_abs_in2out.keys()) \
                & set(self._var_abs_names['input'])
        else:
            # All myproc outputs not in excl_sub
            scope_out = set(self._var_abs_names['output']) \
                - set(excl_sub._var_abs_names['output'])

            # All myproc inputs connected to an output in this system but not in excl_sub
            scope_in = []
            for abs_in in self._var_abs_names['input']:
                if abs_in in self._conn_global_abs_in2out:
                    abs_out = self._conn_global_abs_in2out[abs_in]

                    if abs_out not in excl_sub._var_allprocs_abs2idx['output']:
                        scope_in.append(abs_in)
            scope_in = set(scope_in)

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
        for vec in outputs:
            self._scale_vec(vec, 'output', 'phys')
        for vec in residuals:
            self._scale_vec(vec, 'residual', 'phys')

        yield

        for vec in outputs:
            self._scale_vec(vec, 'output', 'norm')
        for vec in residuals:
            self._scale_vec(vec, 'residual', 'norm')

    @contextmanager
    def _unscaled_context_all(self):
        """
        Context manager that temporarily puts all vectors and Jacobians in an unscaled state.
        """
        for vec_type in ['output', 'residual']:
            for vec in self._vectors[vec_type].values():
                self._scale_vec(vec, vec_type, 'phys')
        yield
        for vec_type in ['output', 'residual']:
            for vec in self._vectors[vec_type].values():
                self._scale_vec(vec, vec_type, 'norm')

    @contextmanager
    def _scaled_context_all(self):
        """
        Context manager that temporarily puts all vectors and Jacobians in a scaled state.
        """
        for vec_type in ['output', 'residual']:
            for vec in self._vectors[vec_type].values():
                self._scale_vec(vec, vec_type, 'norm')

        yield

        for vec_type in ['output', 'residual']:
            for vec in self._vectors[vec_type].values():
                self._scale_vec(vec, vec_type, 'phys')

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
        scope_out : set or None
            Set of absolute output names in the scope of this mat-vec product.
            If None, all are in the scope.
        scope_in : set or None
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
            elif mode == 'rev':
                d_inputs.set_const(0.0)
                d_outputs.set_const(0.0)

        excl_out = self._excluded_vars_out[vec_name]
        excl_in = self._excluded_vars_in[vec_name]

        res_names = set(self._var_abs_names['output']) - excl_out
        out_names = set(self._var_abs_names['output']) - excl_out
        in_names = set(self._var_abs_names['input']) - excl_in
        if scope_out is not None:
            out_names = out_names & scope_out
        if scope_in is not None:
            in_names = in_names & scope_in

        d_inputs._names = in_names
        d_outputs._names = out_names
        d_residuals._names = res_names

        yield d_inputs, d_outputs, d_residuals

        # reset _names so users will see full vector contents
        d_inputs._names = d_inputs._views
        d_outputs._names = d_outputs._views
        d_residuals._names = d_residuals._views

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

    @property
    def proc_allocator(self):
        """
        Get the current system's processor allocator object.
        """
        return self._mpi_proc_allocator

    @proc_allocator.setter
    def proc_allocator(self, value):
        """
        Set the processor allocator object.
        """
        self._mpi_proc_allocator = value

    def _set_partials_meta(self):
        """
        Set subjacobian info into our jacobian.

        Overridden in <Component>.
        """
        pass

    def system_iter(self, local=True, include_self=False, recurse=True,
                    typ=None):
        """
        Yield a generator of subsystems of this system.

        Parameters
        ----------
        local : bool
            If True, only iterate over systems on this proc.
        include_self : bool
            If True, include this system in the iteration.
        recurse : bool
            If True, iterate over the whole tree under this system.
        typ : type
            If not None, only yield Systems that match that are instances of the
            given type.
        """
        if local:
            sysiter = self._subsystems_myproc
        else:
            sysiter = self._subsystems_allprocs

        if include_self and (typ is None or isinstance(self, typ)):
            yield self

        for s in sysiter:
            if typ is None or isinstance(s, typ):
                yield s
            if recurse:
                for sub in s.system_iter(local=local, recurse=True, typ=typ):
                    yield sub

    def add_design_var(self, name, lower=None, upper=None, ref=None,
                       ref0=None, indices=None, adder=None, scaler=None):
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
            if np.all(adder == 0.0):
                adder = None
        elif adder == 0.0:
            adder = None
        dvs['adder'] = adder

        dvs['name'] = name
        dvs['upper'] = upper
        dvs['lower'] = lower
        dvs['ref'] = ref
        dvs['ref0'] = ref0
        if indices is not None:
            dvs['size'] = len(indices)
            indices = np.atleast_1d(indices)
        dvs['indices'] = indices

    def add_response(self, name, type_, lower=None, upper=None, equals=None,
                     ref=None, ref0=None, indices=None, index=None,
                     adder=None, scaler=None, linear=False):
        r"""
        Add a response variable to this system.

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

        Notes
        -----
        The response can be scaled using ref and ref0.
        The argument :code:`ref0` represents the physical value when the scaled value is 0.
        The argument :code:`ref` represents the physical value when the scaled value is 1.

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
            msg = '{0} \'{1}\' already exists.'.format(typemap[type], name)
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
        err = False
        if indices is not None:
            if isinstance(indices, string_types):
                err = True
            elif isinstance(indices, Iterable):
                all_int = all([isinstance(item, int) for item in indices])
                if not all_int:
                    err = True
            else:
                err = True
        if err:
            msg = "If specified, indices must be a sequence of integers."
            raise ValueError(msg)

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

        if self._static_mode:
            responses = self._static_responses
        else:
            responses = self._responses

        responses[name] = resp = OrderedDict()

        if isinstance(scaler, np.ndarray):
            if np.all(scaler == 1.0):
                scaler = None
        elif scaler == 1.0:
            scaler = None
        resp['scaler'] = scaler

        if isinstance(adder, np.ndarray):
            if np.all(adder == 0.0):
                adder = None
        elif adder == 0.0:
            adder = None
        resp['adder'] = adder

        resp['name'] = name
        resp['ref'] = ref
        resp['ref0'] = ref0
        resp['type'] = type_

        if type_ == 'con':
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

    def add_constraint(self, name, lower=None, upper=None, equals=None,
                       ref=None, ref0=None, adder=None, scaler=None,
                       indices=None, linear=False):
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

        Notes
        -----
        The response can be scaled using ref and ref0.
        The argument :code:`ref0` represents the physical value when the scaled value is 0.
        The argument :code:`ref` represents the physical value when the scaled value is 1.
        """
        self.add_response(name=name, type_='con', lower=lower, upper=upper,
                          equals=equals, scaler=scaler, adder=adder, ref=ref,
                          ref0=ref0, indices=indices, linear=linear)

    def add_objective(self, name, ref=None, ref0=None, index=None,
                      adder=None, scaler=None):
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
            If variable is an array, this indicates which entriy is of
            interest for this particular response. This may be a positive
            or negative integer.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.

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
                          ref=ref, ref0=ref0, index=index)

    def get_design_vars(self, recurse=True):
        """
        Get the DesignVariable settings from this system.

        Retrieve all design variable settings from the system and, if recurse
        is True, all of its subsystems.

        Parameters
        ----------
        recurse : bool
            If True, recurse through the subsystems and return the path of
            all design vars relative to the this system.

        Returns
        -------
        dict
            The design variables defined in the current system and, if
            recurse=True, its subsystems.

        """
        pro2abs = self._var_allprocs_prom2abs_list['output']

        # Human readable error message during Driver setup.
        try:
            out = {pro2abs[name][0]: data for name, data in iteritems(self._design_vars)}
        except KeyError as err:
            msg = "Output not found for design variable {0} in system '{1}'."
            raise RuntimeError(msg.format(str(err), self.pathname))

        # Size them all
        vec = self._outputs._views_flat
        for name, data in iteritems(out):
            if 'size' not in data:
                # Depending on where the designvar was added, the name in the
                # vectors might be relative instead of absolute. Lucky we have
                # both.
                if name in vec:
                    data['size'] = vec[name].size
                else:
                    data['size'] = vec[out[name]['name']].size

        if recurse:
            for subsys in self._subsystems_myproc:
                subsys_design_vars = subsys.get_design_vars(recurse=recurse)
                for key in subsys_design_vars:
                    out[key] = subsys_design_vars[key]
            if self.comm.size > 1 and self._subsystems_allprocs:
                iproc = self.comm.rank
                for rank, all_out in enumerate(self.comm.allgather(out)):
                    if rank != iproc:
                        out.update(all_out)

        return out

    def get_responses(self, recurse=True):
        """
        Get the response variable settings from this system.

        Retrieve all response variable settings from the system as a dict,
        keyed by variable name.

        Parameters
        ----------
        recurse : bool, optional
            If True, recurse through the subsystems and return the path of
            all responses relative to the this system.

        Returns
        -------
        dict
            The responses defined in the current system and, if
            recurse=True, its subsystems.

        """
        prom2abs = self._var_allprocs_prom2abs_list['output']

        # Human readable error message during Driver setup.
        try:
            out = {prom2abs[name][0]: data for name, data in iteritems(self._responses)}
        except KeyError as err:
            msg = "Output not found for response {0} in system '{1}'."
            raise RuntimeError(msg.format(str(err), self.pathname))

        # Size them all
        vec = self._outputs._views_flat
        for name in out:
            if 'size' not in out[name]:
                out[name]['size'] = vec[name].size

        if recurse:
            for subsys in self._subsystems_myproc:
                subsys_responses = subsys.get_responses(recurse=recurse)
                for key in subsys_responses:
                    out[key] = subsys_responses[key]

            if self.comm.size > 1 and self._subsystems_allprocs:
                iproc = self.comm.rank
                for rank, all_out in enumerate(self.comm.allgather(out)):
                    if rank != iproc:
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

    def list_inputs(self, explicit=True, implicit=True, out_stream=sys.stdout):
        """
        List inputs.

        Parameters
        ----------
        explicit : bool, optional
            include inputs from explicit components. Default is True.

        implicit : bool, optional
            include inputs from implicit components. Default is True.

        out_stream : file_like
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        list
            list of (name, value) of inputs
        """
        states = self._list_states()

        expl_inputs = []
        impl_inputs = []
        for name, val in iteritems(self._inputs._views):
            if name in states:
                impl_inputs.append((name, val))
            else:
                expl_inputs.append((name, val))

        if out_stream:
            if explicit:
                self._write_inputs('Explicit', expl_inputs, out_stream)
            if implicit:
                self._write_inputs('Implicit', impl_inputs, out_stream)

        if explicit and implicit:
            return expl_inputs + impl_inputs
        elif explicit:
            return expl_inputs
        elif implicit:
            return impl_inputs
        else:
            raise RuntimeError('You have excluded both Explicit and Implicit components.')

    def _write_inputs(self, comp_type, inputs, out_stream=sys.stdout):
        """
        Write formatted input values to out_stream.

        Parameters
        ----------
        comp_type : str, 'Explicit' or 'Implicit'
            the type of component with the input values.

        inputs : list
            list of (name, value) tuples.

        out_stream : file_like
            Where to send human readable output. Default is sys.stdout.
        """
        if out_stream is None:
            return

        count = len(inputs)

        pathname = self.pathname if self.pathname else 'model'

        header = "%d Input(s) to %s Components in '%s'\n" % (count, comp_type, pathname)
        out_stream.write(header)

        if count:
            out_stream.write("-" * len(header) + "\n")
            for name, val in sorted(inputs):
                out_stream.write("%s\n" % name)
                out_stream.write("  value:    " + str(val))
                out_stream.write('\n\n')

    def list_outputs(self, explicit=True, implicit=True, out_stream=sys.stdout):
        """
        List outputs.

        Parameters
        ----------
        explicit : bool, optional
            include outputs from explicit components. Default is True.

        implicit : bool, optional
            include outputs from implicit components. Default is True.

        out_stream : file_like
            Where to send human readable output. Default is sys.stdout.
            Set to None to suppress.

        Returns
        -------
        list
            list of (name, value) of outputs
        """
        states = self._list_states()

        expl_outputs = []
        impl_outputs = []
        for name, val in iteritems(self._outputs._views):
            if name in states:
                impl_outputs.append((name, val))
            else:
                expl_outputs.append((name, val))

        if out_stream:
            if explicit:
                self._write_outputs('Explicit', expl_outputs, out_stream)
            if implicit:
                self._write_outputs('Implicit', impl_outputs, out_stream)

        if explicit and implicit:
            return expl_outputs + impl_outputs
        elif explicit:
            return expl_outputs
        elif implicit:
            return impl_outputs
        else:
            raise RuntimeError('You have excluded both Explicit and Implicit components.')

    def list_residuals(self, explicit=True, implicit=True, out_stream=sys.stdout):
        """
        List residuals.

        Parameters
        ----------
        explicit : bool, optional
            include outputs from explicit components. Default is True.

        implicit : bool, optional
            include outputs from implicit components. Default is True.

        out_stream : file_like
            Where to send human readable output. Default is sys.stdout. Set to None to suppress.

        Returns
        -------
        list
            list of (name, value) of residuals
        """
        states = self._list_states()

        expl_resids = []
        impl_resids = []
        for name, val in iteritems(self._residuals._views):
            if name in states:
                impl_resids.append((name, val))
            else:
                expl_resids.append((name, val))

        if out_stream:
            if explicit:
                self._write_outputs('Explicit', expl_resids, out_stream)
            if implicit:
                self._write_outputs('Implicit', impl_resids, out_stream)

        if explicit and implicit:
            return expl_resids + impl_resids
        elif explicit:
            return expl_resids
        elif implicit:
            return impl_resids
        else:
            raise RuntimeError('You have excluded both Explicit and Implicit components.')

    def _write_outputs(self, comp_type, outputs, out_stream=sys.stdout):
        """
        Write formatted output values and residuals to out_stream.

        Parameters
        ----------
        comp_type : str, 'Explicit' or 'Implicit'
            the type of component with the output values.

        outputs : list
            list of (name, value) tuples.

        out_stream : file_like
            Where to send human readable output. Default is sys.stdout.
        """
        if out_stream is None:
            return

        count = len(outputs)

        pathname = self.pathname if self.pathname else 'model'

        header = "%d %s Output(s) in '%s'\n" % (count, comp_type, pathname)
        out_stream.write(header)

        if count:
            out_stream.write("-" * len(header) + "\n")
            for name, _ in sorted(outputs):
                out_stream.write("%s\n" % name)
                out_stream.write("  value:    " + str(self._outputs._views[name]))
                out_stream.write('\n')
                out_stream.write("  residual: " + str(self._residuals._views[name]))
                out_stream.write('\n\n')

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
            self._apply_linear(vec_names, mode, scope_out, scope_in)

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
            result = self._solve_linear(vec_names, mode)

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

    def _apply_linear(self, vec_names, mode, var_inds=None):
        """
        Compute jac-vec product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        var_inds : [int, int, int, int] or None
            ranges of variable IDs involved in this matrix-vector product.
            The ordering is [lb1, ub1, lb2, ub2].
        """
        pass

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
