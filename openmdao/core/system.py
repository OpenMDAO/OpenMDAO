"""Define the base System class."""
from __future__ import division

from contextlib import contextmanager
from collections import namedtuple, OrderedDict, Iterable
from fnmatch import fnmatchcase
import sys
from itertools import product

from six import iteritems, string_types
from six.moves import range

import numpy as np

from openmdao.proc_allocators.default_allocator import DefaultAllocator
from openmdao.jacobians.dictionary_jacobian import DictionaryJacobian
from openmdao.jacobians.assembled_jacobian import AssembledJacobian
from openmdao.utils.generalized_dict import GeneralizedDictionary
from openmdao.utils.units import convert_units
from openmdao.utils.class_util import overrides_method
from openmdao.utils.general_utils import \
    determine_adder_scaler, format_as_float_or_array, ensure_compatible
from openmdao.utils.mpi import MPI


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
        name of the system, must be different from siblings.
    pathname : str
        global name of the system, including the path.
    comm : MPI.Comm or <FakeComm>
        MPI communicator object.
    metadata : <GeneralizedDictionary>
        dictionary of user-defined arguments.
    _first_setup : bool
        If True, this is the first time we are setting up, so we should not clear the
        _var_rel2data_io and _var_rel_names attributes prior to setup.
    _assembler : <Assembler>
        pointer to the global assembler object.
    _mpi_proc_allocator : <ProcAllocator>
        object that distributes procs among subsystems.
    _mpi_req_procs : (int, int or None)
        number of min and max procs usable by this system.
    _mpi_proc_range : (int, int)
        The range of processors that the comm on this system owns, in the global index space.
    _subsystems_allprocs : [<System>, ...]
        list of all subsystems (children of this system).
    _subsystems_myproc : [<System>, ...]
        list of local subsystems that exist on this proc.
    _subsystems_myproc_inds : [int, ...]
        list of indices of subsystems on this proc among all of this system's
        subsystems (subsystems on all of this system's processors).
    _var_allprocs_idx_range : {'input': [int, int], 'output': [int, int]}
        index range of owned variables with respect to all problem variables.
    _var_promotes : { 'any': [], 'input': [], 'output': [] }
        dictionary of lists of variable names/wildcards specifying promotion
        (used to calculate promoted names)
    _manual_connections : dict
        dictionary of input_name: (output_name, src_indices) connections.
    _manual_connections_abs : [(str, str), ...]
        _manual_connections with absolute variable names.  Entries
        have the form (input, output).
    _var_allprocs_prom2abs_list : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to list of all absolute names.
        For outputs, the list will have length one since promoted output names are unique.
    _var_allprocs_idx_range : {'input': [int, int], 'output': [int, int]}
        Global index range of owned variables with respect to all model variables.
    _var_abs_names : {'input': [str, ...], 'output': [str, ...]}
        List of absolute names of owned variables existing on current proc.
    _var_abs2data_io : dict
        Dictionary mapping absolute names to dicts with keys (prom, rel, my_idx, type_, metadata).
        The my_idx entry is the index among variables in this system, on this processor.
        The type_ entry is either 'input' or 'output'.
    _vectors : {'input': dict, 'output': dict, 'residual': dict}
        dict of vector objects. These are the derivatives vectors.
    _vector_transfers : dict
        dict of transfer objects.
    _vector_var_ids : dict
        dictionary of index arrays of relevant variables for this vector
    _scaling_to_norm : {'input': ndarray[nvar_in, 2], 'output': ndarray[nvar_out, 2]}
        coefficients to convert vectors to normalized values.
        In the integer arrays, nvar_in and nvar_out are counts of variables on myproc.
    _scaling_to_phys : {'input': ndarray[nvar_in, 2], 'output': ndarray[nvar_out, 2]}
        coefficients to convert vectors to physical values.
        In the integer arrays, nvar_in and nvar_out are counts of variables on myproc.
    _lower_bounds : <Vector>
        vector of lower bounds, scaled and dimensionless.
    _upper_bounds : <Vector>
        vector of upper bounds, scaled and dimensionless.
    _inputs : <Vector>
        inputs vector; points to _vectors['input']['nonlinear'].
    _outputs : <Vector>
        outputs vector; points to _vectors['output']['nonlinear'].
    _residuals : <Vector>
        residuals vector; points to _vectors['residual']['nonlinear'].
    _transfers : dict of <Transfer>
        transfer object; points to _vector_transfers['nonlinear'].
    _jacobian : <Jacobian>
        <Jacobian> object to be used in apply_linear.
    _jacobian_changed : bool
        If True, the jacobian has changed since the last call to setup.
    _owns_assembled_jac : bool
        If True, we are owners of the AssembledJacobian in self._jacobian.
    _subjacs_info : OrderedDict of dict
        Sub-jacobian metadata for each (output, input) pair added using
        declare_partials. Members of each pair may be glob patterns.
    _nl_solver : <NonlinearSolver>
        nonlinear solver to be used for solve_nonlinear.
    _ln_solver : <LinearSolver>
        linear solver to be used for solve_linear; not the Newton system.
    _suppress_solver_output : boolean
        flag that turns off all solver output for this System and all
        of its descendants if False.
    _design_vars : dict of namedtuple
        dict of all driver design vars added to the system.
    _responses : dict of namedtuple
        dict of all driver responses added to the system.

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
        self.metadata = GeneralizedDictionary()
        self.metadata.update(kwargs)

        self._first_setup = True

        self._assembler = None

        self._mpi_proc_allocator = DefaultAllocator()
        self._mpi_req_procs = None
        self._mpi_proc_range = None

        self._subsystems_allprocs = []
        self._subsystems_myproc = []
        self._subsystems_myproc_inds = []

        self._var_promotes = {'input': [], 'output': [], 'any': []}

        self._manual_connections = {}
        self._manual_connections_abs = []

        self._var_allprocs_prom2abs_list = {'input': {}, 'output': {}}
        self._var_allprocs_idx_range = {'input': [0, 0], 'output': [0, 0]}
        self._var_abs_names = {'input': [], 'output': []}
        self._var_abs2data_io = {}

        self._vectors = {'input': {}, 'output': {}, 'residual': {}}
        self._vector_transfers = {}
        self._vector_var_ids = {}

        self._scaling_to_norm = {
            'input': None, 'output': None, 'residual': None}
        self._scaling_to_phys = {
            'input': None, 'output': None, 'residual': None}

        self._lower_bounds = None
        self._upper_bounds = None

        self._inputs = None
        self._outputs = None
        self._residuals = None
        self._transfers = None

        self._jacobian = DictionaryJacobian()
        self._jacobian._system = self
        self._jacobian_changed = True
        self._owns_assembled_jac = False

        self._subjacs_info = {}

        self._nl_solver = None
        self._ln_solver = None
        self._suppress_solver_output = False

        self._design_vars = {}
        self._responses = {}

        # # Reconfigurability attributes
        #
        # self.pathname = ''
        # self.comm = None
        # self._mpi_proc_range = [0, 1]
        #
        # self._varx_abs_names = {'input': [], 'output': []}
        # self._varx_allprocs_abs_names = {'input': [], 'output': []}
        # self._varx_allprocs_prom2abs_list = {'input': {}, 'output': {}}
        # self._varx_abs2data_io = {}
        # self._varx_allprocs_abs2meta_io = {}
        #
        # self._varx_allprocs_set2abs_names = {'input': {}, 'output': {}}
        # self._varx_set2iset = None
        #
        # self._varx_allprocs_idx_range = {'input': [0, 0], 'output': [0, 0]}
        # self._varx_allprocs_vst_idx_ranges = {'input': None, 'output': None}
        # self._varx_allprocs_abs2idx_io = {}
        # self._varx_set_indices = {'input': None, 'output': None}
        #
        # self._varx_sizes = {'input': None, 'output': None}
        # self._varx_sizes_byset = {'input': {}, 'output': {}}

    #
    #
    # -------------------------------------------------------------------------------------
    # Start of reconfigurability changes

    def _get_initial_var_indices(self):
        set2iset = {}
        for type_ in ['input', 'output']:
            set2iset[type_] = {}
            for iset, set_name in enumerate(self._num_var_byset[type_]):
                set2iset[type_][set_name] = iset

        var_range = {}
        var_range_byset = {}
        for type_ in ['input', 'output']:
            var_range[type_] = (0, self._num_var[type_])

            var_range_byset[type_] = {}
            for set_name in set2iset[type_]:
                var_range_byset[type_][set_name] = (0, self._num_var_byset[type_][set_name])

        return set2iset, var_range, var_range_byset

    def _get_initial_global(self):
        ext_num_vars = {'input': (0, 0), 'output': (0, 0)}
        ext_sizes = {'input': (0, 0), 'output': (0, 0)}
        ext_num_vars_byset = {
            'input': {set_name: (0, 0) for set_name in self._varx_set2iset['input']},
            'output': {set_name: (0, 0) for set_name in self._varx_set2iset['output']},
        }
        ext_sizes_byset = {
            'input': {set_name: (0, 0) for set_name in self._varx_set2iset['input']},
            'output': {set_name: (0, 0) for set_name in self._varx_set2iset['output']},
        }
        return ext_num_vars, ext_sizes, ext_num_vars_byset, ext_sizes_byset

    def _get_root_vectors(self, vec_names, vector_class):
        root_vectors = {'input': {}, 'output': {}, 'residual': {}}

        for key in ['input', 'output', 'residual']:
            type_ = 'output' if key is 'residual' else key
            for vec_name in vec_names:
                root_vectors[key][vec_name] = vector_class(vec_name, type_, self)

        return root_vectors

    def _get_bounds_root_vectors(self, vector_class):
        lower = vector_class('lower', 'output', self)
        upper = vector_class('upper', 'output', self)
        return lower, upper

    def _get_scaling_root_vectors(self, vec_names, vector_class):
        root_vectors = {
            ('input', 'phys0'): {}, ('input', 'phys1'): {},
            ('input', 'norm0'): {}, ('input', 'norm1'): {},
            ('output', 'phys0'): {}, ('output', 'phys1'): {},
            ('output', 'norm0'): {}, ('output', 'norm1'): {},
            ('residual', 'phys'): {}, ('residual', 'norm'): {}
        }

        for key in root_vectors:
            vec_key, ref_key = key
            type_ = 'output' if vec_key == 'residual' else vec_key

            for vec_name in vec_names:
                root_vectors[key][vec_name] = vector_class(vec_name, type_, self)

        return root_vectors

    def _setupx(self, comm, vector_class):
        # TEMPORARY: this is meant to only be here during the transition to reconfigurability
        from openmdao.vectors.default_vector import DefaultVector, DefaultVectorX
        if vector_class is DefaultVector:
            vector_class = DefaultVectorX
        try:
            from openmdao.vectors.petsc_vector import PETScVector, PETScVectorX
            if vector_class is PETScVector:
                vector_class = PETScVectorX
        except:
            pass

        vec_names = ['nonlinear', 'linear']

        self._mpi_req_procs = self.get_req_procs()
        self._setupx_procs('', comm, (0, comm.size))
        self._setupx_vars()
        self._setupx_var_index_ranges(*self._get_initial_var_indices())
        self._setupx_var_data()
        self._setupx_var_index_maps()
        self._setupx_var_sizes()
        self._setupx_global_connections()
        self._setupx_connections()
        self._setupx_global(*self._get_initial_global())
        self._setupx_vectors(vec_names, self._get_root_vectors(vec_names, vector_class))
        self._setupx_transfers()
        self._setupx_bounds(*self._get_bounds_root_vectors(vector_class))
        self._setupx_scaling(self._get_scaling_root_vectors(vec_names, vector_class))
        self._setupx_solvers()

    def _setupx_procs(self, pathname, comm, proc_range):
        self.pathname = pathname
        self.comm = comm
        self._mpi_proc_range = proc_range

        minp, maxp = self._mpi_req_procs
        if MPI and comm is not None and comm != MPI.COMM_NULL and comm.size < minp:
            raise RuntimeError("%s needs %d MPI processes, but was given only %d." %
                               (self.pathname, minp, comm.size))

    def _setupx_vars(self):
        self._num_var = {'input': 0, 'output': 0}
        self._num_var_byset = {'input': {}, 'output': {}}

    def _setupx_var_index_ranges(self, set2iset, var_range, var_range_byset):
        self._varx_set2iset = set2iset
        self._varx_range = var_range
        self._varx_range_byset = var_range_byset

        num_var_byset = self._num_var_byset
        for type_ in ['input', 'output']:
            for set_name in self._varx_set2iset[type_]:
                if set_name not in num_var_byset[type_]:
                    num_var_byset[type_][set_name] = 0

    def _setupx_var_data(self):
        self._varx_allprocs_abs_names = {'input': [], 'output': []}
        self._varx_abs_names = {'input': [], 'output': []}
        self._varx_allprocs_prom2abs_list = {'input': {}, 'output': {}}
        self._varx_abs2prom = {'input': {}, 'output': {}}
        self._varx_allprocs_abs2meta = {'input': {}, 'output': {}}
        self._varx_abs2meta = {'input': {}, 'output': {}}

    def _setupx_var_index_maps(self):
        self._varx_allprocs_abs2idx = allprocs_abs2idx = {'input': {}, 'output': {}}
        self._varx_allprocs_abs2idx_byset = allprocs_abs2idx_byset = {'input': {}, 'output': {}}

        for type_ in ['input', 'output']:
            allprocs_abs2meta_t = self._varx_allprocs_abs2meta[type_]
            allprocs_abs2idx_t = allprocs_abs2idx[type_]
            allprocs_abs2idx_byset_t = allprocs_abs2idx_byset[type_]

            counter = {set_name: 0 for set_name in self._varx_set2iset[type_]}
            for idx, abs_name in enumerate(self._varx_allprocs_abs_names[type_]):
                allprocs_abs2idx_t[abs_name] = idx

                set_name = allprocs_abs2meta_t[abs_name]['var_set']
                allprocs_abs2idx_byset_t[abs_name] = counter[set_name]
                counter[set_name] += 1

    def _setupx_var_sizes(self):
        self._varx_sizes = {'input': None, 'output': None}
        self._varx_sizes_byset = {'input': {}, 'output': {}}

    def _setupx_global_connections(self):
        self._conn_global_abs_in2out = {}

    def _setupx_connections(self):
        self._conn_abs_in2out = {}

    def _setupx_partials(self):
        self._subjacs_info = {}

    def _setupx_global(self, ext_num_vars, ext_sizes, ext_num_vars_byset, ext_sizes_byset):
        self._ext_num_vars = ext_num_vars
        self._ext_sizes = ext_sizes
        self._ext_num_vars_byset = ext_num_vars_byset
        self._ext_sizes_byset = ext_sizes_byset

    def _setupx_vectors(self, vec_names, root_vectors, rel_out=None, rel_in=None):
        self._vec_names = vec_names
        self._vectors = vectors = {'input': {}, 'output': {}, 'residual': {}}
        self._relevant_vars_out = rel_out
        self._relevant_vars_in = rel_in

        if rel_out is None:
            self._relevant_vars_out = {}
            for vec_name in vec_names:
                self._relevant_vars_out[vec_name] = \
                    set(self._varx_allprocs_abs_names['output'])
        if rel_in is None:
            self._relevant_vars_in = {}
            for vec_name in vec_names:
                self._relevant_vars_in[vec_name] = \
                    set(self._varx_allprocs_abs_names['input'])

        for vec_name in vec_names:
            vector_class = root_vectors['output'][vec_name].__class__

            for key in ['input', 'output', 'residual']:
                type_ = 'output' if key is 'residual' else key

                vectors[key][vec_name] = vector_class(
                    vec_name, type_, self, root_vectors[key][vec_name])

            if vec_name is 'nonlinear':
                self._inputs = vectors['input']['nonlinear']
                self._outputs = vectors['output']['nonlinear']
                self._residuals = vectors['residual']['nonlinear']

                for abs_name, meta in iteritems(self._varx_abs2meta['input']):
                    self._inputs._views[abs_name][:] = meta['value']

                for abs_name, meta in iteritems(self._varx_abs2meta['output']):
                    self._outputs._views[abs_name][:] = meta['value']

    def _setupx_transfers(self):
        self._xfers = {}

    def _setupx_bounds(self, root_lower, root_upper):
        vector_class = root_lower.__class__
        self._lower_boundsx = lower = vector_class('lower', 'output', self, root_lower)
        self._upper_boundsx = upper = vector_class('upper', 'output', self, root_upper)

        for abs_name, meta in iteritems(self._varx_abs2meta['output']):
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

    def _setupx_scaling(self, root_vectors):
        self._scaling_vecs = vecs = {
            ('input', 'phys0'): {}, ('input', 'phys1'): {},
            ('input', 'norm0'): {}, ('input', 'norm1'): {},
            ('output', 'phys0'): {}, ('output', 'phys1'): {},
            ('output', 'norm0'): {}, ('output', 'norm1'): {},
            ('residual', 'phys'): {}, ('residual', 'norm'): {}
        }

        allprocs_abs2meta_out = self._varx_allprocs_abs2meta['output']
        abs2meta_in = self._varx_abs2meta['input']

        for vec_name in self._vec_names:
            vector_class = root_vectors['residual', 'phys'][vec_name].__class__

            for key in vecs:
                type_ = 'output' if key[0] == 'residual' else key[0]
                vecs[key][vec_name] = vector_class(
                    vec_name, type_, self, root_vectors[key][vec_name])

            for abs_name, meta in iteritems(self._varx_abs2meta['output']):
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

                vecs['output', 'phys0'][vec_name]._views[abs_name][:] = ref0
                vecs['output', 'phys1'][vec_name]._views[abs_name][:] = ref - ref0
                vecs['output', 'norm0'][vec_name]._views[abs_name][:] = -ref0 / (ref - ref0)
                vecs['output', 'norm1'][vec_name]._views[abs_name][:] = 1.0 / (ref - ref0)

                vecs['residual', 'phys'][vec_name]._views[abs_name][:] = res_ref
                vecs['residual', 'norm'][vec_name]._views[abs_name][:] = 1.0 / res_ref

            for abs_in, abs_out in iteritems(self._conn_abs_in2out):
                if abs_in not in abs2meta_in:
                    continue

                meta_out = allprocs_abs2meta_out[abs_out]
                shape_out = meta_out['shape']
                ref = meta_out['ref']
                ref0 = meta_out['ref0']

                meta_in = abs2meta_in[abs_in]
                shape_in = meta_in['shape']
                src_indices = meta_in['src_indices']
                if src_indices is not None:
                    entries = [list(range(x)) for x in shape_in]
                    cols = np.vstack(src_indices[i] for i in product(*entries))
                    dimidxs = [cols[:, i] for i in range(cols.shape[1])]
                    src_indices = np.ravel_multi_index(dimidxs, shape_out)
                    if not np.isscalar(ref):
                        ref = ref[src_indices]
                    if not np.isscalar(ref0):
                        ref0 = ref0[src_indices]
                else:
                    if not np.isscalar(ref):
                        ref = ref.reshape(shape)
                    if not np.isscalar(ref0):
                        ref0 = ref0.reshape(shape)

                vecs['input', 'phys0'][vec_name]._views[abs_in][:] = ref0
                vecs['input', 'phys1'][vec_name]._views[abs_in][:] = ref - ref0
                vecs['input', 'norm0'][vec_name]._views[abs_in][:] = -ref0 / (ref - ref0)
                vecs['input', 'norm1'][vec_name]._views[abs_in][:] = 1.0 / (ref - ref0)

    def _setupx_solvers(self):
        if self._nl_solver is not None:
            self._nl_solver._setup_solvers(self, 0)
        if self._ln_solver is not None:
            self._ln_solver._setup_solvers(self, 0)

    # End of reconfigurability changes
    # -------------------------------------------------------------------------------------
    #
    #

    def _scale_vec(self, vec, key, scale_to):
        scal_vecs = self._scaling_vecs
        vec_name = vec._name

        if key is not 'residual':
            vec.elem_mult(scal_vecs[key, scale_to + '1'][vec_name])
            vec += scal_vecs[key, scale_to + '0'][vec_name]
        else:
            vec.elem_mult(scal_vecs[key, scale_to][vec_name])

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
        self._xfers[vec_name][mode, isub](vec_inputs, vec_outputs, mode)

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

    def _setup_jacobians(self, jacobian=None):
        """
        Set and populate jacobians down through the system tree.

        Parameters
        ----------
        jacobian : <AssembledJacobian> or None
            The global jacobian to populate for this system.
        """
        self._jacobian_changed = False

        if self._owns_assembled_jac:

            # At present, we don't support a AssembledJacobian in a group if any subcomponents
            # are matrix-free.
            for subsys in self.system_iter():

                try:
                    if subsys._matrix_free:
                        msg = "AssembledJacobian not supported if any subcomponent is matrix-free."
                        raise RuntimeError(msg)

                # Groups don't have `_matrix_free`
                # Note, we could put this attribute on Group, but this would be True for a
                # default Group, and thus we would need an isinstance on Component, which is the
                # reason for the try block anyway.
                except AttributeError:
                    continue

            jacobian = self._jacobian

        elif jacobian is not None:
            self._jacobian = jacobian

        self._set_partials_meta()

        for subsys in self._subsystems_myproc:
            subsys._setup_jacobians(jacobian)

        if self._owns_assembled_jac:
            self._jacobian._system = self
            self._jacobian._initialize()

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

        maps = {'input': {}, 'output': {}}
        gname = self.name + '.' if self.name else ''
        found = False

        promotes = self._var_promotes['any']
        if promotes:
            names, patterns, renames = split_list(promotes)

        for typ in ('input', 'output'):
            pmap = maps[typ]

            if promotes:
                pass
            elif self._var_promotes[typ]:
                names, patterns, renames = split_list(self._var_promotes[typ])
            else:
                names = patterns = renames = ()

            for name in prom_names[typ]:
                if name in pmap:
                    pass
                elif name in names:
                    pmap[name] = name
                    found = True
                elif name in renames:
                    pmap[name] = renames[name]
                    found = True
                else:
                    for pattern in patterns:
                        # if name matches, promote that variable to parent
                        if fnmatchcase(name, pattern):
                            pmap[name] = name
                            found = True
                            break
                    else:
                        # Default: prepend the parent system's name
                        pmap[name] = gname + name if gname else name

        if not found:
            for io, lst in self._var_promotes.items():
                if lst:
                    if io == 'any':
                        suffix = ''
                    else:
                        suffix = '_%ss' % io
                    raise RuntimeError("%s: no variables were promoted "
                                       "based on promotes%s=%s" %
                                       (self.pathname, suffix, list(lst)))

        return maps

    def _get_scope(self, excl_sub=None):
        if excl_sub is None:
            # All myproc outputs
            scope_out = set(self._varx_abs_names['output'])

            # All myproc inputs connected to an output in this system
            scope_in = set(self._conn_global_abs_in2out.keys()) \
                & set(self._varx_abs_names['input'])
        else:
            # All myproc outputs not in excl_sub
            scope_out = set(self._varx_abs_names['output']) \
                - set(excl_sub._varx_abs_names['output'])

            # All myproc inputs connected to an output in this system but not in excl_sub
            scope_in = []
            for abs_in in self._varx_abs_names['input']:
                if abs_in in self._conn_global_abs_in2out:
                    abs_out = self._conn_global_abs_in2out[abs_in]

                    if abs_out not in excl_sub._varx_allprocs_abs2idx['output']:
                        scope_in.append(abs_in)
            scope_in = set(scope_in)

        return scope_out, scope_in

    @property
    def jacobian(self):
        """
        A Jacobian object or None.
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
    def _units_scaling_context(self, inputs=[], outputs=[], residuals=[], scale_jac=False):
        """
        Context manager for units and scaling for vectors and Jacobians.

        Temporarily puts vectors in a physical and unscaled state, because
        internally, vectors are nominally in a dimensionless and scaled state.
        The same applies (optionally) for Jacobians.

        Parameters
        ----------
        inputs : list of input <Vector> objects
            List of input vectors to apply the unit and scaling conversions.
        outputs : list of output <Vector> objects
            List of output vectors to apply the unit and scaling conversions.
        residuals : list of residual <Vector> objects
            List of residual vectors to apply the unit and scaling conversions.
        scale_jac : bool
            If True, scale the Jacobian as well.
        """
        scal_vecs = self._scaling_vecs

        for vec in inputs:
            self._scale_vec(vec, 'input', 'phys')
        for vec in outputs:
            self._scale_vec(vec, 'output', 'phys')
        for vec in residuals:
            self._scale_vec(vec, 'residual', 'phys')
        if scale_jac:
            self._jacobian._precompute_iter()
            self._jacobian._scale(self._scaling_to_phys)

        yield

        for vec in inputs:
            self._scale_vec(vec, 'input', 'norm')
        for vec in outputs:
            self._scale_vec(vec, 'output', 'norm')
        for vec in residuals:
            self._scale_vec(vec, 'residual', 'norm')
        if scale_jac:
            self._jacobian._precompute_iter()
            self._jacobian._scale(self._scaling_to_norm)

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

        rel_out = self._relevant_vars_out[vec_name]
        rel_in = self._relevant_vars_in[vec_name]

        res_names = set(self._varx_abs_names['output']) & rel_out
        if scope_out is None:
            out_names = set(self._varx_abs_names['output']) & rel_out
        else:
            out_names = set(self._varx_abs_names['output']) & rel_out & scope_out
        if scope_in is None:
            in_names = set(self._varx_abs_names['input']) & rel_in
        else:
            in_names = set(self._varx_abs_names['input']) & rel_in & scope_in

        d_inputs._names = in_names
        d_outputs._names = out_names
        d_residuals._names = res_names

        yield d_inputs, d_outputs, d_residuals

        # reset _names so users will see full vector contents
        d_inputs._names = d_inputs._views
        d_outputs._names = d_outputs._views
        d_residuals._names = d_residuals._views

    @contextmanager
    def nonlinear_vector_context(self):
        """
        Context manager that yields the inputs, outputs, and residuals vectors.

        Yields
        ------
        (inputs, outputs, residuals) : tuple of <Vector> instances
            Yields the inputs, outputs, and residuals nonlinear vectors.
        """
        if self._inputs is None:
            raise RuntimeError("Cannot get vectors because setup has not yet been called.")

        yield self._inputs, self._outputs, self._residuals

    @contextmanager
    def linear_vector_context(self, vec_name='linear'):
        """
        Context manager that yields linear inputs, outputs, and residuals vectors.

        Parameters
        ----------
        vec_name : str
            Name of the linear right-hand-side vector. The default is 'linear'.

        Yields
        ------
        (inputs, outputs, residuals) : tuple of <Vector> instances
            Yields the inputs, outputs, and residuals linear vectors for vec_name.
        """
        if self._inputs is None:
            raise RuntimeError("Cannot get vectors because setup has not yet been called.")

        if vec_name not in self._vectors['input']:
            raise ValueError("There is no linear vector named %s" % vec_name)

        yield (self._vectors['input'][vec_name],
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
        self._jacobian._precompute_iter()
        yield self._jacobian
        self._jacobian._system = oldsys

    @contextmanager
    def _scaled_context(self):
        """
        Context manager that temporarily puts all vectors and Jacobians in a scaled state.
        """
        self._scale_vectors_and_jacobians('to norm')
        yield
        self._scale_vectors_and_jacobians('to phys')

    def _scale_vectors_and_jacobians(self, direction):
        """
        Scale all vectors and Jacobians to or from a scaled state.

        Parameters
        ----------
        direction : str
            'to norm' (to scaled) or 'to phys' (to unscaled).
        """
        if direction == 'to norm':
            scaling = self._scaling_to_norm
        elif direction == 'to phys':
            scaling = self._scaling_to_phys

        for vec_type in ['input', 'output', 'residual']:
            for vec in self._vectors[vec_type].values():
                self._scale_vec(vec, vec_type, direction[3:])

        for system in self.system_iter(include_self=True, recurse=True):
            if system._owns_assembled_jac:
                with system.jacobian_context():
                    system._jacobian._precompute_iter()
                    system._jacobian._scale(scaling)

    @property
    def nl_solver(self):
        """
        The nonlinear solver for this system.
        """
        return self._nl_solver

    @nl_solver.setter
    def nl_solver(self, solver):
        """
        Set this system's nonlinear solver and perform setup.
        """
        self._nl_solver = solver

    @property
    def ln_solver(self):
        """
        The linear (adjoint) solver for this system.
        """
        return self._ln_solver

    @ln_solver.setter
    def ln_solver(self, solver):
        """
        Set this system's linear (adjoint) solver and perform setup.
        """
        self._ln_solver = solver

    @property
    def suppress_solver_output(self):
        """
        The value of the global toggle to disable solver printing.
        """
        return self._suppress_solver_output

    @suppress_solver_output.setter
    def suppress_solver_output(self, value):
        """
        Recursively set the solver print suppression toggle.
        """
        self._suppress_solver_output = value
        # loop over _subsystems_allprocs here because _subsystems_myprocs
        # is empty until setup
        for subsys in self._subsystems_allprocs:
            subsys.suppress_solver_output = value

    @property
    def proc_allocator(self):
        """
        The current system's processor allocator object.
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
        A generator of subsystems of this system.

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
                       ref0=None, indices=None, adder=None, scaler=None,
                       **kwargs):
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
            interest for this particular response.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        **kwargs : optional
            Keyword arguments that are saved as metadata for the
            design variable.

        Notes
        -----
        The response can be scaled using ref and ref0.
        The argument :code:`ref0` represents the physical value when the scaled value is 0.
        The argument :code:`ref` represents the physical value when the scaled value is 1.
        """
        if name in self._design_vars:
            msg = "Design Variable '{}' already exists."
            raise RuntimeError(msg.format(name))

        # Name must be a string
        if not isinstance(name, string_types):
            raise TypeError('The name argument should be a string, got {0}'.format(name))

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

        meta = kwargs if kwargs else None
        self._design_vars[name] = dvs = OrderedDict()
        dvs['name'] = name
        dvs['upper'] = upper
        dvs['lower'] = lower
        dvs['scaler'] = None if scaler == 1.0 else scaler
        dvs['adder'] = None if adder == 0.0 else adder
        dvs['ref'] = ref
        dvs['ref0'] = ref0
        dvs['indices'] = indices
        dvs['metadata'] = meta

    def add_response(self, name, type, lower=None, upper=None, equals=None,
                     ref=None, ref0=None, indices=None, adder=None, scaler=None,
                     linear=False, **kwargs):
        r"""
        Add a response variable to this system.

        Parameters
        ----------
        name : string
            Name of the response variable in the system.
        type : string
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
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        linear : bool
            Set to True if constraint is linear. Default is False.
        **kwargs : optional
            Keyword arguments that are saved as metadata for the
            design variable.

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
        if not isinstance(type, string_types):
            raise TypeError('The type argument should be a string')
        elif type not in ('con', 'obj'):
            raise ValueError('The type must be one of \'con\' or \'obj\': '
                             'Got \'{0}\' instead'.format(name))

        if name in self._responses:
            typemap = {'con': 'Constraint', 'obj': 'Objective'}
            msg = '{0} \'{1}\' already exists.'.format(typemap[type], name)
            raise RuntimeError(msg.format(name))

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

        # Currently ref and ref0 must be scalar
        if ref is not None:
            ref = float(ref)

        if ref0 is not None:
            ref0 = float(ref0)

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

        meta = kwargs if kwargs else None
        self._responses[name] = resp = OrderedDict()
        resp['name'] = name
        resp['scaler'] = None if scaler == 1.0 else scaler
        resp['adder'] = None if adder == 0.0 else adder
        resp['ref'] = ref
        resp['ref0'] = ref0
        resp['indices'] = indices
        resp['metadata'] = meta
        resp['type'] = type

        if type == 'con':
            resp['lower'] = lower
            resp['upper'] = upper
            resp['equals'] = equals
            resp['linear'] = linear

        elif type == 'obj':
            pass
        else:
            raise ValueError('Unrecognized type for response.  Expected'
                             ' one of [\'obj\', \'con\']:  ({0})'.format(type))

    def add_constraint(self, name, lower=None, upper=None, equals=None,
                       ref=None, ref0=None, adder=None, scaler=None,
                       indices=None, linear=False, **kwargs):
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
            interest for this particular response.
        linear : bool
            Set to True if constraint is linear. Default is False.
        **kwargs : optional
            Keyword arguments that are saved as metadata for the
            design variable.

        Notes
        -----
        The response can be scaled using ref and ref0.
        The argument :code:`ref0` represents the physical value when the scaled value is 0.
        The argument :code:`ref` represents the physical value when the scaled value is 1.
        """
        meta = kwargs if kwargs else None

        self.add_response(name=name, type='con', lower=lower, upper=upper,
                          equals=equals, scaler=scaler, adder=adder, ref=ref,
                          ref0=ref0, indices=indices, linear=linear, metadata=meta)

    def add_objective(self, name, ref=None, ref0=None, indices=None,
                      adder=None, scaler=None, **kwargs):
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
        indices : sequence of int, optional
            If variable is an array, these indicate which entries are of
            interest for this particular response.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        **kwargs : optional
            Keyword arguments that are saved as metadata for the
            design variable.

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
        meta = kwargs if kwargs else None
        if 'lower' in kwargs or 'upper' in kwargs or 'equals' in kwargs:
            raise RuntimeError('Bounds may not be set on objectives')
        self.add_response(name, type='obj', scaler=scaler, adder=adder,
                          ref=ref, ref0=ref0, indices=indices, metadata=meta)

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

            # Depending on where the designvar was added, the name in the
            # vectors might be relative instead of absolute. Lucky we have
            # both.
            if name in vec:
                out[name]['size'] = vec[name].size
            else:
                out[name]['size'] = vec[out[name]['name']].size

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
        with self._scaled_context():
            self._apply_nonlinear()

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
        with self._scaled_context():
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
        with self._scaled_context():
            self._apply_linear(vec_names, mode, scope_out, scope_int)

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
        with self._scaled_context():
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
        with self._scaled_context():
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
            relative error.
        float
            absolute error.
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

    def initialize_processors(self):
        """
        Optional user-defined method run after repartitioning/rebalancing.

        Available attributes:
            name
            pathname
            comm
            metadata (local and global)
        """
        pass

    def initialize_variables(self):
        """
        Required method for components to declare inputs and outputs.

        Available attributes:
            name
            pathname
            comm
            metadata (local and global)
        """
        pass

    def initialize_partials(self):
        """
        Optional method for components to declare Jacobian structure/approximations.

        Available attributes:
            name
            pathname
            comm
            metadata (local and global)
            variable names
        """
        pass
