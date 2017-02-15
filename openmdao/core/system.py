"""Define the base System class."""
from __future__ import division

from fnmatch import fnmatchcase
from contextlib import contextmanager
from collections import namedtuple, OrderedDict, Iterable
import numbers
import sys

import numpy

from six import string_types
from six.moves import range

from openmdao.proc_allocators.default_allocator import DefaultAllocator
from openmdao.jacobians.default_jacobian import DefaultJacobian
from openmdao.jacobians.global_jacobian import GlobalJacobian
from openmdao.utils.generalized_dict import GeneralizedDictionary
from openmdao.utils.class_util import overrides_method
from openmdao.utils.units import convert_units

# This is for storing various data mapped to var pathname
PathData = namedtuple("PathData", ['name', 'idx', 'myproc_idx', 'typ'])

DesignVariable = namedtuple('DesignVariable', ['name', 'lower', 'upper',
                                               'scaler', 'adder', 'ref',
                                               'ref0', 'indices', 'metadata'])

Constraint = namedtuple('Constraint', ['name', 'lower', 'upper', 'equals',
                                       'scaler', 'adder', 'ref', 'ref0',
                                       'indices', 'metadata'])

Objective = namedtuple('Objective', ['name', 'scaler', 'adder', 'ref',
                                     'ref0', 'indices', 'metadata'])


def _format_driver_array_option(option_name, var_name, values,
                                val_if_none=0.0):
    """
    Format driver array option values.

    Checks that the given array values are either None, float, or an
    iterable of numeric values.  On output all interables of numeric values
    are converted to numpy.ndarray.  If values is scalar, it is converted
    to float.

    Parameters
    ----------
    option_name : str
        Name of the option being set
    var_name : str
        The path of the variable relative to the current system.
    values : float or numpy ndarray or Iterable
        Values of the array option to be formatted to the expected form.
    val_if_none : float or numpy ndarray
        The default value for the option if values is None.

    Returns
    -------
    float or numpy.ndarray
        Values transformed to the expected form.

    Raises
    ------
    ValueError
        If values is Iterable but cannot be converted to a numpy ndarray
    TypeError
        If values is scalar, not None, and not a Number.
    """
    # Convert adder to ndarray/float as necessary
    if isinstance(values, numpy.ndarray):
        pass
    elif not isinstance(values, string_types) \
            and isinstance(values, Iterable):
        values = numpy.asarray(values, dtype=float)
    elif values is None:
        values = val_if_none
    elif isinstance(values, numbers.Number):
        values = float(values)
    else:
        raise TypeError('Expected values of {0} to be an Iterable of '
                        'numeric values, or a scalar numeric value. '
                        'Got {1} instead.'.format(option_name, values))
    return values


class System(object):
    """
    Base class for all systems in OpenMDAO.

    Never instantiated; subclassed by <Group> or <Component>.
    All subclasses have their attributes defined here.

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
    _assembler : <Assembler>
        pointer to the global assembler object.
    _mpi_proc_allocator : <ProcAllocator>
        object that distributes procs among subsystems.
    _mpi_proc_range : [int, int]
        indices of procs owned by comm with respect to COMM_WORLD.
    _subsystems_allprocs : [<System>, ...]
        list of all subsystems (children of this system).
    _subsystems_myproc : [<System>, ...]
        list of local subsystems that exist on this proc.
    _subsystems_myproc_inds : [int, ...]
        list of indices of subsystems on this proc among all of this system's
        subsystems (subsystems on all of this system's processors).
    _var_allprocs_names : {'input': [str, ...], 'output': [str, ...]}
        list of promoted names of all owned variables, not just on current proc.
    _var_allprocs_pathnames : {'input': [str, ...], 'output': [str, ...]}
        list of pathnames of all owned variables, not just on current proc.
    _var_allprocs_range : {'input': [int, int], 'output': [int, int]}
        index range of owned variables with respect to all problem variables.
    _var_allprocs_indices : {'input': dict, 'output': dict}
        dictionary of global indices keyed by the variable name.
    _var_myproc_names : {'input': [str, ...], 'output': [str, ...]}
        list of unpromoted names of owned variables on current proc.
    _var_myproc_metadata : {'input': list, 'output': list}
        list of metadata dictionaries of variables that exist on this proc.
    _var_pathdict : dict
        maps full variable pathname to local name, index and I/O type
    _var_name2path = dict
        maps local var name to full pathname
    _var_myproc_indices : {'input': ndarray[:], 'output': ndarray[:]}
        integer arrays of global indices of variables on this proc.
    _var_maps : {'input': dict, 'output': dict}
        dictionary of variable names and their aliases (for promotes/renames).
    _var_promotes : { 'any': set(), 'input': set(), 'output': set() }
        dictionary of sets of variable names/wildcards specifying promotion
        (used to calculate _var_maps)
    _var_renames : { 'input': {}, 'output': {} }
        dictionary of mappings used to specify variables to be renamed in the
        parent group. (used to calculate _var_maps)
    _var_connections : dict
        dictionary of input_name: (output_name, src_indices) connections.
    _var_connections_indices : [(int, int), ...]
        _var_connections with variable indices instead of names.  Entries
        have the form (input_index, output_index).
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
    _owns_global_jac : bool
        If True, we are owners of the GlobalJacobian in self._jacobian.
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

        self._assembler = None

        self._mpi_proc_allocator = DefaultAllocator()
        self._mpi_proc_range = [0, 1]

        self._subsystems_allprocs = []
        self._subsystems_myproc = []
        self._subsystems_myproc_inds = []

        self._var_allprocs_names = {'input': [], 'output': []}
        self._var_allprocs_pathnames = {'input': [], 'output': []}
        self._var_allprocs_range = {'input': [0, 0], 'output': [0, 0]}
        self._var_allprocs_indices = {'input': {}, 'output': {}}

        self._var_myproc_names = {'input': [], 'output': []}
        self._var_myproc_metadata = {'input': [], 'output': []}
        self._var_myproc_indices = {'input': None, 'output': None}

        self._var_pathdict = {}
        self._var_name2path = {'input': {}, 'output': {}}

        self._var_maps = {'input': {}, 'output': {}}
        self._var_promotes = {'input': set(), 'output': set(), 'any': set()}
        self._var_renames = {'input': {}, 'output': {}}

        self._var_connections = {}
        self._var_connections_indices = []

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

        self._jacobian = DefaultJacobian()
        self._jacobian._system = self
        self._jacobian_changed = True
        self._owns_global_jac = False

        self._subjacs_info = OrderedDict()

        self._nl_solver = None
        self._ln_solver = None
        self._suppress_solver_output = False

        self._design_vars = {}
        self._responses = {}

        self.initialize()

    def _setup_processors(self, path, comm, global_dict,
                          assembler, proc_range):
        """
        Recursively split comms and define local subsystems.

        Sets the following attributes:
            pathname
            comm
            _assembler
            _mpi_proc_range
            _subsystems_myproc
            _subsystems_myproc_inds

        Parameters
        ----------
        path : str
            parent names to prepend to name to get the pathname
        comm : MPI.Comm or <FakeComm>
            communicator for this system (already split, if applicable).
        global_dict : dict
            dictionary with kwargs of all parents assembled in it.
        assembler : Assembler
            pointer to the global assembler object to distribute to everyone.
        proc_range : [int, int]
            indices of procs owned by comm with respect to COMM_WORLD.
        """
        # Set attributes
        self.pathname = '.'.join((path, self.name)) if path else self.name
        self.comm = comm
        self._assembler = assembler
        self._mpi_proc_range = proc_range

        # Add self's kwargs to dictionary of parents' kwargs (already new copy)
        self.metadata._assemble_global_dict(global_dict)

        # Optional user-defined method
        self.initialize_processors()

        nsub = len(self._subsystems_allprocs)
        # If this is a group:
        if nsub > 0:
            # Call the load balancing algorithm
            tmp = self._mpi_proc_allocator(nsub, comm, proc_range)
            sub_inds, sub_comm, sub_proc_range = tmp

            # Define local subsystems
            self._subsystems_myproc = [self._subsystems_allprocs[ind]
                                       for ind in sub_inds]
            self._subsystems_myproc_inds = sub_inds

            # Perform recursion
            for subsys in self._subsystems_myproc:
                sub_global_dict = self.metadata._global_dict.copy()
                subsys._setup_processors(self.pathname, sub_comm,
                                         sub_global_dict, assembler,
                                         sub_proc_range)

    def _setup_variables(self, recurse=True):
        """
        Assemble variable metadata and names lists.

        Sets the following attributes:
            _var_allprocs_names
            _var_myproc_names
            _var_myproc_metadata

        Parameters
        ----------
        recurse : boolean
            recursion is not performed if traversing up the tree after reconf.
        """
        # Perform recursion
        if recurse:
            for subsys in self._subsystems_myproc:
                subsys._setup_variables()

        if overrides_method('initialize_variables', self, System):
            # TODO: we may want to provide a way for component devs to tell
            # the framework that they don't need to re-configure, since the
            # majority of components won't need to be configured more than once

            # Empty the lists in case this is part of a reconfiguration
            for typ in ['input', 'output']:
                self._var_allprocs_names[typ] = []
                self._var_allprocs_pathnames[typ] = []
                self._var_myproc_names[typ] = []
                self._var_myproc_metadata[typ] = []

            self.initialize_variables()

    def _setup_variable_indices(self, global_index, recurse=True):
        """
        Define the variable indices and range.

        Sets the following attributes:
            _var_allprocs_range
            _var_allprocs_indices
            _var_myproc_indices

        Parameters
        ----------
        global_index : {'input': int, 'output': int}
            current global variable counter.
        recurse : boolean
            recursion is not performed if traversing up the tree after reconf.
        """
        # Define the global variable range for the system
        for typ in ['input', 'output']:
            size = len(self._var_allprocs_names[typ])
            self._var_allprocs_range[typ][0] = global_index[typ]
            self._var_allprocs_range[typ][1] = global_index[typ] + size

        # If group, compute _var_myproc_indices as follows
        if len(self._subsystems_myproc) > 0:
            subsys0 = self._subsystems_myproc[0]

            # Pre-recursion: compute 'index' to pass to subsystems
            # Need offset: number of variables on procs before current proc
            # Necessary because of multiple global counters on different procs
            if self.comm.size > 1:
                for typ in ['input', 'output']:
                    local_var_size = len(subsys0._var_allprocs_names[typ])

                    # Compute the variable count list; 0 on rank > 0 procs
                    sub_comm = subsys0.comm
                    if sub_comm.rank == 0:
                        nvar_myproc = local_var_size
                    else:
                        nvar_myproc = 0
                    nvar_allprocs = self.comm.allgather(nvar_myproc)

                    # Compute the offset
                    iproc = self.comm.rank
                    nvar_myproc = local_var_size
                    global_index[typ] += \
                        numpy.sum(nvar_allprocs[:iproc + 1]) - nvar_myproc

            # Perform the recursion
            if recurse:
                for subsys in self._subsystems_myproc:
                    subsys._setup_variable_indices(global_index)

            # Post-recursion: assemble local variable indices from subsystems
            for typ in ['input', 'output']:
                raw = []
                for subsys in self._subsystems_myproc:
                    raw.append(subsys._var_myproc_indices[typ])
                self._var_myproc_indices[typ] = numpy.concatenate(raw)

        # If component, _var_myproc_indices is simply an arange
        else:
            for typ in ['input', 'output']:
                ind1, ind2 = self._var_allprocs_range[typ]
                self._var_myproc_indices[typ] = numpy.arange(ind1, ind2)

        # Reset index dict to the global variable count on all procs
        # Necessary for younger siblings to have proper index values
        for typ in ['input', 'output']:
            global_index[typ] = self._var_allprocs_range[typ][1]

        # Populate the _var_allprocs_indices dictionary
        for typ in ['input', 'output']:
            idx = self._var_allprocs_range[typ][0]
            for name in self._var_allprocs_names[typ]:
                self._var_allprocs_indices[typ][name] = idx
                idx += 1

    def _setup_connections(self):
        """
        Recursively assemble a list of input-output connections.

        Overridden in <Group>.
        """
        pass

    def _setup_vector(self, vectors, vector_var_ids, use_ref_vector):
        """
        Add this vector and assign sub_vectors to subsystems.

        Sets the following attributes:
            _vectors
            _vector_transfers
            _inputs*
            _outputs*
            _residuals*
            _transfers*

        * If vec_name is None - i.e., we are setting up the nonlinear vector

        Parameters
        ----------
        vectors : {'input': <Vector>, 'output': <Vector>, 'residual': <Vector>}
            Vector objects corresponding to 'name'.
        vector_var_ids : ndarray[:]
            integer array of all relevant variables for this vector.
        use_ref_vector : bool
            if True, allocate vectors to store ref. values.
        """
        vec_name = vectors['output']._name

        # Set the incoming _vectors in the appropriate attribute
        for key in ['input', 'output', 'residual']:
            self._vectors[key][vec_name] = vectors[key]

        if use_ref_vector:
            vectors['input']._compute_ivar_map()
            vectors['output']._compute_ivar_map()
            vectors['residual']._ivar_map = vectors['output']._ivar_map

        # Compute the transfer for this vector set
        self._vector_transfers[vec_name] = self._get_transfers(vectors)

        # Assign relevant variables IDs array
        self._vector_var_ids[vec_name] = vector_var_ids

        # Define shortcuts for convenience
        if vec_name is 'nonlinear':
            self._inputs = self._vectors['input']['nonlinear']
            self._outputs = self._vectors['output']['nonlinear']
            self._residuals = self._vectors['residual']['nonlinear']
            self._transfers = self._vector_transfers['nonlinear']

        # Perform recursion
        for subsys in self._subsystems_myproc:

            sub_vectors = {}
            for key in ('input', 'output', 'residual'):
                sub_vectors[key] = vectors[key]._create_subvector(subsys)

            subsys._setup_vector(sub_vectors, vector_var_ids, use_ref_vector)

    def _setup_scaling(self):
        """
        Set up scaling vectors.
        """
        nvar_in = len(self._var_myproc_metadata['input'])
        nvar_out = len(self._var_myproc_metadata['output'])

        # Initialize scaling arrays
        for scaling in (self._scaling_to_norm, self._scaling_to_phys):
            scaling['input'] = numpy.empty((nvar_in, 2))
            scaling['output'] = numpy.empty((nvar_out, 2))
            scaling['residual'] = numpy.empty((nvar_out, 2))

        # ref0 and ref are the values of the variable in the specified
        # units at which the scaled values are 0 and 1, respectively

        # Scaling coefficients from the src output
        src_units = self._assembler._src_units
        src_scaling = self._assembler._src_scaling

        # Compute scaling arrays for inputs using a0 and a1
        # Example:
        #   Let x, x_src, x_tgt be the variable in dimensionless, source, and target units, resp.
        #   x_src = a0 + a1 x
        #   x_tgt = b0 + b1 x
        #   x_tgt = g(x_src) = d0 + d1 x_src
        #   b0 + b1 x = d0 + d1 a0 + d1 a1 x
        #   b0 = d0 + d1 a0
        #   b0 = g(a0)
        #   b1 = d0 + d1 a1 - d0
        #   b1 = g(a1) - g(0)
        for ind, meta in enumerate(self._var_myproc_metadata['input']):
            global_ind = self._var_myproc_indices['input'][ind]
            self._scaling_to_phys['input'][ind, 0] = \
                convert_units(src_scaling[global_ind, 0], src_units[global_ind], meta['units'])
            self._scaling_to_phys['input'][ind, 1] = \
                convert_units(src_scaling[global_ind, 1], src_units[global_ind], meta['units']) - \
                convert_units(0., src_units[global_ind], meta['units'])

        for ind, meta in enumerate(self._var_myproc_metadata['output']):
            # Compute scaling arrays for outputs; no unit conversion needed
            self._scaling_to_phys['output'][ind, 0] = meta['ref0']
            self._scaling_to_phys['output'][ind, 1] = meta['ref'] - meta['ref0']

            # Compute scaling arrays for residuals; convert units
            self._scaling_to_phys['residual'][ind, 0] = meta['res_ref0']
            self._scaling_to_phys['residual'][ind, 1] = meta['res_ref'] - meta['res_ref0']

        # Compute inverse scaling arrays
        for key in ['input', 'output', 'residual']:
            a = self._scaling_to_phys[key][:, 0]
            b = self._scaling_to_phys[key][:, 1]
            self._scaling_to_norm[key][:, 0] = -a / b
            self._scaling_to_norm[key][:, 1] = 1.0 / b

        for subsys in self._subsystems_myproc:
            subsys._setup_scaling()

    def _setup_bounds_vectors(self, lower_bounds, upper_bounds, is_top):
        """
        Set up the lower and upper bounds vectors.

        Sets the following attributes:
            _lower_bounds
            _upper_bounds

        Parameters
        ----------
        lower_bounds : <Vector>
            lower bound vector allocated in <Problem>.
        upper_bounds : <Vector>
            upper bound vector allocated in <Problem>.
        is_top : bool
            whether this system is the root system.
        """
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

        # if this is the top-most group, we will set the values here as well.
        if is_top:
            for ind, meta in enumerate(self._var_myproc_metadata['output']):
                name = self._var_myproc_names['output'][ind]
                a, b = self._scaling_to_norm['output'][ind, :]

                # We have to convert from physical, unscaled to scaled, dimensionless.
                # We set into the bounds vector first and then apply a and b because
                # meta['lower'] and meta['upper'] could be lists or tuples.
                if meta['lower'] is None:
                    self._lower_bounds[name] = -numpy.inf
                else:
                    self._lower_bounds[name] = meta['lower']
                    self._lower_bounds[name] = a + b * self._lower_bounds[name]
                if meta['upper'] is None:
                    self._upper_bounds[name] = numpy.inf
                else:
                    self._upper_bounds[name] = meta['upper']
                    self._upper_bounds[name] = a + b * self._upper_bounds[name]

        # Perform recursion
        for subsys in self._subsystems_myproc:
            sub_lower_bounds = lower_bounds._create_subvector(subsys)
            sub_upper_bounds = upper_bounds._create_subvector(subsys)
            subsys._setup_bounds_vectors(sub_lower_bounds, sub_upper_bounds, False)

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
        self._owns_global_jac = isinstance(jacobian, GlobalJacobian)
        self._jacobian = jacobian
        self._jacobian_changed = True

    def _setup_jacobians(self, jacobian=None):
        """
        Set and populate jacobians down through the system tree.

        Parameters
        ----------
        jacobian : <GlobalJacobian> or None
            The global jacobian to populate for this system.
        """
        self._jacobian_changed = False

        if self._owns_global_jac:
            jacobian = self._jacobian
        elif jacobian is not None:
            self._jacobian = jacobian

        self._set_partials_meta()

        for subsys in self._subsystems_myproc:
            subsys._setup_jacobians(jacobian)

        if self._owns_global_jac:
            self._jacobian._system = self
            self._jacobian._initialize()

    def _get_transfers(self, vectors):
        """
        Compute transfers.

        Parameters
        ----------
        vectors : {'input': Vector, 'output': Vector, 'residual': Vector}
            dictionary of <Vector> objects

        Returns
        -------
        dict of <Transfer>
            dictionary of full and partial Transfer objects.
        """
        transfer_class = vectors['output'].TRANSFER

        nsub_allprocs = len(self._subsystems_allprocs)
        var_range = self._var_allprocs_range
        subsystems_myproc = self._subsystems_myproc
        subsystems_inds = self._subsystems_myproc_inds

        # Call the assembler's transfer setup routine
        compute_transfers = self._assembler._compute_transfers
        xfer_indices = compute_transfers(nsub_allprocs, var_range,
                                         subsystems_myproc, subsystems_inds)
        (xfer_in_inds, xfer_out_inds,
         fwd_xfer_in_inds, fwd_xfer_out_inds,
         rev_xfer_in_inds, rev_xfer_out_inds) = xfer_indices

        # Create Transfer objects from the raw indices
        transfers = {}
        transfers[None] = transfer_class(vectors['input'], vectors['output'],
                                         xfer_in_inds, xfer_out_inds, self.comm)
        for isub in range(len(fwd_xfer_in_inds)):
            transfers['fwd', isub] = transfer_class(vectors['input'],
                                                    vectors['output'],
                                                    fwd_xfer_in_inds[isub],
                                                    fwd_xfer_out_inds[isub],
                                                    self.comm)
        for isub in range(len(rev_xfer_in_inds)):
            transfers['rev', isub] = transfer_class(vectors['input'],
                                                    vectors['output'],
                                                    rev_xfer_in_inds[isub],
                                                    rev_xfer_out_inds[isub],
                                                    self.comm)
        return transfers

    def _get_maps(self, typ):
        """
        Define variable maps based on promotes and renames lists.

        Parameters
        ----------
        typ : str
            Either 'input' or 'output'.

        Returns
        -------
        dict of {str:str, ...}
            dictionary mapping input/output variable names
            to promoted or renamed variable names.
        """
        maps = {}

        gname = self.name + '.' if self.name else ''

        promotes = self._var_promotes['any']
        promotes_typ = self._var_promotes[typ]
        renames = self._var_renames[typ]

        if promotes:
            names = promotes
            patterns = [n for n in names if '*' in n or '?' in n]
        elif promotes_typ:
            names = promotes_typ
            patterns = [n for n in names if '*' in n or '?' in n]
        else:
            names = ()
            patterns = ()

        for name in self._var_allprocs_names[typ]:
            if name in names:
                maps[name] = name
                continue

            for pattern in patterns:
                # if name matches, promote that variable to parent
                if fnmatchcase(name, pattern):
                    maps[name] = name
                    break
            else:
                if name in renames:
                    # Rename selected variables in the parent system
                    maps[name] = renames[name]
                else:
                    # Default: prepend the parent system's name
                    maps[name] = gname + name if gname else name

        return maps

    @contextmanager
    def _jacobian_context(self):
        """
        Context manager for jacobians.

        Sets this system's _jacobian._system attribute to the current system.

        Returns
        -------
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
        for vec in inputs:
            vec._scale(self._scaling_to_phys['input'])
        for vec in outputs:
            vec._scale(self._scaling_to_phys['output'])
        for vec in residuals:
            vec._scale(self._scaling_to_phys['residual'])
        if scale_jac:
            self._jacobian._precompute_iter()
            self._jacobian._scale(self._scaling_to_phys)

        yield

        for vec in inputs:
            vec._scale(self._scaling_to_norm['input'])
        for vec in outputs:
            vec._scale(self._scaling_to_norm['output'])
        for vec in residuals:
            vec._scale(self._scaling_to_norm['residual'])
        if scale_jac:
            self._jacobian._precompute_iter()
            self._jacobian._scale(self._scaling_to_norm)

    @contextmanager
    def _matvec_context(self, vec_name, var_inds, mode, clear=True):
        """
        Context manager for vectors.

        For the given vec_name, return vectors that use a set of
        internal variables that are relevant to the current matrix-vector
        product.  This is called only from _apply_linear.

        Parameters
        ----------
        vec_name : str
            Name of the vector to use.
        var_inds : [int, int, int, int] or None
            ranges of variable IDs involved in this matrix-vector product.
            The ordering is [lb1, ub1, lb2, ub2].
        mode : str
            Key for specifying derivative direction. Values are 'fwd'
            or 'rev'.
        clear : bool(True)
            If True, zero out residuals (in fwd mode) or inputs and outputs
            (in rev mode).

        Returns
        -------
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

        out_names = []
        res_names = []
        var_ids = self._vector_var_ids[vec_name]
        out_ind = self._var_allprocs_range['output'][0]
        for out_name in self._var_allprocs_names['output']:
            if out_ind in var_ids:
                res_names.append(out_name)
                if var_inds is None or (var_inds[0] <= out_ind < var_inds[1] or
                                        var_inds[2] <= out_ind < var_inds[3]):
                    out_names.append(out_name)
            out_ind += 1

        in_names = []
        in_ind = self._var_allprocs_range['input'][0]
        for in_name in self._var_allprocs_names['input']:
            out_ind = self._assembler._input_src_ids[in_ind]
            if out_ind in var_ids:
                if var_inds is None or (var_inds[0] <= out_ind < var_inds[1] or
                                        var_inds[2] <= out_ind < var_inds[3]):
                    in_names.append(in_name)
            in_ind += 1

        d_inputs._names = set(in_names)
        d_outputs._names = set(out_names)
        d_residuals._names = set(res_names)

        yield d_inputs, d_outputs, d_residuals

        # reset _names so users will see full vector contents
        d_inputs._names = d_inputs._views
        d_outputs._names = d_outputs._views
        d_residuals._names = d_residuals._views

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

    def get_input(self, name, units=None):
        """
        Return the named input value using the unpromoted name.

        Parameters
        ----------
        name : str
            name of the variable.
        units : str or None
            if not None, return the value in the given units.

        Returns
        -------
        float or ndarray
            The value of the named variable.
        """
        if units is not None:
            raise NotImplementedError("units arg not supported yet")
        if self._inputs is None:
            raise RuntimeError("%s: Cannot access input '%s'. Setup has not "
                               "been called." % (self.name, name))
        try:
            return self._inputs[name]
        except KeyError:
            raise KeyError("%s: input '%s' not found." % (self.pathname,
                                                          name))

    def set_input(self, name, value):
        """
        Set the value of the named input using the unpromoted name.

        Parameters
        ----------
        name : str
            name of the variable.
        value : float or ndarray
            value to be set.
        """
        if self._inputs is None:
            raise RuntimeError("%s: Cannot access input '%s'. Setup has not "
                               "been called." % (self.name, name))
        try:
            self._inputs[name] = value
        except KeyError:
            raise KeyError("%s: input '%s' not found." % (self.pathname,
                                                          name))

    def get_output(self, name, scaled=False, units=None):
        """
        Return the named output value using promoted or unpromoted name.

        Parameters
        ----------
        name : str
            name of the variable.
        scaled : bool
            If True, return the scaled value.
        units : str or None
            If not None, return the value in the given units.

        Returns
        -------
        float or ndarray
            The value of the named variable.
        """
        if scaled or units is not None:
            raise NotImplementedError("scaled and units args not supported yet")
        if self._outputs is None:
            raise RuntimeError("%s: Cannot access output '%s'. Setup has not "
                               "been called." % (self.name, name))
        try:
            return self._outputs[name]
        except KeyError:
            # check for promoted name
            start = len(self.pathname) + 1 if self.pathname else 0
            try:
                unprom = self._var_name2path['output'][name][start:]
                return self._outputs[unprom]
            except KeyError:
                raise KeyError("%s: output '%s' not found." % (self.pathname,
                                                               name))

    def set_output(self, name, value):
        """
        Return the named output value using promoted or unpromoted name.

        Parameters
        ----------
        name : str
            name of the variable.
        value : float or ndarray
            the value to be set.
        """
        if self._outputs is None:
            raise RuntimeError("%s: Cannot access output '%s'. Setup has not "
                               "been called." % (self.name, name))
        try:
            self._outputs[name] = value
        except KeyError:
            # check for promoted name
            start = len(self.pathname) + 1 if self.pathname else 0
            try:
                unprom = self._var_name2path['output'][name][start:]
                self._outputs[unprom] = value
            except KeyError:
                raise KeyError("%s: output '%s' not found." % (self.pathname,
                                                               name))

    def get_residual(self, name, scaled=False, units=None):
        """
        Return the named residual value using promoted or unpromoted name.

        Parameters
        ----------
        name : str
            name of the variable.
        scaled : bool
            If True, return the scaled value.
        units : str or None
            If not None, return the value in the given units.

        Returns
        -------
        float or ndarray
            The value of the named residual.
        """
        if scaled or units is not None:
            raise NotImplementedError("scaled and units args not supported yet")
        if self._residuals is None:
            raise RuntimeError("%s: Cannot access residual '%s'. Setup has not "
                               "been called." % (self.name, name))

        try:
            return self._residuals[name]
        except KeyError:
            # check for promoted name
            start = len(self.pathname) + 1 if self.pathname else 0
            try:
                unprom = self._var_name2path['output'][name][start:]
                return self._residuals[unprom]
            except KeyError:
                raise KeyError("%s: residual '%s' not found." % (self.pathname,
                                                                 name))

    def set_residual(self, name, value):
        """
        Set value of named residual using promoted or unpromoted name.

        Parameters
        ----------
        name : str
            name of the variable.
        value : float or ndarray
            the value to be set.
        """
        if self._residuals is None:
            raise RuntimeError("%s: Cannot access residual '%s'. Setup has not "
                               "been called." % (self.name, name))

        try:
            self._residuals[name] = value
        except KeyError:
            # check for promoted name
            start = len(self.pathname) + 1 if self.pathname else 0
            try:
                unprom = self._var_name2path['output'][name][start:]
                self._residuals[unprom] = value
            except KeyError:
                raise KeyError("%s: residual '%s' not found." % (self.pathname,
                                                                 name))

    def _apply_nonlinear(self):
        """
        Compute residuals.
        """
        pass

    def _solve_nonlinear(self):
        """
        Compute outputs.

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
        Compute jac-vec product.

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
        Apply inverse jac product.

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

    def _linearize(self):
        """
        Compute jacobian / factorization.
        """
        pass

    def initialize(self):
        """
        Optional user-defined method run once during instantiation.

        Available attributes:
            name
            metadata (only local)
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
        ref0 : upper or ndarray, optional
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
        The design variable can be scaled using scaler and adder, where

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
        if name in self._design_vars:
            msg = "Design Variable '{}' already exists."
            raise RuntimeError(msg.format(name))

        # Name must be a string
        if not isinstance(name, string_types):
            raise TypeError('The name argument should be a string, got {0}'.format(name))

        # Affine scaling cannot be used with scalers/adders
        if ref0 is not None or ref is not None:
            if scaler is not None or adder is not None:
                raise ValueError('Inputs ref/ref0 are mutually exclusive '
                                 'with scaler/adder')
            # Convert ref/ref0 to scaler/adder so we can scale the bounds
            adder = -ref0
            scaler = 1.0 / (ref + adder)
        else:
            if scaler is None:
                scaler = 1.0
            if adder is None:
                adder = 0.0

        # Convert adder to ndarray/float as necessary
        adder = _format_driver_array_option('adder', name, adder, val_if_none=0.0)

        # Convert scaler to ndarray/float as necessary
        scaler = _format_driver_array_option('scaler', name, scaler, val_if_none=1.0)

        # Convert lower to ndarray/float as necessary
        lower = _format_driver_array_option('lower', name, lower, val_if_none=-sys.float_info.max)

        # Convert upper to ndarray/float as necessary
        upper = _format_driver_array_option('upper', name, upper, val_if_none=sys.float_info.max)

        # Apply scaler/adder to lower and upper
        lower = (lower + adder) * scaler
        upper = (upper + adder) * scaler

        meta = kwargs if kwargs else None
        self._design_vars[name] = DesignVariable(name=name, lower=lower,
                                                 upper=upper, scaler=scaler,
                                                 adder=adder, ref=ref,
                                                 ref0=ref0, indices=indices,
                                                 metadata=meta)

    def add_response(self, name, type, lower=None, upper=None, equals=None,
                     ref=None, ref0=None, indices=None, adder=None, scaler=None,
                     **kwargs):
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
        **kwargs : optional
            Keyword arguments that are saved as metadata for the
            design variable.

        Notes
        -----
        The response can be scaled using scaler and adder, where

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

        # Affine scaling cannot be used with scalers/adders
        if ref0 is not None or ref is not None:
            if scaler is not None or adder is not None:
                raise ValueError('Inputs ref/ref0 are mutually exclusive '
                                 'with scaler/adder')
            # Convert ref/ref0 to scaler/adder so we can scale the bounds
            adder = -ref0
            scaler = 1.0 / (ref + adder)
        else:
            if scaler is None:
                scaler = 1.0
            if adder is None:
                adder = 0.0

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

        # Convert adder to ndarray/float as necessary
        adder = _format_driver_array_option('adder', name, adder, val_if_none=0.0)

        # Convert scaler to ndarray/float as necessary
        scaler = _format_driver_array_option('scaler', name, scaler, val_if_none=1.0)

        # Convert lower to ndarray/float as necessary
        lower = _format_driver_array_option('lower', name, lower, val_if_none=-sys.float_info.max)

        # Convert upper to ndarray/float as necessary
        upper = _format_driver_array_option('upper', name, upper, val_if_none=sys.float_info.max)

        # Convert equals to ndarray/float as necessary
        if equals is not None:
            equals = _format_driver_array_option('equals', name, equals)

        # Scale the bounds
        if lower is not None:
            lower = (lower + adder) * scaler

        if upper is not None:
            upper = (upper + adder) * scaler

        if equals is not None:
            equals = (equals + adder) * scaler

        meta = kwargs if kwargs else None
        if type == 'obj':
            self._responses[name] = Objective(name=name, scaler=scaler,
                                              adder=adder, ref=ref, ref0=ref0,
                                              indices=indices, metadata=meta)
        elif type == 'con':
            self._responses[name] = Constraint(name=name, lower=lower,
                                               upper=upper, equals=equals,
                                               scaler=scaler, adder=adder,
                                               ref=ref, ref0=ref0,
                                               indices=indices, metadata=meta)
        else:
            raise ValueError('Unrecognized type for response.  Expected'
                             ' one of [\'obj\', \'con\']:  ({0})'.format(type))

    def add_constraint(self, name, lower=None, upper=None, equals=None,
                       ref=None, ref0=None, adder=None, scaler=None,
                       indices=None, **kwargs):
        r"""
        Add a constraint variable to this system.

        Parameters
        ----------
        name : string
            Name of the response variable in the system.
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
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        indices : sequence of int, optional
            If variable is an array, these indicate which entries are of
            interest for this particular response.
        **kwargs : optional
            Keyword arguments that are saved as metadata for the
            design variable.

        Notes
        -----
        The constraint can be scaled using scaler and adder, where

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

        self.add_response(name=name, type='con', lower=lower, upper=upper,
                          equals=equals, scaler=scaler, adder=adder, ref=ref,
                          ref0=ref0, indices=indices, metadata=meta)

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
        out = self._design_vars.copy()
        if recurse:
            for subsys in self._subsystems_allprocs:
                subsys_design_vars = subsys.get_design_vars(recurse=recurse)
                for key in subsys_design_vars:
                    out[subsys.name + '.' + key] = subsys_design_vars[key]
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
        out = self._responses.copy()
        if recurse:
            for subsys in self._subsystems_allprocs:
                subsys_design_vars = subsys.get_responses(recurse=recurse)
                for key in subsys_design_vars:
                    out[subsys.name + '.' + key] = subsys_design_vars[key]
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
        return dict((key, response) for (key, response) in
                    self.get_responses(recurse=recurse).items() if isinstance(response, Constraint))

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
        return dict((key, response) for (key, response) in
                    self.get_responses(recurse=recurse).items() if isinstance(response, Objective))
