"""Define the base System class."""
from __future__ import division
import numpy

from Blue.proc_allocators.proc_allocator import DefaultProcAllocator
from Blue.solvers.solver import NonlinearBlockGS
from Blue.jacobians.jacobian import DefaultJacobian



class System(object):
    """Base class for all systems in OpenMDAO.

    Always subclassed by Group or Component, or a subclass thereof.
    All subclasses have their attributes defined here.

    Attributes
    ----------
    comm : MPI.Comm or FakeComm
        MPI communicator object.
    args : list of objects
        user-defined arguments (to be used in apply_nonlinear, ...).
    kwargs : dict of objects
        dictionary of user-defined arguments.
    global_kwargs : dict of objects
        kwargs combined with kwargs of parent systems.

    _sys_name : str
        name of the system, must be different from siblings.
    _sys_depth : int
        distance from the root node in the hierarchy tree.
    _sys_assembler: Assembler
        pointer to the global assembler object.

    _mpi_proc_allocator : ProcAllocator
        object that distributes procs among subsystems.
    _mpi_proc_range : [int, int]
        indices of procs owned by comm with respect to COMM_WORLD.

    _subsystems_allprocs : [System, ...]
        list of all subsystems (children of this system).
    _subsystems_myproc : [System, ...]
        list of local subsystems that exist on this proc.
    _subsystems_inds : [int, ...]
        list of indices of subsystems on this proc among all subsystems.

    _variable_allprocs_names : {'input': [str, ...], 'output': [str, ...]}
        list of names of all owned variables, not just on current proc.
    _variable_allprocs_range : {'input': [int,int], 'output': [int,int]}
        index range of owned variables with respect to all problem variables.

    _variable_myproc_names : {'input': [str, ...], 'output': [str, ...]}
        list of names of owned variables on current proc.
    _variable_myproc_metadata : {'input': list, 'output': list}
        list of metadata dictionaries of variables that exist on this proc.
    _variable_myproc_indices : {'input': ndarray[:], 'output': ndarray[:]}
        integer arrays of global indices of variables on this proc.

    _variable_maps : {'input': dict, 'output': dict}
        dict of variable names and their aliases (for promotes/renames).
    _variable_connections : dict
        dict of input:output connections between subsystems.
    _variable_connections_indices : [(int, int), ...]
        _variable_connections with variable indices instead of names.

    _vectors : {'input': dict, 'output': dict, 'residual': dict}
        dict of vector objects.
    _vector_transfers : dict
        dict of transfer objects.

    inputs : Vector
        inputs vector; points to _vectors['input'][None].
    outputs : Vector
        outputs vector; points to _vectors['output'][None].
    residuals : Vector
        residuals vector; points to _vectors['residual'][None].
    transfers : dict of Transfer
        transfer object; points to _vector_transfers[None].

    _jacobian : Jacobian
        global Jacobian object to be used in apply_linear

    _solvers_nonlinear : NonlinearSolver
        nonlinear solver to be used for solve_nonlinear.
    _solvers_linear : LinearSolver
        linear solver to be used for solve_linear; not the Newton system.
    _solvers_print : boolean
        global overriding flag that turns off all solver output if 'False'.
    """

    def __init__(self, name, *args, **kwargs):
        """Initialize all attributes.

        All subclasses use this __init__ method without overriding it.

        Args
        ----
        name : str
            system name.
        *args : list of arguments
            available in methods as self.args.
        **kwargs: dict of keyword arguments
            available here and in all descendants of this system.
        """
        self.comm = None
        self.args = args
        self.kwargs = kwargs
        self.global_kwargs = {}

        self._sys_name = name
        self._sys_depth = 0
        self._sys_assembler = None

        self._mpi_proc_allocator = DefaultProcAllocator()
        self._mpi_proc_range = None

        self._subsystems_allprocs = []
        self._subsystems_myproc = []
        self._subsystems_inds = []

        self._variable_allprocs_names = {'input': [], 'output': []}
        self._variable_allprocs_range = {'input': [0,0], 'output': [0,0]}

        self._variable_myproc_names = {'input': [], 'output': []}
        self._variable_myproc_metadata = {'input': [], 'output': []}
        self._variable_myproc_indices = {'input': None, 'output': None}

        self._variable_maps = {'input': {}, 'output': {}}
        self._variable_connections = {}
        self._variable_connections_indices = []

        self._vectors = {'input': {}, 'output': {}, 'residual': {}}
        self._vector_transfers = {}
        self._vector_IDs = {}

        self._inputs = None
        self._outputs = None
        self._residuals = None
        self._transfers = None

        self._jacobian = DefaultJacobian()

        self._solvers_nonlinear = NonlinearBlockGS()
        self._solvers_linear = NonlinearBlockGS() # temporary hack!
        self._solvers_print = True

    def get_subsystem(self, name):
        """Return the system called 'name' in the current namespace.

        Args
        ----
        name : str
            name of the desired system in the current namespace.

        Returns
        -------
        System or None
            System if found on this proc else None.
        """
        if name == self._sys_name:
            # If this system's name matches, target found
            return self
        else:
            ind = len(self._sys_name) + 1
            # If first part of name matches this system's name, check subsytems
            if name[:ind] == '%s.' % self._sys_name:
                for subsys in self._subsystems_myproc:
                    result = subsys.get_subsystem(name[ind:])
                    # If result is not None, target found; otherwise continue
                    if result is not None:
                        return result
                # All subsystems failed
                return None
            else:
                return None

    def set_solver_print(self, flag):
        """Recursively set solver print flag for this and all systems below.

        Args
        ----
        flag : boolean
            if False, solver printing is surpressed for this system and below.
        """
        self._solvers_print = flag
        for subsys in self._subsystems_myproc:
            subsys.set_solver_print(flag)

    def set_jacobian(self, jac=None):
        """Recursively set the system's jacobian attribute.

        Args
        ----
        jac : Jacobian or None
            Jacobian object to be set; if None, reset to the DefaultJacobian.
        """
        if jac is None:
            self._jacobian = DefaultJacobian()
            self._jacobian.setup(self)

        for subsys in self.subsystems_myproc:
            subsys.set_jacobian(jac)


    def _setup_processors(self, depth, assembler, global_kwargs, comm,
                         proc_range):
        """Recursively split comms and define local subsystems.

        Args
        ----
        depth : int
            depth level for this system - i.e., distance from root node.
        assembler : Assembler
            pointer to the global assember object to distribute to everyone.
        global_kwargs : dict
            dictionary with kwargs of all parents assembled in it.
        comm : MPI.Comm or FakeComm
            communicator for this system (already split, if applicable).
        proc_range : [int, int]
            indices of procs owned by comm with respect to COMM_WORLD.
        """
        # Set attributes
        self._sys_depth = depth
        self._sys_assembler = assembler
        self.global_kwargs = global_kwargs
        self.comm = comm
        self._mpi_proc_range = proc_range

        self.global_kwargs.update(self.kwargs)

        # Optional user-defined init method
        self.initialize(comm)

        nsub = len(self._subsystems_allprocs)
        if nsub > 0:
            # If this is a group, call the load balancing algorithm
            tmp = self._mpi_proc_allocator(nsub, comm, proc_range)
            sub_inds, sub_comm, sub_proc_range = tmp

            # Define local subsystems and perform recursion
            self._subsystems_myproc = [self._subsystems_allprocs[ind]
                                      for ind in sub_inds]
            self._subsystems_inds = sub_inds
            for subsys in self._subsystems_myproc:
                sub_global_kwargs = self.global_kwargs.copy()
                subsys._setup_processors(depth+1, assembler, sub_global_kwargs,
                                        sub_comm, sub_proc_range)

    def _setup_variables(self, recursion=True):
        """Assemble variable metadata and names lists.

        Args
        ----
        recursion : boolean
            recursion is not performed if traversing up the tree after reconf.
        """
        # Perform recursion
        if recursion:
            for subsys in self._subsystems_myproc:
                subsys._setup_variables()

        # Empty the lists in case this is part of a reconfiguration
        for typ in ['input', 'output']:
            self._variable_allprocs_names[typ] = []
            self._variable_myproc_names[typ] = []
            self._variable_myproc_metadata[typ] = []

        # If this is a component, the user calls add_input/add_output
        if len(self._subsystems_myproc) == 0:
            self.initialize_variables(self.comm)
        # If this is a group, assemble the metadata and names lists
        else:
            for typ in ['input', 'output']:
                for subsys in self._subsystems_myproc:
                    # Assemble the names list from subsystems
                    subsys._utils_compute_maps(typ)
                    for sub_name in subsys._variable_allprocs_names[typ]:
                        name = subsys._variable_maps[typ][sub_name]
                        self._variable_allprocs_names[typ].append(name)
                        self._variable_myproc_names[typ].append(name)

                    # Assemble the metadata list from the subsystems
                    metadata = subsys._variable_myproc_metadata[typ]
                    self._variable_myproc_metadata[typ].extend(metadata)

                # The names list is on all procs, allgather all names
                if self.comm.size > 1:

                    # One representative proc from each sub_comm adds names
                    sub_comm = self._subsystems_myproc[0].comm
                    if sub_comm.rank == 0:
                        names = self._variable_allprocs_names[typ]
                    else:
                        names = []

                    # Every proc on this comm now has global variable names
                    raw = self.comm.allgather(names)
                    self._variable_allprocs_names[typ] = []
                    for names in raw:
                        self._variable_allprocs_names[typ].extend(names)

    def _setup_variable_indices(self, index, recursion=True):
        """Define the variable indices (local) and range (global).

        Args
        ----
        index : int
            current global variable counter.
        recursion : boolean
            recursion is not performed if traversing up the tree after reconf.
        """
        # Define the global variable range for the system
        for typ in ['input', 'output']:
            size = len(self._variable_allprocs_names[typ])
            self._variable_allprocs_range[typ][0] = index[typ]
            self._variable_allprocs_range[typ][1] = index[typ] + size

        # If group, compute _variable_myproc_indices as follows
        if len(self._subsystems_myproc) > 0:
            subsys0 = self._subsystems_myproc[0]

            # Pre-recursion: compute 'index' to pass to subsystems
            # Need offset: number of variables on procs before current proc
            # Necessary because of multiple global counters on different procs
            for typ in ['input', 'output']:
                if self.comm.size > 1:

                    # Compute the variable count list; 0 on rank > 0 procs
                    sub_comm = self._subsystems_myproc[0].comm
                    if sub_comm.rank == 0:
                        nvar_myproc = len(subsys0._variable_allprocs_names[typ])
                    else:
                        nvar_myproc = 0
                    nvar_allprocs = self.comm.allgather(nvar_myproc)

                    # Compute the offset
                    iproc = self.comm.rank
                    nvar_myproc = len(subsys0._variable_allprocs_names[typ])
                    index[typ] += numpy.sum(nvar_allprocs[:iproc+1]) \
                               - nvar_myproc

            # Perform the recursion
            for subsys in self._subsystems_myproc:
                subsys._setup_variable_indices(index)

            # Post-recursion: assemble local variable indices from subsystems
            for typ in ['input', 'output']:
                raw = []
                for subsys in self._subsystems_myproc:
                    raw.append(subsys._variable_myproc_indices[typ])
                self._variable_myproc_indices[typ] = numpy.concatenate(raw)

        # If component, _variable_myproc_indices is simply an arange
        else:
            for typ in ['input', 'output']:
                ind1, ind2 = self._variable_allprocs_range[typ]
                self._variable_myproc_indices[typ] = numpy.arange(ind1, ind2)

        # Reset index dict to the global variable count on all procs
        # Necessary for younger siblings to have proper index values
        for typ in ['input', 'output']:
            index[typ] = self._variable_allprocs_range[typ][1]

    def _setup_connections(self):
        """Recursively assemble a list of input-output connections."""
        # Perform recursion and assemble pairs from subsystems
        pairs = []
        for subsys in self._subsystems_myproc:
            subsys._setup_connections()
            if subsys.comm.rank == 0:
                pairs.extend(subsys._variable_connections_indices)

        # Do an allgather to gather from root procs of all subsystems
        if self.comm.size > 1:
            pairs_raw = self.comm.allgather(pairs)
            pairs = []
            for pairs0 in pairs_raw:
                pairs.extend(pairs0)

        # Loop through user-defined connections
        var_allprocs_names = self._variable_allprocs_names
        for ip_name in self._variable_connections:
            op_name = self._variable_connections[ip_name]

            ip_found = ip_name in var_allprocs_names['input']
            op_found = op_name in var_allprocs_names['output']
            if ip_found and op_found:
                ip_index = var_allprocs_names['input'].index(ip_name)
                op_index = var_allprocs_names['output'].index(op_name)
                ip_index += self._variable_allprocs_range['input'][0]
                op_index += self._variable_allprocs_range['output'][0]
                pairs.append([ip_index, op_index])

        self._variable_connections_indices = pairs

    def _setup_vector(self, vec_name, _vectors):
        """Add this vector and assign sub_vectors to subsystems.

        Args
        ----
        vec_name : str
            name of the Vector (None, '', or name of the RHS for derivatives).
        _vectors : {'input': Vector, 'output': Vector, 'residual': Vector}
            Vector objects corresponding to 'name'.
        """
        # Set the incoming _vectors in the appropriate attribute
        for key in ['input', 'output', 'residual']:
            self._vectors[key][vec_name] = _vectors[key]

        # Compute the transfer for this vector set
        transfers = self._util_compute_transfers(_vectors)
        self._vector_transfers[vec_name] = transfers

        # Define shortcuts for convenience
        if vec_name == None:
            self._inputs = _vectors['input']
            self._outputs = _vectors['output']
            self._residuals = _vectors['residual']
            self._transfers = transfers

        # Perform recursion
        for subsys in self._subsystems_myproc:
            sub_comm = subsys.comm
            p_range = subsys._mpi_proc_range

            sub__vectors = {}
            for key in ['input', 'output', 'residual']:
                typ = 'output' if key is 'residual' else key
                v_range = subsys._variable_allprocs_range[typ]
                v_names = subsys._variable_myproc_names[typ]
                v_inds = subsys._variable_myproc_indices[typ]
                vec = _vectors[key]._create_subvector(sub_comm, p_range,
                                                      v_range, v_inds, v_names)
                sub__vectors[key] = vec

            subsys._setup_vector(vec_name, sub__vectors)

    def _setup_solvers(self):
        """Recursively set up all solvers in this and systems below."""
        self._solvers_nonlinear._setup_solvers(self, 0)
        self._solvers_linear._setup_solvers(self, 0)
        for subsys in self._subsystems_myproc:
            subsys._setup_solvers()

    def _util_compute_transfers(self, _vectors):
        """Compute transfers.

        Args
        ----
        _vectors : {'input': Vector, 'output': Vector, 'residual': Vector}
            dictionary of Vector objects

        Returns
        -------
        dict of Transfer
            dictionary of full and partial Transfer objects.
        """
        Transfer = _vectors['output'].TRANSFER

        nsub_allprocs = len(self._subsystems_allprocs)
        var_range = self._variable_allprocs_range
        _subsystems_myproc = self._subsystems_myproc
        _subsystems_inds = self._subsystems_inds

        # Call the assembler's transfer setup routine
        _compute_transfers = self._sys_assembler._compute_transfers
        xfer_indices = _compute_transfers(nsub_allprocs, var_range,
                                          _subsystems_myproc, _subsystems_inds)
        [xfer_ip_inds, xfer_op_inds,
         fwd_xfer_ip_inds, fwd_xfer_op_inds,
         rev_xfer_ip_inds, rev_xfer_op_inds] = xfer_indices

        # Create Transfer objects from the raw indices
        transfers = {}
        transfers[None] = Transfer(_vectors['input'],
                                   _vectors['output'],
                                   xfer_ip_inds,
                                   xfer_op_inds,
                                   self.comm)
        for isub in xrange(len(fwd_xfer_ip_inds)):
            transfers['fwd', isub] = Transfer(_vectors['input'],
                                              _vectors['output'],
                                              fwd_xfer_ip_inds[isub],
                                              fwd_xfer_op_inds[isub],
                                              self.comm)
        for isub in xrange(len(rev_xfer_ip_inds)):
            transfers['rev', isub] = Transfer(_vectors['input'],
                                              _vectors['output'],
                                              rev_xfer_ip_inds[isub],
                                              rev_xfer_op_inds[isub],
                                              self.comm)
        return transfers

    def _utils_compute_maps(self, typ):
        """Define variable maps based on promotes and renames lists.

        Args
        ----
        typ : str
            Either 'input' or 'output'.
        """
        kwargs = self.kwargs
        maps = {}

        # Give all variables the same names in the parent system
        promotes_all = 'promotes_all_%ss' % typ
        if 'promotes_all' in kwargs and kwargs['promotes_all']:
            for name in self._variable_allprocs_names[typ]:
                maps[name] = name
        elif promotes_all in kwargs and kwargs[promotes_all]:
            for name in self._variable_allprocs_names[typ]:
                maps[name] = name
        else:
            # Default: the parent system's name is prepended to variable name
            for name in self._variable_allprocs_names[typ]:
                maps[name] = self._sys_name + '.' + name

            # Promote selected variables
            promotes = 'promotes_%ss' % typ
            if promotes in kwargs:
                for name in kwargs[promotes]:
                    maps[name] = name

            # Rename selected variables to custom names in the parent system
            renames = 'renames_%ss' % typ
            if renames in kwargs:
                for name in kwargs[renames]:
                    maps[name] = kwargs[renames][name]

        self._variable_maps[typ] = maps

    def _utils_compute_deriv_names(self, var_ind_range):
        op_names = []
        op_ind = self.variable_allprocs_range['output'][0]
        for op_name in self.variable_allprocs_names['output']:
            if op_ind in self._vector_IDs[vec_name]:
                op_names.append(op_name)
            op_ind += 1

        ip_names = []
        ip_ind = self.variable_allprocs_range['input'][0]
        for ip_name in self.variable_allprocs_names['input']:
            input_ID = self._sys_assembler._input_IDs[ip_ind]
            valid = var_ind_range[0] <= ip_ind < var_ind_range[1]
            valid = valid and input_ID in self._vector_IDs[vec_name]
            if valid:
                ip_names.append(ip_name)
            ip_ind += 1

        return op_names, ip_names

    def initialize(self, comm):
        """Optional user-defined init method in groups and components."""
        pass

    def initialize_variables(self, comm):
        """Required method for components to declare inputs and outputs."""
        pass
