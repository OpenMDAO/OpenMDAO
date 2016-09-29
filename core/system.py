"""Define the base System class."""
from __future__ import division
import numpy

from Blue.proc_allocators.proc_allocator import DefaultProcAllocator
from Blue.solvers.solver import NonlinearBlockGS



class System(object):
    """Base class for all systems in OpenMDAO.

    Always subclassed by Group or Component, or a subclass thereof.
    All subclasses have their attributes defined here.

    Attributes
    ----------
    sys_name : str
        name of the system, must be different from siblings.
    sys_depth : int
        distance from the root node in the hierarchy tree.
    sys_assembler: Assembler
        pointer to the global assembler object.

    sys_args : list of objects
        user-defined arguments (to be used in apply_nonlinear, ...).
    sys_kwargs : dict of objects
        dictionary of user-defined arguments.
    sys_global_kwargs : dict of objects
        sys_kwargs combined with kwargs of parent systems.

    mpi_comm : MPI.Comm or FakeComm
        MPI communicator object.
    mpi_proc_allocator : ProcAllocator
        object that distributes procs among subsystems.
    mpi_proc_range : [int, int]
        indices of procs owned by mpi_comm with respect to COMM_WORLD.

    subsystems_allprocs : [System, ...]
        list of all subsystems (children of this system).
    subsystems_myproc : [System, ...]
        list of local subsystems that exist on this proc.
    subsystems_inds : [int, ...]
        list of indices of subsystems on this proc among all subsystems.

    variable_allprocs_names : {'input': [str, ...], 'output': [str, ...]}
        list of names of all owned variables, not just on current proc.
    variable_allprocs_range : {'input': [int,int], 'output': [int,int]}
        index range of owned variables with respect to all problem variables.

    variable_myproc_metadata : {'input': list, 'output': list}
        list of metadata dictionaries of variables that exist on this proc.
    variable_myproc_indices : {'input': ndarray[:], 'output': ndarray[:]}
        integer arrays of global indices of variables on this proc.

    variable_maps : {'input': dict, 'output': dict}
        dict of variable names and their aliases (for promotes/renames).
    variable_connections : dict
        dict of input:output connections between subsystems.
    variable_connections_indices : [(int, int), ...]
        variable_connections with variable indices instead of names.

    vectors : {'input': dict, 'output': dict, 'residual': dict}
        dict of vector objects.
    vector_transfers : dict
        dict of transfer objects.

    inputs : Vector
        inputs vector; points to vectors['input'][None].
    outputs : Vector
        outputs vector; points to vectors['output'][None].
    residuals : Vector
        residuals vector; points to vectors['residual'][None].
    transfers : dict of Transfer
        transfer object; points to vector_transfers[None].

    solvers_nonlinear : NonlinearSolver
        nonlinear solver to be used for solve_nonlinear.
    solvers_linear : LinearSolver
        linear solver to be used for solve_linear; not the Newton system.
    solvers_print : boolean
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
            available in methods as self.sys_args.
        **kwargs: dict of keyword arguments
            available here and in all descendants of this system.
        """
        self.sys_name = name
        self.sys_depth = 0
        self.sys_assembler = None

        self.sys_args = args
        self.sys_kwargs = kwargs
        self.sys_global_kwargs = {}

        self.mpi_comm = None
        self.mpi_proc_allocator = DefaultProcAllocator()
        self.mpi_proc_range = None

        self.subsystems_allprocs = []
        self.subsystems_myproc = []
        self.subsystems_inds = []

        self.variable_allprocs_names = {'input': [], 'output': []}
        self.variable_allprocs_range = {'input': [0,0], 'output': [0,0]}

        self.variable_myproc_metadata = {'input': [], 'output': []}
        self.variable_myproc_indices = {'input': None, 'output': None}

        self.variable_maps = {'input': {}, 'output': {}}
        self.variable_connections = {}
        self.variable_connections_indices = []

        self.vectors = {'input': {}, 'output': {}, 'residual': {}}
        self.vector_transfers = {}

        self.inputs = None
        self.outputs = None
        self.residuals = None
        self.transfers = None

        self.solvers_nonlinear = NonlinearBlockGS()
        self.solvers_linear = NonlinearBlockGS() # temporary hack!
        self.solvers_print = True

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
        if name == self.sys_name:
            # If this system's name matches, target found
            return self
        else:
            ind = len(self.sys_name) + 1
            # If first part of name matches this system's name, check subsytems
            if name[:ind] == '%s.' % self.sys_name:
                for subsys in self.subsystems_myproc:
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
        self.solvers_print = flag
        for subsys in self.subsystems_myproc:
            subsys.set_solver_print(flag)

    def setup_processors(self, depth, assembler, global_kwargs, comm,
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
        self.sys_depth = depth
        self.sys_assembler = assembler
        self.sys_global_kwargs = global_kwargs
        self.mpi_comm = comm
        self.mpi_proc_range = proc_range

        self.sys_global_kwargs.update(self.sys_kwargs)

        # Optional user-defined init method
        self.initialize(comm)

        nsub = len(self.subsystems_allprocs)
        if nsub > 0:
            # If this is a group, call the load balancing algorithm
            tmp = self.mpi_proc_allocator(nsub, comm, proc_range)
            sub_inds, sub_comm, sub_proc_range = tmp

            # Define local subsystems and perform recursion
            self.subsystems_myproc = [self.subsystems_allprocs[ind]
                                      for ind in sub_inds]
            self.subsystems_inds = sub_inds
            for subsys in self.subsystems_myproc:
                sub_global_kwargs = self.sys_global_kwargs.copy()
                subsys.setup_processors(depth+1, assembler, sub_global_kwargs,
                                        sub_comm, sub_proc_range)

    def setup_variables(self, recursion=True):
        """Assemble variable metadata and names lists.

        Args
        ----
        recursion : boolean
            recursion is not performed if traversing up the tree after reconf.
        """
        # Perform recursion
        if recursion:
            for subsys in self.subsystems_myproc:
                subsys.setup_variables()

        # Empty the lists in case this is part of a reconfiguration
        for typ in ['input', 'output']:
            self.variable_myproc_metadata[typ] = []
            self.variable_allprocs_names[typ] = []

        # If this is a component, the user calls add_input/add_output
        if len(self.subsystems_myproc) == 0:
            self.initialize_variables(self.mpi_comm)
        # If this is a group, assemble the metadata and names lists
        else:
            for typ in ['input', 'output']:
                for subsys in self.subsystems_myproc:
                    # Assemble the names list from subsystems
                    subsys.utils_compute_maps(typ)
                    for sub_name in subsys.variable_allprocs_names[typ]:
                        name = subsys.variable_maps[typ][sub_name]
                        self.variable_allprocs_names[typ].append(name)

                    # Assemble the metadata list from the subsystems
                    metadata = subsys.variable_myproc_metadata[typ]
                    self.variable_myproc_metadata[typ].extend(metadata)

                # The names list is on all procs, allgather all names
                if self.mpi_comm.size > 1:

                    # One representative proc from each sub_comm adds names
                    sub_comm = self.subsystems_myproc[0].mpi_comm
                    if sub_comm.rank == 0:
                        names = self.variable_allprocs_names[typ]
                    else:
                        names = []

                    # Every proc on this comm now has global variable names
                    raw = self.mpi_comm.allgather(names)
                    self.variable_allprocs_names[typ] = []
                    for names in raw:
                        self.variable_allprocs_names[typ].extend(names)

    def setup_variable_indices(self, index, recursion=True):
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
            size = len(self.variable_allprocs_names[typ])
            self.variable_allprocs_range[typ][0] = index[typ]
            self.variable_allprocs_range[typ][1] = index[typ] + size

        # If group, compute variable_myproc_indices as follows
        if len(self.subsystems_myproc) > 0:
            subsys0 = self.subsystems_myproc[0]

            # Pre-recursion: compute 'index' to pass to subsystems
            # Need offset: number of variables on procs before current proc
            # Necessary because of multiple global counters on different procs
            for typ in ['input', 'output']:
                if self.mpi_comm.size > 1:

                    # Compute the variable count list; 0 on rank > 0 procs
                    sub_comm = self.subsystems_myproc[0].mpi_comm
                    if sub_comm.rank == 0:
                        nvar_myproc = len(subsys0.variable_allprocs_names[typ])
                    else:
                        nvar_myproc = 0
                    nvar_allprocs = self.mpi_comm.allgather(nvar_myproc)

                    # Compute the offset
                    iproc = self.mpi_comm.rank
                    nvar_myproc = len(subsys0.variable_allprocs_names[typ])
                    index[typ] += numpy.sum(nvar_allprocs[:iproc+1]) \
                               - nvar_myproc

            # Perform the recursion
            for subsys in self.subsystems_myproc:
                subsys.setup_variable_indices(index)

            # Post-recursion: assemble local variable indices from subsystems
            for typ in ['input', 'output']:
                raw = []
                for subsys in self.subsystems_myproc:
                    raw.append(subsys.variable_myproc_indices[typ])
                self.variable_myproc_indices[typ] = numpy.concatenate(raw)

        # If component, variable_myproc_indices is simply an arange
        else:
            for typ in ['input', 'output']:
                ind1, ind2 = self.variable_allprocs_range[typ]
                self.variable_myproc_indices[typ] = numpy.arange(ind1, ind2)

        # Reset index dict to the global variable count on all procs
        # Necessary for younger siblings to have proper index values
        for typ in ['input', 'output']:
            index[typ] = self.variable_allprocs_range[typ][1]

    def setup_connections(self):
        """Recursively assemble a list of input-output connections."""
        # Perform recursion and assemble pairs from subsystems
        pairs = []
        for subsys in self.subsystems_myproc:
            subsys.setup_connections()
            if subsys.mpi_comm.rank == 0:
                pairs.extend(subsys.variable_connections_indices)

        # Do an allgather to gather from root procs of all subsystems
        if self.mpi_comm.size > 1:
            pairs_raw = self.mpi_comm.allgather(pairs)
            pairs = []
            for pairs0 in pairs_raw:
                pairs.extend(pairs0)

        # Loop through user-defined connections
        var_allprocs_names = self.variable_allprocs_names
        for ip_name in self.variable_connections:
            op_name = self.variable_connections[ip_name]

            ip_found = ip_name in var_allprocs_names['input']
            op_found = op_name in var_allprocs_names['output']
            if ip_found and op_found:
                ip_index = var_allprocs_names['input'].index(ip_name)
                op_index = var_allprocs_names['output'].index(op_name)
                ip_index += self.variable_allprocs_range['input'][0]
                op_index += self.variable_allprocs_range['output'][0]
                pairs.append([ip_index, op_index])

        self.variable_connections_indices = pairs

    def setup_vector(self, vec_name, vectors):
        """Add this vector and assign subvectors to subsystems.

        Args
        ----
        vec_name : str
            name of the Vector (None, '', or name of the RHS for derivatives).
        vectors : {'input': Vector, 'output': Vector, 'residual': Vector}
            Vector objects corresponding to 'name'.
        """
        # Set the incoming vectors in the appropriate attribute
        for key in ['input', 'output', 'residual']:
            self.vectors[key][vec_name] = vectors[key]

        # Compute the transfer for this vector set
        transfers = self.util_compute_transfers(vectors)
        self.vector_transfers[vec_name] = transfers

        # Define shortcuts for convenience
        if vec_name == None:
            self.inputs = vectors['input']
            self.outputs = vectors['output']
            self.residuals = vectors['residual']
            self.transfers = transfers

        # Perform recursion
        for subsys in self.subsystems_myproc:
            sub_comm = subsys.mpi_comm
            p_range = subsys.mpi_proc_range

            sub_vectors = {}
            for key in ['input', 'output', 'residual']:
                typ = 'output' if key is 'residual' else key
                v_range = subsys.variable_allprocs_range[typ]
                v_names = subsys.variable_allprocs_names[typ]
                vec = vectors[key].create_subvector(sub_comm, p_range,
                                                    v_range, v_names)
                sub_vectors[key] = vec

            subsys.setup_vector(vec_name, sub_vectors)

    def setup_solvers(self):
        """Recursively set up all solvers in this and systems below."""
        self.solvers_nonlinear.setup_solvers(self, 0)
        self.solvers_linear.setup_solvers(self, 0)
        for subsys in self.subsystems_myproc:
            subsys.setup_solvers()

    def util_compute_transfers(self, vectors):
        """Compute transfers.

        Args
        ----
        vectors : {'input': Vector, 'output': Vector, 'residual': Vector}
            dictionary of Vector objects

        Returns
        -------
        dict of Transfer
            dictionary of full and partial Transfer objects.
        """
        Transfer = vectors['output'].TRANSFER

        nsub_allprocs = len(self.subsystems_allprocs)
        var_range = self.variable_allprocs_range
        subsystems_myproc = self.subsystems_myproc
        subsystems_inds = self.subsystems_inds

        # Call the assembler's transfer setup routine
        compute_transfers = self.sys_assembler.compute_transfers
        xfer_indices = compute_transfers(nsub_allprocs, var_range,
                                         subsystems_myproc, subsystems_inds)
        [xfer_ip_inds, xfer_op_inds,
         fwd_xfer_ip_inds, fwd_xfer_op_inds,
         rev_xfer_ip_inds, rev_xfer_op_inds] = xfer_indices

        # Create Transfer objects from the raw indices
        transfers = {}
        transfers[None] = Transfer(vectors['input'],
                                   vectors['output'],
                                   xfer_ip_inds,
                                   xfer_op_inds,
                                   self.mpi_comm)
        for isub in xrange(len(fwd_xfer_ip_inds)):
            transfers['fwd', isub] = Transfer(vectors['input'],
                                              vectors['output'],
                                              fwd_xfer_ip_inds[isub],
                                              fwd_xfer_op_inds[isub],
                                              self.mpi_comm)
        for isub in xrange(len(rev_xfer_ip_inds)):
            transfers['rev', isub] = Transfer(vectors['input'],
                                              vectors['output'],
                                              rev_xfer_ip_inds[isub],
                                              rev_xfer_op_inds[isub],
                                              self.mpi_comm)
        return transfers

    def utils_compute_maps(self, typ):
        """Define variable maps based on promotes and renames lists.

        Args
        ----
        typ : str
            Either 'input' or 'output'.
        """
        kwargs = self.sys_kwargs
        maps = {}

        # Give all variables the same names in the parent system
        promotes_all = 'promotes_all_%ss' % typ
        if 'promotes_all' in kwargs and kwargs['promotes_all']:
            for name in self.variable_allprocs_names[typ]:
                maps[name] = name
        elif promotes_all in kwargs and kwargs[promotes_all]:
            for name in self.variable_allprocs_names[typ]:
                maps[name] = name
        else:
            # Default: the parent system's name is prepended to variable name
            for name in self.variable_allprocs_names[typ]:
                maps[name] = self.sys_name + '.' + name

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

        self.variable_maps[typ] = maps

    def initialize(self, comm):
        """Optional user-defined init method in groups and components."""
        pass

    def initialize_variables(self, comm):
        """Required method for components to declare inputs and outputs."""
        pass

    def solve_nonlinear(self):
        """Call this system's nonlinear solver."""
        return self.solvers_nonlinear()

    def solve_linear(self):
        """Call this system's linear solver."""
        return self.solvers_linear()
