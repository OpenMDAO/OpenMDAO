from __future__ import division
import numpy



class LoadBalance(object):

    def __call__(self, nsub, comm, proc_range):
        ''' Assigns subsystems and a sub-comm to the current processor '''
        if comm.size == 1:
            isubs = range(nsub)
            sub_comm = comm
            sub_proc_range = [0, 1]
        return isubs, sub_comm, sub_proc_range



class System(object):

    def __init__(self, name, *args, **kwargs):
        ''' Initializes all attributes '''
        self.sys_name = name
        self.sys_args = args
        self.sys_kwargs = kwargs
        self.sys_assembler = None

        self.mpi_comm = None
        self.mpi_load_balance = LoadBalance()
        self.mpi_proc_range = None

        self.subsystems_allprocs = []
        self.subsystems_myproc = []

        self.variable_names = {'input': [], 'output': []}
        self.variable_maps = {'input': {}, 'output': {}}
        self.variable_connections = {}
        self.variable_connections_found = []
        self.variable_range = {'input': [0,0], 'output': [0,0]}

        self.variable_myproc_metadata = {'input': [], 'output': []}
        self.variable_myproc_indices = {'input': None, 'output': None}

        self.vector_names = []
        self.vector_list = {'input': None, 'output': None}
        self.vector_transfers = []

        self.solvers_nonlinear = None
        self.solvers_linear = None

    def get_subsystem(self, name):
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

    def setup_processors(self, assembler, comm, proc_range):
        ''' Splits comms and defines local subsystems '''
        # Set attributes
        self.sys_assembler = assembler
        self.mpi_comm = comm
        self.mpi_proc_range = proc_range

        # Optional user-defined init method
        self.initialize(comm)

        nsub = len(self.subsystems_allprocs)
        if nsub > 0:
            # If this is a group, call the load balance algorithm
            tmp = self.mpi_load_balance(nsub, comm, proc_range)
            sub_inds, sub_comm, sub_proc_range = tmp

            # Define local subsystems and perform recursion
            self.subsystems_myproc = [self.subsystems_allprocs[ind]
                                      for ind in sub_inds]
            for subsys in self.subsystems_myproc:
                subsys.setup_processors(assembler, sub_comm, sub_proc_range)

    def setup_variables(self, recursion=True):
        ''' Assembles variable metadata and names lists '''
        # Perform recursion
        if recursion:
            for subsys in self.subsystems_myproc:
                subsys.setup_variables()

        # Empty the lists in case this is part of a reconfiguration
        for typ in ['input', 'output']:
            self.variable_myproc_metadata[typ] = []
            self.variable_names[typ] = []

        # If this is a component, the user calls add_input/add_output
        if len(self.subsystems_myproc) == 0:
            self.initialize_variables(self.mpi_comm)
        # If this is a group, assemble the metadata and names lists
        else:
            for typ in ['input', 'output']:
                for subsys in self.subsystems_myproc:
                    # Assemble the names list from subsystems
                    subsys.utils_compute_maps(typ)
                    for sub_name in subsys.variable_names[typ]:
                        name = subsys.variable_maps[typ][sub_name]
                        self.variable_names[typ].append(name)

                    # Assemble the metadata list from the subsystems
                    metadata = subsys.variable_myproc_metadata[typ]
                    self.variable_myproc_metadata[typ].extend(metadata)

                # The names list is on all procs, allgather all names
                if self.mpi_comm.size > 1:
                    sub_comm = self.subsystems_myproc[0].mpi_comm
                    if sub_comm.rank == 0:
                        names = self.variable_names[typ]
                    else:
                        names = []
                    self.variable_names[typ] = self.mpi_comm.allgather(names)
                    # TODO: check if this is OK

    def setup_variable_indices(self, index, recursion=True):
        ''' Defines the variable indices (local) and range (global) '''

        # Define the global variable range for the system
        for typ in ['input', 'output']:
            size = len(self.variable_names[typ])
            self.variable_range[typ][0] = index[typ]
            self.variable_range[typ][1] = index[typ] + size

        if len(self.subsystems_myproc) > 0:
            subsys0 = self.subsystems_myproc[0]

            # Compute offset: number of variables on procs before current proc
            for typ in ['input', 'output']:
                # Compute the variable count list; zero on non-rank 0 procs
                if self.mpi_comm.size > 1:
                    sub_comm = self.subsystems_myproc[0].mpi_comm
                    if sub_comm.rank == 0:
                        nvar_myproc = len(subsys0.variable_names[typ])
                    else:
                        nvar_myproc = 0
                    nvar_allprocs = self.mpi_comm.allgather(nvar_myproc)

                    # Compute the offset
                    iproc = self.mpi_comm.rank
                    nvar_myproc = len(subsys0.variable_names[typ])
                    index[typ] += numpy.sum(nvar_allprocs[:iproc+1]) \
                               - nvar_myproc

            # Perform the recursion
            for subsys in self.subsystems_myproc:
                subsys.setup_variable_indices(index)

            # Assemble the local variable indices from the subsystems
            for typ in ['input', 'output']:
                size = len(self.variable_myproc_metadata[typ])
                self.variable_myproc_indices[typ] = numpy.zeros(size, int)
                ind1, ind2 = 0, 0
                for subsys in self.subsystems_myproc:
                    ind2 += len(subsys.variable_myproc_metadata[typ])
                    indices = subsys.variable_myproc_indices[typ]
                    self.variable_myproc_indices[typ][ind1:ind2] = indices
                    ind1 += len(subsys.variable_myproc_metadata[typ])
        else:
            for typ in ['input', 'output']:
                ind1, ind2 = self.variable_range[typ]
                self.variable_myproc_indices[typ] = numpy.arange(ind1, ind2)

        # Reset index dict to the global variable count on all procs
        for typ in ['input', 'output']:
            index[typ] = self.variable_range[typ][1]

    def setup_connections(self):
        ''' Recursively assemble a list of input-output connections '''
        pairs = []
        for subsys in self.subsystems_myproc:
            if subsys.mpi_comm.rank == 0:
                pairs.extend(subsys.variable_connections_found)
        if self.mpi_comm.size > 1:
            pairs = self.mpi_comm.allgather(pairs)

        for ip_name in self.variable_connections:
            op_name = self.variable_connections[ip_name]

            ip_found = ip_name in self.variable_names['input']
            op_found = op_name in self.variable_names['output']
            if ip_found and op_found:
                ip_index = self.variable_names['input'].index(ip_name)
                op_index = self.variable_names['output'].index(op_name)
                ip_index += self.variable_range['input'][0]
                op_index += self.variable_range['output'][0]
                pairs.append([ip_index, op_index])
            else:
                print 'Invalid connection in %s' % self.sys_name

        self.variable_connections_found = pairs

    def utils_compute_maps(self, typ):
        ''' Defines variable maps based on promotes and renames lists '''
        kwargs = self.sys_kwargs
        maps = {}

        # All input/output names are given the same names in the parent system
        promotes_all = 'promotes_all_%ss' % typ
        if promotes_all in kwargs and kwargs[promotes_all]:
            for name in self.variable_names[typ]:
                maps[name] = name
        else:
            # Default: the parent system's name is prepended to variable name
            for name in self.variable_names[typ]:
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
        ''' Optional user-defined init method in groups and components '''
        pass

    def initialize_variables(self, comm):
        ''' Required method for components to declare inputs and outputs '''
        pass


class Group(System):

    def __init__(self, name, *args, **kwargs):
        ''' Subsystems added (1) in method, (2) in kwarg, or (3) in script '''
        super(Group, self).__init__(name, *args, **kwargs)

        self.add_subsystems()
        if 'subsystems' in kwargs:
            self.subsystems_allprocs.extend(kwargs['subsystems'])

    def add_subsystems(self):
        ''' Optional method for adding subsystems '''
        pass

    def add_subsystem(self, subsys):
        ''' Add subsystem '''
        self.subsystems_allprocs.append(subsys)

    def connect(self, op_name, ip_name):
        ''' Connect output op_name to input ip_name in this namespace '''
        self.variable_connections[ip_name] = op_name


class Component(System):

    DEFAULTS = {
        'indices': [0],
        'shape': (1,),
        'units': '',
        'value': 1.0,
        'scale': 1.0,
        'lower': None,
        'upper': None,
        'var_set': 0,
    }

    def add_input(self, name, **kwargs):
        ''' Add an input variable to the component '''
        metadata = self.DEFAULTS.copy()
        metadata.update(kwargs)
        metadata['indices'] = numpy.array(metadata['indices'])
        self.variable_myproc_metadata['input'].append(metadata)
        self.variable_names['input'].append(name)

    def add_output(self, name, **kwargs):
        ''' Add an output variable to the component '''
        metadata = self.DEFAULTS.copy()
        metadata.update(kwargs)
        self.variable_myproc_metadata['output'].append(metadata)
        self.variable_names['output'].append(name)
