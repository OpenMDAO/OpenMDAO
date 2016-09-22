from __future__ import division

from assembler import BaseAssembler
from vector import BaseVector



class Comm(object):
    ''' Fake MPI communicator class used if mpi4py is not installed '''

    def __init__(self):
        ''' Defines processor rank and size '''
        self.rank = 0
        self.size = 1



class Problem(object):
    ''' Top-level container for the systems and drivers '''

    def __init__(self, root, comm=None, Assembler=None, Vector=None):
        if comm is None:
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
            except:
                comm = Comm()
        if Assembler is None:
            Assembler = BaseAssembler
        if Vector is None:
            Vector = BaseVector

        self.root = root
        self.comm = comm
        self.assembler = Assembler(comm)
        self.Vector = Vector

    def setup(self):
        root = self.root
        assembler = self.assembler

        root.setup_processors(self.assembler, self.comm, [0, self.comm.size])
        root.setup_variables()
        root.setup_variable_indices({'input': 0, 'output': 0})
        root.setup_connections()

        sizes = {typ: len(root.variable_names[typ])
                 for typ in ['input', 'output']}
        variable_metadata = root.variable_myproc_metadata
        variable_indices = root.variable_myproc_indices
        assembler.setup_variables(sizes, variable_metadata, variable_indices)

        connections = root.variable_connections_indices
        variable_names = root.variable_names
        assembler.setup_connections(connections, variable_names)

        input_metadata = root.variable_myproc_metadata['input']
        assembler.setup_input_indices(input_metadata)

        self.setup_vector('', self.Vector)

        return self

    def setup_vector(self, name, Vector):
        root = self.root
        assembler = self.assembler

        vectors = {}
        for key in ['input', 'output', 'residual']:
            if key is 'residual':
                typ = 'output'
            else:
                typ = key

            nvar_all = len(root.variable_names[typ])
            vec = Vector(name, self.comm, root.mpi_proc_range,
                         root.variable_range[typ],
                         root.variable_names[typ],
                         assembler.variable_sizes[typ],
                         assembler.variable_set_indices[typ])
            vectors[key] = vec
        self.root.setup_vector(name, vectors)
