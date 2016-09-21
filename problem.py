from __future__ import division
from assembler import BaseAssembler


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

        self.root = root
        self.comm = comm
        self.assembler = Assembler(comm)
        self.vectors = {}
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

        return self

    def setup_vector(self, name, Vector):
        for name in ['']:
            vec_ip = self.Vector(name, comm, assembler.var_sizes['input'])
            vec_op = self.Vector(name, comm, assembler.var_sizes['output'])

        return self
