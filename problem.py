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
            comm = Comm()
        if Assembler is None:
            Assembler = BaseAssembler

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

        return self
