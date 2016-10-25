"""Define the Problem class and a FakeComm class for non-MPI users."""
from __future__ import division

import sys

from openmdao.assemblers.assembler import DefaultAssembler
from openmdao.vectors.vector import DefaultVector


class FakeComm(object):
    """Fake MPI communicator class used if mpi4py is not installed.

    Attributes
    ----------
    rank : int
        index of current proc; value is 0 because there is only 1 proc.
    size : int
        number of procs in the comm; value is 1 since MPI is not available.
    """

    def __init__(self):
        """Initialize attributes."""
        self.rank = 0
        self.size = 1


class Problem(object):
    """Top-level container for the systems and drivers.

    Attributes
    ----------
    root : System
        pointer to the top-level System object (root node in the tree).
    comm : MPI.Comm or FakeComm
        the global communicator; the same as that of assembler and root.
    _assembler : Assembler
        pointer to the global Assembler object.
    """

    def __init__(self, root=None, comm=None, AssemblerClass=None):
        """Initialize attributes."""
        if comm is None:
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
            except ImportError:
                comm = FakeComm()
        if AssemblerClass is None:
            AssemblerClass = DefaultAssembler

        self.root = root
        self.comm = comm
        self._assembler = AssemblerClass(comm)

    # FIXME: getitem/setitem need to properly handle scaling/units and must
    # properly handle getting/setting using arrays > 1D
    def __getitem__(self, name):
        try:
            return self.root._outputs[name]
        except KeyError:
            return self.root._inputs[name]
    
    def __setitem__(self, name, value):
        try:
            self.root._outputs[name] = value
        except KeyError:
            self.root._inputs[name] = value
    
    # FIXME: once we have drivers, this should call self.driver.run() instead
    def run(self):
        self.root._solve_nonlinear()
        
    def setup(self, VectorClass=None, check=False, out_stream=sys.stdout):
        """Set up everything (root, assembler, vector, solvers, drivers).

        Args
        ----
        VectorClass
            reference to an actual Vector class; not an instance.

        Returns
        -------
        self : Problem
            this enables the user to instantiate and setup in one line.
        """
        root = self.root
        comm = self.comm
        assembler = self._assembler

        if VectorClass is None:
            VectorClass = DefaultVector

        # Recursive system setup
        root._setup_processors('', comm, {}, 0, assembler, [0, comm.size])
        root._setup_variables()
        root._setup_variable_indices({'input': 0, 'output': 0})
        root._setup_connections()
        root._setup_solvers()

        # Assembler setup: variable metadata and indices
        sizes = {typ: len(root._variable_allprocs_names[typ])
                 for typ in ['input', 'output']}
        variable_metadata = root._variable_myproc_metadata
        variable_indices = root._variable_myproc_indices
        assembler._setup_variables(sizes, variable_metadata, variable_indices)

        # Assembler setup: variable connections
        connections = root._variable_connections_indices
        variable_allprocs_names = root._variable_allprocs_names
        assembler._setup_connections(connections, variable_allprocs_names)

        # Assembler setup: global transfer indices vector
        input_metadata = root._variable_myproc_metadata['input']
        var_indices = root._variable_myproc_indices['input']
        assembler._setup_input_indices(input_metadata, var_indices)

        # Vector setup for the basic execution vector
        self.setup_vector(None, VectorClass)

        # Vector setup for the linear vector
        self.setup_vector('', VectorClass)

        return self

    def setup_vector(self, vec_name, VectorClass):
        """Set up the 'vec_name' Vector.

        Args
        ----
        vec_name : str
            name of the vector
        VectorClass
            reference to the actual Vector class
        """
        root = self.root
        assembler = self._assembler

        vectors = {}
        for key in ['input', 'output', 'residual']:
            if key is 'residual':
                typ = 'output'
            else:
                typ = key

            vectors[key] = VectorClass(vec_name, typ, self.root)

        # TODO: implement this properly
        ind1, ind2 = self.root._variable_allprocs_range['output']
        import numpy
        vector_var_ids = numpy.arange(ind1, ind2)

        self.root._setup_vector(vectors, vector_var_ids)
