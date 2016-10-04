"""Define the Problem class and a FakeComm class for non-MPI users."""
from __future__ import division

from Blue.assemblers.assembler import DefaultAssembler
from Blue.vectors.vector import DefaultVector



class FakeComm(object):
    """Fake MPI communicator class used if mpi4py is not installed.

    Attributes
    ----------
    rank : int
        index of current proc; value is 0 because there is only 1 proc.
    size : int
        number of procs in the comm; value is 1.
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
    _Vector
        reference to the actual Vector class; not an instance.
    """

    def __init__(self, root, comm=None, Assembler=None, Vector=None):
        """Initialize attributes."""
        if comm is None:
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
            except:
                comm = FakeComm()
        if Assembler is None:
            Assembler = DefaultAssembler
        if Vector is None:
            Vector = DefaultVector

        self.root = root
        self.comm = comm
        self._assembler = Assembler(comm)
        self._Vector = Vector

    def setup(self):
        """Set up everything (root, assembler, vector, solvers, drivers).

        Returns
        -------
        self : Problem
            this enables the user to instantiate and setup in one line.
        """
        root = self.root
        comm = self.comm
        _assembler = self._assembler
        _Vector = self._Vector

        # System setup
        root._setup_processors(0, _assembler, {}, comm, [0, comm.size])
        root._setup_variables()
        root._setup_variable_indices({'input': 0, 'output': 0})
        root._setup_connections()
        root._setup_solvers()

        # Assembler setup: variable sizes and indices
        sizes = {typ: len(root._variable_allprocs_names[typ])
                 for typ in ['input', 'output']}
        variable_metadata = root._variable_myproc_metadata
        variable_indices = root._variable_myproc_indices
        _assembler._setup_variables(sizes, variable_metadata, variable_indices)

        # Assembler setup: variable connections
        connections = root._variable_connections_indices
        _variable_allprocs_names = root._variable_allprocs_names
        _assembler._setup_connections(connections, _variable_allprocs_names)

        # Assembler setup: global transfer indices vector
        input_metadata = root._variable_myproc_metadata['input']
        var_indices = root._variable_myproc_indices['input']
        _assembler._setup_input_indices(input_metadata, var_indices)

        # Vector setup for the basic execution vector
        self.setup_vector(None, _Vector)

        return self

    def setup_vector(self, vec_name, Vector):
        """Set up the 'vec_name' Vector.

        Args
        ----
        vec_name : str
            name of the vector
        Vector
            reference to the actual Vector class
        """
        root = self.root
        _assembler = self._assembler

        _vectors = {}
        for key in ['input', 'output', 'residual']:
            if key is 'residual':
                typ = 'output'
            else:
                typ = key

            nvar_all = len(root._variable_allprocs_names[typ])
            vec = Vector(vec_name, self.comm, root._mpi_proc_range,
                         root._variable_allprocs_range[typ],
                         root._variable_allprocs_names[typ],
                         _assembler._variable_sizes[typ],
                         _assembler._variable_set_indices[typ])
            _vectors[key] = vec

        self.root._setup_vector(vec_name, _vectors)
