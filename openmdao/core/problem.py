"""Define the Problem class and a FakeComm class for non-MPI users."""
from __future__ import division

import sys

from openmdao.assemblers.default_assembler import DefaultAssembler
from openmdao.vectors.default_vector import DefaultVector
from openmdao.error_checking.check_config import check_config


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
    root : <System>
        pointer to the top-level <System> object (root node in the tree).
    comm : MPI.Comm or <FakeComm>
        the global communicator; the same as that of assembler and root.
    _assembler : <Assembler>
        pointer to the global <Assembler> object.
    _use_ref_vector : bool
        if True, allocate vectors to store ref. values.
    """

    def __init__(self, root=None, comm=None, assembler_class=None,
                 use_ref_vector=True):
        """Initialize attributes.

        Args
        ----
        root : <System> or None
            pointer to the top-level <System> object (root node in the tree).
        comm : MPI.Comm or <FakeComm> or None
            the global communicator; the same as that of assembler and root.
        assembler_class : <Assembler> or None
            pointer to the global <Assembler> object.
        use_ref_vector : bool
            if True, allocate vectors to store ref. values.
        """
        if comm is None:
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
            except ImportError:
                comm = FakeComm()
        if assembler_class is None:
            assembler_class = DefaultAssembler

        self.root = root
        self.comm = comm
        self._assembler = assembler_class(comm)
        self._use_ref_vector = use_ref_vector

    # TODO: getitem/setitem need to properly handle scaling/units
    def __getitem__(self, name):
        """Get an output/input variable.

        Args
        ----
        name : str
            name of the variable in the root's namespace.

        Returns
        -------
        float or ndarray
            the requested output/input variable.
        """
        try:
            self.root._outputs[name]
            ind = self.root._var_myproc_names['output'].index(name)
            c0, c1 = self.root._scaling_to_phys['output'][ind, :]
            return c0 + c1 * self.root._outputs[name]
        except KeyError:
            ind = self.root._var_myproc_names['input'].index(name)
            c0, c1 = self.root._scaling_to_phys['input'][ind, :]
            return c0 + c1 * self.root._inputs[name]

    def __setitem__(self, name, value):
        """Set an output/input variable.

        Args
        ----
        name : str
            name of the output/input variable in the root's namespace.
        value : float or ndarray or list
            value to set this variable to.
        """
        try:
            self.root._outputs[name]
            ind = self.root._var_myproc_names['output'].index(name)
            c0, c1 = self.root._scaling_to_norm['output'][ind, :]
            self.root._outputs[name] = c0 + c1 * value
        except KeyError:
            ind = self.root._var_myproc_names['input'].index(name)
            c0, c1 = self.root._scaling_to_norm['input'][ind, :]
            self.root._inputs[name] = c0 + c1 * value

    # TODO: once we have drivers, this should call self.driver.run() instead
    def run(self):
        """Run the model by calling the root's solve_nonlinear.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        return self.root._solve_nonlinear()

    def setup(self, vector_class=DefaultVector, check=True, logger=None):
        """Set up everything (root, assembler, vector, solvers, drivers).

        Args
        ----
        vector_class : type (DefaultVector)
            reference to an actual <Vector> class; not an instance.
        check : boolean (True)
            whether to run error check after setup is complete.
        logger : object
            Object for logging config checks if check is True.

        Returns
        -------
        self : <Problem>
            this enables the user to instantiate and setup in one line.
        """
        root = self.root
        comm = self.comm
        assembler = self._assembler

        # Recursive system setup
        root._setup_processors('', comm, {}, assembler, [0, comm.size])
        root._setup_variables()
        root._setup_variable_indices({'input': 0, 'output': 0})
        root._setup_connections()

        # Assembler setup: variable metadata and indices
        nvars = {typ: len(root._var_allprocs_names[typ])
                 for typ in ['input', 'output']}
        assembler._setup_variables(nvars, root._var_myproc_metadata,
                                   root._var_myproc_indices)

        # Assembler setup: variable connections
        assembler._setup_connections(root._var_connections_indices,
                                     root._var_allprocs_names)

        # Assembler setup: global transfer indices vector
        assembler._setup_src_indices(root._var_myproc_metadata['input'],
                                     root._var_myproc_indices['input'])

        # Assembler setup: compute data required for units/scaling
        assembler._setup_src_data(root._var_myproc_metadata['output'],
                                  root._var_myproc_indices['output'])

        root._setup_scaling()

        # Vector setup for the basic execution vector
        self.setup_vector('nonlinear', vector_class, self._use_ref_vector)

        # Vector setup for the linear vector
        self.setup_vector('linear', vector_class, self._use_ref_vector)

        if check:
            check_config(self, logger)

        return self

    def setup_vector(self, vec_name, vector_class, use_ref_vector):
        """Set up the 'vec_name' <Vector>.

        Args
        ----
        vec_name : str
            name of the vector.
        vector_class : type
            reference to the actual <Vector> class.
        use_ref_vector : bool
            if True, allocate vectors to store ref. values.
        """
        root = self.root
        assembler = self._assembler

        vectors = {}
        for key in ['input', 'output', 'residual']:
            if key is 'residual':
                typ = 'output'
            else:
                typ = key

            vectors[key] = vector_class(vec_name, typ, self.root)

        # TODO: implement this properly
        ind1, ind2 = self.root._var_allprocs_range['output']
        import numpy
        vector_var_ids = numpy.arange(ind1, ind2)

        self.root._setup_vector(vectors, vector_var_ids, use_ref_vector)

    def compute_total_derivs(self, of=None, wrt=None, return_format='flat_dict'):
        """Compute derivatives of desired quantities with respect to desired inputs.

        Args
        ----
        of : list of variable name strings or None
            Variables whose derivatives will be computed. Default is None, which
            uses the driver's objectives and constraints.
        wrt : list of variable name strings or None
            Variables with respect to which the derivatives will be computed.
            Default is None, which uses the driver's desvars.
        return_format : string
            Format to return the derivatives. Default is a 'flat_dict', which
            returns them in a dictionary whose keys are tuples of form (of, wrt).

        Returns
        -------
        derivs : object
            Derivatives in form requested by 'return_format'.
        """
        root = self.root

        #TODO - Pull 'of' and 'wrt' from driver if unspecified.
        if wrt is None:
            raise NotImplementedError("Need to specify 'wrt' for now.")
        if of is None:
            raise NotImplementedError("Need to specify 'of' for now.")

        # A number of features will need to be supported here as development
        # goes forward.
        # TODO: Make sure we can function in parallel when some params or
        # functions are not local.
        # TODO: Support parallel adjoint and parallel forward derivatives
        #       Aside: how are they specified, and do we have to pick up any
        #       that are missed?
        # TODO: Handle driver scaling.
        # TODO: Support for any other return_format we need.
        # TODO: Support constraint sparsity (i.e., skip in/out that are not
        #       relevant for this constraint) (desvars too?)
        # TODO: Don't calculate for inactive constraints
        # TODO: Support full-model FD. Don't know how this'll work, but we
        #       used to need a separate function for that.

