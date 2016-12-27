"""LinearSolver that uses PetSC KSP to solve for a system's derivatives."""

from __future__ import division, print_function

try:
    import petsc4py
    from petsc4py import PETSc
except ImportError:
    PETSc = None

from openmdao.solvers.solver import LinearSolver


KSP_TYPES = [
    "richardson",
    "chebyshev",
    "cg",
    "groppcg",
    "pipecg",
    "pipecgrr",
    "cgne",
    "nash",
    "stcg",
    "gltr",
    "fcg",
    "pipefcg",
    "gmres",
    "pipefgmres",
    "fgmres",
    "lgmres",
    "dgmres",
    "pgmres",
    "tcqmr",
    "bcgs",
    "ibcgs",
    "fbcgs",
    "fbcgsr",
    "bcgsl",
    "cgs",
    "tfqmr",
    "cr",
    "pipecr",
    "lsqr",
    "preonly",
    "qcg",
    "bicg",
    "minres",
    "symmlq",
    "lcd",
    "python",
    "gcr",
    "pipegcr",
    "tsirm",
    "cgls"
]


def _get_petsc_vec_array_new(vec):
    """Helper function to handle a petsc backwards incompatibility."""
    return vec.getArray(readonly=True)


def _get_petsc_vec_array_old(vec):
    """Helper function to handle a petsc backwards incompatibility."""
    return vec.getArray()


if PETSc:
    try:
        petsc_version = petsc4py.__version__
    except AttributeError:  # hack to fix doc-tests
        petsc_version = "3.5"


if PETSc and int((petsc_version).split('.')[1]) >= 6:
    _get_petsc_vec_array = _get_petsc_vec_array_new
else:
    _get_petsc_vec_array = _get_petsc_vec_array_old


class Monitor(object):
    """Prints output from PETSc's KSP solvers.

    Callable object given to KSP as a callback for printing the residual.

    Attributes
    ----------
    _solver : _solver
        the openmdao solver.
    _norm : float
        the current norm.
    _norm0 : float
        the norm for the first iteration.
    """

    def __init__(self, solver):
        """Store pointer to the openmdao solver and initialize norms.

        Args
        ----
        solver : object
            the openmdao solver.
        """
        self._solver = solver
        self._norm = 1.0
        self._norm0 = 1.0

    def __call__(self, ksp, counter, norm):
        """Store norm if first iteration, and print norm.

        Args
        ----
        ksp : object
            the KSP solver.
        counter : int
            the counter.
        norm : float
            the norm.
        """
        if counter == 0 and norm != 0.0:
            self._norm0 = norm
        self._norm = norm

        self._solver._mpi_print(counter, norm / self._norm0, norm)
        self._solver._iter_count += 1


class PetscKSP(LinearSolver):
    """LinearSolver that uses PetSC KSP to solve for a system's derivatives.

    Options
    -------
    options['ksp_type'] :  str
        KSP algorithm to use. Default is 'fgmres'.

    Attributes
    ----------
    _print_name : str ('KSP')
        print name.
    _ksp : dist
        dictionary of KSP instances (keyed on vector name).
    _precon : Solver
        preconditioner.
    """

    SOLVER = 'LN: PetscKSP'

    def __init__(self, **kwargs):
        """Declare the solver options.

        Args
        ----
        kwargs : {}
            dictionary of options set by the instantiating class/script.
        """
        if PETSc is None:
            raise RuntimeError("PETSc is not available.")

        super(PetscKSP, self).__init__(**kwargs)

        self._print_name = 'KSP'

        # add option to specify which KSP algorithm to use
        self.options.declare('ksp_type', value='fgmres', values=KSP_TYPES,
                             desc="KSP algorithm to use. Default is 'fgmres'.")

        # initialize dictionary of KSP instances (keyed on vector name)
        self._ksp = {}

        # initialize preconditioner to None
        self._precon = None

    def mult(self, mat, in_vec, result):
        """Apply Jacobian matrix (KSP Callback).

        The following attributes must be defined when solve is called to
        provide information used in this callback:

        _system : System
            pointer to the owning system.
        _vec_name : str
            the right-hand-side (RHS) vector name.
        _mode : str
            'fwd' or 'rev'.

        Args
        ----
        mat : PETSc.Mat
            PETSc matrix object.
        in_vec : PetSC Vector
            Incoming vector.
        result : PetSC Vector
            Empty array into which we place the matrix-vector product.
        """
        # assign x and b vectors based on mode
        system = self._system
        vec_name = self._vec_name

        if self._mode == 'fwd':
            x_vec = system._vectors['output'][vec_name]
            b_vec = system._vectors['residual'][vec_name]
        elif self._mode == 'rev':
            x_vec = system._vectors['residual'][vec_name]
            b_vec = system._vectors['output'][vec_name]

        # set value of x vector to KSP provided value
        x_vec.set_data(_get_petsc_vec_array(in_vec))

        # apply linear
        ind1, ind2 = system._variable_allprocs_range['output']
        var_inds = [ind1, ind2, ind1, ind2]
        system._apply_linear([vec_name], self._mode, var_inds)

        # stuff resulting value of b vector into result for KSP
        b_vec.get_data(result.array)

    def __call__(self, vec_names, mode):
        """Solve the linear system for the problem in self._system.

        The full solution vector is returned.

        Args
        ----
        vec_names : list
            list of vector names.
        mode : string
            Derivative mode, can be 'fwd' or 'rev'.
        """
        if 'precon' in self.options['subsolvers']:
            self._precon = self.options['subsolvers']['precon']

        self._vec_names = vec_names
        self._mode = mode

        system = self._system
        options = self.options

        maxiter = options['maxiter']
        atol = options['atol']
        rtol = options['rtol']

        for vec_name in self._vec_names:
            self._vec_name = vec_name

            # assign x and b vectors based on mode
            if self._mode == 'fwd':
                x_vec = system._vectors['output'][vec_name]
                b_vec = system._vectors['residual'][vec_name]
            elif self._mode == 'rev':
                x_vec = system._vectors['residual'][vec_name]
                b_vec = system._vectors['output'][vec_name]

            # create numpy arrays to interface with Petsc
            sol_array = x_vec.get_data()
            rhs_array = b_vec.get_data()

            # create Petsc vectors from numpy arrays
            self.sol_petsc_vec = PETSc.Vec().createWithArray(sol_array,
                                                             comm=system.comm)
            self.rhs_petsc_vec = PETSc.Vec().createWithArray(rhs_array,
                                                             comm=system.comm)

            # run Petsc solver
            self._iter_count = 0
            ksp = self._get_ksp_solver(system, vec_name)
            ksp.setTolerances(max_it=maxiter, atol=atol, rtol=rtol)
            ksp.solve(self.rhs_petsc_vec, self.sol_petsc_vec)

            # stuff the result into the x vector
            x_vec.set_data(sol_array)

    def apply(self, mat, in_vec, result):
        """Apply preconditioner.

        Args
        ----
        mat : PETSc.Mat
            PETSc matrix object.
        in_vec : PETSc.Vector
            Incoming vector
        result : PETSc.Vector
            Empty vector in which the preconditioned in_vec is stored.
        """
        if self._precon:
            system = self._system
            vec_name = self._vec_name
            mode = self._mode

            # assign x and b vectors based on mode
            if mode == 'fwd':
                x_vec = system._vectors['output'][vec_name]
                b_vec = system._vectors['residual'][vec_name]
            elif mode == 'rev':
                x_vec = system._vectors['residual'][vec_name]
                b_vec = system._vectors['output'][vec_name]

            # set value of b vector to KSP provided value
            b_vec.set_data(_get_petsc_vec_array(in_vec))

            # call the preconditioner
            self._precon([vec_name], mode)

            # stuff resulting value of x vector into result for KSP
            x_vec.get_data(result.array)
        else:
            # no preconditioner, just pass back the incoming vector
            result.array[:] = _get_petsc_vec_array(in_vec)

    def _get_ksp_solver(self, system, vec_name):
        """Get an instance of the KSP solver for `vec_name` in `system`.

        Instances will be created on first request and cached for future use.

        Args
        ----
        system : `System`
            Parent `System` object.
        vec_name : string
            name of vector.

        Returns
        -------
        KSP
            the KSP solver instance.
        """
        # use cached instance if available
        if vec_name in self._ksp:
            return self._ksp[vec_name]

        rank = system.comm.rank + system._mpi_proc_range[0]

        lsizes = system._sys_assembler._variable_sizes_all['output'][rank]
        sizes = sum(system._sys_assembler._variable_sizes_all['output'])

        lsize = sum(lsizes)
        size = sum(sizes)

        jac_mat = PETSc.Mat().createPython([(lsize, size), (lsize, size)],
                                           comm=system.comm)
        jac_mat.setPythonContext(self)
        jac_mat.setUp()

        ksp = self._ksp[vec_name] = PETSc.KSP().create(comm=system.comm)

        ksp.setOperators(jac_mat)
        ksp.setType(self.options['ksp_type'])
        ksp.setGMRESRestart(1000)
        ksp.setPCSide(PETSc.PC.Side.RIGHT)
        ksp.setMonitor(Monitor(self))

        pc_mat = ksp.getPC()
        pc_mat.setType('python')
        pc_mat.setPythonContext(self)

        return ksp
