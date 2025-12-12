"""LinearSolver that uses PETSc for LU factor/solve."""

import numpy as np
import scipy.sparse

from openmdao.solvers.linear.direct import DirectSolver
from openmdao.solvers.linear.direct import format_singular_error
from openmdao.matrices.dense_matrix import DenseMatrix
from openmdao.solvers.linear.linear_rhs_checker import LinearRHSChecker
from openmdao.utils.om_warnings import issue_warning, SolverWarning


PC_SERIAL_TYPES = [
    "superlu",
    "klu",
    "umfpack",
    "petsc",
]
PC_DISTRIBUTED_TYPES = [
    "mumps",
    "superlu_dist",
]
# Direct solvers that don't come installed with petsc4py or are not supported
# for this problem
# strumpack
# pardiso
# cholmod


def check_err_on_singular(name, value):
    """
    Check the value of the "err_on_singular" option on the PETScDirectSolver.

    Parameters
    ----------
    name : str
        Name of the option being checked.
    value : bool
        Value of the option being checked.
    """
    if not value:
        raise ValueError(
            f'The PETScDirectSolver must always have its "{name}" option set to '
            'True. This option is only maintained for compatibility with parent '
            'solver methods.'
        )


class PETScLU:
    """
    Wrapper for PETSc LU decomposition, using petsc4py.

    Parameters
    ----------
    A : ndarray or <scipy.sparse.csc_matrix>
        Matrix to use in solving x @ A == b.
    sparse_solver_name : str
        Name of the direct solver from PETSc to use.
    comm : <mpi4py.MPI.Intracomm>
        The system MPI communicator.

    Attributes
    ----------
    orig_A : ndarray or <scipy.sparse.csc_matrix>
        Originally provided matrix.
    A : <petsc4py.PETSc.Mat>
        Assembled PETSc AIJ (compressed sparse row format) matrix.
    ksp : <petsc4py.PETSc.KSP>
        PETSc Krylov Subspace Solver context.
    running_mpi : bool
        Is the script currently being run under MPI (True) or not (False).
    comm : <mpi4py.MPI.Intracomm>
        The system MPI communicator.
    _x : <petsc4py.PETSc.Vec>
        Sequential (non-distributed) PETSc vector to store the solve solution.
    _PETSc : <petsc4py.PETSc>
        Lazily imported petsc4py.PETSc module.
    """

    def __init__(self, A: scipy.sparse.spmatrix, sparse_solver_name: str = None,
                 comm=None):
        """
        Initialize and setup the PETSc LU Direct Solver object.
        """
        # Lazy import of PETSc
        try:
            from petsc4py import PETSc
            self._PETSc = PETSc
        except ImportError:
            self._PETSc = None

        if comm is None:
            from openmdao.utils.mpi import MPI, FakeComm
            if MPI is None:
                self.comm = FakeComm()
            else:
                self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm

        self.running_mpi = not self.comm.size == 1
        self.orig_A = A
        # Create PETSc matrix
        # Dense
        if isinstance(A, np.ndarray):
            self.A = self._PETSc.Mat().createDense(A.shape, array=A, comm=self._PETSc.COMM_SELF)

        # Sparse
        else:
            # PETSc wants to use CSR matrices, not CSC
            if not scipy.sparse.isspmatrix_csr(A):
                A = A.tocsr()

            # TODO: Look at how to maybe provide a nnz argument for a rough
            # estimate of number of nonzero rows so it hopefully doesn't have
            # to do frequent reallocations
            if self.running_mpi and sparse_solver_name in PC_DISTRIBUTED_TYPES:
                # Parallel: build local CSR
                self.A = self._PETSc.Mat().create(comm=comm)
                self.A.setSizes(A.shape)
                self.A.setType('aij')
                self.A.setUp()

                rstart, rend = self.A.getOwnershipRange()

                indptr = A.indptr
                indices = A.indices
                data = A.data

                for i in range(rstart, rend):
                    row_start = indptr[i]
                    row_end = indptr[i + 1]
                    cols = indices[row_start: row_end]
                    vals = data[row_start: row_end]
                    self.A.setValues(i, cols, vals)

            else:
                # Serial: build full CSR (if you're running a serial solver
                # while using MPI, then it will solve the full thing on each rank)
                self.A = self._PETSc.Mat().createAIJ(
                    size=A.shape,
                    csr=(A.indptr, A.indices, A.data),
                    comm=self._PETSc.COMM_SELF,
                )

            self.A.assemble()

        # Create PETSc solver (KSP is the iterative linear solver [Krylov SPace
        # solver] and PC is the preconditioner)
        self.ksp = self._PETSc.KSP().create()
        self.ksp.setOperators(self.A)
        # Use only the preconditioner (e.g. direct LU solve) and skip the
        # iterative solve
        self.ksp.setType('preonly')
        pc = self.ksp.getPC()
        # In practice, majority of OpenMDAO applications use general, unsymmetric
        # Jacobians, so LU is usually the only practical choice.
        pc.setType('lu')

        # Backends are only for sparse matrices. For dense matrix should by
        # default use LAPACK
        if sparse_solver_name and not isinstance(A, np.ndarray):
            if sparse_solver_name in PC_DISTRIBUTED_TYPES and not self.running_mpi:
                issue_warning(
                    f'The "{sparse_solver_name}" solver is meant to be run '
                    'distributed, but it is currently being run sequentially.',
                    category=SolverWarning
                )
            pc.setFactorSolverType(sparse_solver_name)

        # Read and apply the user specified options to configure the solver,
        # preconditioner, etc., then perform the internal setup and initialization
        # (setUp can be called automatically by solve(), but we call it
        # explicitly so we can prepare the solver beforehand)
        self.ksp.setFromOptions()
        self.ksp.setUp()

        # Create a single process vector which will store the solve solution
        # vector (createVecLeft automatically creates a properly sized and
        # distributed vector based on A)
        self._x = self.A.createVecLeft()

    def solve(self, b: np.ndarray, transpose: bool = False) -> np.ndarray:
        """
        Solve the linear system using only the preconditioner direct solver.

        Parameters
        ----------
        b : ndarray
            Input data for the right hand side.
        transpose : bool
            Is A.T @ x == b being solved (True) or A @ x == b (False).

        Returns
        -------
        ndarray
            The solution array.
        """
        b_petsc = self.A.createVecRight()
        rstart, rend = b_petsc.getOwnershipRange()
        b_petsc.setValues(range(rstart, rend), b[rstart: rend])
        b_petsc.assemble()

        if transpose:
            self.ksp.solveTranspose(b_petsc, self._x)
        else:
            self.ksp.solve(b_petsc, self._x)

        # OpenMDAO needs x to be a numpy array, so we need to take the distributed
        # x and "scatter" it (basically gather it to one rank). Once it's
        # gathered into one array, it can be converted to a numpy array and
        # passed out.
        pc = self.ksp.getPC()
        if self.running_mpi and pc.getFactorSolverType() in PC_DISTRIBUTED_TYPES:
            # Create the sequential vector on COMM_SELF so it's not part of the
            # distributed communication and is only on one process.
            if self._x.comm.getRank() == 0:
                x_seq = self._PETSc.Vec().createSeq(self._x.getSize(), comm=self._PETSc.COMM_SELF)
            else:
                x_seq = self._PETSc.Vec().createSeq(0, comm=self._PETSc.COMM_SELF)

            # Have to call scatter on all ranks or MPI will error out (each
            # rank is a process being run in the MPI)
            scatter, _ = self._PETSc.Scatter.toZero(self._x)
            scatter.scatter(self._x, x_seq, addv=self._PETSc.InsertMode.INSERT,
                            mode=self._PETSc.ScatterMode.FORWARD)
            scatter.destroy()

            # Rank 0 owns x, broadcast (or send a copy of) it to the other ranks
            # so that they don't break what OpenMDAO expects from the linear solver
            if self._x.comm.getRank() == 0:
                x_array = x_seq.getArray().copy()
            else:
                x_array = np.empty(self._x.getSize())
            self.comm.Bcast(x_array, root=0)
            return x_array

        # If running in sequence, can just directly copy the whole thing to an array
        else:
            return self._x.getArray().copy()


class PETScDirectSolver(DirectSolver):
    """
    LinearSolver that uses PETSc for LU factor/solve.

    Parameters
    ----------
    **kwargs : dict
        Options dictionary.

    Attributes
    ----------
    _PETSc : <petsc4py.PETSc>
        A lazily imported petsc4py.PETSc module.
    """

    SOLVER = 'LN: PETScDirect'

    def __init__(self, **kwargs):
        """
        Declare the solver options.
        """
        super().__init__(**kwargs)

        try:
            from petsc4py import PETSc
            self._PETSc = PETSc
        except ImportError:
            self._PETSc = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare(
            'sparse_solver_name',
            values=PC_SERIAL_TYPES + PC_DISTRIBUTED_TYPES,
            default='superlu',
            desc="Direct solver algorithm from PETSc that will be used for the "
                 "LU factorization and solve if the matrix is sparse. For a "
                 "dense matrix, this option will be ignored and LAPACK will "
                 "be automatically used."
        )

        # Undeclare and redeclare the "err_on_singular" option so that it's
        # still compatible with shared parent methods. Must always be True
        # because during factorization solvers will always error with a singular.
        self.options.undeclare("err_on_singular")
        self.options.declare(
            'err_on_singular',
            default=True,
            types=bool,
            check_valid=check_err_on_singular,
            desc="Raise an error if LU decomposition is singular. Must always "
                 "be 'True' for the PETScDirectSolver. This option is only "
                 "maintained for compatibility with parent solver methods."
        )

    def check_config(self, logger):
        """
        Perform optional error checks.

        Parameters
        ----------
        logger : object
            The object that manages logging output.
        """
        if self.options['rhs_checking'] is False:
            system = self._system()
            redundant_adj = system.pathname in system._relevance.get_redundant_adjoint_systems()
            if redundant_adj:
                logger.info(
                    f"\n'rhs_checking' is disabled for '{system.linear_solver.msginfo}'"
                    ", but that solver has redundant adjoint solves. If it is "
                    "expensive to compute derivatives for this solver, turning on "
                    "'rhs_checking' may improve performance.\n"
                )

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : <System>
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super(DirectSolver, self)._setup_solvers(system, depth)
        if self.options['sparse_solver_name'] in PC_SERIAL_TYPES:
            self._disallow_distrib_solve()
        self._lin_rhs_checker = LinearRHSChecker.create(self._system(),
                                                        self.options['rhs_checking'])

    def raise_petsc_error(self, e, system, matrix):
        """
        Raise an error based on the issue that PETSc had with the linearize.

        Parameters
        ----------
        e : Error
            Error returned by PETSc.
        system : <System>
            System containing the Directsolver.
        matrix : ndarray
            Matrix of interest.
        """
        if "Zero pivot in LU factorization" in str(e):
            # Zero pivot in LU factorization doesn't necessarily guarantee
            # that the matrix is singular, but not sure what else to raise
            raise RuntimeError(format_singular_error(system, matrix)) from e

        elif "Could not locate solver type" in str(e):
            raise RuntimeError("Specified PETSc sparse solver, "
                               f"'{self.options['sparse_solver_name']}', "
                               "is not installed.") from e

        else:
            # Just raise the original exception
            raise

    def _linearize(self):
        """
        Perform factorization.
        """
        system = self._system()
        nproc = system.comm.size

        if self._assembled_jac is not None:
            matrix = system._assembled_jac.get_dr_do_matrix()

            if matrix is None:
                # this happens if we're not rank 0 when using owned_sizes
                self._lu = self._lup = None

            # Perform dense or sparse lu factorization.
            elif isinstance(matrix, (scipy.sparse.csc_matrix, scipy.sparse.csr_matrix)):
                try:
                    self._lu = PETScLU(matrix, self.options['sparse_solver_name'],
                                       comm=system.comm)
                except self._PETSc.Error as e:
                    self.raise_petsc_error(e, system, matrix)

            elif isinstance(matrix, np.ndarray):  # dense
                try:
                    self._lup = PETScLU(matrix)
                except self._PETSc.Error as e:
                    self.raise_petsc_error(e, system, matrix)

            # Note: calling scipy.sparse.linalg.splu on a COO actually transposes
            # the matrix during conversion to csc prior to LU decomp, so we can't use COO.
            else:
                raise RuntimeError("Direct solver not implemented for matrix type %s"
                                   " in %s." % (type(matrix), system.msginfo))
        else:
            if nproc > 1:
                raise RuntimeError("DirectSolvers without an assembled jacobian are not supported "
                                   "when running under MPI if comm.size > 1.")

            matrix = self._build_mtx()
            try:
                self._lup = PETScLU(matrix)

            except self._PETSc.Error as e:
                self.raise_petsc_error(e, system, matrix)

        if self._lin_rhs_checker is not None:
            self._lin_rhs_checker.clear()

    def solve(self, mode, rel_systems=None):
        """
        Run the solver.

        Parameters
        ----------
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Names of systems relevant to the current solve.  Deprecated.
        """
        system = self._system()
        d_residuals = system._dresiduals
        d_outputs = system._doutputs

        if system.under_complex_step:
            raise RuntimeError('{}: PETScDirectSolver is not supported under '
                               'complex step.'.format(self.msginfo))

        # assign x and b vectors based on mode
        if mode == 'fwd':
            x_vec = d_outputs.asarray()
            b_vec = d_residuals.asarray()
            transpose = False
        else:  # rev
            x_vec = d_residuals.asarray()
            b_vec = d_outputs.asarray()
            transpose = True

            if self._lin_rhs_checker is not None:
                sol_array, is_zero = self._lin_rhs_checker.get_solution(b_vec, system)
                if is_zero:
                    x_vec[:] = 0.0
                    return
                if sol_array is not None:
                    x_vec[:] = sol_array
                    return

        # AssembledJacobians are unscaled.
        if system._get_assembled_jac() is not None:
            full_b = b_vec

            with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
                if isinstance(self._assembled_jac._dr_do_mtx, DenseMatrix):
                    sol_array = self._lup.solve(full_b, transpose=transpose)
                    matrix = self._lup.orig_A
                else:
                    sol_array = self._lu.solve(full_b, transpose=transpose)
                    matrix = self._lu.orig_A

                x_vec[:] = sol_array

        # matrix-vector-product generated jacobians are scaled.
        else:
            x_vec[:] = sol_array = self._lup.solve(b_vec, transpose=transpose)
            matrix = self._lup.orig_A

        # Detect singularities (PETSc linear solver "solve" doesn't error out
        # with NaN and inf so need to check for them).
        if np.isinf(x_vec).any() or np.isnan(x_vec).any():
            raise RuntimeError(format_singular_error(system, matrix))

        if not system.under_complex_step and self._lin_rhs_checker is not None and mode == 'rev':
            self._lin_rhs_checker.add_solution(b_vec, sol_array, system, copy=True)

    def preferred_sparse_format(self):
        """
        Return the preferred sparse format for the dr/do matrix of a split jacobian.

        Returns
        -------
        str
            The preferred sparse format for the dr/do matrix of a split jacobian.
        """
        return 'csr'
