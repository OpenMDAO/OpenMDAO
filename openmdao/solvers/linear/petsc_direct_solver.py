"""LinearSolver that uses PETSc for LU factor/solve."""

import warnings

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.sparse

from openmdao.solvers.solver import LinearSolver
from openmdao.matrices.dense_matrix import DenseMatrix
from openmdao.utils.array_utils import identity_column_iter
from openmdao.solvers.linear.linear_rhs_checker import LinearRHSChecker
from openmdao.utils.om_warnings import issue_warning, SolverWarning

try:
    from petsc4py import PETSc
    DEFAULT_COMM = PETSc.COMM_WORLD
except ImportError:
    PETSc = None
    DEFAULT_COMM = None

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
    """

    def __init__(self, A: scipy.sparse.spmatrix, sparse_solver_name: str = None,
                 comm=DEFAULT_COMM):
        """
        Initialize and setup the PETSc LU Direct Solver object.
        """
        self.comm = comm
        self.running_mpi = not comm.size == 1
        self.orig_A = A
        # Create PETSc matrix
        # Dense
        if isinstance(A, np.ndarray):
            self.A = PETSc.Mat().createDense(A.shape, array=A)

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
                self.A = PETSc.Mat().create(comm=comm)
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
                # Serial: build full CSR
                self.A = PETSc.Mat().createAIJ(
                    size=A.shape,
                    csr=(A.indptr, A.indices, A.data)
                )

            self.A.assemble()

        # Create PETSc solver (KSP is the iterative linear solver [Krylov SPace
        # solver] and PC is the preconditioner)
        self.ksp = PETSc.KSP().create()
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
        # gathered into onen array, it can be converted to a numpy array and
        # passed out.
        pc = self.ksp.getPC()
        if self.running_mpi and pc.getFactorSolverType() in PC_DISTRIBUTED_TYPES:
            # Create the sequential vector on COMM_SELF so it's not part of the
            # distributed communication and is only on one process.
            if self._x.comm.getRank() == 0:
                x_seq = PETSc.Vec().createSeq(self._x.getSize(), comm=PETSc.COMM_SELF)
            else:
                x_seq = PETSc.Vec().createSeq(0, comm=PETSc.COMM_SELF)

            # Have to call scatter on all ranks or MPI will error out (each
            # rank is a processing being run in the MPI)
            scatter, _ = PETSc.Scatter.toZero(self._x)
            scatter.scatter(self._x, x_seq, addv=PETSc.InsertMode.INSERT,
                            mode=PETSc.ScatterMode.FORWARD)
            scatter.destroy()

            # Rank 0 owns x, broadcast (or send a copy) of it to the other ranks
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


def index_to_varname(system, loc):
    """
    Given a matrix location, return the name of the variable associated with that index.

    Parameters
    ----------
    system : <System>
        System containing the Directsolver.
    loc : int
        Index of row or column.

    Returns
    -------
    str
        String containing variable absolute name (and promoted name if there is
        one) and index.
    """
    start = end = 0
    varsizes = np.sum(system._owned_output_sizes, axis=0)
    for i, name in enumerate(system._resolver.abs_iter('output')):
        end += varsizes[i]
        if loc < end:
            varname = system._resolver.abs2prom(name, 'output')
            break
        start = end

    if varname == name:
        name_string = "'{}' index {}.".format(varname, loc - start)
    else:
        name_string = "'{}' ('{}') index {}.".format(varname, name, loc - start)

    return name_string


def loc_to_error_msg(system, loc_txt, loc):
    """
    Given a matrix location, format a coherent error message when matrix is singular.

    Parameters
    ----------
    system : <System>
        System containing the Directsolver.
    loc_txt : str
        Either 'row' or 'col'.
    loc : int
        Index of row or column.

    Returns
    -------
    str
        New error string.
    """
    names = index_to_varname(system, loc)
    msg = "Singular entry found in {} for {} associated with state/residual " + names
    return msg.format(system.msginfo, loc_txt)


def format_singular_error(system, matrix):
    """
    Format a coherent error message for any ill-conditioned mmatrix.

    Parameters
    ----------
    system : <System>
        System containing the Directsolver.
    matrix : ndarray
        Matrix of interest.

    Returns
    -------
    str
        New error string.
    """
    if scipy.sparse.issparse(matrix):
        matrix = matrix.toarray()

    if np.any(np.isnan(matrix)):
        # There is a nan in the matrix.
        return format_nan_error(system, matrix)

    zero_rows = np.where(~matrix.any(axis=1))[0]
    zero_cols = np.where(~matrix.any(axis=0))[0]
    if zero_cols.size <= zero_rows.size:

        if zero_rows.size == 0:
            # In this case, some row is a linear combination of the other rows.

            # SVD gives us some information that may help locate the source of
            # the problem.
            try:
                u, _, _ = np.linalg.svd(matrix)

            except Exception:
                msg = f"Jacobian in '{system.pathname}' is not full rank, but OpenMDAO was " + \
                      "not able to determine which rows or columns."
                return msg

            # Nonzero elements in the left singular vector show the rows that
            # contribute strongly to the singular subspace. Note that sometimes
            # extra rows/cols are included in the set, currently don't have a
            # good way to pare them down.
            tol = 1e-15
            u_sing = np.abs(u[:, -1])
            left_idx = np.where(u_sing > tol)[0]

            msg = "Jacobian in '{}' is not full rank. The following set of states/residuals " + \
                  "contains one or more equations that is a linear combination of the others: \n"

            for loc in left_idx:
                name = index_to_varname(system, loc)
                msg += ' ' + name + '\n'

            if len(left_idx) > 2:
                msg += "Note that the problem may be in a single Component."

            return msg.format(system.pathname)

        loc_txt = "row"
        loc = zero_rows[0]
    else:
        loc_txt = "column"
        loc = zero_cols[0]

    return loc_to_error_msg(system, loc_txt, loc)


def format_nan_error(system, matrix):
    """
    Format a coherent error message when the matrix contains NaN.

    Parameters
    ----------
    system : <System>
        System containing the Directsolver.
    matrix : ndarray
        Matrix of interest.

    Returns
    -------
    str
        New error string.
    """
    # Because of how we built the matrix, a NaN in a comp causes the whole row
    # to be NaN, so we need to associate each index with a variable.
    varsizes = np.sum(system._owned_output_sizes, axis=0)

    nanrows = np.zeros(matrix.shape[0], dtype=bool)
    nanrows[np.where(np.isnan(matrix))[0]] = True

    varnames = []
    start = end = 0
    for i, name in enumerate(system._resolver.abs_iter('output')):
        end += varsizes[i]
        if np.any(nanrows[start:end]):
            varnames.append("'%s'" % system._resolver.abs2prom(name, 'output'))
        start = end

    msg = "NaN entries found in {} for rows associated with states/residuals [{}]."
    return msg.format(system.msginfo, ', '.join(varnames))


class PETScDirectSolver(LinearSolver):
    """
    LinearSolver that uses PETSc for LU factor/solve.

    Parameters
    ----------
    **kwargs : dict
        Options dictionary.

    Attributes
    ----------
    _lin_rhs_checker : LinearRHSChecker or None
        Object for checking the right-hand side of the linear solve.
    """

    SOLVER = 'LN: PETScDirect'

    def __init__(self, **kwargs):
        """
        Declare the solver options.
        """
        super().__init__(**kwargs)

        if PETSc is None:
            raise RuntimeError(f"{self.msginfo}: PETSc is not available. ")

        self._lin_rhs_checker = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare(
            'rhs_checking',
            types=(bool, dict),
            default=False,
            desc="If True, check RHS vs. cache and/or zero to avoid some solves."
                 "Can also be set to a dict of options for the LinearRHSChecker to "
                 "allow finer control over it. Allowed options are: "
                 f"{LinearRHSChecker.options}"
        )

        self.options.declare(
            'sparse_solver_name',
            values=PC_SERIAL_TYPES + PC_DISTRIBUTED_TYPES,
            default='superlu',
            desc="Direct solver algorithm from PETSc that will be used for the "
                 "LU factorization and solve if the matrix is sparse. For a "
                 "dense matrix, this option will be ignored and LAPACK will "
                 "be automatically used."
        )

        # this solver does not iterate
        self.options.undeclare("maxiter")
        self.options.undeclare("err_on_non_converge")

        self.options.undeclare("atol")
        self.options.undeclare("rtol")

        # Use an assembled jacobian by default.
        self.options['assemble_jac'] = True

        self.supports['implicit_components'] = True

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
        super()._setup_solvers(system, depth)
        if self.options['sparse_solver_name'] in PC_SERIAL_TYPES:
            self._disallow_distrib_solve()
        self._lin_rhs_checker = LinearRHSChecker.create(self._system(),
                                                        self.options['rhs_checking'])

    def _linearize_children(self):
        """
        Return a flag that is True when we need to call linearize on our subsystems' solvers.

        Returns
        -------
        boolean
            Flag for indicating child linearization.
        """
        return False

    def can_solve_cycle(self):
        """
        Return True if this solver can solve groups with cycles.

        Returns
        -------
        bool
            True if this solver can solve groups with cycles.
        """
        return True

    def use_relevance(self):
        """
        Return True if relevance should be active.

        Returns
        -------
        bool
            True if relevance should be active.
        """
        return False

    def _build_mtx(self):
        """
        Assemble a Jacobian matrix by matrix-vector-product with columns of identity.

        Returns
        -------
        ndarray
            Jacobian matrix.
        """
        system = self._system()
        bvec = system._dresiduals
        xvec = system._doutputs

        # First make a backup of the vectors
        b_data = bvec.asarray(copy=True)
        x_data = xvec.asarray(copy=True)

        nmtx = x_data.size
        seed = np.zeros(x_data.size)
        mtx = np.empty((nmtx, nmtx), dtype=b_data.dtype)
        scope_out, scope_in = system._get_matvec_scope()

        # temporarily disable relevance to avoid creating a singular matrix
        with system._relevance.active(False):
            # Assemble the Jacobian by running the identity matrix through apply_linear
            for i, seed in enumerate(identity_column_iter(seed)):
                # set value of x vector to provided value
                xvec.set_val(seed)

                # apply linear
                system._apply_linear(self._assembled_jac, 'fwd', scope_out, scope_in)

                # put new value in out_vec
                mtx[:, i] = bvec.asarray()

        # Restore the backed-up vectors
        bvec.set_val(b_data)
        xvec.set_val(x_data)

        return mtx

    def _linearize(self):
        """
        Perform factorization.
        """
        system = self._system()
        nproc = system.comm.size

        if self._assembled_jac is not None:
            matrix = self._assembled_jac._int_mtx._matrix

            if matrix is None:
                # this happens if we're not rank 0 when using owned_sizes
                self._lu = self._lup = None

            # Perform dense or sparse lu factorization.
            elif isinstance(matrix, scipy.sparse.csc_matrix):
                try:
                    self._lu = PETScLU(matrix, self.options['sparse_solver_name'],
                                       comm=system.comm)
                except RuntimeError:
                    # Zero pivot in LU factorization doesn't necessarily guarantee
                    # that the matrix is singular, but not sure what else to raise
                    raise RuntimeError(format_singular_error(system, matrix))

            elif isinstance(matrix, np.ndarray):  # dense
                try:
                    self._lup = PETScLU(matrix)
                except PETSc.Error:
                    # Zero pivot in LU factorization doesn't necessarily guarantee
                    # that the matrix is singular, but not sure what else to raise
                    raise RuntimeError(format_singular_error(system, matrix))

            # Note: calling scipy.sparse.linalg.splu on a COO actually transposes
            # the matrix during conversion to csc prior to LU decomp, so we can't use COO.
            else:
                raise RuntimeError("Direct solver not implemented for matrix type %s"
                                   " in %s." % (type(self._assembled_jac._int_mtx),
                                                system.msginfo))
        else:
            if nproc > 1:
                raise RuntimeError("DirectSolvers without an assembled jacobian are not supported "
                                   "when running under MPI if comm.size > 1.")

            matrix = self._build_mtx()
            try:
                self._lup = PETScLU(matrix)

            except PETSc.Error:
                # Zero pivot in LU factorization doesn't necessarily guarantee
                # that the matrix is singular, but not sure what else to raise
                raise RuntimeError(format_singular_error(system, matrix))

        if self._lin_rhs_checker is not None:
            self._lin_rhs_checker.clear()

    def _inverse(self):
        """
        Return the inverse Jacobian.

        This is only used by the Broyden solver when calculating a full model
        Jacobian. Since it is only done for a single RHS, no need for LU.

        Returns
        -------
        ndarray
            Inverse Jacobian.
        """
        system = self._system()
        nproc = system.comm.size

        if self._assembled_jac is not None:

            matrix = self._assembled_jac._int_mtx._matrix

            if matrix is None:
                # This happens if we're not rank 0 and owned_sizes are being used
                sz = np.sum(system._owned_output_sizes)
                inv_jac = np.zeros((sz, sz))

            # Dense and Sparse matrices have their own inverse method.
            elif isinstance(matrix, np.ndarray):
                # Detect singularities and warn user.
                with warnings.catch_warnings():
                    try:
                        inv_jac = scipy.linalg.inv(matrix)
                    except RuntimeWarning:
                        raise RuntimeError(format_singular_error(system, matrix))

                    # NaN in matrix.
                    except ValueError:
                        raise RuntimeError(format_nan_error(system, matrix))

            elif isinstance(matrix, scipy.sparse.csc_matrix):
                try:
                    inv_jac = scipy.sparse.linalg.inv(matrix)
                except RuntimeError:
                    raise RuntimeError(format_singular_error(system, matrix))

                # to prevent broadcasting errors later, make sure inv_jac is 2D
                # scipy.sparse.linalg.inv returns a shape (1,) array if matrix
                # is shape (1,1)
                if inv_jac.size == 1:
                    inv_jac = inv_jac.reshape((1, 1))
            else:
                raise RuntimeError("Direct solver not implemented for matrix type %s"
                                   " in %s." % (type(matrix), system.msginfo))

        else:
            if nproc > 1:
                raise RuntimeError("BroydenSolvers without an assembled jacobian "
                                   "are not supported when running under MPI if "
                                   "comm.size > 1.")
            mtx = self._build_mtx()

            # During inversion detect singularities and warn user.
            with warnings.catch_warnings():
                try:
                    inv_jac = scipy.linalg.inv(mtx)

                except RuntimeWarning:
                    raise RuntimeError(format_singular_error(system, mtx))

                # NaN in matrix.
                except ValueError:
                    raise RuntimeError(format_nan_error(system, mtx))

        return inv_jac

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
        if self._assembled_jac is not None:
            full_b = b_vec

            with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
                if isinstance(self._assembled_jac._int_mtx, DenseMatrix):
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

        # Detect singularities (PETSc linear solvers don't error out with NaN
        # and inf so need to check for them).
        if np.isinf(x_vec).any() or np.isnan(x_vec).any():
            raise RuntimeError(format_singular_error(system, matrix))

        if not system.under_complex_step and self._lin_rhs_checker is not None and mode == 'rev':
            self._lin_rhs_checker.add_solution(b_vec, sol_array, copy=True)
