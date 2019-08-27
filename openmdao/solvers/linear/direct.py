"""LinearSolver that uses linalg.solve or LU factor/solve."""

from __future__ import division, print_function

import sys
import warnings
from six import reraise, PY2
from six.moves import range

import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from openmdao.solvers.solver import LinearSolver
from openmdao.matrices.coo_matrix import COOMatrix
from openmdao.matrices.csr_matrix import CSRMatrix
from openmdao.matrices.csc_matrix import CSCMatrix
from openmdao.matrices.dense_matrix import DenseMatrix
from openmdao.utils.mpi import MPI
from openmdao.utils.array_utils import sizes2offsets
from openmdao.vectors.vector import INT_DTYPE


def loc2error_msg(system, loc_txt, loc):
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
    start = end = 0
    varname = "Unknown"
    varsizes = system._var_sizes['nonlinear']['output']
    var_inds = system._var_allprocs_abs2idx['nonlinear']
    for name in system._var_abs_names['output']:
        if name in system._var_abs2meta:
            end += varsizes[system._owning_rank[name]][var_inds[name]]
            if loc < end:
                varname = system._var_abs2prom['output'][name]
                break
            start = end

    if varname == name:
        names = "'{}' index {}.".format(varname, loc - start)
    else:
        names = "'{}' ('{}') index {}.".format(varname, name, loc - start)

    msg = "Singular entry found in {} for {} associated with state/residual " + names
    return msg.format(system.msginfo, loc_txt)


def format_singular_error(err, system, mtx):
    """
    Format a coherent error message when the matrix is singular.

    Parameters
    ----------
    err : Exception
        Exception object
    system : <System>
        System containing the Directsolver.
    mtx : ndarray
        Matrix of interest.

    Returns
    -------
    str
        New error string.
    """
    if PY2:
        err_msg = err.message
    else:
        err_msg = err.args[0]

    loc = int(err_msg.split('number ')[1].split(' is exactly')[0])

    # Lapack:DGETRF outputs INFO, which uses fortran numbering.
    loc -= 1

    col_norm = np.linalg.norm(mtx[:, loc - 1])
    row_norm = np.linalg.norm(mtx[loc - 1, :])

    if row_norm <= col_norm:
        loc_txt = "row"
    else:
        loc_txt = "column"

    return loc2error_msg(system, loc_txt, loc)


def format_singular_csc_error(system, matrix):
    """
    Format a coherent error message when the CSC matrix is singular.

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
    dense = matrix.toarray()
    zero_rows = np.where(~dense.any(axis=1))[0]
    zero_cols = np.where(~dense.any(axis=0))[0]
    if np.any(np.isnan(dense)):
        # There is a nan in the matrix.
        return(format_nan_error(system, dense))
    elif zero_cols.size <= zero_rows.size:

        if zero_rows.size == 0:
            # Underdetermined: duplicate columns or rows.
            msg = "Identical rows or columns found in jacobian in '{}'. Problem is " + \
                  "underdetermined."
            return msg.format(system.pathname)

        loc_txt = "row"
        loc = zero_rows[0]
    else:
        loc_txt = "column"
        loc = zero_cols[0]

    return loc2error_msg(system, loc_txt, loc)


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
    rows = set(np.where(np.isnan(matrix))[0])

    # Because of how we built the matrix, a NaN in a comp cause the whole row to be NaN, so we
    # need to associate each index with a variable.
    varname = []
    all_vars = system._var_allprocs_abs_names['output']
    varsizes = system._var_sizes['nonlinear']['output']
    for row in rows:
        n = 0
        for j, name in enumerate(all_vars):
            n += varsizes[system._owning_rank[name]][j]
            if row < n:
                relname = system._var_abs2prom['output'][name]
                varname.append("'%s'" % relname)
                break

    msg = "NaN entries found in {} for rows associated with states/residuals [{}]."
    return msg.format(system.msginfo, ', '.join(varname))


class DirectSolver(LinearSolver):
    """
    LinearSolver that uses linalg.solve or LU factor/solve.
    """

    SOLVER = 'LN: Direct'

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super(DirectSolver, self)._declare_options()

        self.options.declare('err_on_singular', types=bool, default=True,
                             desc="Raise an error if LU decomposition is singular.")

        # this solver does not iterate
        self.options.undeclare("maxiter")
        self.options.undeclare("err_on_maxiter")    # Deprecated option.
        self.options.undeclare("err_on_non_converge")

        self.options.undeclare("atol")
        self.options.undeclare("rtol")

        # Use an assembled jacobian by default.
        self.options['assemble_jac'] = True

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : System
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super(DirectSolver, self)._setup_solvers(system, depth)
        self._local2owned_inds = None
        self._owned_size_totals = None

    def _linearize_children(self):
        """
        Return a flag that is True when we need to call linearize on our subsystems' solvers.

        Returns
        -------
        boolean
            Flag for indicating child linearization.
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
        system = self._system
        bvec = system._vectors['residual']['linear']
        xvec = system._vectors['output']['linear']

        # First make a backup of the vectors
        b_data = bvec._data.copy()
        x_data = xvec._data.copy()

        nmtx = x_data.size
        eye = np.eye(nmtx)
        mtx = np.empty((nmtx, nmtx), dtype=b_data.dtype)
        scope_out, scope_in = system._get_scope()
        vnames = ['linear']

        # Assemble the Jacobian by running the identity matrix through apply_linear
        for i in range(nmtx):
            # set value of x vector to provided value
            xvec._data[:] = eye[:, i]

            # apply linear
            system._apply_linear(self._assembled_jac, vnames, self._rel_systems, 'fwd',
                                 scope_out, scope_in)

            # put new value in out_vec
            mtx[:, i] = bvec._data

        # Restore the backed-up vectors
        bvec._data[:] = b_data
        xvec._data[:] = x_data

        return mtx

    def _linearize(self):
        """
        Perform factorization.
        """
        system = self._system

        if self._assembled_jac is not None:

            if system.comm.size == 1:
                matrix = self._assembled_jac._int_mtx._matrix
            else:
                matrix = self._assembled_jac._int_mtx._get_assembled_matrix(system)

            if system.comm.size > 1 and self._owned_size_totals is None:
                self._owned_size_totals = np.sum(system._owned_sizes, axis=1)

            if matrix is None:
                # this happens if we're not rank 0
                self._lu = self._lup = None
                sz = np.sum(system._owned_sizes)
                self._nodup_size = sz

            # Perform dense or sparse lu factorization.
            elif isinstance(matrix, csc_matrix):
                try:
                    self._lu = scipy.sparse.linalg.splu(matrix)
                    self._nodup_size = matrix.shape[1]
                except RuntimeError as err:
                    if 'exactly singular' in str(err):
                        raise RuntimeError(format_singular_csc_error(system, matrix))
                    else:
                        reraise(*sys.exc_info())

            elif isinstance(matrix, np.ndarray):  # dense
                # During LU decomposition, detect singularities and warn user.
                with warnings.catch_warnings():
                    if self.options['err_on_singular']:
                        warnings.simplefilter('error', RuntimeWarning)
                    try:
                        self._lup = scipy.linalg.lu_factor(matrix)
                        self._nodup_size = matrix.shape[1]
                    except RuntimeWarning as err:
                        raise RuntimeError(format_singular_error(err, system, matrix))

                    # NaN in matrix.
                    except ValueError as err:
                        raise RuntimeError(format_nan_error(system, matrix))

            # Note: calling scipy.sparse.linalg.splu on a COO actually transposes
            # the matrix during conversion to csc prior to LU decomp, so we can't use COO.
            else:
                raise RuntimeError("Direct solver not implemented for matrix type %s"
                                   " in %s." % (type(self._assembled_jac._int_mtx), system.msginfo))

        else:
            if system.comm.size > 1:
                raise RuntimeError("DirectSolvers without an assembled jacobian are not supported "
                                   "when running under MPI if comm.size > 1.")

            mtx = self._build_mtx()
            self._nodup_size = mtx.shape[1]

            # During LU decomposition, detect singularities and warn user.
            with warnings.catch_warnings():

                if self.options['err_on_singular']:
                    warnings.simplefilter('error', RuntimeWarning)

                try:
                    self._lup = scipy.linalg.lu_factor(mtx)

                except RuntimeWarning as err:
                    raise RuntimeError(format_singular_error(err, system, mtx))

                # NaN in matrix.
                except ValueError as err:
                    raise RuntimeError(format_nan_error(system, mtx))

    def _inverse(self):
        """
        Return the inverse Jacobian.

        This is only used by the Broyden solver when calculating a full model Jacobian. Since it
        is only done for a single RHS, no need for LU.

        Returns
        -------
        ndarray
            Inverse Jacobian.
        """
        system = self._system

        if self._assembled_jac is not None:

            if system.comm.size == 1:
                matrix = self._assembled_jac._int_mtx._matrix
            else:
                matrix = self._assembled_jac._int_mtx._get_assembled_matrix(system)

            # Dense and Sparse matrices have their own inverse method.
            if isinstance(matrix, np.ndarray):
                # Detect singularities and warn user.
                with warnings.catch_warnings():
                    if self.options['err_on_singular']:
                        warnings.simplefilter('error', RuntimeWarning)
                    try:
                        inv_jac = scipy.linalg.inv(matrix)
                    except RuntimeWarning as err:
                        raise RuntimeError(format_singular_error(err, system, matrix))

                    # NaN in matrix.
                    except ValueError as err:
                        raise RuntimeError(format_nan_error(system, matrix))

            elif isinstance(matrix, csc_matrix):
                try:
                    inv_jac = scipy.sparse.linalg.inv(matrix)
                except RuntimeError as err:
                    if 'exactly singular' in str(err):
                        raise RuntimeError(format_singular_csc_error(system, matrix))
                    else:
                        reraise(*sys.exc_info())
            else:
                raise RuntimeError("Direct solver not implemented for matrix type %s"
                                   " in %s." % (type(matrix), system.msginfo))

        else:
            mtx = self._build_mtx()

            # During inversion detect singularities and warn user.
            with warnings.catch_warnings():

                if self.options['err_on_singular']:
                    warnings.simplefilter('error', RuntimeWarning)

                try:
                    inv_jac = scipy.linalg.inv(mtx)

                except RuntimeWarning as err:
                    raise RuntimeError(format_singular_error(err, system, mtx))

                # NaN in matrix.
                except ValueError as err:
                    raise RuntimeError(format_nan_error(system, mtx))

        return inv_jac

    def solve(self, vec_names, mode, rel_systems=None):
        """
        Run the solver.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Names of systems relevant to the current solve.
        """
        if len(vec_names) > 1 or vec_names[0] != 'linear':
            raise RuntimeError("DirectSolvers with multiple right-hand-sides are not supported.")

        self._vec_names = vec_names

        system = self._system
        iproc = system.comm.rank

        d_residuals = system._vectors['residual']['linear']
        d_outputs = system._vectors['output']['linear']

        # assign x and b vectors based on mode
        if mode == 'fwd':
            x_vec = d_outputs._data
            b_vec = d_residuals._data
            trans_lu = 0
            trans_splu = 'N'
        else:  # rev
            x_vec = d_residuals._data
            b_vec = d_outputs._data
            trans_lu = 1
            trans_splu = 'T'

        # AssembledJacobians are unscaled.
        if self._assembled_jac is not None:
            if system.comm.size > 1:
                _, nodup2local_inds, local2owned_inds, noncontig_dist_inds = \
                    system._get_nodup_out_ranges()
                # gather full_b
                tmp = np.empty(self._nodup_size, dtype=b_vec.dtype)
                mpi_typ = MPI.C_DOUBLE_COMPLEX if np.iscomplex(b_vec[0]) else MPI.DOUBLE
                disps = sizes2offsets(self._owned_size_totals, dtype=INT_DTYPE)
                system.comm.Gatherv((b_vec[local2owned_inds], local2owned_inds.size, mpi_typ),
                                    (tmp, (self._owned_size_totals, disps), mpi_typ),
                                    root=0)
            else:
                full_b = tmp = b_vec

            with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
                if iproc == 0:
                    # convert full_b to the same ordering that the matrix expects, where
                    # dist vars are contiguous.
                    # TODO: only do this if distributed vars are found to avoid unnecessary
                    # inefficiency
                    if system.comm.size > 1:
                        full_b = tmp[noncontig_dist_inds]

                    if isinstance(self._assembled_jac._int_mtx, DenseMatrix):
                        arr = scipy.linalg.lu_solve(self._lup, full_b, trans=trans_lu)
                    else:
                        arr = self._lu.solve(full_b, trans_splu)

                if system.comm.size > 1:
                    if iproc > 0:
                        arr = np.zeros(tmp.size, dtype=tmp.dtype)

                    # this may send more data than necessary, but the alternative is to use a lot
                    # of memory on rank 0 to store the chunk that each proc needs and then do a
                    # Scatterv.
                    system.comm.Bcast((arr, mpi_typ), root=0)
                    x_vec[:] = arr[nodup2local_inds]
                else:
                    x_vec[:] = arr

        # matrix-vector-product generated jacobians are scaled.
        else:
            x_vec[:] = scipy.linalg.lu_solve(self._lup, b_vec, trans=trans_lu)
