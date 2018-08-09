"""LinearSolver that uses linalg.solve or LU factor/solve."""

from __future__ import division, print_function

import sys
import warnings
from six import reraise, PY2
from six.moves import range

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from openmdao.solvers.solver import LinearSolver
from openmdao.matrices.coo_matrix import COOMatrix
from openmdao.matrices.csr_matrix import CSRMatrix
from openmdao.matrices.csc_matrix import CSCMatrix
from openmdao.matrices.dense_matrix import DenseMatrix
from openmdao.recorders.recording_iteration_stack import Recording


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

    n = 0
    varname = "Unknown"
    varsizes = system._var_sizes['nonlinear']['output']
    for j, name in enumerate(system._var_allprocs_abs_names['output']):
        n += varsizes[system._owning_rank[name]][j]
        if loc < n:
            varname = system._var_abs2prom['output'][name]
            break

    msg = "Singular entry found in '{}' for {} associated with state/residual '{}'."
    return msg.format(system.pathname, loc_txt, varname)


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
        loc_txt = "row"
        loc = zero_rows[0]
    else:
        loc_txt = "column"
        loc = zero_cols[0]

    n = 0
    varname = "Unknown"
    varsizes = system._var_sizes['nonlinear']['output']
    for j, name in enumerate(system._var_allprocs_abs_names['output']):
        n += varsizes[system._owning_rank[name]][j]
        if loc < n:
            varname = system._var_abs2prom['output'][name]
            break

    msg = "Singular entry found in '{}' for {} associated with state/residual '{}'."
    return msg.format(system.pathname, loc_txt, varname)


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

    msg = "NaN entries found in '{}' for rows associated with states/residuals [{}]."
    return msg.format(system.pathname, ', '.join(varname))


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

        self.options.declare('err_on_singular', default=True,
                             desc="Raise an error if LU decomposition is singular.")

        # this solver does not iterate
        self.options.undeclare("maxiter")
        self.options.undeclare("err_on_maxiter")

        self.options.undeclare("atol")
        self.options.undeclare("rtol")

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
        mtx = np.empty((nmtx, nmtx))
        scope_out, scope_in = system._get_scope()
        vnames = ['linear']

        # Assemble the Jacobian by running the identity matrix through apply_linear
        for i in range(nmtx):
            # set value of x vector to provided value
            xvec.set_data(eye[:, i])

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

            mtx = self._assembled_jac._int_mtx
            ranges = self._assembled_jac._view_ranges[system.pathname]
            matrix = mtx._matrix[ranges[0]:ranges[1], ranges[0]:ranges[1]]

            # Perform dense or sparse lu factorization
            if isinstance(mtx, DenseMatrix):
                # During LU decomposition, detect singularities and warn user.
                with warnings.catch_warnings():
                    if self.options['err_on_singular']:
                        warnings.simplefilter('error', RuntimeWarning)
                    try:
                        self._lup = scipy.linalg.lu_factor(matrix)
                    except RuntimeWarning as err:
                        raise RuntimeError(format_singular_error(err, system, matrix))

                    # NaN in matrix.
                    except ValueError as err:
                        raise RuntimeError(format_nan_error(system, matrix))

            elif isinstance(mtx, (CSRMatrix, CSCMatrix)):
                try:
                    self._lu = scipy.sparse.linalg.splu(matrix)
                except RuntimeError as err:
                    if 'exactly singular' in str(err):
                        raise RuntimeError(format_singular_csc_error(system, matrix))
                    else:
                        reraise(*sys.exc_info())

            elif isinstance(mtx, COOMatrix):
                # calling scipy.sparse.linalg.splu on a COO actually transposes
                # the matrix during conversion to csc prior to LU decomp
                raise RuntimeError("Direct solver is not compatible with matrix type "
                                   "COOMatrix in system '%s'." % system.pathname)
            else:
                raise RuntimeError("Direct solver not implemented for matrix type %s"
                                   " in system '%s'." % (type(mtx), system.pathname))

        else:
            mtx = self._build_mtx()

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

            mtx = self._assembled_jac._int_mtx
            ranges = self._assembled_jac._view_ranges[system.pathname]
            matrix = mtx._matrix[ranges[0]:ranges[1], ranges[0]:ranges[1]]

            # Dense and Sparse matrices have their own inverse method.
            if isinstance(mtx, DenseMatrix):
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

            elif isinstance(mtx, (CSRMatrix, CSCMatrix)):
                try:
                    inv_jac = scipy.sparse.linalg.inv(matrix)
                except RuntimeError as err:
                    if 'exactly singular' in str(err):
                        raise RuntimeError(format_singular_csc_error(system, matrix))
                    else:
                        reraise(*sys.exc_info())
            else:
                raise RuntimeError("Direct solver not implemented for matrix type %s"
                                   " in system '%s'." % (type(mtx), system.pathname))

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

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        if len(vec_names) > 1:
            raise RuntimeError("DirectSolvers with multiple right-hand-sides are not supported.")

        self._vec_names = vec_names

        system = self._system

        with Recording('DirectSolver', 0, self) as rec:
            for vec_name in vec_names:
                if vec_name not in system._rel_vec_names:
                    continue
                self._vec_name = vec_name
                d_residuals = system._vectors['residual'][vec_name]
                d_outputs = system._vectors['output'][vec_name]

                # assign x and b vectors based on mode
                if mode == 'fwd':
                    x_vec = d_outputs
                    b_vec = d_residuals
                    trans_lu = 0
                    trans_splu = 'N'
                else:  # rev
                    x_vec = d_residuals
                    b_vec = d_outputs
                    trans_lu = 1
                    trans_splu = 'T'

                # AssembledJacobians are unscaled.
                if self._assembled_jac is not None:
                    with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
                        if (isinstance(self._assembled_jac._int_mtx,
                                       (COOMatrix, CSRMatrix, CSCMatrix))):
                            x_vec._data[:] = self._lu.solve(b_vec._data, trans_splu)
                        else:
                            x_vec._data[:] = scipy.linalg.lu_solve(self._lup, b_vec._data,
                                                                   trans=trans_lu)

                # MVP-generated jacobians are scaled.
                else:
                    x_vec._data[:] = scipy.linalg.lu_solve(self._lup, b_vec._data, trans=trans_lu)

                rec.abs = 0.0
                rec.rel = 0.0

        return False, 0., 0.
