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


def format_singluar_error(err, system, mtx):
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
    for name in system._var_allprocs_abs_names['output']:
        n += len(system._outputs._views_flat[name])
        if loc <= n:
            varname = name
            break

    msg = "Singular entry found in '{}' for {} associated with state/residual '{}'."
    return msg.format(system.pathname, loc_txt, varname)


def format_singluar_csc_error(system, matrix):
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
    if zero_cols.size <= zero_rows.size:
        loc_txt = "row"
        loc = zero_rows[0]
    else:
        loc_txt = "column"
        loc = zero_cols[0]

    n = 0
    varname = "Unknown"
    for name in system._var_allprocs_abs_names['output']:
        relname = system._var_abs2prom['output'][name]
        n += len(system._outputs[relname])
        if loc <= n:
            varname = relname
            break

    msg = "Singular entry found in '{}' for {} associated with state/residual '{}'."
    return msg.format(system.pathname, loc_txt, varname)


class DirectSolver(LinearSolver):
    """
    LinearSolver that uses linalg.solve or LU factor/solve.
    """

    SOLVER = 'LN: Direct'

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('err_on_singular', default=True,
                             desc="Raise an error if LU decomposition is singular.")

        # this solver does not iterate
        self.options.undeclare("maxiter")
        self.options.undeclare("err_on_maxiter")

    def _linearize(self):
        """
        Perform factorization.
        """
        system = self._system

        if system._owns_assembled_jac or system._views_assembled_jac:
            mtx = system._jacobian._int_mtx
            # Perform dense or sparse lu factorization
            if isinstance(mtx, DenseMatrix):
                if system._views_assembled_jac:
                    ranges = system._jacobian._view_ranges[system.pathname]
                    matrix = mtx._matrix[ranges[0]:ranges[1], ranges[0]:ranges[1]]
                else:
                    matrix = mtx._matrix

                # During LU decomposition, detect singularities and warn user.
                with warnings.catch_warnings():

                    if self.options['err_on_singular']:
                        warnings.simplefilter('error', RuntimeWarning)

                    try:
                        self._lup = scipy.linalg.lu_factor(matrix)

                    except RuntimeWarning as err:
                        raise RuntimeError(format_singluar_error(err, system, matrix))

            elif isinstance(mtx, (CSRMatrix, CSCMatrix)):
                if system._views_assembled_jac:
                    ranges = system._jacobian._view_ranges[system.pathname]
                    matrix = mtx._matrix[ranges[0]:ranges[1], ranges[0]:ranges[1]]
                else:
                    matrix = mtx._matrix
                try:
                    self._lu = scipy.sparse.linalg.splu(matrix)
                except RuntimeError as err:
                    if 'exactly singular' in str(err):
                        raise RuntimeError(format_singluar_csc_error(system, matrix))
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
            # First make a backup of the vectors
            b_data = system._vectors['residual']['linear'].get_data()
            x_data = system._vectors['output']['linear'].get_data()

            # Assemble the Jacobian by running the identity matrix through apply_linear
            nmtx = x_data.size
            eye = np.eye(nmtx)
            mtx = np.empty((nmtx, nmtx))
            for i in range(nmtx):
                self._mat_vec(eye[:, i], mtx[:, i])

            # Restore the backed-up vectors
            system._vectors['residual']['linear'].set_data(b_data)
            system._vectors['output']['linear'].set_data(x_data)

            # During LU decomposition, detect singularities and warn user.
            with warnings.catch_warnings():

                if self.options['err_on_singular']:
                    warnings.simplefilter('error', RuntimeWarning)

                try:
                    self._lup = scipy.linalg.lu_factor(mtx)

                except RuntimeWarning as err:
                    raise RuntimeError(format_singluar_error(err, system, mtx))

    def _mat_vec(self, in_vec, out_vec):
        """
        Compute matrix-vector product.

        Parameters
        ----------
        in_vec : ndarray
            the incoming array (combines all varsets).
        out_vec : ndarray
            where the outgoing array after the product (combines all varsets) will be stored
        """
        vec_name = 'linear'
        system = self._system

        # assign x and b vectors based on mode
        x_vec = system._vectors['output'][vec_name]
        b_vec = system._vectors['residual'][vec_name]

        # set value of x vector to provided value
        x_vec.set_data(in_vec)

        # apply linear
        scope_out, scope_in = system._get_scope()
        system._apply_linear([vec_name], self._rel_systems, 'fwd', scope_out, scope_in)

        # put new value in out_vec
        b_vec.get_data(out_vec)

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
                if system._owns_assembled_jac or system._views_assembled_jac:
                    with system._unscaled_context(outputs=[d_outputs], residuals=[d_residuals]):
                        b_data = b_vec.get_data()
                        if (isinstance(system._jacobian._int_mtx,
                                       (COOMatrix, CSRMatrix, CSCMatrix))):
                            x_data = self._lu.solve(b_data, trans_splu)
                        else:
                            x_data = scipy.linalg.lu_solve(self._lup, b_data, trans=trans_lu)
                        x_vec.set_data(x_data)

                # MVP-generated jacobians are scaled.
                else:
                    b_data = b_vec.get_data()
                    x_data = scipy.linalg.lu_solve(self._lup, b_data, trans=trans_lu)
                    x_vec.set_data(x_data)

                rec.abs = 0.0
                rec.rel = 0.0

        return False, 0., 0.
