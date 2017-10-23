"""LinearSolver that uses linalg.solve or LU factor/solve."""

from __future__ import division, print_function

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


class DirectSolver(LinearSolver):
    """
    LinearSolver that uses linalg.solve or LU factor/solve.
    """

    SOLVER = 'LN: Direct'

    def _linearize(self):
        """
        Perform factorization.
        """
        system = self._system

        if system._owns_assembled_jac or system._views_assembled_jac:
            mtx = system._jacobian._int_mtx
            # Perform dense or sparse lu factorization
            if isinstance(mtx, DenseMatrix):
                ranges = system._jacobian._view_ranges[system.pathname]
                matrix = mtx._matrix[ranges[0]:ranges[1], ranges[0]:ranges[1]]
                np.set_printoptions(precision=3)
                self._lup = scipy.linalg.lu_factor(matrix)
            elif isinstance(mtx, (CSRMatrix, CSCMatrix)):
                np.set_printoptions(precision=3)
                self._lu = scipy.sparse.linalg.splu(mtx._matrix)
            elif isinstance(mtx, COOMatrix):
                # calling scipy.sparse.linalg.splu on a COO actually transposes
                # the matrix during conversion to csc prior to LU decomp
                raise RuntimeError("Direct solver is not compatible with mtx type "
                                   "COOMatrix in system '%s'." % system.pathname)
            else:
                raise RuntimeError("Direct solver not implemented for mtx type %s"
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

            self._lup = scipy.linalg.lu_factor(mtx)

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
