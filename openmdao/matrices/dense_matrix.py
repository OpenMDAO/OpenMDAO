"""Define the DenseMatrix class."""
from __future__ import division, print_function
import numpy
from scipy.sparse import coo_matrix, csr_matrix

from openmdao.matrices.matrix import Matrix


class DenseMatrix(Matrix):
    """Dense global matrix."""

    def _build(self, num_rows, num_cols):
        """See Matrix."""
        matrix = numpy.zeros((num_rows, num_cols))
        for ind in range(2):
            submats = [self._op_submats, self._ip_submats][ind]
            metadata = [self._op_metadata, self._ip_metadata][ind]

            for key in submats:
                jac, irow, icol, src_indices = submats[key]

                if isinstance(jac, numpy.ndarray):
                    shape = jac.shape
                    irow1 = irow
                    irow2 = irow + shape[0]
                    icol1 = icol
                    icol2 = icol + shape[1]
                    metadata[key] = (slice(irow1, irow2), slice(icol1, icol2))
                    matrix[irow1:irow2, icol1:icol2] = jac
                elif isinstance(jac, (coo_matrix, csr_matrix)):
                    jac = jac.tocoo()
                    irows = irow + jac.row
                    icols = icol + jac.col
                    matrix[irows, icols] = jac.data
                    metadata[key] = (irows, icols)
                elif isinstance(jac, list):
                    irows = irow + jac[1]
                    icols = icol + jac[2]
                    matrix[irows, icols] = jac[0]
                    metadata[key] = (irows, icols)

        self._matrix = matrix

    def _update_submat(self, submats, metadata, key, jac):
        """See Matrix."""
        irows, icols = metadata[key]
        if isinstance(jac, numpy.ndarray):
            self._matrix[irows, icols] = jac
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            self._matrix[irows, icols] = jac.data
        elif isinstance(jac, list):
            self._matrix[irows, icols] = jac[0]

    def _prod(self, in_vec, mode):
        """See Matrix."""
        if mode == 'fwd':
            return self._matrix.dot(in_vec)
        elif mode == 'rev':
            return self._matrix.T.dot(in_vec)
