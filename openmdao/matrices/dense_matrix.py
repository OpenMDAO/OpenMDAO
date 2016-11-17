"""Define the base Matrix class."""
from __future__ import division, print_function
import numpy
import scipy.sparse

from matrix import Matrix


class DenseMatrix(Matrix):
    """Dense global matrix."""

    def _build(self, num_rows, num_cols):
        """See Matrix."""
        matrix = numpy.zeros((num_rows, num_cols))
        for ind in range(2):
            submats = [self._op_submats, self._ip_submats][ind]
            metadata = [self._op_metadata, self._ip_metadata][ind]

            for key in submats:
                jac, irow, icol = submats[key]

                if type(jac) is numpy.ndarray or scipy.sparse.issparse(jac):
                    shape = jac.shape
                    irow1 = irow
                    irow2 = irow + shape[0]
                    icol1 = icol
                    icol2 = icol + shape[1]
                    metadata[key] = matrix[irow1:irow2, icol1:icol2]
                    metadata[key][:, :] = jac
                elif type(jac) is list:
                    irows = irow + jac[1]
                    icols = icol + jac[2]
                    matrix[irows, icols] = jac[0]
                    metadata[key] = (irows, icols)

        self._matrix = matrix

    def _update_submat(self, submats, metadata, key, jac):
        """See Matrix."""
        if type(jac) is numpy.ndarray:
            metadata[key][:, :] = jac
        elif scipy.sparse.issparse(jac):
            metadata[key][:, :] = jac.todense()
        elif type(jac) is list:
            irows, icols = metadata[key]
            self._matrix[irows, icols] = jac[0]

    def _prod(self, vec, mode):
        """See Matrix."""
        if mode == 'fwd':
            return self._matrix.dot(vec)
        elif mode == 'rev':
            return self._matrix.T.dot(vec)
