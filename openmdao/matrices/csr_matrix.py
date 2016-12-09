"""Define the CsrMatrix class."""
from __future__ import division

import numpy
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix

from openmdao.matrices.coo_matrix import CooMatrix


class CsrMatrix(CooMatrix):
    """Sparse matrix in Compressed Row Storage format."""

    def _build(self, num_rows, num_cols):
        """Allocate the matrix.

        Args
        ----
        num_rows : int
            number of rows in the matrix.
        num_cols : int
            number of cols in the matrix.
        """
        super(CsrMatrix, self)._build(num_rows, num_cols)

        rows = self._matrix.row
        cols = self._matrix.col
        data = self._matrix.data

        # get a set of indices that sorts into row major order
        idxs = numpy.lexsort((cols, rows))

        # now sort these back into ascending order (our original stacked order)
        # so in _update_submat() we can just extract the individual index
        # arrays that will map each block into the combined data array.
        self._idxs = numpy.argsort(idxs)

        data = data[idxs]
        rows = rows[idxs]
        cols = cols[idxs]

        # data array for the CSR should be the same as for the COO since
        # it was already in sorted order.
        self._matrix = coo_matrix((data, (rows, cols)),
                                  shape=(num_rows, num_cols)).tocsr()

    def _update_submat(self, submats, metadata, key, jac):
        """Update the values of a sub-jacobian.

        Args
        ----
        submats : dict
            dictionary of sub-jacobian data keyed by (op_ind, ip_ind).
        metadata : dict
            implementation-specific data for the sub-jacobians.
        key : (int, int)
            the global output and input variable indices.
        jac : ndarray or scipy.sparse or tuple
            the sub-jacobian, the same format with which it was declared.
        """
        ind1, ind2, idxs = metadata[key]
        if isinstance(jac, ndarray):
            self._matrix.data[self._idxs[ind1:ind2]] = jac.flat
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            self._matrix.data[self._idxs[ind1:ind2]] = jac.data[idxs]
        elif isinstance(jac, list):
            self._matrix.data[self._idxs[ind1:ind2]] = jac[0][idxs]
