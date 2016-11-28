"""Define the CsrMatrix class."""
from __future__ import division, print_function
import numpy
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix, issparse

from openmdao.matrices.matrix import Matrix


class CsrMatrix(Matrix):
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
        counter = 0

        submat_meta_iter = ((self._op_submats, self._op_metadata),
                            (self._ip_submats, self._ip_metadata))

        for submats, metadata in submat_meta_iter:
            for key in submats:
                jac = submats[key][0]

                ind1 = counter
                if isinstance(jac, ndarray):
                    counter += jac.size
                elif isinstance(jac, (coo_matrix, csr_matrix)):
                    counter += jac.data.size
                elif isinstance(jac, list) and len(jac) == 3:
                    counter += len(jac[0])
                ind2 = counter
                metadata[key] = (ind1, ind2)

        data = numpy.zeros(counter)
        rows = numpy.zeros(counter, int)
        cols = numpy.zeros(counter, int)

        for submats, metadata in submat_meta_iter:
            for key in submats:
                jac, irow, icol = submats[key]
                ind1, ind2 = metadata[key]

                if isinstance(jac, ndarray):
                    irows = numpy.empty(jac.shape, int)
                    icols = numpy.empty(jac.shape, int)
                    for indr in range(jac.shape[0]):
                        for indc in range(jac.shape[1]):
                            irows[indr, indc] = indr
                            icols[indr, indc] = indc
                    size = jac.size
                    data[ind1:ind2] = jac.flat
                    rows[ind1:ind2] = irow + irows.reshape(size)
                    cols[ind1:ind2] = icol + icols.reshape(size)
                elif isinstance(jac, (coo_matrix, csr_matrix)):
                    jac = jac.tocoo()
                    data[ind1:ind2] = jac.data
                    rows[ind1:ind2] = irow + jac.row
                    cols[ind1:ind2] = icol + jac.col
                elif isinstance(jac, list) and len(jac) == 3:
                    data[ind1:ind2] = jac[0]
                    rows[ind1:ind2] = irow + jac[1]
                    cols[ind1:ind2] = icol + jac[2]

        # get a set of indices that sorts into row major order
        idxs = numpy.lexsort((cols, rows))

        data = data[idxs]
        rows = rows[idxs]
        cols = cols[idxs]

        # data array for the CSR should be the same as for the COO since
        # it was already in sorted order.
        self._matrix = coo_matrix((data, (rows, cols)),
                                  shape=(num_rows, num_cols)).tocsr()

        # now sort these back into ascending order (our original stacked order)
        # so we can then just extract the individual index arrays that will
        # map each block into the combined data array.
        self._idxs = numpy.argsort(idxs)

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
        ind1, ind2 = metadata[key]
        if isinstance(jac, ndarray):
            self._matrix.data[self._idxs[ind1:ind2]] = jac.flat
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            self._matrix.data[self._idxs[ind1:ind2]] = jac.data
        elif isinstance(jac, list) and len(jac) == 3:
            self._matrix.data[self._idxs[ind1:ind2]] = jac[0]

    def _prod(self, in_vec, mode):
        """Perform a matrix vector product.

        Args
        ----
        in_vec : ndarray[:]
            incoming vector to multiply.
        mode : str
            'fwd' or 'rev'.

        Returns
        -------
        ndarray[:]
            vector resulting from the product.
        """
        if mode == 'fwd':
            return self._matrix.dot(in_vec)
        elif mode == 'rev':
            return self._matrix.T.dot(in_vec)
