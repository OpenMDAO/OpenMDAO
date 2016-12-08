"""Define the CsrMatrix class."""
from __future__ import division, print_function
import numpy
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix, issparse
from six.moves import range

from openmdao.matrices.matrix import Matrix, _compute_index_map


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
                elif isinstance(jac, list):
                    counter += len(jac[0])
                ind2 = counter
                metadata[key] = (ind1, ind2)

        data = numpy.zeros(counter)
        rows = numpy.zeros(counter, int)
        cols = numpy.zeros(counter, int)

        for submats, metadata in submat_meta_iter:
            for key in submats:
                jac, irow, icol, src_indices = submats[key]
                ind1, ind2 = metadata[key]

                if isinstance(jac, ndarray):
                    rowrange = numpy.arange(jac.shape[0], dtype=int)

                    if src_indices is None:
                        colrange = numpy.arange(jac.shape[1], dtype=int)
                    else:
                        colrange = numpy.array(src_indices, dtype=int) + icol

                    ncols = colrange.size

                    subrows = numpy.empty(rowrange.size*colrange.size, dtype=int)
                    subcols = numpy.empty(subrows.size, dtype=int)

                    for i, row in enumerate(rowrange):
                        subrows[i*ncols: (i+1)*ncols] = row
                        subcols[i*ncols: (i+1)*ncols] = colrange

                    rows[ind1:ind2] = subrows + irow
                    cols[ind1:ind2] = subcols + icol
                    data[ind1:ind2] = jac.flat
                elif isinstance(jac, (coo_matrix, csr_matrix)):
                    jac = jac.tocoo()
                    if src_indices is None:
                        data[ind1:ind2] = jac.data
                        rows[ind1:ind2] = jac.row + irow
                        cols[ind1:ind2] = jac.col + icol
                    else:
                        irows, icols, idxs = _compute_index_map(jac.row,
                                                                jac.col,
                                                                irow, icol,
                                                                src_indices)

                        # get the indices to get us back to our original
                        # data order
                        idxs = numpy.argsort(idxs)

                        data[ind1:ind2] = jac.data[idxs]
                        rows[ind1:ind2] = irows
                        cols[ind1:ind2] = icols

                elif isinstance(jac, list):
                    if src_indices is None:
                        data[ind1:ind2] = jac[0]
                        rows[ind1:ind2] = irow + jac[1]
                        cols[ind1:ind2] = icol + jac[2]
                    else:
                        irows, icols, idxs = _compute_index_map(jac[1],
                                                                jac[2],
                                                                irow, icol,
                                                                src_indices)

                        # get the indices to get us back to our original
                        # data order
                        idxs = numpy.argsort(idxs)

                        data[ind1:ind2] = jac[0][idxs]
                        rows[ind1:ind2] = irows
                        cols[ind1:ind2] = icols


        # get a set of indices that sorts into row major order
        idxs = numpy.lexsort((cols, rows))

        # now sort these back into ascending order (our original stacked order)
        # so in _update_submat() we can just extract the individual index arrays that will
        # map each block into the combined data array.
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
        ind1, ind2 = metadata[key]
        if isinstance(jac, ndarray):
            self._matrix.data[self._idxs[ind1:ind2]] = jac.flat
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            self._matrix.data[self._idxs[ind1:ind2]] = jac.data
        elif isinstance(jac, list):
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
