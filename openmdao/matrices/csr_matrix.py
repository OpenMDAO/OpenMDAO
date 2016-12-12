"""Define the CsrMatrix class."""
from __future__ import division

import numpy
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix

from openmdao.matrices.coo_matrix import Matrix, _compute_index_map


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
        full_slice = slice(None)

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

        data = numpy.empty(counter)
        rows = numpy.empty(counter, int)
        cols = numpy.empty(counter, int)

        for submats, metadata in submat_meta_iter:
            for key in submats:
                jac, irow, icol, src_indices = submats[key]
                ind1, ind2 = metadata[key]

                if isinstance(jac, ndarray):
                    rowrange = numpy.arange(jac.shape[0], dtype=int)

                    if src_indices is None:
                        colrange = numpy.arange(jac.shape[1], dtype=int)
                    else:
                        colrange = numpy.array(src_indices, dtype=int)

                    ncols = colrange.size

                    subrows = rows[ind1:ind2]
                    subcols = cols[ind1:ind2]

                    for i, row in enumerate(rowrange):
                        subrows[i * ncols: (i + 1) * ncols] = row
                        subcols[i * ncols: (i + 1) * ncols] = colrange

                    rows[ind1:ind2] += irow
                    cols[ind1:ind2] += icol
                    data[ind1:ind2] = jac.flat

                    idxs = full_slice

                elif isinstance(jac, (coo_matrix, csr_matrix)):
                    coojac = jac.tocoo()
                    if src_indices is None:
                        data[ind1:ind2] = coojac.data
                        rows[ind1:ind2] = coojac.row + irow
                        cols[ind1:ind2] = coojac.col + icol
                        idxs = full_slice
                    else:
                        irows, icols, idxs = _compute_index_map(coojac.row,
                                                                coojac.col,
                                                                irow, icol,
                                                                src_indices)
                        data[ind1:ind2] = jac.data[idxs]
                        rows[ind1:ind2] = irows
                        cols[ind1:ind2] = icols

                        # get the reverse indices so we can apply them
                        # to the LHS instead of the RHS, thus avoiding
                        # copying the RHS array.
                        idxs = numpy.argsort(idxs)

                elif isinstance(jac, list):
                    if src_indices is None:
                        data[ind1:ind2] = jac[0]
                        rows[ind1:ind2] = irow + jac[1]
                        cols[ind1:ind2] = icol + jac[2]
                        idxs = full_slice
                    else:
                        irows, icols, idxs = _compute_index_map(jac[1],
                                                                jac[2],
                                                                irow, icol,
                                                                src_indices)
                        data[ind1:ind2] = jac[0][idxs]
                        rows[ind1:ind2] = irows
                        cols[ind1:ind2] = icols

                        # get the reverse indices so we can apply them
                        # to the LHS instead of the RHS, thus avoiding
                        # copying the RHS array.
                        idxs = numpy.argsort(idxs)

                if metadata is self._ip_metadata:
                    metadata[key] = (ind1, ind2, idxs)

        # get a set of indices that sorts into row major order
        idxs = numpy.lexsort((cols, rows))

        data = data[idxs]
        rows = rows[idxs]
        cols = cols[idxs]

        # now sort these back into ascending order (our original stacked order)
        # so in _update_submat() we can just extract the individual index
        # arrays that will map each block into the combined data array.
        self._idxs = numpy.argsort(idxs)

        # now apply the reverse index to each part of self._idxs so that
        # we can avoid copying the index array during updates.
        for key in self._ip_metadata:
            ind1, ind2, idxs = self._ip_metadata[key]
            self._idxs[ind1:ind2] = self._idxs[ind1:ind2][idxs]
            self._ip_metadata[key] = (ind1, ind2)  # don't need idxs any more

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
