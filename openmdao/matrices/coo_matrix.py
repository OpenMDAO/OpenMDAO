"""Define the COOmatrix class."""
from __future__ import division

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix

from six import iteritems

from openmdao.matrices.matrix import Matrix, _compute_index_map, sparse_types


class COOMatrix(Matrix):
    """
    Sparse matrix in Coordinate list format.
    """

    def _build_sparse(self, num_rows, num_cols):
        """
        Allocate the data, rows, and cols for the sparse matrix.

        Parameters
        ----------
        num_rows : int
            number of rows in the matrix.
        num_cols : int
            number of cols in the matrix.

        Returns
        -------
        (ndarray, ndarray, ndarray)
            data, rows, cols that can be used to construct a sparse matrix.
        """
        counter = 0

        submats = self._submats
        metadata = self._metadata
        pre_metadata = {}
        for key, (info, irow, icol, src_indices, shape, factor) in iteritems(submats):
            if not info['dependent']:
                continue
            val = info['value']
            rows = info['rows']
            dense = (rows is None and (val is None or
                     isinstance(val, ndarray)))
            ind1 = counter
            if dense:
                counter += np.prod(shape)
            elif rows is None:
                counter += val.data.size
            else:
                counter += len(rows)
            ind2 = counter
            pre_metadata[key] = (ind1, ind2, None)

        data = np.zeros(counter)
        rows = -np.ones(counter, int)
        cols = -np.ones(counter, int)

        for key, (ind1, ind2, idxs) in iteritems(pre_metadata):
            info, irow, icol, src_indices, shape, factor = submats[key]
            val = info['value']
            dense = (info['rows'] is None and (val is None or
                     isinstance(val, ndarray)))

            if dense:
                jac_type = ndarray
                rowrange = np.arange(shape[0], dtype=int)

                if src_indices is None:
                    colrange = np.arange(shape[1], dtype=int)
                else:
                    colrange = src_indices

                ncols = colrange.size
                subrows = rows[ind1:ind2]
                subcols = cols[ind1:ind2]

                for i, row in enumerate(rowrange):
                    subrows[i * ncols: (i + 1) * ncols] = row
                    subcols[i * ncols: (i + 1) * ncols] = colrange

                rows[ind1:ind2] += irow
                cols[ind1:ind2] += icol

            else:  # sparse
                if isinstance(val, sparse_types):
                    jac_type = type(val)
                    jac = val.tocoo()
                    jrows = jac.row
                    jcols = jac.col
                else:
                    jac_type = list
                    jrows = info['rows']
                    jcols = info['cols']

                if src_indices is None:
                    rows[ind1:ind2] = jrows + irow
                    cols[ind1:ind2] = jcols + icol
                else:
                    irows, icols, idxs = _compute_index_map(jrows, jcols,
                                                            irow, icol,
                                                            src_indices)
                    rows[ind1:ind2] = irows
                    cols[ind1:ind2] = icols

            metadata[key] = (ind1, ind2, idxs, jac_type, factor)

        return data, rows, cols

    def _build(self, num_rows, num_cols):
        """
        Allocate the matrix.

        Parameters
        ----------
        num_rows : int
            number of rows in the matrix.
        num_cols : int
            number of cols in the matrix.
        """
        data, rows, cols = self._build_sparse(num_rows, num_cols)

        metadata = self._metadata
        for key, (ind1, ind2, idxs, jac_type, factor) in iteritems(metadata):
            if idxs is None:
                metadata[key] = (slice(ind1, ind2), jac_type, factor)
            else:
                # store reverse indices to avoid copying subjac data during
                # update_submat.
                metadata[key] = (np.argsort(idxs) + ind1, jac_type, factor)

        self._matrix = coo_matrix((data, (rows, cols)),
                                  shape=(num_rows, num_cols))

    def _update_submat(self, key, jac):
        """
        Update the values of a sub-jacobian.

        Parameters
        ----------
        key : (str, str)
            the global output and input variable names.
        jac : ndarray or scipy.sparse or tuple
            the sub-jacobian, the same format with which it was declared.
        """
        idxs, jac_type, factor = self._metadata[key]
        if not isinstance(jac, jac_type):
            raise TypeError("Jacobian entry for %s is of different type (%s) than "
                            "the type (%s) used at init time." % (key,
                                                                  type(jac).__name__,
                                                                  jac_type.__name__))
        if isinstance(jac, ndarray):
            self._matrix.data[idxs] = jac.flat
        elif isinstance(jac, sparse_types):
            self._matrix.data[idxs] = jac.data
        else:  # list format  [data, rows, cols]
            self._matrix.data[idxs] = jac[0]

        if factor is not None:
            self._matrix.data[idxs] *= factor

    def _prod(self, in_vec, mode, ranges):
        """
        Perform a matrix vector product.

        Parameters
        ----------
        in_vec : ndarray[:]
            incoming vector to multiply.
        mode : str
            'fwd' or 'rev'.
        ranges : (int, int, int, int)
            Min row, max row, min col, max col for the current system.

        Returns
        -------
        ndarray[:]
            vector resulting from the product.
        """
        if mode == 'fwd':
            return self._matrix.dot(in_vec)
        else:  # rev
            return self._matrix.T.dot(in_vec)
