"""Define the CooMatrix class."""
from __future__ import division

import numpy
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix
from six.moves import range
from six import iteritems

from openmdao.matrices.matrix import Matrix, _compute_index_map


class CooMatrix(Matrix):
    """Sparse matrix in Coordinate list format."""

    def _build_sparse(self, num_rows, num_cols):
        """Allocate the data, rows, and cols for the sparse matrix.

        Args
        ----
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

        for key, (info, irow, icol, src_indices, shape) in iteritems(submats):
            val = info['value']
            rows = info['rows']
            dense = (rows is None and (val is None or
                     isinstance(val, ndarray)))
            ind1 = counter
            if dense:
                counter += numpy.prod(shape)
            elif rows is None:
                counter += val.data.size
            else:
                counter += len(rows)
            ind2 = counter
            metadata[key] = (ind1, ind2, None)

        data = numpy.zeros(counter)
        rows = -numpy.ones(counter, int)
        cols = -numpy.ones(counter, int)

        for key, (info, irow, icol, src_indices, shape) in iteritems(submats):
            val = info['value']
            dense = (info['rows'] is None and (val is None or
                     isinstance(val, ndarray)))
            ind1, ind2, idxs = metadata[key]

            if dense:
                jac_type = ndarray
                rowrange = numpy.arange(shape[0], dtype=int)

                if src_indices is None:
                    colrange = numpy.arange(shape[1], dtype=int)
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
                if isinstance(val, (coo_matrix, csr_matrix)):
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

            metadata[key] = (ind1, ind2, idxs, jac_type)

        return data, rows, cols

    def _build(self, num_rows, num_cols):
        """Allocate the matrix.

        Args
        ----
        num_rows : int
            number of rows in the matrix.
        num_cols : int
            number of cols in the matrix.
        """
        data, rows, cols = self._build_sparse(num_rows, num_cols)

        metadata = self._metadata
        for key, (ind1, ind2, idxs, jac_type) in iteritems(metadata):
            if idxs is None:
                metadata[key] = (slice(ind1, ind2), jac_type)
            else:
                # store reverse indices to avoid copying subjac data during
                # update_submat.
                metadata[key] = (numpy.argsort(idxs) + ind1, jac_type)

        self._matrix = coo_matrix((data, (rows, cols)),
                                  shape=(num_rows, num_cols))

    def _update_submat(self, key, jac):
        """Update the values of a sub-jacobian.

        Args
        ----
        key : (int, int)
            the global output and input variable indices.
        jac : ndarray or scipy.sparse or tuple
            the sub-jacobian, the same format with which it was declared.
        """
        idxs, jac_type = self._metadata[key]
        if not isinstance(jac, jac_type):
            raise TypeError("Jacobian entry for %s is of different type (%s) than "
                            "the type (%s) used at init time." % (key,
                                                                  type(jac).__name__,
                                                                  jac_type.__name__))
        if isinstance(jac, ndarray):
            self._matrix.data[idxs] = jac.flat
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            self._matrix.data[idxs] = jac.data
        elif isinstance(jac, list):
            self._matrix.data[idxs] = jac[0]

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
