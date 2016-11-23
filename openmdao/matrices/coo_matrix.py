"""Define the CooMatrix class."""
from __future__ import division, print_function
import numpy
from numpy import ndarray
import scipy.sparse
from scipy.sparse import coo_matrix, issparse

from openmdao.matrices.matrix import Matrix


class CooMatrix(Matrix):
    """Sparse matrix in Coordinate list format."""

    def _build(self, num_rows, num_cols):
        """See Matrix."""
        counter = 0

        submat_meta_iter = ((self._op_submats, self._op_metadata),
                            (self._ip_submats, self._ip_metadata))

        for submats, metadata in submat_meta_iter:
            for key in submats:
                jac, irow, icol = submats[key]

                ind1 = counter
                if isinstance(jac, ndarray):
                    counter += jac.size
                elif scipy.sparse.issparse(jac):
                    counter += jac.data.size
                elif isinstance(jac, list):
                    counter += len(jac[0])
                else:
                    raise RuntimeError("unknown subjac type of %s for key %s" %
                                       (type(jac), key))
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
                    data[ind1:ind2] = jac.flatten()
                    rows[ind1:ind2] = irow + irows.reshape(size)
                    cols[ind1:ind2] = icol + icols.reshape(size)
                elif scipy.sparse.issparse(jac):
                    jac = jac.tocoo()
                    data[ind1:ind2] = jac.data
                    rows[ind1:ind2] = irow + jac.row
                    cols[ind1:ind2] = icol + jac.col
                elif isinstance(jac, list):
                    data[ind1:ind2] = jac[0]
                    rows[ind1:ind2] = irow + jac[1]
                    cols[ind1:ind2] = icol + jac[2]

        self._matrix = coo_matrix((data, (rows, cols)),
                                  shape=(num_rows, num_cols))

    def _update_submat(self, submats, metadata, key, jac):
        """See Matrix."""
        ind1, ind2 = metadata[key]
        if isinstance(jac, ndarray):
            self._matrix.data[ind1:ind2] = jac.flat
        elif scipy.sparse.issparse(jac):
            self._matrix.data[ind1:ind2] = jac.data
        elif isinstance(jac, list):
            self._matrix.data[ind1:ind2] = jac[0]

    def _prod(self, in_vec, mode):
        """See Matrix."""
        if mode == 'fwd':
            return self._matrix.dot(in_vec)
        elif mode == 'rev':
            return self._matrix.T.dot(in_vec)
