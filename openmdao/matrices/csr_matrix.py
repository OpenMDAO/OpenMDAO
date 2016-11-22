"""Define the CsrMatrix class."""
from __future__ import division, print_function
import numpy
from numpy import ndarray
import scipy.sparse

from openmdao.matrices.matrix import Matrix


class CsrMatrix(Matrix):
    """Sparse matrix in Compressed Row Storage format."""

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
                    counter += numpy.prod(jac.shape)
                elif scipy.sparse.issparse(jac):
                    counter += jac.data.size
                elif isinstance(jac, list):
                    counter += len(jac[1])
                else:
                    raise RuntimeError("unknown subjac type of %s for key %s" %
                                       (type(jac), key))
                ind2 = counter
                metadata[key] = (ind1, ind2)

        self._num_rows = num_rows
        self._num_cols = num_cols
        self._data = data = numpy.zeros(counter)
        self._rows = rows = numpy.zeros(counter, int)
        self._cols = cols = numpy.zeros(counter, int)

        for submats, metadata in submat_meta_iter:
            for key in submats:
                jac, irow, icol = submats[key]
                ind1, ind2 = metadata[key]

                if isinstance(jac, ndarray) or scipy.sparse.issparse(jac):
                    irows = numpy.zeros(jac.shape, int)
                    icols = numpy.zeros(jac.shape, int)
                    for indr in range(jac.shape[0]):
                        for indc in range(jac.shape[1]):
                            irows[indr, indc] = indr
                            icols[indr, indc] = indc
                    size = numpy.prod(jac.shape)
                    if isinstance(jac, numpy.ndarray):
                        data[ind1:ind2] = jac.flatten()
                    else:
                        data[ind1:ind2] = jac.todense().flatten()
                    rows[ind1:ind2] = irow + irows.reshape(size)
                    cols[ind1:ind2] = icol + icols.reshape(size)
                elif isinstance(jac, list):
                    data[ind1:ind2] = jac[0]
                    rows[ind1:ind2] = irow + jac[1]
                    cols[ind1:ind2] = icol + jac[2]

    def _update_submat(self, submats, metadata, key, jac):
        """See Matrix."""
        if isinstance(jac, ndarray):
            ind1, ind2 = metadata[key]
            self._data[ind1:ind2] = jac.flat
        elif scipy.sparse.issparse(jac):
            ind1, ind2 = metadata[key]
            self._data[ind1:ind2] = jac.data
        elif isinstance(jac, list):
            ind1, ind2 = metadata[key]
            self._data[ind1:ind2] = jac[0]

    def _prod(self, in_vec, mode):
        """See Matrix."""
        if mode == 'fwd':
            out_vec = numpy.zeros(self._num_rows)
            numpy.add.at(out_vec, self._rows, self._data * in_vec[self._cols])
        elif mode == 'rev':
            out_vec = numpy.zeros(self._num_cols)
            numpy.add.at(out_vec, self._cols, self._data * in_vec[self._rows])
        return out_vec
