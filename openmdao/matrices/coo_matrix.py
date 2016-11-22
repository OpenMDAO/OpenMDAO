"""Define the CooMatrix class."""
from __future__ import division, print_function
import numpy
import scipy.sparse

from openmdao.matrices.matrix import Matrix


class CooMatrix(Matrix):
    """Sparse matrix in Coordinate list format."""

    def _build(self, num_rows, num_cols):
        """See Matrix."""
        counter = 0

        for ind in range(2):
            submats = [self._op_submats, self._ip_submats][ind]
            metadata = [self._op_metadata, self._ip_metadata][ind]

            for key in submats:
                jac, irow, icol = submats[key]

                if type(jac) is numpy.ndarray or scipy.sparse.issparse(jac):
                    ind1 = counter
                    counter += numpy.prod(jac.shape)
                    ind2 = counter
                    metadata[key] = (ind1, ind2)
                elif type(jac) is list:
                    ind1 = counter
                    counter += len(jac[1])
                    ind2 = counter
                    metadata[key] = (ind1, ind2)

        data = numpy.zeros(counter)
        rows = numpy.zeros(counter, int)
        cols = numpy.zeros(counter, int)

        for ind in range(2):
            submats = [self._op_submats, self._ip_submats][ind]
            metadata = [self._op_metadata, self._ip_metadata][ind]

            for key in submats:
                jac, irow, icol = submats[key]
                ind1, ind2 = metadata[key]

                if type(jac) is numpy.ndarray or scipy.sparse.issparse(jac):
                    irows = numpy.zeros(jac.shape, int)
                    icols = numpy.zeros(jac.shape, int)
                    for indr in range(jac.shape[0]):
                        for indc in range(jac.shape[1]):
                            irows[indr, indc] = indr
                            icols[indr, indc] = indc
                    size = numpy.prod(jac.shape)
                    if type(jac) is numpy.ndarray:
                        data[ind1:ind2] = jac.flatten()
                    else:
                        data[ind1:ind2] = jac.todense().flatten()
                    rows[ind1:ind2] = irow + irows.reshape(size)
                    cols[ind1:ind2] = icol + icols.reshape(size)
                elif type(jac) is list:
                    data[ind1:ind2] = jac[0]
                    rows[ind1:ind2] = irow + jac[1]
                    cols[ind1:ind2] = icol + jac[2]

        self._matrix = [data, rows, cols, num_rows, num_cols]

    def _update_submat(self, submats, metadata, key, jac):
        """See Matrix."""
        # TODO: improve on todense in the scipy.sparse case
        if type(jac) is numpy.ndarray:
            ind1, ind2 = metadata[key]
            self._matrix[0][ind1:ind2] = jac.flatten()
        elif scipy.sparse.issparse(jac):
            ind1, ind2 = metadata[key]
            self._matrix[0][ind1:ind2] = jac.todense().flatten()
        elif type(jac) is list:
            ind1, ind2 = metadata[key]
            self._matrix[0][ind1:ind2] = jac[0]

    def _prod(self, in_vec, mode):
        """See Matrix."""
        data, rows, cols, num_rows, num_cols = self._matrix

        if mode == 'fwd':
            out_vec = numpy.zeros(num_rows)
            numpy.add.at(out_vec, rows, data * in_vec[cols])
        elif mode == 'rev':
            out_vec = numpy.zeros(num_cols)
            numpy.add.at(out_vec, cols, data * in_vec[rows])
        return out_vec
