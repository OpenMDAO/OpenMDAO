"""Define the DenseMatrix class."""
from __future__ import division, print_function
import numpy
from scipy.sparse import coo_matrix, csr_matrix

from openmdao.matrices.matrix import Matrix, _compute_index_map


class DenseMatrix(Matrix):
    """Dense global matrix."""

    def _build(self, num_rows, num_cols):
        """See Matrix."""
        matrix = numpy.zeros((num_rows, num_cols))
        submat_meta_iter = ((self._op_submats, self._op_metadata),
                            (self._ip_submats, self._ip_metadata))

        for submats, metadata in submat_meta_iter:
            for key in submats:
                jac, irow, icol, src_indices = submats[key]

                if isinstance(jac, numpy.ndarray):
                    nrows, ncols = jac.shape
                    irow2 = irow + nrows
                    if src_indices is None:
                        icol2 = icol + ncols
                        metadata[key] = (slice(irow, irow2),
                                         slice(icol, icol2),
                                         (slice(None), slice(None)))
                    else:
                        metadata[key] = (slice(irow, irow2),
                                         src_indices + icol,
                                         (slice(None),
                                          slice(src_indices.size)))
                    irows, icols, jidxs = metadata[key]
                    matrix[irows, icols] = jac[jidxs]
                elif isinstance(jac, (coo_matrix, csr_matrix)):
                    jac = jac.tocoo()
                    if src_indices is None:
                        irows = irow + jac.row
                        icols = icol + jac.col
                        metadata[key] = (irows, icols, slice(None))
                        matrix[irows, icols] = jac.data
                    else:
                        irows, icols, idxs = _compute_index_map(jac.row,
                                                                jac.col,
                                                                irow, icol,
                                                                src_indices)

                        # get the indices to get us back to our original
                        # data order
                        idxs = numpy.argsort(idxs)

                        metadata[key] = (irows, icols, idxs)
                        matrix[irows, icols] = jac.data[idxs]

                elif isinstance(jac, list):
                    if src_indices is None:
                        irows = jac[1] + irow
                        icols = jac[2] + icol
                        idxs = None
                    else:
                        irows, icols, idxs = _compute_index_map(jac[1], jac[2],
                                                                irow, icol,
                                                                src_indices)
                        # get the indices to get us back to our original
                        # data order
                        idxs = numpy.argsort(idxs)

                    metadata[key] = (irows, icols, idxs)
                    matrix[irows, icols] = jac[0]

        self._matrix = matrix

    def _update_submat(self, submats, metadata, key, jac):
        """See Matrix."""
        irows, icols, jidxs = metadata[key]
        if isinstance(jac, numpy.ndarray):
            self._matrix[irows, icols] = jac[jidxs]
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            self._matrix[irows, icols] = jac.data[jidxs]
        elif isinstance(jac, list):
            # can't just use slice(None) as jidxs because for lists
            # that actually makes a copy
            if jidxs is None:
                self._matrix[irows, icols] = jac[0]
            else:
                self._matrix[irows, icols] = jac[0][jidxs]

    def _prod(self, in_vec, mode):
        """See Matrix."""
        if mode == 'fwd':
            return self._matrix.dot(in_vec)
        elif mode == 'rev':
            return self._matrix.T.dot(in_vec)
