"""Define the DenseMatrix class."""
from __future__ import division, print_function
import numpy
from scipy.sparse import coo_matrix, csr_matrix

from openmdao.matrices.matrix import Matrix, _compute_index_map


class DenseMatrix(Matrix):
    """Dense global matrix."""

    def _build(self, num_rows, num_cols):
        """Allocate the matrix.

        Args
        ----
        num_rows : int
            number of rows in the matrix.
        num_cols : int
            number of cols in the matrix.
        """
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
                                         slice(icol, icol2))
                    else:
                        metadata[key] = (slice(irow, irow2),
                                         src_indices + icol)

                    irows, icols = metadata[key]
                    matrix[irows, icols] = jac
                elif isinstance(jac, (coo_matrix, csr_matrix)):
                    jac = jac.tocoo()
                    if src_indices is None:
                        irows = irow + jac.row
                        icols = icol + jac.col
                    else:
                        irows, icols, idxs = _compute_index_map(jac.row,
                                                                jac.col,
                                                                irow, icol,
                                                                src_indices)
                        revidxs = numpy.argsort(idxs)
                        irows, icols = irows[revidxs], icols[revidxs]

                    metadata[key] = (irows, icols)
                    matrix[irows, icols] = jac.data

                elif isinstance(jac, list):
                    if src_indices is None:
                        irows = jac[1] + irow
                        icols = jac[2] + icol
                    else:
                        irows, icols, idxs = _compute_index_map(jac[1], jac[2],
                                                                irow, icol,
                                                                src_indices)
                        revidxs = numpy.argsort(idxs)
                        irows, icols = irows[revidxs], icols[revidxs]

                    metadata[key] = (irows, icols)
                    matrix[irows, icols] = jac[0]

        self._matrix = matrix

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
        irows, icols = metadata[key]
        if isinstance(jac, numpy.ndarray):
            self._matrix[irows, icols] = jac
        elif isinstance(jac, (coo_matrix, csr_matrix)):
            self._matrix[irows, icols] = jac.data
        elif isinstance(jac, list):
            self._matrix[irows, icols] = jac[0]

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
