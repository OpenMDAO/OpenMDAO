"""Define the DenseMatrix class."""
from __future__ import division, print_function
import numpy as np

from openmdao.matrices.matrix import Matrix, _compute_index_map, sparse_types


class DenseMatrix(Matrix):
    """
    Dense global matrix.
    """

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
        self._matrix = matrix = np.zeros((num_rows, num_cols))
        submats = self._submats
        metadata = self._metadata

        for key in submats:
            info, irow, icol, src_indices, shape, factor = submats[key]
            rows = info['rows']
            cols = info['cols']
            val = info['value']

            if rows is None and (val is None or isinstance(val,
                                                           np.ndarray)):
                nrows, ncols = shape
                irow2 = irow + nrows
                if src_indices is None:
                    icol2 = icol + ncols
                    metadata[key] = (slice(irow, irow2),
                                     slice(icol, icol2), np.ndarray, factor)
                else:
                    metadata[key] = (slice(irow, irow2),
                                     src_indices + icol, np.ndarray, factor)
            elif isinstance(val, sparse_types):
                jac = val.tocoo()
                if src_indices is None:
                    irows = irow + jac.row
                    icols = icol + jac.col
                else:
                    irows, icols, idxs = _compute_index_map(jac.row,
                                                            jac.col,
                                                            irow, icol,
                                                            src_indices)
                    revidxs = np.argsort(idxs)
                    irows, icols = irows[revidxs], icols[revidxs]

                metadata[key] = (irows, icols, type(val), factor)
            elif rows is not None:
                if src_indices is None:
                    irows = rows + irow
                    icols = cols + icol
                else:
                    irows, icols, idxs = _compute_index_map(rows, cols,
                                                            irow, icol,
                                                            src_indices)
                    revidxs = np.argsort(idxs)
                    irows, icols = irows[revidxs], icols[revidxs]

                metadata[key] = (irows, icols, list, factor)

    def _update_submat(self, key, jac):
        """
        Update the values of a sub-jacobian.

        Parameters
        ----------
        key : (int, int)
            the global output and input variable indices.
        jac : ndarray or scipy.sparse or tuple
            the sub-jacobian, the same format with which it was declared.
        """
        irows, icols, jac_type, factor = self._metadata[key]
        if not isinstance(jac, jac_type):
            raise TypeError("Jacobian entry for %s is of different type (%s) than "
                            "the type (%s) used at init time." % (key,
                                                                  type(jac).__name__,
                                                                  jac_type.__name__))
        if isinstance(jac, np.ndarray):
            self._matrix[irows, icols] = jac
        elif isinstance(jac, sparse_types):
            self._matrix[irows, icols] = jac.data
        elif isinstance(jac, list):
            self._matrix[irows, icols] = jac[0]

        if factor is not None:
            self._matrix[irows, icols] *= factor

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
        # when we have a derivative based solver at a level below the
        # group that owns the AssembledJacobian, we need to use only
        # the part of the matrix that is relevant to the lower level
        # system.
        if ranges is None:
            mat = self._matrix
        else:
            rstart, rend, cstart, cend = ranges
            mat = self._matrix[rstart:rend, cstart:cend]
        if mode == 'fwd':
            return mat.dot(in_vec)
        else:  # rev
            return mat.T.dot(in_vec)
