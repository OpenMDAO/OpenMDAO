"""Define the DenseMatrix class."""
from __future__ import division, print_function
import numpy as np
from six import iteritems

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
        self._matrix = np.zeros((num_rows, num_cols))
        submats = self._submats
        metadata = self._metadata

        for key in submats:
            info, loc, src_indices, shape, factor = submats[key]
            irow, icol = loc
            rows = info['rows']

            if rows is None:
                val = info['value']
                if val is None or isinstance(val, np.ndarray):
                    nrows, ncols = shape
                    irow2 = irow + nrows
                    if src_indices is None:
                        icol2 = icol + ncols
                        metadata[key] = (slice(irow, irow2),
                                         slice(icol, icol2), np.ndarray, factor)
                    else:
                        metadata[key] = (slice(irow, irow2),
                                         src_indices + icol, np.ndarray, factor)
                else:  # sparse
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
            else:  # list format [data, rows, cols]
                if src_indices is None:
                    irows = rows + irow
                    icols = info['cols'] + icol
                else:
                    irows, icols, idxs = _compute_index_map(rows, info['cols'],
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
        key : (str, str)
            the global output and input variable names.
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
        elif isinstance(jac, list):
            self._matrix[irows, icols] = jac[0]
        else:  # sparse
            self._matrix[irows, icols] = jac.data

        if factor is not None:
            self._matrix[irows, icols] *= factor

    def _update_add_submat(self, key, jac):
        """
        Add the subjac values to an existing  sub-jacobian.

        Parameters
        ----------
        key : (str, str)
            the global output and input variable names.
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
            val = jac
        elif isinstance(jac, list):
            val = jac[0]
        else:  # sparse
            val = jac.data

        if factor is not None:
            self._matrix[irows, icols] += val * factor
        else:
            self._matrix[irows, icols] += val

    def _prod(self, in_vec, mode, ranges, mask=None):
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
        mask : ndarray of type bool, or None
            Array used to mask out part of the input vector.

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
            if mask is None:
                return mat.dot(in_vec)
            else:
                inputs_masked = np.ma.array(in_vec, mask=mask)

                # Use the special dot product function from masking module so that we
                # ignore masked parts.
                return np.ma.dot(mat, inputs_masked)
        else:  # rev
            if mask is None:
                return mat.T.dot(in_vec)
            else:
                # Mask need to be applied to ext_mtx so that we can ignore multiplication
                # by certain columns.
                arrmask = np.zeros(mat.T.shape, dtype=np.bool)
                arrmask[mask, :] = True
                masked_mtx = np.ma.array(mat, mask=arrmask, fill_value=0.0)

                masked_product = np.ma.dot(masked_mtx.T, in_vec).flatten()
                return np.ma.filled(masked_product, fill_value=0.0)

    def _create_mask_cache(self, d_inputs, d_residuals, mode):
        """
        Create masking array for this matrix.

        Note: this only applies when this Matrix is an 'ext_mtx' inside of a
        Jacobian object.

        Parameters
        ----------
        d_inputs : Vector
            The inputs linear vector.
        d_residuals : Vector
            The residuals linear vector.
        mode : str
            Derivative direction ('fwd' or 'rev').

        Returns
        -------
        ndarray or None
            The mask array or None.
        """
        if len(d_inputs._views) > len(d_inputs._names):
            sub = d_inputs._names
            mask = np.ones(len(d_inputs), dtype=np.bool)
            for key, val in iteritems(self._metadata):
                if key[1] in sub:
                    mask[val[1]] = False

            return mask
