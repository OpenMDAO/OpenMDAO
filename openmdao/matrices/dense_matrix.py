"""Define the DenseMatrix class."""
import numpy as np

from openmdao.matrices.coo_matrix import COOMatrix

# NOTE: DenseMatrix is inherited from COOMatrix so that we can easily handle use cases
#       where partials overlap the same matrix entries, as in the case of repeated
#       src_indices entries.  This does require additional memory above storing just
#       the dense matrix, but it's worth it because the code is simpler and more robust.


class DenseMatrix(COOMatrix):
    """
    Dense global matrix.
    """

    def _build(self, num_rows, num_cols, system=None):
        """
        Allocate the matrix.

        Parameters
        ----------
        num_rows : int
            number of rows in the matrix.
        num_cols : int
            number of cols in the matrix.
        system : <System>
            owning system.
        """
        super()._build(num_rows, num_cols)
        self._coo = self._matrix

    def _prod(self, in_vec, mode, mask=None):
        """
        Perform a matrix vector product.

        Parameters
        ----------
        in_vec : ndarray[:]
            incoming vector to multiply.
        mode : str
            'fwd' or 'rev'.
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
        mat = self._matrix

        if mode == 'fwd':
            if mask is None:
                return mat.dot(in_vec)
            else:
                # Use the special dot product function from masking module so that we
                # ignore masked parts.
                return np.ma.dot(mat, np.ma.array(in_vec, mask=mask))
        else:  # rev
            if mask is None:
                return mat.T.dot(in_vec)
            else:
                # Mask need to be applied to ext_mtx so that we can ignore multiplication
                # by certain columns.
                mat_T = mat.T
                arrmask = np.zeros(mat_T.shape, dtype=np.bool)
                arrmask[mask, :] = True
                masked_mtx = np.ma.array(mat_T, mask=arrmask, fill_value=0.0)

                masked_product = np.ma.dot(masked_mtx, in_vec).flatten()
                return np.ma.filled(masked_product, fill_value=0.0)

    def _create_mask_cache(self, d_inputs):
        """
        Create masking array for this matrix.

        Note : this only applies when this Matrix is an 'ext_mtx' inside of a
        Jacobian object.

        Parameters
        ----------
        d_inputs : Vector
            The inputs linear vector.

        Returns
        -------
        ndarray or None
            The mask array or None.
        """
        if d_inputs._in_matvec_context():
            sub = d_inputs._names
            mask = np.ones(len(d_inputs), dtype=np.bool)
            for key, val in self._metadata.items():
                if key[1] in sub:
                    mask[val[1]] = False

            return mask

    def _pre_update(self):
        """
        Do anything that needs to be done at the end of AssembledJacobian._update.
        """
        self._matrix = self._coo

    def _post_update(self):
        """
        Do anything that needs to be done at the end of AssembledJacobian._update.
        """
        # this will add any repeated entries together
        self._matrix = self._coo.toarray()
