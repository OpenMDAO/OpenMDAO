"""Define the DenseMatrix class."""

import numpy as np

from openmdao.matrices.matrix import Matrix
from openmdao.matrices.coo_matrix import COOMatrix


class GroupDenseMatrix(COOMatrix):
    """
    Dense jacobian matrix for use in Groups.

    Parameters
    ----------
    submats : dict
        Dictionary of sub-jacobian data keyed by (row_name, col_name).
    """

    # NOTE: GroupDenseMatrix is inherited from COOMatrix so that we can easily handle use cases
    #       where partials overlap the same matrix entries, as in the case of repeated
    #       src_indices entries.  This does require additional memory above storing just
    #       the dense matrix, but it's worth it because the code is simpler and more robust.
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

    def toarray(self):
        """
        Return the matrix as a dense array.

        Returns
        -------
        ndarray
            Dense array representation of the matrix.
        """
        return self._matrix


class DenseMatrix(Matrix):
    """
    Dense jacobian matrix for use in Components.

    This cannot be used in Groups because src_indices are not supported in this case.

    Parameters
    ----------
    submats : dict
        Dictionary of sub-jacobian data keyed by (row_name, col_name).
    """

    def _build(self, num_rows, num_cols, dtype=float):
        """
        Allocate the matrix.

        Parameters
        ----------
        num_rows : int
            number of rows in the matrix.
        num_cols : int
            number of cols in the matrix.
        dtype : dtype
            The dtype of the matrix.
        """
        self._matrix = np.zeros((num_rows, num_cols), dtype=dtype)

    def transpose(self):
        """
        Transpose the matrix.

        Returns
        -------
        coo_matrix
            Transposed matrix.
        """
        return self._matrix.T

    def _prod(self, in_vec, mode, mask=None):
        """
        Perform a matrix vector product.

        Parameters
        ----------
        in_vec : ndarray[:]
            incoming vector to multiply.
        mode : str
            'fwd' or 'rev'.
        mask : ndarray or None
            Array used to mask out part of the vector.  If mask is not None then
            the masked values will be set to 0.0.

        Returns
        -------
        ndarray[:]
            vector resulting from the product.
        """
        in_vec = self._get_masked_arr(in_vec, mode, mask)

        if mode == 'fwd':
            return self._matrix @ in_vec
        else:  # rev
            return self._matrix.transpose() @ in_vec

    def set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        When turned on, the value in each subjac is cast as complex, and when turned
        off, they are returned to real values.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        is_complex = 'complex' in self._matrix.dtype.__str__()
        if active:
            if not is_complex:
                self._matrix = self._matrix.astype(complex)
        elif is_complex:
            self._matrix = self._matrix.real.astype(float)

    def toarray(self):
        """
        Return the matrix as a dense array.

        Returns
        -------
        ndarray
            Dense array representation of the matrix.
        """
        return self._matrix

    def _update_from_submat(self, subjac, randgen):
        """
        Update the matrix from a sub-jacobian.
        """
        if subjac.dense:
            self._matrix[subjac.row_slice, subjac.col_slice] = subjac.get_val(randgen)
            if subjac.factor is not None:
                self._matrix[subjac.row_slice, subjac.col_slice] *= subjac.factor
        else:
            data, rows, cols = subjac.as_coo_info(full=True, randgen=randgen)
            # note that this only works properly if used in a Component where src_indices is
            # always None, so no duplicated row/col entries.
            self._matrix[rows, cols] = data
            if subjac.factor is not None:
                self._matrix[rows, cols] *= subjac.factor

    def todense(self):
        """
        Return a dense version of the matrix.

        Returns
        -------
        ndarray
            Dense array representation of the matrix.
        """
        return self._matrix

    def dump(self, msginfo):
        """
        Dump the matrix to stdout.

        Parameters
        ----------
        msginfo : str
            Message info.
        """
        print(f"{msginfo}: DenseMatrix:")
        with np.printoptions(linewidth=9999, threshold=9999):
            print(self._matrix)
