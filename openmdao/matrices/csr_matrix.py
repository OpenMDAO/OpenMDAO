"""Define the CSRmatrix class."""

from scipy.sparse import csr_matrix

from openmdao.matrices.coo_matrix import COOMatrix


class CSRMatrix(COOMatrix):
    """
    Sparse matrix in Compressed Row Storage format.

    Parameters
    ----------
    submats : dict
        Dictionary of sub-jacobian data keyed by (row_name, col_name).
    """

    def _pre_update(self, dtype):
        """
        Do anything that needs to be done at the end of SplitJacobian._update.

        Parameters
        ----------
        dtype : dtype
            The dtype of the jacobian.
        """
        super()._pre_update(dtype)
        self._matrix = self._coo

    def _post_update(self):
        """
        Do anything that needs to be done at the end of SplitJacobian._update.
        """
        # this will add any repeated entries together
        coo = self._coo
        # this will add any repeated entries together
        # NOTE: The CSR matrix was created in the following way instead of using self._coo.tocsr()
        # because on older versions of scipy, self._coo.tocsr() reuses the row/col arrays and the
        # result is that self._coo.row and self._coo.col get scrambled after csr conversion.
        self._matrix = csr_matrix((coo.data, (coo.row, coo.col)), shape=coo.shape)
        self._matrix_T = None  # reset the transpose

    def transpose(self):
        """
        Transpose the matrix.

        Returns
        -------
        csc_matrix
            Transposed matrix.
        """
        if self._matrix_T is None:
            coo_T = self._coo.T
            self._matrix_T = csr_matrix((coo_T.data, (coo_T.row, coo_T.col)), shape=coo_T.shape)
        return self._matrix_T
