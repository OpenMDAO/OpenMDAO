"""Define the CSCmatrix class."""

from scipy.sparse import csc_matrix

from openmdao.matrices.coo_matrix import COOMatrix


class CSCMatrix(COOMatrix):
    """
    Sparse matrix in Compressed Col Storage format.

    Parameters
    ----------
    submats : dict
        Dictionary of sub-jacobian data keyed by (row_name, col_name).
    """

    def _pre_update(self, dtype):
        """
        Do anything that needs to be done at the start of SplitJacobian._update.

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
        coo = self._coo
        # this will add any repeated entries together
        # NOTE: The CSC matrix was created in the following way instead of using self._coo.tocsc()
        # because on older versions of scipy, self._coo.tocsc() reuses the row/col arrays and the
        # result is that self._coo.row and self._coo.col get scrambled after csc conversion.
        self._matrix = csc_matrix((coo.data, (coo.row, coo.col)), shape=coo.shape)
        self._matrix_T = None  # reset the transpose

    def transpose(self):
        """
        Transpose the matrix.

        Returns
        -------
        csr_matrix
            Transposed matrix.
        """
        if self._matrix_T is None:
            # the transpose should only happen in reverse mode for apply_linear, and _matrix.T
            # will be CSR, which is preferred for a matvec product.
            self._matrix_T = self._matrix.T
        return self._matrix_T
