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

    def _build(self, num_rows, num_cols, in_ranges, out_ranges):
        """
        Allocate the matrix.

        Parameters
        ----------
        num_rows : int
            number of rows in the matrix.
        num_cols : int
            number of cols in the matrix.
        in_ranges : dict
            Maps input var name to column range.
        out_ranges : dict
            Maps output var name to row range.
        """
        super()._build(num_rows, num_cols, in_ranges, out_ranges)
        self._coo = self._matrix

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
        coo = self._coo
        # this will add any repeated entries together
        # NOTE: The CSR matrix was created in the following way instead of using self._coo.tocsr()
        # because on older versions of scipy, self._coo.tocsr() reuses the row/col arrays and the
        # result is that self._coo.row and self._coo.col get scrambled after csr conversion.
        self._matrix = csr_matrix((coo.data, (coo.row, coo.col)), shape=coo.shape)
        self._matrix_T = None

    def transpose(self):
        """
        Transpose the matrix.

        Returns
        -------
        csc_matrix
            Transposed matrix.
        """
        if self._matrix_T is None:
            self._matrix_T = self._matrix.T
        return self._matrix_T
