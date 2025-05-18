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
            self._matrix_T = self._matrix.T
        return self._matrix_T

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
                self._matrix.data = self._matrix.data.astype(complex)
                self._coo.data = self._coo.data.astype(complex)
        elif is_complex:
            self._matrix.data = self._matrix.data.astype(float)
            self._coo.data = self._coo.data.astype(float)
