"""Define the CSCmatrix class."""

from scipy.sparse import csc_matrix

from openmdao.matrices.coo_matrix import COOMatrix


class CSCMatrix(COOMatrix):
    """
    Sparse matrix in Compressed Col Storage format.
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
        super(CSCMatrix, self)._build(num_rows, num_cols, in_ranges, out_ranges)
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
        coo = self._coo
        # this will add any repeated entries together
        # NOTE: this form of the ctor was used instead of self._coo.tocsc() because
        # on older versions of scipy the row/col arrays are reused and the result is
        # that self._coo.row and self._coo.col get scrambled after csc conversion.
        self._matrix = csc_matrix((coo.data, (coo.row, coo.col)), shape=coo.shape)
