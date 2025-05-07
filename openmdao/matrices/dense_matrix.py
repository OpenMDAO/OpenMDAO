"""Define the DenseMatrix class."""

from openmdao.matrices.coo_matrix import COOMatrix

# NOTE: DenseMatrix is inherited from COOMatrix so that we can easily handle use cases
#       where partials overlap the same matrix entries, as in the case of repeated
#       src_indices entries.  This does require additional memory above storing just
#       the dense matrix, but it's worth it because the code is simpler and more robust.


class DenseMatrix(COOMatrix):
    """
    Dense global matrix.

    Parameters
    ----------
    submats : dict
        Dictionary of sub-jacobian data keyed by (row_name, col_name).
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
