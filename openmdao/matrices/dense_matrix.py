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

    def _update_submat(self, subjac, randgen=None):
        """
        Update the values of a sub-jacobian.

        Parameters
        ----------
        subjac : Subjac
            the sub-jacobian
        randgen : RandomState or None
            Random number generator.
        """
        subjac.update_dense(self._matrix, randgen)

    def transpose(self):
        """
        Transpose the matrix.

        Returns
        -------
        coo_matrix
            Transposed matrix.
        """
        return self._matrix.T

    def _prod(self, in_vec, mode):
        """
        Perform a matrix vector product.

        Parameters
        ----------
        in_vec : ndarray[:]
            incoming vector to multiply.
        mode : str
            'fwd' or 'rev'.

        Returns
        -------
        ndarray[:]
            vector resulting from the product.
        """
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
