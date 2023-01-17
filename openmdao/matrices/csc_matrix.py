"""Define the CSCmatrix class."""

import numpy as np
from scipy.sparse import csc_matrix

from openmdao.matrices.coo_matrix import COOMatrix


class CSCMatrix(COOMatrix):
    """
    Sparse matrix in Compressed Col Storage format.

    Parameters
    ----------
    comm : MPI.Comm or <FakeComm>
        Communicator of the top-level system that owns the <Jacobian>.
    is_internal : bool
        If True, this is the int_mtx of an AssembledJacobian.
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
        super()._build(num_rows, num_cols, system)
        self._coo = self._matrix

    def _pre_update(self):
        """
        Do anything that needs to be done at the start of AssembledJacobian._update.
        """
        self._matrix = self._coo

    def _post_update(self):
        """
        Do anything that needs to be done at the end of AssembledJacobian._update.
        """
        coo = self._coo
        # this will add any repeated entries together
        # NOTE: The CSC matrix was created in the following way instead of using self._coo.tocsc()
        # because on older versions of scipy, self._coo.tocsc() reuses the row/col arrays and the
        # result is that self._coo.row and self._coo.col get scrambled after csc conversion.
        self._matrix = csc_matrix((coo.data, (coo.row, coo.col)), shape=coo.shape)

    def _convert_mask(self, mask):
        """
        Convert the mask to the format of this sparse matrix (CSC, etc.) from COO.

        Parameters
        ----------
        mask : ndarray
            The mask of indices to zero out.

        Returns
        -------
        ndarray
            The converted mask array.
        """
        coo = self._coo
        csc = csc_matrix((mask, (coo.row, coo.col)), shape=coo.shape)
        return csc.data

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
        if active:
            if 'complex' not in self._matrix.dtype.__str__():
                self._matrix.data = self._matrix.data.astype(complex)
                self._matrix.dtype = complex
                self._coo.data = self._coo.data.astype(complex)
                self._coo.dtype = complex
        else:
            self._matrix.data = self._matrix.data.real
            self._matrix.dtype = float
            self._coo.data = self._coo.data.real
            self._coo.dtype = float
