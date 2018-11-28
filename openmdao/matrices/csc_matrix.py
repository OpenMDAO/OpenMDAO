"""Define the CSCmatrix class."""

import numpy as np
from scipy.sparse import coo_matrix
from six import iteritems
from collections import Counter

from openmdao.matrices.coo_matrix import COOMatrix, _get_dup_partials


class CSCMatrix(COOMatrix):
    """
    Sparse matrix in Compressed Col Storage format.

    Attributes
    ----------
    _coo : coo_matrix
        COO matrix to be converted into CSC.
    """

    def __init__(self, comm):
        """
        Initialize all attributes.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm>
            communicator of the top-level system that owns the <Jacobian>.
        """
        super(CSCMatrix, self).__init__(comm)
        self._coo = None

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
        # this will add any repeated entries together
        self._matrix = self._matrix.tocsc()
