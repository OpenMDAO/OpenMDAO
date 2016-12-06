"""Define the DenseMatrix class."""
from __future__ import division, print_function
import numpy
import scipy.sparse

from openmdao.matrices.matrix import Matrix


class DenseMatrix(Matrix):
    """Dense global matrix."""

    def _build(self, num_rows, num_cols):
        """Allocate the matrix.

        Args
        ----
        num_rows : int
            number of rows in the matrix.
        num_cols : int
            number of cols in the matrix.
        """
        matrix = numpy.zeros((num_rows, num_cols))
        for ind in range(2):
            submats = [self._op_submats, self._ip_submats][ind]
            metadata = [self._op_metadata, self._ip_metadata][ind]

            for key in submats:
                jac, irow, icol = submats[key]

                if type(jac) is numpy.ndarray or scipy.sparse.issparse(jac):
                    shape = jac.shape
                    irow1 = irow
                    irow2 = irow + shape[0]
                    icol1 = icol
                    icol2 = icol + shape[1]
                    metadata[key] = matrix[irow1:irow2, icol1:icol2]
                    if type(jac) is numpy.ndarray:
                        metadata[key][:, :] = jac
                    else:
                        metadata[key][:, :] = jac.todense()
                elif type(jac) is list:
                    irows = irow + jac[1]
                    icols = icol + jac[2]
                    matrix[irows, icols] = jac[0]
                    metadata[key] = (irows, icols)

        self._matrix = matrix

    def _update_submat(self, submats, metadata, key, jac):
        """Update the values of a sub-jacobian.

        Args
        ----
        submats : dict
            dictionary of sub-jacobian data keyed by (op_ind, ip_ind).
        metadata : dict
            implementation-specific data for the sub-jacobians.
        key : (int, int)
            the global output and input variable indices.
        jac : ndarray or scipy.sparse or tuple
            the sub-jacobian, the same format with which it was declared.
        """
        if type(jac) is numpy.ndarray:
            metadata[key][:, :] = jac
        elif scipy.sparse.issparse(jac):
            metadata[key][:, :] = jac.todense()  # TODO: improve on todense
        elif type(jac) is list:
            irows, icols = metadata[key]
            self._matrix[irows, icols] = jac[0]

    def _prod(self, in_vec, mode):
        """Perform a matrix vector product.

        Args
        ----
        vec : ndarray[:]
            incoming vector to multiply.
        mode : str
            'fwd' or 'rev'.

        Returns
        -------
        ndarray[:]
            vector resulting from the product.
        """
        if mode == 'fwd':
            return self._matrix.dot(in_vec)
        elif mode == 'rev':
            return self._matrix.T.dot(in_vec)
