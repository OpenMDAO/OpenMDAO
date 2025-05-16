"""Define the COOmatrix class."""
import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix


from openmdao.core.constants import INT_DTYPE
from openmdao.matrices.matrix import Matrix


class COOMatrix(Matrix):
    """
    Sparse matrix in Coordinate list format.

    Parameters
    ----------
    submats : dict
        Dictionary of sub-jacobian data keyed by (row_name, col_name).

    Attributes
    ----------
    _coo : coo_matrix
        COO matrix. Used as a basis for conversion to CSC, CSR, Dense in inherited classes.
    _matrix_T : sparse matrix
        Transpose of the matrix.  Only computed if needed for reverse mode for CSC or CSR
        matrices.
    """

    def __init__(self, submats):
        """
        Initialize all attributes.
        """
        super().__init__(submats)
        self._coo = None
        self._matrix_T = None

    def _build_coo(self, dtype=float):
        """
        Allocate the data, rows, and cols for the COO matrix.

        Parameters
        ----------
        dtype : dtype
            The dtype of the matrix.

        Returns
        -------
        (ndarray, ndarray, ndarray)
            data, rows, cols that can be used to construct a COO matrix.
        """
        submats = self._submats
        key_ranges = {}

        # compute the ranges of the subjacs within the COO data/rows/cols arrays
        start = end = 0
        for key, submat in submats.items():
            end += submat.get_coo_data_size()
            key_ranges[key] = (start, end, submat)
            start = end

        rows = np.empty(end, dtype=INT_DTYPE)
        cols = np.empty(end, dtype=INT_DTYPE)
        data = np.zeros(end, dtype=dtype)

        metadata = self._metadata
        for key, (start, end, submat) in key_ranges.items():
            row_offset = submat.row_slice.start
            col_offset = submat.col_slice.start
            rows[start:end] = submat.rows() + row_offset
            cols[start:end] = submat.cols() + col_offset
            metadata[key] = (slice(start, end), submat.factor)

        return data, rows, cols

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
        data, rows, cols = self._build_coo(dtype)
        self._matrix = self._coo = coo_matrix((data, (rows, cols)), shape=(num_rows, num_cols))

    def _update_submat(self, key, subjac, randgen=None):
        """
        Update the values of a sub-jacobian.

        Parameters
        ----------
        key : (str, str)
            the of and wrt variable names.
        subjac : Subjac
            the sub-jacobian
        randgen : RandomState or None
            Random number generator.
        """
        jac = subjac.get_as_coo_data(randgen)
        idxs, factor = self._metadata[key]
        if isinstance(jac, ndarray):
            if factor is None:
                self._matrix.data[idxs] = jac.flat
            else:
                self._matrix.data[idxs] = jac.ravel() * factor
        else:  # sparse
            if factor is None:
                self._matrix.data[idxs] = jac.data
            else:
                self._matrix.data[idxs] = jac.data * factor

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
            return self._matrix.dot(in_vec)
        else:  # rev
            return self._matrix.transpose().dot(in_vec)

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
            if 'complex' not in self._coo.dtype.__str__():
                self._coo.data = self._coo.data.astype(complex)
                self._coo.dtype = complex
        else:
            self._coo.data = self._coo.data.real
            self._coo.dtype = float

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
        return mask

    def toarray(self):
        """
        Return the matrix as a dense array.

        Returns
        -------
        ndarray
            Dense array representation of the matrix.
        """
        return self._matrix.toarray()
