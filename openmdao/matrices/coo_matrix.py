"""Define the COOmatrix class."""
import numpy as np
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
    _coo_slices : dict
        Dictionary of slices into the COO matrix data/rows/cols for each sub-jacobian.
    """

    def __init__(self, submats):
        """
        Initialize all attributes.
        """
        super().__init__(submats)
        self._coo = None
        self._matrix_T = None
        self._coo_slices = {}

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
        submats = self._submats
        self._coo_slices = {}

        rows = []
        cols = []

        # compute the ranges of the subjacs within the COO data/rows/cols arrays
        start = end = 0
        for key, submat in submats.items():
            _, r, c = submat.as_coo_info(full=True)
            end += r.size
            rows.append(r)
            cols.append(c)
            self._coo_slices[key] = slice(start, end)
            start = end

        rows = np.concatenate(rows, dtype=INT_DTYPE)
        cols = np.concatenate(cols, dtype=INT_DTYPE)
        data = np.zeros(end, dtype=dtype)

        self._matrix = self._coo = coo_matrix((data, (rows, cols)), shape=(num_rows, num_cols))

    def transpose(self):
        """
        Transpose the matrix.

        Returns
        -------
        coo_matrix
            Transposed matrix.
        """
        return self._matrix.T

    def _prod(self, in_vec, mode, mask=None):
        """
        Perform a matrix vector product.

        Parameters
        ----------
        in_vec : ndarray[:]
            incoming vector to multiply.
        mode : str
            'fwd' or 'rev'.
        mask : ndarray or None
            Array used to mask out part of the vector.  If mask is not None then
            the masked values will be set to 0.0.

        Returns
        -------
        ndarray[:]
            vector resulting from the product.
        """
        in_vec = self._get_masked_arr(in_vec, mode, mask)

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
        is_complex = 'complex' in self._coo.dtype.__str__()
        if active:
            if not is_complex:
                self._coo.data = self._coo.data.astype(complex)
        elif is_complex:
            self._coo.data = self._coo.data.real.astype(float)

    def toarray(self):
        """
        Return the matrix as a dense array.

        Returns
        -------
        ndarray
            Dense array representation of the matrix.
        """
        return self._matrix.toarray()

    def _update_from_submat(self, subjac, randgen):
        """
        Update the matrix from a sub-jacobian.
        """
        self._coo.data[self._coo_slices[subjac.key]] = subjac.get_as_coo_data(randgen)
        if subjac.factor is not None:
            self._coo.data[self._coo_slices[subjac.key]] *= subjac.factor

    def todense(self):
        """
        Return a dense version of the matrix.

        Returns
        -------
        ndarray
            Dense array representation of the matrix.
        """
        return self._coo.toarray()
