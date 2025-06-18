"""Define the COOmatrix class."""
import numpy as np
from scipy.sparse import coo_matrix


from openmdao.core.constants import INT_DTYPE
from openmdao.matrices.matrix import Matrix


class COOMatrix(Matrix):
    """
    Sparse matrix in Coordinate list format.

    Used with the SplitJacobian to represent the dr/do and dr/di matrices, to form a matvec
    product with the d_outputs and d_inputs vectors respectively.

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

        Also, keep track of the slice within the COO data/rows/cols arrays corresponding to each
        sub-jacobian.

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
        in_vec : ndarray
            incoming vector to multiply.
        mode : str
            'fwd' or 'rev'.
        mask : ndarray or None
            Array used to mask out part of the vector.  If mask is not None then
            the masked values will be set to 0.0.

        Returns
        -------
        ndarray
            vector resulting from the product.
        """
        if mode == 'fwd':
            return self._matrix @ self._get_masked_arr(in_vec, mask)
        else:  # rev
            return self.transpose() @ self._get_masked_arr(in_vec, mask)

    def _update_dtype(self, dtype):
        """
        Update the dtype of the matrix.

        This happens during pre_update.

        Parameters
        ----------
        dtype : dtype
            The new dtype of the matrix.
        """
        if dtype.kind != self.dtype.kind:
            self.dtype = dtype
            data = self._coo.data if dtype.kind == 'c' else self._coo.data.real
            self._coo.data = np.ascontiguousarray(data, dtype=dtype)
            self._matrix_T = None

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

    def dump(self, msginfo):
        """
        Dump the matrix to stdout.

        Parameters
        ----------
        msginfo : str
            Message info.
        """
        print(f"{msginfo}: COOMatrix:")
        for r, c, v in sorted(zip(self._coo.row, self._coo.col, self._coo.data)):
            print(f"{r}, {c}, {v}")
