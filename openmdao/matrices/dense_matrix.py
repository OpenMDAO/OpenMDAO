"""Define the DenseMatrix class."""

import numpy as np

from openmdao.core.constants import INT_DTYPE
from openmdao.matrices.matrix import Matrix
from scipy.sparse import coo_matrix, csc_matrix


class DenseMatrix(Matrix):
    """
    The underlying matrix may be a COO or a dense array.

    The type of the underlying matrix will be COO if the array has repeated indices or dense
    otherwise.

    Used with the SplitJacobian to represent the dr/do and dr/di matrices, to form a matvec
    product with the d_outputs and d_inputs vectors respectively.

    Parameters
    ----------
    submats : dict
        Dictionary of sub-jacobian data keyed by (row_name, col_name).

    Attributes
    ----------
    _coo : coo_matrix or None
        COO matrix used for conversion to dense in some cases, e.g., when array has repeated
        indices.
    _coo_slices : dict or None
        Dictionary of slices into the COO matrix data/rows/cols for each sub-jacobian.
    """

    def __init__(self, submats):
        """
        Initialize all attributes.
        """
        super().__init__(submats)
        self._coo = None
        self._coo_slices = {}

    def _build(self, num_rows, num_cols, dtype=float):
        """
        Allocate the matrix.

        Also, possibly keep track of the slice within the COO data/rows/cols arrays corresponding
        to each sub-jacobian.

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

        # now, determine if we need to keep a coo representation of the matrix in order to
        # handle repeated indices.
        csc = csc_matrix((np.ones(end, dtype=dtype), (rows, cols)), shape=(num_rows, num_cols))

        # csc was created with 1.0 in each data entry, and csc adds repeated entries together, so
        # we can check if there are any entries greater than 1.0 to determine if we have repeated
        # indices.
        has_repeated = np.any(csc.data > 1.0)
        del csc

        if has_repeated:
            # we have repeated indices, so we need to keep a COO representation
            self._coo = coo_matrix((np.zeros(end, dtype=dtype), (rows, cols)),
                                   shape=(num_rows, num_cols))
        else:
            self._coo_slices = None
            self._coo = None
            self._matrix = np.zeros((num_rows, num_cols), dtype=dtype)

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
            if mask is None:
                return self._matrix @ in_vec
            else:
                return self._matrix @ self._get_masked_arr(in_vec, mask)
        else:  # rev
            if mask is None:
                return self.transpose() @ in_vec
            else:
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
            if self._coo is None:
                self.dtype = dtype
                mat = self._matrix if dtype.kind == 'c' else self._matrix.real
                self._matrix = np.ascontiguousarray(mat, dtype=dtype)
            else:
                self.dtype = dtype
                data = self._coo.data if dtype.kind == 'c' else self._coo.data.real
                self._coo.data = np.ascontiguousarray(data, dtype=dtype)
                self._matrix = None

    def _update_from_submat(self, subjac, randgen):
        """
        Update the matrix from a sub-jacobian.
        """
        if self._coo is None:
            if subjac.dense:
                view = self._matrix[subjac.row_slice, subjac.col_slice]
                val = subjac.info['val'] if randgen is None else subjac.get_rand_val(randgen)
                if subjac.src_indices is not None:
                    view[:, subjac.src_indices] = val
                else:
                    view[:, :] = val
                if subjac.factor is not None:
                    view *= subjac.factor
            else:
                data, rows, cols = subjac.as_coo_info(full=True, randgen=randgen)
                self._matrix[rows, cols] = data  # only works if there are no repeated indices
                if subjac.factor is not None:
                    self._matrix[rows, cols] *= subjac.factor
        else:
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
        return self._matrix

    def _pre_update(self, dtype):
        """
        Do anything that needs to be done at the end of jacobian update.

        Parameters
        ----------
        dtype : dtype
            The dtype of the jacobian.
        """
        super()._pre_update(dtype)
        if self._coo is not None:
            # during update, stay in coo format if we have repeated indices
            self._matrix = self._coo

    def _post_update(self):
        """
        Do anything that needs to be done at the end of jacobian update.
        """
        if self._coo is not None:
            # this will add any repeated entries together
            self._matrix = self._coo.toarray()

    def dump(self, msginfo):
        """
        Dump the matrix to stdout.

        Parameters
        ----------
        msginfo : str
            Message info.
        """
        print(f"{msginfo}: DenseMatrix:")
        with np.printoptions(linewidth=9999, threshold=9999):
            print(self._matrix)
