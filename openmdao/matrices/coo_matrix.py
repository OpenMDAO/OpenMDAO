"""Define the COOmatrix class."""
import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix, issparse


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
    _matrix_T : coo_matrix
        Transpose of the COO matrix.  Only computed if needed for reverse mode for CSC or CSR
        matrices.
    """

    def __init__(self, submats):
        """
        Initialize all attributes.
        """
        super().__init__(submats)
        self._coo = None
        self._matrix_T = None

    def _build_coo(self, system):
        """
        Allocate the data, rows, and cols for the COO matrix.

        Parameters
        ----------
        system : <System>
            Parent system of this matrix.

        Returns
        -------
        (ndarray, ndarray, ndarray)
            data, rows, cols that can be used to construct a COO matrix.
        """
        submats = self._submats
        key_ranges = {}

        start = end = 0
        for key, submat in submats.items():
            size = submat.get_sparse_data_size()
            end += size
            key_ranges[key] = (start, end, submat)
            start = end

        data_size = end

        rows = np.empty(end, dtype=INT_DTYPE)
        cols = np.empty(end, dtype=INT_DTYPE)

        metadata = self._metadata
        for key, (start, end, submat) in key_ranges.items():
            irow = submat.row_slice.start
            icol = submat.col_slice.start
            val = submat.get_val()
            jac_type = type(val)
            idxs = slice(start, end)

            if not issparse(val) and len(val.shape) == 2:  # dense

                if submat.src_indices is None:
                    colrange = range(icol, icol + submat.shape[1])
                else:
                    colrange = submat.src_indices.shaped_array() + icol

                ncols = len(colrange)

                subrows = rows[start:end]
                subcols = cols[start:end]

                substart = subend = 0
                for row in range(irow, irow + submat.shape[0]):
                    subend += ncols
                    subrows[substart:subend] = row
                    subcols[substart:subend] = colrange
                    substart = subend

            else:  # sparse
                if submat.info['diagonal']:
                    jrows = jcols = np.arange(val.size)
                elif submat.info['rows'] is None:
                    jac = val.tocoo()
                    jrows = jac.row
                    jcols = jac.col
                else:
                    jac_type = list
                    jrows = submat.info['rows']
                    jcols = submat.info['cols']

                if submat.src_indices is None:
                    rows[start:end] = jrows + irow
                    cols[start:end] = jcols + icol
                else:
                    irows, icols, idxs = _compute_index_map(jrows, jcols,
                                                            submat.src_indices)

                    rows[start:end] = irows + irow
                    cols[start:end] = icols + icol

                    # store reverse indices to avoid copying subjac data during
                    # update_submat.
                    idxs = np.argsort(idxs) + start

            metadata[key] = (idxs, jac_type, submat.factor)

        if system is not None and system.under_complex_step:
            data = np.zeros(data_size, dtype=complex)
        else:
            data = np.zeros(data_size)

        return data, rows, cols

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
        data, rows, cols = self._build_coo(system)
        self._matrix = self._coo = coo_matrix((data, (rows, cols)), shape=(num_rows, num_cols))

    def _update_submat(self, key, jac):
        """
        Update the values of a sub-jacobian.

        Parameters
        ----------
        key : (str, str)
            the global output and input variable names.
        jac : ndarray or scipy.sparse
            the sub-jacobian, the same format with which it was declared.
        """
        idxs, jac_type, factor = self._metadata[key]
        if not isinstance(jac, jac_type) and (jac_type is list and not isinstance(jac, ndarray)):
            raise TypeError("Jacobian entry for %s is of different type (%s) than "
                            "the type (%s) used at init time." % (key,
                                                                  type(jac).__name__,
                                                                  jac_type.__name__))
        if isinstance(jac, ndarray):
            self._matrix.data[idxs] = jac.flat
        else:  # sparse
            self._matrix.data[idxs] = jac.data

        if factor is not None:
            self._matrix.data[idxs] *= factor

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


def _compute_index_map(jrows, jcols, src_indices):
    """
    Return row/column indices and coo indicesto map sub-jacobian to an 'internal' subjac.

    For a given subjac which is the partial of the residual wrt the input, the 'internal' subjac
    is the partial of the residual wrt the source of that input.

    Parameters
    ----------
    jrows : index array
        Array of row indices.
    jcols : index array
        Array of column indices.
    src_indices : index array
        Index array of which values to pull from a source into an input
        variable.

    Returns
    -------
    tuple of (ndarray, ndarray, ndarray)
        Row indices, column indices, and indices of columns matching
        src_indices.
    """
    icols = src_indices.shaped_array()[jcols]
    return (jrows, icols, np.arange(icols.size))
