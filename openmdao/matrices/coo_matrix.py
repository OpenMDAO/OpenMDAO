"""Define the COOmatrix class."""
import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix, csc_matrix

from collections import OrderedDict

from openmdao.core.constants import INT_DTYPE
from openmdao.matrices.matrix import Matrix, _compute_index_map


class COOMatrix(Matrix):
    """
    Sparse matrix in Coordinate list format.

    Parameters
    ----------
    comm : MPI.Comm or <FakeComm>
        Communicator of the top-level system that owns the <Jacobian>.
    is_internal : bool
        If True, this is the int_mtx of an AssembledJacobian.

    Attributes
    ----------
    _coo : coo_matrix
        COO matrix. Used as a basis for conversion to CSC, CSR, Dense in inherited classes.
    """

    def __init__(self, comm, is_internal):
        """
        Initialize all attributes.
        """
        super().__init__(comm, is_internal)
        self._coo = None

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
        metadata = self._metadata
        key_ranges = self._key_ranges = OrderedDict()

        start = end = 0
        for key, (info, loc, src_indices, shape, factor) in submats.items():

            val = info['val']
            rows = info['rows']
            dense = (rows is None and (val is None or isinstance(val, ndarray)))

            if dense:
                if src_indices is None:
                    end += val.size
                else:
                    end += shape[0] * src_indices.indexed_src_size

            elif rows is None:  # sparse matrix
                end += val.data.size
            else:  # list sparse format
                end += len(rows)

            key_ranges[key] = (start, end, dense, rows)
            start = end

        data = np.zeros(end)
        rows = np.empty(end, dtype=INT_DTYPE)
        cols = np.empty(end, dtype=INT_DTYPE)

        for key, (start, end, dense, jrows) in key_ranges.items():
            info, loc, src_indices, shape, factor = submats[key]
            irow, icol = loc
            val = info['val']
            idxs = None

            col_offset = row_offset = 0

            if dense:
                jac_type = ndarray

                if src_indices is None:
                    colrange = np.arange(col_offset, col_offset + shape[1], dtype=INT_DTYPE)
                else:
                    colrange = src_indices.shaped_array()

                ncols = colrange.size
                subrows = rows[start:end]
                subcols = cols[start:end]

                for i in range(shape[0]):
                    subrows[i * ncols: (i + 1) * ncols] = i + row_offset
                    subcols[i * ncols: (i + 1) * ncols] = colrange

                subrows += irow
                subcols += icol

            else:  # sparse
                if jrows is None:
                    jac_type = type(val)
                    jac = val.tocoo()
                    jrows = jac.row
                    jcols = jac.col
                else:
                    jac_type = list
                    jcols = info['cols']

                if src_indices is None:
                    rows[start:end] = jrows + (irow + row_offset)
                    cols[start:end] = jcols + (icol + col_offset)
                else:
                    irows, icols, idxs = _compute_index_map(jrows, jcols,
                                                            irow, icol,
                                                            src_indices)
                    rows[start:end] = irows
                    cols[start:end] = icols

            metadata[key] = (start, end, idxs, jac_type, factor)

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

        metadata = self._metadata
        for key, (start, end, idxs, jac_type, factor) in metadata.items():
            if idxs is None:
                metadata[key] = (slice(start, end), jac_type, factor)
            else:
                # store reverse indices to avoid copying subjac data during
                # update_submat.
                metadata[key] = (np.argsort(idxs) + start, jac_type, factor)

        self._matrix = self._coo = coo_matrix((data, (rows, cols)), shape=(num_rows, num_cols))

    def _update_submat(self, key, jac):
        """
        Update the values of a sub-jacobian.

        Parameters
        ----------
        key : (str, str)
            the global output and input variable names.
        jac : ndarray or scipy.sparse or tuple
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

    def _prod(self, in_vec, mode, mask=None):
        """
        Perform a matrix vector product.

        Parameters
        ----------
        in_vec : ndarray[:]
            incoming vector to multiply.
        mode : str
            'fwd' or 'rev'.
        mask : ndarray of type bool, or None
            Array used to zero out part of the matrix data.

        Returns
        -------
        ndarray[:]
            vector resulting from the product.
        """
        # when we have a derivative based solver at a level below the
        # group that owns the AssembledJacobian, we need to use only
        # the part of the matrix that is relevant to the lower level
        # system.
        mat = self._matrix

        # NOTE: mask applies only to ext_mtx.

        if mode == 'fwd':
            if mask is None:
                return mat.dot(in_vec)
            else:
                save = mat.data[mask]
                mat.data[mask] = 0.0
                val = mat.dot(in_vec)
                mat.data[mask] = save
                return val
        else:  # rev
            if mask is None:
                return mat.T.dot(in_vec)
            else:
                save = mat.data[mask]
                mat.data[mask] = 0.0
                val = mat.T.dot(in_vec)
                mat.data[mask] = save
                return val

    def _create_mask_cache(self, d_inputs):
        """
        Create masking array for this matrix.

        Note : this only applies when this Matrix is an 'ext_mtx' inside of a
        Jacobian object.

        Parameters
        ----------
        d_inputs : Vector
            The inputs linear vector.

        Returns
        -------
        ndarray or None
            The mask array or None.
        """
        if d_inputs._in_matvec_context():
            input_names = d_inputs._names
            mask = None
            for key, val in self._key_ranges.items():
                if key[1] in input_names:
                    if mask is None:
                        mask = np.ones(self._matrix.data.size, dtype=bool)
                    ind1, ind2, _, _ = val
                    mask[ind1:ind2] = False

            if mask is not None:
                # convert the mask indices (if necessary) base on sparse matrix type
                # (CSC, CSR, etc.)
                return self._convert_mask(mask)

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
