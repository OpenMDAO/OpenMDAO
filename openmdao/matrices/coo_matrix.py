"""Define the COOmatrix class."""
from __future__ import division, print_function

from collections import Counter, defaultdict
import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix

from six import iteritems
from six.moves import range

from collections import OrderedDict

from openmdao.matrices.matrix import Matrix, _compute_index_map, sparse_types


class COOMatrix(Matrix):
    """
    Sparse matrix in Coordinate list format.

    Attributes
    ----------
    _mat_range_cache : dict
        Dictionary of cached CSC matrices needed for solving on a sub-range of the
        parent CSC matrix.
    _coo : coo_matrix
        COO matrix. Used as a basis for conversion to CSC, CSR, Dense in inherited classes.
    """

    def __init__(self, comm):
        """
        Initialize all attributes.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm>
            communicator of the top-level system that owns the <Jacobian>.
        """
        super(COOMatrix, self).__init__(comm)
        self._mat_range_cache = {}
        self._coo = None

    def _build_sparse(self, num_rows, num_cols):
        """
        Allocate the data, rows, and cols for the sparse matrix.

        Parameters
        ----------
        num_rows : int
            number of rows in the matrix.
        num_cols : int
            number of cols in the matrix.

        Returns
        -------
        (ndarray, ndarray, ndarray)
            data, rows, cols that can be used to construct a sparse matrix.
        """
        submats = self._submats
        metadata = self._metadata
        pre_metadata = self._key_ranges = OrderedDict()

        start = end = 0
        for key, (info, loc, src_indices, shape, factor) in iteritems(submats):
            val = info['value']
            rows = info['rows']
            dense = (rows is None and (val is None or isinstance(val, ndarray)))

            full_size = np.prod(shape)
            if dense:
                if src_indices is None:
                    delta = full_size
                else:
                    delta = shape[0] * len(src_indices)
            elif rows is None:  # sparse matrix
                delta = val.data.size
            else:  # list sparse format
                delta = len(rows)

            end += delta
            pre_metadata[key] = (start, end, dense, rows)
            start = end

        data = np.zeros(end)
        rows = np.empty(end, dtype=int)
        cols = np.empty(end, dtype=int)

        for key, (start, end, dense, jrows) in iteritems(pre_metadata):
            info, loc, src_indices, shape, factor = submats[key]
            irow, icol = loc
            val = info['value']
            idxs = None

            if dense:
                jac_type = ndarray

                if src_indices is None:
                    colrange = np.arange(shape[1], dtype=int)
                else:
                    colrange = src_indices

                ncols = colrange.size
                subrows = rows[start:end]
                subcols = cols[start:end]

                for i in range(shape[0]):
                    subrows[i * ncols: (i + 1) * ncols] = i
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
                    rows[start:end] = jrows + irow
                    cols[start:end] = jcols + icol
                else:
                    irows, icols, idxs = _compute_index_map(jrows, jcols,
                                                            irow, icol,
                                                            src_indices)
                    rows[start:end] = irows
                    cols[start:end] = icols

            metadata[key] = (start, end, idxs, jac_type, factor)

        return data, rows, cols

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
        data, rows, cols = self._build_sparse(num_rows, num_cols)

        metadata = self._metadata
        for key, (ind1, ind2, idxs, jac_type, factor) in iteritems(metadata):
            if idxs is None:
                metadata[key] = (slice(ind1, ind2), jac_type, factor)
            else:
                # store reverse indices to avoid copying subjac data during
                # update_submat.
                metadata[key] = (np.argsort(idxs) + ind1, jac_type, factor)

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

    def _prod(self, in_vec, mode, ranges, mask=None):
        """
        Perform a matrix vector product.

        Parameters
        ----------
        in_vec : ndarray[:]
            incoming vector to multiply.
        mode : str
            'fwd' or 'rev'.
        ranges : (int, int, int, int)
            Min row, max row, min col, max col for the current system.
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
        if ranges is not None:
            rstart, rend, cstart, cend = ranges
            if rstart != 0 or cstart != 0 or rend != mat.shape[0] or cend != mat.shape[1]:
                if ranges in self._mat_range_cache:
                    mat, idxs = self._mat_range_cache[ranges]

                    # update the data array of our smaller cached matrix with current data from
                    # self._matrix
                    mat.data[:] = self._matrix.data[idxs]
                else:
                    rstart, rend, cstart, cend = ranges
                    rmat = mat.tocoo()

                    # find all row and col indices that are within the desired range
                    ridxs = np.nonzero(np.logical_and(rmat.row >= rstart, rmat.row < rend))[0]
                    cidxs = np.nonzero(np.logical_and(rmat.col >= cstart, rmat.col < cend))[0]

                    # take the intersection since both rows and cols must be within range
                    idxs = np.intersect1d(ridxs, cidxs, assume_unique=True)

                    # create a new smaller csc matrix using only the parts of self._matrix that
                    # are within range
                    mat = coo_matrix((rmat.data[idxs], (rmat.row[idxs] - rstart,
                                                        rmat.col[idxs] - cstart)),
                                     shape=(rend - rstart, cend - cstart))
                    mat = mat.tocsc()
                    self._mat_range_cache[ranges] = mat, idxs

        # NOTE: both mask and ranges will never be defined at the same time.  ranges applies only
        #       to int_mtx and mask applies only to ext_mtx.

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

        Note: this only applies when this Matrix is an 'ext_mtx' inside of a
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
        if len(d_inputs._views) > len(d_inputs._names):
            input_names = d_inputs._names
            mask = np.ones(self._matrix.data.size, dtype=np.bool)
            for key, val in iteritems(self._key_ranges):
                if key[1] in input_names:
                    ind1, ind2, _, _ = val
                    mask[ind1:ind2] = False

            return mask

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
            self._coo.data = self._coo.data.astype(np.complex)
            self._coo.dtype = np.complex
        else:
            self._coo.data = self._coo.data.real
            self._coo.dtype = np.float
