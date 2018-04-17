"""Define the COOmatrix class."""
from __future__ import division, print_function

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix

from six import iteritems
from six.moves import range

from collections import OrderedDict, Counter, defaultdict

from openmdao.matrices.matrix import Matrix, _compute_index_map, sparse_types


class COOMatrix(Matrix):
    """
    Sparse matrix in Coordinate list format.
    """

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
        counter = 0

        submats = self._submats
        metadata = self._metadata
        pre_metadata = self._key_ranges = OrderedDict()
        locations = {}

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
            elif rows is None:
                delta = val.data.size
            else:
                delta = len(rows)

            if loc in locations:
                ind1, ind2, otherkey = locations[loc]
                if not (src_indices is None and (ind2 - ind1) == delta == full_size):
                    raise RuntimeError("Keys %s map to the same sub-jacobian of a CSC or "
                                       "CSR partial jacobian and at least one of them is either "
                                       "not dense or uses src_indices.  This can occur when "
                                       "multiple inputs on the same "
                                       "component are connected to the same output." %
                                       sorted((key, otherkey)))
            else:
                ind1 = counter
                counter += delta
                ind2 = counter
                locations[loc] = (ind1, ind2, key)

            pre_metadata[key] = (ind1, ind2, dense)

        data = np.zeros(counter)
        rows = np.empty(counter, dtype=int)
        cols = np.empty(counter, dtype=int)

        for key, (ind1, ind2, dense) in iteritems(pre_metadata):
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
                subrows = rows[ind1:ind2]
                subcols = cols[ind1:ind2]

                for i in range(shape[0]):
                    subrows[i * ncols: (i + 1) * ncols] = i
                    subcols[i * ncols: (i + 1) * ncols] = colrange

                subrows += irow
                subcols += icol

            else:  # sparse
                if isinstance(val, sparse_types):
                    jac_type = type(val)
                    jac = val.tocoo()
                    jrows = jac.row
                    jcols = jac.col
                else:
                    jac_type = list
                    jrows = info['rows']
                    jcols = info['cols']

                if src_indices is None:
                    rows[ind1:ind2] = jrows + irow
                    cols[ind1:ind2] = jcols + icol
                else:
                    irows, icols, idxs = _compute_index_map(jrows, jcols,
                                                            irow, icol,
                                                            src_indices)
                    rows[ind1:ind2] = irows
                    cols[ind1:ind2] = icols

            metadata[key] = (ind1, ind2, idxs, jac_type, factor)

        return data, rows, cols

    def _build(self, num_rows, num_cols):
        """
        Allocate the matrix.

        Parameters
        ----------
        num_rows : int
            number of rows in the matrix.
        num_cols : int
            number of cols in the matrix.
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

        self._matrix = coo_matrix((data, (rows, cols)),
                                  shape=(num_rows, num_cols))

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
        if not isinstance(jac, jac_type):
            raise TypeError("Jacobian entry for %s is of different type (%s) than "
                            "the type (%s) used at init time." % (key,
                                                                  type(jac).__name__,
                                                                  jac_type.__name__))
        if isinstance(jac, ndarray):
            self._matrix.data[idxs] = jac.flat
        elif isinstance(jac, sparse_types):
            self._matrix.data[idxs] = jac.data
        else:  # list format  [data, rows, cols]
            self._matrix.data[idxs] = jac[0]

        if factor is not None:
            self._matrix.data[idxs] *= factor

    def _update_add_submat(self, key, jac):
        """
        Add the subjac values to an existing sub-jacobian.

        Parameters
        ----------
        key : (str, str)
            the global output and input variable names.
        jac : ndarray or scipy.sparse or tuple
            the sub-jacobian, the same format with which it was declared.
        """
        idxs, jac_type, factor = self._metadata[key]
        if not isinstance(jac, jac_type):
            raise TypeError("Jacobian entry for %s is of different type (%s) than "
                            "the type (%s) used at init time." % (key,
                                                                  type(jac).__name__,
                                                                  jac_type.__name__))
        if isinstance(jac, ndarray):
            val = jac.flatten()
        elif isinstance(jac, sparse_types):
            val = jac.data
        else:  # list format  [data, rows, cols]
            val = jac[0]

        if factor is not None:
            self._matrix.data[idxs] += val * factor
        else:
            self._matrix.data[idxs] += val

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
            if not((rend - rstart) == mat.shape[0] and (cend - cstart) == mat.shape[1]):
                raise RuntimeError("Placing a sparse jacobian under another assembled jacobian "
                                   "is currently not supported.")

        if mode == 'fwd':
            return mat.dot(in_vec)
        else:  # rev
            return mat.T.dot(in_vec)
