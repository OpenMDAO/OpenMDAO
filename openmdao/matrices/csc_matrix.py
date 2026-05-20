"""Define the CSCmatrix class."""

import numpy as np
from scipy.sparse import csc_matrix

from openmdao.core.constants import INT_DTYPE
from openmdao.matrices.coo_matrix import COOMatrix


class CSCMatrix(COOMatrix):
    """
    Sparse matrix in Compressed Col Storage format.

    Eliminates redundant COO storage by pre-computing a single array mapping
    each COO entry to its position in the CSC data array. Updates are applied
    in-place directly to the CSC matrix, and COO arrays are discarded after
    build. Duplicate (row, col) entries are handled by zeroing CSC data before
    each update cycle and accumulating contributions with +=.

    Parameters
    ----------
    submats : dict
        Dictionary of sub-jacobian data keyed by (row_name, col_name).

    Attributes
    ----------
    _coo_to_csc_map : ndarray of int
        Single array of size nnz. Entry i holds the CSC data index for COO
        entry i. Multiple COO entries may map to the same CSC index when
        duplicate (row, col) pairs exist across subjacs.
    _has_within_subjac_duplicates : dict
        Maps subjac key to bool. True if that subjac has duplicate (row, col)
        entries within its own COO slice, requiring np.add.at instead of +=.
    """

    def __init__(self, submats):
        """
        Initialize all attributes.
        """
        super().__init__(submats)
        self._coo_to_csc_map = None
        self._has_within_subjac_duplicates = None

    def _build(self, num_rows, num_cols, dtype=float):
        """
        Build COO, convert to CSC once, compute mapping, then discard COO arrays.

        Uses lexsort to compute the COO-to-CSC index mapping, which correctly
        handles duplicate (row, col) entries by mapping them all to the same
        CSC index. All COO arrays are discarded after the mapping is built to
        avoid storing redundant data.
        """
        super()._build(num_rows, num_cols, dtype)

        coo = self._coo
        n_entries = coo.data.size

        # Build CSC once from COO
        self._matrix = csc_matrix((coo.data, (coo.row, coo.col)), shape=coo.shape)

        # Compute COO-to-CSC mapping via lexsort (col-major order, same as CSC).
        # For duplicate (row, col) pairs, all duplicates map to the same CSC index.
        sort_order = np.lexsort((coo.row, coo.col))
        sorted_row = coo.row[sort_order]
        sorted_col = coo.col[sort_order]

        # Mark the first occurrence of each unique (row, col) pair
        is_new = np.ones(n_entries, dtype=bool)
        if n_entries > 1:
            is_new[1:] = (np.diff(sorted_row) != 0) | (np.diff(sorted_col) != 0)

        # Assign CSC indices: increment for each new unique entry, same for duplicates
        csc_idx = np.cumsum(is_new, dtype=INT_DTYPE) - 1

        # Map back to original COO order
        self._coo_to_csc_map = np.empty(n_entries, dtype=INT_DTYPE)
        self._coo_to_csc_map[sort_order] = csc_idx

        # Pre-compute per-subjac duplicate flags so _update_from_submat can use
        # fast += for the common case and np.add.at only when needed.
        self._has_within_subjac_duplicates = {}
        for key, coo_slice in self._coo_slices.items():
            idx = self._coo_to_csc_map[coo_slice]
            n = idx.size
            self._has_within_subjac_duplicates[key] = (
                n > 1 and np.unique(idx).size != n
            )

        # Discard all COO arrays - they are no longer needed
        coo.row = np.array([], dtype=INT_DTYPE)
        coo.col = np.array([], dtype=INT_DTYPE)
        coo.data = np.array([], dtype=dtype)

    def _pre_update(self, dtype):
        """
        Update CSC dtype if needed and zero data to prepare for accumulation.

        Parameters
        ----------
        dtype : dtype
            The dtype of the jacobian.
        """
        super()._pre_update(dtype)
        # Zero CSC data so contributions from each subjac can be accumulated
        self._matrix.data[:] = 0.

    def _update_from_submat(self, subjac, randgen):
        """
        Accumulate subjac data directly into the CSC matrix.

        Uses += so that multiple subjacs writing to the same (row, col) position
        (duplicate entries) are handled correctly.

        Parameters
        ----------
        subjac : Subjac
            The sub-jacobian to update from.
        randgen : numpy.random.RandomState
            Random number generator for complex step.
        """
        csc_indices = self._coo_to_csc_map[self._coo_slices[subjac.key]]
        data = subjac.get_as_coo_data(randgen)
        if subjac.factor is not None:
            data = data * subjac.factor
        if self._has_within_subjac_duplicates[subjac.key]:
            # Rare case: within-subjac duplicate (row, col) entries require unbuffered add
            np.add.at(self._matrix.data, csc_indices, data)
        else:
            self._matrix.data[csc_indices] += data

    def _post_update(self):
        """
        Reset transpose cache.

        CSC data has already been updated in-place via _update_from_submat.
        No rebuild is needed.
        """
        self._matrix_T = None

    def dump(self, msginfo):
        """
        Dump the matrix to stdout.

        Parameters
        ----------
        msginfo : str
            Message info.
        """
        print(f"{msginfo}: CSCMatrix:")
        coo = self._matrix.tocoo()
        for r, c, v in sorted(zip(coo.row, coo.col, coo.data)):
            print(f"{r}, {c}, {v}")

    def todense(self):
        """
        Return a dense version of the matrix.

        Returns
        -------
        ndarray
            Dense array representation of the matrix.
        """
        return self._matrix.toarray()

    def transpose(self):
        """
        Transpose the matrix.

        Returns
        -------
        csr_matrix
            Transposed matrix.
        """
        if self._matrix_T is None:
            # the transpose should only happen in reverse mode for apply_linear, and _matrix.T
            # will be CSR, which is preferred for a matvec product.
            self._matrix_T = self._matrix.T
        return self._matrix_T
