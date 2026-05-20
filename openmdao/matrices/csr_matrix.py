"""Define the CSRmatrix class."""

import numpy as np
from scipy.sparse import csr_matrix

from openmdao.core.constants import INT_DTYPE
from openmdao.matrices.coo_matrix import COOMatrix


class CSRMatrix(COOMatrix):
    """
    Sparse matrix in Compressed Row Storage format.

    Eliminates redundant COO storage by pre-computing a single array mapping
    each COO entry to its position in the CSR data array. Updates are applied
    in-place directly to the CSR matrix, and COO arrays are discarded after
    build. Duplicate (row, col) entries are handled by zeroing CSR data before
    each update cycle and accumulating contributions with np.add.at.

    Parameters
    ----------
    submats : dict
        Dictionary of sub-jacobian data keyed by (row_name, col_name).

    Attributes
    ----------
    _coo_to_csr_map : ndarray of int
        Single array of size nnz. Entry i holds the CSR data index for COO
        entry i. Multiple COO entries may map to the same CSR index when
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
        self._coo_to_csr_map = None
        self._has_within_subjac_duplicates = None

    def _build(self, num_rows, num_cols, dtype=float):
        """
        Build COO, convert to CSR once, compute mapping, then discard COO arrays.

        Uses lexsort to compute the COO-to-CSR index mapping in row-major order,
        which correctly handles duplicate (row, col) entries by mapping them all
        to the same CSR index. All COO arrays are discarded after the mapping
        is built to avoid storing redundant data.
        """
        super()._build(num_rows, num_cols, dtype)

        coo = self._coo
        n_entries = coo.data.size

        # Build CSR once from COO
        self._matrix = csr_matrix((coo.data, (coo.row, coo.col)), shape=coo.shape)

        # Compute COO-to-CSR mapping via lexsort (row-major order, same as CSR).
        # For duplicate (row, col) pairs, all duplicates map to the same CSR index.
        sort_order = np.lexsort((coo.col, coo.row))
        sorted_row = coo.row[sort_order]
        sorted_col = coo.col[sort_order]

        # Mark the first occurrence of each unique (row, col) pair
        is_new = np.ones(n_entries, dtype=bool)
        if n_entries > 1:
            is_new[1:] = (np.diff(sorted_row) != 0) | (np.diff(sorted_col) != 0)

        # Assign CSR indices: increment for each new unique entry, same for duplicates
        csr_idx = np.cumsum(is_new, dtype=INT_DTYPE) - 1

        # Map back to original COO order
        self._coo_to_csr_map = np.empty(n_entries, dtype=INT_DTYPE)
        self._coo_to_csr_map[sort_order] = csr_idx

        # Pre-compute per-subjac duplicate flags so _update_from_submat can use
        # fast += for the common case and np.add.at only when needed.
        self._has_within_subjac_duplicates = {}
        for key, coo_slice in self._coo_slices.items():
            idx = self._coo_to_csr_map[coo_slice]
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
        Update CSR dtype if needed and zero data to prepare for accumulation.

        Parameters
        ----------
        dtype : dtype
            The dtype of the jacobian.
        """
        super()._pre_update(dtype)
        # Zero CSR data so contributions from each subjac can be accumulated
        self._matrix.data[:] = 0.

    def _update_from_submat(self, subjac, randgen):
        """
        Accumulate subjac data directly into the CSR matrix.

        Uses np.add.at so that multiple subjacs writing to the same (row, col)
        position (duplicate entries) are handled correctly.

        Parameters
        ----------
        subjac : Subjac
            The sub-jacobian to update from.
        randgen : numpy.random.RandomState
            Random number generator for complex step.
        """
        csr_indices = self._coo_to_csr_map[self._coo_slices[subjac.key]]
        data = subjac.get_as_coo_data(randgen)
        if subjac.factor is not None:
            data = data * subjac.factor
        if self._has_within_subjac_duplicates[subjac.key]:
            # Rare case: within-subjac duplicate (row, col) entries require unbuffered add
            np.add.at(self._matrix.data, csr_indices, data)
        else:
            self._matrix.data[csr_indices] += data

    def _post_update(self):
        """
        Reset transpose cache.

        CSR data has already been updated in-place via _update_from_submat.
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
        print(f"{msginfo}: CSRMatrix:")
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
        csc_matrix
            Transposed matrix.
        """
        if self._matrix_T is None:
            self._matrix_T = self._matrix.T
        return self._matrix_T
