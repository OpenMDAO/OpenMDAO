"""Define the CsrMatrix class."""
from __future__ import division

import numpy
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix
from six import iteritems

from openmdao.matrices.matrix import Matrix, _compute_index_map
from openmdao.matrices.coo_matrix import CooMatrix


class CsrMatrix(CooMatrix):
    """Sparse matrix in Compressed Row Storage format."""

    def _build(self, num_rows, num_cols):
        """Allocate the matrix.

        Args
        ----
        num_rows : int
            number of rows in the matrix.
        num_cols : int
            number of cols in the matrix.
        """
        data, rows, cols = self._build_sparse(num_rows, num_cols)

        # get a set of indices that sorts into row major order
        srtidxs = numpy.lexsort((cols, rows))

        data = data[srtidxs]
        rows = rows[srtidxs]
        cols = cols[srtidxs]

        # now sort these back into ascending order (our original stacked order)
        # so in _update_submat() we can just extract the individual index
        # arrays that will map each block into the combined data array.
        revidxs = numpy.argsort(srtidxs)

        metadata = self._metadata
        for key, (ind1, ind2, idxs, jac_type) in iteritems(metadata):
            if idxs is None:
                metadata[key] = (revidxs[ind1:ind2], jac_type)
            else:
                # apply the reverse index to each part of revidxs so that
                # we can avoid copying the index array during updates.
                metadata[key] = (revidxs[ind1:ind2][numpy.argsort(idxs)],
                                 jac_type)

        # data array for the CSR should be the same as for the COO since
        # it was already in sorted order.
        self._matrix = coo_matrix((data, (rows, cols)),
                                  shape=(num_rows, num_cols)).tocsr()
