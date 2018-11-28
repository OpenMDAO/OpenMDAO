"""Define the CSCmatrix class."""

import numpy as np
from scipy.sparse import coo_matrix
from six import iteritems
from collections import Counter

from openmdao.matrices.coo_matrix import COOMatrix, _get_dup_partials


class CSCMatrix(COOMatrix):
    """
    Sparse matrix in Compressed Col Storage format.
    """

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
        data[:] = 1.0
        coo = coo_matrix((data, (rows, cols)), shape=(num_rows, num_cols))
        arr = coo.toarray()
        dup_inds = np.nonzero(arr > 1.)

        # get a set of indices that sorts into col major order
        srtidxs = np.lexsort((rows, cols))

        #data = data[srtidxs]
        rows = rows[srtidxs]
        cols = cols[srtidxs]

        # now sort these back into ascending order (our original stacked order)
        # so in _update_submat() we can just extract the individual index
        # arrays that will map each block into the combined data array.
        revidxs = np.argsort(srtidxs)

        metadata = self._metadata
        for key, (ind1, ind2, idxs, jac_type, factor) in iteritems(metadata):
            if idxs is None:
                metadata[key] = (revidxs[ind1:ind2], jac_type, factor)
            else:
                # apply the reverse index to each part of revidxs so that
                # we can avoid copying the index array during updates.
                metadata[key] = (revidxs[ind1:ind2][np.argsort(idxs)],
                                 jac_type, factor)

        # data array for the CSC will be the same as for the COO since
        # it was already in sorted order.
        coo = coo_matrix((data, (rows, cols)), shape=(num_rows, num_cols))
        coo_data_size = coo.data.size
        self._matrix = coo.tocsc()

        # make sure data size is the same between coo and csc, else indexing is
        # messed up
        if False: #coo_data_size != self._matrix.data.size:
            # check to see if we have any duplicated src_indices
            seen = set()
            dup_src_inds = []
            for key, (_, _, src_indices, _, _) in iteritems(self._submats):
                if src_indices is not None and key[1] not in seen:
                    seen.add(key[1])
                    counts = Counter(src_indices)
                    dups = []
                    for entry, count in counts.most_common():
                        if count > 1:
                            dups.append(entry)
                        else:
                            break
                    if dups:
                        dup_src_inds.append((key[1], dups))
                        print(key, src_indices)
            if dup_src_inds:
                extra = (" Some duplicate entries, if not all, are due to the following repeated src indices: %s" %
                         ', '.join(str(d) for d in dup_src_inds))
            else:
                extra = ''
            dups = sorted(_get_dup_partials(rows, cols, in_ranges, out_ranges).items())
            raise ValueError("CSC matrix data contains the following duplicate row/col entries: "
                             "%s\nThis would break internal indexing.%s" % (dups, extra))
