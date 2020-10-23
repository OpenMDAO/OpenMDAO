
import numpy as np

from openmdao.utils.general_utils import _is_slicer_op
from openmdao.core.constants import INT_DTYPE

class _Indexer(object):
    def __init__(self, idxs):
        self.slice = self.arr = self.src_shape = None
        self.final_slice = self.final_arr = None
        self.min = self.max = self._shape = None

        if _is_slicer_op(idxs):
            self.slice = idxs
            if isinstance(idxs, tuple):
                entries = idxs
            else:  # plain slice
                entries = [idxs]

            dims = []
            for entry in entries:
                if isinstance(entry, slice):
                    if entry.start is None or entry.start is Ellipsis or entry.start < 0:
                        break
                    if entry.stop is None or entry.stop is Ellipsis or entry.stop < 0:
                        break
                    span = entry.stop - entry.start
                    if span > 0:
                        dims.append(span)
                        if entry.step != 1:
                            dims[-1] = 1 + (dims[-1] - 1) // entry.step
                    else:
                        dims.append(0)
                elif np.isscalar(entry):
                    if entry < 0:
                        break
                    dims.append(1)
                else:  # index array
                    dims.append(entry.size)
            else:  # slice has no shape dependent entries
                self.final_slice = self.slice
                self._shape = tuple(dims)
        else:
            self.arr = np.atleast_1d(idxs)
            if np.all(self.arr >= 0):
                self.final_arr = self.arr
                self._shape = self.arr.shape
                self.slice = self.final_slice = self._to_slice(self.arr)

    def __repr__(self):
        if self.slice is not None:
            return f"indexer({self.slice})"
        return f"indexer({self.arr})"

    def __call__(self, final=False):
        if final:
            if self.final_slice is not None:
                return self.final_slice
            if self.final_arr is not None:
                return self.final_arr
            else:
                raise RuntimeError(f"{self} source does not have a known src_shape.")
        else:
            if self.slice is not None:
                return self.slice
            return self.arr

    def _to_slice(self, arr):
        # try to convert array to slice.  Return None if not possible.
        if arr.ndim == 1 and arr.size > 1:  # see if 1D array will convert to slice
            span = arr[1] - arr[0]
            if np.all((arr[1:] - arr[:-1]) == span):
                # array is increasing with constant span
                return slice(arr[0], arr[-1] + 1, span)

    def as_flat(self):
        # return slice or index array for use in a flat array
        pass

    def as_array(self, strict=False):
        pass

    def as_slice(self, strict=False):
        pass

    def size(self):
        if self._shape is not None:
            return np.product(self._shape)
        raise RuntimeError(f"{self} does not have a known src_shape so can't compute its size.")

    def shape(self):
        # return shape of the indices themselves
        if self._shape is not None:
            return self._shape
        raise RuntimeError(f"{self} does not have a known src_shape so can't compute its size.")

    def set_src_shape(self, src_shape):
        self.final_slice = self.final_arr = None
        self.src_shape = src_shape
        # compute finals

    def is_contiguous(self):
        pass

    def ndims(self):
        pass

    def apply(self, parent):
        # apply these indices to parent indices
        pass
    

class IndexMaker(object):
    def __getitem__(self, idx):
        return _Indexer(idx)

    def __call__(self, idx):
        return _Indexer(idx)


indexer = IndexMaker()
