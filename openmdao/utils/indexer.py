
import numpy as np

from openmdao.utils.general_utils import _is_slicer_op
from openmdao.core.constants import INT_DTYPE

class _Indexer(object):
    def __init__(self, idxs):
        # if original idx is a slice or fancy index (tuple of index arrays/slices), self.slice is not None
        # if original idx is an array, self.arr is not None
        self.slice = self.arr = self._src_shape = None
        self.final_slice = self.final_arr = None
        self.min = self.max = self._shape = None
        self._shape_dependent = True

        if _is_slicer_op(idxs):
            if isinstance(idxs, tuple):
                entries = idxs
            else:  # plain slice
                entries = (idxs,)
            self.slice = entries

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
                self._shape_dependent = False
        else:
            self.arr = np.atleast_1d(idxs)
            self._shape = self.arr.shape
            if np.all(self.arr >= 0):
                self._shape_dependent = False
                self.final_arr = self.arr
                self.final_slice = self._to_slice(self.arr)

    def __repr__(self):
        if self.slice is not None:
            if len(self.slice) == 1:
                return f"indexer({self.slice[0]})"
            else:
                return f"indexer({self.slice})"
        return f"indexer({self.arr})"

    def __call__(self, final=False):
        if self.final_slice is not None:
            return self.final_slice

        if final:
            if self.final_arr is not None:
                return self.final_arr
            self.src_shape()  # raise error
        else:
            if self.slice is not None:
                if len(self.slice) == 1:
                    return self.slice[0]
                return self.slice
            return self.arr

    def _to_slice(self, arr):
        # try to convert array to slice.  Return None if not possible.
        # array is assumed to have no negative entries.
        if arr.ndim == 1 and arr.size > 1:  # see if 1D array will convert to slice
            span = arr[1] - arr[0]
            if np.all((arr[1:] - arr[:-1]) == span):
                # array is increasing with constant span
                return slice(arr[0], arr[-1] + 1, span)

    def _to_final_arr(self):
        if self.final_arr is not None:
            return self.final_arr

        if self.arr is not None:  # array has negative entries
            # TODO: make this more efficient later...
            return np.arange(self.src_shape(), dtype=INT_DTYPE)[self.arr]

        # must be a slice
        return np.arange(self.src_shape(), dtype=INT_DTYPE)[self.slice]

    def as_array(self, flat=False, final=False):
        if final or flat:
            arr = self._to_final_arr()
            if flat:
                return arr.ravel()
            return arr
        elif self.arr is not None:
            return self.arr

        # we must be a slice or fancy index
        shape = self.shape()  # make sure we have shape
        idx = np.arange(shape, dtype=INT_DTYPE)[self.final_slice]

        if flat:
            return idx.ravel()
        return idx

    def as_slice(self, flat=False, final=False):
        if final or (flat and self.ndims() > 1):
            self.shape()
            if self.final_slice is None:
                raise RuntimeError(f"Indexer {self} is not convertable to a flat slice.")
            return self._to_flat_slice(self.final_slice)

        if self.slice is None:
            raise RuntimeError(f"Indexer {self} is not convertable to a slice.")
        
        return self.slice

    def size(self):
        return np.product(self.shape())

    def shape(self):
        # return shape of the indices themselves
        if self._shape is not None:
            return self._shape
        raise RuntimeError(f"{self} does not have a known src_shape so can't compute its shape.")

    def src_shape(self):
        if self._src_shape is not None:
            return self._src_shape
        raise RuntimeError(f"{self} does not have a known src_shape.")

    def set_src_shape(self, src_shape):
        self.final_slice = self.final_arr = None
        self._src_shape = src_shape
        # compute finals

    def is_contiguous(self):
        if self.slice is not None:
            return self.slice.step in (1, None)

    def ndims(self):
        if self.slice is not None:
            return len(self.slice)

    def apply(self, parent):
        # apply these indices to parent indices
        pass
    

class IndexMaker(object):
    def __getitem__(self, idx):
        return _Indexer(idx)

    def __call__(self, idx):
        return _Indexer(idx)


indexer = IndexMaker()
