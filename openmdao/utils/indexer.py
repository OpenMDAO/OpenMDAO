
import numpy as np

from openmdao.utils.general_utils import _is_slicer_op
from openmdao.core.constants import INT_DTYPE

def array2slice(arr):
    """
    Try to convert an array to slice.  Return None if not possible.

    Conversion is only attempted for a 1D array.

    Parameters
    ----------
    arr : ndarray
        The array to be represented as a slice.
    """
    if arr.ndim == 1 and arr.size > 1:  # see if 1D array will convert to slice
        span = arr[1] - arr[0]
        if span > 0 and np.all((arr[1:] - arr[:-1]) == span):
            # array is increasing with constant span
            return slice(arr[0], arr[-1] + 1, span)


class Indexer(object):
    def __init__(self):
        self._src_shape = None
        self._shape_dependent = True
        self.min = self.max = self._shape = None
        self._final_slice = None
        self._final_arr = None

    def as_array(self, final=False):
        if final:
            if self._final_arr is None:
                self._final_arr = self.as_final_arr()
            return self._final_arr
        return


class SliceIndexer(Indexer):
    def __init__(self, slc):
        self._slice = slc

    def as_slice(self):
        if self._final_slice is
        return self._final_slice

    def final_array(self):
        pass


class ArrayIndexer(Indexer):
    def __init__(self, arr):
        self.arr = arr

    def final_array(self):
        pass


    def as_slice(self):
        if self._final_slice is None:
            self._final_slice = array2slice(self.final_array())

        if self._final_slice is None:
            raise ValueError(f"array index cannot be converted to a slice.")

        return self._final_slice


class MultiIndexer(Indexer):
    def __init__(self, tup):
        # if original idxs is a slice or fancy index (tuple of index arrays/slices),
        #    self.tup is not None
        # if original idxs is an array, self.arr is not None
        self.tup = self.arr = self._src_shape = None
        self.final_tup = self.final_arr = None
        self.min = self.max = self._shape = None
        self._shape_dependent = True

        if _is_slicer_op(idxs):
            if isinstance(idxs, tuple):
                entries = idxs
            else:  # plain slice
                entries = (idxs,)
            self.tup = entries

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
                self.final_tup = self.tup
                self._shape = tuple(dims)
                self._shape_dependent = False
        else:
            self.arr = np.atleast_1d(idxs)
            self._shape = self.arr.shape
            if np.min(self.arr) >= 0:
                self._shape_dependent = False
                self.final_arr = self.arr
                self.final_tup = array2slice(self.arr)

    def __repr__(self):
        if self.tup is not None:
            if len(self.tup) == 1:
                return f"indexer({self.tup[0]})"
            else:
                return f"indexer({self.tup})"
        return f"indexer({self.arr})"

    def __call__(self, final=False):
        if self.final_tup is not None:
            return self.final_tup

        if final:
            if self.final_arr is not None:
                return self.final_arr
            self.src_shape()  # raise error
        else:
            if self.tup is not None:
                if len(self.tup) == 1:
                    return self.tup[0]
                return self.tup
            return self.arr

    def _to_final_arr(self):
        if self.final_arr is not None:
            return self.final_arr

        if self.arr is not None:  # array has negative entries
            # TODO: make this more efficient later...
            return np.arange(self.src_shape(), dtype=INT_DTYPE)[self.arr]

        # must be a slice
        return np.arange(self.src_shape(), dtype=INT_DTYPE)[self.tup]

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
        idx = np.arange(shape, dtype=INT_DTYPE)[self.final_tup]

        if flat:
            return idx.ravel()
        return idx

    def as_slice(self, flat=False, final=False):
        if final or (flat and self.ndim() > 1):
            self.shape()
            if self.final_tup is None:
                raise RuntimeError(f"Indexer {self} is not convertable to a flat slice.")
            return self._to_flat_slice(self.final_tup)

        if self.tup is None:
            raise RuntimeError(f"Indexer {self} is not convertable to a slice.")

        return self.tup

    def size(self):
        return np.product(self.shape())

    def shape(self):
        # return shape of the indices themselves
        if self._shape is not None:
            return self._shape
        raise RuntimeError(f"{self} does not have a known src_shape so can't compute its shape.")

    @property
    def src_shape(self):
        if self._src_shape is not None:
            return self._src_shape
        raise RuntimeError(f"{self} does not have a known src_shape.")

    @src_shape.setter
    def src_shape(self, shape):
        self.final_tup = self.final_arr = None
        self._src_shape = shape
        # compute finals

    def is_contiguous(self):
        if self.tup is not None:
            return self.tup.step in (1, None)

    def ndim(self):
        if self.tup is not None:
            return len(self.tup)
        return self.arr.ndim

    def apply(self, parent):
        # apply these indices to parent indices
        pass


class IndexMaker(object):
    def __getitem__(self, idx):
        return Indexer(idx)

    def __call__(self, idx):
        return Indexer(idx)


indexer = IndexMaker()
