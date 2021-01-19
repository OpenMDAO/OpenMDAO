
import numpy as np
from copy import deepcopy

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

    def set_src_shape(self, shape):
        self._src_shape = shape

    def as_slice(self):
        raise NotImplementedError("No implementation of as_slice found.")

    def as_array(self):
        raise NotImplementedError("No implementation of as_array found.")


class IntIndexer(Indexer):
    def __init__(self, idx):
        super().__init__()
        self._idx = idx
        if idx >= 0:
            self._shaped_idx = idx
        else:
            self._shaped_idx = None

    def __call__(self):
        return self._idx

    def shape(self):
        return 1

    def set_src_shape(self, shape):
        super().set_src_shape(shape)
        if self._idx < 0:
            if np.is_scalar(shape):
                self._shaped_idx = self._idx + shape
            else:
                self._shaped_idx = self._idx + shape[0]

    def as_slice(self):
        if self._idx == -1:
            if self._shaped_idx is None:
                raise RuntimeError(f"Can't express index {self._idx} as a slice because source shape is unknown.")
            return slice(self._shaped_idx, self._shaped_idx + 1)
        return slice(self._idx, self._idx + 1)

    def as_array(self):
        return np.array([self._idx])


class SliceIndexer(Indexer):
    def __init__(self, slc):
        super().__init__()
        self._slice = deepcopy(slc)
        if (slc.start is not None and slc.start < 0) or slc.stop is None or slc.stop < 0:  # need shape
            self._shaped_slice = None
        else:
            self._shaped_slice = self._slice

    def __call__(self):
        return self._slice

    def shape(self):
        if self._shaped_slice is None:
            raise RuntimeError(f"Can't get shape of {self._slice} because source shape is unknown.")
        return ((self._shaped_slice.stop - self._shaped_slice.start) // self._shaped_slice.step)

    def set_src_shape(self, shape):
        super().set_src_shape(shape)
        if np.is_scalar(shape):
            length = shape
        elif len(shape) == 1:
            length = shape[0]
        else:
            raise RuntimeError(f"shape {shape} passed to set_src_shape is not have dimension of 1")

        slc = self._slice

        if slc.start < 0 or slc.stop is None or slc.stop < 0:  # need shape
            start = 0 if slc.start is None else slc.start
            if start < 0:
                start = length - start
            stop = length if slc.stop is None else slc.stop
            if stop < 0:
                stop = length - stop
            step = 1 if slc.step is None else slc.step

            self._shaped_slice = slice(start, stop, step)

    def as_slice(self):
        return self._slice

    def as_array(self):
        if self._shaped_slice is None:
            raise RuntimeError(f"Can't convert {self._slice} to array because source shape is unknown.")

        return np.arange(self._shaped_slice.start, 
                         self._shaped_slice.stop, 
                         self._shaped_slice.step)


class ArrayIndexer(Indexer):
    def __init__(self, arr):
        super().__init__()
        self.arr = np.asarray(arr)
        self._slice = None

    def __call__(self):
        return self.arr

    def shape(self):
        return self.arr.shape

    def as_slice(self):
        if self._slice is None:
            self._slice = array2slice(self.arr)

        if self._slice is None:
            raise ValueError(f"array index cannot be converted to a slice.")

        return self._slice

    def as_array(self):
        return self.arr


class MultiIndexer(Indexer):
    def __init__(self, tup):
        # if original idxs is a slice or fancy index (tuple of index arrays/slices),
        #    self.tup is not None
        # if original idxs is an array, self.arr is not None
        self.tup = self.arr = self._src_shape = None
        self.shaped_tup = self.shaped_arr = None
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
                self.shaped_tup = self.tup
                self._shape = tuple(dims)
                self._shape_dependent = False
        else:
            self.arr = np.atleast_1d(idxs)
            self._shape = self.arr.shape
            if np.min(self.arr) >= 0:
                self._shape_dependent = False
                self.shaped_arr = self.arr
                self.shaped_tup = array2slice(self.arr)

    def __repr__(self):
        if self.tup is not None:
            if len(self.tup) == 1:
                return f"indexer({self.tup[0]})"
            else:
                return f"indexer({self.tup})"
        return f"indexer({self.arr})"

    # def __call__(self, shaped=False):
    #     if self.shaped_tup is not None:
    #         return self.shaped_tup

    #     if shaped:
    #         if self.shaped_arr is not None:
    #             return self.shaped_arr
    #         self.src_shape()  # raise error
    #     else:
    #         if self.tup is not None:
    #             if len(self.tup) == 1:
    #                 return self.tup[0]
    #             return self.tup
    #         return self.arr

    def _to_shaped_arr(self):
        if self.shaped_arr is not None:
            return self.shaped_arr

        if self.arr is not None:  # array has negative entries
            # TODO: make this more efficient later...
            return np.arange(self.src_shape(), dtype=INT_DTYPE)[self.arr]

        # must be a slice
        return np.arange(self.src_shape(), dtype=INT_DTYPE)[self.tup]

    def as_array(self, flat=False, shaped=False):
        if shaped or flat:
            arr = self._to_shaped_arr()
            if flat:
                return arr.ravel()
            return arr
        elif self.arr is not None:
            return self.arr

        # we must be a slice or fancy index
        shape = self.shape()  # make sure we have shape
        idx = np.arange(shape, dtype=INT_DTYPE)[self.shaped_tup]

        if flat:
            return idx.ravel()
        return idx

    def as_slice(self, flat=False, shaped=False):
        if shaped or (flat and self.ndim() > 1):
            self.shape()
            if self.shaped_tup is None:
                raise RuntimeError(f"Indexer {self} is not convertable to a flat slice.")
            return self._to_flat_slice(self.shaped_tup)

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
        self.shaped_tup = self.shaped_arr = None
        self._src_shape = shape
        # compute shapeds

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
    def _get_indexer(self, idx):
        if isinstance(idx, int):
            return IntIndexer(idx)
        if isinstance(idx, slice):
            return SliceIndexer(idx)
        if _is_slicer_op(idx):
            return MultiIndexer(idx)
        return ArrayIndexer(idx)

    def __getitem__(self, idx):
        return self._get_indexer(idx)

    def __call__(self, idx):
        return self._get_indexer(idx)


indexer = IndexMaker()
