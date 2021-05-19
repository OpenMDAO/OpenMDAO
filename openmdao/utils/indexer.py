
import numpy as np
from copy import deepcopy

from openmdao.utils.general_utils import _is_slicer_op
from openmdao.core.constants import INT_DTYPE

def array2slice(arr):
    """
    Try to convert an array to slice.

    Conversion is only attempted for a 1D array.

    Parameters
    ----------
    arr : ndarray
        The array to be represented as a slice.
    
    Returns
    -------
    slice or None
        If slice conversion is possible, return the slice, else return None
    """
    if arr.ndim == 1 and arr.size > 1:  # see if 1D array will convert to slice
        span = arr[1] - arr[0]
        if np.all((arr[1:] - arr[:-1]) == span):
            if span > 0:
                # array is increasing with constant span
                return slice(arr[0], arr[-1] + 1, span)
            elif span < 0:
                 # array is decreasing with constant span
                return slice(arr[0], arr[-1] - 1, span)
               


class Indexer(object):
    def set_src_shape(self, shape):
        raise NotImplementedError("No implementation of set_src_shape found.")

    def __call__(self):
        raise NotImplementedError("No implementation of __call__ found.")

    def as_slice(self):
        raise NotImplementedError("No implementation of as_slice found.")

    def as_array(self):
        raise NotImplementedError("No implementation of as_array found.")

    def shape(self):
        raise NotImplementedError("No implementation of shape found.")

    def shaped(self):
        raise NotImplementedError("No implementation of shaped found.")

    def flat(self):
        return self()


class IntIndexer(Indexer):
    def __init__(self, idx):
        self._idx = idx
        if idx >= 0:
            self._shaped_idx = idx
        else:
            self._shaped_idx = None

    def __call__(self):
        return self._idx

    def shaped(self):
        if self._shaped_idx is None:
            raise ValueError(f"IntIndex value has no source shape and index is {self._idx}.")
        return self._shaped_idx

    def shape(self):
        return 1

    def set_src_shape(self, shape):
        if self._idx < 0:
            if np.isscalar(shape):
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
        start, stop, step = slc.start, slc.stop, slc.step
        self._slice = deepcopy(slc)
        start, stop, step = self._slice.start, self._slice.stop, self._slice.step
        if start is None:
            start = 0
        if step is None:
            step = 1
        if start < 0 or stop is None or stop < 0:  # need shape
            self._shaped_slice = None
        else:
            self._shaped_slice = slice(start, stop, step)

    def __call__(self):
        return self._slice

    def shaped(self):
        if self._shaped_slice is None:
            raise RuntimeError(f"Can't get shape of {self._slice} because source shape is unknown.")
        return self._shaped_slice

    def shape(self):
        if self._shaped_slice is None:
            raise RuntimeError(f"Can't get shape of {self._slice} because source shape is unknown.")
        return abs((self._shaped_slice.stop - self._shaped_slice.start) // self._shaped_slice.step)

    def set_src_shape(self, shape):
        if np.isscalar(shape):
            length = shape
        elif len(shape) == 1:
            length = shape[0]
        else:
            raise RuntimeError(f"shape {shape} passed to set_src_shape does not have dimension of 1")

        slc = self._slice

        if slc.start is None or slc.start < 0 or slc.stop is None or slc.stop < 0:  # need shape
            start = 0 if slc.start is None else slc.start
            if start < 0:
                start = length + start
            stop = length if slc.stop is None else slc.stop
            if stop < 0:
                stop = length + stop
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
        self.arr = np.asarray(arr)
        self._slice = None
        if np.any(self.arr < 0):
            self._shaped_arr = None
        else:
            self._shaped_arr = self.arr
            self._slice = array2slice(self._shaped_arr)

    def __call__(self):
        if self._slice is None:
            return self.arr
        else:
            return self._slice
    
    def shaped(self):
        if self._slice is not None:
            return self._slice
        if self._shaped_arr is not None:
            return self._shaped_arr
        raise RuntimeError(f"Can't determine extent of array because source shape is not known.")

    def shape(self):
        return self.arr.shape

    def as_slice(self):
        if self._slice is None:
            raise ValueError(f"array index cannot be converted to a slice.")
        return self._slice

    def as_array(self):
        return self.arr

    def set_src_shape(self, shape):
        assert np.isscalar(shape) or len(shape) == 1
        size = shape if np.isscalar(shape) else shape[0]
        neg = self.arr < 0
        if np.any(neg):  # must have at least 1 negative index
            self._shaped_arr = self.arr.copy()
            self._shaped_arr[neg] += size
        else:
            self._shaped_arr = self.arr
        self._slice = array2slice(self._shaped_arr)


class MultiIndexer(Indexer):
    def __init__(self, tup):
        self._idx_list = [indexer(i) for i in tup]
        self._src_shape = None

    def __call__(self):
        return tuple(i() for i in self._idx_list)

    def shaped(self):
        return tuple(i.shaped() for i in self._idx_list)

    def shape(self):
        return tuple(i.shape() for i in self._idx_list)

    def as_slice(self):
        return tuple(i.as_slice() for i in self._idx_list)

    def as_array(self):
        # return as a flattened index array into a flat source
        if self._src_shape is None:
            raise RuntimeError(f"Can't determine extent of array because source shape is not known.")

        idxs = np.arange(np.product(self._src_shape), dtype=np.int32).reshape(self._src_shape)
        
        return idxs[self()].ravel()

    def set_src_shape(self, shape):
        assert len(shape) == len(self._idx_list)
        for i, s in zip(self._idx_list, shape):
            i.set_src_shape(s)
        self._src_shape = shape


class IndexMaker(object):
    def _get_indexer(self, idx):
        if isinstance(idx, int):
            return IntIndexer(idx)
        if isinstance(idx, slice):
            return SliceIndexer(idx)
        if isinstance(idx, tuple):
            return MultiIndexer(idx)
        return ArrayIndexer(idx)

    def __getitem__(self, idx):
        return self._get_indexer(idx)

    def __call__(self, idx):
        return self._get_indexer(idx)


indexer = IndexMaker()
