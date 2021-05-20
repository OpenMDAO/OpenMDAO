
import sys
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
    if arr.ndim == 1:
        if arr.size > 1:  # see if 1D array will convert to slice
            span = arr[1] - arr[0]
            if np.all((arr[1:] - arr[:-1]) == span):
                if span > 0:
                    # array is increasing with constant span
                    return slice(arr[0], arr[-1] + 1, span)
                elif span < 0:
                    # array is decreasing with constant span
                    return slice(arr[0], arr[-1] - 1, span)
        else:
            return slice(0, 0)



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
                raise RuntimeError(f"Can't express index {self._idx} as a slice because source "
                                   "shape is unknown.")
            return slice(self._shaped_idx, self._shaped_idx + 1)
        return slice(self._idx, self._idx + 1)

    def as_array(self):
        return np.array([self._idx])


class SliceIndexer(Indexer):
    def __init__(self, slc):
        self._slice = slc
        if (slc.start is not None and slc.start < 0) or slc.stop is None or slc.stop < 0:
            self._shaped_slice = None
        else:
            self._shaped_slice = slc

    def __call__(self):
        return self._slice

    def shaped(self):
        if self._shaped_slice is None:
            raise RuntimeError(f"Can't get shape of {self._slice} because source shape is unknown.")
        return self._shaped_slice

    def shape(self):
        if self._shaped_slice is None:
            raise RuntimeError(f"Can't get shape of {self._slice} because source shape is unknown.")

        # use maxsize here since _shaped_slice always has positive int start and stop
        return len(range(*self._shaped_slice.indices(sys.maxsize)))

    def set_src_shape(self, shape):
        if np.isscalar(shape):
            length = shape
        elif len(shape) == 1:
            length = shape[0]
        else:
            raise RuntimeError(f"shape {shape} passed to set_src_shape does not have dimension "
                               "of 1")

        self._shaped_slice = slice(*self._slice.indices(length))

    def as_slice(self):
        return self._slice

    def as_array(self):
        if self._shaped_slice is None:
            raise RuntimeError(f"Can't convert {self._slice} to array because source shape is "
                               "unknown.")

        # use maxsize here since _shaped_slice always has positive int start and stop
        return np.arange(*self._shaped_slice.indices(sys.maxsize), dtype=int)


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
        return self.arr.size

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
        self._idx_list = [indexer[i] for i in tup]
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

        idx = np.atleast_1d(idx)

        # if array is convertable to a slice, store it as a slice
        slc = array2slice(idx)
        if slc is None:
            return ArrayIndexer(idx)
        else:
            return SliceIndexer(slc)

    def __getitem__(self, idx):
        return self._get_indexer(idx)


indexer = IndexMaker()


# Since this is already user facing we'll leave it as is, and just use the output of
# __getitem__ to initialize our Indexer object that will be used internally.
class Slicer(object):
    """
    Helper class that can be used when a slice is needed for indexing.
    """

    def __getitem__(self, val):
        """
        Pass through indices or slice.

        Parameters
        ----------
        val : int or slice object or tuples of slice objects
            Indices or slice to return.

        Returns
        -------
        indices : int or slice object or tuples of slice objects
            Indices or slice to return.
        """
        return val


# instance of the Slicer class to be used by users
slicer = Slicer()
