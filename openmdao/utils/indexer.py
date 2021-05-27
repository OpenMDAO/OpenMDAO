"""
Classes that handle array indexing.
"""

import sys
import numpy as np
from copy import deepcopy
from numbers import Integral

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
    if arr.ndim == 1 and arr.dtype.kind in ('i', 'u'):
        if arr.size > 1:  # see if 1D array will convert to slice
            if arr[0] >= 0 and arr[1] >= 0:
                span = arr[1] - arr[0]
            else:
                return None
            if np.all((arr[1:] - arr[:-1]) == span):
                if span > 0:
                    # array is increasing with constant span
                    return slice(arr[0], arr[-1] + 1, span)
                elif span < 0:
                    # array is decreasing with constant span
                    return slice(arr[0], arr[-1] - 1, span)
        elif arr.size == 1:
            if arr[0] >= 0:
                return slice(arr[0], arr[0] + 1)
            return slice(arr[0], arr[0] - 1, -1)
        else:
            return slice(0, 0)


class Indexer(object):
    """
    Abstract indexing class.

    Attributes
    ----------
    _src_shape : tuple or None
        Shape of the 'source'.  Used to determine actual index or slice values when indices are
        negative or slice contains negative start or stop values or ':' or '...'.
    _shaped_inst : Indexer or None
        Cached shaped_instance if we've computed it before.
    """

    def __init__(self):
        """
        Initialize attributes.
        """
        self._src_shape = None
        self._shaped_inst = None

    def __len__(self):
        """
        Return the length of the flattened indices.

        Returns
        -------
        int
            Length of flattened indices.
        """
        return self.size()

    def __call__(self):
        """
        Return the indices in their most efficient form.

        For example, if the original indices were an index array that is convertable to a slice,
        then a slice would be returned.

        This could be either an int, a slice, an index array, or a multidimensional 'fancy' index.
        """
        raise NotImplementedError("No implementation of '__call__' found.")

    def size(self):
        """
        Return the size of the flattened indices.

        Returns
        -------
        int
            Size of flattened indices.
        """
        return np.product(self.shape(), dtype=int)

    def _check_ind_type(self, ind, types):
        if not isinstance(ind, types):
            raise TypeError(f"Can't create {type(self).__name__} using this "
                            f"kind of index: {ind}.")

    def flat(self, copy=False):
        """
        Return index array or slice into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.
        """
        raise NotImplementedError("No implementation of 'flat' found.")

    def shape(self):
        """
        Return the shape of the result if the indices were applied to a source array.

        Returns
        -------
        tuple
            The shape of the indices.
        """
        s = self.shaped_instance()
        if s is None:
            raise RuntimeError(f"Can't get shape of {self} because source shape "
                               "is unknown.")
        return s.shape()

    def shaped_instance(self):
        """
        Return a 'shaped' version of this Indexer type.

        This should be overridden for all non-shaped derived classes.

        Returns
        -------
        Indexer
            The 'shaped' Indexer type.  'shaped' Indexers know the extent of the array that
            they are indexing into, or they don't care what the extent is because they don't
            contain negative indices, ':', or '...'.
        """
        return self

    def shaped(self):
        """
        Return a version of the indices that index into a flattened array.

        Could be either a slice or an index array.

        Returns
        -------
        slice or ndarray or int
            Version of these indices that index into a flattened array.
        """
        s = self.shaped_instance()
        if s is None:
            raise ValueError(f"Can't get shaped version of {self} because it has "
                             "no source shape.")
        return s()

    def shaped_array(self, copy=False):
        """
        Return an index array version of the indices that index into a flattened array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.

        Returns
        -------
        ndarray
            Version of these indices that index into a flattened array.
        """
        s = self.shaped_instance()
        if s is None:
            raise ValueError(f"Can't get shaped array of {self} because it has "
                             "no source shape.")
        return s.as_array(copy=copy)

    def shaped_slice(self):
        """
        Return a slice version (if possible) of the indices that index into a flattened array.

        Raises an exception if a slice can't be returned.

        Returns
        -------
        slice
            Slice ersion of these indices that index into a flattened array.
        """
        s = self.shaped_instance()
        if s is None:
            raise ValueError(f"Can't get shaped slice of {self} because it has "
                             "no source shape.")
        return s.as_slice()

    def set_src_shape(self, shape):
        """
        Set the shape of the 'source' array .

        Parameters
        ----------
        shape : tuple or int
            The shape of the 'source' array.

        Returns
        -------
        Indexer
            Self is returned to allow chaining.
        """
        if isinstance(shape, Integral):
            shape = (shape,)
        self._src_shape = shape
        self._shaped_inst = None
        return self

    def to_json(self):
        """
        Return a JSON serializable version of self.
        """
        raise NotImplementedError("No implementation of 'to_json' found.")


class ShapedIntIndexer(Indexer):
    """
    Int indexing class.

    Attributes
    ----------
    _idx : int
        The integer index.
    """

    def __init__(self, idx):
        """
        Initialize attributes.

        Parameters
        ----------
        idx : int
            The index.
        """
        super().__init__()
        self._check_ind_type(idx, Integral)
        self._idx = idx

    def __call__(self):
        """
        Return this index.

        Returns
        -------
        int
            This index.
        """
        return self._idx

    def __str__(self):
        """
        Return string representation.

        Returns
        -------
        str
            String representation.
        """
        return f"{self._idx}"

    def size(self):
        """
        Return the size of the flattened indices.

        Returns
        -------
        int
            Size of flattened indices.
        """
        return 1

    def shape(self):
        """
        Return the shape of the index ().

        Returns
        -------
        tuple
            The shape of the index.
        """
        return ()

    def as_slice(self):
        """
        Return a slice into a flat array.

        Returns
        -------
        slice
            The slice into a flat array.
        """
        return slice(self._idx, self._idx + 1)

    def as_array(self, copy=False):
        """
        Return an index array into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.

        Returns
        -------
        ndarray
            The index array into a flat array.
        """
        return np.array([self._idx])

    def flat(self, copy=False):
        """
        Return index array into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.

        Returns
        -------
        ndarray
            The index into a flat array.
        """
        return np.array([self._idx])

    def to_json(self):
        """
        Return a JSON serializable version of self.

        Returns
        -------
        int
            int version of self.
        """
        return self._idx


class IntIndexer(ShapedIntIndexer):
    """
    Int indexing class that may or may not be 'shaped'.
    """

    def shaped_instance(self):
        """
        Return a 'shaped' version of this Indexer type.

        Returns
        -------
        ShapedIntIndexer or None
            Will return a ShapedIntIndexer if possible, else None.
        """
        if self._shaped_inst is not None:
            return self._shaped_inst

        if self._idx < 0:
            if self._src_shape is None:
                return None
            else:
                self._shaped_inst = ShapedIntIndexer(self._idx + self._src_shape[0])
            return self._shaped_inst

        return ShapedIntIndexer(self._idx)

    def as_slice(self):
        """
        Return this index as a slice.

        Returns
        -------
        slice
            A slice that represents this index.
        """
        return self.shaped_slice()


class ShapedSliceIndexer(Indexer):
    """
    Abstract slice class that is 'shaped'.

    Attributes
    ----------
    _slice : slice
        The wrapped slice object.
    """

    def __init__(self, slc):
        """
        Initialize attributes.

        Parameters
        ----------
        slc : slice
            The slice.
        """
        super().__init__()
        self._check_ind_type(slc, slice)
        self._slice = slc

    def __call__(self):
        """
        Return this slice.

        Returns
        -------
        slice
            This slice.
        """
        return self._slice

    def __str__(self):
        """
        Return string representation.

        Returns
        -------
        str
            String representation.
        """
        return f"{self._slice}"

    def as_slice(self):
        """
        Return a slice into a flat array.

        Returns
        -------
        slice
            The slice into a flat array.
        """
        return self._slice

    def as_array(self, copy=False):
        """
        Return an index array into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.

        Returns
        -------
        ndarray
            The index array into a flat array.
        """
        # use maxsize here since _shaped_slice always has positive int start and stop
        return np.arange(*self._slice.indices(sys.maxsize), dtype=int)

    def flat(self, copy=False):
        """
        Return a slice into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.

        Returns
        -------
        slice
            The slice into a flat array.
        """
        # slices are immutable, so ignore copy arg
        return self._slice

    def shape(self):
        """
        Return the shape of the indices.

        Returns
        -------
        int
            The shape of the indices.
        """
        # use maxsize here since shaped slice always has positive int start and stop
        return len(range(*self._slice.indices(sys.maxsize)))

    def to_json(self):
        """
        Return a JSON serializable version of self.

        Returns
        -------
        list of int or int
            list or int version of self.
        """
        return self.as_array().tolist()


class SliceIndexer(ShapedSliceIndexer):
    """
    Abstract slice class that may or may not be 'shaped'.
    """

    def shaped_instance(self):
        """
        Return a 'shaped' version of this Indexer type.

        Returns
        -------
        ShapedSliceIndexer or None
            Will return a ShapedSliceIndexer if possible, else None.
        """
        if self._shaped_inst is not None:
            return self._shaped_inst

        slc = self._slice
        if (slc.start is not None and slc.start < 0) or slc.stop is None or slc.stop < 0:
            if self._src_shape is None:
                return None
            else:
                self._shaped_inst = \
                    ShapedSliceIndexer(slice(*self._slice.indices(self._src_shape[0])))
                return self._shaped_inst

        return ShapedSliceIndexer(slc)

    def shape(self):
        """
        Return the shape of the indices.

        Returns
        -------
        int
            The shape of the indices.
        """
        return Indexer.shape(self)

    def as_array(self, copy=False):
        """
        Return an index array into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.

        Returns
        -------
        ndarray
            The index array into a flat array.
        """
        return self.shaped_array(copy=copy)


class ShapedArrayIndexer(Indexer):
    """
    Abstract index array class that is 'shaped'.

    Attributes
    ----------
    _arr : ndarray
        The wrapped index array object.
    _convert : bool
        If True, conversion of arrays to slices and ellipses to multi-indexers is allowed.
    """

    def __init__(self, arr, convert=True):
        """
        Initialize attributes.

        Parameters
        ----------
        arr : ndarray
            The index array.
        convert : bool
            If True, conversion of arrays to slices and ellipses to multi-indexers is allowed.
        """
        super().__init__()

        ndarr = np.asarray(arr)

        # check type
        if ndarr.dtype.kind not in ('i', 'u'):
            raise TypeError(f"Can't create an index array using the following indices of "
                            f"non-integral type: {arr}.")

        self._arr = ndarr
        self._convert = convert

    def __call__(self):
        """
        Return this index array.

        Returns
        -------
        int
            This index array.
        """
        return self._arr

    def __str__(self):
        """
        Return string representation.

        Returns
        -------
        str
            String representation.
        """
        return f"{self._arr}"

    def shape(self):
        """
        Return the shape of the indices.

        Returns
        -------
        tuple
            The shape of the indices.
        """
        return self._arr.shape

    def as_array(self, copy=False):
        """
        Return an index array into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.

        Returns
        -------
        ndarray
            The index array into a flat array.
        """
        if copy:
            return self._arr.copy()
        return self._arr

    def as_slice(self):
        """
        Return a slice into a flat array.

        This always fails because if it were possible, we would have already replaced this
        array indexer with a slice indexer.
        """
        raise ValueError(f"Can't convert {self} to a slice.")

    def flat(self, copy=False):
        """
        Return an index array into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.

        Returns
        -------
        ndarray
            The index into a flat array.
        """
        if copy:
            return self._arr.copy()
        return self._arr

    def to_json(self):
        """
        Return a JSON serializable version of self.

        Returns
        -------
        list of int or int
            list or int version of self.
        """
        return self._arr.tolist()


class ArrayIndexer(ShapedArrayIndexer):
    """
    Abstract index array class that may or may not be 'shaped'.
    """

    def shaped_instance(self):
        """
        Return a 'shaped' version of this Indexer type.

        Returns
        -------
        ShapedArrayIndexer or None
            Will return a ShapedArrayIndexer if possible, else None.
        """
        if self._shaped_inst is not None:
            return self._shaped_inst

        negs = self._arr < 0
        if np.any(negs):
            if self._src_shape is None:
                return None
            else:
                sharr = self._arr.copy()
                sharr[negs] += self._src_shape[0]
        else:
            sharr = self._arr

        if self._convert:
            slc = array2slice(sharr)
            if slc is not None:
                self._shaped_inst = ShapedSliceIndexer(slc)
            else:
                self._shaped_inst = ShapedArrayIndexer(sharr)
        else:
            self._shaped_inst = ShapedArrayIndexer(sharr)

        return self._shaped_inst

    def as_slice(self):
        """
        Return a slice into a flat array.

        Returns
        -------
        slice
            The slice into a flat array.
        """
        return self.shaped_slice()


class ShapedMultiIndexer(Indexer):
    """
    Abstract multi indexer class that is 'shaped'.

    Attributes
    ----------
    _tup : tuple
        The wrapped tuple of indices/slices.
    _idx_list : list
        List of Indexers.
    """

    def __init__(self, tup):
        """
        Initialize attributes.

        Parameters
        ----------
        tup : tuple
            Tuple of indices/slices.
        """
        super().__init__()
        self._tup = tup
        self._idx_list = [indexer(i, convert=False) for i in tup]

    def __call__(self):
        """
        Return this mltidimensional index.

        Returns
        -------
        int
            This multidimensional index.
        """
        return tuple(i() for i in self._idx_list)

    def __str__(self):
        """
        Return string representation.

        Returns
        -------
        str
            String representation.
        """
        return f"{self._tup}"

    def shape(self):
        """
        Return the shape of the indices.

        Returns
        -------
        tuple
            The shape of the indices.
        """
        lens = []
        seen_num = False
        for i in self._idx_list:
            if isinstance(i, ShapedSliceIndexer):
                lens.append(len(i))
            elif isinstance(i, ShapedArrayIndexer) and not seen_num:
                # only first array idx counts toward shape
                lens.append(len(i))
                seen_num = True
            # int indexers don't count toward shape (scalar array has shape ())

        return tuple(lens)

    def as_slice(self):
        """
        Return a tuple of slices into a multidimensional array.

        Returns
        -------
        tuple of slices
            The slice into a multidimensional array.
        """
        return tuple(i.as_slice() for i in self._idx_list)

    def as_array(self, copy=False):
        """
        Return an index array into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.

        Returns
        -------
        ndarray
            The index array into a flat array.
        """
        # return as a flattened index array into a flat source
        if self._src_shape is None:
            raise ValueError(f"Can't determine extent of array because source shape is not known.")

        idxs = np.arange(np.product(self._src_shape), dtype=np.int32).reshape(self._src_shape)

        return idxs[self()].ravel()

    def flat(self, copy=False):
        """
        Return an index array into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.

        Returns
        -------
        ndarray
            An index array into a flat array.
        """
        return self.as_array(copy=copy)

    def set_src_shape(self, shape):
        """
        Set the shape of the 'source' array .

        Parameters
        ----------
        shape : tuple or int
            The shape of the 'source' array.

        Returns
        -------
        Indexer
            Self is returned to allow chaining.
        """
        super().set_src_shape(shape)
        for i, s in zip(self._idx_list, shape):
            i.set_src_shape(s)

        return self

    def to_json(self):
        """
        Return a JSON serializable version of self.

        Returns
        -------
        list of int or int
            list or int version of self.
        """
        return self.as_array().tolist()


class MultiIndexer(ShapedMultiIndexer):
    """
    Abstract multi indexer class that may or may not be 'shaped'.
    """

    def shaped_instance(self):
        """
        Return a 'shaped' version of this Indexer type.

        Returns
        -------
        ShapedMultiIndexer or None
            Will return a ShapedMultiIndexer if possible, else None.
        """
        if self._shaped_inst is not None:
            return self._shaped_inst

        try:
            self._shaped_inst = ShapedMultiIndexer(tuple(idxer.shaped_instance()()
                                                         for idxer in self._idx_list))
        except Exception:
            self._shaped_inst = None

        return self._shaped_inst


class EllipsisIndexer(Indexer):
    """
    Abstract multi indexer class that is 'shaped'.

    Attributes
    ----------
    _tup : tuple
        The wrapped tuple of indices/slices (it contains an ellipsis).
    """

    def __init__(self, tup):
        """
        Initialize attributes.

        Parameters
        ----------
        tup : tuple
            Tuple of indices/slices.
        """
        super().__init__()
        self._tup = tup

    def __call__(self):
        """
        Return the 'default' form of the indices.

        Returns
        -------
        tuple
            Tuple of indices and/or slices.
        """
        return self._tup

    def __str__(self):
        """
        Return string representation.

        Returns
        -------
        str
            String representation.
        """
        return f"{self._tup}"

    def shaped_instance(self):
        """
        Return a 'shaped' version of this Indexer type.

        Returns
        -------
        A shaped Indexer or None
            Will return some kind of shaped Indexer if possible, else None.
        """
        if self._shaped_inst is not None:
            return self._shaped_inst

        if self._src_shape is None:
            return None

        lst = [None] * len(self._src_shape)
        # number of full slice dimensions
        nfull = len(self._src_shape) - len(self._tup) + 1
        i = 0
        for ind in self._tup:
            if ind is ...:
                for j in range(nfull):
                    lst[i] = slice(None)
                    i += 1
            else:
                lst[i] = ind
                i += 1
        if len(lst) == 1:
            idxer = indexer(lst[0])
        else:
            idxer = indexer(tuple(lst))
        idxer.set_src_shape(self._src_shape)
        self._shaped_inst = idxer.shaped_instance()
        return self._shaped_inst

    def as_slice(self):
        """
        Return a tuple of slices into a multidimensional array.

        Returns
        -------
        tuple of slices
            The slice into a multidimensional array.
        """
        return self.shaped_slice()

    def as_array(self, copy=False):
        """
        Return an index array into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.

        Returns
        -------
        ndarray
            The index array into a flat array.
        """
        # return as a flattened index array into a flat source
        return self.shaped_array(copy=copy)

    def flat(self, copy=False):
        """
        Return an index array into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.

        Returns
        -------
        ndarray
            An index array into a flat array.
        """
        return self.as_array(copy=copy)

    def to_json(self):
        """
        Return a JSON serializable version of self.

        Returns
        -------
        list of int or int
            list or int version of self.
        """
        return self.as_array().tolist()


class IndexMaker(object):
    """
    A Factory for Indexer objects.
    """

    def _get_indexer(self, idx, convert):
        """
        Return an Indexer instance based on the passed indices/slices.

        Parameters
        ----------
        idx : int, ndarray, slice, or tuple
            Some sort of index/indices/slice.
        convert : bool
            If True, conversion of arrays to slices and ellipses to multi-indexers is allowed.

        Returns
        -------
        Indexer
            The Indexer instance we created based on the args.
        """
        if convert and idx is ...:
            idxer = EllipsisIndexer((idx,))
        elif isinstance(idx, int):
            idxer = IntIndexer(idx)
        elif isinstance(idx, slice):
            idxer = SliceIndexer(idx)

        elif convert and isinstance(idx, tuple):
            if ... in idx:
                idxer = EllipsisIndexer(idx)
            else:
                idxer = MultiIndexer(idx)
        else:
            idx = np.atleast_1d(idx)

            if convert:
                # if array is convertable to a slice, store it as a slice
                slc = array2slice(idx)
                if slc is None:
                    if idx.ndim == 1:
                        idxer = ArrayIndexer(idx)
                    else:
                        idxer = MultiIndexer(tuple(idx))
                else:
                    idxer = SliceIndexer(slc)
            else:
                # can't convert sub-index arrays into sub slices because that can change
                # the result
                if idx.ndim == 1:
                    idxer = ArrayIndexer(idx, False)
                else:
                    idxer = MultiIndexer(tuple(idx))

        shaped = idxer.shaped_instance()
        if shaped is not None:
            return shaped
        return idxer

    def __getitem__(self, idx):
        """
        Return an Indexer based on idx.

        Parameters
        ----------
        idx : int, ndarray, slice or tuple
            The passed indices/slices.

        Returns
        -------
        Indexer
            The Indexer instance we created based on the args.
        """
        return self._get_indexer(idx, convert=True)

    def __call__(self, idx, convert=True):
        """
        Return an Indexer based on the args.

        Parameters
        ----------
        idx : int, ndarray, slice or tuple
            The passed indices/slices.
        convert : bool
            If True, conversion of arrays to slices and ellipses to multi-indexers is allowed.

        Returns
        -------
        Indexer
            The Indexer instance we created based on the args.
        """
        return self._get_indexer(idx, convert)


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
