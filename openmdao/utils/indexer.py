"""
Classes that handle array indexing.
"""

import sys
import numpy as np
from numbers import Integral
from itertools import zip_longest

from openmdao.utils.general_utils import shape2tuple
from openmdao.utils.array_utils import shape_to_len
from openmdao.utils.om_warnings import issue_warning


def array2slice(arr):
    """
    Try to convert an array to slice.

    Conversion is only attempted for a 1D array.

    Parameters
    ----------
    arr : ndarray
        The index array to be represented as a slice.

    Returns
    -------
    slice or None
        If slice conversion is possible, return the slice, else return None.
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
        else:
            return slice(0, 0)


def _truncate(s):
    if len(s) > 40:
        return s[:20] + ' ... ' + s[-20:]
    return s


def combine_ranges(ranges):
    """
    Combine a list of (start, end) tuples into the smallest possible list of contiguous ranges.

    The ranges are assumed to be non-overlapping and in ascending order.

    Parameters
    ----------
    ranges : list
        List of (start, end) tuples.

    Returns
    -------
    list of tuples
        List of combined ranges.
    """
    rnglist = []
    if not ranges:
        return rnglist

    cstart, cend = ranges[0]
    for start, end in ranges[1:]:
        if start == cend:
            cend = end
        else:
            rnglist.append((cstart, cend))
            cstart, cend = start, end

    rnglist.append((cstart, cend))

    return rnglist


def ranges2indexer(ranges, src_shape=None):
    """
    Convert a list of ranges to an indexer.

    Parameters
    ----------
    ranges : list
        List of (start, end) tuples.
    src_shape : tuple or None
        The shape of the source array.

    Returns
    -------
    Indexer
        Indexer object.
    """
    ranges = combine_ranges(ranges)
    if len(ranges) == 1:
        idx = slice(ranges[0][0], ranges[0][1])
    elif len(ranges) == 0:
        idx = slice(0, 0)
    else:
        idx = np.concatenate([np.arange(start, end) for start, end in ranges])

    if src_shape is None:
        src_shape = (ranges[-1][1] - ranges[0][0],)

    return indexer(idx, src_shape=src_shape, flat_src=True)


class Indexer(object):
    """
    Abstract indexing class.

    Parameters
    ----------
    flat_src : bool
        True if we're treating the source as flat.

    Attributes
    ----------
    _src_shape : tuple or None
        Shape of the 'source'.  Used to determine actual index or slice values when indices are
        negative or slice contains negative start or stop values or ':' or '...'.
    _shaped_inst : Indexer or None
        Cached shaped_instance if we've computed it before.
    _flat_src : bool
        If True, index is into a flat source array.
    _dist_shape : tuple
        Distributed shape of the source.
    """

    def __init__(self, flat_src=None):
        """
        Initialize attributes.
        """
        self._src_shape = None
        self._dist_shape = None
        self._shaped_inst = None
        self._flat_src = flat_src

    def __call__(self):
        """
        Return the indices in their most efficient form.

        For example, if the original indices were an index array that is convertable to a slice,
        then a slice would be returned.

        This could be either an int, a slice, an index array, or a multidimensional 'fancy' index.
        """
        raise NotImplementedError("No implementation of '__call__' found.")

    def __repr__(self):
        """
        Return simple string representation.

        Returns
        -------
        str
            String representation.
        """
        return f"{self.__class__.__name__}: {str(self)}"

    def copy(self, *args):
        """
        Copy this Indexer.

        Parameters
        ----------
        *args : position args
            Args that are specific to initialization of a derived Indexer.

        Returns
        -------
        Indexer
            A copy of this Indexer.
        """
        inst = self.__class__(*args)
        inst.__dict__.update(self.__dict__)
        return inst

    def _set_attrs(self, parent):
        """
        Copy certain attributes from the parent to self.

        Parameters
        ----------
        parent : Indexer
            Parent of this indexer.

        Returns
        -------
        Indexer
            This indexer.
        """
        self._src_shape = parent._src_shape
        self._flat_src = parent._flat_src
        self._dist_shape = parent._dist_shape
        return self

    @property
    def indexed_src_shape(self):
        """
        Return the shape of the result if the indices were applied to a source array.

        Returns
        -------
        tuple
            The shape of the result.
        """
        s = self.shaped_instance()
        if s is None:
            raise RuntimeError(f"Can't get indexed_src_shape of {self} because source shape "
                               "is unknown.")
        if self._flat_src:
            return resolve_shape(shape_to_len(self._src_shape)).get_shape(self.flat())
        else:
            return resolve_shape(self._src_shape).get_shape(self())

    @property
    def indexed_src_size(self):
        """
        Return the size of the result if the index were applied to the source.

        Returns
        -------
        int
            Size of flattened indices.
        """
        return shape_to_len(self.indexed_src_shape)

    def flat(self, copy=False):
        """
        Return index array or slice into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.
        """
        raise NotImplementedError("No implementation of 'flat' found.")

    def shaped_instance(self):
        """
        Return a 'shaped' version of this Indexer type.

        This should be overridden for all non-shaped derived classes.

        Returns
        -------
        Indexer
            The 'shaped' Indexer type.  'shaped' Indexers know the extent of the array that
            they are indexing into, or they don't care what the extent is because they don't
            contain negative indices, negative start or stop, ':', or '...'.
        """
        return self

    def shaped_array(self, copy=False, flat=True):
        """
        Return an index array version of the indices that index into a flattened array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.
        flat : bool
            If True, return a flat array.

        Returns
        -------
        ndarray
            Version of these indices that index into a flattened array.
        """
        s = self.shaped_instance()
        if s is None:
            raise ValueError(f"Can't get shaped array of {self} because it has no source shape.")
        return s.as_array(copy=copy, flat=flat)

    def apply(self, subidxer):
        """
        Apply a sub-Indexer to this Indexer and return the resulting indices.

        Parameters
        ----------
        subidxer : Indexer
            The Indexer to be applied to this one.

        Returns
        -------
        ndarray
            The resulting indices (always flat).
        """
        arr = self.shaped_array().ravel()
        return arr[subidxer.flat()]

    def set_src_shape(self, shape, dist_shape=None):
        """
        Set the shape of the 'source' array .

        Parameters
        ----------
        shape : tuple or int
            The shape of the 'source' array.
        dist_shape : tuple or None
            If not None, the full distributed shape of the source.

        Returns
        -------
        Indexer
            Self is returned to allow chaining.
        """
        sshape, self._dist_shape, = self._get_shapes(shape, dist_shape)

        if shape is not None:
            if self._flat_src is None:
                self._flat_src = len(sshape) <= 1

        if sshape != self._src_shape:
            self._src_shape = sshape
            try:
                self._check_bounds()
            except Exception:
                self._src_shape = None
                self._dist_shape = None
                raise
            self._shaped_inst = None

        return self

    def to_json(self):
        """
        Return a JSON serializable version of self.
        """
        raise NotImplementedError("No implementation of 'to_json' found.")

    def _get_shapes(self, shape, dist_shape):
        if shape is None:
            return None, None

        shape = shape2tuple(shape)
        if self._flat_src:
            shape = (shape_to_len(shape),)

        if dist_shape is None:
            return shape, shape

        dist_shape = shape2tuple(dist_shape)
        if self._flat_src:
            dist_shape = (shape_to_len(dist_shape),)

        return shape, dist_shape


class ShapedIntIndexer(Indexer):
    """
    Int indexing class.

    Parameters
    ----------
    idx : int
        The index.
    flat_src : bool
        If True, source is treated as flat.

    Attributes
    ----------
    _idx : int
        The integer index.
    """

    def __init__(self, idx, flat_src=None):
        """
        Initialize attributes.
        """
        super().__init__(flat_src)
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

    def apply_offset(self, offset, flat=True):
        """
        Apply an offset to this index.

        Parameters
        ----------
        offset : int
            The offset to apply.
        flat : bool
            If True, return a flat index.

        Returns
        -------
        int
            The offset index.
        """
        return self._idx + offset

    def copy(self):
        """
        Copy this Indexer.

        Returns
        -------
        Indexer
            A copy of this Indexer.
        """
        return super().copy(self._idx)

    @property
    def min_src_dim(self):
        """
        Return the number of source dimensions.

        Returns
        -------
        int
            The number of dimensions expected in the source array.
        """
        return 1

    @property
    def indexed_src_shape(self):
        """
        Return the shape of the index ().

        Returns
        -------
        tuple
            The shape of the index.
        """
        if self._flat_src:
            return (1,)
        return super().indexed_src_shape

    def as_array(self, copy=False, flat=True):
        """
        Return an index array into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.
        flat : bool
            If True, return a flat array.

        Returns
        -------
        ndarray
            The index array.
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

    def _check_bounds(self):
        """
        Check that indices are within the bounds of the source shape.
        """
        if self._src_shape is not None and (self._idx >= self._dist_shape[0] or
                                            self._idx < -self._dist_shape[0]):
            raise IndexError(f"index {self._idx} is out of bounds of the source shape "
                             f"{self._dist_shape}.")

    def to_json(self):
        """
        Return a JSON serializable version of self.

        Returns
        -------
        int
            Int version of self.
        """
        return self._idx


class IntIndexer(ShapedIntIndexer):
    """
    Int indexing class that may or may not be 'shaped'.

    Parameters
    ----------
    idx : int
        The index.
    flat_src : bool or None
        If True, treat source as flat.
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

        if self._src_shape is None:
            return None

        if self._idx < 0:
            self._shaped_inst = ShapedIntIndexer(self._idx + self._src_shape[0])
        else:
            self._shaped_inst = ShapedIntIndexer(self._idx)

        return self._shaped_inst._set_attrs(self)


class ShapedSliceIndexer(Indexer):
    """
    Abstract slice class that is 'shaped'.

    Parameters
    ----------
    slc : slice
        The slice.
    flat_src : bool
        If True, source is treated as flat.

    Attributes
    ----------
    _slice : slice
        The wrapped slice object.
    """

    def __init__(self, slc, flat_src=None):
        """
        Initialize attributes.
        """
        super().__init__(flat_src)
        if slc.step is None:
            slc = slice(slc.start, slc.stop, 1)
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

    def apply_offset(self, offset, flat=True):
        """
        Apply an offset to this index.

        Parameters
        ----------
        offset : int
            The offset to apply.
        flat : bool
            If True, return a flat index.

        Returns
        -------
        slice
            The offset slice.
        """
        return slice(self._slice.start + offset, self._slice.stop + offset, self._slice.step)

    def copy(self):
        """
        Copy this Indexer.

        Returns
        -------
        Indexer
            A copy of this Indexer.
        """
        return super().copy(self._slice)

    def as_array(self, copy=False, flat=True):
        """
        Return an index array into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.
        flat : bool
            If True, return a flat array.

        Returns
        -------
        ndarray
            The index array.
        """
        if len(self._src_shape) == 1:
            # Case 1: Requested flat or nonflat indices but src_shape is None or flat
            # return a flattened arange
            slc = self._slice
            if slc.stop is None and slc.step < 0:  # special case - neg step down to -1
                return np.arange(self._src_shape[0], dtype=int)[slc]
            else:
                # use maxsize here since a shaped slice always has positive int start and stop
                return np.arange(*slc.indices(sys.maxsize), dtype=int)
        else:
            src_size = shape_to_len(self._src_shape)
            arr = np.arange(src_size, dtype=int).reshape(self._src_shape)[self._slice].ravel()
            if flat:
                # Case 2: Requested flattened indices of multidimensional array
                # Return indices into a flattened src.
                return arr
            else:
                # Case 3: Requested non-flat indices of multidimensional array
                # This is never called within OpenMDAO
                return np.unravel_index(arr, shape=self._src_shape)

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

    @property
    def min_src_dim(self):
        """
        Return the number of source dimensions.

        Returns
        -------
        int
            The number of dimensions expected in the source array.
        """
        return 1

    def _check_bounds(self):
        """
        Check that indices are within the bounds of the source shape.
        """
        # a slice with start or stop outside of the source range is allowed in numpy arrays
        # and just results in an empty array, but in OpenMDAO that behavior would probably be
        # unintended, so for now make it an error.
        if self._src_shape is not None:
            start = self._slice.start
            stop = self._slice.stop
            sz = shape_to_len(self._dist_shape)
            if (start is not None and (start >= sz or start < -sz)
                    or (stop is not None and (stop > sz or stop < -sz))):
                raise IndexError(f"{self._slice} is out of bounds of the source shape "
                                 f"{self._dist_shape}.")

    def to_json(self):
        """
        Return a JSON serializable version of self.

        Returns
        -------
        list of int or int
            List or int version of self.
        """
        return self.as_array().tolist()


class SliceIndexer(ShapedSliceIndexer):
    """
    Abstract slice class that may or may not be 'shaped'.

    Parameters
    ----------
    slc : slice
        The slice.
    flat_src : bool or None
        If True, treat source as flat.
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

        if self._src_shape is None:
            return None

        slc = self._slice
        if slc.stop is None and slc.step < 0:  # special backwards indexing case
            self._shaped_inst = \
                ShapedSliceIndexer(slc)
        elif (slc.start is not None and slc.start < 0) or slc.stop is None or slc.stop < 0:
            self._shaped_inst = \
                ShapedSliceIndexer(slice(*self._slice.indices(self._src_shape[0])))
        else:
            self._shaped_inst = ShapedSliceIndexer(slc)

        return self._shaped_inst._set_attrs(self)

    def as_array(self, copy=False, flat=True):
        """
        Return an index array into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.
        flat : bool
            If True, return a flat array.

        Returns
        -------
        ndarray
            The index array.
        """
        return self.shaped_array(copy=copy, flat=flat)

    @property
    def indexed_src_shape(self):
        """
        Return the shape of the result of indexing into the source.

        Returns
        -------
        tuple
            The shape of the index.
        """
        slc = self._slice
        if self._flat_src and slc.start is not None and slc.stop is not None:
            return (len(range(slc.start, slc.stop, slc.step)),)
        return super().indexed_src_shape


class ShapedArrayIndexer(Indexer):
    """
    Abstract index array class that knows its source shape.

    Parameters
    ----------
    arr : ndarray
        The index array.
    flat_src : bool
        If True, source is treated as flat.

    Attributes
    ----------
    _arr : ndarray
        The wrapped index array object.
    """

    def __init__(self, arr, flat_src=None):
        """
        Initialize attributes.
        """
        super().__init__(flat_src)

        ndarr = np.asarray(arr)

        # check type
        if ndarr.dtype.kind not in ('i', 'u'):
            raise TypeError(f"Can't create an index array using indices of "
                            f"non-integral type '{ndarr.dtype.type.__name__}'.")

        self._arr = ndarr

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
        return _truncate(f"{self._arr}".replace('\n', ''))

    def apply_offset(self, offset, flat=True):
        """
        Apply an offset to this index.

        Parameters
        ----------
        offset : int
            The offset to apply.
        flat : bool
            If True, return a flat index.

        Returns
        -------
        slice
            The offset slice.
        """
        return self.as_array(flat=flat) + offset

    def copy(self):
        """
        Copy this Indexer.

        Returns
        -------
        Indexer
            A copy of this Indexer.
        """
        return super().copy(self._arr)

    @property
    def min_src_dim(self):
        """
        Return the number of source dimensions.

        Returns
        -------
        int
            The number of dimensions expected in the source array.
        """
        return 1

    def as_array(self, copy=False, flat=True):
        """
        Return an index array into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.
        flat : bool
            If True, return a flat array.

        Returns
        -------
        ndarray
            The index array.
        """
        if flat:
            arr = self._arr.ravel()
        else:
            arr = self._arr
        if copy:
            return arr.copy()
        return arr

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
            return self._arr.ravel().copy()
        return self._arr.ravel()

    def _check_bounds(self):
        """
        Check that indices are within the bounds of the source shape.
        """
        if self._src_shape is not None and self._arr.size > 0:
            src_size = shape_to_len(self._dist_shape)
            amax = np.max(self._arr)
            ob = None
            if amax >= src_size or -amax < -src_size:
                ob = amax
            if ob is None:
                amin = np.min(self._arr)
                if amin < 0 and -amin > src_size:
                    ob = amin
            if ob is not None:
                raise IndexError(f"index {ob} is out of bounds for source dimension of size "
                                 f"{src_size}.")

    def to_json(self):
        """
        Return a JSON serializable version of self.

        Returns
        -------
        list of int or int
            List or int version of self.
        """
        return self().tolist()


class ArrayIndexer(ShapedArrayIndexer):
    """
    Abstract index array class that may or may not be 'shaped'.

    Parameters
    ----------
    arr : ndarray
        The index array.
    flat_src : bool or None
        If True, treat source as flat.
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

        if self._src_shape is None:
            return None

        negs = self._arr < 0
        if np.any(negs):
            sharr = self._arr.copy()
            sharr[negs] += self._src_shape[0]
        else:
            sharr = self._arr

        self._shaped_inst = ShapedArrayIndexer(sharr)
        return self._shaped_inst._set_attrs(self)

    @property
    def indexed_src_shape(self):
        """
        Return the shape of the result of indexing into the source.

        Returns
        -------
        tuple
            The shape of the index.
        """
        return self._arr.shape


class ShapedMultiIndexer(Indexer):
    """
    Abstract multi indexer class that is 'shaped'.

    Parameters
    ----------
    tup : tuple
        Tuple of indices/slices.
    flat_src : bool
        If True, treat source array as flat.

    Attributes
    ----------
    _tup : tuple
        The wrapped tuple of indices/slices.
    _idx_list : list
        List of Indexers.
    """

    def __init__(self, tup, flat_src=False):
        """
        Initialize attributes.
        """
        if flat_src and len(tup) > 1:
            raise RuntimeError(f"Can't index into a flat array with an indexer expecting {len(tup)}"
                               " dimensions.")
        super().__init__(flat_src)
        self._tup = tup
        self._set_idx_list()

    def _set_idx_list(self):
        self._idx_list = []
        for i in self._tup:
            if isinstance(i, (np.ndarray, list)):  # need special handling here for ndim > 1 arrays
                self._idx_list.append(ArrayIndexer(i, flat_src=self._flat_src))
            else:
                self._idx_list.append(indexer(i, flat_src=self._flat_src))

    def __call__(self):
        """
        Return this multidimensional index.

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
        return str(self._tup)

    def apply_offset(self, offset, flat=True):
        """
        Apply an offset to this index.

        Parameters
        ----------
        offset : int
            The offset to apply.
        flat : bool
            If True, return a flat index.

        Returns
        -------
        ndarray
            The offset array.
        """
        if flat:
            return self.flat() + offset
        return self.as_array(flat=False) + offset

    def copy(self):
        """
        Copy this Indexer.

        Returns
        -------
        Indexer
            A copy of this Indexer.
        """
        return super().copy(self._tup)

    @property
    def min_src_dim(self):
        """
        Return the number of source dimensions.

        Returns
        -------
        int
            The number of dimensions expected in the source array.
        """
        return len(self._idx_list)

    def as_array(self, copy=False, flat=True):
        """
        Return an index array into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.
        flat : bool
            If True, return a flat array.

        Returns
        -------
        ndarray
            The index array into a flat array.
        """
        if self._src_shape is None:
            raise ValueError(f"Can't determine extent of array because source shape is not known.")

        idxs = np.arange(shape_to_len(self._src_shape), dtype=np.int32).reshape(self._src_shape)

        if flat:
            return idxs[self()].ravel()
        else:
            return idxs[self()]

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
        return self.shaped_array(copy=copy, flat=True)

    def set_src_shape(self, shape, dist_shape=None):
        """
        Set the shape of the 'source' array .

        Parameters
        ----------
        shape : tuple or int
            The shape of the 'source' array.
        dist_shape : tuple or None
            If not None, the full distributed shape of the source.

        Returns
        -------
        Indexer
            Self is returned to allow chaining.
        """
        self._check_src_shape(shape2tuple(shape))
        super().set_src_shape(shape, dist_shape)
        if shape is None:
            return self

        if self._flat_src:
            for i in self._idx_list:
                i.set_src_shape(self._src_shape, self._dist_shape)
        else:
            for i, s, ds in zip(self._idx_list, self._src_shape, self._dist_shape):
                i.set_src_shape(s, ds)

        return self

    def _check_src_shape(self, shape):
        if shape is not None and not self._flat_src and len(shape) < len(self._idx_list):
            raise ValueError(f"Can't set source shape to {shape} because indexer {self} expects "
                             f"{len(self._idx_list)} dimensions.")

    def _check_bounds(self):
        """
        Check that indices are within the bounds of the source shape.
        """
        if self._src_shape is not None:
            for i in self._idx_list:
                i._check_bounds()

    def to_json(self):
        """
        Return a JSON serializable version of self.

        Returns
        -------
        list of int or int
            List or int version of self.
        """
        return self.as_array().tolist()


class MultiIndexer(ShapedMultiIndexer):
    """
    Abstract multi indexer class that may or may not be 'shaped'.

    Parameters
    ----------
    tup : tuple
        Tuple of indices/slices.
    flat_src : bool
        If True, treat source array as flat.
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

        if self._src_shape is None:
            return None

        try:
            self._shaped_inst = ShapedMultiIndexer(tuple(idxer.shaped_instance()()
                                                         for idxer in self._idx_list),
                                                   flat_src=self._flat_src)
        except Exception as err:
            self._shaped_inst = None
        else:
            self._shaped_inst.set_src_shape(self._src_shape)
            self._shaped_inst._set_attrs(self)

        return self._shaped_inst


class EllipsisIndexer(Indexer):
    """
    Abstract multi indexer class that is 'shaped'.

    Parameters
    ----------
    tup : tuple
        Tuple of indices/slices.
    flat_src : bool
        If True, treat source array as flat.

    Attributes
    ----------
    _tup : tuple
        The wrapped tuple of indices/slices (it contains an ellipsis).
    """

    def __init__(self, tup, flat_src=None):
        """
        Initialize attributes.
        """
        super().__init__(flat_src)
        tlist = []
        # convert any internal lists/tuples to arrays
        for i, v in enumerate(tup):
            if isinstance(v, (list, tuple)):
                v = np.atleast_1d(v)
            tlist.append(v)
        self._tup = tuple(tlist)

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

    def apply_offset(self, offset, flat=True):
        """
        Apply an offset to this index.

        Parameters
        ----------
        offset : int
            The offset to apply.
        flat : bool
            If True, return a flat index.

        Returns
        -------
        ndarray
            The offset array.
        """
        return self.as_array(flat=flat) + offset

    def copy(self):
        """
        Copy this Indexer.

        Returns
        -------
        EllipsisIndexer
            A copy of this Indexer.
        """
        return super().copy(self._tup)

    @property
    def min_src_dim(self):
        """
        Return the number of source dimensions.

        Returns
        -------
        int
            The number of dimensions expected in the source array.
        """
        mn = len(self._tup) - 1
        return mn if mn > 1 else 1

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
        return self._shaped_inst._set_attrs(self)

    def as_array(self, copy=False, flat=True):
        """
        Return an index array into a flat array.

        Parameters
        ----------
        copy : bool
            If True, make sure the array returned is a copy.
        flat : bool
            If True, return a flat array.

        Returns
        -------
        ndarray
            The index array.
        """
        return self.shaped_array(copy=copy, flat=flat)

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

    def _check_bounds(self):
        """
        Check that indices are within the bounds of the source shape.
        """
        s = self.shaped_instance()
        if s is not None:
            s._check_bounds()

    def to_json(self):
        """
        Return a JSON serializable version of self.

        Returns
        -------
        list of int or int
            A list or int version of self.
        """
        return self.as_array().tolist()


class IndexMaker(object):
    """
    A Factory for Indexer objects.
    """

    def __call__(self, idx, src_shape=None, flat_src=False):
        """
        Return an Indexer instance based on the passed indices/slices.

        Parameters
        ----------
        idx : int, ndarray, slice, or tuple
            Some sort of index/indices/slice.
        src_shape : tuple or None
            Source shape if known.
        flat_src : bool
            If True, indices are into a flat source.

        Returns
        -------
        Indexer
            The Indexer instance we created based on the args.
        """
        if idx is ...:
            idxer = EllipsisIndexer((idx,), flat_src=flat_src)
        elif isinstance(idx, int):
            idxer = IntIndexer(idx, flat_src=flat_src)
        elif isinstance(idx, slice):
            idxer = SliceIndexer(idx, flat_src=flat_src)

        elif isinstance(idx, tuple):
            multi = len(idx) > 1
            for i in idx:
                if i is ...:
                    multi = len(idx) > 2  # ... doesn't count toward limit of dimensions
                    idxer = EllipsisIndexer(idx, flat_src=flat_src)
                    break
            else:
                idxer = MultiIndexer(idx, flat_src=flat_src)
            if flat_src and multi:
                raise RuntimeError("Can't use a multdimensional index into a flat source.")
        else:
            arr = np.atleast_1d(idx)
            if arr.ndim == 1:
                idxer = ArrayIndexer(arr, flat_src=flat_src)
            else:
                issue_warning("Using a non-tuple sequence for multidimensional indexing is "
                              "deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the "
                              "future this will be interpreted as an array index, "
                              "`arr[np.array(seq)]`, which will result either in an error or a "
                              "different result.")
                idxer = MultiIndexer(tuple(idx), flat_src=flat_src)

        if src_shape is not None:
            if flat_src:
                src_shape = (shape_to_len(src_shape),)
            idxer.set_src_shape(src_shape)

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
        return self(idx)


indexer = IndexMaker()


def _convert_ellipsis_idx(shape, idx):
    lst = [None] * len(shape)
    # number of full slice dimensions
    nfull = len(shape) - len(idx) + 1
    i = 0
    for ind in idx:
        if ind is ...:
            for j in range(nfull):
                lst[i] = slice(None)
                i += 1
        else:
            lst[i] = ind
            i += 1

    return tuple(lst)


class resolve_shape(object):
    """
    Class that computes the result shape from a source shape and an index.

    Parameters
    ----------
    shape : tuple
        The shape of the source.

    Attributes
    ----------
    _shape : tuple
        The shape of the source.
    """

    def __init__(self, shape):
        """
        Initialize attributes.

        Parameters
        ----------
        shape : tuple or int
            Shape of the source.
        """
        self._shape = shape2tuple(shape)

    def get_shape(self, idx):
        """
        Return the shape of the result of indexing into the source with index idx.

        Parameters
        ----------
        idx : int, slice, tuple, ndarray
            The index into the source.

        Returns
        -------
        tuple
            The shape after indexing.
        """
        if not isinstance(idx, tuple):
            idx = (idx,)
            is_tup = False
        else:
            is_tup = True

        for i in idx:
            if i is ...:
                idx = _convert_ellipsis_idx(self._shape, idx)
                break

        if len(self._shape) < len(idx):
            raise ValueError(f"Index {idx} dimension too large to index into shape {self._shape}.")

        lens = []
        seen_arr = False
        arr_shape = None  # to handle multi-indexing where individual sub-arrays have a shape
        for dim, ind in zip_longest(self._shape, idx):
            if ind is None:
                lens.append(dim)
            elif isinstance(ind, slice):
                lens.append(len(range(*ind.indices(dim))))
            elif isinstance(ind, np.ndarray):
                if not seen_arr:
                    seen_arr = True
                    if ind.ndim > 1:
                        if arr_shape is not None and arr_shape != ind.shape:
                            raise ValueError("Multi-index has index sub-arrays of different "
                                             f"shapes ({arr_shape} != {ind.shape}).")
                        arr_shape = ind.shape
                    else:
                        # only first array idx counts toward shape
                        lens.append(ind.size)
            # int indexers don't count toward shape (scalar array has shape ())
            elif not isinstance(ind, Integral):
                raise TypeError(f"Index {ind} of type '{type(ind).__name__}' is invalid.")

        if arr_shape is not None:
            return arr_shape

        if is_tup or len(lens) >= 1:
            return tuple(lens)
        elif is_tup:
            return ()
        return (1,)


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
