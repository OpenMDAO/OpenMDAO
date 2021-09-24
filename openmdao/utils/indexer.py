"""
Classes that handle array indexing.
"""

import sys
import numpy as np
from numbers import Integral

from openmdao.utils.general_utils import shape2tuple
from openmdao.utils.om_warnings import warn_deprecation
from openmdao.utils.om_warnings import issue_warning, OMDeprecationWarning


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


class Indexer(object):
    """
    Abstract indexing class.

    Parameters
    ----------
    flat_src : bool or None
        If True, treat source as flat.

    Attributes
    ----------
    _src_shape : tuple or None
        Shape of the 'source'.  Used to determine actual index or slice values when indices are
        negative or slice contains negative start or stop values or ':' or '...'.
    _shaped_inst : Indexer or None
        Cached shaped_instance if we've computed it before.
    _flat_src : bool
        If True, index is into a flat source array.
    _dist_shape : tuple or None
        Distributed shape.
    _orig_src_shape : tuple or None
        Original shape of the source, before possible flattening based on _flat_src flag.
    """

    def __init__(self, flat_src):
        """
        Initialize attributes.
        """
        self._src_shape = None
        self._shaped_inst = None
        self._flat_src = flat_src
        self._dist_shape = None
        # TODO: remove this after flat src deprecation branch
        self._orig_src_shape = None

    def __len__(self):
        """
        Return the length of the flattened indices.

        Returns
        -------
        int
            Length of flattened indices.
        """
        return self.size

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

    def __eq__(self, other):
        """
        Return True if this indexer is equivalent to other.

        Parameters
        ----------
        other : Indexer
            The other Indexer.

        Returns
        -------
        bool
            True if Indexers are equivalent.
        """
        return np.all(self.as_array() == other.as_array())

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
        self._orig_src_shape = parent._orig_src_shape
        return self

    @property
    def size(self):
        """
        Return the size of the flattened indices.

        Returns
        -------
        int
            Size of flattened indices.
        """
        return np.product(self.shape, dtype=int)

    @property
    def src_ndim(self):
        """
        Return the number of source dimensions.

        Returns
        -------
        int
            The number of dimensions expected in the source array.
        """
        return 1

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

    @property
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
        return s.shape

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
            raise ValueError(f"Can't get shaped array of {self} because it has "
                             "no source shape.")
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
        return arr[self.flat()]

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
        self._orig_src_shape = shape
        self._src_shape, self._dist_shape = self._get_shapes(shape, dist_shape)
        if shape is not None:
            self._check_bounds()

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
        if self._flat_src or self._appears_flat():
            shape = (np.product(shape),)

        if dist_shape is None:
            return shape, shape

        dist_shape = shape2tuple(dist_shape)
        if self._flat_src:
            dist_shape = (np.product(dist_shape),)

        return shape, dist_shape

    def _appears_flat(self):
        return True

    def _check_flat_indices_warning(self, flat, src_shape, vname, prefix=''):
        if src_shape is None or flat or not self._appears_flat():
            return
        src_shape = shape2tuple(src_shape)
        if len(src_shape) > 1:
            if flat is None:
                issue_warning(f"Indexing into source variable '{vname}' of dimension "
                              f"{len(src_shape)} using indices of dimension "
                              f"{len(shape2tuple(self.shape))} without setting `flat_indices=True`."
                              " The source is currently treated as flat, but this automatic "
                              f"flattening is deprecated and will be removed in a future release. "
                              "To keep the old behavior, set `flat_indices=True` when you set "
                              "`indices`.", category=OMDeprecationWarning, prefix=prefix)
            else:  # flat is False
                raise IndexError(f"Indexing into source variable '{vname}' of dimension "
                                 f"{len(src_shape)} using indices of dimension "
                                 f"{len(shape2tuple(self.shape))} and setting "
                                 "`flat_indices=False`. The current version of OpenMDAO assumes "
                                 "a flat source if the indices appear flat, so `flat_indices=False`"
                                 " is not a valid option. This behavior is "
                                 "deprecated and will be removed in a later version.")

    def _check_flat_src_indices_warning(self, flat, src_shape, tgt, prefix=''):
        if src_shape is None or self._flat_src or flat or not self._appears_flat():
            return
        src_shape = shape2tuple(src_shape)
        if len(src_shape) > 1:
            if flat is None:
                issue_warning(f"Indexing into a source array of dimension {len(src_shape)} "
                              f"using indices of dimension {len(shape2tuple(self.shape))} without "
                              f"setting `flat_src_indices=True` when connecting to input '{tgt}'. "
                              "The source array is currently treated as flat, but this automatic "
                              "flattening is deprecated and will be removed in a future release. "
                              "To keep the old behavior, set `flat_src_indices=True` when you set "
                              "`src_indices`.", category=OMDeprecationWarning, prefix=prefix)
            else:  # flat is False
                raise IndexError(f"Indexing into a source array of dimension {len(src_shape)} using"
                                 f" indices of dimension {len(shape2tuple(self.shape))} and setting"
                                 f" `flat_indices=False` when connecting to input '{tgt}'. The "
                                 "current version of OpenMDAO assumes a flat source if the indices "
                                 "appear flat, so `flat_indices=False` is not a valid option. This "
                                 "behavior is deprecated and will be removed in a later version.")


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
    def size(self):
        """
        Return the size of the flattened indices.

        Returns
        -------
        int
            Size of flattened indices.
        """
        return 1

    @property
    def shape(self):
        """
        Return the shape of the index ().

        Returns
        -------
        tuple
            The shape of the index.
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

        if self._idx < 0:
            if self._src_shape is None:
                return None
            else:
                self._shaped_inst = ShapedIntIndexer(self._idx + self._src_shape[0])
            return self._shaped_inst

        return ShapedIntIndexer(self._idx)._set_attrs(self)


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
        # use maxsize here since a shaped slice always has positive int start and stop
        arr = np.arange(*self._slice.indices(sys.maxsize), dtype=int)
        if flat:
            return arr

        if self._orig_shape is None:
            return arr
        return arr.reshape(self._orig_shape)

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
    def shape(self):
        """
        Return the shape of the indices.

        Returns
        -------
        int
            The shape of the indices.
        """
        # use maxsize here since shaped slice always has positive int start and stop
        return (len(range(*self._slice.indices(sys.maxsize))),)

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
            sz = np.product(self._dist_shape)
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

        slc = self._slice
        if (slc.start is not None and slc.start < 0) or slc.stop is None or slc.stop < 0:
            if self._src_shape is None:
                return None
            else:
                self._shaped_inst = ShapedSliceIndexer(
                    slice(*self._slice.indices(self._src_shape[0])))._set_attrs(self)
                return self._shaped_inst

        return ShapedSliceIndexer(slc)._set_attrs(self)

    @property
    def shape(self):
        """
        Return the shape of the indices.

        Returns
        -------
        int
            The shape of the indices.
        """
        s = self.shaped_instance()
        if s is None:
            slc = self._slice
            # if both start and stop are negative we can figure out the shape
            if slc.start is not None and slc.start < 0 and slc.stop is not None and slc.stop < 0:
                step = 1 if slc.step is None else slc.step
                return (len(range(slc.start, slc.stop, step)),)
            raise RuntimeError(f"Can't get shape of {self} because source shape "
                               "is unknown.")
        return s.shape

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


class ShapedArrayIndexer(Indexer):
    """
    Abstract index array class that is 'shaped'.

    Parameters
    ----------
    arr : ndarray
        The index array.
    orig_shape : tuple or None
        Original shape of the array.
    flat_src : bool
        If True, source is treated as flat.

    Attributes
    ----------
    _arr : ndarray
        The wrapped index array object.
    _orig_shape : tuple
        Original shape of the array.
    """

    def __init__(self, arr, orig_shape=None, flat_src=None):
        """
        Initialize attributes.
        """
        super().__init__(flat_src)

        ndarr = np.asarray(arr)

        # check type
        if ndarr.dtype.kind not in ('i', 'u'):
            raise TypeError(f"Can't create an index array using indices of "
                            f"non-integral type '{ndarr.dtype.type.__name__}'.")

        self._orig_shape = ndarr.shape if orig_shape is None else orig_shape

        self._arr = ndarr.flat[:]

    def __call__(self):
        """
        Return this index array.

        Returns
        -------
        int
            This index array.
        """
        v = self._arr.view()
        v.shape = self._orig_shape
        return v

    def __str__(self):
        """
        Return string representation.

        Returns
        -------
        str
            String representation.
        """
        return _truncate(f"{self()}".replace('\n', ''))

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
    def shape(self):
        """
        Return the shape of the indices.

        Returns
        -------
        tuple
            The shape of the indices.
        """
        return self._orig_shape

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
            arr = self._arr
        else:
            arr = self._arr.view()
            arr.shape = self._orig_shape
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
            return self._arr.copy()
        return self._arr

    def _check_bounds(self):
        """
        Check that indices are within the bounds of the source shape.
        """
        if self._src_shape is not None and self._arr.size > 0:
            src_size = np.product(self._dist_shape)
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
    orig_shape : tuple or None
        Original shape of the array.
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

        negs = self._arr < 0
        if np.any(negs):
            if self._src_shape is None:
                return None
            else:
                sharr = self._arr.copy()
                sharr[negs] += self._src_shape[0]
        else:
            sharr = self._arr

        self._shaped_inst = ShapedArrayIndexer(sharr, orig_shape=self._orig_shape)

        return self._shaped_inst._set_attrs(self)


class ShapedMultiIndexer(Indexer):
    """
    Abstract multi indexer class that is 'shaped'.

    Parameters
    ----------
    tup : tuple
        Tuple of indices/slices.
    orig_shape : tuple or None
        The original shape of the array.
    flat_src : bool
        If True, treat source array as flat.

    Attributes
    ----------
    _tup : tuple
        The wrapped tuple of indices/slices.
    _orig_shape : tuple
        Original shape.
    _idx_list : list
        List of Indexers.
    """

    def __init__(self, tup, orig_shape=None, flat_src=None):
        """
        Initialize attributes.
        """
        super().__init__(flat_src)
        self._tup = tup
        self._orig_shape = orig_shape
        self._set_idx_list()

    def _set_idx_list(self):
        self._idx_list = [indexer(i, flat=self._flat_src) for i in self._tup]

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
        return f"{self._tup}".strip('\n')

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
    def shape(self):
        """
        Return the shape of the indices.

        Returns
        -------
        tuple
            The shape of the indices.
        """
        lens = []
        seen_arr = False
        for i in self._idx_list:
            if isinstance(i, ShapedSliceIndexer):
                lens.append(len(i))
            elif isinstance(i, ShapedArrayIndexer) and not seen_arr:
                # only first array idx counts toward shape
                lens.append(len(i))
                seen_arr = True
            # int indexers don't count toward shape (scalar array has shape ())

        return tuple(lens)

    @property
    def src_ndim(self):
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
        # return as a flattened index array into a flat source
        if self._src_shape is None:
            raise ValueError(f"Can't determine extent of array because source shape is not known.")

        idxs = np.arange(np.product(self._src_shape), dtype=np.int32).reshape(self._src_shape)

        # TODO: add different behavior here for flat/nonflat
        if flat:
            return idxs[self()].ravel()
        else:
            if self._orig_shape is None:
                return idxs[self()]
            else:
                return idxs[self()].reshape(self._orig_shape)

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
        self._check_src_shape(shape)
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
        if shape is not None and len(shape) != len(self._idx_list):
            raise ValueError(f"Can't set source shape to {shape} because indexer {self} expects "
                             f"{self.src_ndim} dimensions.")

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

    def _appears_flat(self):
        return False


class MultiIndexer(ShapedMultiIndexer):
    """
    Abstract multi indexer class that may or may not be 'shaped'.

    Parameters
    ----------
    tup : tuple
        Tuple of indices/slices.
    orig_shape : tuple or None
        The original shape of the array.
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

        try:
            self._shaped_inst = ShapedMultiIndexer(tuple(idxer.shaped_instance()()
                                                         for idxer in self._idx_list),
                                                   flat_src=self._flat_src)
        except Exception as err:
            self._shaped_inst = None
            return

        if self._src_shape is not None:
            self._shaped_inst.set_src_shape(self._src_shape)

        return self._shaped_inst._set_attrs(self)


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
    def src_ndim(self):
        """
        Return the number of source dimensions.

        Returns
        -------
        int
            The number of dimensions expected in the source array.
        """
        if self._flat_src:
            return 1

        if self._shaped_inst is None:
            s = self.shaped_instance()
            if s is None:
                raise ValueError(f"Can't get ndim of {self} because it has no source shape.")

        return self._shaped_inst.src_ndim

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

    def _appears_flat(self):
        return False


# the following class will go away when we drop support for our custom non-flat indexing format
class ListOfTuplesArrayIndexer(Indexer):
    """
    Multi indexer using our custom 'list of tuples' format.

    Parameters
    ----------
    tup : tuple
        Tuple of indices/slices.
    flat_src : bool
        If True, source is treated as flat.

    Attributes
    ----------
    _arr : ndarray
        Multidimensional index array.
    _flat_src : bool
        If True, index is into a flat source array.
    _npy_inds : tuple
        Tuple containing the equivalent numpy compatible index arrays.
    """

    def __init__(self, tup, flat_src=None):
        """
        Initialize attributes.
        """
        tup = np.atleast_1d(tup)
        self._arr = tup

        orig_shape = tup.shape[:-1]
        ndims = tup.shape[-1]
        size = np.product(orig_shape)
        totsize = size * ndims
        self._npy_inds = tuple(tup.flat[i:totsize:ndims] for i in range(ndims))
        super().__init__(flat_src)

    def __str__(self):
        """
        Return string representation.

        Returns
        -------
        str
            String representation.
        """
        return _truncate(f"{self._arr}".replace('\n', ''))

    def __call__(self):
        """
        Return this multidimensional index as a valid numpy array index.

        Returns
        -------
        int
            This multidimensional index.
        """
        return self._npy_inds

    @property
    def src_ndim(self):
        """
        Return the number of source dimensions.

        Returns
        -------
        int
            The number of dimensions expected in the source array.
        """
        return self._arr.shape[-1]

    @property
    def shape(self):
        """
        Return the shape of the indices.

        Returns
        -------
        tuple
            The shape of the indices.
        """
        return self._arr.shape[:-1]

    def copy(self):
        """
        Copy this Indexer.

        Returns
        -------
        Indexer
            A copy of this Indexer.
        """
        return super().copy(self._arr)

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
        if shape is not None and len(shape) != self.src_ndim:
            raise ValueError(f"Can't set source shape to {shape} because indexer {self} expects "
                             f"{self.src_ndim} dimensions.")

        return super().set_src_shape(shape, dist_shape)

    def _check_bounds(self):
        """
        Check that indices are within the bounds of the source shape.
        """
        if self._src_shape is not None:
            for inds, s in zip(self._npy_inds, self._dist_shape):
                if np.any(inds >= s):
                    raise RuntimeError(f"Indexer {self} exceeds bounds for axis of dimension {s}.")
                negs = inds[inds < 0]
                if negs.size > 0 and np.any(negs < -s):
                    raise RuntimeError(f"Indexer {self} exceeds bounds for axis of dimension {s}.")

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
            self._shaped_inst = ShapedMultiIndexer(self._npy_inds, self.shape)
        except Exception:
            self._shaped_inst = None

        if self._src_shape is not None:
            self._shaped_inst.set_src_shape(self._src_shape)

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

    def to_json(self):
        """
        Return a JSON serializable version of self.

        Returns
        -------
        list of int or int
            List or int version of self.
        """
        return self.as_array().tolist()

    def _appears_flat(self):
        return len(self._npy_inds) == 1


class IndexMaker(object):
    """
    A Factory for Indexer objects.
    """

    def __call__(self, idx, src_shape=None, flat=None, new_style=True):
        """
        Return an Indexer instance based on the passed indices/slices.

        Parameters
        ----------
        idx : int, ndarray, slice, or tuple
            Some sort of index/indices/slice.
        src_shape : tuple or None
            Source shape if known.
        flat : bool or None
            If True, indices are into a flat source.
        new_style : bool
            If True, indexer follows behavior of numpy indexing.

        Returns
        -------
        Indexer
            The Indexer instance we created based on the args.
        """
        if idx is ...:
            idxer = EllipsisIndexer((idx,), flat_src=flat)
        elif isinstance(idx, int):
            idxer = IntIndexer(idx, flat_src=flat)
        elif isinstance(idx, slice):
            idxer = SliceIndexer(idx, flat_src=flat)

        elif isinstance(idx, tuple):
            if idx and not new_style:
                mat = np.atleast_1d(idx)
                if mat.dtype == int and mat.ndim == 1:
                    idxer = ArrayIndexer(mat, flat_src=flat)
                else:
                    idxer = ListOfTuplesArrayIndexer(mat, flat_src=flat)
            else:
                multi = len(idx) > 1
                for i in idx:
                    if i is ...:
                        multi = len(idx) > 2  # ... doesn't count toward limit of dimensions
                        idxer = EllipsisIndexer(idx, flat_src=flat)
                        break
                else:
                    idxer = MultiIndexer(idx, flat_src=flat)
                if flat and multi:
                    raise RuntimeError("Can't use multdimensional index into a flat source.")
        else:
            arr = np.atleast_1d(idx)
            if flat or new_style or arr.ndim == 1:
                idxer = ArrayIndexer(arr, flat_src=flat)
            else:
                idxer = ListOfTuplesArrayIndexer(arr, flat_src=flat)

        if src_shape is not None:
            if flat:
                src_shape = (np.product(src_shape),)
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


def to_numpy_style(inds):
    """
    Convert old style OpenMDAO-specific indices into numpy-like indices.

    Parameters
    ----------
    inds : list of tuples
        Old style indices.

    Returns
    -------
    tuple of ndarrays
        New style indices.
    """
    inds = indexer(inds, new_style=False)
    if isinstance(inds, ListOfTuplesArrayIndexer):
        return tuple(i.tolist() for i in inds._npy_inds)
    elif isinstance(inds, ShapedArrayIndexer):
        return inds().tolist()
    raise TypeError("Indices are not old-style format.")


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


# TODO: remove this when the 'new_style' src_indices arg is removed
def _update_new_style(src_indices, new_style, prefix=""):
    if not new_style:
        if isinstance(src_indices, tuple):
            for part in src_indices:
                if part is ... or isinstance(part, slice):
                    return True
            warn_deprecation(f"{prefix}: 'src_indices={src_indices}' is specified in"
                             " a deprecated format. In a future release, 'src_indices'"
                             " will be expected to use NumPy array indexing, so replace the "
                             f"existing src_indices with {to_numpy_style(src_indices)}.")

    return new_style
