"""
A collection of classes for mapping indices to variable names and vice versa.
"""

from openmdao.utils.array_utils import shape_to_len


# default size of array for which we use a FlatRangeMapper instead of a RangeTree
_MAX_FLAT_RANGE_SIZE = 10000


class DataRangeMapper(object):
    """
    A mapper of indices to variable names and vice versa.

    Parameters
    ----------
    ranges : list of (data, start, stop)
        Ordered list of (data, start, stop) tuples, where start and stop define the range of
        indices for the data. Ranges must be contiguous.  data must be hashable.

    Attributes
    ----------
    size : int
        Total size of all of the ranges combined.
    _data2range : dict
        Dictionary mapping data to an index range.
    _range2data : dict
        Dictionary mapping an index range to data.
    """

    def __init__(self, ranges):
        """
        Initialize a DataRangeMapper.
        """
        self._data2range = {}
        self._range2data = {}
        self.size = ranges[-1][2] - ranges[0][1]

    @staticmethod
    def create(ranges, max_flat_range_size=_MAX_FLAT_RANGE_SIZE):
        """
        Return a mapper that maps indices to variable names and relative indices.

        Parameters
        ----------
        ranges : list of (data, start, stop)
            Ordered list of (data, start, stop) tuples, where start and stop define the range
            of indices for the data. Ranges must be contiguous.
        max_flat_range_size : int
            If the total array size is less than this, a FlatRangeMapper will be returned instead
            of a RangeTree.

        Returns
        -------
        FlatRangeMapper or RangeTree
            A mapper that maps indices to variable data and relative indices.
        """
        size = ranges[-1][2] - ranges[0][1]
        return FlatRangeMapper(ranges) if size < max_flat_range_size else RangeTree(ranges)

    def add_range(self, data, start, stop):
        """
        Add a range to the mapper.

        Parameters
        ----------
        data : object (must be hashable)
            Data corresponding to an index range.
        start : int
            Starting index of the variable.
        stop : int
            Ending index of the variable.
        """
        self._data2range[data] = (start, stop)
        self._range2data[start, stop] = data

    def data2range(self, data):
        """
        Get the range corresponding to the given name and rank.

        Parameters
        ----------
        data : object (must be hashable)
            Data corresponding to an index range.

        Returns
        -------
        tuple of (int, int)
            The range of indices corresponding to the given data.
        """
        return self._data2range[data]

    def _index2data(self, idx):
        """
        Find the data corresponding to the given index.

        Parameters
        ----------
        idx : int
            The index into the full array.

        Returns
        -------
        object or None
            The data corresponding to the given index, or None if not found.
        """
        raise NotImplementedError("_index2data method must be implemented by subclass.")

    def __getitem__(self, idx):
        """
        Find the data corresponding to the given index.

        Parameters
        ----------
        idx : int
            The index into the full array.
        """
        return self._index2data(idx)

    def indices2data(self, idxs):
        """
        Find the data objects corresponding to the given indices.

        Parameters
        ----------
        idxs : list of int
            The indices into the full array.

        Returns
        -------
        list of (object, int)
            The data corresponding to each of the given indices.
        """
        data = [self._index2data(idx) for idx in idxs]
        if None in data:
            missing = []
            for idx in idxs:
                d = self._index2data(idx)
                if d is None:
                    missing.append(idx)
            raise RuntimeError(f"Indices {sorted(missing)} are not in any range.")

        return data


class RangeTreeNode(DataRangeMapper):
    """
    A node in a binary search tree of ranges, mapping data to an index range.

    Parameters
    ----------
    data : object
        Data corresponding to an index range.
    start : int
        Starting index of the variable.
    stop : int
        Ending index of the variable.

    Attributes
    ----------
    data : object
        Data corresponding to an index range.
    start : int
        Starting index of the variable.
    stop : int
        Ending index of the variable.
    left : RangeTreeNode or None
        Left child node.
    right : RangeTreeNode or None
        Right child node.
    """

    __slots__ = ['data', 'start', 'stop', 'left', 'right']

    def __init__(self, data, start, stop):
        """
        Initialize a RangeTreeNode.
        """
        self.data = data
        self.start = start
        self.stop = stop
        self.left = None
        self.right = None


class RangeTree(DataRangeMapper):
    """
    A binary search tree of ranges, mapping data to an index range.

    Allows for fast lookup of the data corresponding to a given index. The ranges must be
    contiguous, but they can be of different sizes.

    Search complexity is O(log2 n). Uses less memory than FlatRangeMapper when total array size is
    large.

    Parameters
    ----------
    ranges : list of (data, start, stop)
        Ordered list of (data, start, stop) tuples, where start and stop define the range of
        indices for the data. Ranges must be contiguous.  data must be hashable.

    Attributes
    ----------
    size : int
        Total size of all of the ranges combined.
    root : RangeTreeNode
        Root node of the binary search tree.
    """

    def __init__(self, ranges):
        """
        Initialize a RangeTree.
        """
        super().__init__(ranges)
        self.size = ranges[-1][2] - ranges[0][1]
        self.root = self.build(ranges)

    def _index2data(self, idx):
        """
        Find the data corresponding to the given index.

        Parameters
        ----------
        idx : int
            The index into the full array.

        Returns
        -------
        object or None
            The data corresponding to the given index, or None if not found.
        int or None
            The rank corresponding to the given index, or None if not found.
        """
        node = self.root
        while node is not None:
            if idx < node.start:
                node = node.left
            elif idx >= node.stop:
                node = node.right
            else:
                return node.data

    def index2rel_data(self, idx):
        """
        Find the data and relative index corresponding to the matched range.

        Parameters
        ----------
        idx : int
            The index into the full array.

        Returns
        -------
        obj or None
            The data corresponding to the matched range, or None if not found.
        int or None
            The relative index into the matched range, or None if not found.
        """
        node = self.root
        while node is not None:
            if idx < node.start:
                node = node.left
            elif idx >= node.stop:
                node = node.right
            else:
                return node.data, idx - node.start

        return None, None

    def build(self, ranges):
        """
        Build a binary search tree to map indices to variable data.

        Parameters
        ----------
        ranges : list of (data, start, stop)
            List of (data, start, stop) tuples, where start and stop
            define the range of indices for the data. Ranges must be contiguous.
            data must be hashable.

        Returns
        -------
        RangeTreeNode
            Root node of the binary search tree.
        """
        half = len(ranges) // 2
        data, start, stop = ranges[half]

        node = RangeTreeNode(data, start, stop)
        self.add_range(data, start, stop)

        left_slices = ranges[:half]
        if left_slices:
            node.left = self.build(left_slices)

        right_slices = ranges[half + 1:]
        if right_slices:
            node.right = self.build(right_slices)

        return node


class FlatRangeMapper(DataRangeMapper):
    """
    A flat list mapping indices to variable data and relative indices.

    Parameters
    ----------
    ranges : list of (data, start, stop)
        Ordered list of (data, start, stop) tuples, where start and stop define the range of
        indices for that data. Ranges must be contiguous.  data must be hashable.

    Attributes
    ----------
    ranges : list of (data, start, stop)
        List of (data, start, stop) tuples, where start and stop define the range of
        indices for that data. Ranges must be contiguous. data must be hashable.
    """

    def __init__(self, ranges):
        """
        Initialize a FlatRangeMapper.
        """
        super().__init__(ranges)
        self.ranges = [None] * self.size
        for rng in ranges:
            data, start, stop = rng
            self.ranges[start:stop] = [rng] * (stop - start)
            self.add_range(data, start, stop)

    def _index2data(self, idx):
        """
        Find the data corresponding to the given index.

        Parameters
        ----------
        idx : int
            The index into the full array.

        Returns
        -------
        object or None
            The data corresponding to the given index, or None if not found.
        """
        try:
            return self.ranges[idx][0]
        except IndexError:
            return None

    def index2rel_data(self, idx):
        """
        Find the data and relative index corresponding to the matched range.

        Parameters
        ----------
        idx : int
            The index into the full array.

        Returns
        -------
        object or None
            The data corresponding to the matched range, or None if not found.
        int or None
            The relative index into the matched range, or None if not found.
        """
        try:
            data, start, _ = self.ranges[idx]
        except IndexError:
            return (None, None)

        return (data, idx - start)


def metas2ranges(meta_iter, shape_name='shape'):
    """
    Convert an iterator of metadata to an iterator of (name, start, stop) tuples.

    Parameters
    ----------
    meta_iter : iterator of (name, meta)
        Iterator of (name, meta) tuples, where name is the variable name and meta is the
        corresponding metadata dictionary.
    shape_name : str
        Name of the metadata entry that contains the shape of the variable. Value can be either
        'shape' or 'global_shape'.  Default is 'shape'.  The value of the metadata entry must
        be a tuple of integers.

    Yields
    ------
    tuple
        Tuple of the form (name, start, stop), where name is the variable name, start is the start
        of the variable range, and stop is the end of the variable range.
    """
    start = stop = 0
    for name, meta in meta_iter:
        stop += shape_to_len(meta[shape_name])
        yield (name, start, stop)
        start = stop


def metas2shapes(meta_iter, shape_name='shape'):
    """
    Convert an iterator of metadata to an iterator of (name, shape) tuples.

    Parameters
    ----------
    meta_iter : iterator of (name, meta)
        Iterator of (name, meta) tuples, where name is the variable name and meta is the
        corresponding metadata dictionary.
    shape_name : str
        Name of the metadata entry that contains the shape of the variable. Value can be either
        'shape' or 'global_shape'.  Default is 'shape'.  The value of the metadata entry must
        be a tuple of integers.

    Yields
    ------
    tuple
        Tuple of the form (name, shape), where name is the variable name and shape is the shape
        of the variable.
    """
    for name, meta in meta_iter:
        yield (name, meta[shape_name])


if __name__ == '__main__':
    meta = {
        'x': {'shape': (2, 3)},
        'y': {'shape': (4, 5)},
        'z': {'shape': (6,)},
    }

    print(list(metas2shapes(meta.items())))

    ranges = list(metas2ranges(meta.items()))
    print(ranges)

    rtree = RangeTree(ranges)
    flat = FlatRangeMapper(ranges)

    for i in range(34):
        rname, rind = rtree.index2rel_data(i)
        fname, find = flat.index2rel_data(i)
        print(i, rname, rind, fname, find)
