"""
A collection of classes for mapping indices to variable names and vice versa.
"""

from openmdao.utils.array_utils import shape_to_len


# default size of array for which we use a FlatRangeMapper instead of a RangeTree
MAX_FLAT_RANGE_SIZE = 10000


class RangeMapper(object):
    """
    A mapper of indices to variable names and vice versa.

    Parameters
    ----------
    sizes : iterable of (key, size) tuples
        Iterable of (key, size) tuples.  key must be hashable.

    Attributes
    ----------
    size : int
        Total size of all of the sizes combined.
    _key2range : dict
        Dictionary mapping key to an index range.
    """

    def __init__(self, sizes):
        """
        Initialize a RangeMapper.
        """
        self._key2range = {}
        start = 0
        for key, size in sizes:
            self._key2range[key] = (start, start + size)
            start += size
        self.size = start

    @staticmethod
    def create(sizes, max_flat_range_size=MAX_FLAT_RANGE_SIZE):
        """
        Return a mapper that maps indices to variable names and relative indices.

        Parameters
        ----------
        sizes : iterable of (key, size)
            Iterable of (key, size) tuples.
        max_flat_range_size : int
            If the total array size is less than this, a FlatRangeMapper will be returned instead
            of a RangeTree.

        Returns
        -------
        FlatRangeMapper or RangeTree
            A mapper that maps indices to variable key and relative indices.
        """
        size = sum(size for _, size in sizes)
        return FlatRangeMapper(sizes) if size <= max_flat_range_size else RangeTree(sizes)

    def key2range(self, key):
        """
        Get the range corresponding to the given key.

        Parameters
        ----------
        key : object (must be hashable)
            Data corresponding to an index range.

        Returns
        -------
        tuple of (int, int)
            The range of indices corresponding to the given key.
        """
        return self._key2range[key]

    def key2size(self, key):
        """
        Get the size corresponding to the given key.

        Parameters
        ----------
        key : object (must be hashable)
            Key corresponding to an index range.

        Returns
        -------
        int
            The size corresponding to the given key.
        """
        start, stop = self._key2range[key]
        return stop - start

    def __getitem__(self, idx):
        """
        Find the key corresponding to the given index.

        Parameters
        ----------
        idx : int
            The index into the full array.
        """
        raise NotImplementedError("__getitem__ method must be implemented by subclass.")

    def __iter__(self):
        """
        Iterate over (key, start, stop) tuples.

        Yields
        ------
        (obj, int, int)
            (key, start index, stop index), where key is a hashable object.
        """
        raise NotImplementedError("__getitem__ method must be implemented by subclass.")

    def inds2keys(self, inds):
        """
        Find the set of keys corresponding to the given indices.

        Parameters
        ----------
        inds : iter of int
            The array indices.

        Returns
        -------
        set of object
            The set of keys corresponding to the given indices.
        """
        return {self[idx] for idx in inds}

    def between_iter(self, start_key, stop_key):
        """
        Iterate over (key, start, stop) tuples between the given start and stop keys.

        Parameters
        ----------
        start_key : object
            Key corresponding to an index range.
        stop_key : object
            Key corresponding to an index range.

        Yields
        ------
        (obj, int, int)
            (key, relative start index, relative stop index), where key is a hashable object.
        """
        started = False
        for key, (start, stop) in self._key2range.items():
            if key == start_key:
                yield (key, 0, stop - start)
                if start_key == stop_key:
                    break
                started = True
            elif started:
                if key == stop_key:
                    yield (key, 0, stop - start)
                    break
                else:
                    yield (key, 0, stop - start)

    def overlap_iter(self, key, other):
        """
        Find the set of keys that overlap between this mapper and another.

        Parameters
        ----------
        key : object
            Key corresponding to an index range.
        other : RangeMapper
            Another mapper.

        Yields
        ------
        (obj, int, int, obj, int, int)
            (key, start, stop, otherkey, otherstart, otherstop).
        """
        start, stop = self._key2range[key]

        start_key, start_rel = other.index2key_rel(start)
        if start_key is None:
            return

        stop_key, stop_rel = other.index2key_rel(stop - 1)

        overlaps = [list(tup) for tup in other.between_iter(start_key, stop_key)]
        overlaps[0][1] = start_rel
        overlaps[-1][2] = stop_rel + 1

        start = stop = 0
        for k, kstart, kstop in overlaps:
            stop += kstop - kstart
            yield (key, start, stop, k, kstart, kstop)
            start = stop

    def dump(self):
        """
        Dump the contents of the mapper to stdout.
        """
        for key, (start, stop) in self._key2range.items():
            print(f'{key}: {start} - {stop}')


class RangeTreeNode(RangeMapper):
    """
    A node in a binary search tree of sizes, mapping key to an index range.

    Parameters
    ----------
    key : object
        Data corresponding to an index range.
    start : int
        Starting index of the variable.
    stop : int
        Ending index of the variable.

    Attributes
    ----------
    key : object
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

    __slots__ = ['key', 'start', 'stop', 'left', 'right']

    def __init__(self, key, start, stop):
        """
        Initialize a RangeTreeNode.
        """
        self.key = key
        self.start = start
        self.stop = stop
        self.left = None
        self.right = None

    def __repr__(self):
        """
        Return a string representation of the RangeTreeNode.
        """
        return f"RangeTreeNode({self.key}, ({self.start}:{self.stop}))"


class RangeTree(RangeMapper):
    """
    A binary search tree of sizes, mapping key to an index range.

    Allows for fast lookup of the key corresponding to a given index. The sizes must be
    contiguous, but they can be of different sizes.

    Search complexity is O(log2 n). Uses less memory than FlatRangeMapper when total array size is
    large.

    Parameters
    ----------
    sizes : list of (key, start, stop)
        Ordered list of (key, start, stop) tuples, where start and stop define the range of
        indices for the key. Ranges must be contiguous.  key must be hashable.

    Attributes
    ----------
    root : RangeTreeNode
        Root node of the binary search tree.
    """

    def __init__(self, sizes):
        """
        Initialize a RangeTree.
        """
        super().__init__(sizes)
        ranges = []
        start = stop = 0
        for key, size in sizes:
            stop += size
            ranges.append((key, start, stop))
            start = stop

        self.root = self.build(ranges)

    def __getitem__(self, idx):
        """
        Find the key corresponding to the given index.

        Parameters
        ----------
        idx : int
            The index into the full array.

        Returns
        -------
        object or None
            The key corresponding to the given index, or None if not found.
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
                return node.key

    def __iter__(self):
        """
        Iterate over (key, start, stop) tuples.

        Yields
        ------
        (obj, int, int)
            (key, start index, stop index), where key is a hashable object.
        """
        node = self.root
        stack = [[node, node.left, node.right]]
        while stack:
            sub = stack[-1]
            node, left, right = sub
            if left:
                stack.append([left, left.left, left.right])
                sub[1] = None  # zero left
            else:
                if right:
                    stack.append([right, right.left, right.right])
                    sub[2] = None  # zero right
                else:
                    stack.pop()
                yield (node.key, node.start, node.stop)

    def index2key_rel(self, idx):
        """
        Find the key and relative index corresponding to the matched range.

        Parameters
        ----------
        idx : int
            The index into the full array.

        Returns
        -------
        obj or None
            The key corresponding to the matched range, or None if not found.
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
                return node.key, idx - node.start

        return None, None

    def build(self, ranges):
        """
        Build a binary search tree to map indices to variable key.

        Parameters
        ----------
        ranges : list of (key, start, stop)
            List of (key, start, stop) tuples, where start and stop
            define the range of indices for the key. Ranges must be ordered and contiguous.
            key must be hashable.

        Returns
        -------
        RangeTreeNode
            Root node of the binary search tree.
        """
        mid = len(ranges) // 2

        key, start, stop = ranges[mid]

        node = RangeTreeNode(key, start, stop)

        left_slices = ranges[:mid]
        right_slices = ranges[mid + 1:]

        if left_slices:
            node.left = self.build(left_slices)

        if right_slices:
            node.right = self.build(right_slices)

        return node


class FlatRangeMapper(RangeMapper):
    """
    A flat list mapping indices to variable key and relative indices.

    Parameters
    ----------
    sizes : list of (key, size)
        Ordered list of (key, size) tuples.  key must be hashable.

    Attributes
    ----------
    ranges : list of (key, start, stop)
        List of (key, start, stop) tuples, where start and stop define the range of
        indices for that key. Ranges must be contiguous. key must be hashable.
    """

    def __init__(self, sizes):
        """
        Initialize a FlatRangeMapper.
        """
        super().__init__(sizes)
        self.ranges = [None] * self.size
        start = stop = 0
        for key, size in sizes:
            stop += size
            self.ranges[start:stop] = [(key, start, stop)] * size
            start = stop

    def __getitem__(self, idx):
        """
        Find the key corresponding to the given index.

        Parameters
        ----------
        idx : int
            The index into the full array.

        Returns
        -------
        object or None
            The key corresponding to the given index, or None if not found.
        """
        try:
            return self.ranges[idx][0]
        except IndexError:
            return None

    def __iter__(self):
        """
        Iterate over (key, start, stop) tuples.

        Yields
        ------
        (obj, int, int)
            (key, start index, stop index), where key is a hashable object.
        """
        for key, (start, stop) in self._key2range.items():
            yield (key, start, stop)

    def index2key_rel(self, idx):
        """
        Find the key and relative index corresponding to the matched range.

        Parameters
        ----------
        idx : int
            The index into the full array.

        Returns
        -------
        object or None
            The key corresponding to the matched range, or None if not found.
        int or None
            The relative index into the matched range, or None if not found.
        """
        try:
            key, start, _ = self.ranges[idx]
        except IndexError:
            return (None, None)

        return (key, idx - start)
