
from openmdao.utils.array_utils import shape_to_len


# if the total array size is less than this, we'll just use a flat list mapping
# indices to names instead of a binary search tree
_MAX_FLAT_RANGE_SIZE = 10000


class NameRangeMapper(object):
    def __init__(self, ranges):
        self._name2range = {}
        self._range2name = {}
        self.size = ranges[-1][2] - ranges[0][1]

    @staticmethod
    def create(ranges):
        """
        Return a mapper that maps indices to variable names and relative indices.

        Parameters
        ----------
        ranges : list of (name, start, stop)
            Ordered list of (name, start, stop) tuples, where start and stop define the range of
            indices for that name. Ranges must be contiguous.

        Returns
        -------
        FlatRangeMapper or RangeTree
            A mapper that maps indices to variable names and relative indices.
        """
        size = ranges[-1][2] - ranges[0][1]
        return FlatRangeMapper(ranges) if size < _MAX_FLAT_RANGE_SIZE else RangeTree(ranges)

    def add_range(self, name, start, stop):
        """
        Add a range to the mapper.

        Parameters
        ----------
        name : str
            Name of the variable.
        start : int
            Starting index of the variable.
        stop : int
            Ending index of the variable.
        """
        self._name2range[name] = (start, stop)
        self._range2name[(start, stop)] = name

    def name2range(self, name):
        """
        Get the range corresponding to the given name.

        Parameters
        ----------
        name : str
            The name of the variable.

        Returns
        -------
        tuple of (int, int)
            The range of indices corresponding to the given name.
        """
        return self._name2range[name]

    def index2name(self, idx):
        """
        Find the name corresponding to the given index.

        Parameters
        ----------
        idx : int
            The index into the full array.

        Returns
        -------
        str or None
            The name corresponding to the given index, or None if not found.
        """
        raise NotImplementedError("index2name method must be implemented by subclass.")

    def index2names(self, idxs):
        """
        Find the names corresponding to the given indices.

        Parameters
        ----------
        idxs : list of int
            The indices into the full array.

        Returns
        -------
        list of str
            The names corresponding to the given indices.
        """
        names = {self.index2name(idx) for idx in idxs}
        if None in names:
            missing = []
            for idx in idxs:
                if self.index2name(idx) is None:
                    missing.append(idx)
            raise RuntimeError("Indices %s are not in any range." % sorted(missing))

        return names


class RangeTree(NameRangeMapper):
    """
    A binary search tree of ranges, mapping a name to an index range.

    Allows for fast lookup of the name corresponding to a given index. The ranges must be
    contiguous, but they can be of different sizes.

    Search complexity is O(log2 n). Uses less memory than FlatRangeMapper when total array size is
    large.
    """
    def __init__(self, ranges):
        """
        Initialize a RangeTree.

        Parameters
        ----------
        ranges : list of (name, start, stop)
            List of (name, start, stop) tuples, where name is the variable name and start and stop
            define the range of indices for that variable.
        """
        super().__init__(ranges)
        self.size = ranges[-1][2] - ranges[0][1]
        self.root = self.build(ranges)

    def index2name(self, idx):
        """
        Find the name corresponding to the given index.

        Parameters
        ----------
        idx : int
            The index into the full array.

        Returns
        -------
        str or None
            The name corresponding to the given index, or None if not found.
        """
        node = self.root
        while node is not None:
            if idx < node.start:
                node = node.left
            elif idx >= node.stop:
                node = node.right
            else:
                return node.name

    def index2name_and_rel_ind(self, idx):
        """
        Find the name and relative index corresponding to the matched range.

        Parameters
        ----------
        idx : int
            The index into the full array.

        Returns
        -------
        str or None
            The name corresponding to the matched range, or None if not found.
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
                return node.name, idx - node.start

        return None, None

    def build(self, ranges):
        """
        Build a binary search tree to map indices to variable names.

        Parameters
        ----------
        ranges : list of (name, start, stop)
            List of (name, start, stop) tuples, where name is the variable name and start and stop
            define the range of indices for that variable. Ranges must be contiguous.

        Returns
        -------
        RangeTreeNode
            Root node of the binary search tree.
        """
        half = len(ranges) // 2
        name, start, stop = ranges[half]

        node = RangeTreeNode(name, start, stop)
        self.add_range(name, start, stop)

        left_slices = ranges[:half]
        if left_slices:
            node.left = self.build(left_slices)

        right_slices = ranges[half + 1:]
        if right_slices:
            node.right = self.build(right_slices)

        return node


class RangeTreeNode(NameRangeMapper):
    """
    A node in a binary search tree of ranges, mapping a name to an index range.

    Parameters
    ----------
    name : str
        Name of the variable.
    start : int
        Starting index of the variable.
    stop : int
        Ending index of the variable.

    Attributes
    ----------
    name : str
        Name of the variable.
    start : int
        Starting index of the variable.
    stop : int
        Ending index of the variable.
    left : RangeTreeNode or None
        Left child node.
    right : RangeTreeNode or None
        Right child node.
    """

    __slots__ = ['name', 'start', 'stop', 'left', 'right']

    def __init__(self, name, start, stop):
        """
        Initialize a RangeTreeNode.
        """
        self.name = name
        self.start = start
        self.stop = stop
        self.left = None
        self.right = None


class FlatRangeMapper(NameRangeMapper):
    """
    A flat list mapping indices to variable names and relative indices.

    Parameters
    ----------
    ranges : list of (name, start, stop)
        Ordered list of (name, start, stop) tuples, where start and stop define the range of
        indices for that name. Ranges must be contiguous.

    Attributes
    ----------
    size : int
        Total size of all of the ranges combined.
    ranges : list of (name, start, stop)
        List of (name, start, stop) tuples, where start and stop define the range of
        indices for that name. Ranges must be contiguous.
    """

    def __init__(self, ranges):
        """
        Initialize a FlatRangeMapper.
        """
        super().__init__(ranges)
        self.ranges = [None] * self.size
        for rng in ranges:
            name, start, stop = rng
            self.ranges[start:stop] = [rng] * (stop - start)
            self.add_range(name, start, stop)

    def index2name(self, idx):
        """
        Find the name corresponding to the given index.

        Parameters
        ----------
        idx : int
            The index into the full array.

        Returns
        -------
        str or None
            The name corresponding to the given index, or None if not found.
        """
        try:
            return self.ranges[idx][0]
        except IndexError:
            return None

    def index2name_and_rel_ind(self, idx):
        """
        Find the name and relative index corresponding to the matched range.

        Parameters
        ----------
        idx : int
            The index into the full array.

        Returns
        -------
        str or None
            The name corresponding to the matched range, or None if not found.
        int or None
            The relative index into the matched range, or None if not found.
        """
        try:
            name, start, _ = self.ranges[idx]
        except IndexError:
            return (None, None)

        return (name, idx - start)


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
        rname, rind = rtree.index2name_and_rel_ind(i)
        fname, find = flat.index2name_and_rel_ind(i)
        assert rname == fname and rind == find, f'i = {i}, rname = {rname}, rind = {rind}, fname = {fname}, find = {find}'
        print(i, rname, rind, fname, find)
