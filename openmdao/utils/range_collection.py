
from openmdao.utils.array_utils import shape_to_len


class RangeTree(object):
    """
    A binary search tree of ranges, mapping a name to an index range.

    Allows for fast lookup of the name corresponding to a given index. The ranges must be
    contiguous, but they can be of different sizes.

    Search complexity is O(log2 n). Better than FlatRangeMapper when total array size is large.
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
        self.size = ranges[-1][2] - ranges[0][1]
        self._rangedict = {}
        self.root = RangeTree.build(ranges, self._rangedict)

    def get_range(self, name):
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
        return self._rangedict[name]
            
    def find_name(self, idx):
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
    
    def find_name_and_rel_ind(self, idx):
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

    @staticmethod
    def build(ranges, rngdict):
        """
        Build a binary search tree to map indices to variable names.

        Parameters
        ----------
        ranges : list of (name, start, stop)
            List of (name, start, stop) tuples, where name is the variable name and start and stop
            define the range of indices for that variable. Ranges must be contiguous.
        rngdict : dict
            Dictionary to be populated with the ranges, keyed by name.

        Returns
        -------
        RangeTreeNode
            Root node of the binary search tree.
        """
        half = len(ranges) // 2
        name, start, stop = ranges[half]

        node = RangeTreeNode(name, start, stop)
        rngdict[name] = (start, stop)

        left_slices = ranges[:half]
        if left_slices:
            node.left = RangeTree.build(left_slices, rngdict)

        right_slices = ranges[half + 1:]
        if right_slices:
            node.right = RangeTree.build(right_slices, rngdict)

        return node

    def compute_transfer_inds(self, target_mapper, sources, targets):
        """
        Compute the transfer indices for the given sources and targets.

        Parameters
        ----------
        target_mapper : NameRangeMapper
            The NameRangeMapper for the target side of the transfer.
        sources : list of str
            List of source names.
        targets : list of str
            List of target names.

        Returns
        -------
        tuple of (dict, dict)
            A tuple of (src_inds, tgt_inds), where src_inds is a dict mapping source names to
            lists of indices and tgt_inds is a dict mapping target names to lists of indices.
        """
        pass    


class RangeTreeNode(object):
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


# if the total array size is less than this, we'll just use a flat list mapping
# indices to names instead of a binary search tree
_MAX_FLAT_RANGE_SIZE = 10000


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


class FlatRangeMapper(object):
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
        self.size = ranges[-1][2] - ranges[0][1]
        self.ranges = [None] * self.size
        for rng in ranges:
            _, start, stop = rng
            self.ranges[start:stop] = [rng] * (stop - start)

    def find_name(self, idx):
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

    def find_name_and_rel_ind(self, idx):
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


def NameRangeMapper(ranges):
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
        rname, rind = rtree.find_name_and_rel_ind(i)
        fname, find = flat.find_name_and_rel_ind(i)
        assert rname == fname and rind == find, 'i = %d, rname = %s, rind = %s, fname = %s, find = %s' % (i, rname, rind, fname, find)
        print(i, rname, rind, fname, find)


