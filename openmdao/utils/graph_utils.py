"""
Various graph related utilities.
"""
import networkx as nx


def get_sccs_topo(graph):
    """
    Return strongly connected subsystems of the given Group in topological order.

    Parameters
    ----------
    graph : networkx.DiGraph
        Directed graph of Systems.

    Returns
    -------
    list of sets of str
        A list of strongly connected components in topological order.
    """
    # Tarjan's algorithm returns SCCs in reverse topological order, so
    # the list returned here is reversed.
    sccs = list(nx.strongly_connected_components(graph))
    sccs.reverse()
    return sccs


class RangeTree(object):
    """
    A binary search tree of ranges, mapping a name to an index range.

    Search complexity is O(log2 n).

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
    left : RangeTree or None
        Left child node.
    right : RangeTree or None
        Right child node.
    """

    __slots__ = ['name', 'start', 'stop', 'left', 'right']

    def __init__(self, name, start, stop):
        """
        Initialize a RangeTree.
        """
        self.name = name
        self.start = start
        self.stop = stop
        self.left = None
        self.right = None

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
        if idx < self.start:
            return None if self.left is None else self.left.find_name(idx)

        if idx >= self.stop:
            return None if self.right is None else self.right.find_name(idx)

        return self.name

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
        if idx < self.start:
            return (None, None) if self.left is None else self.left.find_name_and_rel_ind(idx)

        if idx >= self.stop:
            return (None, None) if self.right is None else self.right.find_name_and_rel_ind(idx)

        return (self.name, idx - self.start)

    @staticmethod
    def build(ranges):
        """
        Build a binary search tree to map indices to variable names.

        Parameters
        ----------
        ranges : list of (name, start, stop)
            List of (name, start, stop) tuples, where name is the variable name and start and stop
            define the range of indices for that variable.

        Returns
        -------
        RangeTree
            Root node of the binary search tree.
        """
        half = len(ranges) // 2
        name, start, stop = ranges[half]

        node = RangeTree(name, start, stop)

        left_slices = ranges[:half]
        right_slices = ranges[half + 1:]
        if left_slices:
            node.left = RangeTree.build(left_slices)
        if right_slices:
            node.right = RangeTree.build(right_slices)

        return node


# if the total array size is less than this, we'll just use a flat list mapping
# indices to names instead of a binary search tree
_MAX_RANGE_SIZE = 10000


def NameRangeMapper(ranges):
    """
    Return a mapper that maps indices to variable names and relative indices.

    Parameters
    ----------
    ranges : list of (name, start, stop)
        Ordered list of (name, start, stop) tuples, where start and stop define the range of
        indices for that name.

    Returns
    -------
    FlatRangeMapper or RangeTree
        A mapper that maps indices to variable names and relative indices.
    """
    size = ranges[-1][2] - ranges[0][1]
    if size < _MAX_RANGE_SIZE:
        return FlatRangeMapper(ranges)

    return RangeTree.build(ranges)


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
        indices for that name.
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
