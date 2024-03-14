"""
Various utilities for working with iterators.
"""


def meta2item_iter(metaiter, item):
    """
    Convert a metadata iterator to an iterator over (name, <item>).

    Parameters
    ----------
    metaiter : iter of (name, meta)
        Iterator over variable names and their metadata dicts.
    item : str
        The item to extract from the metadata.

    Yields
    ------
    tuple
        Tuple of (name, <item>) for each variable.
    """
    for name, meta in metaiter:
        yield name, meta[item]


def meta2items_iter(metaiter, items):
    """
    Convert a metadata iterator to an iterator over [name, meta[item0], meta[item1], ...].

    Parameters
    ----------
    metaiter : iter of (name, meta)
        Iterator over variable names and their metadata dicts.
    items : list of str
        The items to extract from the metadata.

    Yields
    ------
    list
        [name, meta[item0], meta[item1], ...] for each variable.
    """
    for name, meta in metaiter:
        toyield = [name]
        for item in items:
            toyield.append(meta[item])
        yield toyield


def size2range_iter(size_iter):
    """
    Convert a size iterator to a range iterator.

    Parameters
    ----------
    size_iter : iter of (name, size)
        Iterator over variable names and their sizes.

    Yields
    ------
    tuple
        Tuple of (name, (start, end)) for each variable.
    """
    start = 0
    for name, size in size_iter:
        yield name, (start, start + size)
        start += size
