


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


