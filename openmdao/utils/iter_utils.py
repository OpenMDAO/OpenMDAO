"""
Various utilities for working with iterators.
"""


def meta2item_iter(meta_iter, item):
    """
    Convert a metadata iterator to an iterator over (name, <item>).

    Parameters
    ----------
    meta_iter : iter of (name, meta)
        Iterator over variable names and their metadata dicts.
    item : str
        The name of the item to extract from the metadata.

    Yields
    ------
    tuple
        Tuple of (name, <item>) for each variable.
    """
    for name, meta in meta_iter:
        yield name, meta[item]


def meta2items_iter(meta_iter, items):
    """
    Convert a metadata iterator to an iterator over [name, meta[item0], meta[item1], ...].

    Parameters
    ----------
    meta_iter : iter of (name, meta)
        Iterator over variable names and their metadata dicts.
    items : list of str
        The names of the items to extract from the metadata.

    Yields
    ------
    list
        [name, meta[item0], meta[item1], ...] for each variable.
    """
    for name, meta in meta_iter:
        to_yield = [name]
        for item in items:
            to_yield.append(meta[item])
        yield to_yield


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
    start = end = 0
    for name, size in size_iter:
        end += size
        yield name, (start, end)
        start = end


def meta2range_iter(meta_iter, size_name='size', subset=None):
    """
    Iterate over variables and their ranges, based on size metadata for each variable.

    Parameters
    ----------
    meta_iter : iterator over (name, meta)
        Iterator over variable name and metadata (which contains size information).
    size_name : str
        Name of the size metadata entry.  Defaults to 'size', but could also be 'global_size'.
    subset : iter of str or None
        If not None, restrict the ranges to those variables contained in subset.

    Yields
    ------
    str
        Name of variable.
    int
        Starting index.
    int
        Ending index.
    """
    start = end = 0

    if subset is None:
        for name, meta in meta_iter:
            end += meta[size_name]
            yield name, start, end
            start = end
    else:
        if not isinstance(subset, (set, dict)):
            subset = set(subset)

        seen = set()
        for name, meta in meta_iter:
            end += meta[size_name]
            if name in subset:
                yield name, start, end
            seen.add(name)
            start = end

        if subset - seen:
            raise KeyError(f"In meta2range_iter, subset members {sorted(subset - seen)} not found.")
