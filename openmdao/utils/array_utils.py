"""
Utils for dealing with arrays.
"""

def evenly_distrib_idxs(num_divisions, arr_size):
    """
    Return evenly distributed entries for the given array size.

    Given a number of divisions and the size of an array, chop the array up
    into pieces according to number of divisions, keeping the distribution
    of entries as even as possible.

    Parameters
    ----------
    num_divisions: int
        Number of parts to divide the array into.
    arr_size: int
        Number of entries in the array.

    Returns
    -------
    tuple
        a tuple of (sizes, offsets), where sizes and offsets contain values for all
        divisions.
    """
    base = arr_size // num_divisions
    leftover = arr_size % num_divisions
    sizes = np.ones(num_divisions, dtype="int") * base

    # evenly distribute the remainder across size-leftover procs,
    # instead of giving the whole remainder to one proc
    sizes[:leftover] += 1

    offsets = np.zeros(num_divisions, dtype="int")
    offsets[1:] = np.cumsum(sizes)[:-1]

    return sizes, offsets

def to_slice(idxs):
    """Convert an index array to a slice if possible. Otherwise,
    return the index array. Indices are assumed to be sorted in
    ascending order.
    """
    if len(idxs) == 1:
        return slice(idxs[0], idxs[0]+1)
    elif len(idxs) == 0:
        return idxs

    stride = idxs[1]-idxs[0]

    if stride <= 0:
        return idxs

    #make sure stride is consistent throughout the array
    if any(idxs[1:]-idxs[:-1] != stride):
        return idxs

    # set the upper bound to idxs[-1]+stride instead of idxs[-1]+1 because
    # later, we compare upper and lower bounds when collapsing slices
    return slice(idxs[0], idxs[-1]+stride, stride)

