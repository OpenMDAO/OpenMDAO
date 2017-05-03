"""
Utils for dealing with arrays.
"""

import numpy as np


def evenly_distrib_idxs(num_divisions, arr_size):
    """
    Return evenly distributed entries for the given array size.

    Given a number of divisions and the size of an array, chop the array up
    into pieces according to number of divisions, keeping the distribution
    of entries as even as possible.

    Parameters
    ----------
    num_divisions : int
        Number of parts to divide the array into.
    arr_size : int
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
