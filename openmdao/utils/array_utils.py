"""
Utils for dealing with arrays.
"""
from __future__ import print_function, division

import sys
import numpy as np
import six
from six.moves import range


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
    sizes = np.full(num_divisions, base, dtype=int)

    # evenly distribute the remainder across size-leftover procs,
    # instead of giving the whole remainder to one proc
    sizes[:leftover] += 1

    offsets = np.zeros(num_divisions, dtype=int)
    offsets[1:] = np.cumsum(sizes)[:-1]

    return sizes, offsets


def take_nth(rank, size, seq):
    """
    Iterate returning every nth value.

    Return an iterator over the sequence that returns every
    nth element of seq based on the given rank within a group of
    the given size.  For example, if size = 2, a rank of 0 returns
    even indexed elements and a rank of 1 returns odd indexed elements.

    Parameters
    ----------
    rank : int
        MPI rank of this process.
    size : int
        Size of the array we're taking nth entries from.
    seq : iter
        Iterator containing the values being returned.
    """
    assert(rank < size)
    it = iter(seq)
    while True:
        for proc in range(size):
            if rank == proc:
                yield six.next(it)
            else:
                six.next(it)


def convert_neg(arr, dim):
    """
    Convert any negative indices into their positive equivalent.

    Parameters
    ----------
    arr : ndarray
        Array having negative indices converted.
    dim : int
        Dimension of the array.

    Returns
    -------
    ndarray
        The converted array.
    """
    arr[arr < 0] += dim
    return arr


def array_viz(arr, stream=sys.stdout):
    """
    Display the structure of an array in a compact form.

    Parameters
    ----------
    arr : ndarray
        Array being visualized.
    stream : file-like
        Stream where output will be written.
    """
    if len(arr.shape) != 2:
        raise RuntimeError("array_viz only works for 2d arrays.")

    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            if arr[r, c] == 0.0:
                stream.write('.')
            else:
                stream.write('x')
        stream.write(' %d\n' % r)
