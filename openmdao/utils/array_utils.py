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


def array_viz(arr, prob=None, of=None, wrt=None, stream=sys.stdout):
    """
    Display the structure of a boolean array in a compact form.

    If prob, of, and wrt are supplied, print the name of the response alongside
    each row and print the names of the design vars, aligned with each column, at
    the bottom.

    Parameters
    ----------
    arr : ndarray
        Array being visualized.
    prob : Problem or None
        Problem object.
    of : list of str or None
        Names of response variables used in derivative calculation.
    wrt : list of str or None
        Names of design variables used in derivative calculation.
    stream : file-like
        Stream where output will be written.
    """
    if len(arr.shape) != 2:
        raise RuntimeError("array_viz only works for 2d arrays.")

    if prob is None or of is None or wrt is None:
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                if arr[r, c]:
                    stream.write('x')
                else:
                    stream.write('.')
            stream.write(' %d\n' % r)
    else:

        row = 0
        for res in of:
            for r in range(row, row + prob.driver._responses[res]['size']):
                col = 0
                for dv in wrt:
                    for c in range(col, col + prob.driver._designvars[dv]['size']):
                        if arr[r, c]:
                            stream.write('x')
                        else:
                            stream.write('.')
                    col = c + 1
                stream.write(' %d  %s\n' % (r, res))
            row = r + 1

        start = 0
        for name in wrt:
            tab = ' ' * start
            stream.write('%s|%s\n' % (tab, name))
            start += prob.driver._designvars[name]['size']


def array_connection_compatible(shape1, shape2):
    """
    Return True if the two arrays shapes are compatible.

    Array shapes are compatible if the underlying data has the same size and is
    stored in the same contiguous order for the two shapes.

    Parameters
    ----------
    shape1 : tuple of int
        Shape of the first array.
    shape2 : tuple of int
        Shape of the second array.

    Returns
    -------
    bool
        True if the two shapes are compatible for connection, else False.
    """
    ashape1 = np.asarray(shape1, dtype=int)
    ashape2 = np.asarray(shape2, dtype=int)

    size1 = np.prod(ashape1)
    size2 = np.prod(ashape2)

    # Shapes are not connection-compatible if size is different
    if size1 != size2:
        return False

    nz1 = np.where(ashape1 > 1)[0]
    nz2 = np.where(ashape2 > 1)[0]

    if len(nz1) > 0:
        fundamental_shape1 = ashape1[np.min(nz1): np.max(nz1) + 1]
    else:
        fundamental_shape1 = np.ones((1,))

    if len(nz2) > 0:
        fundamental_shape2 = ashape2[np.min(nz2): np.max(nz2) + 1]
    else:
        fundamental_shape2 = np.ones((1,))

    return np.all(fundamental_shape1 == fundamental_shape2)
