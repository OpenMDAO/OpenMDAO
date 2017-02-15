"""Some miscellaneous utility functions."""
from __future__ import division

import warnings
from six import string_types

from collections import Iterable
import numbers

import numpy as np


def warn_deprecation(msg):
    """
    Raise a warning and prints a deprecation message to stdout.

    Parameters
    ----------
    msg : str
        Message that will be printed to stdout.
    """
    # Deprecation warnings need to be printed regardless of debug level
    warnings.simplefilter('always', DeprecationWarning)

    # note, stack level 3 should take us back to original caller.
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
    warnings.simplefilter('ignore', DeprecationWarning)


def make_compatible(name, value, shape=None, indices=None):
    """
    Make value compatible with the specified shape or the shape of indices.

    Parameters
    ----------
    value : str
        The name of the value.
    value : float or list or tuple or ndarray
        The value of a variable.
    shape : int or tuple or list or None
        The expected or desired shape of the value.
    indices : int or list of ints or tuple of ints or int ndarray or None
        The indices of a source variable, used to determine shape if shape is None.
        If shape is not None, the shape of the indices must match shape.

    Returns
    -------
    ndarray
        The value in a shape compatible with the specified shape and/or indices.
    tuple
        The resulting shape of the value.

    Raises
    ------
    ValueError
        If value cannot be made to conform to shape or if shape and indices
        are incompatible.
    """
    # if shape is not given, infer from indices
    if shape:
        if isinstance(shape, int):
            shape = (shape,)
        elif isinstance(shape, list):
            shape = tuple(shape)
    elif indices is not None:
        shape = np.atleast_1d(indices).shape

    if shape is None:
        # shape is not determined, assume the shape of value was intended
        value = np.atleast_1d(value)
        shape = value.shape
    else:
        # shape is determined, if value is scalar assign it to array of shape
        # otherwise make sure value is an array of the determined shape
        if np.isscalar(value):
            value = np.ones(shape) * value
        else:
            value = np.atleast_1d(value)
            if value.shape != shape:
                raise ValueError("Incompatible shape for '%s': "
                                 "Expected %s but got %s." %
                                 (name, shape, value.shape))

    return value, shape


def format_as_float_or_array(name, values, val_if_none=0.0):
    """
    Format values as a numpy array.

    Checks that the given array values are either None, float, or an
    iterable of numeric values.  On output all iterables of numeric values
    are converted to numpy.ndarray.  If values is scalar, it is converted
    to float.

    Parameters
    ----------
    name : str
        Name of the values being formatted
    values : float or numpy ndarray or Iterable
        Values to be formatted to the expected form.
    val_if_none : bool
        Value to return if values is None

    Returns
    -------
    float or numpy.ndarray
        Values transformed to the expected form.

    Raises
    ------
    ValueError
        If values is Iterable but cannot be converted to a numpy ndarray
    TypeError
        If values is scalar, not None, and not a Number.
    """
    # Convert to ndarray/float as necessary
    if isinstance(values, np.ndarray):
        pass
    elif not isinstance(values, string_types) \
            and isinstance(values, Iterable):
        values = np.asarray(values, dtype=float)
    elif values is None:
        values = val_if_none
    elif isinstance(values, numbers.Number):
        values = float(values)
    else:
        raise TypeError('Expected values of {0} to be an Iterable of '
                        'numeric values, or a scalar numeric value. '
                        'Got {1} instead.'.format(name, values))
    return values
