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


def make_compatible(meta, value):
    """
    Make value compatible with the variable described in metadata.

    Parameters
    ----------
    meta : dict
        metadata for a variable.
    value : float or ndarray or list
        value.

    Returns
    -------
    ndarray
        value in a form compatible with the specified metadata.
    """
    tgt_shape = meta['shape']
    if np.isscalar(value):
        value = np.ones(tgt_shape) * value
    else:
        value = np.atleast_1d(value)
    if value.shape != tgt_shape:
        raise ValueError("Incompatible shape for assignment. "
                         "Expected %s but got %s." %
                         (tgt_shape, value.shape))
    return value


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
