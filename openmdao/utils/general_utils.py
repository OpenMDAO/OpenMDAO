"""Some miscellaneous utility functions."""

import warnings

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

    Args
    ----
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
