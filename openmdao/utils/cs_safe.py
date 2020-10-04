"""
collection of complex-step safe functions to replace standard numpy operations.
"""

import numpy as np


def abs(x):
    """
    complex-step safe version of numpy.abs function.

    Parameters
    ----------
    x: ndarray
        array value to be computed on

    Returns
    -------
    ndarray
    """
    if isinstance(x, np.ndarray):
        return x * np.sign(x)
    elif x.real < 0.0:
        return -x
    return x


def norm(x, axis=None):
    """
    complex-step safe version of numpy.linalg.norm function.

    Parameters
    ----------
    x: ndarray
        array value to be computed on
    axis: None or int
        axis to perform the norm up to

    Returns
    -------
    ndarray
    """
    return np.sqrt(np.sum(x**2, axis=axis))
