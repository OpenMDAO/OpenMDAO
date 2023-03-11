"""
collection of complex-step safe functions to replace standard numpy operations.
"""
from .om_warnings import warn_deprecation
from ..func import abs as cs_abs, norm as cs_norm, arctan2 as cs_arctan2


def abs(x):
    """
    Complex-step safe version of numpy.abs function.

    Parameters
    ----------
    x : ndarray
        Array value to be computed on.

    Returns
    -------
    ndarray
        Absolute value.
    """
    warn_deprecation('openmdao.utils.cs_safe.abs is deprecated. Use openmdao.func.abs instead.')
    return cs_abs(x)


def norm(x, axis=None):
    """
    Complex-step safe version of numpy.linalg.norm function.

    Parameters
    ----------
    x : ndarray
        Array value to be computed on.
    axis : None or int
        Axis to perform the norm up to.

    Returns
    -------
    ndarray
        Matrix or vector norm.
    """
    warn_deprecation('openmdao.utils.cs_safe.norm is deprecated. Use openmdao.func.norm instead.')
    return cs_norm(x, axis=axis)


def arctan2(y, x):
    """
    Numpy-compatible, complex-compatible arctan2 function for use with complex-step.

    Parameters
    ----------
    y : float or complex
        The length of the side opposite the angle being determined.
    x : float or complex
        The length of the side adjacent to the angle being determined.

    Returns
    -------
    ndarray
        The angle whose opposite side has length y and whose adjacent side has length x.
    """
    warn_deprecation('openmdao.utils.cs_safe.arctan2 is deprecated. Use openmdao.func.arctan2 instead.')
    return cs_arctan2(y, x)
