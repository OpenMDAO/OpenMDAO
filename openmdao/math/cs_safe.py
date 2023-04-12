"""
collection of complex-step safe functions to replace standard numpy operations.
"""
import os
from ..utils.general_utils import is_truthy


use_jax = os.environ.get('OPENMDAO_MATH_USE_JAX', 'False')


if is_truthy(use_jax):
    try:
        import jax.numpy as np
        from jax import config
        config.update("jax_enable_x64", True)
    except:
        raise EnvironmentError('Environment variable OPENMDAO_MATH_USE_JAX is True but jax is not available.')
else:
    import numpy as np

from .numpy import sum as om_sum, d_sum


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
    _x = np.asarray(x)
    return _x * np.sign(_x)


def d_abs(x):
    """
    Compute derivative of the absolute value function.

    Note this function is non-differentiable at x=0. This implementation
    returns zero for that value but strictly speaking it is undefined.

    Users wanting a fully differentiable implementation should use openmdao.math.smooth_abs,
    which sacrifices some accuracy near zero in exchange for being fully differentiable.

    Parameters
    ----------
    x : ndarray
        Array value to be computed on.

    Returns
    -------
    ndarray
        Derivative of absolute value wrt x with the shape of x preserved.
    """
    return np.sign(x)


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
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        a = np.real(y)
        b = np.imag(y)
        c = np.real(x)
        d = np.imag(x)
        return np.arctan2(a, c) + 1j * (c * b - a * d) / (a**2 + c**2)
    else:
        return np.arctan2(y, x)


def arctanh(x):
    """
    Numpy-compatible, complex-compatible arctanh function for use with complex-step.

    Parameters
    ----------
    x : ndarray
        The value of x at which arctanh is evaluated

    Returns
    -------
    ndarray
        The value of arctanh at the given x.
    """
    return 0.5 * (np.log(1 + x) - np.log(1 - x))


def d_arctanh(x):
    """
    Analytic derivative of the arctanh function.

    Parameters
    ----------
    x : ndarray
        The value of x at which arctanh is evaluated

    Returns
    -------
    ndarray
        The derivative of arctanh(x) wrt x.
    """
    return 1. / (1. - x ** 2)


def d_arctan2(y, x):
    """
    Compute the derivative of the two-argument inverse tangent function.

    Parameters
    ----------
    y : ndarray
        Array value of y in arctan2(y, x).
    x : ndarray
        Array value of x in arctan2(y, x).

    Returns
    -------
    d_dy : ndarray
        Derivative of arctan2 wrt y with the shape of y preserved.
    d_dx : ndarray
        Derivative of arctan2 wrt x with the shape of x preserved.
    """
    return x / (x**2 + y**2), -y / (x**2 + y**2)


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
    return np.sqrt(om_sum(x ** 2, axis=axis))


def d_norm(x, axis=None):
    """
    Compute the derivative of the norm of an ndarray along a given axis.

    Parameters
    ----------
    x : ndarray
        Array value argument.
    axis : int or None.
        The axis along which the norm is computed, or None if the sum is computed over all elements.

    Returns
    -------
    ndarray
        Derivative of norm wrt x.
    """
    x_sq = (x ** 2)
    return d_sum(x_sq, axis=axis) * (x / np.sqrt(om_sum(x_sq, axis=axis)))
