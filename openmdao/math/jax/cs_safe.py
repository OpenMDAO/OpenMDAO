"""
collection of complex-step safe functions to replace standard numpy operations.
"""
import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


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
    _x = jnp.asarray(x)
    return _x * jnp.sign(_x)


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
    if jnp.iscomplexobj(x) or jnp.iscomplexobj(y):
        a = jnp.real(y)
        b = jnp.imag(y)
        c = jnp.real(x)
        d = jnp.imag(x)
        return jnp.arctan2(a, c) + 1j * (c * b - a * d) / (a ** 2 + c ** 2)
    else:
        return jnp.arctan2(y, x)


def arctanh(x):
    """
    Numpy-compatible, complex-compatible arctanh function for use with complex-step.

    Parameters
    ----------
    x : ndarray
        The value of x at which arctanh is evaluated.

    Returns
    -------
    ndarray
        The value of arctanh at the given x.
    """
    return 0.5 * (jnp.log(1 + x) - jnp.log(1 - x))


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
    return np.sqrt(np.sum(x ** 2, axis=axis, keepdims=axis is not None))