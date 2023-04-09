"""
OpenMDAO-friendly interfaces to numpy functions and their derivatives.
"""


import numpy as np

from numpy import arccos, arcsin, arccosh, arcsinh, arctan, cos, cosh, cumsum, exp, \
    log, log10, sin, sinh, tan, tanh

from scipy.special import erf, erfc

_2_div_sqrt_pi = 2. / np.sqrt(np.pi)


def d_cumsum(x, axis=None):
    """
    Compute derivative of the cumulative sum function.

    Parameters
    ----------
    x : np.ndarray
        Array of inputs to the cumulative sum function.
    axis : int or None
        Axis along which the input should be summed.

    Returns
    -------
    np.ndarray
        The cumulative sum of x along the given axis.

    """
    n = np.prod(x.shape, dtype=int)
    if axis is None or len(x.shape) == 1:
        J = np.tri(n)
    else:
        J = np.zeros(x.shape + x.shape)
        J_view = np.reshape(J, (n, n))

        k = len(x.shape)
        kron = np.kron
        eye = np.eye

        len_axes_before_axis = 1 if axis == 0 \
            else np.prod([size for i, size in enumerate(x.shape) if i < axis], dtype=int)
        len_axes_after_axis = 1 if axis == k \
            else np.prod([size for i, size in enumerate(x.shape) if i > axis], dtype=int)
        pattern = kron(np.tri(x.shape[axis]), eye(len_axes_after_axis))
        J_view[...] = kron(eye(len_axes_before_axis), pattern)

    return J


def d_arccos(x):
    """
    Compute derivative of the inverse cosine function.

    Parameters
    ----------
    x : ndarray
        Array value to be computed on.

    Returns
    -------
    ndarray
        Derivative of arccos wrt x with the shape of x preserved.
    """
    return -1 / np.sqrt(1 - x**2)


def d_arccosh(x):
    """
    Compute derivative of the inverse hyperbolic cosine function.

    Parameters
    ----------
    x : ndarray
        Array value to be computed on.

    Returns
    -------
    ndarray
        Derivative of arccosh wrt x with the shape of x preserved.
    """
    return 1 / (np.sqrt(x - 1) * np.sqrt(x + 1))


def d_arcsin(x):
    """
    Compute derivative of the inverse sine function.

    Parameters
    ----------
    x : ndarray
        Array value to be computed on.

    Returns
    -------
    ndarray
        Derivative of arcsin wrt x with the shape of x preserved.
    """
    return 1 / np.sqrt(1 - x**2)


def d_arcsinh(x):
    """
    Compute derivative of the inverse hyperbolic sine function.

    Parameters
    ----------
    x : ndarray
        Array value to be computed on.

    Returns
    -------
    ndarray
        Derivative of arcsinh wrt x with the shape of x preserved.
    """
    return 1 / np.sqrt(x**2 + 1)


def d_arctan(x):
    """
    Compute derivative of the inverse tangent function.

    Parameters
    ----------
    x : ndarray
        Array value argument.

    Returns
    -------
    ndarray
        Derivative of arctan wrt x with the shape of x preserved.
    """
    return 1 / (1 + x**2)


def d_cos(x):
    """
    Compute derivative of the cosine function.

    Parameters
    ----------
    x : ndarray
        Array value argument.

    Returns
    -------
    ndarray
        Derivative of cosine wrt x with the shape of x preserved.
    """
    return -np.sin(x)


def d_cosh(x):
    """
    Compute the derivative of the hyperbolic cosine function.

    Parameters
    ----------
    x : ndarray
        Array value argument.

    Returns
    -------
    ndarray
        Derivative of cosh wrt x with the shape of x preserved.
    """
    return np.sinh(x)


def dot(a, b):
    """
    Compute the dot product of two n x m vectors a and b.

    Parameters
    ----------
    a : ndarray
        The first array argument.
    b : ndarray
        The second array argument.

    Returns
    -------
    ndarray
        The dot product of a and b along all but the first dimension.
    """
    return np.einsum('ni,ni->n', a, b)


def d_dot(a, b):
    """
    Compute the dot product of two n x m vectors a and b.

    Parameters
    ----------
    a : ndarray
        The first array argument.
    b : ndarray
        The second array argument.

    Returns
    -------
    d_da : ndarray
        Derivative of the dot product of a and b along all but the first dimension wrt a.
    d_db : ndarray
        Derivative of the dot product of a and b along all but the first dimension wrt b.
    """
    return b.ravel(), a.ravel()


def d_erf(x):
    """
    Compute the derivative of the error function.

    Parameters
    ----------
    x : ndarray
        Array value argument.

    Returns
    -------
    ndarray
        Derivative of erf wrt x with the shape of x preserved.
    """
    return _2_div_sqrt_pi * np.exp(-(x ** 2))


def d_erfc(x):
    """
    Compute the derivative of the complementary error function.

    Parameters
    ----------
    x : ndarray
        Array value argument.

    Returns
    -------
    ndarray
        Derivative of erfc wrt x with the shape of x preserved.
    """
    return -d_erf(x)


def d_exp(x):
    """
    Compute the derivative of the exponential function.

    Parameters
    ----------
    x : ndarray
        Array value argument.

    Returns
    -------
    ndarray
        Derivative of exp wrt x with the shape of x preserved.
    """
    return np.exp(x)


def d_log(x):
    """
    Compute the derivative of the natural logarithm function.

    Parameters
    ----------
    x : ndarray
        Array value argument.

    Returns
    -------
    ndarray
        Derivative of log wrt x with the shape of x preserved.
    """
    return 1. / x


def d_log10(x):
    """
    Compute the derivative of the log-base-10 function.

    Parameters
    ----------
    x : ndarray
        Array value argument.

    Returns
    -------
    ndarray
        Derivative of log10 wrt x with the shape of x preserved.
    """
    return 1. / (np.log(10) * x)


def d_sin(x):
    """
    Compute the derivative of the sine function.

    Parameters
    ----------
    x : ndarray
        Array value argument.

    Returns
    -------
    ndarray
        Derivative of sin wrt x.
    """
    return np.cos(x)


def d_sinh(x):
    """
    Compute the derivative of the hyperbolic sine function.

    Parameters
    ----------
    x : ndarray
        Array value argument.

    Returns
    -------
    ndarray
        Derivative of sinh wrt x.
    """
    return np.cosh(x)


def sqrt(x):
    """
    Compute the square root of input x.

    Parameters
    ----------
    x : ndarray
        Array valued argument.

    Returns
    -------
    ndarray
        The square root of each element in x.
    """
    return np.sqrt(x)


def d_sqrt(x):
    """
    Compute the derivative of the square root function.

    Parameters
    ----------
    x : ndarray
        Array value argument.

    Returns
    -------
    ndarray
        Derivative of sqrt wrt x.
    """
    return 0.5 / sqrt(x)


def sum(x, axis=None):
    """
    Sum of the elements over the given axis of x.

    Parameters
    ----------
    x : ndarray
        Array valued argument.
    axis : int
        If None, return the sum of all elements in x.

    Returns
    -------
    ndarray
        The sum of the elements in x over the given axis.
    """
    return np.sum(x, axis=axis, keepdims=axis is not None)


def d_sum(x, axis=None):
    """
    Compute the derivative of the sum of the elements in x along the given axis.

    Parameters
    ----------
    x : ndarray
        Array value argument.
    axis : int or None.
        The axis along which the sum is computed, or None if the sum is computed over all elements.

    Returns
    -------
    ndarray
        Derivative of sum wrt x along the specified axis. The shape of the jacobian is the
        shape of output of sum (which keeps dimensions when axis is specified) concatenated
        with the shape of the input.
    """
    kron = np.kron
    eye = np.eye

    if axis is None or len(x.shape) == 1:
        return np.ones((1,) + x.shape)
    else:
        # This builds up J to the shape of (total_output_size, total_input_size)
        # and then reshapes it to the appropriate dimensions. There may be a
        # more efficient way to do this.
        #
        # Build up a list of arguments for the kronecker products.
        #
        # If the axis of x is the summation axis, it appears in the kronecker products
        # as a row of ones of its given dimension.
        #
        # Otherwise, the axis appears in the kronecker products as an identity matrix
        # of its given dimension.
        kron_args = []
        for i in range(len(x.shape)):
            if i == axis:
                kron_args.append(np.atleast_2d(np.ones(x.shape[i])))
            else:
                kron_args.append(eye(x.shape[i]))

        # Start with the kronecker product of the last two arguments
        arg2 = kron_args.pop()
        arg1 = kron_args.pop()
        jac = kron(arg1, arg2)

        # Now proceed through the remaining ones, fromr right to left
        while kron_args:
            arg1 = kron_args.pop()
            jac = kron(arg1, jac)

        ax0 = np.prod([ax_size for i, ax_size in enumerate(x.shape) if i != axis])

        jac = np.reshape(jac, (ax0,) + x.shape)
        return jac


def d_tan(x):
    """
    Compute the derivative of the tangent function.

    Parameters
    ----------
    x : ndarray
        Array value argument

    Returns
    -------
    ndarray
        Derivative of tan wrt x.
    """
    return 1 / (np.cos(x) ** 2)


def d_tanh(x):
    """
    Compute the derivative of the hyperbolic tangent function.

    Parameters
    ----------
    x : ndarray
        Array value argument

    Returns
    -------
    ndarray
        Derivative of tanh wrt x.
    """
    return 1 / (np.cosh(x) ** 2)
