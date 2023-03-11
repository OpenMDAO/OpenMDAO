import numpy as np
import scipy.sparse

from numpy import arccos, arcsin, arccosh, arcsinh, arctan, cos, cosh, exp, \
    log, log10, sin, sinh, sqrt, sum, tan, tanh

from scipy.special import erf, erfc


_2_div_sqrt_pi = 2. / np.sqrt(np.pi)


def d_arccos(x):
    """
    The derivative of the inverse cosine function.

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
    The derivative of the inverse hyperbolic cosine function.

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
    The derivative of the inverse sine function.

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
    The derivative of the inverse hyperbolic sine function.

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
    The derivative of the inverse tangent function.

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
    The derivative of the cosine function.

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
    The derivative of the hyperbolic cosine function.

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
    The dot product of two n x m vectors a and b.

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
    The dot product of two n x m vectors a and b.

    Parameters
    ----------
    a : ndarray
        The first array argument.
    b : ndarray
        The second array argument.

    Returns
    -------
    d_da : ndarray
        The derivative of the dot product of a and b along all but the first dimension wrt a.
    d_db : ndarray
        The derivative of the dot product of a and b along all but the first dimension wrt b.
    """
    return b.ravel(), a.ravel()


def d_erf(x):
    """
    The derivative of the error function.

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
    The derivative of the complementary error function.

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
    The derivative of the exponential function.

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
    The derivative of the natural logarithm function.

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
    The derivative of the log-base-10 function.

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
    The derivative of the sine function.

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
    The derivative of the hyperbolic sine function.

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


def d_sqrt(x):
    """
    The derivative of the square root function.

    Parameters
    ----------
    x : ndarray
        Array value argument.

    Returns
    -------
    ndarray
        Derivative of sqrt wrt x.
    """
    return 0.5 / np.sqrt(x)


def sum(x, axis=None):
    return np.sum(x, axis=axis)


def d_sum(x, axis=None, sparse=False):
    """
    The derivative of the sum of the elements in x along the given axis.

    Parameters
    ----------
    x : ndarray
        Array value argument.
    axis : int or None.
        The axis along which the sum is computed, or None if the sum is computed over all elements.
    sparse : str or None
        If None, return a dense array. Otherwise, sparse provides the scipy sparse format for the returned jacobian.

    Returns
    -------
    ndarray
        Derivative of sum wrt x along the specified axis.
    """
    if sparse is None:
        kron = np.kron
        eye = np.eye
    else:
        def kron(a, b):
            return scipy.sparse.kron(a, b, format=sparse)

        def eye(size):
            return scipy.sparse.eye(size, format=sparse)

    if axis is None or len(x.shape) == 1:
        n = np.size(x)
        return np.ones((1, n))
    else:
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
        J = kron(arg1, arg2)

        # Now proceed through the remaining ones, fromr right to left
        while kron_args:
            arg1 = kron_args.pop()
            J = kron(arg1, J)
        return J


def d_tan(x):
    """
    The derivative of the tangent function.
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
    The derivative of the hyperbolic tangent function.
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
