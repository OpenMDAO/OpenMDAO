"""
collection of complex-step safe functions to replace standard numpy operations.
"""

import numpy as np

if np.__version__[0] == '2':
    NumPy2 = True
else:
    NumPy2 = False


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
    if isinstance(x, np.ndarray):
        # Changed in NumPy 2.0: Definition of complex sign changed to follow the Array API standard.
        # 1.x: For complex inputs, the sign function returns sign(x.real) + 0j if x.real != 0
        #      else sign(x.imag) + 0j.
        # 2.0: For complex inputs, the sign function returns x / abs(x), and 0 if x==0.
        if NumPy2 and np.any(np.iscomplex(x)):
            # replicate NumPy 1.x behavior for complex arrays
            z0_idx = x.real == 0
            nz_idx = x.real != 0
            signs = np.zeros(x.shape, dtype=complex)
            signs[nz_idx] = np.sign(x[nz_idx]).real + 0j
            signs[z0_idx] = np.sign(x[z0_idx].imag) + 0j
            return x * signs
        else:
            return x * np.sign(x)
    elif x.real < 0.0:
        return -x
    return x


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
    return np.sqrt(np.sum(x**2, axis=axis))


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
