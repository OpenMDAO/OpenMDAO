import numpy as np

from ..utils.cs_safe import abs, arctan2, norm

from numpy import arccos, arcsin, arccosh, arcsinh, cos, cosh, dot, erf, erfc, exp, expm1, \
    inner, kron, log, log10, log1p, matmul, outer, power, prod, sin, sinh, sqrt, sum, tan, tanh, tensordot


def d_abs(x, d_x=None):
    if d_x:
        d_x[...] = np.sign(x)


def d_arccos(x, d_x=None):
    if d_x:
        d_x[...] = -1 / np.sqrt(1 - x**2)


def d_arccosh(x, d_x=None):
    if d_x:
        d_x[...] = 1 / (np.sqrt(x - 1) * np.sqrt(x + 1))


def d_arcsin(x, d_x=None):
    if d_x:
        d_x[...] = 1 / np.sqrt(1 - x**2)


def d_arcsinh(x, d_x=None):
    if d_x:
        d_x[...] = 1 / np.sqrt(x**2 + 1)


def d_arctan2(y, x, d_y=None, d_x=None):
    if d_y:
        d_x[...] = -x / (x**2 + y**2)
    if d_x:
        d_x[...] = y / (x**2 + y**2)


def d_arctanh(x, d_x=None):
    if d_x:
        d_x[...] = 1 / (1 - x**2)


def d_cos(x, d_x=None):
    if d_x:
        d_x[...] = -np.sin(x)


def d_cosh(x, d_x=None):
    if d_x:
        d_x[...] = np.sinh(x)


def dot(a, b):
    return np.einsum('ni,ni->n', a, b)


def d_dot(a, b, d_a=None, d_b=None):
    if d_a:
        d_a[...] = b.ravel()
    if d_b:
        d_b[...] = a.ravel()


def d_dot(a, b):
    return b.ravel(), a.ravel()


def d_erf(x, d_x=None):


def d_efc(x, d_x=None):


def d_exp(x, d_x=None):
    if d_x:
        d_x[...] = np.exp(x)


def d_expm1(x, d_x=None):


def d_inner(x, d_x=None):


def d_kron(x, d_x=None):


def d_log(x, d_x=None):
    if d_x:
        return 1. / x


def d_log10(x, d_x=None):
    if d_x:
        d_x[...] = 1. / (np.log(10) * x)


def d_log1p(x, d_x=None):
    if d_x:
        d_x[...] = 1. / (1. + x)


def d_norm(x, axis=None, d_x=None):
    if d_x:
        d_x[...] = 2 * x * d_sum(x, axis=axis)['x']


def d_sin(x, d_x=None):
    if d_x:
        d_x[...] = np.cos(x)


def d_sinh(x, d_x=None):
    if d_x:
        d_x[...] = np.cosh(x)


def d_sqrt(x, d_x=None):
    if d_x:
        d_x[...] = 0.5 / np.sqrt(x)


def d_sum(x, axis=None, d_x=None):
    if d_x:
        if axis is None:
            n = np.size(x)
            d_x[...] = np.ones((1, n))
        else:
            d_x[...] = np.squeeze(np.ones_like(x), axis=axis)


def d_tan(x, d_x=None):
    if d_x:
        d_x[...] = 1 / (np.cos(x) ** 2)


def d_tanh(x, d_x=None):
    if d_x:
        d_x[...] = 1 / (np.cosh(x) ** 2)