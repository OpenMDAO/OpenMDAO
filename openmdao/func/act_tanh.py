import numpy as np


def act_tanh(x, mu=1., z=0., a=-1., b=1.):
    """
    A function which provides a differentiable activation function based on the hyperbolic tangent.

    Parameters
    ----------
    x : float or np.array
        The input at which the value of the activation function is to be computed.
    mu : float
        A shaping parameter which impacts the "abruptness" of the activation function. As this value approaches zero
        the response approaches that of a step function.
    z : float
        The value of the independent variable about which the activation response is centered.
    a : float
        The initial value that the input asymptotically approaches as x approaches negative infinity.
    b : float
        The final value that the input asymptotically approaches as x approaches positive infinity.

    Returns
    -------
        The value of the activation response at the given input.
    """
    dy = b - a
    tanh_term = np.tanh((x - z) / mu)
    return 0.5 * dy * (1 + tanh_term) + a


def d_act_tanh(x, mu=1.0, z=0.0, a=-1.0, b=1.0, d_x=None, d_mu=None, d_z=None, d_a=None, d_b=None):
    """
    A function which provides a differentiable activation function based on the hyperbolic tangent.

    Parameters
    ----------
    x : float or np.array
        The input at which the value of the activation function is to be computed.
    mu : float
        A shaping parameter which impacts the "abruptness" of the activation function. As this value approaches zero
        the response approaches that of a step function.
    z : float
        The value of the independent variable about which the activation response is centered.
    a : float
        The initial value that the input asymptotically approaches negative infinity.
    b : float
        The final value that the input asymptotically approaches positive infinity.
    dx : np.array or None
        If provided, an array with the same length as x which will be populated with the _diagonal_ of the partial
        derivative jacobian matrix for tanh_act with respect to x.
    dmu : np.array or None
        If provided, an array with the same length as x which will be populated with the column vector of the partial
        derivative jacobian matrix for tanh_act with respect to mu.
    dz : np.array or None
        If provided, an array with the same length as z which will be populated with the column vector of the partial
        derivative jacobian matrix for tanh_act with respect to z.
    da : np.array or None
        If provided, an array with the same length as z which will be populated with the column vector of the partial
        derivative jacobian matrix for tanh_act with respect to a.
    db : np.array or None
        If provided, an array with the same length as z which will be populated with the column vector of the partial
        derivative jacobian matrix for tanh_act with respect to b.
    Returns
    -------
    dict
        A dictionary which contains the partial derivatives of the tanh activation function wrt inputs, stored in the
        keys 'x', 'mu', 'z', 'a', 'b'.
    """
    dy = b - a
    xmz = x - z
    tanh_term = np.tanh(xmz / mu)
    cosh2 = np.cosh(xmz / mu) ** 2

    if d_x is not None:
        d_x[...] = (0.5 * dy) / (mu * cosh2)
    if d_mu is not None:
        d_mu[...] = -(0.5 * dy * xmz) / (mu ** 2 * cosh2)
    if d_z is not None:
        d_z[...] = (-0.5 * dy) / (mu * cosh2)
    if d_a is not None:
        d_a[...] = 0.5 * (1 - tanh_term)
    if d_b is not None:
        d_b[...] = 0.5 * (1 + tanh_term)