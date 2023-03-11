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


def d_act_tanh(x, mu=1.0, z=0.0, a=-1.0, b=1.0, dx=True, dmu=True, dz=True, da=True, db=True):
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
    dx : bool
        True if the derivative of act_tanh wrt x should be calculated. Setting this to False can save time when the
        derivative is not needed.
    dmu : bool
        True if the derivative of act_tanh wrt mu should be calculated. Setting this to False can save time when
        the derivative is not needed.
    dz : bool
        True if the derivative of act_tanh wrt z should be calculated. Setting this to False can save time when
        the derivative is not needed.
    da : bool
        True if the derivative of act_tanh wrt a should be calculated. Setting this to False can save time when
        the derivative is not needed.
    db : bool
        True if the derivative of act_tanh wrt b should be calculated. Setting this to False can save time when
        the derivative is not needed.
    Returns
    -------
    dict
        A dictionary which contains the partial derivatives of the tanh activation function wrt inputs, stored in the
        keys 'x', 'mu', 'z', 'a', 'b'.
    """
    dy = b - a
    dy_d_2 = 0.5 * dy
    xmz = x - z
    tanh_term = np.tanh(xmz / mu)
    cosh2 = np.cosh(xmz / mu) ** 2
    mu_cosh2 = mu * cosh2

    return (dy_d_2 / mu_cosh2 if dx else None,  # d_dx
            -(dy_d_2 * xmz) / (mu * mu_cosh2) if dmu else None,  # d_dmu
            (-dy_d_2) / mu_cosh2 if dz else None,  # d_dz
            0.5 * (1 - tanh_term) if da else None,  # d_da
            0.5 * (1 + tanh_term) if db else None)  # d_db
