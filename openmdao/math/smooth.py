"""
Smooth approximations to functions that do not have continuous derivatives.
"""


import numpy as np


def act_tanh(x, mu=1.0E-2, z=0., a=-1., b=1.):
    """
    Differentiable activation function based on the hyperbolic tangent.

    Parameters
    ----------
    x : float or np.array
        The input at which the value of the activation function
        is to be computed.
    mu : float
        A shaping parameter which impacts the "abruptness" of
        the activation function. As this value approaches zero
        the response approaches that of a step function.
    z : float
        The value of the independent variable about which the
        activation response is centered.
    a : float
        The initial value that the input asymptotically approaches
        as x approaches negative infinity.
    b : float
        The final value that the input asymptotically approaches
        as x approaches positive infinity.

    Returns
    -------
        The value of the activation response at the given input.
    """
    dy = b - a
    tanh_term = np.tanh((x - z) / mu)
    return 0.5 * dy * (1 + tanh_term) + a


def d_act_tanh(x, mu=1.0E-2, z=0.0, a=-1.0, b=1.0,
               dx=True, dmu=True, dz=True, da=True, db=True):
    """
    Differentiable activation function based on the hyperbolic tangent.

    Parameters
    ----------
    x : float or np.array
        The input at which the value of the activation function is to be computed.
    mu : float
        A shaping parameter which impacts the "abruptness" of the activation function.
        As this value approaches zero the response approaches that of a step function.
    z : float or np.array
        The value of the independent variable about which the activation response is centered.
    a : float
        The initial value that the input asymptotically approaches negative infinity.
    b : float
        The final value that the input asymptotically approaches positive infinity.
    dx : bool
        True if the Compute the derivative of act_tanh wrt x should be calculated. Setting this to
        False can save time when the derivative is not needed.
    dmu : bool
        True if the Compute the derivative of act_tanh wrt mu should be calculated. Setting this
        to False can save time when the derivative is not needed.
    dz : bool
        True if the Compute the derivative of act_tanh wrt z should be calculated. Setting this
        to False can save time when the derivative is not needed.
    da : bool
        True if the Compute the derivative of act_tanh wrt a should be calculated. Setting this
        to False can save time when the derivative is not needed.
    db : bool
        True if the Compute the derivative of act_tanh wrt b should be calculated. Setting this
        to False can save time when the derivative is not needed.
    sparse : bool
        If True, return only the known nonzero elements of the derivative. These will
        be flat arrays which correspond to the diagonal of d_dx and d_dz, and the
        column vectors of d_mu, d_a, and d_b.

    Returns
    -------
    d_dx : float or np.ndarray or None
        Derivatives of act_tanh wrt x or None if argument dx is False.
    d_dmu : float or np.ndarray or None
        Derivatives of act_tanh wrt mu or None if argument dmu is False.
    d_dz : float or np.ndarray or None
        Derivatives of act_tanh wrt z or None if argument dz is False.
    d_da : float or np.ndarray or None
        Derivatives of act_tanh wrt a or None if argument da is False.
    d_db : float or np.ndarray or None
        Derivatives of act_tanh wrt b or None if argument db is False.
    """
    dy = b - a
    dy_d_2 = 0.5 * dy
    xmz = x - z
    xmz_d_mu = xmz / mu
    tanh_term = np.tanh(xmz_d_mu)

    # Avoid overflow warnings from cosh
    oo_mu_cosh2 = np.zeros_like(xmz_d_mu)
    idxs_small = np.where(np.abs(xmz_d_mu) < 20)
    if idxs_small:
        cosh2 = np.cosh(xmz_d_mu[idxs_small]) ** 2
        oo_mu_cosh2[idxs_small] = 1. / (mu * cosh2)

    d_dx = ((dy_d_2 * oo_mu_cosh2).ravel()
            if sparse else np.diagflat(dy_d_2 * oo_mu_cosh2)) if dx else None

    if dz:
        if z.size == x.size:
            d_dz = (-dy_d_2 * oo_mu_cosh2).ravel() \
                if sparse else np.diagflat(-dy_d_2 * oo_mu_cosh2)
        else:
            d_dz = (-dy_d_2 * oo_mu_cosh2).ravel() \
                if sparse else np.reshape(-dy_d_2 * oo_mu_cosh2, (x.size, 1))

    return (d_dx,  # d_dx
            -(dy_d_2 * xmz_d_mu) * oo_mu_cosh2 if dmu else None,  # d_dmu
            d_dz if dz else None,  # d_dz
            0.5 * (1 - tanh_term) if da else None,  # d_da
            0.5 * (1 + tanh_term) if db else None)  # d_db


def smooth_max(x, y, mu=1.0E-2):
    """
    Differentiable maximum between two arrays of the same shape.

    Parameters
    ----------
    x : float or np.array
        The first value or array of values for comparison.
    y : float or np.array
        The second value or array of values for comparison.
    mu : float
        A shaping parameter which impacts the "abruptness" of the activation function.
        As this value approaches zero the response approaches that of a step function.

    Returns
    -------
    float or np.array
        For each element in x or y, the greater of the values of x or y at that point.
        This function is smoothed, so near the point where x and y have equal values
        this will be approximate. The accuracy of this approximation can be adjusted
        by changing the mu parameter. Smaller values of mu will lead to more accuracy
        at the expense of the smoothness of the approximation.
    """
    x_greater = act_tanh(x=x, mu=mu, z=y, a=0, b=1)
    y_greater = 1 - x_greater
    return x_greater * x + y_greater * y


def d_smooth_max(x, y, mu=1.0E-2, dx=True, dy=True, dmu=True):
    """
    Compute the derivative of the smooth max function.

    Parameters
    ----------
    x : float or np.array
        The first value or array of values for comparison.
    y : float or np.array
        The second value or array of values for comparison.
    mu : float
        A shaping parameter which impacts the "abruptness" of the activation function. As this
        value approaches zero the response approaches that of a step function.
    dx : bool
        True if the derivative of smooth_max wrt x should be calculated. Setting this to False can
        save time when the derivative is not needed. This function will return None in place of
        this derivative when this argument is False.
    dy : bool
        True if the derivative of smooth_max wrt y should be calculated. Setting this to False can
        save time when the derivative is not needed. This function will return None in place of
        this derivative when this argument is False.
    dmu : bool
        True if the derivative of smooth_max wrt mu should be calculated. Setting this to False
        can save time when the derivative is not needed. This function will return None in
        place of this derivative when this argument is False.

    Returns
    -------
    d_dx : np.array
        The derivative of the smooth max function wrt x.
        These are the nonzero derivatives (the diagonal of the partial derivative jacobian).
        They can be transformed into a dense jacobian using np.diagflat.
    d_dy : np.array
        The derivative of the smooth max function wrt y.
        These are the nonzero derivatives (the diagonal of the partial derivative jacobian).
        They can be transformed into a dense jacobian using np.diagflat.
    d_dmu : np.array
        The derivative of the smooth max function wrt mu.
        These are the nonzero derivatives (the diagonal of the partial derivative jacobian).
        They can be transformed into a dense jacobian using np.diagflat.
    """
    d_dx = d_dy = d_dmu = None

    x_greater = act_tanh(x=x, mu=mu, z=y, a=0, b=1)
    y_greater = 1 - x_greater

    # Compute the partials of x_greater
    pxgreater_px, pxgreater_pmu, pxgreater_py, _, _ = d_act_tanh(x=x, mu=mu, z=y, a=0, b=1, dx=dx,
                                                                 dz=dy, dmu=dmu, da=False, db=False)

    pygreater_px = -pxgreater_px
    pygreater_py = -pxgreater_py
    pygreater_pmu = -pxgreater_pmu

    _x = x.ravel()
    _xg = x_greater.ravel()
    _y = y.ravel()
    _yg = y_greater.ravel()

    if dx:
        d_dx = pxgreater_px.ravel() * _x + _xg + _y * pygreater_px.ravel()
        d_dx = d_dx if sparse else np.diagflat(d_dx)

    if dy:
        d_dy = pxgreater_py.ravel() * _x + _y * pygreater_py.ravel() + _yg
        d_dy = d_dy if sparse else np.diagflat(d_dy)

    if dmu:
        d_dmu = pxgreater_pmu.ravel() * _x + pygreater_pmu.ravel() * _y

    return d_dx, d_dy, d_dmu


def smooth_min(x, y, mu=1.0E-2):
    """
    Differentiable minimum between two arrays of the same shape.

    Parameters
    ----------
    x : float or np.array
        The first value or array of values for comparison.
    y : float or np.array
        The second value or array of values for comparison.
    mu : float
        A shaping parameter which impacts the "abruptness" of the activation function.
        As this value approaches zero the response approaches that of a step function.

    Returns
    -------
    float or np.array
        For each element in x or y, the greater of the values of x or y at that point. This
        function is smoothed, so near the point where x and y have equal values this will
        be approximate. The accuracy of this approximation can be adjusted by changing the
        mu parameter. Smaller values of mu will lead to more accuracy at the expense of the
        smoothness of the approximation.
    """
    x_greater = act_tanh(x=x, mu=mu, z=y, a=0, b=1)
    y_greater = 1 - x_greater
    return x_greater * y + y_greater * x


def d_smooth_min(x, y, mu, dx=True, dy=True, dmu=True):
    """
    Compute the derivative of the smooth min function.

    Parameters
    ----------
    x : float or np.array
        The first value or array of values for comparison.
    y : float or np.array
        The second value or array of values for comparison.
    mu : float
        A shaping parameter which impacts the "abruptness"
        of the activation function. As this value approaches zero
        the response approaches that of a step function.
    dx : bool
        True if the derivative of smooth_max wrt x should
        be calculated. Setting this to False can save time when the
        derivative is not needed. This function will return
        None in place of this derivative when this argument is False.
    dy : bool
        True if the derivative of smooth_max wrt y should be
        calculated. Setting this to False can save time when the
        derivative is not needed. This function will return
        None in place of this derivative when this argument is False.
    dmu : bool
        True if the derivative of smooth_max wrt mu should be
        calculated. Setting this to False can save time when the
        derivative is not needed. This function will return None
        in place of this derivative when this argument is False.

    Returns
    -------
    dx : float or np.array or None
        The derivatives of the smooth_min function wrt x,
        or None if argument dx is False.
    dy : float or np.array or None
        The derivatives of the smooth_min function wrt y,
        or None if argument dy is False.
    dmu : float or np.array or None
        The derivatives of the smooth_min function wrt mu,
        or None if argument dmu is False.
    """
    d_dx = d_dy = d_dmu = None

    x_greater = act_tanh(x=x, mu=mu, z=y, a=0, b=1)
    y_greater = 1 - x_greater

    # Compute the partials of x_greater
    pxgreater_px, pxgreater_pmu, pxgreater_py, _, _ = d_act_tanh(x=x, mu=mu, z=y, a=0, b=1, dx=dx,
                                                                 dz=dy, dmu=dmu, da=False, db=False)

    pygreater_px = -pxgreater_px
    pygreater_py = -pxgreater_py
    pygreater_pmu = -pxgreater_pmu

    _x = x.ravel()
    _xg = x_greater.ravel()
    _y = y.ravel()
    _yg = y_greater.ravel()

    if dx:
        d_dx = pxgreater_px.ravel() * _y + pygreater_px.ravel() * _x + _yg
        d_dx = d_dx if sparse else np.diagflat(d_dx)
    if dy:
        d_dy = pxgreater_py.ravel() * _y + _xg + pygreater_py.ravel() * _x
        d_dy = d_dy if sparse else np.diagflat(d_dy)
    if dmu:
        d_dmu = -pxgreater_pmu.ravel() * _x - pygreater_pmu.ravel() * _y

    return d_dx, d_dy, d_dmu


def smooth_abs(x, mu=1.0E-2):
    """
    Differentiable approximation to the absolute value function.

    Parameters
    ----------
    x : float or np.array
        The argument to absolute value.
    mu : float
        A shaping parameter which impacts the tradeoff between the
        smoothness and accuracy of the function. As this value
        approaches zero the response approaches that of the true
        absolute value.

    Returns
    -------
    float or np.array
        An approximation of the absolute value. Near zero, the value will
        differ from the true absolute value but its derivative will be continuous.
    """
    act = act_tanh(x=x, mu=mu, z=0, a=-1, b=1)
    return x * act


def d_smooth_abs(x, mu=1.0E-2, dx=True, dmu=True):
    """
    Differentiable approximation to the absolute value function.

    Parameters
    ----------
    x : float or np.array
        The argument to absolute value.
    mu : float
        A shaping parameter which impacts the tradeoff between
        the smoothness and accuracy of the function.
        As this value approaches zero the response
        approaches that of the true absolute value.
    dx : bool
        If True, return the derivative with respect to x.
        Otherwise, return None in its place.
    dmu : bool
        If True, return the derivative with respect to mu.
        Otherwise, return None in its place.

    Returns
    -------
    d_dx : float or np.array or None
        The derivatives of the smooth_abs function wrt x,
        or None if argument dx is False.
    d_dmu : float or np.array or None
        The derivatives of the smooth_abs function wrt mu,
        or None if argument dmu is False.
    """
    act = act_tanh(x=x, mu=mu, z=0, a=-1, b=1)
    pact_px, pact_pmu, _, _, _ = d_act_tanh(x=x, mu=mu, z=0, a=-1, b=1,
                                            dx=True, dmu=True, dz=False,
                                            da=False, db=False)

    d_dx = pact_px.ravel() * x.ravel() + act.ravel() if dx else None
    d_dmu = x.ravel() * pact_pmu.ravel() if dmu else None

    return d_dx, d_dmu
