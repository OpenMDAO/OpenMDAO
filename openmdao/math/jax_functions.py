"""
Smooth approximations to functions that do not have continuous derivatives.
"""

CITATIONS = """
@conference {Martins:2005:SOU,
        title = {On Structural Optimization Using Constraint Aggregation},
        booktitle = {Proceedings of the 6th World Congress on Structural and Multidisciplinary
                     Optimization},
        year = {2005},
        month = {May},
        address = {Rio de Janeiro, Brazil},
        author = {Joaquim R. R. A. Martins and Nicholas M. K. Poon}
}
"""

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
except ImportError as e:
    raise ImportError('jax is not available, but is necessary for `openmdao.math`.\n'
                      'Try using `pip install jax jaxlib` to use this capability.') from e


@jax.jit
def act_tanh(x, mu=1.0E-2, z=0., a=-1., b=1.):
    """
    Compute a differentiable activation function based on the hyperbolic tangent.

    act_tanh can be used to approximate a step function from `a` to `b`, occurring at x=z.
    Smaller values of parameter `mu` more accurately represent a step functionbut the
    "sharpness" of the corners in the response may be more difficult for gradient-based
    approaches to resolve.

    Parameters
    ----------
    x : float or array
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
    float or jnp.array
        The value of the activation response at the given input.
    """
    dy = b - a
    tanh_term = jnp.tanh((x - z) / mu)
    return 0.5 * dy * (1 + tanh_term) + a


@jax.jit
def smooth_max(x, y, mu=1.0E-2):
    """
    Compute a differentiable maximum between two arrays of the same shape.

    Parameters
    ----------
    x : float or jnp.array
        The first value or array of values for comparison.
    y : float or jnp.array
        The second value or array of values for comparison.
    mu : float
        A shaping parameter which impacts the "abruptness" of the activation function.
        As this value approaches zero the response approaches that of a step function.

    Returns
    -------
    float or jnp.array
        For each element in x or y, the greater of the values of x or y at that point.
        This function is smoothed, so near the point where x and y have equal values
        this will be approximate. The accuracy of this approximation can be adjusted
        by changing the mu parameter. Smaller values of mu will lead to more accuracy
        at the expense of the smoothness of the approximation.
    """
    x_greater = act_tanh(x, mu, y, 0.0, 1.0)
    y_greater = 1 - x_greater
    return x_greater * x + y_greater * y


@jax.jit
def smooth_min(x, y, mu=1.0E-2):
    """
    Compute a differentiable minimum between two arrays of the same shape.

    Parameters
    ----------
    x : float or array
        The first value or array of values for comparison.
    y : float or array
        The second value or array of values for comparison.
    mu : float
        A shaping parameter which impacts the "abruptness" of the activation function.
        As this value approaches zero the response approaches that of a step function.

    Returns
    -------
    float or jnp.array
        For each element in x or y, the greater of the values of x or y at that point. This
        function is smoothed, so near the point where x and y have equal values this will
        be approximate. The accuracy of this approximation can be adjusted by changing the
        mu parameter. Smaller values of mu will lead to more accuracy at the expense of the
        smoothness of the approximation.
    """
    x_greater = act_tanh(x, mu, y, 0.0, 1.0)
    y_greater = 1 - x_greater
    return x_greater * y + y_greater * x


@jax.jit
def smooth_abs(x, mu=1.0E-2):
    """
    Compute a differentiable approximation to the absolute value function.

    Parameters
    ----------
    x : float or array
        The argument to absolute value.
    mu : float
        A shaping parameter which impacts the tradeoff between the
        smoothness and accuracy of the function. As this value
        approaches zero the response approaches that of the true
        absolute value.

    Returns
    -------
    float or array
        An approximation of the absolute value. Near zero, the value will
        differ from the true absolute value but its derivative will be continuous.
    """
    act = act_tanh(x, mu, 0.0, -1.0, 1.0)
    return x * act


@jax.jit
def ks_max(x, rho=100.0):
    """
    Compute a differentiable maximum value in an array.

    Given some array of values `x`, compute a differentiable, _conservative_ maximum using the
    Kreisselmeier-Steinhauser function.

    Parameters
    ----------
    x : ndarray
        Array of values.
    rho : float
        Aggregation Factor. Larger values of rho more closely match the true maximum value.

    Returns
    -------
    float
        A conservative approximation to the minimum value in x.

    """
    x_max = jnp.max(x)
    x_diff = x - x_max
    exponents = jnp.exp(rho * x_diff)
    summation = jnp.sum(exponents)
    return x_max + 1.0 / rho * jnp.log(summation)


@jax.jit
def ks_min(x, rho=100.0):
    """
    Compute a differentiable minimum value in an array.

    Given some array of values `x`, compute a differentiable,
    _conservative_ minimum using the Kreisselmeier-Steinhauser function.

    Parameters
    ----------
    x : ndarray
        Array of values.
    rho : float
        Aggregation Factor. Larger values of rho more closely match the true minimum value.

    Returns
    -------
    float
        A conservative approximation to the minimum value in x.
    """
    x_min = jnp.min(x)
    x_diff = x_min - x
    exponents = jnp.exp(rho * x_diff)
    summation = jnp.sum(exponents)
    return x_min - 1.0 / rho * jnp.log(summation)
