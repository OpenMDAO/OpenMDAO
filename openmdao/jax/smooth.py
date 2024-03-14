"""
Smooth approximations to functions that do not have continuous derivatives.
"""

try:
    import jax
    from jax import jit
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
except (ImportError, ModuleNotFoundError):
    jax = None
    from openmdao.utils.jax_utils import jit_stub as jit


@jit
def act_tanh(x, mu=1.0E-2, z=0., a=-1., b=1.):
    """
    Compute a differentiable activation function based on the hyperbolic tangent.

    act_tanh can be used to approximate a step function from `a` to `b`, occurring at x=z.
    Smaller values of parameter `mu` more accurately represent a step function but the
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
    float or array
        The value of the activation response at the given input.
    """
    dy = b - a
    tanh_term = jnp.tanh((x - z) / mu)
    return 0.5 * dy * (1 + tanh_term) + a


@jit
def smooth_max(x, y, mu=1.0E-2):
    """
    Compute a differentiable maximum between two arrays of the same shape.

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
    float or array
        For each element in x or y, the greater of the values of x or y at that point.
        This function is smoothed, so near the point where x and y have equal values
        this will be approximate. The accuracy of this approximation can be adjusted
        by changing the mu parameter. Smaller values of mu will lead to more accuracy
        at the expense of the smoothness of the approximation.
    """
    x_greater = act_tanh(x, mu, y, 0.0, 1.0)
    y_greater = 1 - x_greater
    return x_greater * x + y_greater * y


@jit
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
    float or array
        For each element in x or y, the greater of the values of x or y at that point. This
        function is smoothed, so near the point where x and y have equal values this will
        be approximate. The accuracy of this approximation can be adjusted by changing the
        mu parameter. Smaller values of mu will lead to more accuracy at the expense of the
        smoothness of the approximation.
    """
    x_greater = act_tanh(x, mu, y, 0.0, 1.0)
    y_greater = 1 - x_greater
    return x_greater * y + y_greater * x


@jit
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
