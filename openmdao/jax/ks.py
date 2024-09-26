"""
jax implementations of the Kreisselmeier-Steinhauser for the min and max values in an array.
"""

try:
    import jax
    from jax import jit
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
except (ImportError, ModuleNotFoundError):
    jax = None
    from openmdao.utils.jax_utils import jit_stub as jit

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


@jit
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


@jit
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
