"""
This package contains various functions that may be useful in component compute methods.

These functions are implemented using jax.numpy for the underlying calculations.
This will allow these functions to be used in a way that supports jax's automatic differentiation
and just-in-time compilation capabilities.
"""
try:
    import jax
except ImportError as e:
    raise ImportError('jax is not available, but is necessary for `openmdao.math`.\n'
                      'Try using `pip install jax jaxlib` to use this capability.')

from .jax_functions import act_tanh, ks_max, ks_min, smooth_max, smooth_min, smooth_abs
