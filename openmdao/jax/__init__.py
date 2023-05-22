"""
This package contains various functions that may be useful in component compute methods.

These functions are implemented using jax.numpy for the underlying calculations.
This will allow these functions to be used in a way that supports jax's automatic differentiation
and just-in-time compilation capabilities.
"""
from .smooth import act_tanh, smooth_max, smooth_min, smooth_abs
from .ks import ks_max, ks_min
