"""
This package contains jax versions of various functions that may be useful in component compute methods.

These functions are implemented using numpy for jax.numpy for the underlying calculations.
This will allow these functions to be used in a way that supports jax's automatic differentiation methods and
just-in-time compilation capabilities.
"""

from .smooth import act_tanh, smooth_max, smooth_min, smooth_abs

from .cs_safe import abs, arctanh, arctan2, norm
