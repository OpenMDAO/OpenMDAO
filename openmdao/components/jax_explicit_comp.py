"""
An ExplicitComponent that uses JAX for derivatives.
"""

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.jax_utils import jax


class JaxExplicitComponent(ExplicitComponent):
    """
    Base class for explicit components when using JAX for derivatives.

    Parameters
    ----------
    fallback_deriv_method : str
        The method to use if JAX is not available. Default is 'fd'.
    **kwargs : dict
        Additional arguments to be passed to the base class.
    """

    def __init__(self, fallback_deriv_method='fd', **kwargs):  # noqa: D107
        super().__init__(**kwargs)
        self.options['derivs_method'] = 'jax' if jax else fallback_deriv_method
        self.options.declare('use_jit', types=bool, default=True,
                             desc='If True, use jit on the function')
