"""
An ImplicitComponent that uses JAX for derivatives.
"""

import sys
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.utils.jax_utils import jax


class JaxImplicitComponent(ImplicitComponent):
    """
    Base class for implicit components when using JAX for derivatives.

    Parameters
    ----------
    fallback_deriv_method : str
        The method to use if JAX is not available. Default is 'fd'.
    **kwargs : dict
        Additional arguments to be passed to the base class.
    """

    def __init__(self, fallback_deriv_method='fd', **kwargs):  # noqa
        if sys.version_info < (3, 9):
            raise RuntimeError("JaxImplicitComponent requires Python 3.9 or newer.")
        super().__init__(**kwargs)
        self.options['derivs_method'] = 'jax' if jax else fallback_deriv_method
