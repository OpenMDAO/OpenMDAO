"""
An ImplicitComponent that uses JAX for derivatives.
"""

import sys
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.utils.jax_utils import jax
from openmdao.utils.om_warnings import issue_warning


class JaxImplicitComponent(ImplicitComponent):
    """
    Base class for implicit components when using JAX for derivatives.

    Parameters
    ----------
    fallback_derivs_method : str
        The method to use if JAX is not available. Default is 'fd'.
    **kwargs : dict
        Additional arguments to be passed to the base class.
    """

    def __init__(self, fallback_derivs_method='fd', **kwargs):  # noqa
        if sys.version_info < (3, 9):
            raise RuntimeError("JaxImplicitComponent requires Python 3.9 or newer.")
        super().__init__(**kwargs)
        # if derivs_method is explicitly passed in, just use it
        if 'derivs_method' in kwargs:
            return

        if jax:
            self.options['derivs_method'] = 'jax'
        else:
            issue_warning(f"{self.msginfo}: JAX is not available, so "
                          f"'{fallback_derivs_method}' will be used for derivatives.")
            self.options['derivs_method'] = fallback_derivs_method
