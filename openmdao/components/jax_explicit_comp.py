"""
An ExplicitComponent that uses JAX for derivatives.
"""

import sys
from types import MethodType

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.om_warnings import issue_warning
from openmdao.utils.jax_utils import jax, jit, compute_partials as _jax_compute_partials, \
    compute_jacvec_product as _jax_compute_jacvec_product, ReturnChecker, \
    _jax_register_pytree_class, _compute_sparsity

class JaxExplicitComponent(ExplicitComponent):
    """
    Base class for explicit components when using JAX for derivatives.

    Parameters
    ----------
    fallback_derivs_method : str
        The method to use if JAX is not available. Default is 'fd'.
    **kwargs : dict
        Additional arguments to be passed to the base class.
    """

    def __init__(self, fallback_derivs_method='fd', **kwargs):  # noqa
        if sys.version_info < (3, 9):
            raise RuntimeError("JaxExplicitComponent requires Python 3.9 or newer.")
        super().__init__(**kwargs)
        # if derivs_method is explicitly passed in, just use it
        if 'derivs_method' in kwargs:
            return

        if jax:
            self.options['derivs_method'] = 'jax'
        else:
            issue_warning(f"{self.msginfo}: JAX is not available, so '{fallback_derivs_method}' "
                          "will be used for derivatives.")
            self.options['derivs_method'] = fallback_derivs_method

    def _setup_jax(self):
        """
        Set up the jax interface for this component.
        """
        if self.compute_primal is None:
            raise RuntimeError(f"{self.msginfo}: compute_primal is not defined for this component.")

        if self.matrix_free is True:
            self.compute_jacvec_product = MethodType(_jax_compute_jacvec_product, self)
        else:
            self.compute_partials = MethodType(_jax_compute_partials, self)
            self._has_compute_partials = True

        # determine if the compute_primal method returns a tuple
        self._compute_primal_returns_tuple = ReturnChecker(self.compute_primal).returns_tuple()

        if self.options['use_jit']:
            static_argnums = []
            idx = len(self._var_rel_names['input']) + 1
            static_argnums.extend(range(idx, idx + len(self._discrete_inputs)))
            self.compute_primal = MethodType(jit(self.compute_primal.__func__,
                                                 static_argnums=static_argnums), self)

        _jax_register_pytree_class(self.__class__)

    def _get_jac_func(self):
        """
        Return the jacobian function for this component.

        In forward mode, jax.jacfwd is used, and in reverse mode, jax.jacrev is used.  The direction
        is chosen automatically based on the sizes of the inputs and outputs.

        Returns
        -------
        function
            The jacobian function.
        """
        # TODO: modify this to use relevance and possibly compile multiple jac functions depending
        # on DV/response so that we don't compute any derivatives that are always zero.
        if self._jac_func_ is None:
            fjax = jax.jacfwd if self.best_partial_deriv_direction() == 'fwd' else jax.jacrev
            nstatic = len(self._discrete_inputs)
            wrt_idxs = list(range(1, len(self._var_abs2meta['input']) + 1))
            self._jac_func_ = MethodType(fjax(self.compute_primal.__func__, argnums=wrt_idxs), self)

            if self.options['use_jit']:
                static_argnums = tuple(range(1 + len(wrt_idxs), 1 + len(wrt_idxs) + nstatic))
                self._jac_func_ = MethodType(jit(self._jac_func_.__func__,
                                                 static_argnums=static_argnums),
                                             self)

        return self._jac_func_

    def compute_sparsity(self, direction=None):
        """
        Get the sparsity of the Jacobian.
        """
        return _compute_sparsity(self, direction)
