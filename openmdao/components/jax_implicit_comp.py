"""
An ImplicitComponent that uses JAX for derivatives.
"""

import sys
from types import MethodType

from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.utils.om_warnings import issue_warning
from openmdao.utils.jax_utils import jax, jit, \
    linearize as _jax_linearize, apply_linear as _jax_apply_linear, _jax_register_pytree_class, \
    _compute_sparsity, ReturnChecker


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

    def _setup_jax(self):
        if self.compute_primal is None:
            raise RuntimeError(f"{self.msginfo}: compute_primal is not defined for this component.")

        if self.matrix_free is True:
            self.apply_linear = MethodType(_jax_apply_linear, self)
        else:
            self.linearize = MethodType(_jax_linearize, self)
            self._has_linearize = True

        # determine if the compute_primal method returns a tuple
        self._compute_primal_returns_tuple = ReturnChecker(self.compute_primal).returns_tuple()

        if self.options['use_jit']:
            static_argnums = []
            idx = len(self._var_rel_names['input']) + len(self._var_rel_names['output']) + 1
            static_argnums.extend(range(idx, idx + len(self._discrete_inputs)))
            self.compute_primal = MethodType(jit(self.compute_primal.__func__,
                                                 static_argnums=static_argnums), self)

        _jax_register_pytree_class(self.__class__)

    def _get_jac_func(self):
        # TODO: modify this to use relevance and possibly compile multiple jac functions depending
        # on DV/response so that we don't compute any derivatives that are always zero.
        if self._jac_func_ is None:
            fjax = jax.jacfwd if self.best_partial_deriv_direction() == 'fwd' else jax.jacrev
            wrt_idxs = list(range(1, len(self._var_abs2meta['input']) +
                                  len(self._var_abs2meta['output']) + 1))
            primal_func = self.compute_primal.__func__
            self._jac_func_ = MethodType(fjax(primal_func, argnums=wrt_idxs), self)

            if self.options['use_jit']:
                static_argnums = tuple(range(1 + len(wrt_idxs), 1 + len(wrt_idxs) +
                                             len(self._discrete_inputs)))
                self._jac_func_ = MethodType(jit(self._jac_func_.__func__,
                                                 static_argnums=static_argnums),
                                             self)
        return self._jac_func_

    def compute_sparsity(self, direction=None):
        """
        Get the sparsity of the Jacobian.
        """
        return _compute_sparsity(self, direction)
