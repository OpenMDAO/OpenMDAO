"""
An ImplicitComponent that uses JAX for derivatives.
"""

import sys
from types import MethodType
from itertools import chain
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.utils.om_warnings import issue_warning
from openmdao.utils.jax_utils import jax, jit, _jax_register_pytree_class, \
    _compute_sparsity, ReturnChecker, get_vmap_tangents


class JaxImplicitComponent(ImplicitComponent):
    """
    Base class for implicit components when using JAX for derivatives.

    Parameters
    ----------
    fallback_derivs_method : str
        The method to use if JAX is not available. Default is 'fd'.
    **kwargs : dict
        Additional arguments to be passed to the base class.

    Attributes
    ----------
    _compute_primal_returns_tuple : bool
        Whether the compute_primal method returns a tuple.
    _tangents : dict
        The tangents for the inputs and outputs.
    """

    def __init__(self, fallback_derivs_method='fd', **kwargs):  # noqa
        if sys.version_info < (3, 9):
            raise RuntimeError("JaxImplicitComponent requires Python 3.9 or newer.")
        super().__init__(**kwargs)

        self._compute_primal_returns_tuple = False
        self._tangents = {'fwd': None, 'rev': None}
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
            self.apply_linear = MethodType(apply_linear, self)
        else:
            self.linearize = MethodType(linearize, self)
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

        Parameters
        ----------
        direction : str
            The direction to compute the sparsity for.

        Returns
        -------
        coo_matrix
            The sparsity of the Jacobian.
        """
        return _compute_sparsity(self, direction)

    def _get_tangents(self, direction):
        if self._tangents[direction] is None:
            if direction == 'fwd':
                self._tangents[direction] = get_vmap_tangents(tuple(chain(self._inputs.values(),
                                                                          self._outputs.values())),
                                                              direction, fill=1.)
            else:
                self._tangents[direction] = get_vmap_tangents(tuple(self._outputs.values()),
                                                              direction, fill=1.)
        return self._tangents[direction]


# we define linearize here instead of making this the base class version as we
# did with apply_nonlinear, because the existence of a linearize method that is not the
# base class method is used to determine if a given component computes its own partials.
def linearize(inst, inputs, outputs, partials, discrete_inputs=None, discrete_outputs=None):
    """
    Compute sub-jacobian parts and any applicable matrix factorizations for an implicit component.

    The model is assumed to be in an unscaled state.

    Parameters
    ----------
    inst : ImplicitComponent
        The component instance.
    inputs : Vector
        Unscaled, dimensional input variables read via inputs[key].
    outputs : Vector
        Unscaled, dimensional output variables read via outputs[key].
    partials : partial Jacobian
        Sub-jac components written to jacobian[output_name, input_name].
    discrete_inputs : dict or None
        If not None, dict containing discrete input values.
    discrete_outputs : dict or None
        If not None, dict containing discrete output values.
    """
    deriv_vals = inst._get_jac_func()(*inst._get_compute_primal_invals(inputs, outputs,
                                                                       discrete_inputs))
    nested_tup = isinstance(deriv_vals, tuple) and len(deriv_vals) > 0 and \
        isinstance(deriv_vals[0], tuple)

    nof = len(inst._var_rel_names['output'])
    ofidx = len(inst._discrete_outputs) - 1
    for ofname in inst._var_rel_names['output']:
        ofidx += 1
        ofmeta = inst._var_rel2meta[ofname]
        for wrtidx, wrtname in enumerate(chain(inst._var_rel_names['input'],
                                               inst._var_rel_names['output'])):
            key = (ofname, wrtname)
            if key not in partials:
                # FIXME: this means that we computed a derivative that we didn't need
                continue

            wrtmeta = inst._var_rel2meta[wrtname]
            dvals = deriv_vals
            # if there's only one 'of' value, we only take the indexed value if the
            # return value of compute_primal is single entry tuple. If a single array or
            # scalar is returned, we don't apply the 'of' index.
            if nof > 1 or nested_tup:
                dvals = dvals[ofidx]

            # print(ofidx, ofname, ofmeta['shape'], wrtidx, wrtname, wrtmeta['shape'],
            #       'subjac_shape', dvals[wrtidx].shape)
            dvals = dvals[wrtidx].reshape(ofmeta['size'], wrtmeta['size'])

            sjmeta = partials.get_metadata(key)
            if sjmeta['rows'] is None:
                partials[ofname, wrtname] = dvals
            else:
                partials[ofname, wrtname] = dvals[sjmeta['rows'], sjmeta['cols']]


def apply_linear(inst, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
    r"""
    Compute jac-vector product (implicit). The model is assumed to be in an unscaled state.

    If mode is:
        'fwd': (d_inputs, d_outputs) \|-> d_residuals

        'rev': d_residuals \|-> (d_inputs, d_outputs)

    Parameters
    ----------
    inst : ImplicitComponent
        The component instance.
    inputs : Vector
        Unscaled, dimensional input variables read via inputs[key].
    outputs : Vector
        Unscaled, dimensional output variables read via outputs[key].
    d_inputs : Vector
        See inputs; product must be computed only if var_name in d_inputs.
    d_outputs : Vector
        See outputs; product must be computed only if var_name in d_outputs.
    d_residuals : Vector
        See outputs.
    mode : str
        Either 'fwd' or 'rev'.
    """
    if mode == 'fwd':
        dx = tuple(chain(d_inputs.values(), d_outputs.values()))
        full_invals = tuple(inst._get_compute_primal_invals(inputs, outputs, inst._discrete_inputs))
        x = full_invals[:len(dx)]
        other = full_invals[len(dx):]
        _, deriv_vals = jax.jvp(lambda *args: inst.compute_primal(*args, *other),
                                primals=x, tangents=dx)
        if isinstance(deriv_vals, tuple):
            d_residuals.set_vals(deriv_vals)
        else:
            d_residuals.asarray()[:] = deriv_vals.flatten()
    else:
        inhash = (inputs.get_hash(), outputs.get_hash()) + tuple(inst._discrete_inputs.values())
        if inhash != inst._vjp_hash:
            # recompute vjp function only if inputs or outputs have changed
            dx = tuple(chain(d_inputs.values(), d_outputs.values()))
            full_invals = tuple(inst._get_compute_primal_invals(inputs, outputs,
                                                                inst._discrete_inputs))
            x = full_invals[:len(dx)]
            other = full_invals[len(dx):]
            _, inst._vjp_fun = jax.vjp(lambda *args: inst.compute_primal(*args, *other), *x)
            inst._vjp_hash = inhash

            if inst._compute_primals_out_shape is None:
                shape = jax.eval_shape(lambda *args: inst.compute_primal(*args, *other), *x)
                if isinstance(shape, tuple):
                    shape = (tuple(s.shape for s in shape), True, len(inst._var_rel_names['input']))
                else:
                    shape = (shape.shape, False, len(inst._var_rel_names['input']))
                inst._compute_primals_out_shape = shape

        shape, istup, ninputs = inst._compute_primals_out_shape

        if istup:
            deriv_vals = (inst._vjp_fun(tuple(d_residuals.values())))
        else:
            deriv_vals = inst._vjp_fun(tuple(d_residuals.values())[0])

        d_inputs.set_vals(deriv_vals[:ninputs])
        d_outputs.set_vals(deriv_vals[ninputs:])
