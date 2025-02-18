"""
An ExplicitComponent that uses JAX for derivatives.
"""

import sys
from types import MethodType

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.om_warnings import issue_warning
from openmdao.utils.jax_utils import jax, jit, \
    _jax_register_pytree_class, _compute_sparsity, get_vmap_tangents, \
    _update_subjac_sparsity, _jax_derivs2partials, _uncompress_jac, _jax2np, \
    _ensure_returns_tuple


class JaxExplicitComponent(ExplicitComponent):
    """
    Base class for explicit components when using JAX for derivatives.

    Parameters
    ----------
    fallback_derivs_method : str
        The method to use if JAX is not available. Default is 'fd'.
    **kwargs : dict
        Additional arguments to be passed to the base class.

    Attributes
    ----------
    _tangents : dict
        The tangents for the inputs and outputs.
    _sparsity : coo_matrix or None
        The sparsity of the Jacobian.
    _jac_func_ : function or None
        The function that computes the jacobian.
    _jac_colored_ : function or None
        The function that computes the colored jacobian.
    _static_hash : tuple
        The hash of the static values.
    _orig_compute_primal : function
        The original compute_primal method.
    _ret_tuple_compute_primal : function
        The compute_primal method that returns a tuple.
    """

    def __init__(self, fallback_derivs_method='fd', **kwargs):  # noqa
        if sys.version_info < (3, 9):
            raise RuntimeError("JaxExplicitComponent requires Python 3.9 or newer.")
        super().__init__(**kwargs)

        self._re_init()

        if self.compute_primal is None:
            raise RuntimeError(f"{self.msginfo}: compute_primal is not defined for this component.")

        self._orig_compute_primal = self.compute_primal
        self._ret_tuple_compute_primal = \
            MethodType(_ensure_returns_tuple(self.compute_primal.__func__), self)
        self.compute_primal = self._ret_tuple_compute_primal

        # if derivs_method is explicitly passed in, just use it
        if 'derivs_method' in kwargs:
            return

        if jax:
            self.options['derivs_method'] = 'jax'
        else:
            issue_warning(f"{self.msginfo}: JAX is not available, so '{fallback_derivs_method}' "
                          "will be used for derivatives.")
            self.options['derivs_method'] = fallback_derivs_method

    def _re_init(self):
        """
        Re-initialize the component for a new run.
        """
        self._tangents = {'fwd': None, 'rev': None}
        self._sparsity = None
        self._jac_func_ = None
        self._static_hash = None
        self._jac_colored_ = None

    def _setup_jax(self):
        """
        Set up the jax interface for this component.

        This happens in final_setup after all var sizes and partials are set.
        """
        _jax_register_pytree_class(self.__class__)

        self._re_init()

        self.compute = self._jax_compute

        if not self._discrete_inputs and not self.get_self_statics():
            # avoid unnecessary statics checks
            self._statics_changed = self._statics_noop

        if self.matrix_free:
            if self._coloring_info.use_coloring():
                issue_warning(f"{self.msginfo}: coloring has been set but matrix_free is True, "
                              "so coloring will be ignored.")
                self._coloring_info.deactivate()
            self.compute_jacvec_product = self._compute_jacvec_product
        else:
            if self._coloring_info.use_coloring():
                # ensure coloring (and sparsity) is computed before partials
                self._get_coloring()
            else:
                if not self._declared_partials_patterns:
                    # auto determine subjac sparsities
                    self.compute_sparsity()
            self.compute_partials = self._compute_partials
            self._has_compute_partials = True

    def _statics_changed(self, discrete_inputs):
        """
        Determine if jitting is needed based on changes in static values since the last call.

        Parameters
        ----------
        discrete_inputs : dict
            dict containing discrete input values.

        Returns
        -------
        bool
            Whether jitting is needed.
        """
        # if static values change, we need to rejit
        inhash = tuple(discrete_inputs) if discrete_inputs else ()
        inhash = inhash + self.get_self_statics()
        if inhash != self._static_hash:
            self._static_hash = inhash
            return True
        return False

    def _statics_noop(self, discrete_inputs):
        """
        Use this function if the component has no discrete inputs or self statics.

        Parameters
        ----------
        discrete_inputs : dict
            dict containing discrete input values.

        Returns
        -------
        bool
            Always returns False.
        """
        return False

    def _jax_compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Compute outputs given inputs. The model is assumed to be in an unscaled state.

        An inherited component may choose to either override this function or to define a
        compute_primal function.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        discrete_inputs : dict-like or None
            If not None, dict-like object containing discrete input values.
        discrete_outputs : dict-like or None
            If not None, dict-like object containing discrete output values.
        """
        if discrete_outputs:
            returns = self.compute_primal(*self._get_compute_primal_invals(inputs, discrete_inputs))
            outputs.set_vals(returns[:outputs.nvars()])
            self._discrete_outputs.set_vals(returns[outputs.nvars():])
        else:
            outputs.set_vals(self.compute_primal(*self._get_compute_primal_invals(inputs,
                                                                                  discrete_inputs)))

    def _get_differentiable_compute_primal(self, discrete_inputs):
        """
        Get the compute_primal function for the jacobian.

        This version of the compute primal should take no discrete inputs and return no discrete
        outputs. It will be called when computing the jacobian.

        Parameters
        ----------
        discrete_inputs : iter of discrete values
            The discrete input values.

        Returns
        -------
        function
            The compute_primal function to be used to compute the jacobian.
        """
        # exclude the discrete inputs from the inputs and the discrete outputs from the outputs
        if discrete_inputs:
            if self._discrete_outputs:
                ncontouts = self._outputs.nvars()

                def differentiable_compute_primal(*contvals):
                    return self.compute_primal(*contvals, *discrete_inputs)[:ncontouts]

            else:

                def differentiable_compute_primal(*contvals):
                    return self.compute_primal(*contvals, *discrete_inputs)

            return differentiable_compute_primal

        elif self._discrete_outputs:
            ncontouts = self._outputs.nvars()

            def differentiable_compute_primal(*contvals):
                return self.compute_primal(*contvals)[:ncontouts]

            return differentiable_compute_primal

        return self.compute_primal

    def _get_jax_compute_primal(self, discrete_inputs, need_jit):
        """
        Get the jax version of the compute_primal method.
        """
        compute_primal = self._ret_tuple_compute_primal.__func__

        if need_jit:
            # jit the compute_primal method
            idx = self._inputs.nvars() + 1
            if discrete_inputs:
                static_argnums = list(range(idx, idx + len(discrete_inputs)))
            else:
                static_argnums = []
            compute_primal = jit(compute_primal, static_argnums=static_argnums)

        return MethodType(compute_primal, self)

    def _update_jac_functs(self, discrete_inputs):
        """
        Update the jax function that computes the jacobian for this component if necessary.

        An update is required if jitting is enabled and any static values have changed.

        Parameters
        ----------
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.

        Returns
        -------
        tuple
            The jax functions (jax_compute_primal, jax_compute_jac). Note that these are not
            methods, but rather functions. To make them methods you need to assign
            MethodType(function, self) to an attribute of the instance.
        """
        need_jit = self.options['use_jit']
        if need_jit and self._statics_changed(discrete_inputs):
            self._jac_func_ = None

        if self._jac_func_ is None:
            self.compute_primal = self._get_jax_compute_primal(discrete_inputs, need_jit)
            differentiable_cp = self._get_differentiable_compute_primal(discrete_inputs)

            if self._coloring_info.use_coloring():
                if self._coloring_info.coloring is None:
                    # need to dynamically compute the coloring first
                    self._compute_coloring()

                if self.best_partial_deriv_direction() == 'fwd':
                    self._get_tangents('fwd', self._coloring_info.coloring)

                    # here we'll use the same inputs and a single tangent vector from the vmap
                    # batch to compute a single jvp, which corresponds to a column of the
                    # jacobian (the compressed jacobian in the colored case).
                    def jvp_at_point(tangent, icontvals):
                        # [1] is the derivative, [0] is the primal (we don't need the primal)
                        return jax.jvp(differentiable_cp, icontvals, tangent)[1]

                    # vectorize over the last axis of the tangent vectors and use the same
                    # inputs for all cases.
                    self._jac_func_ = jax.vmap(jvp_at_point, in_axes=[-1, None], out_axes=-1)
                    self._jac_colored_ = self._jacfwd_colored
                else:  # rev
                    def vjp_at_point(cotangent, icontvals):
                        # Returns primal and a function to compute VJP so just take [1],
                        # the vjp function
                        return jax.vjp(differentiable_cp, *icontvals)[1](cotangent)

                    self._get_tangents('rev', self._coloring_info.coloring)

                    # Batch over last axis of cotangents
                    self._jac_func_ = jax.vmap(vjp_at_point, in_axes=[-1, None], out_axes=-1)
                    self._jac_colored_ = self._jacrev_colored
            else:
                self._jac_colored_ = None
                fjax = jax.jacfwd if self.best_partial_deriv_direction() == 'fwd' else jax.jacrev
                wrt_idxs = list(range(len(self._var_abs2meta['input'])))
                self._jac_func_ = fjax(differentiable_cp, argnums=wrt_idxs)

            if need_jit:
                self._jac_func_ = jax.jit(self._jac_func_)

    def declare_coloring(self, **kwargs):
        """
        Declare coloring for this component.

        The 'method' argument is set to 'jax' and passed to the base class.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments to be passed to the base class.
        """
        if 'method' in kwargs and kwargs['method'] != self.options['derivs_method']:
            raise ValueError(f"method must be '{self.options['derivs_method']}' for this component "
                             "but got '{kwargs['method']}'.")
        kwargs['method'] = self.options['derivs_method']
        super().declare_coloring(**kwargs)

    # we define _compute_partials here and possibly later rename it to compute_partials instead of
    # making this the base class version as we did with compute, because the existence of a
    # compute_partials method that is not the base class method is used to determine if a given
    # component computes its own partials.
    def _compute_partials(self, inputs, partials, discrete_inputs=None):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        self : ImplicitComponent
            The component instance.
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Sub-jac components written to partials[output_name, input_name]..
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        """
        discrete_inputs = discrete_inputs.values() if discrete_inputs else ()
        self._update_jac_functs(discrete_inputs)

        if self._jac_colored_ is not None:
            return self._jac_colored_(inputs, partials, discrete_inputs)

        derivs = self._jac_func_(*inputs.values())

        # check to see if we even need this with jax.  A jax component doesn't need to map string
        # keys to partials.  We could just use the jacobian as an array to compute the derivatives.
        # Maybe make a simple JaxJacobian that is just a thin wrapper around the jacobian array.
        # The only issue is do higher level jacobians need the subjacobian info?
        _jax_derivs2partials(self, derivs, partials, self._var_rel_names['output'],
                             self._var_rel_names['input'])

    def _jacfwd_colored(self, inputs, partials, discrete_inputs=None):
        """
        Compute the forward jacobian using vmap with jvp and coloring.

        Parameters
        ----------
        inputs : dict
            The inputs to the component.
        partials : dict
            The partials to compute.
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        """
        J = self._jac_func_(self._tangents['fwd'], tuple(inputs.values()))
        partials.set_dense_jac(self, _uncompress_jac(self, _jax2np(J), 'fwd'))

    def _jacrev_colored(self, inputs, partials, discrete_inputs=None):
        """
        Compute the reverse jacobian using vmap with vjp and coloring.

        Parameters
        ----------
        inputs : dict
            The inputs to the component.
        partials : dict
            The partials to compute.
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        """
        J = self._jac_func_(self._tangents['rev'], tuple(inputs.values()))
        partials.set_dense_jac(self, _uncompress_jac(self, _jax2np(J).T, 'rev'))

    def compute_sparsity(self, direction=None, num_iters=1, perturb_size=1e-9):
        """
        Get the sparsity of the Jacobian.

        Parameters
        ----------
        direction : str
            The direction to compute the sparsity for.
        num_iters : int
            The number of times to run the perturbation iteration.
        perturb_size : float
            The size of the perturbation to use.

        Returns
        -------
        coo_matrix
            The sparsity of the Jacobian.
        """
        if self._sparsity is None:
            if self.options['derivs_method'] == 'jax':
                self._sparsity = _compute_sparsity(self, direction, num_iters, perturb_size)
            else:
                self._sparsity = super().compute_sparsity(direction=direction, num_iters=num_iters,
                                                          perturb_size=perturb_size)
        return self._sparsity

    def _update_subjac_sparsity(self, sparsity_iter):
        if self.options['derivs_method'] == 'jax':
            _update_subjac_sparsity(sparsity_iter, self.pathname, self._subjacs_info)
        else:
            super()._update_subjac_sparsity(sparsity_iter)

    def _get_tangents(self, direction, coloring=None):
        """
        Get the tangents for the inputs or outputs.

        If coloring is not None, then the tangents will be compressed based on the coloring.

        Parameters
        ----------
        direction : str
            The direction to get the tangents for.
        coloring : Coloring
            The coloring to use.

        Returns
        -------
        tuple
            The tangents.
        """
        if self._tangents[direction] is None:
            if direction == 'fwd':
                self._tangents[direction] = \
                    get_vmap_tangents(tuple(self._inputs.values()),
                                      direction, fill=1.,
                                      coloring=coloring)
            else:
                self._tangents[direction] = \
                    get_vmap_tangents(tuple(self._outputs.values()),
                                      direction, fill=1.,
                                      coloring=coloring)
        return self._tangents[direction]

    def _compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None):
        r"""
        Compute jac-vector product (explicit). The model is assumed to be in an unscaled state.

        If mode is:
            'fwd': d_inputs \|-> d_outputs

            'rev': d_outputs \|-> d_inputs

        Parameters
        ----------
        self : ExplicitComponent
            The component instance.
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        d_inputs : Vector
            See inputs; product must be computed only if var_name in d_inputs.
        d_outputs : Vector
            See outputs; product must be computed only if var_name in d_outputs.
        mode : str
            Either 'fwd' or 'rev'.
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        """
        if mode == 'fwd':
            dx = tuple(d_inputs.values())
            full_invals = tuple(self._get_compute_primal_invals(inputs, discrete_inputs))
            x = full_invals[:len(dx)]
            other = full_invals[len(dx):]
            _, deriv_vals = jax.jvp(lambda *args: self.compute_primal(*args, *other),
                                    primals=x, tangents=dx)
            d_outputs.set_vals(deriv_vals)
        else:
            inhash = ((inputs.get_hash(),) + tuple(self._discrete_inputs.values()) +
                      self.get_self_statics())
            if inhash != self._static_hash:
                ncont_ins = d_inputs.nvars()
                full_invals = tuple(self._get_compute_primal_invals(inputs, discrete_inputs))
                x = full_invals[:ncont_ins]
                other = full_invals[ncont_ins:]
                # recompute vjp function if inputs have changed
                _, self._vjp_fun = jax.vjp(lambda *args: self.compute_primal(*args, *other), *x)
                self._static_hash = inhash

            deriv_vals = self._vjp_fun(tuple(d_outputs.values()) +
                                       tuple(self._discrete_outputs.values()))

            d_inputs.set_vals(deriv_vals)
