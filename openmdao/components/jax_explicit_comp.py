"""
An ExplicitComponent that uses JAX for derivatives.
"""

import sys
from types import MethodType

from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.om_warnings import issue_warning
from openmdao.utils.jax_utils import jax, jit, ReturnChecker, \
    _jax_register_pytree_class, _compute_sparsity, get_vmap_tangents, _compute_jac, \
        _update_subjac_sparsity


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
    _compute_primal_returns_tuple : bool
        Whether the compute_primal method returns a tuple.
    _tangents : dict
        The tangents for the inputs and outputs.
    """

    def __init__(self, fallback_derivs_method='fd', **kwargs):  # noqa
        if sys.version_info < (3, 9):
            raise RuntimeError("JaxExplicitComponent requires Python 3.9 or newer.")
        super().__init__(**kwargs)

        self._compute_primal_returns_tuple = False
        self._tangents = {'fwd': None, 'rev': None}
        self._sparsity = None

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

        This happens in final_setup after all var sizes and partials are set.
        """
        if self.compute_primal is None:
            raise RuntimeError(f"{self.msginfo}: compute_primal is not defined for this component.")

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
                if self.best_partial_deriv_direction() == 'fwd':
                    self.compute_partials = self._jacfwd_colored
                else:
                    self.compute_partials = self._jacrev_colored
            else:
                if not self._declared_partials_patterns:
                    # auto determine subjac sparsities
                    self.compute_sparsity()
                self.compute_partials = self._compute_partials
            self._has_compute_partials = True

        # determine if the compute_primal method returns a tuple
        self._compute_primal_returns_tuple = ReturnChecker(self.compute_primal).returns_tuple()

        if self.options['use_jit']:
            # jit the compute_primal method
            idx = len(self._var_rel_names['input']) + 1
            static_argnums = list(range(idx, idx + len(self._discrete_inputs)))
            self.compute_primal = MethodType(jit(self.compute_primal.__func__,
                                                 static_argnums=static_argnums), self)

        _jax_register_pytree_class(self.__class__)

    def _get_jac_func(self):
        """
        Return the jacobian function for this component.

        In forward mode without coloring, jax.jacfwd is used, and in reverse mode without coloring,
        jax.jacrev is used.

        If coloring is used, then the jacobian is computed using vmap with jvp or vjp depending on
        the direction, with the tangents determined by the coloring information.

        Returns
        -------
        function
            The jacobian function.
        """
        # TODO: modify this to use relevance and possibly compile multiple jac functions depending
        # on DV/response so that we don't compute any derivatives that are always zero.
        if self._jac_func_ is None:
            if self._coloring_info.use_coloring():
                if self._coloring_info.coloring is None:
                    # need to dynamically compute the coloring first
                    self._compute_coloring()

                if self.best_partial_deriv_direction() == 'fwd':
                    self._jac_func_ = self._jacfwd_colored
                else:
                    self._jac_func_ = self._jacrev_colored
            else:  # just use jacfwd or jacrev
                fjax = jax.jacfwd if self.best_partial_deriv_direction() == 'fwd' else jax.jacrev
                nstatic = len(self._discrete_inputs)
                wrt_idxs = list(range(1, len(self._var_abs2meta['input']) + 1))
                self._jac_func_ = MethodType(fjax(self.compute_primal.__func__, argnums=wrt_idxs),
                                             self)

                if self.options['use_jit']:
                    static_argnums = tuple(range(1 + len(wrt_idxs), 1 + len(wrt_idxs) + nstatic))
                    self._jac_func_ = MethodType(jit(self._jac_func_.__func__,
                                                     static_argnums=static_argnums), self)

        return self._jac_func_

    def declare_coloring(self, **kwargs):
        """
        Declare coloring for this component.

        The 'method' argument is set to 'jax' and passed to the base class.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments to be passed to the base class.
        """
        kwargs['method'] = 'jax'
        super().declare_coloring(**kwargs)

    # we define compute_partials here instead of making this the base class version as we
    # did with compute, because the existence of a compute_partials method that is not the
    # base class method is used to determine if a given component computes its own partials.
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
        derivs = self._get_jac_func()(*self._get_compute_primal_invals(inputs,
                                                                       self._discrete_inputs))
        # check to see if we even need this with jax.  A jax component doesn't need to map string
        # keys to partials.  We could just use the jacobian as an array to compute the derivatives.
        # Maybe make a simple JaxJacobian that is just a thin wrapper around the jacobian array.
        # The only issue is do higher level jacobians need the subjacobian info?
        self.jax_derivs2partials(derivs, partials)

    def _jacfwd_colored(self, inputs, partials, discrete_inputs=None):
        """
        Compute the forward jacobian using vmap with jvp.

        Parameters
        ----------
        inputs : dict
            The inputs to the component.
        partials : dict
            The partials to compute.
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        """
        J = _compute_jac(self, 'fwd', inputs=inputs, discrete_inputs=discrete_inputs)
        self.dense_jac2partials(J, partials)

    def _jacrev_colored(self, inputs, partials, discrete_inputs=None):
        """
        Compute the reverse jacobian using vmap with vjp.

        Parameters
        ----------
        inputs : dict
            The inputs to the component.
        partials : dict
            The partials to compute.
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        """
        J = _compute_jac(self, 'rev', inputs=inputs, discrete_inputs=discrete_inputs)
        self.dense_jac2partials(J, partials)

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
                self._tangents[direction] = get_vmap_tangents(tuple(self._inputs.values()),
                                                              direction, fill=1., coloring=coloring)
            else:
                self._tangents[direction] = get_vmap_tangents(tuple(self._outputs.values()),
                                                              direction, fill=1., coloring=coloring)
        return self._tangents[direction]

    def dense_jac2partials(self, J, partials):
        """
        Copy a dense Jacobian into partials.

        Parameters
        ----------
        J : ndarray
            The dense Jacobian.
        partials : dict
            The partials to copy the Jacobian into, keyed by (of_name, wrt_name).
        """
        ofstart = ofend = 0
        for ofname in self._var_rel_names['output']:
            ofmeta = self._var_rel2meta[ofname]
            ofend += ofmeta['size']
            wrtstart = wrtend = 0
            for wrtname in self._var_rel_names['input']:
                wrtmeta = self._var_rel2meta[wrtname]
                wrtend += wrtmeta['size']
                key = (ofname, wrtname)
                if key not in partials:
                    # FIXME: this means that we computed a derivative that we didn't need
                    continue

                dvals = J[ofstart:ofend, wrtstart:wrtend]
                sjmeta = partials.get_metadata(key)
                rows = sjmeta['rows']
                if rows is None:
                    partials[ofname, wrtname] = dvals
                else:
                    partials[ofname, wrtname] = dvals[rows, sjmeta['cols']]
                wrtstart = wrtend
            ofstart = ofend

    def jax_derivs2partials(self, deriv_vals, partials):
        """
        Copy JAX derivatives into partials.

        Parameters
        ----------
        deriv_vals : tuple
            The derivatives.
        partials : dict
            The partials to copy the derivatives into, keyed by (of_name, wrt_name).
        """
        nested_tup = isinstance(deriv_vals, tuple) and len(deriv_vals) > 0 and \
            isinstance(deriv_vals[0], tuple)
        nof = len(self._var_rel_names['output'])

        for ofidx, ofname in enumerate(self._var_rel_names['output']):
            ofmeta = self._var_rel2meta[ofname]
            for wrtidx, wrtname in enumerate(self._var_rel_names['input']):
                key = (ofname, wrtname)
                if key not in partials:
                    # FIXME: this means that we computed a derivative that we didn't need
                    continue

                wrtmeta = self._var_rel2meta[wrtname]
                dvals = deriv_vals
                # if there's only one 'of' value, we only take the indexed value if the
                # return value of compute_primal is single entry tuple. If a single array or
                # scalar is returned, we don't apply the 'of' index.
                if nof > 1 or nested_tup:
                    dvals = dvals[ofidx]

                dvals = dvals[wrtidx].reshape(ofmeta['size'], wrtmeta['size'])

                sjmeta = partials.get_metadata(key)
                rows = sjmeta['rows']
                if rows is None:
                    partials[ofname, wrtname] = dvals
                else:
                    partials[ofname, wrtname] = dvals[rows, sjmeta['cols']]

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
            if inhash != self._vjp_hash:
                ncont_ins = d_inputs.nvars()
                full_invals = tuple(self._get_compute_primal_invals(inputs, discrete_inputs))
                x = full_invals[:ncont_ins]
                other = full_invals[ncont_ins:]
                # recompute vjp function if inputs have changed
                _, self._vjp_fun = jax.vjp(lambda *args: self.compute_primal(*args, *other), *x)
                self._vjp_hash = inhash

            if self._compute_primal_returns_tuple:
                deriv_vals = self._vjp_fun(tuple(d_outputs.values()) +
                                           tuple(self._discrete_outputs.values()))
            else:
                deriv_vals = self._vjp_fun(tuple(d_outputs.values())[0])

            d_inputs.set_vals(deriv_vals)
