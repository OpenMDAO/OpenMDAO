"""Define the ExplicitComponent class."""

import numpy as np

from openmdao.core.component import Component, _full_slice
from openmdao.utils.class_util import overrides_method
from openmdao.utils.general_utils import ContainsAll
from openmdao.recorders.recording_iteration_stack import Recording

_inst_functs = ['compute_jacvec_product', 'compute_multi_jacvec_product']


class ExplicitComponent(Component):
    """
    Class to inherit from when all output variables are explicit.

    Attributes
    ----------
    _inst_functs : dict
        Dictionary of names mapped to bound methods.
    _has_compute_partials : bool
        If True, the instance overrides compute_partials.
    """

    def __init__(self, **kwargs):
        """
        Store some bound methods so we can detect runtime overrides.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        super(ExplicitComponent, self).__init__(**kwargs)

        self._inst_functs = {name: getattr(self, name, None) for name in _inst_functs}
        self._has_compute_partials = overrides_method('compute_partials', self, ExplicitComponent)
        self.options.undeclare('assembled_jac_type')

    def _configure(self):
        """
        Configure this system to assign children settings and detect if matrix_free.
        """
        new_jacvec_prod = getattr(self, 'compute_jacvec_product', None)
        new_multi_jacvec_prod = getattr(self, 'compute_multi_jacvec_product', None)

        self.supports_multivecs = (overrides_method('compute_multi_jacvec_product',
                                                    self, ExplicitComponent) or
                                   (new_multi_jacvec_prod is not None and
                                    new_multi_jacvec_prod !=
                                    self._inst_functs['compute_multi_jacvec_product']))
        self.matrix_free = self.supports_multivecs or (
            overrides_method('compute_jacvec_product', self, ExplicitComponent) or
            (new_jacvec_prod is not None and
             new_jacvec_prod != self._inst_functs['compute_jacvec_product']))

    def _get_partials_varlists(self):
        """
        Get lists of 'of' and 'wrt' variables that form the partial jacobian.

        Returns
        -------
        tuple(list, list)
            'of' and 'wrt' variable lists.
        """
        of = list(self._var_rel_names['output'])
        wrt = list(self._var_rel_names['input'])
        return of, wrt

    def _get_partials_var_sizes(self):
        """
        Get sizes of 'of' and 'wrt' variables that form the partial jacobian.

        Returns
        -------
        tuple(ndarray, ndarray)
            'of' and 'wrt' variable sizes.
        """
        iproc = self.comm.rank
        out_sizes = self._var_sizes['nonlinear']['output'][iproc]
        in_sizes = self._var_sizes['nonlinear']['input'][iproc]
        return out_sizes, in_sizes

    def _jacobian_wrt_iter(self, wrt_matches=None):
        """
        Iterate over (name, offset, end, idxs) for each column var in the systems's jacobian.

        Parameters
        ----------
        wrt_matches : set or None
            Only include row vars that are contained in this set.  This will determine what
            the actual offsets are, i.e. the offsets will be into a reduced jacobian
            containing only the matching columns.
        """
        if wrt_matches is None:
            wrt_matches = ContainsAll()
        abs2meta = self._var_allprocs_abs2meta
        offset = end = 0
        for wrt in self._var_allprocs_abs_names['input']:
            if wrt in wrt_matches:
                end += abs2meta[wrt]['size']
                yield wrt, offset, end, _full_slice
                offset = end

    def _setup_partials(self):
        """
        Call setup_partials in components.
        """
        super(ExplicitComponent, self)._setup_partials()

        abs2meta = self._var_abs2meta
        abs2prom_out = self._var_abs2prom['output']

        # Note: These declare calls are outside of setup_partials so that users do not have to
        # call the super version of setup_partials. This is still in the final setup.
        for out_abs in self._var_abs_names['output']:
            meta = abs2meta[out_abs]
            out_name = abs2prom_out[out_abs]
            arange = np.arange(meta['size'])

            # No need to FD outputs wrt other outputs
            abs_key = (out_abs, out_abs)
            if abs_key in self._subjacs_info:
                if 'method' in self._subjacs_info[abs_key]:
                    del self._subjacs_info[abs_key]['method']

            dct = {
                'rows': arange,
                'cols': arange,
                'value': np.full(meta['size'], -1.),
                'dependent': True,
            }

            # ExplicitComponent jacobians have -1 on the diagonal.
            if arange.size > 0:
                self._declare_partials(out_name, out_name, dct)

    def _setup_jacobians(self, recurse=True):
        """
        Set and populate jacobian.

        Parameters
        ----------
        recurse : bool
            If True, setup jacobians in all descendants. (ignored)
        """
        if self._has_approx and self._use_derivatives:
            self._set_approx_partials_meta()

    def add_output(self, name, val=1.0, shape=None, units=None, res_units=None, desc='',
                   lower=None, upper=None, ref=1.0, ref0=0.0, res_ref=None, tags=None):
        """
        Add an output variable to the component.

        For ExplicitComponent, res_ref defaults to the value in res unless otherwise specified.

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        val : float or list or tuple or ndarray
            The initial value of the variable being added in user-defined units. Default is 1.0.
        shape : int or tuple or list or None
            Shape of this variable, only required if val is not an array.
            Default is None.
        units : str or None
            Units in which the output variables will be provided to the component during execution.
            Default is None, which means it has no units.
        res_units : str or None
            Units in which the residuals of this output will be given to the user when requested.
            Default is None, which means it has no units.
        desc : str
            description of the variable.
        lower : float or list or tuple or ndarray or None
            lower bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no lower bound.
            Default is None.
        upper : float or list or tuple or ndarray or None
            upper bound(s) in user-defined units. It can be (1) a float, (2) an array_like
            consistent with the shape arg (if given), or (3) an array_like matching the shape of
            val, if val is array_like. A value of None means this output has no upper bound.
            Default is None.
        ref : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 1. Default is 1.
        ref0 : float
            Scaling parameter. The value in the user-defined units of this output variable when
            the scaled value is 0. Default is 0.
        res_ref : float
            Scaling parameter. The value in the user-defined res_units of this output's residual
            when the scaled value is 1. Default is None, which means residual scaling matches
            output scaling.
        tags : str or list of strs
            User defined tags that can be used to filter what gets listed when calling
            list_inputs and list_outputs and also when listing results from case recorders.

        Returns
        -------
        dict
            metadata for added variable
        """
        if res_ref is None:
            res_ref = ref

        return super(ExplicitComponent, self).add_output(name,
                                                         val=val, shape=shape, units=units,
                                                         res_units=res_units, desc=desc,
                                                         lower=lower, upper=upper,
                                                         ref=ref, ref0=ref0, res_ref=res_ref,
                                                         tags=tags)

    def _approx_subjac_keys_iter(self):
        for abs_key, meta in self._subjacs_info.items():
            if 'method' in meta:
                method = meta['method']
                if (method is not None and method in self._approx_schemes and abs_key[1]
                        not in self._outputs._views_flat):
                    yield abs_key

    def _apply_nonlinear(self):
        """
        Compute residuals. The model is assumed to be in a scaled state.
        """
        outputs = self._outputs
        residuals = self._residuals
        with Recording(self.pathname + '._apply_nonlinear', self.iter_count, self):
            with self._unscaled_context(outputs=[outputs], residuals=[residuals]):
                residuals.set_vec(outputs)

                # Sign of the residual is minus the sign of the output vector.
                residuals *= -1.0

                self._inputs.read_only = True
                try:
                    if self._discrete_inputs or self._discrete_outputs:
                        self.compute(self._inputs, self._outputs, self._discrete_inputs,
                                     self._discrete_outputs)
                    else:
                        self.compute(self._inputs, self._outputs)
                finally:
                    self._inputs.read_only = False

                residuals += outputs
                outputs -= residuals

    def _solve_nonlinear(self):
        """
        Compute outputs. The model is assumed to be in a scaled state.
        """
        with Recording(self.pathname + '._solve_nonlinear', self.iter_count, self):
            with self._unscaled_context(outputs=[self._outputs], residuals=[self._residuals]):
                self._residuals.set_const(0.0)
                self._inputs.read_only = True
                try:
                    if self._discrete_inputs or self._discrete_outputs:
                        self.compute(self._inputs, self._outputs, self._discrete_inputs,
                                     self._discrete_outputs)
                    else:
                        self.compute(self._inputs, self._outputs)
                finally:
                    self._inputs.read_only = False

    def _apply_linear(self, jac, vec_names, rel_systems, mode, scope_out=None, scope_in=None):
        """
        Compute jac-vec product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            If None, use local jacobian, else use jac.
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.
        mode : str
            'fwd' or 'rev'.
        scope_out : set or None
            Set of absolute output names in the scope of this mat-vec product.
            If None, all are in the scope.
        scope_in : set or None
            Set of absolute input names in the scope of this mat-vec product.
            If None, all are in the scope.
        """
        J = self._jacobian if jac is None else jac

        for vec_name in vec_names:
            if vec_name not in self._rel_vec_names:
                continue

            with self._matvec_context(vec_name, scope_out, scope_in, mode) as vecs:
                d_inputs, d_outputs, d_residuals = vecs

                # Jacobian and vectors are all scaled, unitless
                J._apply(self, d_inputs, d_outputs, d_residuals, mode)

                # if we're not matrix free, we can skip the bottom of
                # this loop because compute_jacvec_product does nothing.
                if not self.matrix_free:
                    continue

                # Jacobian and vectors are all unscaled, dimensional
                with self._unscaled_context(
                        outputs=[self._outputs], residuals=[d_residuals]):

                    # set appropriate vectors to read_only to help prevent user error
                    self._inputs.read_only = True
                    if mode == 'fwd':
                        d_inputs.read_only = True
                    elif mode == 'rev':
                        d_residuals.read_only = True

                    try:
                        args = [self._inputs, d_inputs, d_residuals, mode]
                        if self._discrete_inputs:
                            args.append(self._discrete_inputs)

                        # We used to negate the residual here, and then re-negate after the hook
                        if d_inputs._ncol > 1:
                            if self.supports_multivecs:
                                self.compute_multi_jacvec_product(*args)
                            else:
                                for i in range(d_inputs._ncol):
                                    # need to make the multivecs look like regular single vecs
                                    # since the component doesn't know about multivecs.
                                    d_inputs._icol = i
                                    d_residuals._icol = i
                                    self.compute_jacvec_product(*args)
                                d_inputs._icol = None
                                d_residuals._icol = None
                        else:
                            self.compute_jacvec_product(*args)
                    finally:
                        self._inputs.read_only = False
                        d_inputs.read_only = d_residuals.read_only = False

    def _solve_linear(self, vec_names, mode, rel_systems):
        """
        Apply inverse jac product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.

        """
        for vec_name in vec_names:
            if vec_name in self._rel_vec_names:
                d_outputs = self._vectors['output'][vec_name]
                d_residuals = self._vectors['residual'][vec_name]

                if mode == 'fwd':
                    if self._has_resid_scaling:
                        with self._unscaled_context(outputs=[d_outputs],
                                                    residuals=[d_residuals]):
                            d_outputs.set_vec(d_residuals)
                    else:
                        d_outputs.set_vec(d_residuals)

                    # ExplicitComponent jacobian defined with -1 on diagonal.
                    d_outputs *= -1.0

                else:  # rev
                    if self._has_resid_scaling:
                        with self._unscaled_context(outputs=[d_outputs],
                                                    residuals=[d_residuals]):
                            d_residuals.set_vec(d_outputs)
                    else:
                        d_residuals.set_vec(d_outputs)

                    # ExplicitComponent jacobian defined with -1 on diagonal.
                    d_residuals *= -1.0

    def _linearize(self, jac=None, sub_do_ln=False):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            Ignored.
        sub_do_ln : boolean
            Flag indicating if the children should call linearize on their linear solvers.
        """
        if not (self._has_compute_partials or self._approx_schemes):
            return

        self._check_first_linearize()

        with self._unscaled_context(outputs=[self._outputs], residuals=[self._residuals]):
            # Computing the approximation before the call to compute_partials allows users to
            # override FD'd values.
            for approximation in self._approx_schemes.values():
                approximation.compute_approximations(self, jac=self._jacobian)

            if self._has_compute_partials:
                self._inputs.read_only = True

                # We don't need to set the _system attribute on jac here because jac (if not None)
                # shares the _subjacs_info metadata with our _jacobian, and our _jacobian knows
                # how to properly convert relative names (used by the component in compute_partials)
                # to absolute names (used by all jacobians internally).
                try:
                    # We used to negate the jacobian here, and then re-negate after the hook.
                    if self._discrete_inputs:
                        self.compute_partials(self._inputs, self._jacobian, self._discrete_inputs)
                    else:
                        self.compute_partials(self._inputs, self._jacobian)
                finally:
                    self._inputs.read_only = False

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Compute outputs given inputs. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        discrete_outputs : dict or None
            If not None, dict containing discrete output values.
        """
        pass

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        partials : Jacobian
            sub-jac components written to partials[output_name, input_name]
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        """
        pass

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None):
        r"""
        Compute jac-vector product. The model is assumed to be in an unscaled state.

        If mode is:
            'fwd': d_inputs \|-> d_outputs

            'rev': d_outputs \|-> d_inputs

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        d_inputs : Vector
            see inputs; product must be computed only if var_name in d_inputs
        d_outputs : Vector
            see outputs; product must be computed only if var_name in d_outputs
        mode : str
            either 'fwd' or 'rev'
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        """
        pass
