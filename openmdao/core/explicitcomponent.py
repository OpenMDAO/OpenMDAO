"""Define the ExplicitComponent class."""

from __future__ import division

import numpy as np
from six import itervalues, iteritems
from six.moves import range

from openmdao.core.component import Component
from openmdao.utils.class_util import overrides_method
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
    _negated_subjacs : set
        Keeps track of negated subjacs so we don't negate more than once during setup.
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
        self._negated_subjacs = set()

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

    def _setup_partials(self, recurse=True):
        """
        Call setup_partials in components.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(ExplicitComponent, self)._setup_partials()

        abs2meta = self._var_abs2meta
        abs2prom_out = self._var_abs2prom['output']

        # Note: These declare calls are outside of setup_partials so that users do not have to
        # call the super version of setup_partials. This is still in the final setup.
        other_names = []
        for out_abs in self._var_abs_names['output']:
            meta = abs2meta[out_abs]
            out_name = abs2prom_out[out_abs]
            arange = np.arange(meta['size'])

            # No need to FD outputs wrt other outputs
            abs_key = (out_abs, out_abs)
            if abs_key in self._subjacs_info:
                if 'method' in self._subjacs_info[abs_key]:
                    del self._subjacs_info[abs_key]['method']

            self._declare_partials(out_name, out_name, rows=arange, cols=arange, val=1.)

    def add_output(self, name, val=1.0, shape=None, units=None, res_units=None, desc='',
                   lower=None, upper=None, ref=1.0, ref0=0.0, res_ref=None, var_set=0):
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
        var_set : hashable object
            For advanced users only. ID or color for this variable, relevant for reconfigurability.
            Default is 0.

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
                                                         var_set=var_set)

    def _negate_jac(self, J):
        """
        Negate this component's part of the jacobian.

        Parameters
        ----------
        J : Jacobian
            The jacobian to be negated.
        """
        if J._subjacs:
            for res_name in self._var_abs_names['output']:
                for in_name in self._var_abs_names['input']:
                    abs_key = (res_name, in_name)
                    if abs_key in J._subjacs:
                        J._multiply_subjac(abs_key, -1.)

    def _setup_jacobians(self, parent_asm_jac=None):
        """
        Set and populate jacobians down through the system tree.

        Parameters
        ----------
        parent_asm_jac : AssembledJacobian or None
            The assembled jacobian from a parent group to populate for this system.
        """
        self._negated_subjacs = set()
        super(ExplicitComponent, self)._setup_jacobians(parent_asm_jac)

    def _set_partials_meta(self, jacs):
        """
        Set subjacobian info into our jacobian.

        Parameters
        ----------
        jacs : list of Jacobian
            Jacobians needing metadata update.
        """
        abs2prom = self._var_abs2prom

        # set context of jacobians once to avoid doing inside of loop
        old_systems = []
        for J in jacs:
            old_systems.append(J._system)
            J._system = self

        for abs_key, meta in iteritems(self._subjacs_info):
            if meta['value'] is None:
                out_size = self._var_abs2meta[abs_key[0]]['size']
                in_size = self._var_abs2meta[abs_key[1]]['size']
                meta['value'] = np.zeros((out_size, in_size))

            # if wrt is an input, we need to negate the subjac, but only once, since we don't
            # copy the subjacobian values in each jacobian (but we do copy the values in the
            # internal matrices)
            negate = abs_key[1] in abs2prom['input']
            for J in jacs:
                if negate:
                    if abs_key in self._negated_subjacs:
                        negate = False
                    else:
                        self._negated_subjacs.add(abs_key)
                J._set_partials_meta(abs_key, meta, negate)

            if 'method' in meta and meta['method']:
                # Don't approximate output wrt output.
                if abs_key[1] not in self._var_allprocs_abs_names['output']:
                    self._approx_schemes[meta['method']].add_approximation(abs_key, meta)

        # reset context of jacobians
        for i, J in enumerate(jacs):
            J._system = old_systems[i]

        for approx in itervalues(self._approx_schemes):
            approx._init_approximations()

    def _apply_nonlinear(self):
        """
        Compute residuals. The model is assumed to be in a scaled state.
        """
        outputs = self._outputs
        residuals = self._residuals
        with Recording(self.pathname + '._apply_nonlinear', self.iter_count, self):
            with self._unscaled_context(outputs=[outputs], residuals=[residuals]):
                residuals.set_vec(outputs)
                self.compute(self._inputs, outputs)

                # Restore any complex views if under complex step.
                if outputs._vector_info._under_complex_step:
                    outputs._remove_complex_views()
                    residuals._remove_complex_views()

                residuals -= outputs
                outputs += residuals

    def _solve_nonlinear(self):
        """
        Compute outputs. The model is assumed to be in a scaled state.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        super(ExplicitComponent, self)._solve_nonlinear()

        with Recording(self.pathname + '._solve_nonlinear', self.iter_count, self):
            with self._unscaled_context(
                    outputs=[self._outputs], residuals=[self._residuals]):
                self._residuals.set_const(0.0)
                failed = self.compute(self._inputs, self._outputs)
        return bool(failed), 0., 0.

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
        J = self.jacobian if jac is None else jac

        with Recording(self.pathname + '._apply_linear', self.iter_count, self):
            for vec_name in vec_names:
                if vec_name not in self._rel_vec_names:
                    continue

                with self._matvec_context(vec_name, scope_out, scope_in, mode) as vecs:
                    d_inputs, d_outputs, d_residuals = vecs

                    # Jacobian and vectors are all scaled, unitless
                    with self.jacobian_context(J):
                        J._apply(d_inputs, d_outputs, d_residuals, mode)

                    # if we're not matrix free, we can skip the bottom of
                    # this loop because compute_jacvec_product does nothing.
                    if not self.matrix_free:
                        continue

                    # Jacobian and vectors are all unscaled, dimensional
                    with self._unscaled_context(
                            outputs=[self._outputs], residuals=[d_residuals]):
                        d_residuals *= -1.0
                        if d_inputs._ncol > 1:
                            if self.supports_multivecs:
                                self.compute_multi_jacvec_product(self._inputs, d_inputs,
                                                                  d_residuals, mode)
                            else:
                                for i in range(d_inputs._ncol):
                                    # need to make the multivecs look like regular single vecs
                                    # since the component doesn't know about multivecs.
                                    d_inputs._icol = i
                                    d_residuals._icol = i
                                    self.compute_jacvec_product(self._inputs, d_inputs,
                                                                d_residuals, mode)
                                d_inputs._icol = None
                                d_residuals._icol = None
                        else:
                            self.compute_jacvec_product(self._inputs, d_inputs, d_residuals, mode)
                        d_residuals *= -1.0

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

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        with Recording(self.pathname + '._solve_linear', self.iter_count, self):
            for vec_name in vec_names:
                if vec_name in self._rel_vec_names:
                    if mode == 'fwd':
                        if self._has_resid_scaling:
                            d_outputs = self._vectors['output'][vec_name]
                            d_residuals = self._vectors['residual'][vec_name]

                            with self._unscaled_context(outputs=[d_outputs],
                                                        residuals=[d_residuals]):
                                d_outputs.set_vec(d_residuals)
                        else:
                            self._vectors['output'][vec_name].set_vec(
                                self._vectors['residual'][vec_name])
                    else:  # rev
                        if self._has_resid_scaling:
                            d_outputs = self._vectors['output'][vec_name]
                            d_residuals = self._vectors['residual'][vec_name]

                            with self._unscaled_context(outputs=[d_outputs],
                                                        residuals=[d_residuals]):
                                d_residuals.set_vec(d_outputs)
                        else:
                            self._vectors['residual'][vec_name].set_vec(
                                self._vectors['output'][vec_name])
        return False, 0., 0.

    def _linearize(self, jac=None, sub_do_ln=False):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            If None, use local jacobian, else use assembled jacobian jac.
        sub_do_ln : boolean
            Flag indicating if the children should call linearize on their linear solvers.
        """
        if not self._has_compute_partials and not self._approx_schemes:
            return

        if jac is None:
            jac = self.jacobian

        with self.jacobian_context(jac):
            with self._unscaled_context(
                    outputs=[self._outputs], residuals=[self._residuals]):
                # Since the residuals are already negated, this call should come before negate_jac
                # Additionally, computing the approximation before the call to compute_partials
                # allows users to override FD'd values.
                for approximation in itervalues(self._approx_schemes):
                    approximation.compute_approximations(self, jac=jac)

                if self._has_compute_partials:
                    # negate constant subjacs (and others that will get overwritten)
                    # back to normal
                    self._negate_jac(jac)
                    self.compute_partials(self._inputs, jac)

                    # re-negate the jacobian
                    self._negate_jac(jac)

            if self._assembled_jac is not None:
                self._assembled_jac._update()

    def compute(self, inputs, outputs):
        """
        Compute outputs given inputs. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]

        Returns
        -------
        bool or None
            None or False if run successfully; True if there was a failure.
        """
        pass

    def compute_partials(self, inputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        partials : Jacobian
            sub-jac components written to partials[output_name, input_name]
        """
        pass

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
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
        """
        pass
