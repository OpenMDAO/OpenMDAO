"""Define the ExplicitComponent class."""

from __future__ import division

import inspect

import numpy as np
from six import iteritems, itervalues

from openmdao.core.component import Component
from openmdao.utils.class_util import overrides_method
from openmdao.utils.general_utils import warn_deprecation


class ExplicitComponent(Component):
    """
    Class to inherit from when all output variables are explicit.
    """

    def __init__(self, **kwargs):
        """
        Check if we are matrix-free.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            available here and in all descendants of this system.
        """
        super(ExplicitComponent, self).__init__(**kwargs)

        if overrides_method('compute_jacvec_product', self, ExplicitComponent):
            self._matrix_free = True

    def add_output(self, name, val=1.0, shape=None, units=None, res_units=None, desc='',
                   lower=None, upper=None, ref=1.0, ref0=0.0,
                   res_ref=None, res_ref0=None, var_set=0):
        """
        Add an output variable to the component.

        For ExplicitComponent, res_ref and res_ref0 default to the values in res and res0 unless
        otherwise specified.

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
        res_ref0 : float
            Scaling parameter. The value in the user-defined res_units of this output's residual
            when the scaled value is 0. Default is None, which means residual scaling matches
            output scaling.
        var_set : hashable object
            For advanced users only. ID or color for this variable, relevant for reconfigurability.
            Default is 0.
        """
        if res_ref is None:
            res_ref = ref
        if res_ref0 is None:
            res_ref0 = ref0

        if inspect.stack()[1][3] == '__init__':
            warn_deprecation("In the future, the 'add_output' method must be "
                             "called from 'initialize_variables' rather than "
                             "in the '__init__' function.")

        super(ExplicitComponent, self).add_output(name, val=val, shape=shape, units=units,
                                                  res_units=res_units, desc=desc, lower=lower,
                                                  upper=upper, ref=ref, ref0=ref0, res_ref=res_ref,
                                                  res_ref0=res_ref0, var_set=var_set)

    def _apply_nonlinear(self):
        """
        Compute residuals. The model is assumed to be in a scaled state.
        """
        with self._units_scaling_context(inputs=[self._inputs], outputs=[self._outputs],
                                         residuals=[self._residuals]):
            self._residuals.set_vec(self._outputs)
            self.compute(self._inputs, self._outputs)
            self._residuals -= self._outputs
            self._outputs += self._residuals

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
        with self._units_scaling_context(inputs=[self._inputs], outputs=[self._outputs],
                                         residuals=[self._residuals]):
            self._residuals.set_const(0.0)
            failed = self.compute(self._inputs, self._outputs)

        return bool(failed), 0., 0.

    def _apply_linear(self, vec_names, mode, var_inds=None):
        """
        Compute jac-vec product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        var_inds : [int, int, int, int] or None
            ranges of variable IDs involved in this matrix-vector product.
            The ordering is [lb1, ub1, lb2, ub2].
        """
        for vec_name in vec_names:
            with self._matvec_context(vec_name, var_inds, mode) as vecs:
                d_inputs, d_outputs, d_residuals = vecs

                # Jacobian and vectors are all scaled, unitless
                with self.jacobian_context() as J:
                    J._apply(d_inputs, d_outputs, d_residuals, mode)

                # Jacobian and vectors are all unscaled, dimensional
                with self._units_scaling_context(inputs=[self._inputs, d_inputs],
                                                 outputs=[self._outputs],
                                                 residuals=[d_residuals]):
                    d_residuals *= -1.0
                    self.compute_jacvec_product(self._inputs, self._outputs,
                                                d_inputs, d_residuals, mode)
                    d_residuals *= -1.0

    def _solve_linear(self, vec_names, mode):
        """
        Apply inverse jac product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        for vec_name in vec_names:
            d_outputs = self._vectors['output'][vec_name]
            d_residuals = self._vectors['residual'][vec_name]

            with self._units_scaling_context(outputs=[d_outputs], residuals=[d_residuals]):
                if mode == 'fwd':
                    d_outputs.set_vec(d_residuals)
                elif mode == 'rev':
                    d_residuals.set_vec(d_outputs)

        return False, 0., 0.

    def _linearize(self, do_nl=False, do_ln=False):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        do_nl : boolean
            flag indicating if the nonlinear solver should be linearized.
        do_ln : boolean
            flag indicating if the linear solver should be linearized.
        """
        with self.jacobian_context() as J:
            with self._units_scaling_context(inputs=[self._inputs], outputs=[self._outputs],
                                             residuals=[self._residuals], scale_jac=True):
                # Since the residuals are already negated, this call should come before negate_jac
                # Additionally, computing the approximation before the call to compute_partials
                # allows users to override FD'd values.
                for approximation in itervalues(self._approx_schemes):
                    approximation.compute_approximations(self, jac=J)

                # negate constant subjacs (and others that will get overwritten)
                # back to normal
                self._negate_jac()
                self.compute_partial_derivs(self._inputs, self._outputs, J)

                # re-negate the jacobian
                self._negate_jac()

            if self._owns_assembled_jac:
                J._update()

    def _setup_partials(self):
        """
        Set up partial derivative sparsity structures and approximation schemes.
        """
        self.initialize_partials()

        abs2data = self._var_abs2data_io

        # Note: These declare calls are outside of initialize_partials so that users do not have to
        # call the super version of initialize_partials. This is still post-initialize_variables.
        other_names = []
        for out_abs in self._var_abs_names['output']:
            meta = abs2data[out_abs]['metadata']
            out_name = abs2data[out_abs]['prom']
            size = np.prod(meta['shape'])
            arange = np.arange(size)

            # No need to FD outputs wrt other outputs
            abs_key = (out_abs, out_abs)
            if abs_key in self._subjacs_info:
                if 'method' in self._subjacs_info[abs_key]:
                    del self._subjacs_info[abs_key]['method']
            self.declare_partials(out_name, out_name, rows=arange, cols=arange, val=1.)
            for other_name in other_names:
                self.declare_partials(out_name, other_name, dependent=False)
                self.declare_partials(other_name, out_name, dependent=False)
            other_names.append(out_name)

    def _negate_jac(self):
        """
        Negate this component's part of the jacobian.
        """
        if self._jacobian._subjacs:
            for res_name in self._var_abs_names['output']:
                for in_name in self._var_abs_names['input']:
                    abs_key = (res_name, in_name)
                    if abs_key in self._jacobian._subjacs:
                        self._jacobian._multiply_subjac(abs_key, -1.)

    def _set_partials_meta(self):
        """
        Set subjacobian info into our jacobian.
        """
        with self.jacobian_context() as J:
            for abs_key, meta in iteritems(self._subjacs_info):
                J._set_partials_meta(abs_key, meta, meta['type'] == 'input')

                method = meta.get('method', False)
                if method and meta['dependent']:
                    self._approx_schemes[method].add_approximation(abs_key, meta)

        for approx in itervalues(self._approx_schemes):
            approx._init_approximations()

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

    def compute_partial_derivs(self, inputs, outputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        partials : Jacobian
            sub-jac components written to partials[output_name, input_name]
        """
        pass

    def compute_jacvec_product(self, inputs, outputs,
                               d_inputs, d_outputs, mode):
        r"""
        Compute jac-vector product. The model is assumed to be in an unscaled state.

        If mode is:
            'fwd': d_inputs \|-> d_outputs

            'rev': d_outputs \|-> d_inputs

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        d_inputs : Vector
            see inputs; product must be computed only if var_name in d_inputs
        d_outputs : Vector
            see outputs; product must be computed only if var_name in d_outputs
        mode : str
            either 'fwd' or 'rev'
        """
        pass
