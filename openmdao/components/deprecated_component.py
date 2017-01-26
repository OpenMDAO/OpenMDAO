"""Define a deprecated Component class for backwards compatibility."""

from __future__ import division

import numpy

from openmdao.core.component import Component as BaseComponent
from openmdao.utils.general_utils import warn_deprecation


class Component(BaseComponent):
    """Component Class for backwards compatibility.

    Attributes
    ----------
    _state_names : [str, ...]
        list of names of the states (deprecated OpenMDAO 1.0 concept).
    _output_names : [str, ...]
        list of names of the outputs (deprecated OpenMDAO 1.0 concept).
    """

    def __init__(self, **kwargs):
        """Add a few more attributes."""
        super(Component, self).__init__(**kwargs)
        self._state_names = []
        self._output_names = []

        warn_deprecation('Components should inherit from ImplicitComponent '
                         'or ExplicitComponent. This class provides '
                         'backwards compabitibility with OpenMDAO <= 1.x as '
                         'this Component class is deprecated')

    def add_param(self, name, val=1.0, **kwargs):
        """Add an param variable to the component.

        Args
        ----
        name : str
            name of the variable in this component's namespace.
        val : object
            The value of the variable being added.
        **kwargs : dict
            additional args, documented [INSERT REF].
        """
        self.add_input(name, val, **kwargs)

    def add_state(self, name, val=1.0, **kwargs):
        """Add a state variable to the component.

        Args
        ----
        name : str
            name of the variable in this component's namespace.
        val : object
            The value of the variable being added.
        **kwargs : dict
            additional args, documented [INSERT REF].
        """
        if 'resid_scaler' in kwargs:
            kwargs['res_ref'] = kwargs['resid_scaler']

        super(Component, self).add_output(name, val, **kwargs)
        self._state_names.append(name)

    def add_output(self, name, val=1.0, **kwargs):
        """Add an output variable to the component.

        Args
        ----
        name : str
            name of the variable in this component's namespace.
        val : object
            The value of the variable being added.
        **kwargs : dict
            additional args, documented [INSERT REF].
        """
        if 'resid_scaler' in kwargs:
            kwargs['res_ref'] = kwargs['resid_scaler']

        super(Component, self).add_output(name, val, **kwargs)
        self._output_names.append(name)

    def _apply_nonlinear(self):
        """See System._apply_nonlinear."""
        self._inputs.scale(self._scaling_to_phys['input'])
        self._outputs.scale(self._scaling_to_phys['output'])
        self._residuals.scale(self._scaling_to_phys['residual'])

        self.apply_nonlinear(self._inputs, self._outputs, self._residuals)

        self._inputs.scale(self._scaling_to_norm['input'])
        self._outputs.scale(self._scaling_to_norm['output'])
        self._residuals.scale(self._scaling_to_norm['residual'])

    def _solve_nonlinear(self):
        """See System._solve_nonlinear."""
        if self._nl_solver is not None:
            self._nl_solver.solve()
        else:
            self._inputs.scale(self._scaling_to_phys['input'])
            self._outputs.scale(self._scaling_to_phys['output'])
            self._residuals.scale(self._scaling_to_phys['residual'])

            self.solve_nonlinear(self._inputs, self._outputs, self._residuals)

            self._inputs.scale(self._scaling_to_norm['input'])
            self._outputs.scale(self._scaling_to_norm['output'])
            self._residuals.scale(self._scaling_to_norm['residual'])

    def _apply_linear(self, vec_names, mode, var_inds=None):
        """See System._apply_linear."""
        for vec_name in vec_names:
            with self._matvec_context(vec_name, var_inds, mode) as vecs:
                d_inputs, d_outputs, d_residuals = vecs
                self._jacobian._system = self
                self._jacobian._apply(d_inputs, d_outputs, d_residuals,
                                      mode)

                self._inputs.scale(self._scaling_to_phys['input'])
                self._outputs.scale(self._scaling_to_phys['output'])
                d_inputs.scale(self._scaling_to_phys['input'])
                d_outputs.scale(self._scaling_to_phys['output'])
                d_residuals.scale(self._scaling_to_phys['residual'])

                for name in d_residuals:
                    if name in self._output_names:
                        d_residuals[name] *= -1.0

                self.apply_linear(self._inputs, self._outputs,
                                  d_inputs, d_outputs, d_residuals, mode)

                for name in d_residuals:
                    if name in self._output_names:
                        d_residuals[name] *= -1.0

                self._inputs.scale(self._scaling_to_norm['input'])
                self._outputs.scale(self._scaling_to_norm['output'])
                d_inputs.scale(self._scaling_to_norm['input'])
                d_outputs.scale(self._scaling_to_norm['output'])
                d_residuals.scale(self._scaling_to_norm['residual'])

    def _solve_linear(self, vec_names, mode):
        """See System._solve_linear."""
        if self._ln_solver is not None:
            return self._ln_solver(vec_names, mode)
        else:
            for vec_name in vec_names:
                d_outputs = self._vectors['output'][vec_name]
                d_residuals = self._vectors['residual'][vec_name]

                d_outputs.scale(self._scaling_to_phys['output'])
                d_residuals.scale(self._scaling_to_phys['residual'])

            success = self.solve_linear(self._vectors['output'],
                                        self._vectors['residual'],
                                        vec_names, mode)

            for vec_name in vec_names:
                for name in d_outputs:
                    if name in self._output_names:
                        if mode == 'fwd':
                            d_outputs[name] = d_residuals[name]
                        elif mode == 'rev':
                            d_residuals[name] = d_outputs[name]

                d_outputs.scale(self._scaling_to_norm['output'])
                d_residuals.scale(self._scaling_to_norm['residual'])

            return success

    def _linearize(self):
        """See System._linearize."""
        self._jacobian._system = self

        self._inputs.scale(self._scaling_to_phys['input'])
        self._outputs.scale(self._scaling_to_phys['output'])

        J = self.linearize(self._inputs, self._outputs, self._residuals)
        if J is not None:
            for k in J:
                self._jacobian[k] = J[k]

        self._inputs.scale(self._scaling_to_norm['input'])
        self._outputs.scale(self._scaling_to_norm['output'])

        for out_name in self._var_myproc_names['output']:
            if out_name in self._output_names:
                size = len(self._outputs._views_flat[out_name])
                ones = numpy.ones(size)
                arange = numpy.arange(size)
                self._jacobian[out_name, out_name] = (ones, arange, arange)

        for out_name in self._var_myproc_names['output']:
            if out_name in self._output_names:
                for in_name in self._var_myproc_names['input']:
                    if (out_name, in_name) in self._jacobian:
                        self._jacobian._negate((out_name, in_name))

        if self._jacobian._top_name == self.pathname:
            self._jacobian._update()

    def apply_nonlinear(self, params, unknowns, residuals):
        """Compute residuals given params and unknowns.

        Args
        ----
        params : Vector
            unscaled, dimensional param variables read via params[key]
        unknowns : Vector
            unscaled, dimensional unknown variables read via unknowns[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
        """
        pass

    def solve_nonlinear(self, params, unknowns, residuals):
        """Compute unknowns given params.

        Args
        ----
        params : Vector
            unscaled, dimensional param variables read via params[key]
        unknowns : Vector
            unscaled, dimensional unknown variables read via unknowns[key]
        residuals : Vector
            unscaled, dimensional residuals written to via residuals[key]
        """
        pass

    def apply_linear(self, params, unknowns,
                     d_params, d_unknowns, d_residuals, mode):
        r"""Compute jac-vector product.

        If mode is:
            'fwd': (d_params, unknowns) \|-> d_residuals

            'rev': d_residuals \|-> (d_params, unknowns)

        Args
        ----
        params : Vector
            unscaled, dimensional param variables read via params[key]
        unknowns : Vector
            unscaled, dimensional unknown variables read via unknowns[key]
        d_params : Vector
            see params; product must be computed only if var_name in d_params
        d_unknowns : Vector
            see unknowns; product must be computed only if var_name in unknowns
        d_residuals : Vector
            see unknowns
        mode : str
            either 'fwd' or 'rev'
        """
        pass

    def solve_linear(self, d_unknowns_dict, d_residuals_dict, vec_names, mode):
        r"""Apply inverse jac product.

        If mode is:
            'fwd': d_residuals \|-> d_unknowns

            'rev': d_unknowns \|-> d_residuals

        Args
        ----
        d_unknowns_dict : dict of <Vector>
            unscaled, dimensional quantities read via d_unknowns[key]
        d_residuals_dict : dict of <Vector>
            unscaled, dimensional quantities read via d_residuals[key]
        vec_names : [str, ...]
            list of right-hand-side vector names to perform solve linear on.
        mode : str
            either 'fwd' or 'rev'
        """
        pass

    def linearize(self, params, unknowns, jacobian):
        """Compute sub-jacobian parts / factorization.

        Args
        ----
        params : Vector
            unscaled, dimensional param variables read via params[key]
        unknowns : Vector
            unscaled, dimensional unknown variables read via unknowns[key]
        jacobian : Jacobian
            sub-jac components written to jacobian[output_name, input_name]
        """
        pass
