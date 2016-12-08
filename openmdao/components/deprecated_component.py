"""Define the ExplicitComponent class."""

from __future__ import division

import numpy

import warnings

from openmdao.core.component import Component as BaseComponent


class Component(BaseComponent):
    """Class to inherit from when all output variables are explicit.

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

        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn("'low' and 'high' are deprecated. "
                      "Use 'lower' and 'upper' instead.",
                      DeprecationWarning, stacklevel=2)
        warnings.simplefilter('ignore', DeprecationWarning)

    def add_param(self, name, val=1.0, **kwargs):
        """Add an input variable to the component.

        Args
        ----
        name : str
            name of the variable in this component's namespace.
        val : object
            The value of the variable being added.
        **kwargs : dict
            additional args, documented [INSERT REF].
        """
        self._add_variable(name, 'input', val, kwargs)

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

        self._add_variable(name, 'output', val, kwargs)
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

        self._add_variable(name, 'output', val, kwargs)
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
            self._nl_solver(self._inputs, self._outputs)
        else:
            self._inputs.scale(self._scaling_to_phys['input'])
            self._outputs.scale(self._scaling_to_phys['output'])
            self._residuals.scale(self._scaling_to_phys['residual'])

            self.solve_nonlinear(self._inputs, self._outputs)

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
            success = True
            for vec_name in vec_names:
                d_outputs = self._vectors['output'][vec_name]
                d_residuals = self._vectors['residual'][vec_name]

                d_outputs.scale(self._scaling_to_phys['output'])
                d_residuals.scale(self._scaling_to_phys['residual'])

                tmp = self.solve_linear(d_outputs, d_residuals, mode)

                for name in d_outputs:
                    if name in self._output_names:
                        if mode == 'fwd':
                            d_outputs[name] = d_residuals[name]
                        elif mode == 'rev':
                            d_residuals[name] = d_outputs[name]

                d_outputs.scale(self._scaling_to_norm['output'])
                d_residuals.scale(self._scaling_to_norm['residual'])

                success = success and tmp
            return success

    def _linearize(self, initial=False):
        """See System._linearize."""
        self._jacobian._system = self

        self._inputs.scale(self._scaling_to_phys['input'])
        self._outputs.scale(self._scaling_to_phys['output'])

        self.linearize(self._inputs, self._outputs, self._jacobian)

        self._inputs.scale(self._scaling_to_norm['input'])
        self._outputs.scale(self._scaling_to_norm['output'])

        for op_name in self._variable_myproc_names['output']:
            if op_name in self._output_names:
                size = len(self._outputs._views_flat[op_name])
                ones = numpy.ones(size)
                arange = numpy.arange(size)
                self._jacobian[op_name, op_name] = (ones, arange, arange)

        for op_name in self._variable_myproc_names['output']:
            if op_name in self._output_names:
                for ip_name in self._variable_myproc_names['input']:
                    if (op_name, ip_name) in self._jacobian:
                        self._jacobian._negate((op_name, ip_name))

        self._jacobian._precompute_iter()
        if not initial and self._jacobian._top_name == self.path_name:
            self._jacobian._update()
