"""Define the core Component classes.

Classes
-------
Component - base Component class
ImplicitComponent - used to define output variables that are all implicit
ExplicitComponent - used to define output variables that are all explicit
IndepVarComponent - used to define output variables that are all independent
"""
from __future__ import division
import numpy

from Blue.core.system import System



class Component(System):
    """Base Component class; not to be directly instantiated."""

    DEFAULTS = {
        'indices': [0],
        'shape': [1],
        'units': '',
        'value': 1.0,
        'scale': 1.0,
        'lower': None,
        'upper': None,
        'var_set': 0,
    }

    def _add_variable(self, name, typ, kwargs):
        """Add an input/output variable to the component.

        Args
        ----
        name : str
            name of the variable in this component's namespace.
        typ : str
            either 'input' or 'output'
        **kwargs : dict
            variable metadata with DEFAULTS defined above.
        """
        metadata = self.DEFAULTS.copy()
        metadata.update(kwargs)
        if typ == 'input':
            metadata['indices'] = numpy.array(metadata['indices'])
        self._variable_allprocs_names[typ].append(name)
        self._variable_myproc_names[typ].append(name)
        self._variable_myproc_metadata[typ].append(metadata)

    def add_input(self, name, **kwargs):
        """See _add_variable."""
        self._add_variable(name, 'input', kwargs)

    def add_output(self, name, **kwargs):
        """See _add_variable."""
        self._add_variable(name, 'output', kwargs)



class ImplicitComponent(Component):
    """Class to inherit from when all output variables are implicit."""


    def _apply_nonlinear(self):
        self.apply_nonlinear()

    def _solve_nonlinear(self):
        if self._solvers_nonlinear is not None:
            self._solvers_nonlinear()
        else:
            self.solve_nonlinear()

    def _apply_linear(self, vec_names, mode, var_ind_range):
        if self._jacobian.GLOBAL:
            for vec_name in vec_names:
                op_names, ip_names = self._utils_compute_deriv_names(var_ind_range)

                d_inputs = self._vectors['input'][vec_name]
                d_outputs = self._vectors['output'][vec_name]
                d_residuals = self._vectors['residual'][vec_name]

                self._jacobian._apply(d_inputs, d_outputs, d_residuals,
                                      op_names, ip_names, mode, var_ind_range)
        else:
            for vec_name in vec_names:
                op_names, ip_names = self._utils_compute_deriv_names(var_ind_range)

                d_inputs = self._vectors['input'][vec_name]
                d_outputs = self._vectors['output'][vec_name]
                d_residuals = self._vectors['residual'][vec_name]

                if mode == 'fwd':
                    d_residuals.set_const(0.0)
                elif mode == 'rev':
                    d_inputs.set_const(0.0)
                    d_outputs.set_const(0.0)
                self.apply_linear(mode, ip_names, op_names, inputs, outputs,
                                  d_inputs, d_outputs, d_residuals)

    def _solve_linear(self, vec_names, mode):
        if self._solvers_linear is not None:
            return self._solvers_linear(vec_names, mode)
        else:
            for vec_name in vec_names:
                d_outputs = self._vectors['output'][vec_name]
                d_residuals = self._vectors['residual'][vec_name]
                success = self.solve_linear(mode, d_output, d_residuals)
                if not success: return False
            return True

    def _linearize(self):
        self.linearize()

        if self._jacobian.GLOBAL:
            self._jacobian._update()

    def apply_nonlinear(self):
        pass

    def solve_nonlinear(self):
        pass

    def apply_linear(self, mode, ip_names, op_names, inputs, outputs,
                     d_inputs, d_outputs, d_residuals):
        pass

    def solve_linear(self, mode, d_output, d_residuals):
        pass

    def linearize(self):
        pass



class ExplicitComponent(Component):
    """Class to inherit from when all output variables are explicit."""

    def _apply_nonlinear(self):
        inputs = self._inputs
        outputs = self._outputs
        residuals = self._residuals

        residuals.set_vec(outputs)
        self.compute(inputs, outputs)
        residuals -= outputs
        outputs += residuals

    def _solve_nonlinear(self):
        inputs = self._inputs
        outputs = self._outputs
        residuals = self._residuals

        residuals.set_val(0.0)
        self.compute(inputs, outputs)

    def _apply_linear(self, vec_names, mode, var_ind_range):
        if self._jacobian.GLOBAL:
            for vec_name in vec_names:
                op_names, ip_names = self._utils_compute_deriv_names(var_ind_range)

                d_inputs = self._vectors['input'][vec_name]
                d_outputs = self._vectors['output'][vec_name]
                d_residuals = self._vectors['residual'][vec_name]

                self._jacobian._apply(d_inputs, d_outputs, d_residuals,
                                      op_names, ip_names, mode, var_ind_range)
        else:
            if mode == 'fwd':
                for vec_name in vec_names:
                    op_names, ip_names = self._utils_compute_deriv_names(var_ind_range)

                    d_inputs = self._vectors['input'][vec_name]
                    d_outputs = self._vectors['output'][vec_name]
                    d_residuals = self._vectors['residual'][vec_name]

                    d_residuals.set_const(0.)
                    self.compute_jacvec_product(mode, ip_names, op_names,
                                                inputs, d_inputs, d_residuals)
                    d_residuals *= -1.0
                    d_residuals += d_outputs
            elif mode == 'rev':
                for vec_name in vec_names:
                    op_names, ip_names = self._utils_compute_deriv_names(var_ind_range)

                    d_inputs = self._vectors['input'][vec_name]
                    d_outputs = self._vectors['output'][vec_name]
                    d_residuals = self._vectors['residual'][vec_name]

                    d_inputs.set_const(0.)
                    d_outputs.set_const(0.)
                    d_residuals *= -1.0
                    self.compute_jacvec_product(mode, ip_names, op_names,
                                                inputs, d_inputs, d_residuals)
                    d_residuals *= -1.0
                    d_outputs.set_vec(d_residuals)

    def _linearize(self):
        self.compute_jacobian(self._inputs, self._jacobian)

        for op_name in self._variable_myproc_names['output']:
            for ip_name in self._variable_myproc_names['input']:
                self._jacobian._explicit[op_name, ip_name] = True

    def compute(self, inputs, outputs):
        pass

    def compute_jacobian(self, inputs, jacobian):
        pass

    def compute_jacvec_product(self, mode, ip_names, op_names,
                               inputs, d_inputs, d_residuals):
        pass




class IndepVarComponent(ExplicitComponent):
    """Class to inherit from when all output variables are independent."""

    def initialize_variables(self, comm):
        indep_vars = self.args[0]
        for name, value in indep_vars:
            if type(value) == numpy.ndarray:
                self.add_output(name, value=value, shape=value.shape)
            else:
                self.add_output(name, value=value, shape=[1])
