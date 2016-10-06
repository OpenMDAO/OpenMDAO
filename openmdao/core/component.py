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

from openmdao.core.system import System



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
        self.apply_nonlinear(self._inputs, self._outputs, self._residuals)

    def _solve_nonlinear(self):
        if self._solvers_nonlinear is not None:
            self._solvers_nonlinear(self._inputs, self._outputs)
        else:
            self.solve_nonlinear(self._inputs, self._outputs)

    def _apply_linear(self, vec_names, mode, var_ind_range):
        for vec_name in vec_names:
            tmp = self._utils_get_vectors(vec_name, var_ind_range, mode)
            d_inputs, d_outputs, d_residuals = tmp
            self.apply_linear(self._inputs, self._outputs,
                              d_inputs, d_outputs, d_residuals, mode)

    def _solve_linear(self, vec_names, mode):
        if self._solvers_linear is not None:
            return self._solvers_linear(vec_names, mode)
        else:
            for vec_name in vec_names:
                d_outputs = self._vectors['output'][vec_name]
                d_residuals = self._vectors['residual'][vec_name]
                success = self.solve_linear(d_output, d_residuals, mode)
                if not success:
                    return False
            return True

    def _linearize(self):
        self._jacobian._system = self
        self.linearize(self._inputs, self._outputs, self._jacobian)

        if self._jacobian._top_name == self._path_name:
            self._jacobian._update()

    def apply_nonlinear(self, inputs, outputs, residuals):
        pass

    def solve_nonlinear(self, inputs, outputs):
        pass

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals,
                     mode):
        self._jacobian._apply(d_inputs, d_outputs, d_residuals, mode)

    def solve_linear(self, d_output, d_residuals, mode):
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
        if self._jacobian._top_name == self._path_name:
            for vec_name in vec_names:
                tmp = self._utils_get_vectors(vec_name, var_ind_range, mode)
                d_inputs, d_outputs, d_residuals = tmp
                self._jacobian._apply(d_inputs, d_outputs, d_residuals,
                                      op_names, ip_names, mode, var_ind_range)
        else:
            if mode == 'fwd':
                for vec_name in vec_names:
                    tmp = self._utils_get_vectors(vec_name, var_ind_range, mode)
                    d_inputs, d_outputs, d_residuals = tmp

                    self.compute_jacvec_product(inputs, outputs,
                                                d_inputs, d_residuals, mode)
                    d_residuals *= -1.0
                    d_residuals += d_outputs
            elif mode == 'rev':
                for vec_name in vec_names:
                    tmp = self._utils_get_vectors(vec_name, var_ind_range, mode)
                    d_inputs, d_outputs, d_residuals = tmp

                    d_residuals *= -1.0
                    self.compute_jacvec_product(inputs, outputs,
                                                d_inputs, d_residuals, mode)
                    d_residuals *= -1.0
                    d_outputs.set_vec(d_residuals)

    def _linearize(self):
        self.compute_jacobian(self._inputs, self._outputs, self._jacobian)

        for op_name in self._variable_myproc_names['output']:
            size = len(self._outputs[op_name])
            ones = numpy.ones(size)
            arange = numpy.arange(size)
            self._jacobian[op_name, op_name] = (ones, arange, arange)

        for op_name in self._variable_myproc_names['output']:
            for ip_name in self._variable_myproc_names['input']:
                if (op_name, ip_name) in self._jacobian:
                    self._jacobian._negate(op_name, ip_name)

    def compute(self, inputs, outputs):
        pass

    def compute_jacobian(self, inputs, outputs, jacobian):
        pass

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_residuals,
                               mode):
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
