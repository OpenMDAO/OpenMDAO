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
        self._variable_myproc_metadata[typ].append(metadata)
        self._variable_allprocs_names[typ].append(name)

    def add_input(self, name, **kwargs):
        """See _add_variable."""
        self._add_variable(name, 'input', kwargs)

    def add_output(self, name, **kwargs):
        """See _add_variable."""
        self._add_variable(name, 'output', kwargs)



class ImplicitComponent(Component):
    """Class to inherit from when all output variables are implicit."""

    pass



class ExplicitComponent(Component):
    """Class to inherit from when all output variables are explicit."""

    def apply_nonlinear(self):
        outputs = self.outputs
        residuals = self.residuals

        residuals.set_vec(outputs)
        self.compute()
        residuals -= outputs
        outputs += residuals

    def solve_nonlinear(self):
        self.residuals.set_val(0.0)
        self.compute()

    def compute(self):
        pass

    def compute_derivative(self):
        pass



class IndepVarComponent(ExplicitComponent):
    """Class to inherit from when all output variables are independent."""

    def initialize_variables(self, comm):
        indep_vars = self.sys_args[0]
        for name, value in indep_vars:
            if type(value) == numpy.ndarray:
                self.add_output(name, value=value, shape=value.shape)
            else:
                self.add_output(name, value=value, shape=[1])
