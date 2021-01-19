"""
Misc components.

Contains some general test components that are used in multiple places for testing, but aren't
featured as examples, and are not meant to be showcased as the proper way to write components
in OpenMDAO.
"""
import numpy as np

import openmdao.api as om


class Comp4LinearCacheTest(om.ImplicitComponent):
    """
    Component needed for testing cached linear solutions.

    Generally, needed an implicit component that was challenging enough that it took a few
    iterations to solve with the petsc and scipy iterative linear solvers. Equation just
    came from playing around. It does not represent any academic or real world problem, so
    it does not need to be explained.
    """
    def setup(self):
        """
        Set up the model and define derivatives.
        """
        self.add_input('x', val=1.0)
        self.add_output('y', val=np.sqrt(3))

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Compute residuals.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        residuals : Vector
            Unscaled, dimensional residuals written to via residuals[key].
        """
        x = inputs['x']
        y = outputs['y']
        residuals['y'] = x * y ** 3 - 3.0 * y * x ** 3

    def linearize(self, inputs, outputs, partials):
        """
        Compute derivatives.

        These derivatives are correct.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        partials : `Jacobian`
            Contains sub-jacobians.
        """
        x = inputs['x']
        y = outputs['y']
        partials['y', 'x'] = y ** 3 - 9.0 * y * x ** 2
        partials['y', 'y'] = 3.0 * x * y ** 2 - 3.0 * y * x ** 3
