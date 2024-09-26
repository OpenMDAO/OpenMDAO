"""
Misc components.

Contains some general test components that are used in multiple places for testing, but aren't
featured as examples, and are not meant to be showcased as the proper way to write components
in OpenMDAO.
"""
from collections import defaultdict
import numpy as np

import openmdao.api as om
from openmdao.core.constants import _UNDEFINED


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


class ExplicitCounterComp(om.ExplicitComponent):
    """
    This component keeps counters for a number of core framework methods.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._counts = defaultdict(int)

    def _reset_counts(self, names=None):
        if names is None:
            names = self._counts.keys()

        for name in names:
            self._counts[name] = 0

    def _configure(self):
        self._counts['_configure'] += 1
        super()._configure()

    def _compute_wrapper(self):
        self._counts['_compute_wrapper'] += 1
        super()._compute_wrapper()

    def _apply_nonlinear(self):
        self._counts['_apply_nonlinear'] += 1
        super()._apply_nonlinear()

    def _compute_jacvec_product_wrapper(self, inputs, d_inputs, d_resids, mode,
                                        discrete_inputs=None):
        self._counts['_compute_jacvec_product_wrapper'] += 1
        super()._compute_jacvec_product_wrapper(inputs, d_inputs, d_resids, mode,
                                                discrete_inputs=discrete_inputs)

    def _apply_linear(self, jac, mode, scope_out=None, scope_in=None):
        self._counts['_apply_linear'] += 1
        super()._apply_linear(jac, mode, scope_out=scope_out, scope_in=scope_in)

    def _solve_linear(self, mode, scope_out=_UNDEFINED, scope_in=_UNDEFINED):
        self._counts['_solve_linear'] += 1
        super()._solve_linear(mode, scope_out=scope_out, scope_in=scope_in)

    def _compute_partials_wrapper(self):
        self._counts['_compute_partials_wrapper'] += 1
        super()._compute_partials_wrapper()

    def _linearize(self, jac=None, sub_do_ln=False):
        self._counts['_linearize'] += 1
        super()._linearize(jac=jac, sub_do_ln=sub_do_ln)


class MultComp(ExplicitCounterComp):
    def __init__(self, mult, **kwargs):
        super().__init__(**kwargs)
        self._mult = mult

    def setup(self):
        self.add_input('x')
        self.add_input('y')
        self.add_output('fxy')

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        outputs['fxy'] = (inputs['x'] - .5 * inputs['y']) * self._mult

    def compute_partials(self, inputs, partials):
        partials['fxy', 'x'] = self._mult
        partials['fxy', 'y'] = -self._mult * .5
