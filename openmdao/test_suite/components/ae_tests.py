"""
Some classes that are used to test Analysis Errors on multiple processes.
"""

import openmdao.api as om
from openmdao.core.driver import Driver


class AEComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_output('y', val=0.0)

    def compute(self, inputs, outputs):
        """
        This will error if x is more than 2.
        """
        x = inputs['x']

        if x > 2.0:
            raise om.AnalysisError('Try again.')

        outputs['y'] = x*x + 2.0


class AEDriver(Driver):
    """
    Handle an Analysis Error from below.
    """

    def run(self):
        """
        Just handle it and return an error state.
        """
        try:
            self._problem().model.run_solve_nonlinear()
        except om.AnalysisError:
            return True

        return False