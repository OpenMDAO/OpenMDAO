"""
Some classes that are used to test Analysis Errors on multiple processes.
"""

from openmdao.api import ExplicitComponent, AnalysisError
from openmdao.core.driver import Driver


class AEComp(ExplicitComponent):

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_output('y', val=0.0)

    def compute(self, inputs, outputs):
        """
        This will error if x is more than 2.
        """
        x = inputs['x']

        if x > 2.0:
            raise AnalysisError('Try again.')

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
            failure_flag, _, _ = self._problem().model._solve_nonlinear()
        except AnalysisError:
            return True

        return False