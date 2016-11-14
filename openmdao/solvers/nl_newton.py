"""Define the NewtonSolver class."""
from openmdao.solvers.solver import NonlinearSolver


class NewtonSolver(NonlinearSolver):
    """Newton solver."""

    SOLVER = 'NL: Newton'

    def _iter_execute(self):
        """See openmdao.solvers.solver.Solver."""
        system = self._system
        system._vectors['residual'][''].set_vec(system._residuals)
        system._vectors['residual'][''] *= -1.0
        system._linearize()
        self.options['subsolvers']['linear']([''], 'fwd')
        if 'linesearch' in self.options['subsolvers']:
            self.options['subsolvers']['linesearch']()
        else:
            system._outputs += system._vectors['output']['']
