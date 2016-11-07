"""Define the NewtonSolver class."""
from openmdao.solvers.solver import NonlinearSolver


class NewtonSolver(NonlinearSolver):
    """Newton solver."""

    METHOD = 'NL: Newton'

    def _iter_execute(self):
        """See openmdao.solvers.solver.Solver."""
        system = self._system
        system._vectors['residual'][''].set_vec(system._residuals)
        system._vectors['residual'][''] *= -1.0
        system.linearize()
        self.subsolvers['linear']([''], 'fwd')
        if 'linesearch' in self.subsolvers:
            self.subsolvers['linesearch']()
        else:
            system._outputs += system._vectors['output']['']
