"""Define the NonlinearBlockJac class."""
from solver import NonlinearSolver


class NonlinearBlockJac(NonlinearSolver):
    """Nonlinear block Jacobi solver."""

    METHOD = 'NL: NLBJ'

    def _iter_execute(self):
        """See openmdao.solvers.solver.Solver."""
        system = self._system
        system._transfers[None](system._inputs, system._outputs, 'fwd')
        for subsys in system._subsystems_myproc:
            subsys._solve_nonlinear()
