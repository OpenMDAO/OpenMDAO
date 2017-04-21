"""Define the NonlinearBlockJac class."""
from openmdao.solvers.solver import NonlinearSolver


class NonlinearBlockJac(NonlinearSolver):
    """
    Nonlinear block Jacobi solver.
    """

    SOLVER = 'NL: NLBJ'

    def _iter_execute(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system
        system._transfer('nonlinear', 'fwd')
        for subsys in system._subsystems_myproc:
            subsys._solve_nonlinear()
        system._check_reconf_update()
