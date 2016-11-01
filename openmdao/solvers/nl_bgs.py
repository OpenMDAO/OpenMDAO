"""Define the NonlinearBlockGS class."""
from six.moves import range

from solver import NonlinearSolver


class NonlinearBlockGS(NonlinearSolver):
    """Nonlinear block Gauss-Seidel solver."""

    METHOD = 'NL: NLBGS'

    def _iter_execute(self):
        """See openmdao.solvers.solver.Solver."""
        system = self._system
        for isub in range(len(system._subsystems_allprocs)):
            system._transfers['fwd', isub](system._inputs,
                                           system._outputs, 'fwd')

            if isub in system._subsystems_inds:
                index = system._subsystems_inds.index(isub)
                subsys = system._subsystems_myproc[index]
                subsys._solve_nonlinear()
