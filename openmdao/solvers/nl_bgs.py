"""Define the NonlinearBlockGS class."""

from six.moves import range

from openmdao.solvers.solver import NonlinearSolver


class NonlinearBlockGS(NonlinearSolver):
    """
    Nonlinear block Gauss-Seidel solver.
    """

    SOLVER = 'NL: NLBGS'

    def _iter_execute(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system
        for isub, subsys in enumerate(system._subsystems_allprocs):
            system._transfers['fwd', isub](system._inputs,
                                           system._outputs, 'fwd')

            if subsys in system._subsystems_myproc:
                subsys._solve_nonlinear()
