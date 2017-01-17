"""Define the NonlinearBlockGS class."""

from six.moves import range

from openmdao.solvers.solver import NonlinearSolver


class NonlinearBlockGS(NonlinearSolver):
    """Nonlinear block Gauss-Seidel solver."""

    SOLVER = 'NL: NLBGS'

    def _iter_execute(self):
        """Perform the operations in the iteration loop."""
        system = self._system
        for isub in range(len(system._subsystems_allprocs)):
            system._transfers['fwd', isub](system._inputs,
                                           system._outputs, 'fwd')

            if isub in system._subsystems_myproc_inds:
                index = system._subsystems_myproc_inds.index(isub)
                subsys = system._subsystems_myproc[index]
                subsys._solve_nonlinear()
