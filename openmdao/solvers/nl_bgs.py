"""Define the NonlinearBlockGS class."""

from openmdao.solvers.solver import NonlinearSolver


class NonlinearBlockGS(NonlinearSolver):
    """
    Nonlinear block Gauss-Seidel solver.
    """

    SOLVER = 'NL: NLBGS'

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : <System>
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super(NonlinearBlockGS, self)._setup_solvers(system, depth)

        if len(system._subsystems_allprocs) != len(system._subsystems_myproc):
            raise RuntimeError('Nonlinear Gauss-Seidel cannot be used on a parallel group.')

    def _iter_execute(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system
        for isub, subsys in enumerate(system._subsystems_myproc):
            system._transfer('nonlinear', 'fwd', isub)
            subsys._solve_nonlinear()
