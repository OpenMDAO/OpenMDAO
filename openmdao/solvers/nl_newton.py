"""Define the NewtonSolver class."""

from openmdao.solvers.solver import NonlinearSolver


class NewtonSolver(NonlinearSolver):
    """Newton solver.

    The default linear solver is the ln_solver in the containing system.
    """

    SOLVER = 'NL: Newton'

    def _setup_solvers(self, system, depth):
        """Assign system instance, set depth, and optionally perform setup.

        Args
        ----
        system : <System>
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super(NewtonSolver, self)._setup_solvers(system, depth)

        if 'linear' not in self.options['subsolvers']:
            self.set_subsolver('linear', system.ln_solver)

    def _iter_execute(self):
        """Perform the operations in the iteration loop."""
        system = self._system
        system._vectors['residual'][''].set_vec(system._residuals)
        system._vectors['residual'][''] *= -1.0
        system._linearize()
        self.options['subsolvers']['linear']([''], 'fwd')
        if 'linesearch' in self.options['subsolvers']:
            self.options['subsolvers']['linesearch']()
        else:
            system._outputs += system._vectors['output']['']
