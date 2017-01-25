"""Define the NewtonSolver class."""

from openmdao.solvers.solver import NonlinearSolver
from openmdao.utils.general_utils import warn_deprecation


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
        system._vectors['residual']['linear'].set_vec(system._residuals)
        system._vectors['residual']['linear'] *= -1.0
        system._linearize()
        self.options['subsolvers']['linear'].solve(['linear'], 'fwd')
        if 'linesearch' in self.options['subsolvers']:
            self.options['subsolvers']['linesearch'].solve()
        else:
            system._outputs += system._vectors['output']['linear']

    # ------ Deprecated methods --------

    @property
    def ln_solver(self):
        """Get the nonlinear solver for this system.

        This method is deprecated.

        Returns
        -------
        LinearSolver
            The requested linear solver instance.
        """
        warn_deprecation('This method provides backwards compabitibility with '
                         'OpenMDAO <= 1.x ; use '
                         ".options['subsolvers']['linear'].")

        return self.options['subsolvers']['linear']

    @ln_solver.setter
    def ln_solver(self, solver):
        """Set this system's nonlinear solver and perform setup.

        This method is deprecated.

        Args
        ----
        solver : `LinearSolver`
            Linear solver to be used for Newton.
        """
        warn_deprecation('This method provides backwards compabitibility with '
                         'OpenMDAO <= 1.x ; use set_subsolver instead.')

        self.set_subsolver('linear', solver)
