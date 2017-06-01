"""Define the base Solver, NonlinearSolver, and LinearSolver classes."""

from __future__ import division, print_function

from openmdao.solvers.solver import Solver

class NonlinearSolver(Solver):
    """
    Base class for nonlinear solvers.
    """

    def solve(self):
        """
        Run the solver.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        return self._run_iterator()

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        if self.options['maxiter'] > 0:
            norm = self._iter_get_norm()
        else:
            norm = 1.0
        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def _iter_get_norm(self):
        """
        Return the norm of the residual.

        Returns
        -------
        float
            norm.
        """
        self._system._apply_nonlinear()
        return self._system._residuals.get_norm()
