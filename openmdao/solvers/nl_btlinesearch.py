"""Backtracking line search using the Armijo-Goldstein condition."""

from math import isnan

import numpy as np

from openmdao.solvers.solver import NonlinearSolver


class BacktrackingLineSearch(NonlinearSolver):
    """Backtracking line search."""

    SOLVER = 'NL: BK_TKG'

    def _declare_options(self):
        """Declare options before kwargs are processed in the init method."""
        opt = self.options
        opt['maxiter'] = 5

        opt.declare(
            'bound_enforcement', value='vector', values=['vector', 'scalar', 'wall'],
            desc="If this is set to 'vector', the entire vector is backtracked together " +
                "when a bound is violated. If this is set to 'scalar', only the violating " +
                "entries are set to the bound and then the backtracking occurs on the vector " +
                "as a whole. If this is set to 'wall', only the violating entries are set " +
                "to the bound, and then the backtracking follows the wall - i.e., the " +
                "violating entries do not change during the line search.")
        opt.declare('rho', value=0.5, desc="Backtracking multiplier.")
        opt.declare('alpha', value=1.0, desc="Initial line search step.")
        # opt.declare('c', value=0.5, desc="Slope check trigger.")

    def _iter_initialize(self):
        """Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        system = self._system
        self.alpha = self.options['alpha']

        u = system._outputs
        du = system._vectors['output']['linear']

        norm0 = self._iter_get_norm()
        if norm0 == 0.0:
            norm0 = 1.0

        u.add_scal_vec(self.alpha, du)

        if self.options['bound_enforcement'] == 'vector':
            u._enforce_bounds_vector(du, self.alpha, system._lower_bounds, system._upper_bounds)
        elif self.options['bound_enforcement'] == 'scalar':
            u._enforce_bounds_scalar(du, self.alpha, system._lower_bounds, system._upper_bounds)
        elif self.options['bound_enforcement'] == 'wall':
            u._enforce_bounds_wall(du, self.alpha, system._lower_bounds, system._upper_bounds)

        norm = self._iter_get_norm()
        return norm0, norm

    def _iter_execute(self):
        """Perform the operations in the iteration loop."""
        system = self._system
        u = system._outputs
        du = system._vectors['output']['linear']

        u.add_scal_vec(-self.alpha, du)
        self.alpha *= self.options['rho']
        u.add_scal_vec(self.alpha, du)
