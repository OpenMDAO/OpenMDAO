"""Backtracking line search using the Armijo-Goldstein condition."""

from math import isnan

import numpy as np

from openmdao.solvers.solver import NonlinearSolver


class BacktrackingLineSearch(NonlinearSolver):
    """Backtracking line search."""

    SOLVER = 'NL: BK_TKG'

    def __init__(self, **kwargs):
        """Backtracking line search."""
        super(BacktrackingLineSearch, self).__init__(**kwargs)

    def _declare_options(self):
        """Declare options before kwargs are processed in the init method."""
        opt = self.options
        opt['maxiter'] = 5

        opt.declare('bound_enforcement', value='vector', values=['vector', 'scalar'],
            desc="If this is set to 'vector', the entire vector is backtracked together" +
            "when a bound is violated. If this is set to 'scalar', only the violating" +
            "entries are backtracked to the bound.")
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
            u.enforce_bounds_all(du, system._lower_bounds, system._upper_bounds)
        else:
            u.enforce_bounds(du, system._lower_bounds, system._upper_bounds)

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
