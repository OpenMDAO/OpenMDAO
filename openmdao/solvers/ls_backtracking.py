"""
A few different backtracking line search subsolvers.

BoundsCheck - Only checks bounds and enforces them by one of three methods.
BacktrackingLineSearch -- Checks bounds and also backtracks but terminates using atol and rtol like
                          other nonlinear solvers.
ArmijoGoldstein -- Like above, but terminates with the ArmijoGoldstein condition.

"""

from math import isnan

import numpy as np

from openmdao.solvers.solver import NonlinearSolver


class BoundsCheck(NonlinearSolver):
    """
    Bounds check only.

    Not so much a linesearch; just check the bounds and if they are violated, then pull back to a
    non-violating point and evaluate.
    """

    SOLVER = 'LS: BCHK'

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        opt = self.options
        opt.declare(
            'bound_enforcement', value='vector', values=['vector', 'scalar', 'wall'],
            desc="If this is set to 'vector', then the the output vector is backtracked to the "
            "first point where violation occured. If it is set to 'scalar' or 'wall', then only "
            "the violated variables are backtracked to their point of violation.")

    def _run_iterator(self):
        """
        Run the iterative solver.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        self._iter_count = 0
        system = self._system

        u = system._outputs
        du = system._vectors['output']['linear']

        norm0 = self._iter_get_norm()
        if norm0 == 0.0:
            norm0 = 1.0

        u += du

        if self.options['bound_enforcement'] == 'vector':
            u._enforce_bounds_vector(du, 1.0, system._lower_bounds, system._upper_bounds)
        elif self.options['bound_enforcement'] == 'scalar':
            u._enforce_bounds_scalar(du, 1.0, system._lower_bounds, system._upper_bounds)
        elif self.options['bound_enforcement'] == 'wall':
            u._enforce_bounds_wall(du, 1.0, system._lower_bounds, system._upper_bounds)

        norm = self._iter_get_norm()
        self._mpi_print(self._iter_count, norm, norm / norm0)

        fail = (np.isinf(norm) or np.isnan(norm))

        return fail, norm, norm / norm0


class BacktrackingLineSearch(NonlinearSolver):
    """
    Backtracking line search.
    """

    SOLVER = 'LS: BKTKG'

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        opt = self.options
        opt['maxiter'] = 5
        opt['rtol'] = 0.95
        opt.declare(
            'bound_enforcement', value='vector', values=['vector', 'scalar', 'wall'],
            desc="If this is set to 'vector', the entire vector is backtracked together " +
                 "when a bound is violated. If this is set to 'scalar', only the violating " +
                 "entries are set to the bound and then the backtracking occurs on the vector " +
                 "as a whole. If this is set to 'wall', only the violating entries are set " +
                 "to the bound, and then the backtracking follows the wall - i.e., the " +
                 "violating entries do not change during the line search.")
        opt.declare('rho', value=0.5, lower=0.0, upper=1.0, desc="Backtracking multiplier.")
        opt.declare('alpha', value=1.0, desc="Initial line search step.")

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
        """
        Perform the operations in the iteration loop.
        """
        system = self._system
        u = system._outputs
        du = system._vectors['output']['linear']

        u.add_scal_vec(-self.alpha, du)
        self.alpha *= self.options['rho']
        u.add_scal_vec(self.alpha, du)


class ArmijoGoldstein(BacktrackingLineSearch):
    """
    Backtracking line search that terminates using the Armijo-Goldstein condition..
    """

    SOLVER = 'LS: AG'

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super(ArmijoGoldstein, self)._declare_options()
        opt = self.options
        opt.declare('c', value=0.1, desc="Slope parameter for line of sufficient decrease. The "
                    "larger the step, the more decrease is required to terminate the line search. "
                    "This parameter is 'c' in: '||res_k|| < ||res_0|| (1 - c * alpha)' for the "
                    "termination criterion.")

    def _run_iterator(self):
        """
        Run the iterative solver.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        maxiter = self.options['maxiter']
        atol = self.options['atol']
        rtol = self.options['rtol']
        c = self.options['c']

        self._iter_count = 0
        norm0, norm = self._iter_initialize()
        self._mpi_print(self._iter_count, norm, norm / norm0)

        # Further backtracking if needed.
        # The Armijo-Goldstein is basically a slope comparison --actual vs predicted.
        # We don't have an actual gradient, but we have the Newton vector that should
        # take us to zero, and our "runs" are the same, and we can just compare the
        # "rise".
        while self._iter_count < maxiter and (norm0 - norm) < c * self.alpha * norm0:
            self._iter_execute()
            self._iter_count += 1
            norm = self._iter_get_norm()
            self._mpi_print(self._iter_count, norm, norm / norm0)

        fail = (np.isinf(norm) or np.isnan(norm) or
                (norm > atol and norm / norm0 > rtol))

        return fail, norm, norm / norm0
