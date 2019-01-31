"""
A few different backtracking line search subsolvers.

BoundsEnforceLS - Only checks bounds and enforces them by one of three methods.
ArmijoGoldsteinLS -- Like above, but terminates with the ArmijoGoldsteinLS condition.

"""
from __future__ import print_function

import sys
import numpy as np
from six import iteritems, reraise

from openmdao.core.analysis_error import AnalysisError
from openmdao.solvers.solver import NonlinearSolver
from openmdao.recorders.recording_iteration_stack import Recording


def _print_violations(unknowns, lower, upper):
    """
    Print out which variables exceed their bounds.

    Parameters
    ----------
    unknowns : Vector
        Vector containing the unknowns.
    lower : Vector
        Vector containing the lower bounds.
    upper : Vector
        Vector containing the upper bounds.
    """
    for name, val in iteritems(unknowns._views_flat):
        if any(val > upper._views_flat[name]):
            print("'%s' exceeds upper bounds" % name)
            print("  Val:", val)
            print("  Upper:", upper._views_flat[name], '\n')

        if any(val < lower._views_flat[name]):
            print("'%s' exceeds lower bounds" % name)
            print("  Val:", val)
            print("  Lower:", lower._views_flat[name], '\n')


class BoundsEnforceLS(NonlinearSolver):
    """
    Bounds enforcement only.

    Not so much a linesearch; just check the bounds and if they are violated, then pull back to a
    non-violating point and evaluate.

    Attributes
    ----------
    _do_subsolve : bool
        Flag used by parent solver to tell the line search whether to solve subsystems while
        backtracking.
    """

    SOLVER = 'LS: BCHK'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            Options dictionary.
        """
        super(BoundsEnforceLS, self).__init__(**kwargs)

        # Parent solver sets this to control whether to solve subsystems.
        self._do_subsolve = False

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super(BoundsEnforceLS, self)._declare_options()
        opt = self.options
        opt.declare(
            'bound_enforcement', default='scalar', values=['vector', 'scalar', 'wall'],
            desc="If this is set to 'vector', then the output vector is backtracked to the "
            "first point where violation occured. If it is set to 'scalar' or 'wall', then only "
            "the violated variables are backtracked to their point of violation.")
        opt.declare('print_bound_enforce', default=False,
                    desc="Set to True to print out names and values of variables that are pulled "
                    "back to their bounds.")

        # Remove unused options from base options here, so that users
        # attempting to set them will get KeyErrors.
        opt.undeclare("atol")
        opt.undeclare("rtol")
        opt.undeclare("maxiter")
        opt.undeclare("err_on_maxiter")

    def _solve(self):
        """
        Run the iterative solver.
        """
        self._iter_count = 0
        system = self._system

        u = system._outputs
        du = system._vectors['output']['linear']

        self._run_apply()

        norm0 = self._iter_get_norm()
        if norm0 == 0.0:
            norm0 = 1.0
        self._norm0 = norm0
        u += du

        if self.options['print_bound_enforce']:
            _print_violations(u, system._lower_bounds, system._upper_bounds)

        with Recording('BoundsEnforceLS', self._iter_count, self) as rec:
            if self.options['bound_enforcement'] == 'vector':
                u._enforce_bounds_vector(du, 1.0, system._lower_bounds, system._upper_bounds)
            elif self.options['bound_enforcement'] == 'scalar':
                u._enforce_bounds_scalar(du, 1.0, system._lower_bounds, system._upper_bounds)
            elif self.options['bound_enforcement'] == 'wall':
                u._enforce_bounds_wall(du, 1.0, system._lower_bounds, system._upper_bounds)

            self._run_apply()
            norm = self._iter_get_norm()
            # With solvers, we want to record the norm AFTER
            # the call, but the call needs to
            # be wrapped in the with for stack purposes,
            # so we locally assign  norm & norm0 into the class.
            rec.abs = norm
            rec.rel = norm / norm0

        self._mpi_print(self._iter_count, norm, norm / norm0)


class ArmijoGoldsteinLS(NonlinearSolver):
    """
    Backtracking line search that terminates using the Armijo-Goldstein condition..

    Attributes
    ----------
    _analysis_error_raised : bool
        Flag is set to True if a subsystem raises an AnalysisError.
    _do_subsolve : bool
        Flag used by parent solver to tell the line search whether to solve subsystems while
        backtracking.
    """

    SOLVER = 'LS: AG'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            Options dictionary.
        """
        super(ArmijoGoldsteinLS, self).__init__(**kwargs)

        # Parent solver sets this to control whether to solve subsystems.
        self._do_subsolve = False

        self._analysis_error_raised = False

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
        self.alpha = 1.

        self._run_apply()
        norm0 = self._iter_get_norm()
        if norm0 == 0.0:
            norm0 = 1.0

        u.add_scal_vec(self.alpha, du)

        if self.options['print_bound_enforce']:
            _print_violations(u, system._lower_bounds, system._upper_bounds)

        if self.options['bound_enforcement'] == 'vector':
            u._enforce_bounds_vector(du, self.alpha, system._lower_bounds, system._upper_bounds)
        elif self.options['bound_enforcement'] == 'scalar':
            u._enforce_bounds_scalar(du, self.alpha, system._lower_bounds, system._upper_bounds)
        elif self.options['bound_enforcement'] == 'wall':
            u._enforce_bounds_wall(du, self.alpha, system._lower_bounds, system._upper_bounds)

        try:
            cache = self._solver_info.save_cache()

            self._run_apply()
            norm = self._iter_get_norm()

        except AnalysisError as err:
            self._solver_info.restore_cache(cache)

            if self.options['retry_on_analysis_error']:
                self._analysis_error_raised = True
            else:
                exc = sys.exc_info()
                reraise(*exc)

            norm = np.nan

        return norm0, norm

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super(ArmijoGoldsteinLS, self)._declare_options()
        opt = self.options
        opt['maxiter'] = 5
        opt.declare('c', default=0.1, desc="Slope parameter for line of sufficient decrease. The "
                    "larger the step, the more decrease is required to terminate the line search. "
                    "This parameter is 'c' in: '||res_k|| < ||res_0|| (1 - c * alpha)' for the "
                    "termination criterion.")
        opt.declare(
            'bound_enforcement', default='vector', values=['vector', 'scalar', 'wall'],
            desc="If this is set to 'vector', the entire vector is backtracked together " +
                 "when a bound is violated. If this is set to 'scalar', only the violating " +
                 "entries are set to the bound and then the backtracking occurs on the vector " +
                 "as a whole. If this is set to 'wall', only the violating entries are set " +
                 "to the bound, and then the backtracking follows the wall - i.e., the " +
                 "violating entries do not change during the line search.")
        opt.declare('rho', default=0.5, lower=0.0, upper=1.0, desc="Backtracking multiplier.")
        opt.declare('alpha', default=1.0, desc="Initial line search step.")
        opt.declare('print_bound_enforce', default=False,
                    desc="Set to True to print out names and values of variables that are pulled "
                    "back to their bounds.")
        opt.declare('retry_on_analysis_error', default=True,
                    desc="Backtrack and retry if an AnalysisError is raised.")

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        self._analysis_error_raised = False
        system = self._system

        # Hybrid newton support.
        if self._do_subsolve and self._iter_count > 0:
            self._solver_info.append_solver()

            try:
                cache = self._solver_info.save_cache()
                self._gs_iter()
                self._run_apply()

            except AnalysisError as err:
                self._solver_info.restore_cache(cache)

                if self.options['retry_on_analysis_error']:
                    self._analysis_error_raised = True

                else:
                    exc = sys.exc_info()
                    reraise(*exc)

            finally:
                self._solver_info.pop()

        else:
            self._run_apply()

    def _solve(self):
        """
        Run the iterative solver.
        """
        maxiter = self.options['maxiter']
        atol = self.options['atol']
        rtol = self.options['rtol']
        c = self.options['c']

        system = self._system
        u = system._outputs
        du = system._vectors['output']['linear']

        self._iter_count = 0
        norm0, norm = self._iter_initialize()
        self._norm0 = norm0

        # Further backtracking if needed.
        while (self._iter_count < maxiter and
               ((norm > norm0 - c * self.alpha * norm0) or self._analysis_error_raised)):
            with Recording('ArmijoGoldsteinLS', self._iter_count, self) as rec:

                u.add_scal_vec(-self.alpha, du)
                if self._iter_count > 0:
                    self.alpha *= self.options['rho']
                u.add_scal_vec(self.alpha, du)

                cache = self._solver_info.save_cache()

                try:
                    self._single_iteration()
                    self._iter_count += 1

                    norm = self._iter_get_norm()

                    # With solvers, we want to report the norm AFTER
                    # the iter_execute call, but the i_e call needs to
                    # be wrapped in the with for stack purposes.
                    rec.abs = norm
                    rec.rel = norm / norm0

                except AnalysisError as err:
                    self._solver_info.restore_cache(cache)
                    self._iter_count += 1

                    if self.options['retry_on_analysis_error']:
                        self._analysis_error_raised = True
                        rec.abs = np.nan
                        rec.rel = np.nan

                    else:
                        exc = sys.exc_info()
                        reraise(*exc)

            # self._mpi_print(self._iter_count, norm, norm / norm0)
            self._mpi_print(self._iter_count, norm, self.alpha)
