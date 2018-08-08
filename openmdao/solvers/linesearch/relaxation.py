"""
RelaxationLS is a linesearch that implements a relaxation scheme that gradually ramps the
relaxation factor from a user selected value up to 1.0 across two selectable cut-off values
of the residual norm.

RelaxationLS Also checks bounds and enforces them by one of three methods.

"""
from __future__ import print_function

from openmdao.solvers.linesearch.backtracking import _print_violations
from openmdao.solvers.solver import LineSearch
from openmdao.recorders.recording_iteration_stack import Recording


class RelaxationLS(LineSearch):
    """
    Relaxation linesearch.

    RelaxationLS is a linesearch that implements a relaxation scheme that gradually ramps the
    relaxation factor from a user selected value up to 1.0 across two selectable cut-off values
    of the residual norm.

    RelaxationLS Also checks bounds and enforces them by one of three methods.

    Attributes
    ----------
    _iter_count : int
        Number of iterations for the current invocation of the solver.
    """

    SOLVER = 'LS: RLX'

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super(RelaxationLS, self)._declare_options()
        opt = self.options
        opt.declare(
            'bound_enforcement', default='vector', values=['vector', 'scalar', 'wall'],
            desc="If this is set to 'vector', the entire vector is backtracked together " +
                 "when a bound is violated. If this is set to 'scalar', only the violating " +
                 "entries are set to the bound and then the backtracking occurs on the vector " +
                 "as a whole. If this is set to 'wall', only the violating entries are set " +
                 "to the bound, and then the backtracking follows the wall - i.e., the " +
                 "violating entries do not change during the line search.")
        opt.declare('print_bound_enforce', default=False,
                    desc="Set to True to print out names and values of variables that are pulled "
                    "back to their bounds.")

        # Main control parameters.
        opt.declare('initial_relaxation', default=0.25,
                    desc="Initial value of relaxation parameter (i.e., far from solution.)")
        opt.declare('norm_far', default=1e-2,
                    desc='Value of absolute residual norm above which the initial relaxation is '
                    'used.')
        opt.declare('norm_near', default=1e-3,
                    desc='Value of absolute residual norm below which the no relaxation is used. '
                    '(i.e., relaxation parameter = 1.0')

        # Remove unused options from base options here, so that users
        # attempting to set them will get KeyErrors.
        opt.undeclare("atol")
        opt.undeclare("rtol")
        opt.undeclare("maxiter")
        opt.undeclare("err_on_maxiter")

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : System
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super(RelaxationLS, self)._setup_solvers(system, depth)

        opts = self.options

        if opts['norm_far'] < opts['norm_near']:
            msg = "In options, norm_far must be greater than or equal to norm_near."
            raise ValueError(msg)

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

        self._run_apply()

        norm0 = self._iter_get_norm()
        if norm0 == 0.0:
            norm0 = 1.0
        self._norm0 = norm0

        # Determine relaxation parameter.
        norm_far = self.options['norm_far']
        norm_near = self.options['norm_near']
        alpha_far = self.options['initial_relaxation']
        if norm0 >= norm_far:
            alpha = alpha_far
        elif norm0 > norm_near:
            alpha = alpha_far + (1.0 - alpha_far) * (norm0 - norm_near) / (norm_far - norm_near)
        else:
            alpha = 1.0

        u += alpha*du

        if self.options['print_bound_enforce']:
            _print_violations(u, system._lower_bounds, system._upper_bounds)

        with Recording('RelaxationLS', self._iter_count, self) as rec:
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

        fail = (np.isinf(norm) or np.isnan(norm))
        return fail, norm, norm / norm0
