"""Define the NonlinearBlockGS class."""

import numpy as np

from openmdao.solvers.solver import NonlinearSolver
from openmdao.utils.mpi import MPI


class NonlinearBlockGS(NonlinearSolver):
    """
    Nonlinear block Gauss-Seidel solver.

    Attributes
    ----------
    _delta_outputs_n_1 : ndarray
        Cached change in the full output vector for the previous iteration. Only used if the aitken
        acceleration option is turned on.
    _theta_n_1 : float
        Cached relaxation factor from previous iteration. Only used if the aitken acceleration
        option is turned on.
    """

    SOLVER = 'NL: NLBGS'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super(NonlinearBlockGS, self).__init__(**kwargs)

        self._theta_n_1 = 1.0
        self._delta_outputs_n_1 = None

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

        rank = MPI.COMM_WORLD.rank if MPI is not None else 0

        if len(system._subsystems_allprocs) != len(system._subsystems_myproc):
            raise RuntimeError('{}: Nonlinear Gauss-Seidel cannot be used on a '
                               'parallel group.'.format(self.msginfo))

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super(NonlinearBlockGS, self)._declare_options()

        self.options.declare('use_aitken', types=bool, default=False,
                             desc='set to True to use Aitken relaxation')
        self.options.declare('aitken_min_factor', default=0.1,
                             desc='lower limit for Aitken relaxation factor')
        self.options.declare('aitken_max_factor', default=1.5,
                             desc='upper limit for Aitken relaxation factor')
        self.options.declare('cs_reconverge', types=bool, default=True,
                             desc='When True, when this driver solves under a complex step, nudge '
                             'the Solution vector by a small amount so that it reconverges.')
        self.options.declare('use_apply_nonlinear', types=bool, default=False,
                             desc="Set to True to always call apply_nonlinear on the solver's "
                             "system after solve_nonlinear has been called.")
        self.options.declare('reraise_child_analysiserror', types=bool, default=False,
                             desc='When the option is true, a solver will reraise any '
                             'AnalysisError that arises during subsolve; when false, it will '
                             'continue solving.')

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
        system = self._system()

        if self.options['use_aitken']:
            self._delta_outputs_n_1 = system._outputs._data.copy()
            self._theta_n_1 = 1.

        # When under a complex step from higher in the hierarchy, sometimes the step is too small
        # to trigger reconvergence, so nudge the outputs slightly so that we always get at least
        # one iteration.
        if system.under_complex_step and self.options['cs_reconverge']:
            system._outputs._data += np.linalg.norm(system._outputs._data) * 1e-10

        # Execute guess_nonlinear if specified.
        system._guess_nonlinear()

        return super(NonlinearBlockGS, self)._iter_initialize()

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system()
        outputs = system._outputs
        residuals = system._residuals
        use_aitken = self.options['use_aitken']

        if use_aitken:

            aitken_min_factor = self.options['aitken_min_factor']
            aitken_max_factor = self.options['aitken_max_factor']

            # some variables that are used for Aitken's relaxation
            delta_outputs_n_1 = self._delta_outputs_n_1
            theta_n_1 = self._theta_n_1

            # store a copy of the outputs, used to compute the change in outputs later
            delta_outputs_n = outputs._data.copy()

        if use_aitken or not self.options['use_apply_nonlinear']:
            # store a copy of the outputs
            if not self.options['use_apply_nonlinear']:
                with system._unscaled_context(outputs=[outputs]):
                    outputs_n = outputs._data.copy()
            else:
                outputs_n = outputs._data.copy()

        self._solver_info.append_subsolver()
        self._gs_iter()
        self._solver_info.pop()

        if use_aitken:
            # compute the change in the outputs after the NLBGS iteration
            delta_outputs_n -= outputs._data
            delta_outputs_n *= -1

            if self._iter_count >= 2:
                # Compute relaxation factor. This method is used by Kenway et al. in
                # "Scalable Parallel Approach for High-Fidelity Steady-State Aero-
                # elastic Analysis and Adjoint Derivative Computations" (ln 22 of Algo 1)

                temp = delta_outputs_n.copy()
                temp -= delta_outputs_n_1

                # If MPI, piggyback on the residual vector to perform a distributed norm.
                if system.comm.size > 1:
                    backup_r = residuals._data.copy()
                    residuals._data[:] = temp
                    temp_norm = residuals.get_norm()
                else:
                    temp_norm = np.linalg.norm(temp)

                if temp_norm == 0.:
                    temp_norm = 1e-12  # prevent division by 0 below

                # If MPI, piggyback on the output and residual vectors to perform a distributed
                # dot product.
                if system.comm.size > 1:
                    backup_o = outputs._data.copy()
                    outputs._data[:] = delta_outputs_n
                    tddo = residuals.dot(outputs)
                    residuals._data[:] = backup_r
                    outputs._data[:] = backup_o
                else:
                    tddo = temp.dot(delta_outputs_n)

                theta_n = theta_n_1 * (1 - tddo / temp_norm ** 2)

                # limit relaxation factor to the specified range
                theta_n = max(aitken_min_factor, min(aitken_max_factor, theta_n))

                # save relaxation factor for the next iteration
                theta_n_1 = theta_n
            else:
                theta_n = 1.

            if not self.options['use_apply_nonlinear']:
                with system._unscaled_context(outputs=[outputs]):
                    outputs._data[:] = outputs_n
            else:
                outputs._data[:] = outputs_n

            # compute relaxed outputs
            outputs._data += theta_n * delta_outputs_n

            # save update to use in next iteration
            delta_outputs_n_1[:] = delta_outputs_n

        if not self.options['use_apply_nonlinear']:
            # Residual is the change in the outputs vector.
            with system._unscaled_context(outputs=[outputs], residuals=[residuals]):
                residuals._data[:] = outputs._data - outputs_n

    def _run_apply(self):
        """
        Run the apply_nonlinear method on the system.
        """
        system = self._system()
        maxiter = self.options['maxiter']
        itercount = self._iter_count

        if self.options['use_apply_nonlinear'] or (itercount < 1 and maxiter < 2):

            # This option runs apply_nonlinear to calculate the residuals, and thus ends up
            # executing ExplicitComponents twice per iteration.

            self._recording_iter.push(('_run_apply', 0))
            try:
                system._apply_nonlinear()
            finally:
                self._recording_iter.pop()

        elif itercount < 1:
            # Run instead of calling apply, so that we don't "waste" the extra run. This also
            # further increments the iteration counter.
            itercount += 1
            outputs = system._outputs
            residuals = system._residuals

            with system._unscaled_context(outputs=[outputs]):
                outputs_n = outputs._data.copy()

            self._solver_info.append_subsolver()
            for isub, subsys in enumerate(system._subsystems_allprocs):
                system._transfer('nonlinear', 'fwd', isub)
                if subsys._is_local:
                    subsys._solve_nonlinear()

            self._solver_info.pop()
            with system._unscaled_context(residuals=[residuals]):
                residuals._data[:] = outputs._data - outputs_n

    def _mpi_print_header(self):
        """
        Print header text before solving.
        """
        if (self.options['iprint'] > 0 and self._system().comm.rank == 0):

            pathname = self._system().pathname
            if pathname:
                nchar = len(pathname)
                prefix = self._solver_info.prefix
                header = prefix + "\n"
                header += prefix + nchar * "=" + "\n"
                header += prefix + pathname + "\n"
                header += prefix + nchar * "="
                print(header)
