"""Define the NonlinearBlockGS class."""

import numpy as np

from openmdao.solvers.solver import NonlinearSolver


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
        self._convrg_vars = None
        self._convrg_rtols = None

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
        self.options.declare('cs_reconverge', default=True,
                             desc='When True, when this driver solves under a complex step, nudge '
                             'the Solution vector by a small amount so that it reconverges.')
        self.options.declare('use_apply_nonlinear', False,
                             desc="Set to True to always call apply_linear on the solver's system "
                             "after solve_nonlinear has been called.")
        self.options.declare('convrg_vars', types=list, default=[],
                             desc='list of variables (names) used by relative error criterium.')
        self.options.declare('convrg_rtols', types=list, default=[],
                             desc='list of relative error tolerances corresponding to each'
                             ' variable specified in convrg_vars option (rtol is used otherwise)')

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
        if self.options['use_aitken']:
            outputs = self._system._outputs
            self._delta_outputs_n_1 = outputs._data.copy()
            self._theta_n_1 = 1.

        # When under a complex step from higher in the hierarchy, sometimes the step is too small
        # to trigger reconvergence, so nudge the outputs slightly so that we always get at least
        # one iteration.
        if self._system.under_complex_step and self.options['cs_reconverge']:
            self._system._outputs._data += np.linalg.norm(self._system._outputs._data) * 1e-10

        self._convrg_vars = self.options['convrg_vars']
        if self._convrg_vars and not self.options['convrg_rtols']:
            rtol = self.options['rtol']
            self._convrg_rtols = rtol * np.ones(len(self._convrg_vars))
        else:
            self._convrg_rtols = self.options['convrg_rtols']
            if len(self._convrg_rtols) != len(self._convrg_vars):
                raise RuntimeError('Convergence rtols bad size : should be {}, '
                                   'found {}.'.format(len(self._convrg_vars),
                                                      len(self._convrg_rtols)))

        return super(NonlinearBlockGS, self)._iter_initialize()

    def _is_rtol_converged(self, norm, norm0):
        """
        Check convergence regarding relative error tolerance.

        Parameters
        ----------
        norm : float
            error (residuals norm)
        norm0 : float
            initial error

        Returns
        -------
        bool
            whether convergence is reached regarding relative error tolerance
        """
        if self._convrg_vars:
            nbvars = len(self._convrg_vars)
            rerrs = np.ones(nbvars)
            outputs = np.ones(nbvars)
            for i, name in enumerate(self._convrg_vars):
                outputs[i] = self._system._outputs._views[name]
                residual = self._system._residuals._views[name]
                rerrs[i] = np.linalg.norm(residual) / np.linalg.norm(outputs[i])
            is_rtol_converged = (rerrs < self._convrg_rtols).all()
        else:
            is_rtol_converged = super(NonlinearBlockGS, self)._is_rtol_converged(norm, norm0)
        return is_rtol_converged

    def _iter_get_norm(self):
        """
        Return the norm of the residual regarding convergence variable settings.

        Returns
        -------
        float
            norm.
        """
        if self._convrg_vars:
            total = []
            for name in self._convrg_vars:
                total.append(self._system._residuals._views_flat[name])
            norm = np.linalg.norm(np.concatenate(total))
        else:
            norm = super(NonlinearBlockGS, self)._iter_get_norm()
        return norm

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system
        outputs = system._outputs
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
                temp_norm = np.linalg.norm(temp)
                if temp_norm == 0.:
                    temp_norm = 1e-12  # prevent division by 0 in the next line
                theta_n = theta_n_1 * (1 - temp.dot(delta_outputs_n) / temp_norm ** 2)

                # limit relaxation factor to the specified range
                theta_n = max(aitken_min_factor, min(aitken_max_factor, theta_n))

                # save relaxation factor for the next iteration
                theta_n_1 = theta_n
            else:
                theta_n = 1.

            outputs._data[:] = outputs_n

            # compute relaxed outputs
            outputs._data += theta_n * delta_outputs_n

            # save update to use in next iteration
            delta_outputs_n_1[:] = delta_outputs_n

        if not self.options['use_apply_nonlinear']:
            # Residual is the change in the outputs vector.
            system._residuals._data[:] = outputs._data - outputs_n

    def _run_apply(self):
        """
        Run the apply_nonlinear method on the system.
        """
        system = self._system
        maxiter = self.options['maxiter']
        itercount = self._iter_count

        if self.options['use_apply_nonlinear'] or (itercount < 1 and maxiter < 2):

            # This option runs apply_linear to calculate the residuals, and thus ends up executing
            # ExplicitComponents twice per iteration.

            self._recording_iter.stack.append(('_run_apply', 0))
            try:
                system._apply_nonlinear()
            finally:
                self._recording_iter.stack.pop()

        elif itercount < 1:
            # Run instead of calling apply, so that we don't "waste" the extra run. This also
            # further increments the iteration counter.
            itercount += 1
            outputs = system._outputs

            outputs_n = outputs._data.copy()

            self._solver_info.append_subsolver()
            for isub, subsys in enumerate(system._subsystems_myproc):
                system._transfer('nonlinear', 'fwd', isub)
                subsys._solve_nonlinear()
                system._check_reconf_update()

            self._solver_info.pop()
            system._residuals._data[:] = outputs._data - outputs_n

    def _mpi_print_header(self):
        """
        Print header text before solving.
        """
        if (self.options['iprint'] > 0 and self._system.comm.rank == 0):

            pathname = self._system.pathname
            if pathname:
                nchar = len(pathname)
                prefix = self._solver_info.prefix
                header = prefix + "\n"
                header += prefix + nchar * "=" + "\n"
                header += prefix + pathname + "\n"
                header += prefix + nchar * "="
                print(header)
