"""Define the NonlinearBlockGS class."""

from openmdao.solvers.solver import NonlinearSolver


class NonlinearBlockGS(NonlinearSolver):
    """
    Nonlinear block Gauss-Seidel solver.
    """

    SOLVER = 'NL: NLBGS'

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
            self._aitken_work1 = self._system._outputs._clone()
            self._aitken_work2 = self._system._outputs._clone()
            self._aitken_work3 = self._system._outputs._clone()
            self._aitken_work4 = self._system._outputs._clone()
            self._theta_n_1 = 1.

        return super(NonlinearBlockGS, self)._iter_initialize()

    def _iter_execute(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system
        use_aitken = self.options['use_aitken']

        if use_aitken:
            outputs = self._system._outputs
            aitken_min_factor = self.options['aitken_min_factor']
            aitken_max_factor = self.options['aitken_max_factor']

            # some variables that are used for Aitken's relaxation
            delta_outputs_n_1 = self._aitken_work1
            delta_outputs_n = self._aitken_work2
            outputs_n = self._aitken_work3
            temp = self._aitken_work4
            theta_n_1 = self._theta_n_1
            # store a copy of the outputs, used to compute the change in outputs later
            delta_outputs_n.set_vec(outputs)
            # store a copy of the outputs
            outputs_n.set_vec(outputs)

        self._solver_info.append_subsolver()
        for isub, subsys in enumerate(system._subsystems_myproc):
            system._transfer('nonlinear', 'fwd', isub)
            subsys._solve_nonlinear()
            system._check_reconf_update()

        self._solver_info.pop()

        if use_aitken:
            # compute the change in the outputs after the NLBGS iteration
            delta_outputs_n -= outputs
            delta_outputs_n *= -1

            if self._iter_count >= 2:
                # Compute relaxation factor. This method is used by Kenway et al. in
                # "Scalable Parallel Approach for High-Fidelity Steady-State Aero-
                # elastic Analysis and Adjoint Derivative Computations" (ln 22 of Algo 1)

                temp.set_vec(delta_outputs_n)
                temp -= delta_outputs_n_1
                temp_norm = temp.get_norm()
                if temp_norm == 0.:
                    temp_norm = 1e-12  # prevent division by 0 in the next line
                theta_n = theta_n_1 * (1 - temp.dot(delta_outputs_n) / temp_norm ** 2)
                # limit relaxation factor to the specified range
                theta_n = max(aitken_min_factor, min(aitken_max_factor, theta_n))
                # save relaxation factor for the next iteration
                theta_n_1 = theta_n
            else:
                theta_n = 1.

            outputs.set_vec(outputs_n)

            # compute relaxed outputs
            outputs.add_scal_vec(theta_n, delta_outputs_n)

            # save update to use in next iteration
            delta_outputs_n_1.set_vec(delta_outputs_n)

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
