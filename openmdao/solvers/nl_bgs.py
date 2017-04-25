"""Define the NonlinearBlockGS class."""

from openmdao.solvers.solver import NonlinearSolver


class NonlinearBlockGS(NonlinearSolver):
    """
    Nonlinear block Gauss-Seidel solver.
    """

    SOLVER = 'NL: NLBGS'

    def _iter_execute(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system
        self._solver_info.prefix += '|  '

        for isub, subsys in enumerate(system._subsystems_allprocs):
            system._transfer('nonlinear', 'fwd', isub)
            if subsys in system._subsystems_myproc:
                subsys._solve_nonlinear()

        self._solver_info.prefix = self._solver_info.prefix[:-3]

    def _mpi_print_header(self):
        """
        Print header text before solving.
        """
        if (self.options['iprint'] > 0 and self._system.comm.rank == 0 and
            not self._system._suppress_solver_output):

            pathname = self._system.pathname
            if pathname:
                nchar = len(pathname)
                prefix = self._solver_info.prefix
                header = prefix + "\n"
                header += prefix + nchar*"=" + "\n"
                header += prefix + pathname + "\n"
                header += prefix+ nchar*"="
                print(header)