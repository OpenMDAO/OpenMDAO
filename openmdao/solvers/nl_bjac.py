"""Define the NonlinearBlockJac class."""
from openmdao.solvers.solver import NonlinearSolver


class NonlinearBlockJac(NonlinearSolver):
    """
    Nonlinear block Jacobi solver.
    """

    SOLVER = 'NL: NLBJ'

    def _iter_execute(self):
        """
        Perform the operations in the iteration loop.
        """
        self._solver_info.prefix += '|  '

        self._system._transfer('nonlinear', 'fwd')
        for subsys in self._system._subsystems_myproc:
            subsys._solve_nonlinear()
        self._system._check_reconf_update()

        self._solver_info.prefix = self._solver_info.prefix[:-3]

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
