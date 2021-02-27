"""Define the NonlinearBlockJac class."""
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.solver import NonlinearSolver
from openmdao.utils.mpi import multi_proc_fail_check


class NonlinearBlockJac(NonlinearSolver):
    """
    Nonlinear block Jacobi solver.
    """

    SOLVER = 'NL: NLBJ'

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system()
        self._solver_info.append_subsolver()
        system._transfer('nonlinear', 'fwd')

        with Recording('NonlinearBlockJac', 0, self) as rec:

            # If this is a parallel group, check for analysis errors and reraise.
            if len(system._subsystems_myproc) != len(system._subsystems_allprocs):
                with multi_proc_fail_check(system.comm):
                    for subsys in system._subsystems_myproc:
                        subsys._solve_nonlinear()
            else:
                for subsys in system._subsystems_myproc:
                    subsys._solve_nonlinear()

            rec.abs = 0.0
            rec.rel = 0.0

        self._solver_info.pop()

    def _run_apply(self):
        """
        Run the apply_nonlinear method on the system.
        """
        system = self._system()

        # If this is a parallel group, check for analysis errors and reraise.
        if len(system._subsystems_myproc) != len(system._subsystems_allprocs):
            with multi_proc_fail_check(system.comm):
                super()._run_apply()
        else:
            super()._run_apply()
