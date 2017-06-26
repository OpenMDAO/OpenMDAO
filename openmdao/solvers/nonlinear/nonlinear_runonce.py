"""
Define the NonLinearRunOnce class.

This is a simple nonlinear solver that just runs the system once.
"""

from six.moves import range

from openmdao.solvers.solver import NonlinearSolver
from openmdao.utils.record_util import create_local_meta


class NonLinearRunOnce(NonlinearSolver):
    """
    Simple solver that runs the containing system once.

    This is done without iteration or norm calculation.
    """

    SOLVER = 'NL: RUNONCE'

    def solve(self):
        """
        Run the solver.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        system = self._system


        from openmdao.recorders.base_recorder import recording

        with recording('NLRunOnce', 0):
            # If this is a parallel group, transfer all at once then run each subsystem.
            if len(system._subsystems_myproc) != len(system._subsystems_allprocs):
                system._transfer('nonlinear', 'fwd')
                for subsys in system._subsystems_myproc:
                    subsys._solve_nonlinear()
                system._check_reconf_update()
            # If this is not a parallel group, transfer for each subsystem just prior to running it.
            else:
                for isub, subsys in enumerate(system._subsystems_myproc):
                    system._transfer('nonlinear', 'fwd', isub)
                    subsys._solve_nonlinear()
                    system._check_reconf_update()

            metadata = create_local_meta('NonLinearRunOnce')
            self._rec_mgr.record_iteration(self, metadata)

        return False, 0.0, 0.0
