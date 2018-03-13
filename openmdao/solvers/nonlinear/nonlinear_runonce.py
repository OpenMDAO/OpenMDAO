"""
Define the NonlinearRunOnce class.

This is a simple nonlinear solver that just runs the system once.
"""
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.solver import NonlinearSolver
from openmdao.utils.general_utils import warn_deprecation
from openmdao.utils.mpi import multi_proc_fail_check


class NonlinearRunOnce(NonlinearSolver):
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

        with Recording('NLRunOnce', 0, self) as rec:
            # If this is a parallel group, transfer all at once then run each subsystem.
            if len(system._subsystems_myproc) != len(system._subsystems_allprocs):
                system._transfer('nonlinear', 'fwd')

                with multi_proc_fail_check(system.comm):
                    for subsys in system._subsystems_myproc:
                        subsys._solve_nonlinear()

                system._check_reconf_update()

            # If this is not a parallel group, transfer for each subsystem just prior to running it.
            else:
                for isub, subsys in enumerate(system._subsystems_myproc):
                    system._transfer('nonlinear', 'fwd', isub)
                    subsys._solve_nonlinear()
                    system._check_reconf_update()
            rec.abs = 0.0
            rec.rel = 0.0

        return False, 0.0, 0.0

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        # changing the default maxiter from the base class
        self.options.declare('maxiter', default=0, values=(0,),
                             desc='maximum number of iterations '
                                  '(this solver does not iterate)')


class NonLinearRunOnce(NonlinearRunOnce):
    """
    Deprecated.  See NonlinearRunOnce.
    """

    def __init__(self, *args, **kwargs):
        """
        Deprecated.

        Parameters
        ----------
        *args : list of object
            Positional args.
        **kwargs : dict
            Named args.
        """
        super(NonLinearRunOnce, self).__init__(*args, **kwargs)
        warn_deprecation('NonLinearRunOnce is deprecated.  Use NonlinearRunOnce instead.')
