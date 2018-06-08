"""
Define the BroydenSolver class.

Based on implementation in Scipy via OpenMDAO 0.8x
"""
from __future__ import print_function

from openmdao.solvers.solver import NonlinearSolver


class BroydenSolver(NonlinearSolver):
    """
    Broyden solver.
    """

    SOLVER = 'BROYDEN'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super(BroydenSolver, self).__init__(**kwargs)

        self.n = 0

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super(BroydenSolver, self)._declare_options()

        self.options.declare('alpha', default=0.4,
                             desc="Value to scale the starting Jacobian, which is Identity. This "
                                  "option does nothing if you precompute the initial Jacobian.")
        self.options.declare('compute_initial_jacobian', default=False,
                             desc="Set to True to compute the initial Jacobian, otherwise start "
                                  "with Identity scaled by alpha.")
        self.options.declare('state_vars', [], desc="List of the state-variable/residuals that "
                                                    "are solver states.")

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : <System>
            Pointer to the owning system.
        depth : int
            Depth of the current system (already incremented).
        """
        super(BroydenSolver, self)._setup_solvers(system, depth)

        states = self.options['state_vars']
        prom = system._var_allprocs_prom2abs_list['output']

        # Check names of states.
        bad_names = [name for name in states if name not in prom]
        if len(bad_names) > 0:
            msg = "The following variable names were not found: {}"
            raise ValueError(msg.format(', '.join(bad_names)))

        # Size linear system
        n = 0
        outputs = system._outputs
        for name in states:
            n += len(outputs[name])

        self.n = n

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            Initial relative error in the user-specified residuals.
        float
            Initial absolute error in the user-specified residuals.
        """
        return super(BroydenSolver, self)._iter_initialize()

    def _iter_execute(self):
        """
        Perform the operations in the iteration loop.
        """
        pass


"""
        xm = self.xin.T
        Fxm = numpy.matrix(self.F).T
        Gm = -self.alpha*numpy.matrix(numpy.identity(len(self.xin)))

        for n in range(self.itmax):

            if self._stop:
                self.raise_exception('Stop requested', RunStopped)

            deltaxm = -Gm*Fxm
            xm = xm + deltaxm.T

            # update the new independents in the model
            self.set_parameters(numpy.asarray(xm).flat)

            # run the model
            self.run_iteration()

            # get dependents
            self.F[:] = self.eval_eq_constraints()

            # successful termination if independents are below tolerance
            #print "iter", n, norm(self.F)
            if norm(self.F) < self.tol:
                return

            Fxm1 = numpy.matrix(self.F).T
            deltaFxm = Fxm1 - Fxm

            if norm(deltaFxm) == 0:
                msg = "Broyden iteration has stopped converging. Change in " \
                      "input has produced no change in output. This could " \
                      "indicate a problem with your component connections. " \
                      "It could also mean that this solver method is " \
                      "inadequate for your problem."
                raise RuntimeError(msg)

            Fxm = Fxm1.copy()
            Gm = Gm + (deltaxm-Gm*deltaFxm)*deltaFxm.T/norm(deltaFxm)**2
"""