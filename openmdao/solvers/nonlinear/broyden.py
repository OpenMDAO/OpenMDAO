"""
Define the BroydenSolver class.

Based on implementation in Scipy via OpenMDAO 0.8x
"""
from __future__ import print_function

import numpy as np

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
        self._recompute_jacobian = True

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
        self._recompute_jacobian = True

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
        self.Gm = np.empty((n, n))
        self.xm = np.empty((n, ))
        self.fxm = np.empty((n, ))
        self.delta_xm = None
        self.delta_fxm = None

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

    def _compute_jacobian(self):
        """
        Compute the Jacobian for the state/residual equations specified in the options.
        """
        # Use Broyden Update.
        if not self._recompute_jacobian:
            dfxm = self.delta_fxm
            Gm = self.Gm
            Gm += (self.delta_xm - Gm * dfxm) * dfxm.T / np.linalg.norm(dfxm)**2
            return

        # Solve for total derivatives of user-requested residuals wrt states.
        if self.options['compute_initial_jacobian']:
            # TODO: this
            pass

        # Reset Jacobian to identiy scaled by alpha.
        else:
            self.Gm = -self.alpha * np.identity(self.n)

    def get_states(self):
        """
        Return a vector containing the values of the states specified in options.

        This is used to get the initial state guesses.

        Returns
        -------
        ndarray
            Array containing values of states.
        """
        states = self.options['state_vars']
        xm = self.xm
        outputs = self.system._outputs
        i = 0
        for name in states:
            val = outputs[name]
            n_size = len(val)
            xm[i:i + n_size] = val
            i += n_size

        return xm

    def set_states(self):
        """
        Return a vector containing the values of the states specified in options.

        This is used to get the initial state guesses.

        Returns
        -------
        ndarray
            Array containing values of states.
        """
        states = self.options['state_vars']
        xm = self.xm
        outputs = self.system._outputs
        i = 0
        for name in states:
            val = outputs[name]
            n_size = len(val)
            xm[i:i + n_size] = val
            i += n_size

    def get_residuals(self):
        """
        Return a vector containing the values of the residuals specified in options.

        Returns
        -------
        ndarray
            Array containing values of residuals.
        """
        states = self.options['state_vars']
        fm = self.fm
        residuals = self.system._residuals
        i = 0
        for name in states:
            val = residuals[name]
            n_size = len(val)
            fm[i:i + n_size] = val
            i += n_size

        return fm

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