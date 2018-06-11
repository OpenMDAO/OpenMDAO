"""
Define the BroydenSolver class.

Based on implementation in Scipy via OpenMDAO 0.8x
"""
from __future__ import print_function

import numpy as np

from openmdao.recorders.recording_iteration_stack import Recording
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
        if self.options['debug_print']:
            self._err_cache['inputs'] = deepcopy(self._system._inputs)
            self._err_cache['outputs'] = deepcopy(self._system._outputs)

        system = self._system

        # Execute guess_nonlinear if specified.
        system._guess_nonlinear()

        # Cache initial states.
        self.xm = self.get_states()

        with Recording('Broyden', 0, self):

            self._solver_info.append_solver()

            # should call the subsystems solve before computing the first residual
            for isub, subsys in enumerate(system._subsystems_myproc):
                system._transfer('nonlinear', 'fwd', isub)
                subsys._solve_nonlinear()
                system._check_reconf_update()

            self._solver_info.pop()

        self._run_apply()
        norm = self._iter_get_norm()

        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def _iter_get_norm(self):
        """
        Return the norm of only the residuals requested in options.

        Returns
        -------
        float
            norm.
        """
        fxm = self.get_residuals()
        return np.sum(fxm**2) ** 0.5

    def _iter_execute(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system
        Gm = self._update_jacobian()
        fxm = self.fxm

        delta_xm = -Gm.dot(fxm)
        xm = self.xm + delta_xm.T

        # Update the new states in the model.
        self.set_states(xm)

        # Run the model.
        with Recording('Broyden', 0, self):
            self._solver_info.append_solver()

            for isub, subsys in enumerate(system._subsystems_allprocs):
                system._transfer('nonlinear', 'fwd', isub)

                if subsys in system._subsystems_myproc:
                    subsys._solve_nonlinear()

            self._solver_info.pop()

        self._run_apply()

        fxm1 = fxm.copy()
        fxm = self.get_residuals()
        delta_fxm = fxm1 - fxm

        # Determine whether to update Jacobian.
        self._recompute_jacobian = False

        # Cache for next jacobian update.
        self.delta_xm = delta_xm
        self.delta_fxm = delta_fxm
        self.fxm = fxm

    """
        xm = self.xin.T
        Fxm = numpy.matrix(self.F).T
        Gm = -self.alpha*numpy.matrix(numpy.identity(len(self.xin)))

        for n in range(self.itmax):

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

    def _update_jacobian(self):
        """
        Update the Jacobian for a new Broyden iteration.

        Returns
        -------
        ndarray
            Updated Jacobian.
        """
        # Use Broyden Update.
        if not self._recompute_jacobian:
            dfxm = self.delta_fxm
            Gm = self.Gm
            Gm += (self.delta_xm - Gm * dfxm) * dfxm.T / np.linalg.norm(dfxm)**2

        # Solve for total derivatives of user-requested residuals wrt states.
        elif self.options['compute_initial_jacobian']:
            # TODO: do this
            pass

        # Reset Jacobian to identiy scaled by alpha.
        else:
            Gm = -self.options['alpha'] * np.identity(self.n)

        return Gm

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
        outputs = self._system._outputs
        i = 0
        for name in states:
            val = outputs[name]
            n_size = len(val)
            xm[i:i + n_size] = val
            i += n_size

        return xm

    def set_states(self, new_val):
        """
        Set new values for states specified in options.

        Parameters
        ----------
        new_val : ndarray
            New values for states.
        """
        states = self.options['state_vars']
        outputs = self._system._outputs
        i = 0
        for name in states:
            val = outputs[name]
            n_size = len(val)
            outputs[name] = new_val[i:i + n_size]
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
        fxm = self.fxm
        residuals = self._system._residuals
        i = 0
        for name in states:
            val = residuals[name]
            n_size = len(val)
            fxm[i:i + n_size] = val
            i += n_size

        return fxm

