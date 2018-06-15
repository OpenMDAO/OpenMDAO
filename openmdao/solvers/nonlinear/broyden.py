"""
Define the BroydenSolver class.

Based on implementation in Scipy via OpenMDAO 0.8x with improvements based on NPSS solver.
"""
from __future__ import print_function

from six.moves import range

import numpy as np

from openmdao.core.analysis_error import AnalysisError
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.solver import NonlinearSolver

CITATION = """@ARTICLE{
              Broyden1965ACo,
              AUTHOR = "C. Broyden",
              TITLE = "A Class of Methods for Solving Nonlinear Simultaneous Equations",
              JOURNAL = "Mathematics of Computation",
              VOLUME = "19",
              YEAR = "1965",
              PAGES = "577--593",
              REFERRED = "[Coleman1996SaE]."
              }"""


class BroydenSolver(NonlinearSolver):
    """
    Broyden solver.

    Attributes
    ----------
    delta_fxm : ndarray
        Most recent change in residual vector.
    delta_xm : ndarray
        Most recent change in state vector.
    fxm : ndarray
        Most recent residual.
    Gm : ndarray
        Most recent Jacobian matrix.
    linear_solver : LinearSolver
        Linear solver to use for calculating jacobian.
    n : int
        Total length of the states being solved.
    xm : ndarray
        Most recent state.
    _idx : dict
        Cache of vector indices for each state name.
    _computed_jacobians : int
        Number of computed jacobians.
    _converge_failures : int
        Number of consecutive iterations that failed to converge to the tol definied in options.
    _recompute_jacobian : bool
        Flag that becomes True when Broyden detects it needs to recompute the Jacobian.
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

        # Slot for linear solver
        self.linear_solver = None

        self.cite = CITATION

        self.n = 0
        self._idx = {}
        self._recompute_jacobian = True
        self.Gm = None
        self.xm = None
        self.fxm = None
        self.delta_xm = None
        self.delta_fxm = None
        self._converge_failures = 0
        self._computed_jacobians = 0

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super(BroydenSolver, self)._declare_options()

        self.options.declare('alpha', default=0.4,
                             desc="Value to scale the starting Jacobian, which is Identity. This "
                                  "option does nothing if you compute the initial Jacobian "
                                  "instead.")
        self.options.declare('compute_jacobian', default=False,
                             desc="Set to True to compute an initial Jacobian, otherwise start "
                                  "with Identity scaled by alpha. Further Jacobians may also be "
                                  "computed depending on the other options.")
        self.options.declare('converge_limit', default=1.0,
                             desc="Ratio of current residual to previous residual above which the "
                                  "convergence is considered a failure. The Jacobian will be "
                                  "regenerated once this condition has been reached a number of "
                                  "consecutive times as specified in max_converge_failures.")
        self.options.declare('diverge_limit', default=2.0,
                             desc="Ratio of current residual to previous residual above which the "
                                  "Jacobian will be immediately regenerated.")
        self.options.declare('max_converge_failures', default=3,
                             desc="The number of convergence failures before regenerating the "
                                  "Jacobian.")
        self.options.declare('max_jacobians', default=10,
                             desc="Maximum number of jacobians to compute.")
        self.options.declare('state_vars', [], desc="List of the state-variable/residuals that "
                                                    "are to be solved here.")
        self.options.declare('update_broyden', default=True,
                             desc="Flag controls whether to perform Broyden update to the "
                                  "Jacobian. There are some applications where it may be useful "
                                  "to turn this off.")

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
        self._computed_jacobians = 0

        if self.linear_solver is not None:
            self.linear_solver._setup_solvers(self._system, self._depth + 1)
        else:
            self.linear_solver = system.linear_solver

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
            size = len(outputs[name])
            self._idx[name] = (n, n + size)
            n += size

        self.n = n
        self.Gm = np.empty((n, n))
        self.xm = np.empty((n, ))
        self.fxm = np.empty((n, ))
        self.delta_xm = None
        self.delta_fxm = None

    def _assembled_jac_solver_iter(self):
        """
        Return a generator of linear solvers using assembled jacs.
        """
        if self.linear_solver is not None:
            for s in self.linear_solver._assembled_jac_solver_iter():
                yield s

    def _set_solver_print(self, level=2, type_='all'):
        """
        Control printing for solvers and subsolvers in the model.

        Parameters
        ----------
        level : int
            iprint level. Set to 2 to print residuals each iteration; set to 1
            to print just the iteration totals; set to 0 to disable all printing
            except for failures, and set to -1 to disable all printing including failures.
        type_ : str
            Type of solver to set: 'LN' for linear, 'NL' for nonlinear, or 'all' for all.
        """
        super(BroydenSolver, self)._set_solver_print(level=level, type_=type_)

        if self.linear_solver is not None and type_ != 'NL':
            self.linear_solver._set_solver_print(level=level, type_=type_)

    def _linearize(self):
        """
        Perform any required linearization operations such as matrix factorization.
        """
        if self.linear_solver is not None:
            self.linear_solver._linearize()

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
        self._converge_failures = 0
        self._computed_jacobians = 0

        # Execute guess_nonlinear if specified.
        system._guess_nonlinear()

        # Start with initial states.
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
        xm = self.xm + delta_xm

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
        delta_fxm = fxm - fxm1

        # Determine whether to update Jacobian.
        self._recompute_jacobian = False
        opt = self.options
        if self._computed_jacobians <= opt['max_jacobians']:
            converge_ratio = np.linalg.norm(fxm) / np.linalg.norm(fxm1)

            if converge_ratio > opt['diverge_limit']:
                self._recompute_jacobian = True
            elif converge_ratio > opt['converge_limit']:
                self._converge_failures += 1

                if self._converge_failures > opt['max_converge_failures']:
                    self._recompute_jacobian = True
            else:
                self._converge_failures = 0

        # Cache for next iteration.
        self.delta_xm = delta_xm
        self.delta_fxm = delta_fxm
        self.fxm = fxm
        self.xm = xm
        self.Gm = Gm

    def _update_jacobian(self):
        """
        Update the Jacobian for a new Broyden iteration.

        Returns
        -------
        ndarray
            Updated Jacobian.
        """
        Gm = self.Gm

        # Use Broyden Update.
        if not self._recompute_jacobian:
            dfxm = self.delta_fxm
            fact = 1.0 / np.linalg.norm(dfxm)**2
            Gm += np.outer((self.delta_xm - Gm.dot(dfxm)), dfxm * fact)

        # Solve for total derivatives of user-requested residuals wrt states.
        elif self.options['compute_jacobian']:
            Gm = self._compute_jacobian()
            self._computed_jacobians += 1

        # Set Jacobian to identity scaled by alpha.
        # This is the default initial Jacobian used by scipy.
        elif self.options['update_broyden']:
            Gm = np.diag(-self.options['alpha'] * np.ones(self.n))

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
        for name in states:
            val = outputs[name]
            i, j = self._idx[name]
            xm[i:j] = val

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
        for name in states:
            i, j = self._idx[name]
            outputs[name] = new_val[i:j]

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
        for name in states:
            val = residuals[name]
            i, j = self._idx[name]
            fxm[i:j] = val

        return fxm

    def _compute_jacobian(self):
        """
        Compute Jacobian for system using OpenMDAO.

        Returns
        -------
        ndarray
            New Jacobian.
        """
        # TODO : Consider promoting this capability out into OpenMDAO so other solvers can use the
        # same code.
        # TODO : Build this Jacobian in parallel if procs are available.
        system = self._system
        states = self.options['state_vars']
        d_res = system._vectors['residual']['linear']
        d_out = system._vectors['output']['linear']

        jac = self.Gm
        d_res.set_const(0.0)

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        # Linearize model.
        ln_solver = self.linear_solver
        do_sub_ln = ln_solver._linearize_children()
        my_asm_jac = ln_solver._assembled_jac
        system._linearize(my_asm_jac, sub_do_ln=do_sub_ln)
        if (my_asm_jac is not None and ln_solver._assembled_jac is not my_asm_jac):
            my_asm_jac._update(system)
        self._linearize()

        for wrt_name in states:
            i_wrt, j_wrt = self._idx[wrt_name]
            d_wrt = d_res[wrt_name]
            for j in range(j_wrt - i_wrt):

                # Increment each variable.
                d_wrt[j] = 1.0

                # Solve for total derivatives.
                ln_solver.solve(['linear'], 'fwd')

                # Extract results.
                for of_name in states:
                    i_of, j_of = self._idx[of_name]
                    jac[i_of:j_of, i_wrt + j] = d_out[of_name]

                d_wrt[j] = 0.0

        # Enable local fd
        system._owns_approx_jac = approx_status

        return jac
