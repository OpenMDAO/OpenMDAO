"""
Define the BroydenSolver class.

Based on implementation in Scipy via OpenMDAO 0.8x with improvements based on NPSS solver.
"""
import numpy as np

from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.solvers.linesearch.backtracking import BoundsEnforceLS
from openmdao.solvers.solver import NonlinearSolver
from openmdao.utils.class_util import overrides_method
from openmdao.warnings import issue_warning, SetupWarning
from openmdao.utils.mpi import MPI


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
        Linear solver to use for calculating inverse Jacobian.
    linesearch : NonlinearSolver
        Line search algorithm. Default is None for no line search.
    size : int
        Total length of the states being solved.
    xm : ndarray
        Most recent state.
    _idx : dict
        Cache of vector indices for each state name.
    _computed_jacobians : int
        Number of computed jacobians.
    _converge_failures : int
        Number of consecutive iterations that failed to converge to the tol definied in options.
    _full_inverse : bool
        When True, Broyden considers the whole vector rather than a list of states.
    _recompute_jacobian : bool
        Flag that becomes True when Broyden detects it needs to recompute the inverse Jacobian.
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
        super().__init__(**kwargs)

        # Slot for linear solver
        self.linear_solver = None

        # Slot for linesearch
        self.linesearch = BoundsEnforceLS()

        self.cite = CITATION

        self.size = 0
        self._idx = {}
        self._recompute_jacobian = True
        self.Gm = None
        self.xm = None
        self.fxm = None
        self.delta_xm = None
        self.delta_fxm = None
        self._converge_failures = 0
        self._computed_jacobians = 0

        # This gets set to True if the user doesn't declare any states.
        self._full_inverse = False

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare('alpha', default=0.4,
                             desc="Value to scale the starting Jacobian, which is Identity. This "
                                  "option does nothing if you compute the initial Jacobian "
                                  "instead.")
        self.options.declare('compute_jacobian', types=bool, default=True,
                             desc="When True, compute an initial Jacobian, otherwise start "
                                  "with Identity scaled by alpha. Further Jacobians may also be "
                                  "computed depending on the other options.")
        self.options.declare('converge_limit', default=1.0,
                             desc="Ratio of current residual to previous residual above which the "
                                  "convergence is considered a failure. The Jacobian will be "
                                  "regenerated once this condition has been reached a number of "
                                  "consecutive times as specified in max_converge_failures.")
        self.options.declare('cs_reconverge', types=bool, default=True,
                             desc='When True, when this driver solves under a complex step, nudge '
                             'the Solution vector by a small amount so that it reconverges.')
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
        self.options.declare('reraise_child_analysiserror', types=bool, default=False,
                             desc='When the option is true, a solver will reraise any '
                             'AnalysisError that arises during subsolve; when false, it will '
                             'continue solving.')

        self.supports['gradients'] = True
        self.supports['implicit_components'] = True

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
        super()._setup_solvers(system, depth)
        self._recompute_jacobian = True
        self._computed_jacobians = 0
        iproc = system.comm.rank

        rank = MPI.COMM_WORLD.rank if MPI is not None else 0
        self._disallow_discrete_outputs()

        if self.linear_solver is not None:
            self.linear_solver._setup_solvers(system, self._depth + 1)
        else:
            self.linear_solver = system.linear_solver

        if self.linesearch is not None:
            self.linesearch._setup_solvers(system, self._depth + 1)
            self.linesearch._do_subsolve = True

        # this check is incorrect (for broyden) and needs to be done differently.
        # self._disallow_distrib_solve()

        states = self.options['state_vars']
        prom2abs = system._var_allprocs_prom2abs_list['output']

        # Check names of states.
        bad_names = [name for name in states if name not in prom2abs]
        if len(bad_names) > 0:
            msg = "{}: The following variable names were not found: {}"
            raise ValueError(msg.format(self.msginfo, ', '.join(bad_names)))

        # Size linear system
        if len(states) > 0:
            # User has specified states, so we must size them.
            n = 0
            meta = system._var_allprocs_abs2meta['output']

            for i, name in enumerate(states):
                size = meta[prom2abs[name][0]]['global_size']
                self._idx[name] = (n, n + size)
                n += size
        else:
            # Full system size.
            self._full_inverse = True
            n = np.sum(system._owned_sizes)

        self.size = n
        self.Gm = np.empty((n, n))
        self.xm = np.empty((n, ))
        self.fxm = np.empty((n, ))
        self.delta_xm = None
        self.delta_fxm = None

        if self._full_inverse:

            # Can only use DirectSolver here.
            from openmdao.solvers.linear.direct import DirectSolver
            if not isinstance(self.linear_solver, DirectSolver):
                msg = "{}: Linear solver must be DirectSolver when solving the full model."
                raise ValueError(msg.format(self.msginfo, ', '.join(bad_names)))

            return

        # Always look for states that aren't being solved so we can warn the user.
        def sys_recurse(system, all_states):
            subs = system._subsystems_myproc
            if len(subs) == 0:

                # Skip implicit components that appear to solve themselves.
                from openmdao.core.implicitcomponent import ImplicitComponent
                if overrides_method('solve_nonlinear', system, ImplicitComponent):
                    return

                all_states.extend(system._list_states())

            else:
                for subsys in subs:
                    sub_nl = subsys.nonlinear_solver
                    if sub_nl and sub_nl.supports['implicit_components']:
                        continue
                    sys_recurse(subsys, all_states)

        all_states = []
        sys_recurse(system, all_states)
        all_states = [system._var_abs2prom['output'][name] for name in all_states]

        missing = set(all_states).difference(states)
        if len(missing) > 0:
            msg = "The following states are not covered by a solver, and may have been " + \
                  "omitted from the BroydenSolver 'state_vars': "
            msg += ', '.join(sorted(missing))
            issue_warning(msg, category=SetupWarning)

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
        super()._set_solver_print(level=level, type_=type_)

        if self.linear_solver is not None and type_ != 'NL':
            self.linear_solver._set_solver_print(level=level, type_=type_)

        if self.linesearch is not None:
            self.linesearch._set_solver_print(level=level, type_=type_)

    def _linearize(self):
        """
        Perform any required linearization operations such as matrix factorization.
        """
        if self.linear_solver is not None:
            self.linear_solver._linearize()

        if self.linesearch is not None:
            self.linesearch._linearize()

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
        system = self._system()
        if self.options['debug_print']:
            self._err_cache['inputs'] = system._inputs._copy_views()
            self._err_cache['outputs'] = system._outputs._copy_views()

        # Convert local storage if we are under complex step.
        if system.under_complex_step:
            self.Gm = self.Gm.astype(np.complex)
            self.xm = self.xm.astype(np.complex)
            self.fxm = self.fxm.astype(np.complex)
        elif np.iscomplexobj(self.xm):
            self.Gm = self.Gm.real
            self.xm = self.xm.real
            self.fxm = self.fxm.real

        self._converge_failures = 0
        self._computed_jacobians = 0

        # Execute guess_nonlinear if specified.
        system._guess_nonlinear()

        # When under a complex step from higher in the hierarchy, sometimes the step is too small
        # to trigger reconvergence, so nudge the outputs slightly so that we always get at least
        # one iteration of Broyden.
        if system.under_complex_step and self.options['cs_reconverge']:
            system._outputs += np.linalg.norm(system._outputs.asarray()) * 1e-10

        # Start with initial states.
        self.xm = self.get_vector(system._outputs)

        with Recording('Broyden', 0, self):
            self._solver_info.append_solver()

            # should call the subsystems solve before computing the first residual
            self._gs_iter()

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
            Norm of the residuals.
        """
        # Need to cache the initial residuals, which is done in this function.
        self.fxm = fxm = self.get_vector(self._system()._residuals)
        if not self._full_inverse:
            # Use full model residual for driving the main loop convergence.
            fxm = self._system()._residuals.asarray()

        return self.compute_norm(fxm)

    def compute_norm(self, vec):
        """
        Compute norm of the vector.

        Under MPI, compute the norm on rank 0, and broadcast it to all other ranks.

        Parameters
        ----------
        vec : ndarray
            Array of real or complex values. For MPI on rank 0, should be full dimension of the
            openmdao vector with duplicate indices removed.

        Returns
        -------
        float
            Norm of vec, computed on rank 0 and broadcast to all other ranks.
        """
        return np.linalg.norm(vec)

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system()
        Gm = self._update_inverse_jacobian()
        fxm = self.fxm

        delta_xm = -Gm.dot(fxm)

        if self.linesearch:
            self._solver_info.append_subsolver()

            self.set_states(self.xm)
            self.set_linear_vector(delta_xm)
            self.linesearch.solve()
            xm = self.get_vector(system._outputs)

            self._solver_info.pop()

        else:
            # Update the new states in the model.
            xm = self.xm + delta_xm
            self.set_states(xm)

        # Run the model.
        with Recording('Broyden', 0, self):
            self._solver_info.append_solver()
            self._gs_iter()
            self._solver_info.pop()

        self._run_apply()

        fxm1 = fxm.copy()
        self.fxm = fxm = self.get_vector(system._residuals)
        delta_fxm = fxm - fxm1

        # States may have been further converged hierarchically.
        xm = self.get_vector(system._outputs)
        delta_xm = xm - self.xm

        # Determine whether to update Jacobian.
        self._recompute_jacobian = False
        opt = self.options
        if self._computed_jacobians <= opt['max_jacobians']:

            converge_ratio = self.compute_norm(fxm) / self.compute_norm(fxm1)

            if converge_ratio > opt['diverge_limit']:
                self._recompute_jacobian = True
            elif converge_ratio > opt['converge_limit']:
                self._converge_failures += 1

                if self._converge_failures >= opt['max_converge_failures']:
                    self._recompute_jacobian = True
            else:
                self._converge_failures = 0

        # Cache for next iteration.
        self.delta_xm = delta_xm
        self.delta_fxm = delta_fxm
        self.fxm = fxm
        self.xm = xm
        self.Gm = Gm

    def _update_inverse_jacobian(self):
        """
        Update the inverse Jacobian for a new Broyden iteration.

        Returns
        -------
        ndarray
            Updated inverse Jacobian.
        """
        Gm = self.Gm

        # Apply the Broyden Update approximation to the previous value of the inverse jacobian.
        if self.options['update_broyden'] and not self._recompute_jacobian:
            dfxm = self.delta_fxm
            fact = np.linalg.norm(dfxm)

            # Sometimes you can get stuck, particularly when enforcing bounds in a linesearch.
            # Make sure we don't update in this case because of divide by zero.
            if fact > self.options['atol']:
                Gm += np.outer((self.delta_xm - Gm.dot(dfxm)), dfxm * (1.0 / fact**2))

        # Solve for total derivatives of user-requested residuals wrt states.
        elif self.options['compute_jacobian']:
            if self._full_inverse:
                Gm = self._compute_full_inverse_jacobian()
            else:
                Gm = self._compute_inverse_jacobian()

            self._computed_jacobians += 1

        # Set inverse Jacobian to identity scaled by alpha.
        # This is the default starting point used by scipy and the general broyden algorithm.
        else:
            Gm = np.diag(np.full(self.size, -self.options['alpha'], dtype=Gm.dtype))

        return Gm

    def get_vector(self, vec):
        """
        Return a vector containing the values of vec at the states specified in options.

        This is the full incoming vec when no states are defined. When under MPI, the values are
        appopriately gathered without duplicates to rank 0.

        Parameters
        ----------
        vec : <Vector>
            Vector from which to extract state values.

        Returns
        -------
        ndarray
            Array containing values of vector at desired states.
        """
        if self._full_inverse:
            xm = vec.asarray(copy=True)
        else:
            states = self.options['state_vars']
            xm = self.xm.copy()
            for name in states:
                i, j = self._idx[name]
                xm[i:j] = vec[name]

        return xm

    def set_states(self, new_val):
        """
        Set new values for states specified in options.

        Parameters
        ----------
        new_val : ndarray
            New values for states.
        """
        outputs = self._system()._outputs

        if self._full_inverse:
            outputs.set_val(new_val)
        else:
            states = self.options['state_vars']
            for name in states:
                i, j = self._idx[name]
                outputs[name] = new_val[i:j]

    def set_linear_vector(self, dx):
        """
        Copy values from step into the linear vector for backtracking.

        Parameters
        ----------
        dx : ndarray
            Full step in the states for this iteration.
        """
        linear = self._system()._vectors['output']['linear']

        if self._full_inverse:
            linear.set_val(dx)
        else:
            linear.set_val(0.0)
            for name in self.options['state_vars']:
                i, j = self._idx[name]
                linear[name] = dx[i:j]

    def _compute_inverse_jacobian(self):
        """
        Compute inverse Jacobian for system by doing a linear solve for each state.

        Returns
        -------
        ndarray
            New inverse Jacobian.
        """
        # TODO: Consider promoting this capability out into OpenMDAO so other solvers can use the
        # same code.
        # TODO: Can do each state in parallel if procs are available.
        system = self._system()
        states = self.options['state_vars']
        d_res = system._vectors['residual']['linear']
        d_out = system._vectors['output']['linear']

        inv_jac = self.Gm
        d_res.set_val(0.0)

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        # Linearize model.
        ln_solver = self.linear_solver
        do_sub_ln = ln_solver._linearize_children()
        my_asm_jac = ln_solver._assembled_jac
        system._linearize(my_asm_jac, sub_do_ln=do_sub_ln)
        if my_asm_jac is not None and system.linear_solver._assembled_jac is not my_asm_jac:
            my_asm_jac._update(system)
        self._linearize()

        for wrt_name in states:
            i_wrt, j_wrt = self._idx[wrt_name]
            if wrt_name in d_res:
                d_wrt = d_res[wrt_name]

            for j in range(j_wrt - i_wrt):

                # Increment each variable.
                if wrt_name in d_res:
                    d_wrt[j] = 1.0

                # Solve for total derivatives.
                ln_solver.solve(['linear'], 'fwd')

                # Extract results.
                for of_name in states:
                    i_of, j_of = self._idx[of_name]
                    inv_jac[i_of:j_of, i_wrt + j] = d_out[of_name]

                if wrt_name in d_res:
                    d_wrt[j] = 0.0

        # Enable local fd
        system._owns_approx_jac = approx_status

        return inv_jac

    def _compute_full_inverse_jacobian(self):
        """
        Compute inverse Jacobian for entire system vector.

        Only the DirectSolver is supported here.

        Returns
        -------
        ndarray
            New inverse Jacobian.
        """
        system = self._system()

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        # Linearize model.
        ln_solver = self.linear_solver
        do_sub_ln = ln_solver._linearize_children()
        my_asm_jac = ln_solver._assembled_jac
        system._linearize(my_asm_jac, sub_do_ln=do_sub_ln)
        if my_asm_jac is not None and system.linear_solver._assembled_jac is not my_asm_jac:
            my_asm_jac._update(system)

        inv_jac = self.linear_solver._inverse()

        # Enable local fd
        system._owns_approx_jac = approx_status

        return inv_jac

    def cleanup(self):
        """
        Clean up resources prior to exit.
        """
        super().cleanup()

        if self.linear_solver:
            self.linear_solver.cleanup()
        if self.linesearch:
            self.linesearch.cleanup()
