"""Define the NonlinearSchur class."""


import numpy as np

from openmdao.solvers.solver import NonlinearSolver
from openmdao.recorders.recording_iteration_stack import Recording
import scipy


class NonlinearSchur(NonlinearSolver):
    """
    Nonlinear Schur complement solver implementation.

    The default linear solver is the linear_solver in the containing system.

    Parameters
    ----------
    mode_nonlinear : str
        Specify whether this system is being solved in 'fwd' or 'rev' mode.
    sys_names : Sequence
        A sequence of the coupled systems to be solved with the NonlienarSchurSolver.
        If not provided, these will be determined automatically. If provided,
        there must be two systems.
    bounds : dict
        A dictionary of lower and upper bounds.
    **kwargs : dict
        Options passed to the creation of the solver.

    Attributes
    ----------
    linear_solver : LinearSolver
        Linear solver to use to find the Newton search direction. The default
        is the parent system's linear solver.
    _mode_nonlinear : str
        Specify whether this system is being solved in 'fwd' or 'rev' mode.
    _sys_names : Sequence
        A sequence of the coupled systems to be solved with the NonlienarSchurSolver.
        If not provided, these will be determined automatically. If provided,
        there must be two systems.
    _bounds : dict
        A dictionary of lower and upper bounds
    """

    SOLVER = 'NL: NLSCHUR'

    def __init__(self, mode_nonlinear='rev', sys_names=None, bounds=None, **kwargs):
        """
        Initialize all attributes.
        """
        self._bounds = bounds
        self._mode_nonlinear = mode_nonlinear
        self._sys_names = sys_names
        super().__init__(**kwargs)

        # Slot for linear solver
        self.linear_solver = None

        # Slot for linesearch
        # self.linesearch = BoundsEnforceLS()

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare('solve_subsystems', types=bool,
                             desc='Set to True to turn on sub-solvers (Hybrid Newton).')
        self.options.declare('max_sub_solves', types=int, default=10,
                             desc='Maximum number of subsystem solves.')
        self.options.declare('lamda', types=float, default=1.0,
                             desc='Constant parameter for inexact schur step. 0 < lamda <=1.'
                                  ' That is x(i+1) = x(i) + lamda * dx')
        self.options.declare('factor_lamda', types=float, default=None,
                             desc='When to use "lamda". When the nonlinear solution '
                             'is too far away, we may want to take smaller steps to '
                             'reach to the solution. However, when the the nonlinear '
                             'solution is close to the fianl solution, we may want '
                             'to allow the solver to take larger steps. This factor '
                             'limits the allowable ratio of the new step to the '
                             'current state variable value. For example, setting 1 '
                             'will let the solver to use "lamda" only when the ratio '
                             'of the new step to the current state variable value '
                             'is greater than equal to 1.')
        self.options.declare(
            'cs_reconverge',
            types=bool,
            default=True,
            desc='When True, when this driver solves under a complex step, nudge '
            'the Solution vector by a small amount so that it reconverges.',
        )
        self.options.declare(
            'reraise_child_analysiserror',
            types=bool,
            default=False,
            desc='When the option is true, a solver will reraise any '
            'AnalysisError that arises during subsolve; when false, it will '
            'continue solving.',
        )

        self.supports['gradients'] = True
        self.supports['implicit_components'] = True

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : System
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super()._setup_solvers(system, depth)

        self._disallow_discrete_outputs()

        if self._sys_names is None:
            self._sys_names = [s for s in system._subsystems_allprocs.keys() if s != '_auto_ivc']

        if len(self._sys_names) != 2:
            raise ValueError(f'System {self.pathname} has a NonlinearSchur solver and is '
                             'required to contain two subsystems, but it has '
                             f'{len(self._sys_names)}.\n{self._sys_names}')

        if not isinstance(self.options._dict['solve_subsystems']['val'], bool):
            msg = '{}: solve_subsystems must be set by the user.'
            raise ValueError(msg.format(self.msginfo))

        if self.linear_solver is not None:
            self.linear_solver._setup_solvers(system, self._depth + 1)
        else:
            self.linear_solver = system.linear_solver

        if self.linesearch is not None:
            self.linesearch._setup_solvers(system, self._depth + 1)

    def _assembled_jac_solver_iter(self):
        """
        Return a generator of linear solvers using assembled jacs.
        """
        if self.linear_solver is not None:
            for s in self.linear_solver._assembled_jac_solver_iter():
                yield s

    def _linearize_children(self):
        """
        Return a flag that is True when we need to call linearize on our subsystems' solvers.

        Returns
        -------
        bool
            Flag for indicating child linerization
        """
        return (
            self.options['solve_subsystems']
            and not self._system().under_complex_step
            and self._iter_count <= self.options['max_sub_solves']
        )

    def _set_solver_print(self, level=2, type_='all', debug_print=None):
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
        debug_print : bool or None
            If None, leave solver debug printing unchanged, otherwise turn if on or off
            depending on whether debug_print is True or False.
        """
        super()._set_solver_print(level=level, type_=type_, debug_print=debug_print)

        if self.linear_solver is not None and type_ != 'NL':
            self.linear_solver._set_solver_print(level=level, type_=type_)

        if self.linesearch is not None:
            self.linesearch._set_solver_print(level=level, type_=type_, debug_print=debug_print)

    def _run_apply(self):
        """
        Run the apply_nonlinear method on the system.
        """
        self._recording_iter.push(('_run_apply', 0))

        system = self._system()

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        try:
            system._apply_nonlinear()
        finally:
            self._recording_iter.pop()

        # Enable local fd
        system._owns_approx_jac = approx_status

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
            initial error.
        float
            error at the first iteration.
        """
        system = self._system()
        solve_subsystems = self.options['solve_subsystems'] and not system.under_complex_step

        if self.options['debug_print']:
            self._err_cache['inputs'] = system._inputs._copy_views()
            self._err_cache['outputs'] = system._outputs._copy_views()

        # Execute guess_nonlinear if specified.
        system._guess_nonlinear()

        with Recording('Newton_subsolve', 0, self) as rec:
            if solve_subsystems and self._iter_count <= self.options['max_sub_solves']:
                self._solver_info.append_solver()

                # should call the subsystems solve before computing the first residual
                self._gs_iter()

                self._solver_info.pop()

            self._run_apply()
            norm = self._iter_get_norm()

            rec.abs = norm
            norm0 = norm if norm != 0.0 else 1.0
            rec.rel = norm / norm0

        return norm0, norm

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system()
        self._solver_info.append_subsolver()
        do_subsolve = (
            self.options['solve_subsystems']
            and not system.under_complex_step
            and (self._iter_count < self.options['max_sub_solves'])
        )
        do_sub_ln = self.linear_solver._linearize_children()

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        system._vectors['residual']['linear'].set_vec(system._residuals)
        system._vectors['residual']['linear'] *= -1.0
        my_asm_jac = self.linear_solver._assembled_jac

        # my_asm_jac should not be passed to _linearize?
        # system._linearize(my_asm_jac, sub_do_ln=do_sub_ln)
        # if my_asm_jac is not None and system.linear_solver._assembled_jac is not my_asm_jac:
        #     my_asm_jac._update(system)

        system._linearize(sub_do_ln=do_sub_ln)
        if my_asm_jac is not None and system.linear_solver._assembled_jac is not my_asm_jac:
            my_asm_jac._update(system)

        # self._linearize()
        # extract the first and second subsystems
        subsys1, _ = system._subsystems_allprocs[self._sys_names[0]]
        subsys2, _ = system._subsystems_allprocs[self._sys_names[1]]

        # ideally, we are solving for all outputs in subsys2
        # so this will be our jacobian size
        # TODO this may not be the most general case. think about just solving for a subset
        subsys2_outputs = subsys2._outputs
        subsys2_residuals = subsys2._residuals
        # list of variables we solve for here. this should include all variables in
        # subsys2 ideally because we dont do anything else for this subsystem here
        vars_to_solve = [*subsys2_outputs.keys()]

        resd_to_solve = [*subsys2_residuals.keys()]

        # total size of the jacobian
        n_vars = 0
        for var in vars_to_solve:
            n_vars += subsys2_outputs[var].size

        # initialize the schur complement jacobian for these variables
        # TODO better way to get the dtype?
        schur_jac = np.zeros((n_vars, n_vars),
                             dtype=system._vectors['residual']['linear'].asarray(copy=True).dtype)

        # backup the vectors we are working with
        rvec = system._vectors['residual']['linear']
        ovec = system._vectors['output']['linear']
        ivec = system._vectors['input']['linear']

        r_data = rvec.asarray(copy=True)
        o_data = ovec.asarray(copy=True)
        i_data = ivec.asarray(copy=True)

        # mode
        mode = self._mode_nonlinear

        if mode == 'fwd':
            # schur_jacobian

            # Schur_Jac = D - C A^-1 B ##

            # set the ovec to zeros
            ovec.set_val(np.zeros(len(ovec)))

            for ii, var in enumerate(vars_to_solve):
                # set the linear seed of the variable we want to solve for in subsys 2
                ovec[f'{subsys2.name}.{var}'] = 1.0

                # transfer this seed to the first subsystem
                system._transfer('linear', mode, subsys1.name)

                # run the jac-vec computation in the first subsystem,
                # this ll give us the B[:,{ii}] vector
                scope_out, scope_in = system._get_matvec_scope(subsys1)

                subsys1._apply_linear(mode, scope_out, scope_in)

                # amd then, by performing solve_linear we get A^-1 B[:,{ii}]
                subsys1._solve_linear(mode, scope_out, scope_in)

                # Do another mat-mult with the solution of this linear system.
                # We want to get the final jacobian using the schur method here,
                # so we will need to do a bit more math.

                # first negate the vector from the linear solve
                subsys1._vectors['output']['linear'] *= -1.0

                # finally, set the seed of the variable to 1 as well to get
                # the diagonal contribution
                # system._vectors['output']['linear'][f'{subsys2.name}.{var}']
                # this should already be at one since we perturbed it above!

                # transfer the outputs to inputs
                system._transfer('linear', mode)

                # run the apply linear. we do it on the complete system here
                # the result is the final jacobian for this using the
                # schur complement method D[:,{ii}] - C A^-1 B[:,{ii}]
                scope_out, scope_in = system._get_matvec_scope()
                system._apply_linear(mode, scope_out, scope_in)

                # put this value into the jacobian.
                schur_jac[:, ii] = subsys2._vectors['residual']['linear'].asarray()

                # set back the seed to zero for the next vector
                ovec[f'{subsys2.name}.{var}'] = 0.0
        else:  # rev mode
            # Schur_Jac = D - C A^-1 B ##

            rvec.set_val(np.zeros(len(rvec)))

            for ii, var in enumerate(resd_to_solve):
                # set the linear seed of the variable we want to solve for in subsys 2

                rvec[f'{subsys2.name}.{var}'] = 1.0

                # we get the C[{ii},:] vector by apply_linear on the system
                scope_out, scope_in = system._get_matvec_scope()
                system._apply_linear(mode, scope_out, scope_in)

                # do a solve_linear to find C[{ii},:] A^-1
                scope_out, scope_in = system._get_matvec_scope(subsys1)
                subsys1._solve_linear(mode, scope_out, scope_in)

                # negate the resdiual first
                subsys1._vectors['residual']['linear'] *= -1.0

                # do a apply_linear on the subsys1 to find the D[{ii},:] - C[{ii},:] A^-1 B
                scope_out, scope_in = system._get_matvec_scope(subsys1)
                subsys1._apply_linear(mode, scope_out, scope_in)

                system._transfer('linear', mode, subsys2.name)

                # put this value into the jacobian.
                schur_jac[ii, :] = subsys2._vectors['output']['linear'].asarray()

                # set back the seed to zero for the next vector
                rvec[f'{subsys2.name}.{var}'] = 0.0

            # schur_jacobian

        # put back the vectors
        rvec.set_val(r_data)
        ovec.set_val(o_data)
        ivec.set_val(i_data)

        # we now have the schur complement of the jacobian for the second component.
        # do a newton update with it!
        subsys2._vectors['residual']['linear'].set_vec(subsys2._residuals)
        subsys2._vectors['residual']['linear'] *= -1.0

        iD_schur = np.eye(n_vars,
                          dtype=system._vectors['residual']['linear'].asarray(copy=True).dtype)
        iD_schur *= 1.E-16
        schur_jac = schur_jac + iD_schur
        d_subsys2 = scipy.linalg.solve(schur_jac, subsys2._vectors['residual']['linear'].asarray())

        # loop over the variables just to be safe with the ordering
        for ii, var in enumerate(vars_to_solve):
            var_name = f'{subsys2.name}.{var}'
            if self.options['factor_lamda'] is None:
                system._outputs[var_name] += self.options['lamda'] * d_subsys2[ii]
            else:
                if abs(d_subsys2[ii]) >= abs(self.options['factor_lamda'] *
                                             system._outputs[var_name]):
                    system._outputs[var_name] += self.options['lamda'] * d_subsys2[ii]
                else:
                    system._outputs[var_name] += d_subsys2[ii]

        if self._bounds is not None:
            for key in self._bounds.keys():
                if key == 'lower':
                    lowerB = self._bounds['lower']
                elif key == 'upper':
                    upperB = self._bounds['upper']
            for ii, var in enumerate(vars_to_solve):
                if system._outputs[f'{subsys2.name}.{var}'] < lowerB[ii]:
                    system._outputs[f'{subsys2.name}.{var}'] = lowerB[ii]
                elif system._outputs[f'{subsys2.name}.{var}'] > upperB[ii]:
                    system._outputs[f'{subsys2.name}.{var}'] = upperB[ii]

        # print outputs
        # if system.comm.rank == 0:
        #     print('\n+  -------------------')
        #     print('+  Balance variables:')
        #     print('+  -------------------\n+')
        #     for ii, var in enumerate(vars_to_solve):
        #         ivar = var.split('.')
        #         print('+ ', ivar[-1], ' = ', system._outputs[f'{subsys2.name}.{var}'][0])
        #     print('\n')

        self._solver_info.pop()

        # Hybrid newton support.
        if do_subsolve:
            with Recording('Newton_subsolve', 0, self):
                self._solver_info.append_solver()
                self._gs_iter()
                self._solver_info.pop()

        # Enable local fd
        system._owns_approx_jac = approx_status

    def _set_complex_step_mode(self, active):
        """
        Turn on or off complex stepping mode.

        Recurses to turn on or off complex stepping mode in all subsystems and their vectors.

        Parameters
        ----------
        active : bool
            Complex mode flag; set to True prior to commencing complex step.
        """
        if self.linear_solver is not None:
            self.linear_solver._set_complex_step_mode(active)
            if self.linear_solver._assembled_jac is not None:
                self.linear_solver._assembled_jac.set_complex_step_mode(active)

    def cleanup(self):
        """
        Clean up resources prior to exit.
        """
        super().cleanup()

        if self.linear_solver:
            self.linear_solver.cleanup()
        if self.linesearch:
            self.linesearch.cleanup()
