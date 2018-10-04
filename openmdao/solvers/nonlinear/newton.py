"""Define the NewtonSolver class."""

from __future__ import print_function

from copy import deepcopy

from openmdao.solvers.solver import NonlinearSolver
from openmdao.recorders.recording_iteration_stack import Recording, recording_iteration
from openmdao.utils.general_utils import warn_deprecation


class NewtonSolver(NonlinearSolver):
    """
    Newton solver.

    The default linear solver is the linear_solver in the containing system.

    Attributes
    ----------
    linear_solver : LinearSolver
        Linear solver to use to find the Newton search direction. The default
        is the parent system's linear solver.
    linesearch : NonlinearSolver
        Line search algorithm. Default is None for no line search.
    options : OptionsDictionary
        options dictionary.
    _system : System
        pointer to the owning system.
    _depth : int
        how many subsolvers deep this solver is (0 means not a subsolver).
    _vec_names : [str, ...]
        list of right-hand-side (RHS) vector names.
    _mode : str
        'fwd' or 'rev', applicable to linear solvers only.
    _iter_count : int
        Number of iterations for the current invocation of the solver.
    """

    SOLVER = 'NL: Newton'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super(NewtonSolver, self).__init__(**kwargs)

        # Slot for linear solver
        self.linear_solver = None

        # Slot for linesearch
        self.linesearch = None

    @property
    def line_search(self):
        """
        Return the current linesearch object.
        """
        warn_deprecation("The 'line_search' attribute provides backwards compatibility "
                         "with OpenMDAO 1.x ; use 'linesearch' instead.")
        return self.linesearch

    @line_search.setter
    def line_search(self, solver):
        """
        Set the linesearch solver.
        """
        warn_deprecation("The 'line_search' attribute provides backwards compatibility "
                         "with OpenMDAO 1.x ; use 'linesearch' instead.")
        self.linesearch = solver

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super(NewtonSolver, self)._declare_options()

        self.options.declare('solve_subsystems', types=bool, default=False,
                             desc='Set to True to turn on sub-solvers (Hybrid Newton).')
        self.options.declare('max_sub_solves', types=int, default=10,
                             desc='Maximum number of subsystem solves.')

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
        super(NewtonSolver, self)._setup_solvers(system, depth)

        self._disallow_discrete_outputs()

        if self.linear_solver is not None:
            self.linear_solver._setup_solvers(self._system, self._depth + 1)
        else:
            self.linear_solver = system.linear_solver

        if self.linesearch is not None:
            self.linesearch._setup_solvers(self._system, self._depth + 1)

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
        super(NewtonSolver, self)._set_solver_print(level=level, type_=type_)

        if self.linear_solver is not None and type_ != 'NL':
            self.linear_solver._set_solver_print(level=level, type_=type_)

        if self.linesearch is not None:
            self.linesearch._set_solver_print(level=level, type_=type_)

    def _run_apply(self):
        """
        Run the apply_nonlinear method on the system.
        """
        recording_iteration.stack.append(('_run_apply', 0))

        system = self._system

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        try:
            system._apply_nonlinear()
        finally:
            recording_iteration.stack.pop()

        # Enable local fd
        system._owns_approx_jac = approx_status

    def _linearize_children(self):
        """
        Return a flag that is True when we need to call linearize on our subsystems' solvers.

        Returns
        -------
        boolean
            Flag for indicating child linerization
        """
        return (self.options['solve_subsystems']
                and self._iter_count <= self.options['max_sub_solves'])

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
        if self.options['debug_print']:
            self._err_cache['inputs'] = deepcopy(self._system._inputs)
            self._err_cache['outputs'] = deepcopy(self._system._outputs)

        system = self._system

        # Execute guess_nonlinear if specified.
        system._guess_nonlinear()

        with Recording('Newton_subsolve', 0, self):
            if self.options['solve_subsystems'] and \
               (self._iter_count <= self.options['max_sub_solves']):

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

    def _iter_execute(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system
        self._solver_info.append_subsolver()
        do_subsolve = self.options['solve_subsystems'] and \
            (self._iter_count < self.options['max_sub_solves'])
        do_sub_ln = self.linear_solver._linearize_children()

        # Disable local fd
        approx_status = system._owns_approx_jac
        system._owns_approx_jac = False

        system._vectors['residual']['linear'].set_vec(system._residuals)
        system._vectors['residual']['linear'] *= -1.0
        my_asm_jac = self.linear_solver._assembled_jac

        system._linearize(my_asm_jac, sub_do_ln=do_sub_ln)
        if (my_asm_jac is not None and system.linear_solver._assembled_jac is not my_asm_jac):
            my_asm_jac._update(system)
        self._linearize()

        self.linear_solver.solve(['linear'], 'fwd')

        if self.linesearch:
            self.linesearch._do_subsolve = do_subsolve
            self.linesearch.solve()
        else:
            system._outputs += system._vectors['output']['linear']

        self._solver_info.pop()

        # Hybrid newton support.
        if do_subsolve:
            with Recording('Newton_subsolve', 0, self):
                self._solver_info.append_solver()

                for isub, subsys in enumerate(system._subsystems_allprocs):
                    system._transfer('nonlinear', 'fwd', isub)

                    if subsys in system._subsystems_myproc:
                        subsys._solve_nonlinear()

                self._solver_info.pop()

        # Enable local fd
        system._owns_approx_jac = approx_status

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

    def cleanup(self):
        """
        Clean up resources prior to exit.
        """
        super(NewtonSolver, self).cleanup()

        if self.linear_solver:
            self.linear_solver.cleanup()
        if self.linesearch:
            self.linesearch.cleanup()
