"""Define the NewtonSolver class."""

from openmdao.solvers.solver import NonlinearSolver


class NewtonSolver(NonlinearSolver):
    """
    Newton solver.

    The default linear solver is the ln_solver in the containing system.

    Attributes
    ----------
    ln_solver : <LinearSolver>
        Linear solver to use to find the Newton search direction. The default
        is the parent system's linear solver.
    linesearch : <NonlinearSolver>
        Line search algorithm. Default is None for no line search.
    options : <OptionsDictionary>
        options dictionary.
    _system : <System>
        pointer to the owning system.
    _depth : int
        how many subsolvers deep this solver is (0 means not a subsolver).
    _vec_names : [str, ...]
        list of right-hand-side (RHS) vector names.
    _mode : str
        'fwd' or 'rev', applicable to linear solvers only.
    _iter_count : int
        Number of iterations for the current invocation of the solver.
    _ln_solver_from_parent : bool
        This is set to True if we are using the parent system's linear solver.
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
        self.ln_solver = None

        # Slot for linesearch
        self.linesearch = None

        # We only need to call linearize on the ln_solver if its not shared with the parent group.
        self._ln_solver_from_parent = True

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('solve_subsystems', type_=bool, value=False,
                             desc='Set to True to turn on sub-solvers (Hybrid Newton).')
        self.options.declare('max_sub_solves', type_=int, value=10,
                             desc='Maximum number of subsystem solves.')
        self.supports['gradients'] = True

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : <System>
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super(NewtonSolver, self)._setup_solvers(system, depth)

        if self.ln_solver is not None:
            self.ln_solver._setup_solvers(self._system, self._depth + 1)
            self._ln_solver_from_parent = False
        else:
            self.ln_solver = system.ln_solver

        if self.linesearch is not None:
            self.linesearch._setup_solvers(self._system, self._depth + 1)

    def _iter_get_norm(self):
        """
        Return the norm of the residual.

        Returns
        -------
        float
            norm.
        """
        system = self._system
        system._apply_nonlinear()
        return system._residuals.get_norm()

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
        if not self._ln_solver_from_parent:
            self.ln_solver._linearize()

        if self.linesearch is not None:
            self.linesearch._linearize()

    def _iter_execute(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system

        # Hybrid newton support.
        if self.options['solve_subsystems'] and self._iter_count <= self.options['max_sub_solves']:
            for isub, subsys in enumerate(system._subsystems_allprocs):
                system._transfer('nonlinear', 'fwd', isub)

                if subsys in system._subsystems_myproc:
                    subsys._solve_nonlinear()
            system._apply_nonlinear()

        system._vectors['residual']['linear'].set_vec(system._residuals)
        system._vectors['residual']['linear'] *= -1.0
        system._linearize()

        self.ln_solver.solve(['linear'], 'fwd')
        if self.linesearch:
            self.linesearch.solve()
        else:
            system._outputs += system._vectors['output']['linear']
