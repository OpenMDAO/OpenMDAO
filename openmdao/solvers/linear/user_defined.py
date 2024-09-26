"""Define the LinearUserDefined class."""

from openmdao.solvers.solver import LinearSolver


class LinearUserDefined(LinearSolver):
    """
    LinearUserDefined solver.

    This is a solver that wraps a user-written linear solve function.

    Parameters
    ----------
    solve_function : function
        Custom function containing the solve_linear function. The default is None, which means
        the name defaults to "solve_linear".
    **kwargs : dict
        Options dictionary.

    Attributes
    ----------
    solve_function : function
        Custom function containing the solve_linear function. The default is None, which means
        the name defaults to "solve_linear".
    """

    SOLVER = 'LN: USER'

    def __init__(self, solve_function=None, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(**kwargs)

        self.solve_function = solve_function

    def solve(self, mode, rel_systems=None):
        """
        Solve the linear system for the problem in self._system().

        The full solution vector is returned.

        Parameters
        ----------
        mode : str
            Derivative mode, can be 'fwd' or 'rev'.
        rel_systems : set of str
            Set of names of relevant systems based on the current linear solve.  Deprecated.
        """
        self._mode = mode

        system = self._system()
        solve = self.solve_function

        if solve is None:
            solve = system.solve_linear

        d_outputs = system._doutputs
        d_resids = system._dresiduals

        self._iter_count = 0

        # run custom solver
        with system._unscaled_context(outputs=[d_outputs], residuals=[d_resids]):
            solve(d_outputs, d_resids, mode)
