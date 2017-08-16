"""Define the LinearUserDefined class."""

from openmdao.solvers.solver import LinearSolver


class LinearUserDefined(LinearSolver):
    """
    LinearUserDefined solver.

    This is a solver that wraps a user-written linear solve function.
    """

    SOLVER = 'LN: USER'

    def __init__(self, solve_function, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        solve_function : function
            Custom function containing the solve_linear function.
        **kwargs : dict
            Options dictionary.
        """
        super(LinearUserDefined, self).__init__(**kwargs)

        self.solve_function = solve_function

    def solve(self, vec_names, mode):
        """
        Solve the linear system for the problem in self._system.

        The full solution vector is returned.

        Parameters
        ----------
        vec_names : list
            list of vector names.
        mode : string
            Derivative mode, can be 'fwd' or 'rev'.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        self._vec_names = vec_names
        self._mode = mode

        system = self._system

        for vec_name in self._vec_names:
            self._vec_name = vec_name

            d_outputs = system._vectors['output'][vec_name]
            d_resids = system._vectors['residual'][vec_name]

            self._iter_count = 0

            # run custom solver
            fail, abs_error, rel_error = self.solve_function(d_outputs, d_resids, mode)

        return fail, abs_error, rel_error
