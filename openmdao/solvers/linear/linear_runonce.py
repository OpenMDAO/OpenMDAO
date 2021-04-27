"""Define the LinearRunOnce class."""

from openmdao.solvers.linear.linear_block_gs import LinearBlockGS


class LinearRunOnce(LinearBlockGS):
    """
    Simple linear solver that performs a single iteration of Guass-Seidel.

    This is done without iteration or norm calculation.
    """

    SOLVER = 'LN: RUNONCE'

    def solve(self, vec_names, mode, rel_systems=None):
        """
        Run the solver.

        Parameters
        ----------
        vec_names : [str, ...]
            List of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Names of systems relevant to the current solve.
        """
        self._vec_names = vec_names
        self._mode = mode
        self._rel_systems = rel_systems

        self._update_rhs_vecs()

        # Single iteration of GS
        self._single_iteration()

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        # Remove unused options from base options here, so that users
        # attempting to set them will get KeyErrors.
        self.options.undeclare("atol")
        self.options.undeclare("rtol")

        # this solver does not iterate
        self.options.undeclare("maxiter")
        self.options.undeclare("err_on_non_converge")
