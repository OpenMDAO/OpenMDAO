"""Define the LinearRunOnce class."""
from openmdao.solvers.linear.linear_block_gs import LinearBlockGS
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.jacobians.assembled_jacobian import AssembledJacobian


class LinearRunOnce(LinearBlockGS):
    """
    Simple linear solver that performs a single iteration of Guass-Seidel.

    This is done without iteration or norm calculation.
    """

    SOLVER = 'LN: RUNONCE'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            Options dictionary.
        """
        super(LinearRunOnce, self).__init__(**kwargs)

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

        Returns
        -------
        float
            Initial error.
        float
            Error at the first iteration.
        """
        self._vec_names = vec_names
        self._mode = mode
        self._rel_systems = rel_systems
        system = self._system

        if isinstance(system._jacobian, AssembledJacobian):
            raise RuntimeError("A block linear solver '%s' is being used with "
                               "an AssembledJacobian in system '%s'" %
                               (self.SOLVER, self._system.pathname))

        # Pre-processing
        if self._mode == 'fwd':
            b_vecs = system._vectors['residual']
        else:  # rev
            b_vecs = system._vectors['output']

        for vec_name in self._vec_names:
            if vec_name in system._rel_vec_names:
                self._rhs_vecs[vec_name].set_vec(b_vecs[vec_name])

        with Recording('LinearRunOnce', 0, self) as rec:
            # Single iteration of GS
            self._iter_execute()

            rec.abs = 0.0
            rec.rel = 0.0

        return False, 0.0, 0.0

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        # Remove unused options from base options here, so that users
        # attempting to set them will get KeyErrors.
        self.options.undeclare("atol")
        self.options.undeclare("rtol")

        # this solver does not iterate
        self.options.undeclare("maxiter")
        self.options.undeclare("err_on_maxiter")
