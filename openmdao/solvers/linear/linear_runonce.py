"""Define the LinearRunOnce class."""
from openmdao.utils.record_util import create_local_meta
from openmdao.solvers.linear.linear_block_gs import LinearBlockGS


class LinearRunOnce(LinearBlockGS):
    """
    Simple linear solver that performs a single iteration of Guass-Seidel.

    This is done without iteration or norm calculation.
    """

    SOLVER = 'LN: RUNONCE'

    def solve(self, vec_names, mode):
        """
        Run the solver.

        Parameters
        ----------
        vec_names : [str, ...]
            List of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.

        Returns
        -------
        float
            Initial error.
        float
            Error at the first iteration.
        """
        self._vec_names = vec_names
        self._mode = mode
        system = self._system

        # Preprocessing
        self._rhs_vecs = {}
        if self._mode == 'fwd':
            b_vecs = system._vectors['residual']
        else:  # rev
            b_vecs = system._vectors['output']

        for vec_name in self._vec_names:
            self._rhs_vecs[vec_name] = b_vecs[vec_name]._clone()

        # Single iteration of GS
        self._iter_execute()

        from openmdao.recorders.base_recorder import recording
        with recording('LinearRunOnce', 0):
            metadata = create_local_meta('LinearRunOnce')
            self._rec_mgr.record_iteration(self, metadata)  # no norms

        return False, 0.0, 0.0
