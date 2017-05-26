"""Define the LNRunOnce class."""

from openmdao.solvers.ln_bgs import LinearBlockGS
from openmdao.utils.record_util import create_local_meta, update_local_meta


class LNRunOnce(LinearBlockGS):
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

        # TODO_RECORDERS - need to replace None in this with metadata from above
        metadata = self.metadata = create_local_meta(None, type(self).__name__)
        update_local_meta(metadata, (self._iter_count,))
        self._rec_mgr.record_iteration(self, metadata) # no norms

        return False, 0.0, 0.0
