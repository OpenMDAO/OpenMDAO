"""Define the LinearBlockJac class."""
from openmdao.solvers.solver import BlockLinearSolver


class LinearBlockJac(BlockLinearSolver):
    """
    Linear block Jacobi solver.

    Parameters
    ----------
    **kwargs : dict
        Options dictionary.
    """

    SOLVER = 'LN: LNBJ'

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system()
        mode = self._mode

        subs = [s for s in system._subsystems_myproc
                if self._rel_systems is None or s.pathname in self._rel_systems]

        if mode == 'fwd':
            system._transfer('linear', mode)

            for subsys in subs:
                scope_out, scope_in = system._get_scope(subsys)
                subsys._apply_linear(None, self._rel_systems, mode, scope_out, scope_in)

            b_vec = system._vectors['residual']['linear']
            b_vec *= -1.0
            b_vec += self._rhs_vec

            for subsys in subs:
                subsys._solve_linear(mode, self._rel_systems)

        else:  # rev
            for subsys in subs:
                scope_out, scope_in = system._get_scope(subsys)
                subsys._apply_linear(None, self._rel_systems, mode, scope_out, scope_in)

            system._transfer('linear', mode)

            b_vec = system._vectors['output']['linear']
            b_vec *= -1.0
            b_vec += self._rhs_vec

            for subsys in subs:
                subsys._solve_linear(mode, self._rel_systems)
