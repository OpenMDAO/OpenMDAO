"""Define the LinearBlockJac class."""
from openmdao.solvers.solver import BlockLinearSolver


class LinearBlockJac(BlockLinearSolver):
    """
    Linear block Jacobi solver.
    """

    SOLVER = 'LN: LNBJ'

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system()
        mode = self._mode
        vec_names = self._vec_names

        subs = [s for s in system._subsystems_myproc
                if self._rel_systems is None or s.pathname in self._rel_systems]

        if mode == 'fwd':
            for vec_name in vec_names:
                system._transfer(vec_name, mode)

            for subsys in subs:
                scope_out, scope_in = system._get_scope(subsys)
                subsys._apply_linear(None, vec_names, self._rel_systems, mode,
                                     scope_out, scope_in)

            for vec_name in vec_names:
                b_vec = system._vectors['residual'][vec_name]
                b_vec *= -1.0
                b_vec._data += self._rhs_vecs[vec_name]

            for subsys in subs:
                subsys._solve_linear(vec_names, mode, self._rel_systems)

        else:  # rev
            for subsys in subs:
                scope_out, scope_in = system._get_scope(subsys)
                subsys._apply_linear(None, vec_names, self._rel_systems, mode,
                                     scope_out, scope_in)

            for vec_name in vec_names:
                system._transfer(vec_name, mode)

                b_vec = system._vectors['output'][vec_name]
                b_vec *= -1.0
                b_vec._data += self._rhs_vecs[vec_name]

            for subsys in subs:
                subsys._solve_linear(vec_names, mode, self._rel_systems)
