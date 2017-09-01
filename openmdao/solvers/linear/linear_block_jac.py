"""Define the LinearBlockJac class."""
from openmdao.solvers.solver import BlockLinearSolver


class LinearBlockJac(BlockLinearSolver):
    """
    Linear block Jacobi solver.
    """

    SOLVER = 'LN: LNBJ'

    def _iter_execute(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system
        mode = self._mode
        vec_names = self._vec_names

        if mode == 'fwd':
            for vec_name in vec_names:
                system._transfer(vec_name, mode)

            for subsys in system._subsystems_myproc:
                if self._rel_systems is None or subsys.pathname in self._rel_systems:
                    scope_out, scope_in = system._get_scope(subsys)
                    sub_vec_names = [v for v in vec_names if v in subsys._lin_rel_vec_name_list]
                    subsys._apply_linear(sub_vec_names, self._rel_systems, mode,
                                         scope_out, scope_in)
            for vec_name in vec_names:
                b_vec = system._vectors['residual'][vec_name]
                b_vec *= -1.0
                b_vec += self._rhs_vecs[vec_name]

            for subsys in system._subsystems_myproc:
                if self._rel_systems is None or subsys.pathname in self._rel_systems:
                    subsys._solve_linear(sub_vec_names, mode, self._rel_systems)

        else:  # rev
            for subsys in system._subsystems_myproc:
                if self._rel_systems is None or subsys.pathname in self._rel_systems:
                    scope_out, scope_in = system._get_scope(subsys)
                    sub_vec_names = [v for v in vec_names if v in subsys._lin_rel_vec_name_list]
                    subsys._apply_linear(sub_vec_names, self._rel_systems, mode,
                                         scope_out, scope_in)
            for vec_name in vec_names:
                system._transfer(vec_name, mode)

                b_vec = system._vectors['output'][vec_name]
                b_vec *= -1.0
                b_vec += self._rhs_vecs[vec_name]

            for subsys in system._subsystems_myproc:
                if self._rel_systems is None or subsys.pathname in self._rel_systems:
                    subsys._solve_linear(sub_vec_names, mode, self._rel_systems)
