"""Define the LinearBlockGS class."""

from six.moves import range

from openmdao.solvers.solver import BlockLinearSolver


class LinearBlockGS(BlockLinearSolver):
    """
    Linear block Gauss-Seidel solver.
    """

    SOLVER = 'LN: LNBGS'

    def _iter_execute(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system
        mode = self._mode
        vec_names = self._vec_names

        if mode == 'fwd':
            for ind, subsys in enumerate(system._subsystems_myproc):
                isub = system._subsystems_myproc_inds[ind]
                for vec_name in vec_names:
                    system._transfer(vec_name, mode, isub)
                scope_out, scope_in = system._get_scope(subsys)
                subsys._apply_linear(vec_names, mode, scope_out, scope_in)
                for vec_name in vec_names:
                    b_vec = system._vectors['residual'][vec_name]
                    b_vec *= -1.0
                    b_vec += self._rhs_vecs[vec_name]
                subsys._solve_linear(vec_names, mode)

        elif mode == 'rev':
            subsystems = system._subsystems_allprocs
            subinds = system._subsystems_myproc_inds
            for revidx in range(len(system._subsystems_myproc) - 1, -1, -1):
                isub = subinds[revidx]
                subsys = subsystems[isub]
                for vec_name in vec_names:
                    b_vec = system._vectors['output'][vec_name]
                    b_vec.set_const(0.0)
                    system._transfer(vec_name, mode, isub)
                    b_vec *= -1.0
                    b_vec += self._rhs_vecs[vec_name]
                subsys._solve_linear(vec_names, mode)
                scope_out, scope_in = system._get_scope(subsys)
                subsys._apply_linear(vec_names, mode, scope_out, scope_in)
