"""Define the LinearBlockJac class."""
from openmdao.solvers.solver import LinearSolver


class LinearBlockGS(LinearSolver):
    """Linear block Gauss-Seidel solver."""

    SOLVER = 'LN: LNBGS'

    def _iter_execute(self):
        """Perform the operations in the iteration loop."""
        system = self._system
        mode = self._mode
        vec_names = self._vec_names

        if mode == 'fwd':
            for ind, subsys in enumerate(system._subsystems_myproc):
                isub = system._subsystems_inds[ind]
                for vec_name in vec_names:
                    d_inputs = system._vectors['input'][vec_name]
                    d_outputs = system._vectors['output'][vec_name]
                    system._vector_transfers[vec_name][mode, isub](
                        d_inputs, d_outputs, mode)
                var_inds = [
                    system._variable_allprocs_range['output'][0],
                    subsys._variable_allprocs_range['output'][0],
                    subsys._variable_allprocs_range['output'][1],
                    system._variable_allprocs_range['output'][1],
                ]
                subsys._apply_linear(vec_names, mode, var_inds)
                for vec_name in vec_names:
                    b_vec = system._vectors['residual'][vec_name]
                    b_vec *= -1.0
                    b_vec += self._rhs_vecs[vec_name]
                subsys._solve_linear(vec_names, mode)
        elif mode == 'rev':
            system._subsystems_myproc.reverse()
            system._subsystems_inds.reverse()
            for ind, subsys in enumerate(system._subsystems_myproc):
                isub = system._subsystems_inds[ind]
                for vec_name in vec_names:
                    d_inputs = system._vectors['input'][vec_name]
                    d_outputs = system._vectors['output'][vec_name]
                    b_vec = system._vectors['output'][vec_name]
                    b_vec.set_const(0.0)
                    system._vector_transfers[vec_name][mode, isub](
                        d_inputs, d_outputs, mode)
                    b_vec *= -1.0
                    b_vec += self._rhs_vecs[vec_name]
                subsys._solve_linear(vec_names, mode)
                var_inds = [
                    system._variable_allprocs_range['output'][0],
                    subsys._variable_allprocs_range['output'][0],
                    subsys._variable_allprocs_range['output'][1],
                    system._variable_allprocs_range['output'][1],
                ]
                subsys._apply_linear(vec_names, mode, var_inds)
            system._subsystems_myproc.reverse()
            system._subsystems_inds.reverse()
