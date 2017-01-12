"""Define the LinearBlockJac class."""
from openmdao.solvers.solver import LinearSolver


class LinearBlockJac(LinearSolver):
    """Linear block Jacobi solver."""

    SOLVER = 'LN: LNBJ'

    def _iter_execute(self):
        """Perform the operations in the iteration loop."""
        system = self._system
        mode = self._mode
        vec_names = self._vec_names

        if mode == 'fwd':
            for vec_name in vec_names:
                d_inputs = system._vectors['input'][vec_name]
                d_outputs = system._vectors['output'][vec_name]
                system._vector_transfers[vec_name]['nonlinear'](
                    d_inputs, d_outputs, mode)
            for subsys in system._subsystems_myproc:
                var_inds = [
                    system._var_allprocs_range['output'][0],
                    subsys._var_allprocs_range['output'][0],
                    subsys._var_allprocs_range['output'][1],
                    system._var_allprocs_range['output'][1],
                ]
                subsys._apply_linear(vec_names, mode, var_inds)
            for vec_name in vec_names:
                b_vec = system._vectors['residual'][vec_name]
                b_vec *= -1.0
                b_vec += self._rhs_vecs[vec_name]
            for subsys in system._subsystems_myproc:
                subsys._solve_linear(vec_names, mode)
        elif mode == 'rev':
            for subsys in system._subsystems_myproc:
                var_inds = [
                    system._var_allprocs_range['output'][0],
                    subsys._var_allprocs_range['output'][0],
                    subsys._var_allprocs_range['output'][1],
                    system._var_allprocs_range['output'][1],
                ]
                subsys._apply_linear(vec_names, mode, var_inds)
            for vec_name in vec_names:
                d_inputs = system._vectors['input'][vec_name]
                d_outputs = system._vectors['output'][vec_name]
                system._vector_transfers[vec_name]['nonlinear'](
                    d_inputs, d_outputs, mode)

                b_vec = system._vectors['output'][vec_name]
                b_vec *= -1.0
                b_vec += self._rhs_vecs[vec_name]
            for subsys in system._subsystems_myproc:
                subsys._solve_linear(vec_names, mode)
