"""Define the LinearBlockJac class."""
from solver import LinearSolver


class LinearBlockJac(LinearSolver):
    """Linear block Jacobi solver."""

    METHOD = 'LN: LNBJ'

    def _iter_execute(self):
        """See openmdao.solvers.solver.Solver."""
        system = self._system
        mode = self._mode
        vec_names = self._vec_names

        if mode == 'fwd':
            for vec_name in vec_names:
                d_inputs = system._vectors['input'][vec_name]
                d_outputs = system._vectors['output'][vec_name]
                system._transfers[None](d_inputs, d_outputs, mode)
            for subsys in system._subsystems_myproc:
                IDs = [ID for ID in system._variable_myproc_indices['output']
                       if ID not in subsys._variable_myproc_indices['output']]
                subsys._apply_linear(vec_names, mode, IDs)
            for vec_name in vec_names:
                b_vec = system._vectors['residual'][vec_name]
                b_vec *= -1.0
                b_vec += self._rhs_vecs[vec_name]
            for subsys in system._subsystems_myproc:
                subsys._solve_linear(vec_names, mode)
        elif mode == 'rev':
            for subsys in system._subsystems_myproc:
                IDs = [ID for ID in system._variable_myproc_indices['output']
                       if ID not in subsys._variable_myproc_indices['output']]
                subsys._apply_linear(vec_names, mode, IDs)
            for vec_name in vec_names:
                d_inputs = system._vectors['input'][vec_name]
                d_outputs = system._vectors['output'][vec_name]
                system._transfers[None](d_inputs, d_outputs, mode)

                b_vec = system._vectors['output'][vec_name]
                b_vec *= -1.0
                b_vec += self._rhs_vecs[vec_name]
            for subsys in system._subsystems_myproc:
                subsys._solve_linear(vec_names, mode)
