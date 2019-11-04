"""Define the LinearBlockGS class."""
from __future__ import print_function

from six import iteritems
from six.moves import range


from openmdao.solvers.solver import BlockLinearSolver


class LinearBlockGS(BlockLinearSolver):
    """
    Linear block Gauss-Seidel solver.
    """

    SOLVER = 'LN: LNBGS'

    def _single_iteration(self):
        """
        Perform the operations in the iteration loop.
        """
        system = self._system()
        mode = self._mode
        vec_names = self._vec_names

        if mode == 'fwd':
            for isub, (subsys, local)in enumerate(system._all_subsystem_iter()):
                if self._rel_systems is not None and subsys.pathname not in self._rel_systems:
                    continue
                for vec_name in vec_names:
                    # must always do the transfer on all procs even if subsys not local
                    system._transfer(vec_name, mode, isub)

                if not local:
                    continue

                scope_out, scope_in = system._get_scope(subsys)
                subsys._apply_linear(None, vec_names, self._rel_systems, mode, scope_out, scope_in)
                for vec_name in vec_names:
                    if vec_name in subsys._rel_vec_names:
                        b_vec = system._vectors['residual'][vec_name]
                        b_vec *= -1.0
                        b_vec._data += self._rhs_vecs[vec_name]
                subsys._solve_linear(vec_names, mode, self._rel_systems)

        else:  # rev
            subsystems = system._subsystems_allprocs
            for isub in range(len(system._subsystems_allprocs) - 1, -1, -1):
                subsys = subsystems[isub]
                local = subsys.name in system._loc_subsys_map

                if self._rel_systems is not None and subsys.pathname not in self._rel_systems:
                    continue

                for vec_name in vec_names:
                    if local and vec_name in subsys._rel_vec_names:
                        b_vec = system._vectors['output'][vec_name]
                        b_vec.set_const(0.0)
                    system._transfer(vec_name, mode, isub)
                    if local and vec_name in subsys._rel_vec_names:
                        b_vec *= -1.0
                        b_vec._data += self._rhs_vecs[vec_name]

                if local:
                    subsys._solve_linear(vec_names, mode, self._rel_systems)
                    scope_out, scope_in = system._get_scope(subsys)
                    subsys._apply_linear(None, vec_names, self._rel_systems, mode,
                                         scope_out, scope_in)
