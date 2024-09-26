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

        subs = list(system._relevance.filter(system._subsystems_myproc))
        scopelist = [None] * len(subs)

        if mode == 'fwd':
            system._transfer('linear', mode)

            for i, subsys in enumerate(subs):
                scope_out, scope_in = system._get_matvec_scope(subsys)
                scope_out = self._vars_union(self._scope_out, scope_out)
                scope_in = self._vars_union(self._scope_in, scope_in)
                scopelist[i] = (scope_out, scope_in)
                if subsys._iter_call_apply_linear():
                    subsys._apply_linear(None, mode, scope_out, scope_in)
                else:
                    subsys._dresiduals.set_val(0.0)

            system._dresiduals *= -1.0
            system._dresiduals += self._rhs_vec

            for i, subsys in enumerate(subs):
                scope_out, scope_in = scopelist[i]
                subsys._solve_linear(mode, scope_out, scope_in)

        else:  # rev
            for i, subsys in enumerate(subs):
                scope_out, scope_in = system._get_matvec_scope(subsys)
                scope_out = self._vars_union(self._scope_out, scope_out)
                scope_in = self._vars_union(self._scope_in, scope_in)
                scopelist[i] = (scope_out, scope_in)
                if subsys._iter_call_apply_linear():
                    subsys._apply_linear(None, mode, scope_out, scope_in)
                else:
                    subsys._doutputs.set_val(0.0)

            system._transfer('linear', mode)

            system._doutputs *= -1.0
            system._doutputs += self._rhs_vec

            for i, subsys in enumerate(subs):
                scope_out, scope_in = scopelist[i]
                subsys._solve_linear(mode, scope_out, scope_in)
