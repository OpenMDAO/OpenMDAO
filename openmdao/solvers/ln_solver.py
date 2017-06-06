"""Define the base Solver, NonlinearSolver, and LinearSolver classes."""

from __future__ import division, print_function

from openmdao.jacobians.assembled_jacobian import AssembledJacobian
from openmdao.solvers.solver import Solver


class LinearSolver(Solver):
    """
    Base class for linear solvers.
    """

    def solve(self, vec_names, mode):
        """
        Run the solver.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            initial error.
        float
            error at the first iteration.
        """
        self._vec_names = vec_names
        self._mode = mode
        return self._run_iterator()

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        system = self._system

        self._rhs_vecs = {}
        if self._mode == 'fwd':
            b_vecs = system._vectors['residual']
        else:  # rev
            b_vecs = system._vectors['output']

        for vec_name in self._vec_names:
            self._rhs_vecs[vec_name] = b_vecs[vec_name]._clone()

        if self.options['maxiter'] > 1:
            norm = self._iter_get_norm()
        else:
            norm = 1.0
        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def _iter_get_norm(self):
        """
        Return the norm of the residual.

        Returns
        -------
        float
            norm.
        """
        system = self._system
        scope_out, scope_in = system._get_scope()
        system._apply_linear(self._vec_names, self._mode, scope_out, scope_in)

        if self._mode == 'fwd':
            b_vecs = system._vectors['residual']
        else:  # rev
            b_vecs = system._vectors['output']

        norm = 0
        for vec_name in self._vec_names:
            b_vec = b_vecs[vec_name]
            b_vec -= self._rhs_vecs[vec_name]
            norm += b_vec.get_norm()**2

        return norm ** 0.5


class BlockLinearSolver(LinearSolver):
    """
    A base class for LinearBlockGS and LinearBlockJac.
    """

    def _iter_initialize(self):
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        if isinstance(self._system._jacobian, AssembledJacobian):
            raise RuntimeError("A block linear solver '%s' is being used with "
                               "an AssembledJacobian in system '%s'" %
                               (self.SOLVER, self._system.pathname))
        return super(BlockLinearSolver, self)._iter_initialize()
