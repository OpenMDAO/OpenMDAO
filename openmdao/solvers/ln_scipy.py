"""Define the scipy iterative solver class."""
from __future__ import division, print_function
import numpy
from six.moves import range
from scipy.sparse.linalg import LinearOperator, gmres

from openmdao.solvers.solver import LinearSolver


class ScipyIterativeSolver(LinearSolver):
    """The Krylov iterative solvers in scipy.sparse.linalg."""

    SOLVER = 'LN: SCIPY'

    def __init__(self, subsolvers=None, **kwargs):
        """Declare the solver option.

        Args
        ----
        subsolvers : dict or None
            dictionary of subsolvers (nest solvers).
        kwargs : {}
            dictionary of options set by the instantiating class/script.
        """
        super(ScipyIterativeSolver, self).__init__(subsolvers=subsolvers,
                                                   **kwargs)
        self.options.declare('solver', typ=object, value=gmres)

    def _mat_vec(self, in_vec):
        """Compute matrix-vector product.

        Args
        ----
        in_vec : ndarray
            the incoming array (combines all varsets).

        Returns
        -------
        ndarray
            the outgoing array after the product (combines all varsets).
        """
        vec_name = self._vec_name
        system = self._system
        ind1, ind2 = system._variable_allprocs_range['output']

        if self._mode == 'fwd':
            x_vec = system._vectors['output'][vec_name]
            b_vec = system._vectors['residual'][vec_name]
        elif self._mode == 'rev':
            x_vec = system._vectors['residual'][vec_name]
            b_vec = system._vectors['output'][vec_name]

        x_vec._update_varset_data(in_vec)
        system._apply_linear([vec_name], self._mode, numpy.arange(ind1, ind2))
        return b_vec._combined_varset_data()

    def _monitor(self, res):
        """Print the residual and iteration number (callback from SciPy).

        Args
        ----
        res : ndarray
            the current residual vector.
        """
        norm = numpy.linalg.norm(res)
        if self._counter == 0:
            if norm != 0.0:
                self._norm0 = norm
            else:
                self._norm0 = 1.0
        self._mpi_print(self._counter, norm/self._norm0, norm)
        self._counter += 1

    def __call__(self, vec_names, mode):
        """See LinearSolver."""
        self._vec_names = vec_names
        self._mode = mode

        system = self._system
        solver = self.options['solver']

        ilimit = self.options['ilimit']
        atol = self.options['atol']
        rtol = self.options['rtol']

        for vec_name in self._vec_names:
            self._vec_name = vec_name

            if self._mode == 'fwd':
                x_vec = system._vectors['output'][vec_name]
                b_vec = system._vectors['residual'][vec_name]
            elif self._mode == 'rev':
                x_vec = system._vectors['residual'][vec_name]
                b_vec = system._vectors['output'][vec_name]

            x_vec_combined = x_vec._combined_varset_data()
            size = x_vec_combined.size
            linop = LinearOperator((size, size), dtype=float,
                                   matvec=self._mat_vec)
            self._counter = 0
            x_vec._update_varset_data(
                solver(linop, b_vec._combined_varset_data(),
                       x0=x_vec_combined, maxiter=ilimit, tol=atol,
                       callback=self._monitor)[0])
