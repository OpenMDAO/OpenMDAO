"""Define the scipy iterative solver class."""
from __future__ import division, print_function
import numpy
from scipy.sparse.linalg import LinearOperator, gmres

from openmdao.solvers.solver import LinearSolver


class ScipyIterativeSolver(LinearSolver):
    """The Krylov iterative solvers in scipy.sparse.linalg."""

    SOLVER = 'LN: SCIPY'

    def __init__(self, **kwargs):
        """Declare the solver option.

        Args
        ----
        kwargs : {}
            dictionary of options set by the instantiating class/script.
        """
        print('ln_scipy __init__')
        super(ScipyIterativeSolver, self).__init__(**kwargs)
        self.options.declare('solver', typ=object, value=gmres,
                             desc='function handle for actual solver')

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
        print('---------------------------')
        print('ln_scipy _mat_vec')
        vec_name = self._vec_name
        system = self._system
        ind1, ind2 = system._variable_allprocs_range['output']

        if self._mode == 'fwd':
            x_vec = system._vectors['output'][vec_name]
            b_vec = system._vectors['residual'][vec_name]
        elif self._mode == 'rev':
            x_vec = system._vectors['residual'][vec_name]
            b_vec = system._vectors['output'][vec_name]

        x_vec.set_data(in_vec)
        var_inds = [
            system._variable_allprocs_range['output'][0],
            system._variable_allprocs_range['output'][1],
            system._variable_allprocs_range['output'][0],
            system._variable_allprocs_range['output'][1],
        ]
        system._apply_linear([vec_name], self._mode, var_inds)
        print('ln_scipy _mat_vec result:\n', b_vec.get_data())
        print('---------------------------')
        return b_vec.get_data()

    def _monitor(self, res):
        """Print the residual and iteration number (callback from SciPy).

        Args
        ----
        res : ndarray
            the current residual vector.
        """
        print('---------------------------')
        norm = numpy.linalg.norm(res)
        print('ln_scipy _monitor, counter =', self._counter, 'norm:', norm)
        if self._counter == 0:
            if norm != 0.0:
                self._norm0 = norm
            else:
                self._norm0 = 1.0
        self._mpi_print(self._counter, norm / self._norm0, norm)
        self._counter += 1
        print('---------------------------')

    def __call__(self, vec_names, mode):
        """See LinearSolver."""
        print('ln_scipy __call__', vec_names, mode)
        self._vec_names = vec_names
        self._mode = mode

        system = self._system
        solver = self.options['solver']

        maxiter = self.options['maxiter']
        atol = self.options['atol']
        rtol = self.options['rtol']

        print('ln_scipy __call__ vec_names=', self._vec_names)

        for vec_name in self._vec_names:
            print('ln_scipy __call__ path_name=', system.path_name, "vec_name =", vec_name)
            self._vec_name = vec_name

            if self._mode == 'fwd':
                x_vec = system._vectors['output'][vec_name]
                b_vec = system._vectors['residual'][vec_name]
            elif self._mode == 'rev':
                x_vec = system._vectors['residual'][vec_name]
                b_vec = system._vectors['output'][vec_name]

            from pprint import pprint
            print('x_vec:')
            pprint(x_vec.get_data())
            print('b_vec:')
            pprint(b_vec.get_data())

            x_vec_combined = x_vec.get_data()
            size = x_vec_combined.size
            linop = LinearOperator((size, size), dtype=float,
                                   matvec=self._mat_vec)
            self._counter = 0
            result = solver(linop, b_vec.get_data(),
                            x0=x_vec_combined, maxiter=maxiter, tol=atol,
                            callback=self._monitor)
            print('result:')
            pprint(result)
            x_vec.set_data(result[0])
