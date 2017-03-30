"""Define the scipy iterative solver class."""

from __future__ import division, print_function

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from openmdao.solvers.solver import LinearSolver


class ScipyIterativeSolver(LinearSolver):
    """
    The Krylov iterative solvers in scipy.sparse.linalg.

    Attributes
    ----------
    precon : Solver
        Preconditioner for linear solve. Default is None for no preconditioner.
    """

    SOLVER = 'LN: SCIPY'

    def __init__(self, **kwargs):
        """
        Declare the solver option.

        Parameters
        ----------
        **kwargs : {}
            dictionary of options set by the instantiating class/script.
        """
        super(ScipyIterativeSolver, self).__init__(**kwargs)

        # initialize preconditioner to None
        self.precon = None

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('solver', type_=object, value=gmres,
                             desc='function handle for actual solver')

        self.options.declare('restart', value=20, type_=int,
                             desc='Number of iterations between restarts. Larger values increase '
                                  'iteration cost, but may be necessary for convergence')

        # changing the default maxiter from the base class
        self.options['maxiter'] = 1000
        self.options['atol'] = 1.0e-12

    def _setup_solvers(self, system, depth):
        """
        Assign system instance, set depth, and optionally perform setup.

        Parameters
        ----------
        system : <System>
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        super(ScipyIterativeSolver, self)._setup_solvers(system, depth)

        if self.precon is not None:
            self.precon._setup_solvers(self._system, self._depth + 1)

    def _need_child_linearize(self):
        """
        Return a flag indicating if you would like your child solvers to get a linearization or not.

        Returns
        -------
        boolean
            flag for indicating child linerization
        """
        if self.precon is not None:
            return self.precon._need_child_linearize()
        return False

    def _linearize(self):
        """
        Perform any required linearization operations such as matrix factorization.
        """
        if self.precon is not None:
            self.precon._linearize()

    def _mat_vec(self, in_vec):
        """
        Compute matrix-vector product.

        Parameters
        ----------
        in_vec : ndarray
            the incoming array (combines all varsets).

        Returns
        -------
        ndarray
            the outgoing array after the product (combines all varsets).
        """
        vec_name = self._vec_name
        system = self._system
        ind1, ind2 = system._var_allprocs_idx_range['output']

        if self._mode == 'fwd':
            x_vec = system._vectors['output'][vec_name]
            b_vec = system._vectors['residual'][vec_name]
        elif self._mode == 'rev':
            x_vec = system._vectors['residual'][vec_name]
            b_vec = system._vectors['output'][vec_name]

        x_vec.set_data(in_vec)
        var_inds = [
            system._var_allprocs_idx_range['output'][0],
            system._var_allprocs_idx_range['output'][1],
            system._var_allprocs_idx_range['output'][0],
            system._var_allprocs_idx_range['output'][1],
        ]
        system._apply_linear([vec_name], self._mode, var_inds)

        # self._mpi_print(b_vec.get_data())
        return b_vec.get_data()

    def _monitor(self, res):
        """
        Print the residual and iteration number (callback from SciPy).

        Parameters
        ----------
        res : ndarray
            the current residual vector.
        """
        norm = np.linalg.norm(res)
        if self._iter_count == 0:
            if norm != 0.0:
                self._norm0 = norm
            else:
                self._norm0 = 1.0
        self._mpi_print(self._iter_count, norm, norm / self._norm0)
        self._iter_count += 1

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
            absolute error.
        float
            relative error.
        """
        self._vec_names = vec_names
        self._mode = mode

        system = self._system
        solver = self.options['solver']

        maxiter = self.options['maxiter']
        atol = self.options['atol']
        rtol = self.options['rtol']
        restart = self.options['restart']

        for vec_name in self._vec_names:
            self._vec_name = vec_name

            if self._mode == 'fwd':
                x_vec = system._vectors['output'][vec_name]
                b_vec = system._vectors['residual'][vec_name]
            elif self._mode == 'rev':
                x_vec = system._vectors['residual'][vec_name]
                b_vec = system._vectors['output'][vec_name]

            x_vec_combined = x_vec.get_data()
            size = x_vec_combined.size
            linop = LinearOperator((size, size), dtype=float,
                                   matvec=self._mat_vec)

            # Support a preconditioner
            if self.precon:
                M = LinearOperator((size, size),
                                   matvec=self._apply_precon,
                                   dtype=float)
            else:
                M = None

            self._iter_count = 0
            x_vec.set_data(
                solver(linop, b_vec.get_data(), M=M, restart=restart,
                       x0=x_vec_combined, maxiter=maxiter, tol=atol,
                       callback=self._monitor)[0])

        # TODO: implement this properly

        return False, 0., 0.

    def _apply_precon(self, in_vec):
        """
        Apply preconditioner.

        Parameters
        ----------
        in_vec : ndarray
            Incoming vector.

        Returns
        -------
        ndarray
            The preconditioned Vector.
        """
        system = self._system
        vec_name = self._vec_name
        mode = self._mode

        # Need to clear out any junk from the inputs.
        system._vectors['input'][vec_name].set_const(0.0)

        # assign x and b vectors based on mode
        if mode == 'fwd':
            x_vec = system._vectors['output'][vec_name]
            b_vec = system._vectors['residual'][vec_name]
        elif mode == 'rev':
            x_vec = system._vectors['residual'][vec_name]
            b_vec = system._vectors['output'][vec_name]

        # set value of b vector to KSP provided value
        b_vec.set_data(in_vec)

        # call the preconditioner
        self.precon.solve([vec_name], mode)

        # return resulting value of x vector
        return x_vec.get_data()
