"""Define the scipy iterative solver class."""

from distutils.version import LooseVersion
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator, gmres

from openmdao.solvers.solver import LinearSolver

_SOLVER_TYPES = {
    # 'bicg': bicg,
    # 'bicgstab': bicgstab,
    # 'cg': cg,
    # 'cgs': cgs,
    'gmres': gmres,
}


class ScipyKrylov(LinearSolver):
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
        super().__init__(**kwargs)

        # initialize preconditioner to None
        self.precon = None

    def _assembled_jac_solver_iter(self):
        """
        Return a generator of linear solvers using assembled jacs.
        """
        if self.options['assemble_jac']:
            yield self
        if self.precon is not None:
            for s in self.precon._assembled_jac_solver_iter():
                yield s

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        super()._declare_options()

        self.options.declare('solver', default='gmres', values=tuple(_SOLVER_TYPES.keys()),
                             desc='function handle for actual solver')

        self.options.declare('restart', default=20, types=int,
                             desc='Number of iterations between restarts. Larger values increase '
                                  'iteration cost, but may be necessary for convergence. This '
                                  'option applies only to gmres.')

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
        super()._setup_solvers(system, depth)

        if self.precon is not None:
            self.precon._setup_solvers(self._system(), self._depth + 1)

    def _set_solver_print(self, level=2, type_='all'):
        """
        Control printing for solvers and subsolvers in the model.

        Parameters
        ----------
        level : int
            iprint level. Set to 2 to print residuals each iteration; set to 1
            to print just the iteration totals; set to 0 to disable all printing
            except for failures, and set to -1 to disable all printing including failures.
        type_ : str
            Type of solver to set: 'LN' for linear, 'NL' for nonlinear, or 'all' for all.
        """
        super()._set_solver_print(level=level, type_=type_)

        if self.precon is not None and type_ != 'NL':
            self.precon._set_solver_print(level=level, type_=type_)

    def _linearize_children(self):
        """
        Return a flag that is True when we need to call linearize on our subsystems' solvers.

        Returns
        -------
        boolean
            Flag for indicating child linerization
        """
        precon = self.precon
        return (precon is not None) and (precon._linearize_children())

    def _linearize(self):
        """
        Perform any required linearization operations such as matrix factorization.
        """
        if self.precon is not None:
            self.precon._linearize()

    def _mat_vec(self, in_arr):
        """
        Compute matrix-vector product.

        Parameters
        ----------
        in_arr : ndarray
            the incoming array.

        Returns
        -------
        ndarray
            the outgoing array after the product.
        """
        vec_name = self._vec_name
        system = self._system()

        if self._mode == 'fwd':
            x_vec = system._vectors['output'][vec_name]
            b_vec = system._vectors['residual'][vec_name]
        else:  # rev
            x_vec = system._vectors['residual'][vec_name]
            b_vec = system._vectors['output'][vec_name]

        x_vec.set_val(in_arr)
        scope_out, scope_in = system._get_scope()
        system._apply_linear(self._assembled_jac, [vec_name], self._rel_systems, self._mode,
                             scope_out, scope_in)

        # DO NOT REMOVE: frequently used for debugging
        # print('in', in_arr)
        # print('out', b_vec.asarray())

        return b_vec.asarray()

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

    def solve(self, vec_names, mode, rel_systems=None):
        """
        Run the solver.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Names of systems relevant to the current solve.
        """
        self._vec_names = vec_names
        self._rel_systems = rel_systems
        self._mode = mode

        system = self._system()
        solver = _SOLVER_TYPES[self.options['solver']]
        if solver is gmres:
            restart = self.options['restart']

        maxiter = self.options['maxiter']
        atol = self.options['atol']

        fail = False

        for vec_name in self._vec_names:

            self._vec_name = vec_name

            if self._mode == 'fwd':
                x_vec = system._vectors['output'][vec_name]
                b_vec = system._vectors['residual'][vec_name]
            else:  # rev
                x_vec = system._vectors['residual'][vec_name]
                b_vec = system._vectors['output'][vec_name]

            x_vec_combined = x_vec.asarray()
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
            if solver is gmres:
                if LooseVersion(scipy.__version__) < LooseVersion("1.1"):
                    x, info = solver(linop, b_vec.asarray(True), M=M, restart=restart,
                                     x0=x_vec_combined, maxiter=maxiter, tol=atol,
                                     callback=self._monitor)
                else:
                    x, info = solver(linop, b_vec.asarray(True), M=M, restart=restart,
                                     x0=x_vec_combined, maxiter=maxiter, tol=atol, atol='legacy',
                                     callback=self._monitor)
            else:
                x, info = solver(linop, b_vec.asarray(True), M=M,
                                 x0=x_vec_combined, maxiter=maxiter, tol=atol,
                                 callback=self._monitor)

            fail |= (info != 0)
            x_vec.set_val(x)

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
        system = self._system()
        vec_name = self._vec_name
        mode = self._mode

        # Need to clear out any junk from the inputs.
        system._vectors['input'][vec_name].set_val(0.0)

        # assign x and b vectors based on mode
        if mode == 'fwd':
            x_vec = system._vectors['output'][vec_name]
            b_vec = system._vectors['residual'][vec_name]
        else:  # rev
            x_vec = system._vectors['residual'][vec_name]
            b_vec = system._vectors['output'][vec_name]

        # set value of b vector to KSP provided value
        b_vec.set_val(in_vec)

        # call the preconditioner
        self._solver_info.append_precon()
        self.precon.solve([vec_name], mode)
        self._solver_info.pop()

        # return resulting value of x vector
        return x_vec.asarray(copy=True)
