"""Define the scipy iterative solver class."""

from packaging.version import Version
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator, gmres
from openmdao.solvers.linear.linear_rhs_checker import LinearRHSChecker

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

    Parameters
    ----------
    **kwargs : {}
        Dictionary of options set by the instantiating class/script.

    Attributes
    ----------
    precon : Solver
        Preconditioner for linear solve. Default is None for no preconditioner.
    _lin_rhs_checker : LinearRHSChecker or None
        Object for checking the right-hand side of the linear solve.
    """

    SOLVER = 'LN: SCIPY'

    def __init__(self, **kwargs):
        """
        Declare the solver option.
        """
        super().__init__(**kwargs)

        self.precon = None
        self._lin_rhs_checker = None

    def _assembled_jac_solver_iter(self):
        """
        Return a generator of linear solvers using assembled jacs.
        """
        if self.options['assemble_jac']:
            yield self, self.preferred_sparse_format()
        if self.precon is not None:
            for tup in self.precon._assembled_jac_solver_iter():
                yield tup

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

        self.options.declare('rhs_checking', types=(bool, dict),
                             default=False,
                             desc="If True, check RHS vs. cache and/or zero to avoid some solves."
                             "Can also be set to a dict of options for the LinearRHSChecker to "
                             "allow finer control over it. Allowed options are: "
                             f"{LinearRHSChecker.options}")

        # changing the default maxiter from the base class
        self.options['maxiter'] = 1000
        self.options['atol'] = 1.0e-12

        self.supports['implicit_components'] = True

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

        self._lin_rhs_checker = LinearRHSChecker.create(self._system(),
                                                        self.options['rhs_checking'])

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
        bool
            Flag for indicating child linerization
        """
        return (self.precon is not None) and (self.precon._linearize_children())

    def _linearize(self):
        """
        Perform any required linearization operations such as matrix factorization.
        """
        if self.precon is not None:
            self.precon._linearize()

        if self._lin_rhs_checker is not None:
            self._lin_rhs_checker.clear()

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
        system = self._system()

        if self._mode == 'fwd':
            x_vec = system._doutputs
            b_vec = system._dresiduals
        else:  # rev
            x_vec = system._dresiduals
            b_vec = system._doutputs

        x_vec.set_val(in_arr)
        scope_out, scope_in = system._get_matvec_scope()
        system._apply_linear(self._mode, scope_out, scope_in)

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

    def solve(self, mode, rel_systems=None):
        """
        Run the solver.

        Parameters
        ----------
        mode : str
            'fwd' or 'rev'.
        rel_systems : set of str
            Names of systems relevant to the current solve.  Deprecated.
        """
        self._mode = mode

        system = self._system()
        solver = _SOLVER_TYPES[self.options['solver']]
        if solver is gmres:
            restart = self.options['restart']

        maxiter = self.options['maxiter']
        atol = self.options['atol']
        rtol = self.options['rtol']

        if mode == 'fwd':
            x_vec = system._doutputs
            b_vec = system._dresiduals
        else:  # rev
            x_vec = system._dresiduals
            b_vec = system._doutputs

            if self._lin_rhs_checker is not None:
                sol_array, is_zero = self._lin_rhs_checker.get_solution(b_vec.asarray(), system)
                if is_zero:
                    x_vec.set_val(0.0)
                    return
                if sol_array is not None:
                    x_vec.set_val(sol_array)
                    return

        x_vec_combined = x_vec.asarray()
        size = x_vec_combined.size
        linop = LinearOperator((size, size), dtype=float, matvec=self._mat_vec)

        # Support a preconditioner
        if self.precon:
            M = LinearOperator((size, size), matvec=self._apply_precon, dtype=float)
        else:
            M = None

        self._iter_count = 0
        if solver is gmres:
            if Version(Version(scipy.__version__).base_version) < Version("1.12"):
                x, info = solver(linop, b_vec.asarray(True), M=M, restart=restart,
                                 x0=x_vec_combined, maxiter=maxiter, tol=atol, atol='legacy',
                                 callback=self._monitor, callback_type='legacy')
            else:
                x, info = solver(linop, b_vec.asarray(True), M=M, restart=restart,
                                 x0=x_vec_combined, maxiter=maxiter, atol=atol, rtol=rtol,
                                 callback=self._monitor, callback_type='legacy')
        else:
            x, info = solver(linop, b_vec.asarray(True), M=M,
                             x0=x_vec_combined, maxiter=maxiter, tol=atol, atol='legacy',
                             callback=self._monitor, callback_type='legacy')

        if info == 0:
            x_vec.set_val(x)
        elif info > 0:
            self._convergence_failure()
        else:
            msg = (f"Solver '{self.SOLVER}' on system '{self._system().pathname}': "
                   f"had an illegal input or breakdown (info={info}) after {self._iter_count} "
                   "iterations.")
            self.report_failure(msg)

        if not system.under_complex_step and self._lin_rhs_checker is not None and mode == 'rev':
            self._lin_rhs_checker.add_solution(b_vec.asarray(), x, copy=True)

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
        mode = self._mode

        # Need to clear out any junk from the inputs.
        system._dinputs.set_val(0.0)

        # assign x and b vectors based on mode
        if mode == 'fwd':
            x_vec = system._doutputs
            b_vec = system._dresiduals
        else:  # rev
            x_vec = system._dresiduals
            b_vec = system._doutputs

        # set value of b vector to KSP provided value
        b_vec.set_val(in_vec)

        # call the preconditioner
        self._solver_info.append_precon()
        self.precon.solve(mode)
        self._solver_info.pop()

        # return resulting value of x vector
        return x_vec.asarray(copy=True)

    def use_relevance(self):
        """
        Return True if relevance should be active.

        Returns
        -------
        bool
            True if relevance should be active.
        """
        return True

    def preferred_sparse_format(self):
        """
        Return the preferred sparse format for the dr/do matrix of a split jacobian.

        Returns
        -------
        str
            The preferred sparse format for the dr/do matrix of a split jacobian.
        """
        return 'csr'
