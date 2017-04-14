"""Define the base Solver, NonlinearSolver, and LinearSolver classes."""

from __future__ import division, print_function
import numpy as np

from openmdao.utils.generalized_dict import OptionsDictionary
from openmdao.jacobians.assembled_jacobian import AssembledJacobian


class Solver(object):
    """
    Base solver class.

    This class is subclassed by NonlinearSolver and LinearSolver,
    which are in turn subclassed by actual solver implementations.

    Attributes
    ----------
    _system : <System>
        pointer to the owning system.
    _depth : int
        how many subsolvers deep this solver is (0 means not a subsolver).
    _vec_names : [str, ...]
        list of right-hand-side (RHS) vector names.
    _mode : str
        'fwd' or 'rev', applicable to linear solvers only.
    _iter_count : int
        number of iterations for the current invocation of the solver.
    options : <OptionsDictionary>
        options dictionary.
    supports : <OptionsDictionary>
        options dictionary describing what features are supported by this
        solver.
    """

    SOLVER = 'base_solver'

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        self._system = None
        self._depth = 0
        self._vec_names = None
        self._mode = 'fwd'
        self._iter_count = 0

        self.options = OptionsDictionary()
        self.options.declare('maxiter', type_=int, value=10,
                             desc='maximum number of iterations')
        self.options.declare('atol', value=1e-10,
                             desc='absolute error tolerance')
        self.options.declare('rtol', value=1e-10,
                             desc='relative error tolerance')
        self.options.declare('iprint', type_=int, value=1,
                             desc='whether to print output')

        # What the solver supports.
        self.supports = OptionsDictionary()
        self.supports.declare('gradients', type_=bool, value=False)

        self._declare_options()
        self.options.update(kwargs)

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.

        This is optionally implemented by subclasses of Solver.
        """
        pass

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
        self._system = system
        self._depth = depth

    def _mpi_print(self, iteration, abs_res, rel_res):
        """
        Print residuals from an iteration.

        Parameters
        ----------
        iteration : int
            iteration counter, 0-based.
        abs_res : float
            current absolute residual norm.
        rel_res : float
            current relative residual norm.
        """
        if (self.options['iprint'] and self._system.comm.rank == 0 and
                not self._system._suppress_solver_output):
            rawname = self._system.name
            name_len = 10
            if len(rawname) > name_len:
                sys_name = rawname[:name_len]
            else:
                sys_name = rawname + ' ' * (name_len - len(rawname))

            solver_name = self.SOLVER
            name_len = 12
            if len(solver_name) > name_len:
                solver_name = solver_name[:name_len]
            else:
                solver_name = solver_name.ljust(name_len)

            depth = self._system.pathname.count('.')
            if self._system.pathname != '':
                depth += 1

            print_str = ' ' * depth + '-' * self._depth
            print_str += sys_name + solver_name
            print_str += ' %3d | %.9g %.9g' % (iteration, abs_res, rel_res)
            print(print_str)

    def _run_iterator(self):
        """
        Run the iterative solver.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        maxiter = self.options['maxiter']
        atol = self.options['atol']
        rtol = self.options['rtol']

        self._iter_count = 0
        norm0, norm = self._iter_initialize()
        self._mpi_print(self._iter_count, norm, norm / norm0)
        while self._iter_count < maxiter and \
                norm > atol and norm / norm0 > rtol:
            self._iter_execute()
            self._iter_count += 1
            norm = self._iter_get_norm()
            self._mpi_print(self._iter_count, norm, norm / norm0)
        fail = (np.isinf(norm) or np.isnan(norm) or
                (norm > atol and norm / norm0 > rtol))
        return fail, norm, norm / norm0

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
        pass

    def _iter_execute(self):
        """
        Perform the operations in the iteration loop.
        """
        pass

    def _iter_get_norm(self):
        """
        Return the norm of the residual.

        Returns
        -------
        float
            norm.
        """
        pass

    def _linearize(self):
        """
        Perform any required linearization operations such as matrix factorization.
        """
        pass

    def _linearize_children(self):
        """
        Return a flag that is True when we need to call linearize on our subsystems' solvers.t.

        Returns
        -------
        boolean
            Flag for indicating child linerization
        """
        return True

    def solve(self):
        """
        Run the solver.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
        pass

    def __str__(self):
        """
        Return a string representation of the solver.

        Returns
        -------
        str
            String representation of the solver.
        """
        return self.SOLVER


class NonlinearSolver(Solver):
    """
    Base class for nonlinear solvers.
    """

    def solve(self):
        """
        Run the solver.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            absolute error.
        float
            relative error.
        """
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
        if self.options['maxiter'] > 0:
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
        self._system._apply_nonlinear()
        return self._system._residuals.get_norm()


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
