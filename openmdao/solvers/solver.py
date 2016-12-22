"""Define the base Solver, NonlinearSolver, and LinearSolver classes."""
from __future__ import division, print_function
import numpy

from openmdao.utils.generalized_dict import GeneralizedDictionary


class Solver(object):
    """Base solver class.

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
    options : <GeneralizedDictionary>
        options dictionary.
    """

    SOLVER = 'base_solver'

    def __init__(self, **kwargs):
        """Initialize all attributes.

        Args
        ----
        **kwargs : dict
            options dictionary.
        """
        self._system = None
        self._depth = 0
        self._vec_names = None
        self._mode = 'fwd'
        self._iter_count = 0

        self.options = GeneralizedDictionary(kwargs)
        self.options.declare('maxiter', typ=int, value=10,
                             desc='maximum number of iterations')
        self.options.declare('atol', value=1e-10,
                             desc='absolute error tolerance')
        self.options.declare('rtol', value=1e-10,
                             desc='relative error tolerance')
        self.options.declare('iprint', typ=int, value=1,
                             desc='whether to print output')
        self.options.declare('subsolvers', typ=dict, value={},
                             desc='dictionary of solvers called by this one')

    def _setup_solvers(self, system, depth):
        """Assign system instance, set depth, and optionally perform setup.

        Args
        ----
        system : <System>
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        self._system = system
        self._depth = depth

        for solver in self.options['subsolvers'].values():
            solver._setup_solvers(system, depth + 1)

    def _mpi_print(self, iteration, res, res0):
        """Print residuals from an iteration.

        Args
        ----
        iteration : int
            iteration counter, 0-based.
        res : float
            current residual norm.
        res0 : float
            initial residual norm.
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

            print_str = ' ' * self._system._sys_depth + '-' * self._depth
            print_str += sys_name + solver_name
            print_str += ' %3d | %.9g %.9g' % (iteration, res, res0)
            print(print_str)

    def _run_iterator(self):
        """Run the iterative solver.

        Returns
        -------
        boolean
            True is unconverged or diverged; False is successful.
        float
            relative error at termination.
        float
            absolute error at termination.
        """
        maxiter = self.options['maxiter']
        atol = self.options['atol']
        rtol = self.options['rtol']

        norm0, norm = self._iter_initialize()
        self._iter_count = 0
        self._mpi_print(self._iter_count, norm / norm0, norm0)
        while self._iter_count < maxiter and \
                norm > atol and norm / norm0 > rtol:
            self._iter_execute()
            norm = self._iter_get_norm()
            self._iter_count += 1
            self._mpi_print(self._iter_count, norm / norm0, norm)
        fail = (numpy.isinf(norm) or numpy.isnan(norm) or
                (norm > atol and norm / norm0 > rtol))
        return fail, norm / norm0, norm

    def _iter_initialize(self):
        """Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        pass

    def _iter_execute(self):
        """Perform the operations in the iteration loop."""
        pass

    def _iter_get_norm(self):
        """Return the norm of the residual.

        Returns
        -------
        float
            norm.
        """
        pass

    def __call__(self):
        """Run the solver.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        pass

    def set_subsolver(self, name, solver):
        """Add a subsolver to this solver.

        Args
        ----
        name : str
            name of the subsolver.
        solver : <Solver>
            the subsolver instance.

        Returns
        -------
        <Solver>
            the subsolver instance.
        """
        self.options['subsolvers'][name] = solver
        self.options['subsolvers'][name]._setup_solvers(self._system,
                                                        self._depth + 1)
        return solver

    def get_subsolver(self, name):
        """Get a subsolver.

        Args
        ----
        name : str
            name of the subsolver.

        Returns
        -------
        <Solver>
            the instance of the requested subsolver.
        """
        return self.options['subsolvers'][name]


class NonlinearSolver(Solver):
    """Base class for nonlinear solvers."""

    def __call__(self):
        """Run the solver.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        return self._run_iterator()

    def _iter_initialize(self):
        """Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        if self.options['maxiter'] > 1:
            norm = self._iter_get_norm()
        else:
            norm = 1.0
        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def _iter_get_norm(self):
        """Return the norm of the residual.

        Returns
        -------
        float
            norm.
        """
        self._system._apply_nonlinear()
        return self._system._residuals.get_norm()


class LinearSolver(Solver):
    """Base class for linear solvers."""

    def __call__(self, vec_names, mode):
        """Run the solver.

        Args
        ----
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        self._vec_names = vec_names
        self._mode = mode
        return self._run_iterator()

    def _iter_initialize(self):
        """Perform any necessary pre-processing operations.

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
        """Return the norm of the residual.

        Returns
        -------
        float
            norm.
        """
        system = self._system
        var_inds = [
            system._variable_allprocs_range['output'][0],
            system._variable_allprocs_range['output'][1],
            system._variable_allprocs_range['output'][0],
            system._variable_allprocs_range['output'][1],
        ]
        system._apply_linear(self._vec_names, self._mode, var_inds)

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
