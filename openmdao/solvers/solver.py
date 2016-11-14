"""Define the base Solver, NonlinearSolver, and LinearSolver classes."""
from __future__ import division, print_function
import numpy
from six.moves import range
from scipy.sparse.linalg import LinearOperator

from openmdao.utils.generalized_dict import GeneralizedDictionary


class Solver(object):
    """Base solver class.

    This class is subclassed by NonlinearSolver and LinearSolver,
    which are in turn subclassed by actual solver implementations.

    Attributes
    ----------
    _system : System
        pointer to the owning system.
    _depth : int
        how many subsolvers deep this solver is (0 means not a subsolver).
    _vec_names : [str, ...]
        list of right-hand-side (RHS) vector names.
    _mode : str
        'fwd' or 'rev', applicable to linear solvers only.
    _subsolvers : dict
        dictionary of pointers to subsolvers.
    options : GeneralizedDictionary
        options dictionary.
    """

    SOLVER = 'base_solver'

    def __init__(self, subsolvers=None, **kwargs):
        """Initialize all attributes.

        Args
        ----
        subsolvers : dict
            dictionary of pointers to subsolvers.
        **kwargs : dict
            options dictionary.
        """
        self._system = None
        self._depth = 0
        self._vec_names = None
        self._mode = 'fwd'
        self._subsolvers = subsolvers if subsolvers is not None else {}

        self.options = GeneralizedDictionary(kwargs)
        self.options.declare('ilimit', typ=int, value=10)
        self.options.declare('atol', value=1e-6)
        self.options.declare('rtol', value=1e-6)
        self.options.declare('iprint', typ=int, value=1)

    def _setup_solvers(self, system, depth):
        """Assign system instance, set depth, and optionally perform setup.

        Args
        ----
        system : System
            pointer to the owning system.
        depth : int
            depth of the current system (already incremented).
        """
        self._system = system
        self._depth = depth

        for solver in self._subsolvers.values():
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
            solver_name = solver_name + ' ' * (name_len - len(solver_name))

        iproc = self._system.comm.rank
        iprint = self.options['iprint']
        suppress_solver_output = self._system._suppress_solver_output
        if iproc == 0 and iprint and not suppress_solver_output:
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
        ilimit = self.options['ilimit']
        atol = self.options['atol']
        rtol = self.options['rtol']

        norm0, norm = self._iter_initialize()
        iteration = 0
        self._mpi_print(iteration, norm / norm0, norm0)
        while iteration < ilimit and norm > atol and norm / norm0 > rtol:
            self._iter_execute()
            norm = self._iter_get_norm()
            iteration += 1
            self._mpi_print(iteration, norm / norm0, norm)
        success = not(norm > atol and norm / norm0 > rtol)
        success = success and (not numpy.isinf(norm))
        success = success and (not numpy.isnan(norm))
        return not success, norm / norm0, norm

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
        solver : Solver
            the subsolver instance.
        """
        self._subsolvers[name] = solver
        self._subsolvers[name]._setup_solvers(self._system, self._depth + 1)

    def get_subsolver(self, name):
        """Get a subsolver.

        Args
        ----
        name : str
            name of the subsolver.

        Returns
        -------
        Solver
            the instance of the requested subsolver.
        """
        return self._subsolvers[name]


class NonlinearSolver(Solver):
    """Base class for nonlinear solvers."""

    def __call__(self):
        """See openmdao.solvers.solver.Solver."""
        return self._run_iterator()

    def _iter_initialize(self):
        """See openmdao.solvers.solver.Solver."""
        if self.options['ilimit'] > 1:
            norm = self._iter_get_norm()
        else:
            norm = 1.0
        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def _iter_get_norm(self):
        """See openmdao.solvers.solver.Solver."""
        self._system._apply_nonlinear()
        return self._system._residuals.get_norm()


class LinearSolver(Solver):
    """Base class for linear solvers."""

    def __call__(self, vec_names, mode):
        """See openmdao.solvers.solver.Solver."""
        self._vec_names = vec_names
        self._mode = mode
        return self._run_iterator()

    def _iter_initialize(self):
        """See openmdao.solvers.solver.Solver."""
        system = self._system

        self._rhs_vecs = {}
        for vec_name in self._vec_names:
            if self._mode == 'fwd':
                b_vec = system._vectors['residual'][vec_name]
            elif self._mode == 'rev':
                b_vec = system._vectors['output'][vec_name]

            self._rhs_vecs[vec_name] = b_vec._clone()

        if self.options['ilimit'] > 1:
            norm = self._iter_get_norm()
        else:
            norm = 1.0
        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def _iter_get_norm(self):
        """See openmdao.solvers.solver.Solver."""
        system = self._system
        var_inds = [
            system._variable_allprocs_range['output'][0],
            system._variable_allprocs_range['output'][1],
            system._variable_allprocs_range['output'][0],
            system._variable_allprocs_range['output'][1],
        ]
        system._apply_linear(self._vec_names, self._mode, var_inds)

        norm = 0
        for vec_name in self._vec_names:
            if self._mode == 'fwd':
                b_vec = system._vectors['residual'][vec_name]
            elif self._mode == 'rev':
                b_vec = system._vectors['output'][vec_name]

            b_vec -= self._rhs_vecs[vec_name]
            norm += b_vec.get_norm()**2

        return norm ** 0.5
