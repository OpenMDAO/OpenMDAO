from __future__ import division, print_function
import numpy
from scipy.sparse.linalg import LinearOperator
from six.moves import range


class Solver(object):

    SOLVER = 'base_solver'

    def __init__(self, options={}, subsolvers={}, **kwargs):
        self._system = None
        self._depth = 0
        self._options = {'ilimit': 10, 'atol': 1e-6, 'rtol':1e-6, 'iprint': 1}
        self._options.update(options)
        self._kwargs = kwargs
        self.subsolvers = subsolvers

    def _setup_solvers(self, system, depth):
        self._system = system
        self._depth = depth

        for solver in self.subsolvers:
            solver._setup_solvers(system, depth+1)

    def _mpi_print(self, iteration, res, res0):
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
        iprint = self._options['iprint']
        _solvers_print = self._system._solvers_print
        if iproc == 0 and iprint and _solvers_print:
            print_str = ' ' * self._system._sys_depth + '-' * self._depth
            print_str += sys_name + solver_name
            print_str += ' %3d | %.9g %.9g' % (iteration, res, res0)
            print(print_str)

    def _run_iterator(self):
        ilimit = self._options['ilimit']
        atol = self._options['atol']
        rtol = self._options['rtol']

        norm0, norm = self._iter_initialize()
        iteration = 0
        self._mpi_print(iteration, norm/norm0, norm0)
        while iteration < ilimit and norm > atol and norm/norm0 > rtol:
            self._iter_execute()
            norm = self._iter_get_norm()
            iteration += 1
            self._mpi_print(iteration, norm/norm0, norm)
        success = not(norm > atol and norm/norm0 > rtol)
        success = success and (not numpy.isinf(norm))
        success = success and (not numpy.isnan(norm))
        return not success, norm/norm0, norm

    def _iter_initialize(self):
        pass

    def _iter_execute(self):
        pass

    def _iter_get_norm(self):
        pass


class NonlinearSolver(Solver):

    def __call__(self):
        return self._run_iterator()

    def _iter_initialize(self):
        if self._options['ilimit'] > 1:
            norm = self._iter_get_norm()
        else:
            norm = 1.0
        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def _iter_get_norm(self):
        self._system._apply_nonlinear()
        return self._system._residuals.get_norm()


class NewtonSolver(NonlinearSolver):

    METHOD = 'NL: Newton'

    def _iter_execute(self):
        system = self._system
        system._vectors['residual'][''].set_vec(system._residuals)
        system._vectors['residual'][''] *= -1.0
        system.linearize()
        self.subsolvers['linear']([''], 'fwd')
        if 'linesearch' in self.subsolvers:
            self.subsolvers['linesearch']()
        else:
            system._outputs += system._vectors['output']['']


class NonlinearBlockJac(NonlinearSolver):

    METHOD = 'NL: NLBJ'

    def _iter_execute(self):
        system = self._system
        system._transfers[None](system._inputs, system._outputs, 'fwd')
        for subsys in system._subsystems_myproc:
            subsys._solve_nonlinear()


class NonlinearBlockGS(NonlinearSolver):

    METHOD = 'NL: NLBGS'

    def _iter_execute(self):
        system = self._system
        for isub in range(len(system._subsystems_allprocs)):
            system._transfers['fwd', isub](system._inputs, system._outputs, 'fwd')

            if isub in system._subsystems_inds:
                index = system._subsystems_inds.index(isub)
                subsys = system._subsystems_myproc[index]
                subsys._solve_nonlinear()


class LinearSolver(Solver):

    def __call__(self, vec_names, mode):
        self._vec_names = vec_names
        self._mode = mode
        return self._run_iterator()

    def _iter_initialize(self):
        system = self._system

        self._rhs_vecs = {}
        for vec_name in self._vec_names:
            if self._mode == 'fwd':
                x_vec = system._vectors['output'][vec_name]
                b_vec = system._vectors['residual'][vec_name]
            elif self._mode == 'rev':
                x_vec = system._vectors['residual'][vec_name]
                b_vec = system._vectors['output'][vec_name]

            self._rhs_vecs[vec_name] = b_vec._clone()
            self._rhs_vecs[vec_name].set_vec(b_vec)

        if self._options['ilimit'] > 1:
            norm = self._iter_get_norm()
        else:
            norm = 1.0
        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def _iter_get_norm(self):
        system = self._system
        ind1, ind2 = system._variable_allprocs_range['output']

        system._apply_linear(self._vec_names, self._mode, [ind1, ind2])

        norm = 0
        for vec_name in vec_names:
            if self._mode == 'fwd':
                x_vec = system._vectors['output'][vec_name]
                b_vec = system._vectors['residual'][vec_name]
            elif self._mode == 'rev':
                x_vec = system._vectors['residual'][vec_name]
                b_vec = system._vectors['output'][vec_name]

            b_vec -= self._rhs_vecs[vec_name]
            norm += b_vec.get_norm()**2

        return norm ** 0.5



class ScipyIterativeSolver(LinearSolver):

    METHOD = 'LN: SCIPY'

    def _mat_vec(self, in_vec):
        vec_name = self._vec_name
        system = self._system
        ind1, ind2 = system._variable_allprocs_range['output']

        if self._mode == 'fwd':
            x_vec = system._vectors['output'][vec_name]
            b_vec = system._vectors['residual'][vec_name]
        elif self._mode == 'rev':
            x_vec = system._vectors['residual'][vec_name]
            b_vec = system._vectors['output'][vec_name]

        # TODO: generalize this to multiple var_sets
        x_vec._data[0][:] = in_vec
        system._apply_linear([vec_name], self._mode, [ind1, ind2])
        return b_vec._data[0][:]

    def _monitor(self, res):
        norm = numpy.linalg.norm(res)
        if self._counter == 0:
            if norm != 0.0:
                self._norm0 = norm
            else:
                self._norm0 = 1.0
        self._mpi_print(self._counter, norm/self._norm0, norm)
        self._counter += 1

    def __call__(self, vec_names, mode):
        self._vec_names = vec_names
        self._mode = mode

        system = self._system
        solver = self._options['solver']

        ilimit = self._options['ilimit']
        atol = self._options['atol']
        rtol = self._options['rtol']

        for vec_name in self._vec_names:
            self._vec_name = vec_name

            if self._mode == 'fwd':
                x_vec = system._vectors['output'][vec_name]
                b_vec = system._vectors['residual'][vec_name]
            elif self._mode == 'rev':
                x_vec = system._vectors['residual'][vec_name]
                b_vec = system._vectors['output'][vec_name]

            # TODO: generalize this to multiple var_sets
            size = x_vec._data[0].shape[0]
            linop = LinearOperator((size, size), dtype=float,
                                   matvec=self._mat_vec)
            self._counter = 0
            x_vec.data[:] = solver(linop, numpy.array(b_vec._data[0]),
                                   x0=numpy.array(x_vec._data[0]),
                                   maxiter=ilimit, tol=atol,
                                   callback=self._monitor)[0]
