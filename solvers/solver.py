from __future__ import division
import numpy



class Solver(object):

    SOLVER = 'base_solver'

    def __init__(self, options={}, subsolvers={}, **kwargs):
        self.system = None
        self.depth = 0
        self.options = {'ilimit': 10, 'atol': 1e-6, 'rtol':1e-6, 'iprint': 1}
        self.options.update(options)
        self.subsolvers = subsolvers
        self.kwargs = kwargs

    def setup_solvers(self, system, depth):
        self.system = system
        self.depth = depth

        for solver in self.subsolvers:
            solver.setup_solvers(system, depth+1)

    def mpi_print(self, iteration, res, res0):
        raw_sys_name = self.system.sys_name
        name_len = 10
        if len(raw_sys_name) > name_len:
            sys_name = raw_sys_name[:name_len]
        else:
            sys_name = raw_sys_name + ' ' * (name_len - len(raw_sys_name))

        solver_name = self.SOLVER
        name_len = 12
        if len(solver_name) > name_len:
            solver_name = solver_name[:name_len]
        else:
            solver_name = solver_name + ' ' * (name_len - len(solver_name))

        iproc = self.system.mpi_comm.rank
        iprint = self.options['iprint']
        solvers_print = self.system.solvers_print
        if iproc == 0 and iprint and solvers_print:
            print_str = ' ' * self.system.sys_depth + '-' * self.depth
            print_str += sys_name + solver_name
            print_str += ' %3d | %.9g %.9g' % (iteration, res, res0)
            print print_str

    def run_iterator(self):
        ilimit = self.options['ilimit']
        atol = self.options['atol']
        rtol = self.options['rtol']

        norm0, norm = self.iter_initialize()
        iteration = 0
        self.mpi_print(iteration, norm/norm0, norm0)
        while iteration < ilimit and norm > atol and norm/norm0 > rtol:
            self.iter_execute()
            norm = self.iter_get_norm()
            iteration += 1
            self.mpi_print(iteration, norm/norm0, norm)
        success = not(norm > atol and norm/norm0 > rtol)
        success = success and (not numpy.isinf(norm))
        success = success and (not numpy.isnan(norm))
        return not success, norm/norm0, norm

    def iter_initialize(self):
        pass

    def iter_execute(self):
        pass

    def iter_get_norm(self):
        pass



class NonlinearSolver(Solver):

    def __call__(self):
        return self.run_iterator()

    def iter_initialize(self):
        if self.options['ilimit'] > 1:
            norm = self.iter_get_norm()
        else:
            norm = 1.0
        norm0 = norm if norm != 0.0 else 1.0
        return norm0, norm

    def iter_get_norm(self):
        self.system.apply_nonlinear()
        return self.system.residuals.get_norm()



class NewtonSolver(NonlinearSolver):

    METHOD = 'NL: Newton'

    def iter_execute(self):
        system = self.system
        system.vectors['residual'][''].set_vec(system.residuals)
        system.vectors['residual'][''] *= -1.0
        system.linearize()
        self.subsolvers['linear']([None], 'fwd')
        if 'linesearch' in self.subsolvers:
            self.subsolvers['linesearch']()
        else:
            system.outputs += system.vector_list['output'][0]



class NonlinearBlockJac(NonlinearSolver):

    METHOD = 'NL: NLBJ'

    def iter_execute(self):
        system = self.system
        system.transfers[None](system.inputs, system.outputs, 'fwd')
        for subsys in system.subsystems_myproc:
            subsys.solve_nonlinear()



class NonlinearBlockGS(NonlinearSolver):

    METHOD = 'NL: NLBGS'

    def iter_execute(self):
        system = self.system
        for isub in xrange(len(system.subsystems_allprocs)):
            system.transfers['fwd', isub](system.inputs, system.outputs, 'fwd')

            if isub in system.subsystems_inds:
                index = system.subsystems_inds.index(isub)
                subsys = system.subsystems_myproc[index]
                subsys.solve_nonlinear()
