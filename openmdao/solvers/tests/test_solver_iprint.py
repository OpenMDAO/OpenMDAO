""" Unit test for the solver printing behavior. """

import unittest

import numpy as np

from openmdao.api import Problem, NewtonSolver, ScipyIterativeSolver, Group, PetscKSP, \
                         IndepVarComp, NonlinearBlockGS, NonlinearBlockJac, LinearBlockGS
from openmdao.test_suite.components.double_sellar import SubSellar
from openmdao.test_suite.components.sellar import SellarDerivatives


class TestSolverPrint(unittest.TestCase):

    def test_feature_iprint_neg1(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        newton = prob.model.nl_solver = NewtonSolver()
        ln_scipy = prob.model.ln_solver = ScipyIterativeSolver()

        newton.options['maxiter'] = 2
        prob.setup(check=False)

        # use a real bad initial guess
        prob['y1'] = 10000
        prob['y2'] = -26

        newton.options['iprint'] = -1
        ln_scipy.options['iprint'] = -1
        prob.run_model()
        print('done')

    def test_feature_iprint_0(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        newton = prob.model.nl_solver = NewtonSolver()
        ln_scipy = prob.model.ln_solver = ScipyIterativeSolver()

        newton.options['maxiter'] = 1
        prob.setup(check=False)

        prob['y1'] = 10000
        prob['y2'] = -26

        newton.options['iprint'] = 0
        ln_scipy.options['iprint'] = 0

        prob.run_model()

    def test_feature_iprint_1(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        newton = prob.model.nl_solver = NewtonSolver()
        ln_scipy = prob.model.ln_solver = ScipyIterativeSolver()

        newton.options['maxiter'] = 20
        prob.setup(check=False)

        prob['y1'] = 10000
        prob['y2'] = -26

        newton.options['iprint'] = 1
        ln_scipy.options['iprint'] = 0
        prob.run_model()

    def test_feature_iprint_2(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        newton = prob.model.nl_solver = NewtonSolver()
        ln_scipy = prob.model.ln_solver = ScipyIterativeSolver()

        newton.options['maxiter'] = 20
        prob.setup(check=False)

        prob['y1'] = 10000
        prob['y2'] = -20

        newton.options['iprint'] = 2
        ln_scipy.options['iprint'] = 1
        prob.run_model()

    def test_hierarchy_iprint(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', Group())
        sub2 = sub1.add_subsystem('sub2', Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nl_solver = NewtonSolver()
        model.ln_solver = ScipyIterativeSolver()
        model.nl_solver.options['solve_subsystems'] = True
        model.nl_solver.options['max_sub_solves'] = 0

        g1.nl_solver = NewtonSolver()
        g1.ln_solver = LinearBlockGS()

        g2.nl_solver = NewtonSolver()
        g2.ln_solver = ScipyIterativeSolver()
        g2.ln_solver.precon = LinearBlockGS()
        g2.ln_solver.precon.options['maxiter'] = 2

        prob.set_solver_print(level=2)

        prob.setup(check=False)
        prob.run_model()

    def test_hierarchy_iprint2(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', Group())
        sub2 = sub1.add_subsystem('sub2', Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nl_solver = NonlinearBlockGS()
        g1.nl_solver = NonlinearBlockGS()
        g2.nl_solver = NonlinearBlockGS()

        prob.set_solver_print(level=2)

        prob.setup(check=False)
        prob.run_model()

    def test_hierarchy_iprint3(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', Group())
        sub2 = sub1.add_subsystem('sub2', Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nl_solver = NonlinearBlockJac()
        sub1.nl_solver = NonlinearBlockJac()
        sub2.nl_solver = NonlinearBlockJac()
        g1.nl_solver = NonlinearBlockJac()
        g2.nl_solver = NonlinearBlockJac()

        prob.set_solver_print(level=2)

        prob.setup(check=False)
        prob.run_model()

        print('hey')

    def test_feature_set_solver_print1(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', Group())
        sub2 = sub1.add_subsystem('sub2', Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nl_solver = NewtonSolver()
        model.ln_solver = ScipyIterativeSolver()
        model.nl_solver.options['solve_subsystems'] = True
        model.nl_solver.options['max_sub_solves'] = 0

        g1.nl_solver = NewtonSolver()
        g1.ln_solver = LinearBlockGS()

        g2.nl_solver = NewtonSolver()
        g2.ln_solver = ScipyIterativeSolver()
        g2.ln_solver.precon = LinearBlockGS()
        g2.ln_solver.precon.options['maxiter'] = 2

        prob.set_solver_print(level=2)

        prob.setup(check=False)
        prob.run_model()

    def test_feature_set_solver_print2(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', Group())
        sub2 = sub1.add_subsystem('sub2', Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nl_solver = NewtonSolver()
        model.ln_solver = ScipyIterativeSolver()
        model.nl_solver.options['solve_subsystems'] = True
        model.nl_solver.options['max_sub_solves'] = 0

        g1.nl_solver = NewtonSolver()
        g1.ln_solver = LinearBlockGS()

        g2.nl_solver = NewtonSolver()
        g2.ln_solver = ScipyIterativeSolver()
        g2.ln_solver.precon = LinearBlockGS()
        g2.ln_solver.precon.options['maxiter'] = 2

        prob.set_solver_print(level=2)
        prob.set_solver_print(level=-1, type_='LN')

        prob.setup(check=False)
        prob.run_model()

    def test_feature_set_solver_print3(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', Group())
        sub2 = sub1.add_subsystem('sub2', Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('pz.z', 'sub1.sub2.g1.z')
        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nl_solver = NewtonSolver()
        model.ln_solver = ScipyIterativeSolver()
        model.nl_solver.options['solve_subsystems'] = True
        model.nl_solver.options['max_sub_solves'] = 0

        g1.nl_solver = NewtonSolver()
        g1.ln_solver = LinearBlockGS()

        g2.nl_solver = NewtonSolver()
        g2.ln_solver = ScipyIterativeSolver()
        g2.ln_solver.precon = LinearBlockGS()
        g2.ln_solver.precon.options['maxiter'] = 2

        prob.set_solver_print(level=0)
        prob.set_solver_print(level=2, depth=2)

        prob.setup(check=False)
        prob.run_model()

if __name__ == "__main__":
    unittest.main()
