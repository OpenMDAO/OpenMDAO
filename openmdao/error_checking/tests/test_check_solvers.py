
import unittest

from openmdao.api import ImplicitComponent, Problem, Group, IndepVarComp, ExecComp, \
    LinearBlockGS, NonlinearBlockGS, ScipyKrylov, NewtonSolver
from openmdao.utils.logger_utils import TestLogger
from openmdao.test_suite.components.sellar import StateConnection, SellarDerivativesGrouped


class StateConnWithSolveLinear(StateConnection):
    def solve_linear(self, dumat, drmat, vois, mode=None):
        pass


class TestCheckSolvers(unittest.TestCase):

    def test_state_single(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('indep', IndepVarComp('y', 1.0))
        model.add_subsystem('statecomp', StateConnection())

        model.connect('indep.y', 'statecomp.y2_actual')

        # perform setup with checks but don't run model
        testlogger = TestLogger()
        prob.setup(check=True, logger=testlogger)
        prob.final_setup()

        # should trigger a warning due to having states without iterative solvers
        warnings = testlogger.get('warning')
        self.assertEqual(len(warnings), 2)

        self.assertEqual(warnings[0],
                         "StateConnection 'statecomp' contains implicit variables, "
                         "but does not have an iterative nonlinear solver and does not "
                         "implement 'solve_nonlinear'.")

        self.assertEqual(warnings[1],
                         "StateConnection 'statecomp' contains implicit variables, "
                         "but does not have an iterative linear solver and does not "
                         "implement 'solve_linear'.")

    def test_state_single_w_ancestor_iter(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('indep', IndepVarComp('y', 1.0))

        model.add_subsystem("G1", Group())
        model.G1.add_subsystem('statecomp', StateConnection(),
                               promotes_inputs=['y2_actual'])

        model.connect('indep.y', 'G1.y2_actual')

        # provide iterative linear solver for implicit group
        model.linear_solver = LinearBlockGS()
        model.linear_solver.options['maxiter'] = 5

        # perform setup with checks but don't run model
        testlogger = TestLogger()
        prob.setup(check=True, logger=testlogger)
        prob.final_setup()

        # should not trigger any solver warnings because maxiter > 1
        warnings = testlogger.get('warning')
        self.assertEqual(len(warnings), 0)

    def test_state_not_single(self):
        prob = Problem()
        model = prob.model

        model.linear_solver = ScipyKrylov()

        model.add_subsystem('indep', IndepVarComp('y', 1.0))
        model.add_subsystem('statecomp', StateConnection())

        model.connect('indep.y', 'statecomp.y2_actual')

        # perform setup with checks but don't run model
        testlogger = TestLogger()
        prob.setup(check=True, logger=testlogger)
        prob.final_setup()

        # should not trigger any solver warnings
        warnings = testlogger.get('warning')
        self.assertEqual(len(warnings), 0)

    def test_state_single_maxiter_gt_1(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('indep', IndepVarComp('y', 1.0))
        model.add_subsystem('statecomp', StateConnection())

        model.connect('indep.y', 'statecomp.y2_actual')

        # provide iterative linear solver for implicit group
        model.linear_solver = LinearBlockGS()
        model.linear_solver.options['maxiter'] = 2

        # perform setup with checks but don't run model
        testlogger = TestLogger()
        prob.setup(check=True, logger=testlogger)
        prob.final_setup()

        # should not trigger any solver warnings because maxiter > 1
        warnings = testlogger.get('warning')
        self.assertEqual(len(warnings), 0)

    def test_cycle(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem("C1", ExecComp('y=2.0*x'))
        model.add_subsystem("C2", ExecComp('y=2.0*x'))
        model.add_subsystem("C3", ExecComp('y=2.0*x'))

        model.connect('C1.y','C2.x')
        model.connect('C2.y','C3.x')
        model.connect('C3.y','C1.x')

        # perform setup with checks but don't run model
        testlogger = TestLogger()
        prob.setup(check=True, logger=testlogger)
        prob.final_setup()

        # should trigger warnings because cycle requires iterative solvers
        warnings = testlogger.get('warning')
        self.assertEqual(len(warnings), 2)

        self.assertEqual(warnings[0],
                         "Group '' contains cycles [['C1', 'C2', 'C3']], but "
                         "does not have an iterative nonlinear solver.")

        self.assertEqual(warnings[1],
                         "Group '' contains cycles [['C1', 'C2', 'C3']], but "
                         "does not have an iterative linear solver.")

    def test_cycle_maxiter_gt_1(self):
        prob = Problem()
        model = prob.model

        C1 = model.add_subsystem("C1", ExecComp('y=2.0*x'))
        C2 = model.add_subsystem("C2", ExecComp('y=2.0*x'))
        C3 = model.add_subsystem("C3", ExecComp('y=2.0*x'))

        model.connect('C1.y','C2.x')
        model.connect('C2.y','C3.x')
        model.connect('C3.y','C1.x')

        # provide iterative solvers to handle cycle
        model.linear_solver = LinearBlockGS()
        model.linear_solver.options['maxiter'] = 2

        model.nonlinear_solver = NonlinearBlockGS()
        model.nonlinear_solver.options['maxiter'] = 2

        # perform setup with checks but don't run model
        testlogger = TestLogger()
        prob.setup(check=True, logger=testlogger)
        prob.final_setup()

        # should not trigger any solver warnings because maxiter > 1
        warnings = testlogger.get('warning')
        self.assertEqual(len(warnings), 0)

    def test_cycle_maxiter_gt_1_subgroup(self):
        prob = Problem()
        model = prob.model

        G1 = model.add_subsystem("G1", Group())
        C1 = G1.add_subsystem("C1", ExecComp('y=2.0*x'))
        C2 = G1.add_subsystem("C2", ExecComp('y=2.0*x'))
        C3 = G1.add_subsystem("C3", ExecComp('y=2.0*x'))

        G1.connect('C1.y','C2.x')
        G1.connect('C2.y','C3.x')
        G1.connect('C3.y','C1.x')

        # provide iterative solvers to handle cycle (default maxiter = 10)
        model.linear_solver = LinearBlockGS()
        model.nonlinear_solver = NonlinearBlockGS()

        # perform setup with checks but don't run model
        testlogger = TestLogger()
        prob.setup(check=True, logger=testlogger)
        prob.final_setup()

        # should not trigger solver warning because maxiter > 1 in ancestor group
        warnings = testlogger.get('warning')
        self.assertEqual(len(warnings), 0)


if __name__ == "__main__":
    unittest.main()
