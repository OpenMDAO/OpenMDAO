"""Test the code we put in out main solver feature document."""

import unittest

import numpy as np

import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_no_warning
from openmdao.test_suite.components.sellar import SellarDerivatives, SellarDerivativesGrouped
from openmdao.test_suite.components.double_sellar import DoubleSellar, SubSellar
from openmdao.utils.testing_utils import use_tempdirs


class TestSolverFeatures(unittest.TestCase):

    def test_specify_solver(self):

        prob = om.Problem()
        model = prob.model = SellarDerivatives()

        model.nonlinear_solver = newton = om.NewtonSolver(solve_subsystems=False)

        # using a different linear solver for Newton with a looser tolerance
        newton.linear_solver = om.ScipyKrylov(atol=1e-4)

        # used for analytic derivatives
        model.linear_solver = om.DirectSolver()

        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

    def test_reuse_solver(self):

        prob = om.Problem()
        model = prob.model = SellarDerivativesGrouped()

        newton = om.NewtonSolver(solve_subsystems=False)

        model.nonlinear_solver = newton

        prob.setup()

        # reusing newton solver is not allowed
        model.mda.nonlinear_solver = newton

        with self.assertRaises(RuntimeError) as context:
            prob.run_model()

        self.assertEqual(str(context.exception),
                         "NewtonSolver has already been assigned to "
                         "<model> <class SellarDerivativesGrouped> "
                         "and cannot also be assigned to 'mda' <class Group>.")

    def test_specify_subgroup_solvers(self):

        prob = om.Problem()
        model = prob.model = DoubleSellar()

        # each SubSellar group converges itself
        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g1.linear_solver = om.DirectSolver()  # used for derivatives

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g2.linear_solver = om.DirectSolver()

        # Converge the outer loop with Gauss Seidel, with a looser tolerance.
        model.nonlinear_solver = om.NonlinearBlockGS(rtol=1.0e-5)
        model.linear_solver = om.ScipyKrylov()
        model.linear_solver.precon = om.LinearBlockGS()

        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val('g1.y1'), 0.64, .00001)
        assert_near_equal(prob.get_val('g1.y2'), 0.80, .00001)
        assert_near_equal(prob.get_val('g2.y1'), 0.64, .00001)
        assert_near_equal(prob.get_val('g2.y2'), 0.80, .00001)

    def test_nonlinear_solver_stall_detection(self):

        for stall_tol_type in ('rel', 'abs'):
            with self.subTest(f'Stall detection for stall_tol_type={stall_tol_type}'):
                prob = om.Problem()
                prob.model.add_subsystem('comp', om.ExecComp('y=3*x+1'), promotes=['*'])

                balance = prob.model.add_subsystem('balance', om.BalanceComp(), promotes=['*'])
                balance.add_balance('x', lower=-.1, upper=10, rhs_val=0, lhs_name='y')

                newton = prob.model.nonlinear_solver = om.NewtonSolver()
                newton.options['solve_subsystems'] = True
                newton.options['stall_limit'] = 3
                newton.options['stall_tol'] = 1e-8
                newton.options['stall_tol_type'] = stall_tol_type
                newton.options['maxiter'] = 100
                newton.options['err_on_non_converge'] = True

                prob.model.linear_solver = om.DirectSolver()

                prob.setup()

                with self.assertRaises(om.AnalysisError) as context:
                    prob.run_model()

                msg = "Solver 'NL: Newton' on system '' stalled after 4 iterations."
                self.assertEqual(str(context.exception), msg)

    def test_nonlinear_solver_bounds_stall_warning(self):
        prob = om.Problem()

        prob.model.add_subsystem('comp', om.ExecComp('y=3*x+1'), promotes=['*'])

        balance = prob.model.add_subsystem('balance', om.BalanceComp(), promotes=['*'])
        balance.add_balance('x', lower=-.1, upper=10, rhs_val=0, lhs_name='y')

        newton = prob.model.nonlinear_solver = om.NewtonSolver()
        newton.options['solve_subsystems'] = True
        newton.options['stall_limit'] = 5
        newton.options['stall_tol'] = 1e-8
        newton.options['maxiter'] = 100
        newton.options['err_on_non_converge'] = False

        prob.model.linear_solver = om.DirectSolver()

        prob.setup()

        msg = ("Your model has stalled three times and may be violating the bounds. "
               "In the future, turn on print_bound_enforce in your solver options "
               "here: \nnonlinear_solver.linesearch.options"
               "['print_bound_enforce']=True. "
               "\nThe bound(s) being violated now are:\n")
        with assert_warning(UserWarning, msg):
            prob.run_model()

        newton.linesearch.options['print_bound_enforce'] = True

        prob.setup()

        with assert_no_warning(UserWarning, msg):
            prob.run_model()

    def test_nonlinear_solver_lower_level_bounds_stall_warning(self):

        prob = om.Problem()

        group1 = prob.model.add_subsystem('balance_group', subsys=om.Group())

        group1.add_subsystem('comp', om.ExecComp('y=3*x+1'), promotes=['*'])

        balance = group1.add_subsystem('balance', om.BalanceComp(), promotes=['*'])
        balance.add_balance('x', lower=-.1, upper=10, rhs_val=0, lhs_name='y')

        newton = group1.nonlinear_solver = om.NewtonSolver()
        newton.options['solve_subsystems'] = True
        newton.options['stall_limit'] = 5
        newton.options['stall_tol'] = 1e-8
        newton.options['maxiter'] = 100
        newton.options['err_on_non_converge'] = False

        group1.linear_solver = om.DirectSolver()

        prob.setup()

        msg = ("Your model has stalled three times and may be violating the bounds. "
               "In the future, turn on print_bound_enforce in your solver options "
               "here: \nbalance_group.nonlinear_solver.linesearch.options"
               "['print_bound_enforce']=True. "
               "\nThe bound(s) being violated now are:\n")

        with assert_warning(UserWarning, msg):
            prob.run_model()

    def test_feature_stall_detection_newton(self):

        prob = om.Problem()

        prob.model.add_subsystem('comp', om.ExecComp('y=3*x+1'), promotes=['*'])

        balance = prob.model.add_subsystem('balance', om.BalanceComp(),
                                           promotes=['*'])
        balance.add_balance('x', lower=-.1, upper=10, rhs_val=0, lhs_name='y')

        newton = prob.model.nonlinear_solver = om.NewtonSolver()
        newton.options['solve_subsystems'] = True
        newton.options['stall_limit'] = 3
        newton.options['stall_tol'] = 1e-8
        newton.options['maxiter'] = 100

        prob.model.linear_solver = om.DirectSolver()

        prob.setup()
        prob.set_solver_print()

        prob.run_model()

    def test_feature_stall_detection_broyden(self):

        prob = om.Problem()

        prob.model.add_subsystem('comp', om.ExecComp('y=3*x+1'), promotes=['*'])

        balance = prob.model.add_subsystem('balance', om.BalanceComp(),
                                           promotes=['*'])
        balance.add_balance('x', lower=-.1, upper=10, rhs_val=0, lhs_name='y')

        nl_solver = prob.model.nonlinear_solver = om.BroydenSolver()
        nl_solver.options['stall_limit'] = 3
        nl_solver.options['stall_tol'] = 1e-8
        nl_solver.options['maxiter'] = 100

        prob.model.linear_solver = om.DirectSolver()

        prob.setup()
        prob.set_solver_print()

        prob.run_model()

    def test_linesearch_property(self):
        import openmdao.api as om

        ns = om.NewtonSolver()
        bs = om.BroydenSolver()
        nlbgs = om.NonlinearBlockGS()
        nlbj = om.NonlinearBlockJac()

        for solver in [ns, bs, nlbgs, nlbj]:
            with self.subTest(msg=solver.msginfo):
                if solver in (ns, bs):
                    self.assertIsInstance(solver.linesearch, om.BoundsEnforceLS)
                else:
                    self.assertIsNone(solver.linesearch)

                new_ls = om.ArmijoGoldsteinLS()

                if solver in (nlbgs, nlbj):
                    with self.assertRaises(AttributeError) as e:
                        solver.linesearch = new_ls
                    expected = (f'{solver.msginfo}: This solver does not '
                                'support a linesearch.')
                    self.assertEqual(expected, str(e.exception))
                else:
                    solver.linesearch = new_ls
                    self.assertIs(solver.linesearch, new_ls)

    def test_solver_broken_weakref(self):

        prob = om.Problem()
        sys = prob.model.add_subsystem('sellar', SellarDerivatives())

        sys.nonlinear_solver =om.NewtonSolver(solve_subsystems=False)

        # Simulate Broken Weakref
        solver = sys.nonlinear_solver
        solver._system = lambda: None

        # Setup should still run with broken ref
        prob.setup()

        # Message info should still be readible
        info = solver.msginfo
        assert info == type(solver).__name__

    @use_tempdirs
    def test_solver_get_outputs_dir(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])))

        sub1 = model.add_subsystem('sub1', om.Group())
        sub2 = sub1.add_subsystem('sub2', om.Group())
        g1 = sub2.add_subsystem('g1', SubSellar())
        g2 = model.add_subsystem('g2', SubSellar())

        model.connect('sub1.sub2.g1.y2', 'g2.x')
        model.connect('g2.y2', 'sub1.sub2.g1.x')

        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov()
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 0

        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g1.linear_solver = om.LinearBlockGS()

        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g2.linear_solver = om.ScipyKrylov()
        g2.linear_solver.precon = om.LinearBlockGS()
        g2.linear_solver.precon.options['maxiter'] = 2

        prob.set_solver_print(level=-1, type_='all')
        g2.set_solver_print(level=2, type_='NL')

        prob.setup()

        with self.assertRaises(RuntimeError) as e:
            g1.nonlinear_solver.get_outputs_dir()

        self.assertEqual('The output directory for Solvers cannot be accessed before final_setup.',
                         str(e.exception))

        prob.final_setup()

        outputs_dir = prob.get_outputs_dir()

        nlg1_dir = g1.nonlinear_solver.get_outputs_dir()
        lg1_dir = g1.linear_solver.get_outputs_dir()

        nlg2_dir = g2.nonlinear_solver.get_outputs_dir()
        lg2_dir = g2.linear_solver.get_outputs_dir()
        lpcg2_dir = g2.linear_solver.precon.get_outputs_dir()

        self.assertEqual(outputs_dir, nlg1_dir)
        self.assertEqual(outputs_dir, lg1_dir)
        self.assertEqual(outputs_dir, nlg2_dir)
        self.assertEqual(outputs_dir, lg2_dir)
        self.assertEqual(outputs_dir, lpcg2_dir)


if __name__ == "__main__":
    unittest.main()
