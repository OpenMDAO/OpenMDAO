"""Test the code we put in out main solver feature document."""

import unittest

import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.test_suite.components.sellar import SellarDerivatives
from openmdao.test_suite.components.double_sellar import DoubleSellar


class TestSolverFeatures(unittest.TestCase):

    def test_specify_solver(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

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

    def test_specify_subgroup_solvers(self):
        import openmdao.api as om
        from openmdao.test_suite.components.double_sellar import DoubleSellar

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
        prob = om.Problem()

        prob.model.add_subsystem('comp', om.ExecComp('y=3*x+1'), promotes=['*'])

        balance = prob.model.add_subsystem('balance', om.BalanceComp(), promotes=['*'])
        balance.add_balance('x', lower=-.1, upper=10, rhs_val=0, lhs_name='y')

        newton = prob.model.nonlinear_solver = om.NewtonSolver()
        newton.options['solve_subsystems'] = True
        newton.options['stall_limit'] = 3
        newton.options['stall_tol'] = 1e-8
        newton.options['maxiter'] = 100
        newton.options['err_on_non_converge'] = True

        prob.model.linear_solver = om.DirectSolver()

        prob.setup()

        with self.assertRaises(om.AnalysisError) as context:
            prob.run_model()

        msg = "Solver 'NL: Newton' on system '' stalled after 4 iterations."
        self.assertEqual(str(context.exception), msg)

    def test_feature_stall_detection_newton(self):
        import openmdao.api as om

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
        import openmdao.api as om

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


if __name__ == "__main__":
    unittest.main()
