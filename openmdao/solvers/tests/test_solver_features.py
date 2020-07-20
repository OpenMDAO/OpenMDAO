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


if __name__ == "__main__":
    unittest.main()
