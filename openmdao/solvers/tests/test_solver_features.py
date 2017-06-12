"""Test the code we put in out main dolver featuer document."""

import unittest

from openmdao.api import Problem
from openmdao.solvers.nonlinear.newton import NewtonSolver
from openmdao.solvers.linear.direct import DirectSolver
from openmdao.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS
from openmdao.solvers.linear.scipy_iter_solver import ScipyIterativeSolver
from openmdao.solvers.linear.linear_block_gs import LinearBlockGS

from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.sellar import SellarDerivatives
from openmdao.test_suite.components.double_sellar import DoubleSellar


class TestSolverFeatures(unittest.TestCase):

    def test_specify_solver(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.nonlinear_solver = NewtonSolver()
        # using a different linear solver for Newton with a looser tolerance
        model.nonlinear_solver.linear_solver = ScipyIterativeSolver()
        model.nonlinear_solver.linear_solver.options['atol'] = 1e-4

        # used for analytic derivatives
        model.linear_solver = DirectSolver()

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_specify_subgroup_solvers(self):

        prob = Problem()
        model = prob.model = DoubleSellar()

        # each SubSellar group converges itself
        g1 = model.get_subsystem('g1')
        g1.nonlinear_solver = NewtonSolver()
        g1.linear_solver = DirectSolver()  # used for derivatives

        g2 = model.get_subsystem('g2')
        g2.nonlinear_solver = NewtonSolver()
        g2.linear_solver = DirectSolver()

        # Converge the outer loop with Gauss Seidel, with a looser tolerance.
        model.nonlinear_solver = NonlinearBlockGS()
        model.nonlinear_solver.options['rtol'] = 1.0e-5
        model.linear_solver = ScipyIterativeSolver()
        model.linear_solver.precon = LinearBlockGS()

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['g1.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g1.y2'], 0.80, .00001)
        assert_rel_error(self, prob['g2.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g2.y2'], 0.80, .00001)


if __name__ == "__main__":
    unittest.main()
