"""Test the code we put in out main solver feature document."""

import unittest

from openmdao.api import Problem
from openmdao.solvers.nonlinear.newton import NewtonSolver
from openmdao.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS
from openmdao.solvers.linear.direct import DirectSolver
from openmdao.solvers.linear.scipy_iter_solver import ScipyKrylov
from openmdao.solvers.linear.linear_block_gs import LinearBlockGS

from openmdao.utils.assert_utils import assert_rel_error
from openmdao.test_suite.components.sellar import SellarDerivatives
from openmdao.test_suite.components.double_sellar import DoubleSellar


class TestSolverFeatures(unittest.TestCase):

    def test_specify_solver(self):
        from openmdao.api import Problem, NewtonSolver, ScipyKrylov, DirectSolver
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.nonlinear_solver = newton = NewtonSolver()

        # using a different linear solver for Newton with a looser tolerance
        newton.linear_solver = ScipyKrylov(atol=1e-4)

        # used for analytic derivatives
        model.linear_solver = DirectSolver()

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_specify_subgroup_solvers(self):
        from openmdao.api import Problem, NewtonSolver, ScipyKrylov, DirectSolver, NonlinearBlockGS, LinearBlockGS
        from openmdao.test_suite.components.double_sellar import DoubleSellar

        prob = Problem()
        model = prob.model = DoubleSellar()

        # each SubSellar group converges itself
        g1 = model.g1
        g1.nonlinear_solver = NewtonSolver()
        g1.linear_solver = DirectSolver()  # used for derivatives

        g2 = model.g2
        g2.nonlinear_solver = NewtonSolver()
        g2.linear_solver = DirectSolver()

        # Converge the outer loop with Gauss Seidel, with a looser tolerance.
        model.nonlinear_solver = NonlinearBlockGS(rtol=1.0e-5)
        model.linear_solver = ScipyKrylov()
        model.linear_solver.precon = LinearBlockGS()

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['g1.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g1.y2'], 0.80, .00001)
        assert_rel_error(self, prob['g2.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g2.y2'], 0.80, .00001)


if __name__ == "__main__":
    unittest.main()
