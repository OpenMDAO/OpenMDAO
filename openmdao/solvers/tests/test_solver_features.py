"""Test the code we put in out main dolver featuer document."""

import unittest

from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp
from openmdao.solvers.nl_newton import NewtonSolver
from openmdao.solvers.ln_direct import DirectSolver
from openmdao.solvers.nl_bgs import NonlinearBlockGS
from openmdao.solvers.ln_scipy import ScipyIterativeSolver
from openmdao.solvers.ln_bgs import LinearBlockGS

from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.sellar import SellarDerivatives
from openmdao.test_suite.components.double_sellar import DoubleSellar


class TestSolverFeatures(unittest.TestCase):

    def test_specify_solver(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.nl_solver = NewtonSolver()
        model.ln_solver = DirectSolver()

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    @unittest.skip('something is broken in the solver setting api')
    def test_specify_subgroup_solvers(self):

        prob = Problem()
        model = prob.model = DoubleSellar()

        # each SubSellar group converges itself
        g1 = model.get_subsystem('g1')
        g1.nl_solver = NonlinearBlockGS()
        g1.ln_solver = DirectSolver()

        g2 = model.get_subsystem('g2')
        g2.nl_solver = NonlinearBlockGS()
        g2.ln_solver = DirectSolver()

        # Converge the outer loop with Gauss Seidel, with a looser tolerance.
        model.nl_solver = NonlinearBlockGS()
        model.nl_solver.options['rtol'] = 1.0e-5
        model.ln_solver = ScipyIterativeSolver()
        model.ln_solver.options['subsolvers']['preconditioner'] = LinearBlockGS()

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['g1.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g1.y2'], 0.80, .00001)
        assert_rel_error(self, prob['g2.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g2.y2'], 0.80, .00001)


if __name__ == "__main__":
    unittest.main()