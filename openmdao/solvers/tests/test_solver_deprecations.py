""" Unit deprecated behavior for solvers. """

import unittest

from openmdao.api import Problem, NewtonSolver, ScipyIterativeSolver
from openmdao.test_suite.components.sellar import SellarStateConnection


class TestSolverDeprecations(unittest.TestCase):

    def test_newton_specify_linear_solver(self):
        # Newton solver no longer has a .ln_solver slot.

        prob = Problem()
        prob.model = SellarStateConnection()
        prob.model.nl_solver = NewtonSolver()

        # Use bad settings for this one so that problem doesn't converge.
        # That way, we test that we are really using Newton's Lin Solver
        # instead.
        prob.model.ln_solver = ScipyIterativeSolver()
        prob.model.ln_solver.options['maxiter'] = 1

        # The good solver
        prob.model.nl_solver.options['subsolvers']['linear'] = ScipyIterativeSolver()

        prob.model.suppress_solver_output = True
        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['state_eq.y2_command'], 12.05848819, .00001)

        ## Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nl_solver._iter_count, 8)
        self.assertEqual(prob.model.ln_solver._iter_count, 0)
        self.assertGreater(prob.model.nl_solver.options['subsolvers']['linear']._iter_count, 0)


if __name__ == "__main__":
    unittest.main()
