""" Unit test for the solver printing behavior. """

import sys
import unittest

from six.moves import cStringIO

from openmdao.api import Problem, Group, NewtonSolver, ScipyIterativeSolver
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.sellar import SellarDerivatives


class TestSolverPrint(unittest.TestCase):

    def test_feature_iprint_neg1(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        newton = prob.model.nl_solver = NewtonSolver()
        ln_scipy = prob.model.ln_solver = ScipyIterativeSolver()

        newton.options['maxiter'] = 2
        prob.setup(check=False)

        # use a reall bad initial guess
        prob['y1'] = 10000
        prob['y2'] = -26

        newton.options['iprint'] = -1
        ln_scipy.options['iprint'] = -1
        prob.run_model()

        # TODO: capture stdout for both the test assert and docs embedding

    def test_feature_iprint_0(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        newton = prob.model.nl_solver = NewtonSolver()
        ln_scipy = prob.model.ln_solver = ScipyIterativeSolver()

        newton.options['maxiter'] = 2
        prob.setup(check=False)

        prob['y1'] = 10000
        prob['y2'] = -26

        newton.options['iprint'] = 0
        ln_scipy.options['iprint'] = -1

        prob.run_model()

        # TODO: capture stdout for both the test assert and docs embedding

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

        # TODO: capture stdout for both the test assert and docs embedding

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

        # TODO: capture stdout for both the test assert and docs embedding


if __name__ == "__main__":
    unittest.main()