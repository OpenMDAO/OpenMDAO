"""Test the Nonlinear Block Jacobi solver. """

import unittest

from openmdao.api import Problem, Group
from openmdao.devtools.testutil import assert_rel_error
from openmdao.solvers.nl_bjac import NonlinearBlockJac
from openmdao.test_suite.components.sellar import SellarDerivatives


class TestNLBlockJacobi(unittest.TestCase):

    def test_feature_basic(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        nlgbs = prob.model.nl_solver = NonlinearBlockJac()

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_feature_maxiter(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        nlgbs = prob.model.nl_solver = NonlinearBlockJac()

        nlgbs.options['maxiter'] = 4

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.5723813937, .00001)
        assert_rel_error(self, prob['y2'], 12.0542542372, .00001)

    def test_feature_rtol(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        nlgbs = prob.model.nl_solver = NonlinearBlockJac()

        nlgbs.options['rtol'] = 1e-3

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.5891491526, .00001)
        assert_rel_error(self, prob['y2'], 12.0569142166, .00001)

    def test_feature_atol(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        nlgbs = prob.model.nl_solver = NonlinearBlockJac()

        nlgbs.options['atol'] = 1e-2

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.5886171567, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

if __name__ == "__main__":
    unittest.main()