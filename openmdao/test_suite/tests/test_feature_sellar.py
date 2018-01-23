"""Test the Sellar systems used in the Sellar feature doc. """

import unittest

from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.test_suite.components.sellar_feature import SellarMDA


class TestSellarFeature(unittest.TestCase):

    def test_sellar(self):
        # Just tests Newton on Sellar with FD derivs.

        prob = Problem()
        prob.model = SellarMDA()

        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nonlinear_solver._iter_count, 8)

if __name__ == "__main__":
    unittest.main()
