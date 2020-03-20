"""Test the Sellar systems used in the Sellar feature doc. """

import unittest

import openmdao.api as om
from openmdao.test_suite.components.sellar_feature import SellarMDA, SellarMDALinearSolver
from openmdao.utils.assert_utils import assert_near_equal


class TestSellarFeature(unittest.TestCase):

    def test_sellar(self):
        prob = om.Problem()
        prob.model = SellarMDA()

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['y1'], 25.58830273, .00001)
        assert_near_equal(prob['y2'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nonlinear_solver._iter_count, 8)


    def test_sellar_linear_solver(self):

        prob = om.Problem()
        prob.model = SellarMDALinearSolver()

        prob.setup()
        prob.model.cycle.nonlinear_solver.options['use_apply_nonlinear'] = True
        prob.run_model()

        assert_near_equal(prob['y1'], 25.58830273, .00001)
        assert_near_equal(prob['y2'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nonlinear_solver._iter_count, 8)

        jac = prob.compute_totals(of=['obj', 'con1', 'con2'], wrt=['x', 'z'])

        assert_near_equal(jac['obj', 'x'][0], 2.98061391, 1e-4)
        assert_near_equal(jac['obj', 'z'][0], [9.61001154, 1.78448533], 1e-4)
        assert_near_equal(jac['con1', 'x'][0], -0.98061447, 1e-4)
        assert_near_equal(jac['con1', 'z'][0], [-9.61002284, -0.78449158], 1e-4)
        assert_near_equal(jac['con2', 'x'][0], 0.09692762, 1e-4)
        assert_near_equal(jac['con2', 'z'][0], [1.94989079, 1.0775421], 1e-4)


if __name__ == "__main__":
    unittest.main()
