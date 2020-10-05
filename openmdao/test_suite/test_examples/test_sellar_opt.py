import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class TestSellarOpt(unittest.TestCase):

    def test_sellar_opt(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar_feature import SellarMDA

        prob = om.Problem()
        prob.model = SellarMDA()

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        # prob.driver.options['maxiter'] = 100
        prob.driver.options['tol'] = 1e-8

        prob.model.add_design_var('x', lower=0, upper=10)
        prob.model.add_design_var('z', lower=0, upper=10)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0)
        prob.model.add_constraint('con2', upper=0)

        # Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer
        prob.model.approx_totals()

        prob.setup()
        prob.set_solver_print(level=0)

        prob.run_driver()

        print('minimum found at')
        assert_near_equal(prob.get_val('x')[0], 0., 1e-5)
        assert_near_equal(prob.get_val('z'), [1.977639, 0.], 1e-5)

        print('minumum objective')
        assert_near_equal(prob.get_val('obj')[0], 3.18339395045, 1e-5)


if __name__ == "__main__":
    unittest.main()
