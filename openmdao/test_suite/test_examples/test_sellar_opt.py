from __future__ import print_function, division, absolute_import

import unittest

from openmdao.utils.assert_utils import assert_rel_error

from openmdao.api import Problem, ScipyOptimizer, ExecComp, IndepVarComp, DirectSolver


class TestSellarOpt(unittest.TestCase):

    def test_sellar_opt(self):
        from openmdao.api import Problem, ScipyOptimizer, ExecComp, IndepVarComp, DirectSolver
        from openmdao.test_suite.components.sellar_feature import SellarMDA

        prob = Problem()
        prob.model = SellarMDA()


        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        # prob.driver.options['maxiter'] = 100
        prob.driver.options['tol'] = 1e-8

        prob.model.add_design_var('x', lower=0, upper=10)
        prob.model.add_design_var('z', lower=0, upper=10)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0)
        prob.model.add_constraint('con2', upper=0)


        prob.setup()
        prob.set_solver_print(level=0)

        # Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer
        prob.model.approx_totals()


        prob.run_driver()



        print('minimum found at')
        assert_rel_error(self, prob['x'][0], 0., 1e-5)
        assert_rel_error(self, prob['z'], [1.977639, 0.], 1e-5)

        print('minumum objective')
        assert_rel_error(self, prob['obj'][0], 3.18339395045, 1e-5)


if __name__ == "__main__":
    unittest.main()


