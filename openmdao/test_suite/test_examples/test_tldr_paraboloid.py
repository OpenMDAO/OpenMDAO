from __future__ import print_function, division, absolute_import

import unittest

from openmdao.utils.assert_utils import assert_rel_error


class TestParaboloidTLDR(unittest.TestCase):

    def test_tldr(self):
        from openmdao.api import Problem, ScipyOptimizeDriver, ExecComp, IndepVarComp

        # build the model
        prob = Problem()
        indeps = prob.model.add_subsystem('indeps', IndepVarComp())
        indeps.add_output('x', 3.0)
        indeps.add_output('y', -4.0)

        prob.model.add_subsystem('paraboloid', ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))

        prob.model.connect('indeps.x', 'paraboloid.x')
        prob.model.connect('indeps.y', 'paraboloid.y')

        # setup the optimization
        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('indeps.x', lower=-50, upper=50)
        prob.model.add_design_var('indeps.y', lower=-50, upper=50)
        prob.model.add_objective('paraboloid.f')

        prob.setup()
        prob.run_driver()

        # minimum value
        assert_rel_error(self, prob['paraboloid.f'], -27.33333, 1e-6)

        # location of the minimum
        assert_rel_error(self, prob['indeps.x'], 6.6667, 1e-4)
        assert_rel_error(self, prob['indeps.y'], -7.33333, 1e-4)

if __name__ == "__main__":

    unittest.main()
