import unittest

from openmdao.utils.assert_utils import assert_near_equal


class TestParaboloidTLDR(unittest.TestCase):

    def test_tldr(self):
        import openmdao.api as om

        # build the model
        prob = om.Problem()

        prob.model.add_subsystem('paraboloid', om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))

        # setup the optimization
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('paraboloid.x', lower=-50, upper=50)
        prob.model.add_design_var('paraboloid.y', lower=-50, upper=50)
        prob.model.add_objective('paraboloid.f')

        prob.setup()

        # Set initial values.
        prob.set_val('paraboloid.x', 3.0)
        prob.set_val('paraboloid.y', -4.0)

        prob.run_driver()

        # minimum value
        assert_near_equal(prob['paraboloid.f'], -27.33333, 1e-6)

        # location of the minimum
        assert_near_equal(prob['paraboloid.x'], 6.6667, 1e-4)
        assert_near_equal(prob['paraboloid.y'], -7.33333, 1e-4)

if __name__ == "__main__":

    unittest.main()
