import unittest

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.test_suite.components.paraboloid import Paraboloid


class BasicOptParaboloid(unittest.TestCase):

    def test_unconstrainted(self):
        import openmdao.api as om

        # We'll use the component that was defined in the last tutorial
        from openmdao.test_suite.components.paraboloid import Paraboloid

        # build the model
        prob = om.Problem()
        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', 3.0)
        indeps.add_output('y', -4.0)

        prob.model.add_subsystem('paraboloid', Paraboloid())

        prob.model.connect('indeps.x', 'paraboloid.x')
        prob.model.connect('indeps.y', 'paraboloid.y')

        # setup the optimization
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'COBYLA'

        prob.model.add_design_var('indeps.x', lower=-50, upper=50)
        prob.model.add_design_var('indeps.y', lower=-50, upper=50)
        prob.model.add_objective('paraboloid.f_xy')

        prob.setup()
        prob.run_driver()

        # minimum value
        assert_near_equal(prob['paraboloid.f_xy'], -27.33333, 1e-6)

        # location of the minimum
        assert_near_equal(prob['indeps.x'], 6.6667, 1e-4)
        assert_near_equal(prob['indeps.y'], -7.33333, 1e-4)

    def test_constrained(self):
        import openmdao.api as om

        # We'll use the component that was defined in the last tutorial
        from openmdao.test_suite.components.paraboloid import Paraboloid

        # build the model
        prob = om.Problem()
        prob.model.add_subsystem('parab', Paraboloid(), promotes_inputs=['x', 'y'])

        # define the component whose output will be constrained
        prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

        # Design variables 'x' and 'y' span components, so we need to provide a common initial
        # value for them.
        prob.model.set_input_defaults('x', 3.0)
        prob.model.set_input_defaults('y', -4.0)

        # setup the optimization
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'COBYLA'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('parab.f_xy')

        # to add the constraint to the model
        prob.model.add_constraint('const.g', lower=0, upper=10.)

        prob.setup()
        prob.run_driver()

        # minimum value
        assert_near_equal(prob.get_val('parab.f_xy'), -27., 1e-6)

        # location of the minimum
        assert_near_equal(prob.get_val('x'), 7, 1e-4)
        assert_near_equal(prob.get_val('y'), -7, 1e-4)


if __name__ == "__main__":

    unittest.main()
