import unittest

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.utils.assert_utils import assert_near_equal

class TestSimpleParaboloid(unittest.TestCase):

    def setUp(self):
        prob = self._prob = om.Problem()
        prob.model.add_subsystem("paraboloid_comp", Paraboloid(), promotes_inputs=["x", "y"], promotes_outputs=["f_xy"])

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('f_xy')

        prob.setup()
        # Need to run final_setup to avoid an error when creating the "_TotalJacInfo" object.
        prob.final_setup()

    def test_f_all_vars(self):
        prob = self._prob
        f = prob.get_callback("f", input_vars=["x", "y"], output_vars=["f_xy"])
        x = f.create_input_vector()

        x[0] = 8.0
        x[1] = 9.0
        assert_near_equal(f(x), (x[0]-3.0)**2 + x[0]*x[1] + (x[1]+4.0)**2 - 3.0)
        # Setting a value that conflicts with what's passed to the functional interface should be overwritten.
        prob.set_val("x", -1.0)
        assert_near_equal(f(x), (x[0]-3.0)**2 + x[0]*x[1] + (x[1]+4.0)**2 - 3.0)

        x[0] = 9.0
        x[1] = 10.0
        y = f.create_output_vector()
        f(x, y=y)
        # Setting a value that conflicts with what's passed to the functional interface should be overwritten.
        prob.set_val("y", -1.0)
        assert_near_equal(y, (x[0]-3.0)**2 + x[0]*x[1] + (x[1]+4.0)**2 - 3.0)

    def test_f_just_x(self):
        prob = self._prob
        x1 = 3.0
        prob.set_val("y", x1)
        f = prob.get_callback("f", input_vars=["x"], output_vars=["f_xy"])
        x = f.create_input_vector()

        x[0] = 8.0
        # x[1] = 9.0
        assert_near_equal(f(x), (x[0]-3.0)**2 + x[0]*x1 + (x1+4.0)**2 - 3.0)
        # Setting a value that conflicts with what's passed to the functional interface should be overwritten.
        prob.set_val("x", -1.0)
        assert_near_equal(f(x), (x[0]-3.0)**2 + x[0]*x1 + (x1+4.0)**2 - 3.0)

        x[0] = 9.0
        # x[1] = 10.0
        y = f.create_output_vector()
        f(x, y=y)
        assert_near_equal(y, (x[0]-3.0)**2 + x[0]*x1 + (x1+4.0)**2 - 3.0)
        # Setting a value that conflicts with what's passed to the functional interface should be overwritten.
        prob.set_val("x", -1.0)
        f(x, y=y)
        assert_near_equal(y, (x[0]-3.0)**2 + x[0]*x1 + (x1+4.0)**2 - 3.0)

    def test_f_just_y(self):
        prob = self._prob
        x0 = 3.0
        prob.set_val("x", x0)
        f = prob.get_callback("f", input_vars=["y"], output_vars=["f_xy"])
        x = f.create_input_vector()

        x[0] = 8.0
        assert_near_equal(f(x), (x0-3.0)**2 + x0*x[0] + (x[0]+4.0)**2 - 3.0)
        # Setting a value that conflicts with what's passed to the functional interface should be overwritten.
        prob.set_val("y", -1.0)
        assert_near_equal(f(x), (x0-3.0)**2 + x0*x[0] + (x[0]+4.0)**2 - 3.0)

        x[0] = 9.0
        y = f.create_output_vector()
        f(x, y=y)
        assert_near_equal(y, (x0-3.0)**2 + x0*x[0] + (x[0]+4.0)**2 - 3.0)
        # Setting a value that conflicts with what's passed to the functional interface should be overwritten.
        prob.set_val("y", -1.0)
        f(x, y=y)
        assert_near_equal(y, (x0-3.0)**2 + x0*x[0] + (x[0]+4.0)**2 - 3.0)

    def test_dfdx(self):
        prob = self._prob
        dfdx = prob.get_callback("dfdx")
        x = dfdx.create_input_vector()

        x[0] = 1.5
        x[1] = 2.5
        J = dfdx(x)
        J_expected = np.array([2*(x[0]-3.0) + x[1], x[0] + 2*(x[1] + 4.0)]).reshape(1, 2)
        assert_near_equal(J, J_expected)

        J = dfdx.create_jacobian_matrix()
        x[0] = 1.6
        x[1] = 2.6
        dfdx(x, J=J)
        J_expected = np.array([2*(x[0]-3.0) + x[1], x[0] + 2*(x[1] + 4.0)]).reshape(1, 2)
        assert_near_equal(J, J_expected)



if __name__ == "__main__":
    unittest.main()
