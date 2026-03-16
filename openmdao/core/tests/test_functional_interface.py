import sys
import unittest

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
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


@use_tempdirs
class TestMultiDimensionalCircleOptimization(unittest.TestCase):
    def setUp(self):
        SHAPE = self._SHAPE = (2, 2, 2, 2)
        SIZE = self._SIZE = np.prod(SHAPE)

        prob = self._prob = om.Problem()

        prob.model.add_subsystem('arctan_yox', om.ExecComp('theta_actual=arctan2(y, x)', has_diag_partials=True,
                                                        theta_actual=np.ones(SHAPE), x=np.ones(SHAPE), y=np.ones(SHAPE)),
                              promotes_inputs=['x', 'y'])

        prob.model.add_subsystem('circle', om.ExecComp('area=pi*r**2'), promotes_inputs=['r'])

        prob.model.add_subsystem('r_con', om.ExecComp('g=x**2 + y**2 - r**2', has_diag_partials=True,
                                                   g=np.ones(SHAPE), x=np.ones(SHAPE), y=np.ones(SHAPE)),
                              promotes_inputs=['r', 'x', 'y'])

        theta_target = np.linspace(0, np.pi/4, SIZE).reshape(SHAPE)
        prob.model.add_subsystem('theta_con', om.ExecComp('g = theta_actual - theta_target', has_diag_partials=True,
                                                       g=np.ones(SHAPE), theta_actual=np.ones(SHAPE),
                                                       theta_target=theta_target))

        # So this would allow me to go from multi-indices to flat indices:
        FLAT_IND = np.arange(SIZE, dtype=int).reshape(SHAPE)
        # So now FLAT_IND is a ndarray of shape `SHAPE`, with each entry being the flat index for the corresponding entry.
        # Next:
        ALL_IND = np.unravel_index(FLAT_IND, SHAPE)
        # That gives me `len(SHAPE)` arrays of shape SHAPE, which I think are the first, second, third, and fourth indices of the indices for SHAPE.

        # Now, split all the indices into even and odd ones.
        # This assumes that the length of the last dimension is even.
        EVEN_IND = tuple([inds[..., 0::2] for inds in ALL_IND])
        ODD_IND = tuple([inds[..., 1::2] for inds in ALL_IND])
        ODD_EVEN_SHAPE = EVEN_IND[0].shape
        prob.model.add_subsystem('delta_theta_con', om.ExecComp('g = even - odd', has_diag_partials=True,
                                                             g=np.ones(ODD_EVEN_SHAPE), even=np.ones(ODD_EVEN_SHAPE),
                                                             odd=np.ones(ODD_EVEN_SHAPE)))

        prob.model.add_subsystem('l_conx', om.ExecComp('g=x-1', has_diag_partials=True, g=np.ones(SHAPE), x=np.ones(SHAPE)),
                              promotes_inputs=['x'])

        prob.model.connect('arctan_yox.theta_actual', 'theta_con.theta_actual')
        prob.model.connect('arctan_yox.theta_actual', 'delta_theta_con.even', src_indices=EVEN_IND)
        prob.model.connect('arctan_yox.theta_actual', 'delta_theta_con.odd', src_indices=ODD_IND)

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['disp'] = True
        prob.driver.options['maxiter'] = 500

        # set up dynamic total coloring here
        prob.driver.declare_coloring()

        prob.model.add_design_var('x')
        prob.model.add_design_var('y')
        prob.model.add_design_var('r', lower=.5, upper=10)

        # nonlinear constraints
        prob.model.add_constraint('r_con.g', equals=0)

        prob.model.add_constraint('theta_con.g', lower=-1e-5, upper=1e-5, indices=EVEN_IND)
        prob.model.add_constraint('delta_theta_con.g', lower=-1e-5, upper=1e-5)

        # this constrains x[0] to be 1 (see definition of l_conx)
        prob.model.add_constraint('l_conx.g', equals=0, linear=False, indices=[0,], flat_indices=True)

        # linear constraint
        prob.model.add_constraint('y', equals=0, indices=[0,], linear=True, flat_indices=True)

        prob.model.add_objective('circle.area', ref=-1, index=0, flat_indices=True)

        prob.setup(mode='fwd')

        # This optimization is pretty sensitive to the initial guess.
        # I generated random values and ran it a bunch of times, then hardcoded the initial `x` and `y` values that worked well.
        # x_random = np.random.random(SHAPE)*2 - 1
        # y_random = np.random.random(SHAPE)*2 - 1
        # print(x_random.__repr__())
        # print(y_random.__repr__())
        x_random = np.array(
              [ 0.43909809, -0.05599184, -0.77305987, -0.60332838,  0.70381156,
                0.04907009,  0.28421059, -0.9870356 , -0.62145017, -0.49074292,
                0.78349727, -0.29591685, -0.65462711,  0.50431846, -0.77965126,
               -0.79501575]).reshape(SHAPE)
        y_random = np.array(
              [ 0.07820026,  0.87434394, -0.29007429, -0.45083871, -0.58627707,
                0.45836869,  0.69042418, -0.35848781, -0.04942154, -0.99711998,
               -0.56794221,  0.88148516,  0.79334896, -0.33992779,  0.80233977,
               -0.18448596]).reshape(SHAPE)

        prob.set_val('x', x_random)
        prob.set_val('y', y_random)
        prob.set_val('r', 0.7)

        prob.run_driver()

    def test_dfdx(self):
        SHAPE = self._SHAPE
        SIZE = self._SIZE
        prob = self._prob

        # This says I want `x[0, 0, 0, 0]` and `x[1, 1, 1, 1]`, in that order.
        x_indices = ([0, 1], [0, 1], [0, 1], [0, 1])

        # This says I want `y[0, 1, 0, 1]` and `y[1, 0, 1, 0]`, in that order.
        y_indices = ([0, 1], [1, 0], [0, 1], [1, 0])

        dfdx = prob.get_callback("dfdx",
                                 input_vars=[
                                     {
                                      "x": {"indices": x_indices},
                                      "y": {"indices": y_indices},
                                     },
                                     "r",
                                 ],
                                 output_vars=["circle.area", "r_con.g"])
        x0 = dfdx.create_input_vector()

        # Indexing this stuff is a pain.
        # The functional interface has all the metadata needed to do this for me.
        darea_dx_expected = np.zeros((1, 2))
        assert_near_equal(dfdx(x0)[0:1, 0:2], darea_dx_expected)

        darea_dy_expected = np.zeros((1, 2))
        assert_near_equal(dfdx(x0)[0:1, 2:4], darea_dy_expected)

        darea_dr_expected = 2*np.pi*prob.get_val('r').reshape((1, 1))
        assert_near_equal(dfdx(x0)[0:1, 4:5], -darea_dr_expected)

        # This is tricky.
        # I think I'll just flatten, then assign the correct values.
        x = prob.get_val('x')
        x.shape = (-1,)
        rcong = prob.get_val('r_con.g')
        rcong.shape = (-1,)
        drcong_dx_expected = np.zeros((rcong.size, x.size))
        for i in range(x.size):
            drcong_dx_expected[i, i] = 2*x[i]

        print(dfdx(x0).shape) # Should be (1+SIZE, SIZE+SIZE+1) = (17, 33)
        # Ah, no, that's wrong, since we're indexing part of `x` and `y`.
        assert_near_equal(dfdx(x0)[1:(1+SIZE), 4:(4+SIZE)], drcong_dx_expected)

        # Actually, should I just compare to `prob.compute_totals`?
        # Should think about that.


if __name__ == "__main__":
    unittest.main()
