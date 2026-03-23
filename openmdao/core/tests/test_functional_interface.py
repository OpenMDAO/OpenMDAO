import sys
import unittest

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs


# Set the threshold to sys.maxsize to print all elements
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)


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


def _get_multid_circle_problem():
    SHAPE = (2, 2, 2, 2)
    SIZE = np.prod(SHAPE)

    prob = om.Problem()

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

    # Messing with the refs here to make sure we get unscaled values for the Jacobian in the tests.
    prob.model.add_design_var('x', ref=1.23)
    prob.model.add_design_var('y', ref=2.34)
    prob.model.add_design_var('r', lower=.5, upper=10, ref=3.45)

    # nonlinear constraints
    prob.model.add_constraint('r_con.g', equals=0, ref=2.31)

    prob.model.add_constraint('theta_con.g', lower=-1e-5, upper=1e-5, indices=EVEN_IND, ref=2.2)
    prob.model.add_constraint('delta_theta_con.g', lower=-1e-5, upper=1e-5, ref=1.1)

    # this constrains x[0] to be 1 (see definition of l_conx)
    prob.model.add_constraint('l_conx.g', equals=0, linear=False, indices=[0,], flat_indices=True, ref=1.2)

    # linear constraint
    prob.model.add_constraint('y', equals=0, indices=[0,], linear=True, flat_indices=True, ref=2.4)

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

    return SHAPE, SIZE, ALL_IND, EVEN_IND, ODD_IND, prob


@use_tempdirs
class TestMultiDimensionalCircleOptimization(unittest.TestCase):

    def setUp(self):
        self._SHAPE, self._SIZE, self._ALL_IND, self._EVEN_IND, self._ODD_IND, self._prob = _get_multid_circle_problem()
        self._prob.run_driver()

    def test_f(self):
        SHAPE = self._SHAPE
        SIZE = self._SIZE
        ALL_IND = self._ALL_IND
        prob = self._prob

        # This says I want `x[0, 0, 0, 0]` and `x[1, 1, 1, 1]`, in that order.
        x_indices = ([0, 1], [0, 1], [0, 1], [0, 1])

        # This says I want `y[0, 1, 0, 1]` and `y[1, 0, 1, 0]`, in that order.
        y_indices = ([0, 1], [1, 0], [0, 1], [1, 0])

        f = prob.get_callback("f",
                              input_vars=[
                                  {
                                   "x": {"indices": x_indices},
                                   "y": {"indices": y_indices},
                                  },
                                  "r",
                              ],
                              output_vars=["circle.area", "r_con.g"])

        # First, check that the indices worked properly.
        x0 = f.create_input_vector()
        y0 = f(x0)

        # Check the `x` input var.
        x_expected = prob.get_val("x", indices=x_indices)
        x_start, x_end = f._input_metadata["x"]["offsets"]
        assert_near_equal(x0[x_start:x_end], x_expected)
        assert_near_equal(f.get_input_val("x"), x_expected)

        # Check the `y` input var.
        y_expected = prob.get_val("y", indices=y_indices)
        y_start, y_end = f._input_metadata["y"]["offsets"]
        assert_near_equal(x0[y_start:y_end], y_expected)
        assert_near_equal(f.get_input_val("y"), y_expected)

        # Check the `r` input var.
        r_expected = prob.get_val("r")
        r_start, r_end = f._input_metadata["r"]["offsets"]
        assert_near_equal(x0[r_start:r_end], r_expected)
        assert_near_equal(f.get_input_val("r"), r_expected)

        # Check the `circle.area` output var.
        area_expected = prob.get_val("circle.area")
        area_start, area_end = f._output_metadata["circle.area"]["offsets"]
        assert_near_equal(y0[area_start:area_end], area_expected)
        assert_near_equal(f.get_output_val("circle.area"), area_expected)

        # Check the `r_con.g` output var.
        rcong_expected = prob.get_val("r_con.g").flatten()
        rcong_start, rcong_end = f._output_metadata["r_con.g"]["offsets"]
        assert_near_equal(y0[rcong_start:rcong_end], rcong_expected)
        assert_near_equal(f.get_output_val("r_con.g"), rcong_expected)

        # Make sure the in-place form gets the same answer.
        y02 = np.zeros_like(y0)
        f(x0, y=y02)
        assert_near_equal(y02, y0)

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

        # First, check that the indices worked properly.
        x0 = dfdx.create_input_vector()
        y0 = dfdx.create_output_vector()

        # Check the `x` input var.
        x_expected = prob.get_val("x", indices=x_indices)
        x_start, x_end = dfdx._input_metadata["x"]["offsets"]
        assert_near_equal(x0[x_start:x_end], x_expected)
        assert_near_equal(dfdx.get_input_val("x"), x_expected)

        # Check the `y` input var.
        y_expected = prob.get_val("y", indices=y_indices)
        y_start, y_end = dfdx._input_metadata["y"]["offsets"]
        assert_near_equal(x0[y_start:y_end], y_expected)
        assert_near_equal(dfdx.get_input_val("y"), y_expected)

        # Check the `r` input var.
        r_expected = prob.get_val("r")
        r_start, r_end = dfdx._input_metadata["r"]["offsets"]
        assert_near_equal(x0[r_start:r_end], r_expected)
        assert_near_equal(dfdx.get_input_val("r"), r_expected)

        # Check the `circle.area` output var.
        area_expected = prob.get_val("circle.area")
        area_start, area_end = dfdx._output_metadata["circle.area"]["offsets"]
        assert_near_equal(y0[area_start:area_end], area_expected)
        assert_near_equal(dfdx.get_output_val("circle.area"), area_expected)

        # Check the `r_con.g` output var.
        rcong_expected = prob.get_val("r_con.g").flatten()
        rcong_start, rcong_end = dfdx._output_metadata["r_con.g"]["offsets"]
        assert_near_equal(y0[rcong_start:rcong_end], rcong_expected)
        assert_near_equal(dfdx.get_output_val("r_con.g"), rcong_expected)

        # Now check that we get the correct derivatives.
        J = dfdx(x0)

        # Derivative of area wrt x is zero.
        darea_dx_expected = np.zeros((area_expected.size, x_expected.size))
        assert_near_equal(J[area_start:area_end, x_start:x_end], darea_dx_expected)

        # Derivative of area wrt y is zero.
        darea_dy_expected = np.zeros((area_expected.size, y_expected.size))
        assert_near_equal(J[area_start:area_end, y_start:y_end], darea_dy_expected)

        # area = pi*r**2, so darea_dr = 2*pi*r.
        darea_dr_expected = np.zeros((area_expected.size, r_expected.size))
        darea_dr_expected[:, :] = 2*np.pi*r_expected
        assert_near_equal(J[area_start:area_end, r_start:r_end], darea_dr_expected)

        # Derivative of `r_con.g` wrt `x`.
        drcong_dx_expected = np.zeros((rcong_expected.size, x_expected.size))
        for i in range(rcong_expected.size):
            for j in range(x_expected.size):
                # So, we know that this sub-jacobian is diagonal-ish.
                # This will give me the current index for the entire `x`, not the sub-array defined by `x_indices`.
                x_idx = tuple(idx[j] for idx in x_indices)
                # Now we need to convert the multi-index into a flattened index.
                x_idx_flat = np.ravel_multi_index(x_idx, SHAPE)
                # Now, for this diagonal Jacobian, if the x flat index is the same as the r_con.g index, then it's non-zero.
                if i == x_idx_flat:
                    drcong_dx_expected[i, j] = 2*x_expected[j]
        assert_near_equal(J[rcong_start:rcong_end, x_start:x_end], drcong_dx_expected)

        # Derivative of `r_con.g` wrt `y`.
        drcong_dy_expected = np.zeros((rcong_expected.size, y_expected.size))
        for i in range(rcong_expected.size):
            for j in range(y_expected.size):
                # So, we know that this sub-jacobian is diagonal-ish.
                # This will give me the current index for the entire `y`, not the sub-array defined by `y_indices`.
                y_idx = tuple(idx[j] for idx in y_indices)
                # Now we need to convert the multi-index into a flattened index.
                y_idx_flat = np.ravel_multi_index(y_idx, SHAPE)
                # Now, for this diagonal Jacobian, if the y flat index is the same as the r_con.g index, then it's non-zero.
                if i == y_idx_flat:
                    drcong_dy_expected[i, j] = 2*y_expected[j]
        assert_near_equal(J[rcong_start:rcong_end, y_start:y_end], drcong_dy_expected)

        # Derivative of `r_con.g` wrt `r`.
        drcong_dr_expected = np.zeros((rcong_expected.size, r_expected.size))
        for i in range(rcong_expected.size):
            for j in range(r_expected.size):
                drcong_dr_expected[i, j] = -2*r_expected[j]
        assert_near_equal(J[rcong_start:rcong_end, r_start:r_end], drcong_dr_expected)

        # Test that the in-place Jacobian call gets the same answer.
        J2 = dfdx.create_jacobian_matrix()
        dfdx(x0, J=J2)
        assert_near_equal(J2, J)

    def test_fdfdx(self):
        SHAPE = self._SHAPE
        SIZE = self._SIZE
        prob = self._prob

        # This says I want `x[0, 0, 0, 0]` and `x[1, 1, 1, 1]`, in that order.
        x_indices = ([0, 1], [0, 1], [0, 1], [0, 1])

        # This says I want `y[0, 1, 0, 1]` and `y[1, 0, 1, 0]`, in that order.
        y_indices = ([0, 1], [1, 0], [0, 1], [1, 0])

        fdfdx = prob.get_callback("fdfdx",
                                 input_vars=[
                                     {
                                      "x": {"indices": x_indices},
                                      "y": {"indices": y_indices},
                                     },
                                     "r",
                                 ],
                                 output_vars=["circle.area", "r_con.g"])

        # First, check that the indices worked properly.
        x0 = fdfdx.create_input_vector()

        y0, J = fdfdx(x0)

        # Check the `x` input var.
        x_expected = prob.get_val("x", indices=x_indices)
        x_start, x_end = fdfdx._input_metadata["x"]["offsets"]
        assert_near_equal(x0[x_start:x_end], x_expected)
        assert_near_equal(fdfdx.get_input_val("x"), x_expected)

        # Check the `y` input var.
        y_expected = prob.get_val("y", indices=y_indices)
        y_start, y_end = fdfdx._input_metadata["y"]["offsets"]
        assert_near_equal(x0[y_start:y_end], y_expected)
        assert_near_equal(fdfdx.get_input_val("y"), y_expected)

        # Check the `r` input var.
        r_expected = prob.get_val("r")
        r_start, r_end = fdfdx._input_metadata["r"]["offsets"]
        assert_near_equal(x0[r_start:r_end], r_expected)
        assert_near_equal(fdfdx.get_input_val("r"), r_expected)

        # Check the `circle.area` output var.
        area_expected = prob.get_val("circle.area")
        area_start, area_end = fdfdx._output_metadata["circle.area"]["offsets"]
        assert_near_equal(y0[area_start:area_end], area_expected)
        assert_near_equal(fdfdx.get_output_val("circle.area"), area_expected)

        # Check the `r_con.g` output var.
        rcong_expected = prob.get_val("r_con.g").flatten()
        rcong_start, rcong_end = fdfdx._output_metadata["r_con.g"]["offsets"]
        assert_near_equal(y0[rcong_start:rcong_end], rcong_expected)
        assert_near_equal(fdfdx.get_output_val("r_con.g"), rcong_expected)

        # Now check that we get the correct derivatives.

        # Derivative of area wrt x is zero.
        darea_dx_expected = np.zeros((area_expected.size, x_expected.size))
        assert_near_equal(J[area_start:area_end, x_start:x_end], darea_dx_expected)

        # Derivative of area wrt y is zero.
        darea_dy_expected = np.zeros((area_expected.size, y_expected.size))
        assert_near_equal(J[area_start:area_end, y_start:y_end], darea_dy_expected)

        # area = pi*r**2, so darea_dr = 2*pi*r.
        darea_dr_expected = np.zeros((area_expected.size, r_expected.size))
        darea_dr_expected[:, :] = 2*np.pi*r_expected
        assert_near_equal(J[area_start:area_end, r_start:r_end], darea_dr_expected)

        # Derivative of `r_con.g` wrt `x`.
        drcong_dx_expected = np.zeros((rcong_expected.size, x_expected.size))
        for i in range(rcong_expected.size):
            for j in range(x_expected.size):
                # So, we know that this sub-jacobian is diagonal-ish.
                # This will give me the current index for the entire `x`, not the sub-array defined by `x_indices`.
                x_idx = tuple(idx[j] for idx in x_indices)
                # Now we need to convert the multi-index into a flattened index.
                x_idx_flat = np.ravel_multi_index(x_idx, SHAPE)
                # Now, for this diagonal Jacobian, if the x flat index is the same as the r_con.g index, then it's non-zero.
                if i == x_idx_flat:
                    drcong_dx_expected[i, j] = 2*x_expected[j]
        assert_near_equal(J[rcong_start:rcong_end, x_start:x_end], drcong_dx_expected)

        # Derivative of `r_con.g` wrt `y`.
        drcong_dy_expected = np.zeros((rcong_expected.size, y_expected.size))
        for i in range(rcong_expected.size):
            for j in range(y_expected.size):
                # So, we know that this sub-jacobian is diagonal-ish.
                # This will give me the current index for the entire `y`, not the sub-array defined by `y_indices`.
                y_idx = tuple(idx[j] for idx in y_indices)
                # Now we need to convert the multi-index into a flattened index.
                y_idx_flat = np.ravel_multi_index(y_idx, SHAPE)
                # Now, for this diagonal Jacobian, if the y flat index is the same as the r_con.g index, then it's non-zero.
                if i == y_idx_flat:
                    drcong_dy_expected[i, j] = 2*y_expected[j]
        assert_near_equal(J[rcong_start:rcong_end, y_start:y_end], drcong_dy_expected)

        # Derivative of `r_con.g` wrt `r`.
        drcong_dr_expected = np.zeros((rcong_expected.size, r_expected.size))
        for i in range(rcong_expected.size):
            for j in range(r_expected.size):
                drcong_dr_expected[i, j] = -2*r_expected[j]
        assert_near_equal(J[rcong_start:rcong_end, r_start:r_end], drcong_dr_expected)

        # Test that the in-place Jacobian call gets the same answer.
        J2 = fdfdx.create_jacobian_matrix()
        y02 = np.zeros_like(y0)
        fdfdx(x0, y=y02, J=J2)
        assert_near_equal(y02, y0)
        assert_near_equal(J2, J)

    def test_dfdx_driver_vars(self):
        SHAPE = self._SHAPE
        SIZE = self._SIZE
        EVEN_IND = self._EVEN_IND
        ODD_IND = self._ODD_IND
        prob = self._prob

        dfdx = prob.get_callback("dfdx")

        input_names_expected = ["x", "y", "r"]
        input_names = dfdx.input_var_names
        self.assertEqual(len(input_names), len(input_names_expected))
        self.assertTrue(all(name in input_names_expected for name in input_names))

        # This doesn't include linear constraints, namely the constraint on the first index of y.
        # Not sure what to do about that.
        output_names_expected = ["r_con.g", "theta_con.g", "delta_theta_con.g", "l_conx.g", "circle.area"]
        output_names = dfdx.output_var_names
        self.assertEqual(len(output_names), len(output_names_expected))
        self.assertTrue(all(name in output_names_expected for name in output_names))

        # Now check the variables.
        # First, check that the indices worked properly.
        x0 = dfdx.create_input_vector()
        y0 = dfdx.create_output_vector()

        # Check the `x` input var.
        x_expected = prob.get_val("x").flatten()
        x_start, x_end = dfdx._input_metadata["x"]["offsets"]
        assert_near_equal(x0[x_start:x_end], x_expected)
        assert_near_equal(dfdx.get_input_val("x"), x_expected)

        # Check the `y` input var.
        y_expected = prob.get_val("y").flatten()
        y_start, y_end = dfdx._input_metadata["y"]["offsets"]
        assert_near_equal(x0[y_start:y_end], y_expected)
        assert_near_equal(dfdx.get_input_val("y"), y_expected)

        # Check the `r` input var.
        r_expected = prob.get_val("r")
        r_start, r_end = dfdx._input_metadata["r"]["offsets"]
        assert_near_equal(x0[r_start:r_end], r_expected)
        assert_near_equal(dfdx.get_input_val("r"), r_expected)

        # Check the `r_con.g` output var.
        rcong_expected = prob.get_val("r_con.g").flatten()
        rcong_start, rcong_end = dfdx._output_metadata["r_con.g"]["offsets"]
        assert_near_equal(y0[rcong_start:rcong_end], rcong_expected)
        assert_near_equal(dfdx.get_output_val("r_con.g"), rcong_expected)

        # Check the `theta_con.g` output var.
        thetacong_expected = prob.get_val("theta_con.g", indices=EVEN_IND).flatten()
        thetacong_start, thetacong_end = dfdx._output_metadata["theta_con.g"]["offsets"]
        assert_near_equal(y0[thetacong_start:thetacong_end], thetacong_expected)
        assert_near_equal(dfdx.get_output_val("theta_con.g"), thetacong_expected)

        # Check the `delta_theta_con.g` output var.
        deltathetacong_expected = prob.get_val("delta_theta_con.g").flatten()
        deltathetacong_start, deltathetacong_end = dfdx._output_metadata["delta_theta_con.g"]["offsets"]
        assert_near_equal(y0[deltathetacong_start:deltathetacong_end], deltathetacong_expected)
        assert_near_equal(dfdx.get_output_val("delta_theta_con.g"), deltathetacong_expected)

        # Check the `l_conx.g` output var.
        lconxg_expected = prob.get_val("l_conx.g", indices=([0], [0], [0], [0]))
        lconxg_start, lconxg_end = dfdx._output_metadata["l_conx.g"]["offsets"]
        assert_near_equal(y0[lconxg_start:lconxg_end], lconxg_expected)
        assert_near_equal(dfdx.get_output_val("l_conx.g"), lconxg_expected)

        # Check the `circle.area` output var.
        area_expected = prob.get_val("circle.area")
        area_start, area_end = dfdx._output_metadata["circle.area"]["offsets"]
        assert_near_equal(y0[area_start:area_end], area_expected)
        assert_near_equal(dfdx.get_output_val("circle.area"), area_expected)

        # Now check that we get the correct derivatives.
        J = dfdx(x0)

        # # Derivative of `r_con.g` wrt `x`.
        # drcong_dx_expected = np.zeros((rcong_expected.size, x_expected.size))
        # for i in range(rcong_expected.size):
        #     for j in range(x_expected.size):
        #         if i == j:
        #             drcong_dx_expected[i, j] = 2*x_expected[j]
        # assert_near_equal(J[rcong_start:rcong_end, x_start:x_end], drcong_dx_expected)

        # # Derivative of `r_con.g` wrt `y`.
        # drcong_dy_expected = np.zeros((rcong_expected.size, y_expected.size))
        # for i in range(rcong_expected.size):
        #     for j in range(y_expected.size):
        #         if i == j:
        #             drcong_dy_expected[i, j] = 2*y_expected[j]
        # assert_near_equal(J[rcong_start:rcong_end, y_start:y_end], drcong_dy_expected)

        # # Derivative of `r_con.g` wrt `r`.
        # drcong_dr_expected = np.zeros((rcong_expected.size, r_expected.size))
        # for i in range(rcong_expected.size):
        #     for j in range(r_expected.size):
        #         drcong_dr_expected[i, j] = -2*r_expected[j]
        # assert_near_equal(J[rcong_start:rcong_end, r_start:r_end], drcong_dr_expected)

        # dtheta_actual_dx = np.zeros((SIZE, x_expected.size))
        # for i in range(dtheta_actual_dx.size):
        #     for j in range(x_expected.size):
        #         if i == j:
        #             dtheta_actual_dx[i, j] = -y0[i]/(x0[i]**2 + y0[i]**2)

        # Easier if I have the `x` and `y` values in the multidimensional shape.
        x = prob.get_val("x")
        y = prob.get_val("y")
        r = prob.get_val("r")
        theta_actual = np.arctan2(y, x)
        even = theta_actual[EVEN_IND]
        odd = theta_actual[ODD_IND]
        EVEN_SHAPE = even.shape
        ODD_SHAPE = odd.shape
        EVEN_SIZE = even.size
        ODD_SIZE = odd.size

        # Get the derivative of theta_actual wrt x.
        dtheta_actual_dx = np.zeros((*SHAPE, *SHAPE))
        for j in np.ndindex(SHAPE):
            for i in np.ndindex(SHAPE):
                if i == j:
                    dtheta_actual_dx[(*i, *j)] = -y[j]/(x[j]**2 + y[j]**2)

        # Get the derivative of theta_actual wrt x.
        dtheta_actual_dy = np.zeros((*SHAPE, *SHAPE))
        for j in np.ndindex(SHAPE):
            for i in np.ndindex(SHAPE):
                if i == j:
                    dtheta_actual_dy[(*i, *j)] = x[j]/(x[j]**2 + y[j]**2)

        # Now get the derivative of even wrt theta_actual.
        # even is just the even-indexed entries of theta_actual.
        # So we should be able to build up an array that does that.
        # But how do we decide?
        # Hmm... I guess i already have the even indices...
        # So I could iterate over those, which would give me, in order, the even indices I want.
        # Check that all the indices in EVEN_IND have the same shape.
        self.assertTrue(all([idx.shape == EVEN_IND[0].shape for idx in EVEN_IND]))
        # EVEN_SHAPE = EVEN_IND[0].shape
        deven_dtheta_actual = np.zeros((*EVEN_SHAPE, *SHAPE))
        # So, I think EVEN_IND is a tuple of ndarrays, which each ndarray having the the SHAPE of even.
        # So doing `for j in zip(*EVEN_IND)` should give me a tuple `j` for each even index.
        # Oh, no, maybe not, because iterating over ndarrays gives you a slice of all but the first dimension.
        # Hmm... so what do I want to do?
        #   I could iterate over each index in EVEN_IND
        #   which would give me `(i, j, k, l)` for each even index of theta_actual.
        # But I also need to keep track of the indices for the output, even.
        # So that's indicated by the index of EVEN index.
        # So I could try `for js, vs in zip([np.ndenumerate(idxs) for idxs in EVEN_IND]):` maybe?
        # What would that do?
        # Well, `np.ndenumerate(idx)` gives you the tuple index and value for the array `idx`.
        # So, for one entry of `EVEN_IND` I would get the tuple multi-index that corresponds to the even output index, and the input index for one dimension of `theta_actual`.
        # And then doing that zip + list comprehension would give me that for all dimensions of theta_actual.
        # But they would be combined somehow.
        # How would that be?
        # Well, each evaluation of the zip will give me a tuple of the output of each `np.ndenumerate`.
        # And each np.ndenumerate will give a tuple of (idx, val).
        # So the total thing will be ((idx_out0, idx_in0), (idx_out1, idx_in1), (idx_out2, idx_in2), (idx_out3, idx_in3)).
        # But all the `idx_out*` tuples will be the same.
        for idx_out_idx_in in zip(*[np.ndenumerate(idxs) for idxs in EVEN_IND]):
            # The total input index will be
            idx_in = tuple(io_ii[1] for io_ii in idx_out_idx_in)
            # The total output index is any of the first entries in idx_out_idx_in.
            idx_out = idx_out_idx_in[0][0]
            deven_dtheta_actual[(*idx_out, *idx_in)] = 1.0

        # Now need to do the same thing for the odd indices.
        # ODD_SHAPE = ODD_IND[0].shape
        dodd_dtheta_actual = np.zeros((*ODD_SHAPE, *SHAPE))
        for idx_out_idx_in in zip(*[np.ndenumerate(idxs) for idxs in ODD_IND]):
            # The total input index will be
            idx_in = tuple(io_ii[1] for io_ii in idx_out_idx_in)
            # The total output index is any of the first entries in idx_out_idx_in.
            idx_out = idx_out_idx_in[0][0]
            dodd_dtheta_actual[(*idx_out, *idx_in)] = 1.0

        # OK, so, since `delta_theta_con.g` = even - odd, then:
        # Get the derivative of theta_actual wrt x.
        ddelta_theta_con_g_deven = np.zeros((*EVEN_SHAPE, *EVEN_SHAPE))
        for j in np.ndindex(EVEN_SHAPE):
            for i in np.ndindex(EVEN_SHAPE):
                if i == j:
                    ddelta_theta_con_g_deven[(*i, *j)] = 1.0
        ddelta_theta_con_g_dodd = np.zeros((*ODD_SHAPE, *ODD_SHAPE))
        for j in np.ndindex(ODD_SHAPE):
            for i in np.ndindex(ODD_SHAPE):
                if i == j:
                    ddelta_theta_con_g_dodd[(*i, *j)] = -1.0

        # So now I should be able to chain rule all this.
        # But I need to reshape things for matrix multiplication to do the right thing for the chain rule.
        dtheta_actual_dx.shape = (SIZE, SIZE)
        deven_dtheta_actual.shape = (EVEN_SIZE, SIZE)
        dodd_dtheta_actual.shape = (ODD_SIZE, SIZE)
        ddelta_theta_con_g_deven.shape = (EVEN_SIZE, EVEN_SIZE)
        ddelta_theta_con_g_dodd.shape = (ODD_SIZE, ODD_SIZE)

        ddelta_theta_con_g_dx = ddelta_theta_con_g_deven @ deven_dtheta_actual @ dtheta_actual_dx + ddelta_theta_con_g_dodd @ dodd_dtheta_actual @ dtheta_actual_dx
        assert_near_equal(J[deltathetacong_start:deltathetacong_end, x_start:x_end], ddelta_theta_con_g_dx)

        # Now do the same thing for the derivative of delta_theta_con_g wrt y.
        dtheta_actual_dy.shape = (SIZE, SIZE)
        ddelta_theta_con_g_dy = ddelta_theta_con_g_deven @ deven_dtheta_actual @ dtheta_actual_dy + ddelta_theta_con_g_dodd @ dodd_dtheta_actual @ dtheta_actual_dy
        assert_near_equal(J[deltathetacong_start:deltathetacong_end, y_start:y_end], ddelta_theta_con_g_dy)

        # delta_theta_con_g is not a function of r, so that derivative should be zero.
        ddelta_theta_con_g_dr = np.zeros((EVEN_SIZE, 1))
        assert_near_equal(J[deltathetacong_start:deltathetacong_end, r_start:r_end], ddelta_theta_con_g_dr)

        # OK, so now I want the derivatives of `r_con.g` wrt `x`, `y`, `r`.
        dr_con_g_dx = np.zeros((*SHAPE, *SHAPE))
        for j in np.ndindex(SHAPE):
            for i in np.ndindex(SHAPE):
                if i == j:
                    dr_con_g_dx[(*i, *j)] = 2*x[j]
        # Need to reshape into a matrix.
        dr_con_g_dx.shape = (SIZE, SIZE)
        assert_near_equal(J[rcong_start:rcong_end, x_start:x_end], dr_con_g_dx)

        dr_con_g_dy = np.zeros((*SHAPE, *SHAPE))
        for j in np.ndindex(SHAPE):
            for i in np.ndindex(SHAPE):
                if i == j:
                    dr_con_g_dy[(*i, *j)] = 2*y[j]
        # Need to reshape into a matrix.
        dr_con_g_dy.shape = (SIZE, SIZE)
        assert_near_equal(J[rcong_start:rcong_end, y_start:y_end], dr_con_g_dy)

        dr_con_g_dr = np.zeros((*SHAPE, *r.shape))
        for j in np.ndindex(*r.shape):
            for i in np.ndindex(*SHAPE):
                dr_con_g_dr[(*i, *j)] = -2*r[j]
        # Need to reshape into a matrix.
        dr_con_g_dr.shape = (SIZE, r.size)
        assert_near_equal(J[rcong_start:rcong_end, r_start:r_end], dr_con_g_dr)

        dtheta_con_g_even_dtheta_actual = np.zeros((*EVEN_SHAPE, *SHAPE))
        for idx_out_idx_in in zip(*[np.ndenumerate(idxs) for idxs in EVEN_IND]):
            # The total input index will be
            idx_in = tuple(io_ii[1] for io_ii in idx_out_idx_in)
            # The total output index is any of the first entries in idx_out_idx_in.
            idx_out = idx_out_idx_in[0][0]
            dtheta_con_g_even_dtheta_actual[(*idx_out, *idx_in)] = 1.0
        dtheta_con_g_even_dtheta_actual.shape = (EVEN_SIZE, SIZE)
        # Now do the chain rule to get the derivatives wrt x, y, z.

        # dtheta_actual_dr is just zero.
        dtheta_actual_dr = np.zeros((SIZE, r.size))

        dtheta_con_g_dx = dtheta_con_g_even_dtheta_actual @ dtheta_actual_dx
        dtheta_con_g_dy = dtheta_con_g_even_dtheta_actual @ dtheta_actual_dy
        dtheta_con_g_dr = dtheta_con_g_even_dtheta_actual @ dtheta_actual_dr

        assert_near_equal(J[thetacong_start:thetacong_end, x_start:x_end], dtheta_con_g_dx)
        assert_near_equal(J[thetacong_start:thetacong_end, y_start:y_end], dtheta_con_g_dy)
        assert_near_equal(J[thetacong_start:thetacong_end, r_start:r_end], dtheta_con_g_dr)

        # Next, get the derivative of `l_conx.g`.
        dl_conx_g_0_dx = np.zeros((1, *SHAPE))
        # So, the indices I want are just the first one.
        ALL_IND = self._ALL_IND
        idx0s = tuple(idx.flat[0] for idx in ALL_IND)
        dl_conx_g_0_dx[(0, *idx0s)] = 1.0
        dl_conx_g_0_dx.shape = (1, SIZE)
        assert_near_equal(J[lconxg_start:lconxg_end, x_start:x_end], dl_conx_g_0_dx)

        # Other derivatives of l_conx.g are zero.
        dl_conx_g_0_dy = np.zeros((1, *SHAPE))
        dl_conx_g_0_dy.shape = ((1, SIZE))
        dl_conx_g_0_dr = np.zeros((1, 1))
        assert_near_equal(J[lconxg_start:lconxg_end, y_start:y_end], dl_conx_g_0_dy)
        assert_near_equal(J[lconxg_start:lconxg_end, r_start:r_end], dl_conx_g_0_dr)

        # Next derivative: circle.area = pi*r**2, so derivative is just 2*pi*r of course.
        dcircle_area_dr = np.zeros((1, 1))
        dcircle_area_dr[0, 0] = 2*np.pi*r_expected[0]

        dcircle_area_dx = np.zeros((1, SIZE))
        dcircle_area_dy = np.zeros((1, SIZE))
        assert_near_equal(J[area_start:area_end, x_start:x_end], dcircle_area_dx)
        assert_near_equal(J[area_start:area_end, y_start:y_end], dcircle_area_dy)
        assert_near_equal(J[area_start:area_end, r_start:r_end], dcircle_area_dr)

    def test_linear_constraint_check(self):
        prob = self._prob

        dfdx = prob.get_callback("dfdx", input_vars=["y", "x"], output_vars=["y", "x"])

        input_names_expected = ["x", "y"]
        input_names = dfdx.input_var_names
        self.assertEqual(len(input_names), len(input_names_expected))
        self.assertTrue(all(name in input_names_expected for name in input_names))

        # This doesn't include linear constraints.
        # Not sure what to do about that.
        output_names_expected = ["x", "y"]
        output_names = dfdx.output_var_names
        self.assertEqual(len(output_names), len(output_names_expected))
        self.assertTrue(all(name in output_names_expected for name in output_names))

        x0 = dfdx.create_input_vector()
        # Why does this have a dense column in the first one?
        # print(dfdx(x0))

if __name__ == "__main__":
    unittest.main()
