"""
Unit test for scipy interpolator. Note that an api is defined for standalone use.
"""
from __future__ import division, print_function, absolute_import
from copy import deepcopy
import unittest

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_allclose, assert_array_equal, assert_equal)

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error

scipy_gte_019 = True
try:
    from scipy.interpolate._bsplines import make_interp_spline
except ImportError:
    scipy_gte_019 = False

if scipy_gte_019:
    from openmdao.components.structured_metamodel_util.scipy_interp import ScipyGridInterp
    from openmdao.components.structured_metamodel_util.outofbounds_error import OutOfBoundsError


def rel_error(actual, computed):
    return np.linalg.norm(actual - computed) / np.linalg.norm(actual)


@unittest.skipIf(not scipy_gte_019, "only run if scipy>=0.19.")
class TestScipyGridInterpolator(unittest.TestCase):
    """Tests the functionality of the scipy grid interpolator."""

    def setUp(self):
        self.interp_configs = {
            "slinear": 1,
            "cubic": 3,
            "quintic": 5,
        }
        self.valid_methods = self.interp_configs.keys()

    def _get_sample_4d_large(self):
        def f(x, y, z, w):
            return x**2 + y**2 + z**2 + w**2
        X = np.linspace(-10, 10, 6)
        Y = np.linspace(-10, 10, 7)
        np.random.seed(0)
        Z = np.random.uniform(-10, 10, 6)
        Z.sort()
        W = np.linspace(-10, 10, 8)
        points = [X, Y, Z, W]
        values = f(*np.meshgrid(*points, indexing='ij'))
        return points, values

    def _get_sample_2d(self):
        # test problem with enough points for smooth spline fits
        def f(u, v):
            return u * np.cos(u * v) + v * np.sin(u * v)

        def df(u, v):
            return (-u * v * np.sin(u * v) + v**2 * np.cos(u * v) +
                    np.cos(u * v),
                    -u**2 * np.sin(u * v) + u * v * np.cos(u * v) +
                    np.sin(u * v))

        # uniformly spaced axis
        u = np.linspace(0, 3, 50)
        # randomly spaced axis
        np.random.seed(7590)
        v = np.random.uniform(0, 3, 33)
        v.sort()

        points = [u, v]
        values = f(*np.meshgrid(*points, indexing='ij'))
        return points, values, f, df

    def test_list_input(self):
        points, values = self._get_sample_4d_large()

        sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             [0.5, 0.5, .5, .5]])

        for method in self.valid_methods:
            interp = ScipyGridInterp(points, values.tolist(), method=method)
            v1 = interp.interpolate(sample.tolist(), compute_gradients=False)

            interp = ScipyGridInterp(points, values, method=method)
            v2 = interp.interpolate(sample, compute_gradients=False)

            assert_allclose(v1, v2)

    def test_auto_reduce_spline_order(self):
        # if a spline method is used and spline_dim_error=False and a dimension
        # does not have enough points, the spline order for that dimension
        # should be automatically reduced
        np.random.seed(314)

        # x dimension is too small for cubic, should fall back to linear
        x = [0, 1]
        y = np.linspace(-10, 4, 10)
        z = np.linspace(1000, 2000, 20)

        points = [x, y, z]
        values = np.random.randn(2, 10, 20)

        # verify that this raises error with dimension checking
        self.assertRaises(ValueError, ScipyGridInterp,
                          points, values, 'cubic')

        interp = ScipyGridInterp(
            points, values, method='cubic', spline_dim_error=False)

        # first dimension (x) should be reduced to k=1 (linear)
        assert_equal(interp._ki[0], 1)

        # should operate as normal
        x = [0.5, 0, 1001]
        result = interp.interpolate(x)
        assert_almost_equal(result, -0.046325695741704434, decimal=5)

        interp = ScipyGridInterp(
            points, values, method='slinear', spline_dim_error=False)

        value1 = interp.interpolate(x)
        # cycle through different methods that require order reduction
        # in the first dimension
        value2 = interp.interpolate(x, method='quintic')
        interp.gradient(x, method='quintic')
        value3 = interp.interpolate(x, method='cubic')
        interp.gradient(x, method='cubic')
        # use default method again
        value4 = interp.interpolate(x)

        # values from different methods should be different
        self.assertRaises(AssertionError, assert_equal, value1, value2)
        self.assertRaises(AssertionError, assert_equal, value2, value3)

        # first value should match last with no side effects from the
        # order reduction or gradient caluclations
        assert_equal(value1, value4)

    def test_complex_exception_spline(self):
        points, values = self._get_sample_4d_large()
        values = values - 2j * values
        sample = np.asarray([[0.1, 0.1, 1., .9]])

        # spline methods dont support complex values
        for method in self.valid_methods:
            self.assertRaises(ValueError, ScipyGridInterp, points, values,
                              method)

    def test_minimum_required_gridsize(self):
        for method in self.valid_methods:
            k = self.interp_configs[method]
            x = np.linspace(0, 1, k)
            y = np.linspace(0, 1, k)
            points = [x, y]
            X, Y = np.meshgrid(*points, indexing='ij')
            values = X + Y
            self.assertRaises(ValueError, ScipyGridInterp, points, values,
                              method)

    def test_method_switching(self):
        # should be able to switch interpolation methods on each __call__
        # and gradient call, without overriding defaults permenantly.
        # exceptions and gradient caching should work as expected.

        np.random.seed(314)
        x = np.linspace(-100, 2, 10)
        y = np.linspace(-10, 4, 6)
        z = np.linspace(1000, 2000, 50)

        points = [x, y, z]
        values = np.random.randn(10, 6, 50)

        x = [0.5, 0, 1001]

        # create as cubic
        interp = ScipyGridInterp(
            points, values, method='cubic')

        # value and gradient work as expected
        result1 = interp.interpolate(x)
        gradient1 = interp.gradient(x)
        result_actual_1 = 0.2630309995970872
        result_gradient_1 = np.array([0.22505535, -0.46465198, 0.02523666])

        assert_almost_equal(result1, result_actual_1)
        assert_almost_equal(gradient1, result_gradient_1)

        # changing the method should work as expected
        result2 = interp.interpolate(x, method='slinear')
        gradient2 = interp.gradient(x, method='slinear')
        result_actual_2 = 0.27801704674026684
        result_gradient_2 = np.array([0.12167214, -0.44221416, -0.00323078])

        assert_almost_equal(result2, result_actual_2)
        assert_almost_equal(gradient2, result_gradient_2)

        # should be able to switch back and get the original results without
        # explicitly setting the method
        result3 = interp.interpolate(x)
        gradient3 = interp.gradient(x)
        assert_almost_equal(result3, result_actual_1)
        assert_almost_equal(gradient3, result_gradient_1)

        # new interpolator and evaluation point
        interp = ScipyGridInterp(points, values, method='slinear')

        # values will be cast to float for splines/gradient methods
        # otherwise, will get null vector gradient [0,0,0] at all pts
        x = [-50, 0, 1501]
        result6 = interp.interpolate(x)
        result_actual_6 = 0.3591176338294626
        assert_almost_equal(result6, result_actual_6)

        # should be able to switch and get value and gradient
        result7 = interp.interpolate(x, method='quintic')
        gradient7 = interp.gradient(x, method='quintic')
        result_actual_7 = 0.6157594079479937
        result_gradient_7 = np.array([-0.35731922, 0.23131539, -0.14088582])
        assert_almost_equal(result7, result_actual_7)
        assert_almost_equal(gradient7, result_gradient_7)

        # switch again; gradient should be different
        gradient8 = interp.gradient(x, method='slinear')
        result_gradient_8 = np.array([-0.11299396, 0.24352342, -0.07446338])
        assert_almost_equal(gradient8, result_gradient_8)

        # should be able to switch back to original without setting it
        result9 = interp.interpolate(x)
        assert_almost_equal(result9, result6)

    def test_spline_deriv_xi1d(self):
        # tests gradient values
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(1234)
        test_pt = np.random.uniform(0, 3, 2)
        actual = np.array(df(*test_pt))
        tol = 1e-1
        for method in self.valid_methods:
            if method == 'slinear':
                tol = 1.5
            interp = ScipyGridInterp(points, values, method)
            computed = interp.gradient(test_pt)
            r_err = rel_error(actual, computed)
            assert r_err < tol

            # test that gradients have been cached
            assert_array_equal(interp._xi, test_pt)
            assert_array_equal(
                interp._all_gradients.flatten(), computed.flatten())

    def test_gradients_returned_by_xi(self):
        # verifies that gradients with respect to xi are returned if cached
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(4321)
        for method in self.valid_methods:
            interp = ScipyGridInterp(points, values, method)
            x = np.array([0.9, 0.1])
            interp._xi = x
            interp._g_order = method
            dy = np.array([0.997901, 0.08915])
            interp._all_gradients = dy
            assert_almost_equal(interp.gradient(x), dy)

    def test_spline_xi1d(self):
        # test interpolated values
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(1)
        test_pt = np.random.uniform(0, 3, 2)
        actual = func(*test_pt)
        for method in self.valid_methods:
            if method == 'slinear':
                tol = 0.5
            else:
                tol = 1e-2
            interp = ScipyGridInterp(points, values, method)
            computed = interp.interpolate(test_pt, compute_gradients=False)
            r_err = rel_error(actual, computed)
            assert r_err < tol

    def test_spline_out_of_bounds_extrap(self):
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(5)
        test_pt = np.random.uniform(3, 3.1, 2)
        actual = func(*test_pt)
        gradient = np.array(df(*test_pt))
        for method in self.valid_methods:
            k = self.interp_configs[method]
            if method == 'slinear':
                tol = 2
            else:
                tol = 1e-1
            interp = ScipyGridInterp(points, values, method,
                                        bounds_error=False,
                                        fill_value=None)
            computed = interp.interpolate(test_pt)
            computed_grad = interp.gradient(test_pt)
            r_err = rel_error(actual, computed)
            assert r_err < tol

            r_err = rel_error(gradient, computed_grad)
            # extrapolated gradients are even trickier, but usable still
            assert r_err < 2 * tol

    def test_spline_xi3d(self):
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(1)
        test_pt = np.random.uniform(0, 3, 6).reshape(3, 2)
        actual = func(*test_pt.T)
        for method in self.valid_methods:
            tol = 1e-1
            if method == 'slinear':
                tol = 0.5
            interp = ScipyGridInterp(points, values, method)
            computed = interp.interpolate(test_pt, compute_gradients=True)
            r_err = rel_error(actual, computed)
            assert r_err < tol

    def test_out_of_bounds_fill2(self):
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(1)
        test_pt = np.random.uniform(3, 3.1, 2)
        actual = np.asarray([np.nan])
        methods = self.valid_methods
        for method in methods:
            interp = ScipyGridInterp(points, values, method,
                                        bounds_error=False,
                                        fill_value=np.nan)
            computed = interp.interpolate(test_pt, compute_gradients=False)
            assert_array_almost_equal(computed, actual)

    def test_invalid_fill_value(self):
        np.random.seed(1234)
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)
        values = np.random.rand(5, 7)

        # integers can be cast to floats
        ScipyGridInterp((x, y), values, fill_value=1)

        # complex values cannot
        self.assertRaises(ValueError, ScipyGridInterp,
                          (x, y), values, fill_value=1 + 2j)

    def test_NaN_exception(self):
        np.random.seed(1234)
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)
        values = np.random.rand(5, 7)
        interp = ScipyGridInterp((x, y), values)

        with self.assertRaises(OutOfBoundsError) as cm:
            interp.interpolate([1, np.nan])

        err = cm.exception

        self.assertEqual(str(err), 'One of the requested xi contains a NaN')
        self.assertEqual(err.idx, 1)
        self.assertTrue(np.isnan(err.value))
        self.assertEqual(err.lower, 0)
        self.assertEqual(err.upper, 1)

    def test_error_messages(self):
        # For coverage. Most of these errors are probably not reachable in openmdao, but
        # proper unit testing requires them for standalone usage of the Interpolation.
        points, values = self._get_sample_4d_large()

        with self.assertRaises(ValueError) as cm:
            interp = ScipyGridInterp(points, values.tolist(), method='junk')

        msg = ('Method "junk" is not defined. Valid methods are')
        self.assertTrue(str(cm.exception).startswith(msg))

        with self.assertRaises(ValueError) as cm:
            interp = ScipyGridInterp(points, values.tolist()[1])

        msg = ('There are 4 point arrays, but values has 3 dimensions')
        self.assertEqual(str(cm.exception), msg)

        badpoints = deepcopy(points)
        badpoints[0][0] = 55.0
        with self.assertRaises(ValueError) as cm:
            interp = ScipyGridInterp(badpoints, values.tolist())

        msg = ('The points in dimension 0 must be strictly ascending')
        self.assertEqual(str(cm.exception), msg)

        badpoints[0] = np.vstack((np.arange(6), np.arange(6)))
        with self.assertRaises(ValueError) as cm:
            interp = ScipyGridInterp(badpoints, values.tolist())

        msg = ('The points in dimension 0 must be 1-dimensional')
        self.assertEqual(str(cm.exception), msg)

        badpoints[0] = (np.arange(4))
        with self.assertRaises(ValueError) as cm:
            interp = ScipyGridInterp(badpoints, values.tolist())

        msg = ('There are 4 points and 6 values in dimension 0')
        self.assertEqual(str(cm.exception), msg)

        badvalues = np.array(values, dtype=np.complex)
        with self.assertRaises(ValueError) as cm:
            interp = ScipyGridInterp(badpoints, badvalues.tolist())

        msg = ("method 'slinear' does not support complex values.")
        self.assertEqual(str(cm.exception), msg)

        interp = ScipyGridInterp(points, values.tolist())
        x = [0.5, 0, 0.5, 0.9]

        with self.assertRaises(ValueError) as cm:
            computed = interp.interpolate(x, method='junk')

        msg = ('Method "junk" is not defined. Valid methods are')
        self.assertTrue(str(cm.exception).startswith(msg))

        self.assertEqual(set(interp.methods()), set(["quintic", "cubic", "slinear"]))


if __name__ == "__main__":
    unittest.main()
