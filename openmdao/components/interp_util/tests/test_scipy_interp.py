"""
Unit test for scipy interpolator. Note that an api is defined for standalone use.
"""
from __future__ import division, print_function, absolute_import
from copy import deepcopy
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose, \
                          assert_almost_equal, assert_array_equal

from openmdao.utils.assert_utils import assert_rel_error

scipy_gte_019 = True
try:
    from scipy.interpolate._bsplines import make_interp_spline
except ImportError:
    scipy_gte_019 = False

if scipy_gte_019:
    from openmdao.components.interp_util.scipy_interp import ScipyGridInterp
    from openmdao.components.interp_util.outofbounds_error import OutOfBoundsError


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
        self.tol = {
            "slinear": 5e-2,
            "cubic": 1e-4,
            "quintic": 1e-6,
        }

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

        interp = ScipyGridInterp(
            points, values, interp_method='cubic')

        # first dimension (x) should be reduced to k=1 (linear)
        self.assertEqual(interp._ki[0], 1)

        # should operate as normal
        x = np.array([0.5, 0, 1001])
        result = interp.interpolate(x)
        assert_almost_equal(result, -0.046325695741704434, decimal=5)

        interp = ScipyGridInterp(points, values, interp_method='slinear')
        value1 = interp.interpolate(x)

        # cycle through different methods that require order reduction
        # in the first dimension
        interp = ScipyGridInterp(points, values, interp_method='quintic')
        value2 = interp.interpolate(x)
        interp.gradient(x)
        interp = ScipyGridInterp(points, values, interp_method='cubic')
        value3 = interp.interpolate(x)
        interp.gradient(x)

        # values from different methods should be different
        self.assertTrue(value1[0] != value2[0])
        self.assertTrue(value2[0] != value3[0])

    def test_complex_exception_spline(self):
        points, values = self._get_sample_4d_large()
        values = values - 2j * values
        sample = np.asarray([[0.1, 0.1, 1., .9]])

        # spline methods dont support complex values
        for method in self.valid_methods:
            self.assertRaises(ValueError, ScipyGridInterp, points, values,
                              method)

    def test_spline_deriv_xi1d(self):
        # tests gradient values
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(1234)
        test_pt = np.random.uniform(0, 3, 2)
        actual = np.array(df(*test_pt))
        for method in self.valid_methods:
            interp = ScipyGridInterp(points, values, method)
            computed = interp.gradient(test_pt)
            r_err = rel_error(actual, computed)
            assert r_err < self.tol[method]

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
            interp = ScipyGridInterp(points, values, method)
            computed = interp.interpolate(test_pt)
            r_err = rel_error(actual, computed)
            assert r_err < self.tol[method]

    def test_spline_out_of_bounds_extrap(self):
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(5)
        test_pt = np.random.uniform(3, 3.1, 2)
        actual = func(*test_pt)
        gradient = np.array(df(*test_pt))
        for method in self.valid_methods:
            k = self.interp_configs[method]
            interp = ScipyGridInterp(points, values, method,
                                        bounds_error=False)
            computed = interp.interpolate(test_pt)
            computed_grad = interp.gradient(test_pt)
            r_err = rel_error(actual, computed)
            assert r_err < 1e3 * self.tol[method]

            r_err = rel_error(gradient, computed_grad)
            # extrapolated gradients are even trickier, but usable still
            assert r_err < 2e3 * self.tol[method]

    def test_spline_xi3d(self):
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(1)
        test_pt = np.random.uniform(0, 3, 6).reshape(3, 2)
        actual = func(*test_pt.T)
        for method in self.valid_methods:
            interp = ScipyGridInterp(points, values, method)
            computed = interp.interpolate(test_pt)
            r_err = rel_error(actual, computed)
            assert r_err < self.tol[method]

    def test_NaN_exception(self):
        np.random.seed(1234)
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)
        values = np.random.rand(5, 7)
        interp = ScipyGridInterp((x, y), values)

        with self.assertRaises(OutOfBoundsError) as cm:
            interp.interpolate(np.array([1, np.nan]))

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
            interp = ScipyGridInterp(points, values.tolist(), interp_method='junk')

        msg = ('Interpolation method "junk" is not defined. Valid methods are')
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

        msg = ("Interpolation method 'slinear' does not support complex values.")
        self.assertEqual(str(cm.exception), msg)

        interp = ScipyGridInterp(points, values.tolist())
        x = [0.5, 0, 0.5, 0.9]

        methods = set(interp._interp_methods()[0])
        self.assertEqual(methods, set(["quintic", "cubic", "slinear"]))


if __name__ == "__main__":
    unittest.main()
