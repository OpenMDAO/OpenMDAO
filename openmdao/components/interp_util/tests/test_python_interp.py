"""
Unit test for scipy interpolator. Note that an api is defined for standalone use.
"""
from __future__ import division, print_function, absolute_import
from copy import deepcopy
import unittest

import numpy as np

from openmdao.components.interp_util.python_interp import PythonGridInterp
from openmdao.components.interp_util.outofbounds_error import OutOfBoundsError


def rel_error(actual, computed):
    return np.linalg.norm(actual - computed) / np.linalg.norm(actual)


class TestPythonGridInterpolator(unittest.TestCase):
    """Tests the functionality of the python grid interpolator."""

    def setUp(self):
        self.interp_configs = {
            "slinear": 2,
            "cubic": 3,
            "lagrange2": 3,
            "lagrange3": 4,
            "akima": 4,
        }
        self.valid_methods = self.interp_configs.keys()
        self.tol = {
            "slinear": 5e-2,
            "lagrange2": 5e-2,
            "lagrange3": 1e-4,
            "cubic": 1e-4,
            "akima": 1e-3,
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
        v = np.random.uniform(0, 3, 50)
        v.sort()

        points = [u, v]
        values = f(*np.meshgrid(*points, indexing='ij'))
        return points, values, f, df

    def test_minimum_required_gridsize(self):
        for method in self.valid_methods:
            k = self.interp_configs[method] - 1
            x = np.linspace(0, 1, k)
            y = np.linspace(0, 1, k)
            points = [x, y]
            X, Y = np.meshgrid(*points, indexing='ij')
            values = X + Y
            self.assertRaises(ValueError, PythonGridInterp, points, values, method)

    def test_spline_xi1d(self):
        # test interpolated values
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(1)
        test_pt = np.random.uniform(0, 3, 2)
        actual = func(*test_pt)
        for order in self.valid_methods:
            interp = PythonGridInterp(points, values, order)
            computed = interp.interpolate(test_pt)
            r_err = rel_error(actual, computed)
            assert r_err < self.tol[order]

    def test_spline_out_of_bounds_extrap(self):
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(5)
        test_pt = np.random.uniform(3, 3.1, 2)
        actual = func(*test_pt)
        gradient = np.array(df(*test_pt))
        tol = 1e-1
        for method in self.valid_methods:
            k = self.interp_configs[method]
            if method == 'slinear':
                tol = 2
            interp = PythonGridInterp(points, values, method,
                                      bounds_error=False)
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
            interp = PythonGridInterp(points, values, method)
            computed = interp.interpolate(test_pt)
            r_err = rel_error(actual, computed)
            #print(method, computed, actual, r_err)
            assert r_err < self.tol[method]

    def test_spline_xi3d_akima_delta_x(self):
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(1)
        test_pt = np.random.uniform(0, 3, 6).reshape(3, 2)
        actual = func(*test_pt.T)
        interp = PythonGridInterp(points, values, 'akima', delta_x=0.01)
        computed = interp.interpolate(test_pt)
        r_err = rel_error(actual, computed)
        #print('akima', computed, actual, r_err)
        assert r_err < self.tol['akima']

    def test_NaN_exception(self):
        np.random.seed(1234)
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)
        values = np.random.rand(5, 7)
        interp = PythonGridInterp((x, y), values)

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
            interp = PythonGridInterp(points, values.tolist(), interp_method='junk')

        msg = ('Interpolation method "junk" is not defined. Valid methods are')
        self.assertTrue(str(cm.exception).startswith(msg))

        with self.assertRaises(ValueError) as cm:
            interp = PythonGridInterp(points, values.tolist()[1])

        msg = ('There are 4 point arrays, but values has 3 dimensions')
        self.assertEqual(str(cm.exception), msg)

        badpoints = deepcopy(points)
        badpoints[0][0] = 55.0
        with self.assertRaises(ValueError) as cm:
            interp = PythonGridInterp(badpoints, values.tolist())

        msg = ('The points in dimension 0 must be strictly ascending')
        self.assertEqual(str(cm.exception), msg)

        badpoints[0] = np.vstack((np.arange(6), np.arange(6)))
        with self.assertRaises(ValueError) as cm:
            interp = PythonGridInterp(badpoints, values.tolist())

        msg = ('The points in dimension 0 must be 1-dimensional')
        self.assertEqual(str(cm.exception), msg)

        badpoints[0] = (np.arange(4))
        with self.assertRaises(ValueError) as cm:
            interp = PythonGridInterp(badpoints, values.tolist())

        msg = ('There are 4 points and 6 values in dimension 0')
        self.assertEqual(str(cm.exception), msg)

        badvalues = np.array(values, dtype=np.complex)
        with self.assertRaises(ValueError) as cm:
            interp = PythonGridInterp(badpoints, badvalues.tolist())

        msg = ("Interpolation method 'slinear' does not support complex values.")
        self.assertEqual(str(cm.exception), msg)

        interp = PythonGridInterp(points, values.tolist())
        x = [0.5, 0, 0.5, 0.9]

        methods = set(interp._interp_methods()[0])
        self.assertEqual(methods, set(["cubic", "slinear", 'lagrange2', 'lagrange3', 'akima']))

        with self.assertRaises(KeyError) as cm:
            interp = PythonGridInterp(points, values.tolist(), interp_method="slinear",
                                      bad_arg=1)

        msg = ("\"InterpLinear: Option 'bad_arg' cannot be set because it has not been declared.")
        self.assertTrue(str(cm.exception).startswith(msg))


if __name__ == "__main__":
    unittest.main()
