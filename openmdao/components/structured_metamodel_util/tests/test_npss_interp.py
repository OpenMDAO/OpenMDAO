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
from openmdao.components.structured_metamodel_util.npss_interp import NPSSGridInterp
from openmdao.components.structured_metamodel_util.outofbounds_error import OutOfBoundsError


def rel_error(actual, computed):
    return np.linalg.norm(actual - computed) / np.linalg.norm(actual)


class TestNPSSGridInterpolator(unittest.TestCase):
    """Tests the functionality of the npss grid interpolator."""

    def setUp(self):
        self.interp_configs = {
            "slinear": 2,
            "cubic": 3,
            "lagrange2": 3,
            "lagrange3": 4,
            "akima": 4,
        }
        self.valid_orders = self.interp_configs.keys()
        self.tol = {
            "slinear": 5e-2,
            "lagrange2": 5e-2,
            "lagrange3": 1e-4,
            "cubic": 1e-4,
            "akima": 1e-2,
        }

    #def _get_sample_4d_large(self):
        #def f(x, y, z, w):
            #return x**2 + y**2 + z**2 + w**2
        #X = np.linspace(-10, 10, 6)
        #Y = np.linspace(-10, 10, 7)
        #np.random.seed(0)
        #Z = np.random.uniform(-10, 10, 6)
        #Z.sort()
        #W = np.linspace(-10, 10, 8)
        #points = [X, Y, Z, W]
        #values = f(*np.meshgrid(*points, indexing='ij'))
        #return points, values

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

    #def test_list_input(self):
        #points, values = self._get_sample_4d_large()

        #sample = np.asarray([[0.1, 0.1, 1., .9], [0.2, 0.1, .45, .8],
                             #[0.5, 0.5, .5, .5]])

        #for method in self.valid_methods:
            #interp = ScipyGridInterp(points, values.tolist(), method=method)
            #v1 = interp.interpolate(sample.tolist(), compute_gradients=False)

            #interp = ScipyGridInterp(points, values, method=method)
            #v2 = interp.interpolate(sample, compute_gradients=False)

            #assert_allclose(v1, v2)

    #def test_auto_reduce_spline_order(self):
        ## if a spline method is used and spline_dim_error=False and a dimension
        ## does not have enough points, the spline order for that dimension
        ## should be automatically reduced
        #np.random.seed(314)

        ## x dimension is too small for cubic, should fall back to linear
        #x = [0, 1]
        #y = np.linspace(-10, 4, 10)
        #z = np.linspace(1000, 2000, 20)

        #points = [x, y, z]
        #values = np.random.randn(2, 10, 20)

        ## verify that this raises error with dimension checking
        #self.assertRaises(ValueError, ScipyGridInterp,
                          #points, values, 'cubic')

        #interp = ScipyGridInterp(
            #points, values, method='cubic', spline_dim_error=False)

        ## first dimension (x) should be reduced to k=1 (linear)
        #assert_equal(interp._ki[0], 1)

        ## should operate as normal
        #x = [0.5, 0, 1001]
        #result = interp.interpolate(x)
        #assert_almost_equal(result, -0.046325695741704434, decimal=5)

        #interp = ScipyGridInterp(
            #points, values, method='slinear', spline_dim_error=False)

        #value1 = interp.interpolate(x)
        ## cycle through different methods that require order reduction
        ## in the first dimension
        #value2 = interp.interpolate(x, method='quintic')
        #interp.gradient(x, method='quintic')
        #value3 = interp.interpolate(x, method='cubic')
        #interp.gradient(x, method='cubic')
        ## use default method again
        #value4 = interp.interpolate(x)

        ## values from different methods should be different
        #self.assertRaises(AssertionError, assert_equal, value1, value2)
        #self.assertRaises(AssertionError, assert_equal, value2, value3)

        ## first value should match last with no side effects from the
        ## order reduction or gradient caluclations
        #assert_equal(value1, value4)

    #def test_complex_exception_spline(self):
        #points, values = self._get_sample_4d_large()
        #values = values - 2j * values
        #sample = np.asarray([[0.1, 0.1, 1., .9]])

        ## spline methods dont support complex values
        #for method in self.valid_methods:
            #self.assertRaises(ValueError, ScipyGridInterp, points, values,
                              #method)

    #def test_minimum_required_gridsize(self):
        #for method in self.valid_methods:
            #k = self.interp_configs[method]
            #x = np.linspace(0, 1, k)
            #y = np.linspace(0, 1, k)
            #points = [x, y]
            #X, Y = np.meshgrid(*points, indexing='ij')
            #values = X + Y
            #self.assertRaises(ValueError, ScipyGridInterp, points, values,
                              #method)

     #def test_spline_deriv_xi1d(self):
        ## tests gradient values
        #points, values, func, df = self. _get_sample_2d()
        #np.random.seed(1234)
        #test_pt = np.random.uniform(0, 3, 2)
        #actual = np.array(df(*test_pt))
        #tol = 1e-1
        #for method in self.valid_methods:
            #if method == 'slinear':
                #tol = 1.5
            #interp = ScipyGridInterp(points, values, method)
            #computed = interp.gradient(test_pt)
            #r_err = rel_error(actual, computed)
            #assert r_err < tol

            ## test that gradients have been cached
            #assert_array_equal(interp._xi, test_pt)
            #assert_array_equal(
                #interp._all_gradients.flatten(), computed.flatten())

    #def test_gradients_returned_by_xi(self):
        ## verifies that gradients with respect to xi are returned if cached
        #points, values, func, df = self. _get_sample_2d()
        #np.random.seed(4321)
        #for method in self.valid_methods:
            #interp = ScipyGridInterp(points, values, method)
            #x = np.array([0.9, 0.1])
            #interp._xi = x
            #interp._g_order = method
            #dy = np.array([0.997901, 0.08915])
            #interp._all_gradients = dy
            #assert_almost_equal(interp.gradient(x), dy)

    def test_spline_xi1d(self):
        # test interpolated values
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(1)
        test_pt = np.random.uniform(0, 3, 2)
        actual = func(*test_pt)
        for order in self.valid_orders:
            interp = NPSSGridInterp(points, values, order)
            computed = interp.interpolate(test_pt, compute_gradients=False)
            r_err = rel_error(actual, computed)
            assert r_err < self.tol[order]

    #def test_spline_out_of_bounds_extrap(self):
        #points, values, func, df = self. _get_sample_2d()
        #np.random.seed(5)
        #test_pt = np.random.uniform(3, 3.1, 2)
        #actual = func(*test_pt)
        #gradient = np.array(df(*test_pt))
        #tol = 1e-1
        #for method in self.valid_methods:
            #k = self.interp_configs[method]
            #if method == 'slinear':
                #tol = 2
            #interp = ScipyGridInterp(points, values, method,
                                        #bounds_error=False,
                                        #fill_value=None)
            #computed = interp.interpolate(test_pt)
            #computed_grad = interp.gradient(test_pt)
            #r_err = rel_error(actual, computed)
            #assert r_err < tol

            #r_err = rel_error(gradient, computed_grad)
            ## extrapolated gradients are even trickier, but usable still
            #assert r_err < 2 * tol

    def test_spline_xi3d(self):
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(1)
        test_pt = np.random.uniform(0, 3, 6).reshape(3, 2)
        actual = func(*test_pt.T)
        for order in self.valid_orders:
            interp = NPSSGridInterp(points, values, order)
            computed = interp.interpolate(test_pt, compute_gradients=False)
            r_err = rel_error(actual, computed)
            print( actual, computed, r_err, order)
            assert r_err < self.tol[order]

    #def test_out_of_bounds_fill2(self):
        #points, values, func, df = self. _get_sample_2d()
        #np.random.seed(1)
        #test_pt = np.random.uniform(3, 3.1, 2)
        #actual = np.asarray([np.nan])
        #methods = self.valid_methods
        #for method in methods:
            #interp = ScipyGridInterp(points, values, method,
                                        #bounds_error=False,
                                        #fill_value=np.nan)
            #computed = interp.interpolate(test_pt, compute_gradients=False)
            #assert_array_almost_equal(computed, actual)

    #def test_invalid_fill_value(self):
        #np.random.seed(1234)
        #x = np.linspace(0, 2, 5)
        #y = np.linspace(0, 1, 7)
        #values = np.random.rand(5, 7)

        ## integers can be cast to floats
        #ScipyGridInterp((x, y), values, fill_value=1)

        ## complex values cannot
        #self.assertRaises(ValueError, ScipyGridInterp,
                          #(x, y), values, fill_value=1 + 2j)

    #def test_NaN_exception(self):
        #np.random.seed(1234)
        #x = np.linspace(0, 2, 5)
        #y = np.linspace(0, 1, 7)
        #values = np.random.rand(5, 7)
        #interp = ScipyGridInterp((x, y), values)

        #with self.assertRaises(OutOfBoundsError) as cm:
            #interp.interpolate([1, np.nan])

        #err = cm.exception

        #self.assertEqual(str(err), 'One of the requested xi contains a NaN')
        #self.assertEqual(err.idx, 1)
        #self.assertTrue(np.isnan(err.value))
        #self.assertEqual(err.lower, 0)
        #self.assertEqual(err.upper, 1)

    #def test_error_messages(self):
        ## For coverage. Most of these errors are probably not reachable in openmdao, but
        ## proper unit testing requires them for standalone usage of the Interpolation.
        #points, values = self._get_sample_4d_large()

        #with self.assertRaises(ValueError) as cm:
            #interp = ScipyGridInterp(points, values.tolist(), method='junk')

        #msg = ('Method "junk" is not defined. Valid methods are')
        #self.assertTrue(str(cm.exception).startswith(msg))

        #with self.assertRaises(ValueError) as cm:
            #interp = ScipyGridInterp(points, values.tolist()[1])

        #msg = ('There are 4 point arrays, but values has 3 dimensions')
        #self.assertEqual(str(cm.exception), msg)

        #badpoints = deepcopy(points)
        #badpoints[0][0] = 55.0
        #with self.assertRaises(ValueError) as cm:
            #interp = ScipyGridInterp(badpoints, values.tolist())

        #msg = ('The points in dimension 0 must be strictly ascending')
        #self.assertEqual(str(cm.exception), msg)

        #badpoints[0] = np.vstack((np.arange(6), np.arange(6)))
        #with self.assertRaises(ValueError) as cm:
            #interp = ScipyGridInterp(badpoints, values.tolist())

        #msg = ('The points in dimension 0 must be 1-dimensional')
        #self.assertEqual(str(cm.exception), msg)

        #badpoints[0] = (np.arange(4))
        #with self.assertRaises(ValueError) as cm:
            #interp = ScipyGridInterp(badpoints, values.tolist())

        #msg = ('There are 4 points and 6 values in dimension 0')
        #self.assertEqual(str(cm.exception), msg)

        #badvalues = np.array(values, dtype=np.complex)
        #with self.assertRaises(ValueError) as cm:
            #interp = ScipyGridInterp(badpoints, badvalues.tolist())

        #msg = ("method 'slinear' does not support complex values.")
        #self.assertEqual(str(cm.exception), msg)

        #interp = ScipyGridInterp(points, values.tolist())
        #x = [0.5, 0, 0.5, 0.9]

        #with self.assertRaises(ValueError) as cm:
            #computed = interp.interpolate(x, method='junk')

        #msg = ('Method "junk" is not defined. Valid methods are')
        #self.assertTrue(str(cm.exception).startswith(msg))

        #self.assertEqual(set(interp.methods()), set(["quintic", "cubic", "slinear"]))


if __name__ == "__main__":
    unittest.main()
