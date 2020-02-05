"""
Unit tests for the spline interpolator component.
"""
from __future__ import division, print_function, absolute_import
from copy import deepcopy
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose, \
                          assert_almost_equal, assert_array_equal

from openmdao.components.interp_util.interp import InterpND
from openmdao.components.interp_util.outofbounds_error import OutOfBoundsError

def rel_error(actual, computed):
    return np.linalg.norm(actual - computed) / np.linalg.norm(actual)

scipy_gte_019 = True
try:
    from scipy.interpolate._bsplines import make_interp_spline
except ImportError:
    scipy_gte_019 = False


class InterpNDStandaloneFeatureTestcase(unittest.TestCase):

    def test_interp_spline_akima(self):

        import numpy as np

        from openmdao.components.interp_util.interp import InterpND

        xcp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        ycp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        n = 50
        x = np.linspace(1.0, 12.0, n)

        akima_options = {'delta_x': 0.1}
        interp = InterpND(interp_method='akima', points=[xcp], values=ycp, x_interp=x,
                          extrapolate=False, **akima_options)
        y = interp.evaluate_spline(np.expand_dims(ycp, axis=0))

        akima_y = np.array([[ 5.       ,  7.20902005,  9.21276849, 10.81097162, 11.80335574,
                            12.1278001 , 12.35869145, 12.58588536, 12.81022332, 13.03254681,
                            13.25369732, 13.47451633, 13.69584534, 13.91852582, 14.14281484,
                            14.36710105, 14.59128625, 14.81544619, 15.03965664, 15.26399335,
                            15.48853209, 15.7133486 , 15.93851866, 16.16573502, 16.39927111,
                            16.63928669, 16.8857123 , 17.1384785 , 17.39751585, 17.66275489,
                            17.93412619, 18.21156029, 18.49498776, 18.78433915, 19.07954501,
                            19.38053589, 19.68724235, 19.99959495, 20.31752423, 20.64096076,
                            20.96983509, 21.37579297, 21.94811407, 22.66809748, 23.51629844,
                            24.47327219, 25.51957398, 26.63575905, 27.80238264, 29.        ]])

        assert_array_almost_equal(akima_y.flatten(), y.flatten())

    def test_interp_spline_bspline(self):

        import numpy as np

        from openmdao.components.interp_util.interp import InterpND

        xcp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        ycp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        n = 50
        x = np.linspace(1.0, 12.0, n)

        interp = InterpND(interp_method='bsplines', num_cp=6, x_interp=x)

        y = interp.evaluate_spline(np.expand_dims(ycp, axis=0))

        akima_y = np.array([[ 5.       ,  7.20902005,  9.21276849, 10.81097162, 11.80335574,
                            12.1278001 , 12.35869145, 12.58588536, 12.81022332, 13.03254681,
                            13.25369732, 13.47451633, 13.69584534, 13.91852582, 14.14281484,
                            14.36710105, 14.59128625, 14.81544619, 15.03965664, 15.26399335,
                            15.48853209, 15.7133486 , 15.93851866, 16.16573502, 16.39927111,
                            16.63928669, 16.8857123 , 17.1384785 , 17.39751585, 17.66275489,
                            17.93412619, 18.21156029, 18.49498776, 18.78433915, 19.07954501,
                            19.38053589, 19.68724235, 19.99959495, 20.31752423, 20.64096076,
                            20.96983509, 21.37579297, 21.94811407, 22.66809748, 23.51629844,
                            24.47327219, 25.51957398, 26.63575905, 27.80238264, 29.        ]])

        assert_array_almost_equal(akima_y.flatten(), y.flatten())


class TestInterpNDPython(unittest.TestCase):
    """Tests for the non-scipy interolation algorithms."""

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
            self.assertRaises(ValueError, InterpND, points, values, method)

    def test_spline_xi1d(self):
        # test interpolated values
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(1)
        test_pt = np.random.uniform(0, 3, 2)
        actual = func(*test_pt)
        for method in self.valid_methods:
            interp = InterpND(method=method, points=points, values=values)
            computed = interp.interpolate(test_pt)
            r_err = rel_error(actual, computed)
            assert r_err < self.tol[method]

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
            interp = InterpND(points, values, method,
                              extrapolate=True)
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
            interp = InterpND(points, values, method)
            computed = interp.interpolate(test_pt)
            r_err = rel_error(actual, computed)
            #print(method, computed, actual, r_err)
            assert r_err < self.tol[method]

    def test_spline_xi3d_akima_delta_x(self):
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(1)
        test_pt = np.random.uniform(0, 3, 6).reshape(3, 2)
        actual = func(*test_pt.T)
        interp = InterpND(points, values, 'akima', delta_x=0.01)
        computed = interp.interpolate(test_pt)
        r_err = rel_error(actual, computed)
        #print('akima', computed, actual, r_err)
        assert r_err < self.tol['akima']

    def test_akima_interpolating_spline(self):
        n_cp = 80
        n_point = 160

        t = np.linspace(0, 3.0*np.pi, n_cp)
        tt = np.linspace(0, 3.0*np.pi, n_point)

        x = np.sin(t)

        # Now, test newer interface for order_reducing spline.

        interp = InterpND((t, ), x, 'akima', x_interp=tt.reshape((n_point, 1)))
        computed = interp.evaluate_spline(x.reshape((1, 80)))

        x_expected = np.sin(tt)
        delta = computed.flatten() - x_expected

        # Here we test that we don't have crazy interpolation error.
        self.assertLess(max(delta), .15)
        # And that it gets middle points a little better.
        self.assertLess(max(delta[15:-15]), .06)

    def test_bsplines_interpolating_spline(self):
        n_cp = 80
        n_point = 160

        t = np.linspace(0, 3.0*np.pi, n_cp)
        tt = np.linspace(0, 3.0*np.pi, n_point)

        x = np.sin(t)

        # Now, test newer interface for order_reducing spline.

        interp = InterpND((t, ), x, 'bsplines', x_interp=tt.reshape((n_point, 1)))
        computed = interp.evaluate_spline(x.reshape((1, 80)))

        x_expected = np.sin(tt)
        delta = computed.flatten() - x_expected

        # Here we test that we don't have crazy interpolation error.
        self.assertLess(max(delta), .15)
        # And that it gets middle points a little better.
        self.assertLess(max(delta[15:-15]), .06)

    def test_NaN_exception(self):
        np.random.seed(1234)
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)
        values = np.random.rand(5, 7)
        interp = InterpND((x, y), values)

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
            interp = InterpND(points, values.tolist(), interp_method='junk')

        msg = ('Interpolation method "junk" is not defined. Valid methods are')
        self.assertTrue(str(cm.exception).startswith(msg))

        with self.assertRaises(ValueError) as cm:
            interp = InterpND(points, values.tolist()[1])

        msg = ('There are 4 point arrays, but values has 3 dimensions')
        self.assertEqual(str(cm.exception), msg)

        badpoints = deepcopy(points)
        badpoints[0][0] = 55.0
        with self.assertRaises(ValueError) as cm:
            interp = InterpND(badpoints, values.tolist())

        msg = ('The points in dimension 0 must be strictly ascending')
        self.assertEqual(str(cm.exception), msg)

        badpoints[0] = np.vstack((np.arange(6), np.arange(6)))
        with self.assertRaises(ValueError) as cm:
            interp = InterpND(badpoints, values.tolist())

        msg = ('The points in dimension 0 must be 1-dimensional')
        self.assertEqual(str(cm.exception), msg)

        badpoints[0] = (np.arange(4))
        with self.assertRaises(ValueError) as cm:
            interp = InterpND(badpoints, values.tolist())

        msg = ('There are 4 points and 6 values in dimension 0')
        self.assertEqual(str(cm.exception), msg)

        badvalues = np.array(values, dtype=np.complex)
        with self.assertRaises(ValueError) as cm:
            interp = InterpND(badpoints, badvalues.tolist())

        msg = ("Interpolation method 'slinear' does not support complex values.")
        self.assertEqual(str(cm.exception), msg)

        interp = InterpND(points, values.tolist())
        x = [0.5, 0, 0.5, 0.9]

        with self.assertRaises(KeyError) as cm:
            interp = InterpND(points, values.tolist(), interp_method="slinear",
                              bad_arg=1)

        msg = ("\"InterpLinear: Option 'bad_arg' cannot be set because it has not been declared.")
        self.assertTrue(str(cm.exception).startswith(msg))


@unittest.skipIf(not scipy_gte_019, "only run if scipy>=0.19.")
class TestInterpNDScipy(unittest.TestCase):
    """Tests for the InterpScipy operated through InterpND."""

    def setUp(self):
        self.interp_configs = {
            "scipy_slinear": 1,
            "scipy_cubic": 3,
            "scipy_quintic": 5,
        }
        self.valid_methods = self.interp_configs.keys()
        self.tol = {
            "scipy_slinear": 5e-2,
            "scipy_cubic": 1e-4,
            "scipy_quintic": 1e-6,
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

        interp = InterpND(points, values, interp_method='scipy_cubic')

        # first dimension (x) should be reduced to k=1 (linear)
        self.assertEqual(interp.table._ki[0], 1)

        # should operate as normal
        x = np.array([0.5, 0, 1001])
        result = interp.interpolate(x)
        assert_almost_equal(result, -0.046325695741704434, decimal=5)

        interp = InterpND(points, values, interp_method='scipy_slinear')
        value1 = interp.interpolate(x)

        # cycle through different methods that require order reduction
        # in the first dimension
        interp = InterpND(points, values, interp_method='scipy_quintic')
        value2 = interp.interpolate(x)
        interp.gradient(x)
        interp = InterpND(points, values, interp_method='scipy_cubic')
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
            self.assertRaises(ValueError, InterpND, points, values,
                              method)

    def test_spline_deriv_xi1d(self):
        # tests gradient values
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(1234)
        test_pt = np.random.uniform(0, 3, 2)
        actual = np.array(df(*test_pt))
        for method in self.valid_methods:
            interp = InterpND(points, values, method)
            computed = interp.gradient(test_pt)
            r_err = rel_error(actual, computed)
            assert r_err < self.tol[method]

            # test that gradients have been cached
            assert_array_equal(interp._xi, test_pt)
            assert_array_equal(
                interp._d_dx.flatten(), computed.flatten())

    def test_gradients_returned_by_xi(self):
        # verifies that gradients with respect to xi are returned if cached
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(4321)
        for method in self.valid_methods:
            interp = InterpND(points, values, method)
            x = np.array([0.9, 0.1])
            interp._xi = x
            dy = np.array([0.997901, 0.08915])
            interp._d_dx = dy
            assert_almost_equal(interp.gradient(x), dy)

    def test_spline_xi1d(self):
        # test interpolated values
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(1)
        test_pt = np.random.uniform(0, 3, 2)
        actual = func(*test_pt)
        for method in self.valid_methods:
            interp = InterpND(points, values, method)
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
            interp = InterpND(points, values, method,
                              extrapolate=True)
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
            interp = InterpND(points, values, method)
            computed = interp.interpolate(test_pt)
            r_err = rel_error(actual, computed)
            assert r_err < self.tol[method]

    def test_NaN_exception(self):
        np.random.seed(1234)
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)
        values = np.random.rand(5, 7)
        interp = InterpND((x, y), values)

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
            interp = InterpND(points, values.tolist(), interp_method='junk')

        msg = ('Interpolation method "junk" is not defined. Valid methods are')
        self.assertTrue(str(cm.exception).startswith(msg))

        with self.assertRaises(ValueError) as cm:
            interp = InterpND(points, values.tolist()[1])

        msg = ('There are 4 point arrays, but values has 3 dimensions')
        self.assertEqual(str(cm.exception), msg)

        badpoints = deepcopy(points)
        badpoints[0][0] = 55.0
        with self.assertRaises(ValueError) as cm:
            interp = InterpND(badpoints, values.tolist())

        msg = ('The points in dimension 0 must be strictly ascending')
        self.assertEqual(str(cm.exception), msg)

        badpoints[0] = np.vstack((np.arange(6), np.arange(6)))
        with self.assertRaises(ValueError) as cm:
            interp = InterpND(badpoints, values.tolist())

        msg = ('The points in dimension 0 must be 1-dimensional')
        self.assertEqual(str(cm.exception), msg)

        badpoints[0] = (np.arange(4))
        with self.assertRaises(ValueError) as cm:
            interp = InterpND(badpoints, values.tolist())

        msg = ('There are 4 points and 6 values in dimension 0')
        self.assertEqual(str(cm.exception), msg)

        badvalues = np.array(values, dtype=np.complex)
        with self.assertRaises(ValueError) as cm:
            interp = InterpND(badpoints, badvalues.tolist(), interp_method='scipy_slinear')

        msg = ("Interpolation method 'scipy_slinear' does not support complex values.")
        self.assertEqual(str(cm.exception), msg)

        interp = InterpND(points, values.tolist())
        x = [0.5, 0, 0.5, 0.9]


if __name__ == '__main__':
    unittest.main()