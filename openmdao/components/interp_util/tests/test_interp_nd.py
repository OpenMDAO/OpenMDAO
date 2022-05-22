"""
Unit tests for the standalone interpolator.
"""
from copy import deepcopy
import unittest

import numpy as np

# Only import openmdao to test using standalone interp in a component with complex step.
import openmdao.api as om

from openmdao.components.interp_util.interp import InterpND, SPLINE_METHODS, TABLE_METHODS
from openmdao.components.interp_util.interp_semi import InterpNDSemi
from openmdao.components.interp_util.outofbounds_error import OutOfBoundsError
from openmdao.utils.assert_utils import assert_near_equal, assert_equal_arrays, assert_warning, assert_check_partials
from openmdao.utils.om_warnings import OMDeprecationWarning

def rel_error(actual, computed):
    return np.linalg.norm(actual - computed) / np.linalg.norm(actual)

scipy_gte_019 = True
try:
    from scipy.interpolate._bsplines import make_interp_spline
except ImportError:
    scipy_gte_019 = False


class InterpNDStandaloneFeatureTestcase(unittest.TestCase):

    def test_interp_spline_akima(self):

        xcp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        ycp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        n = 50
        x = np.linspace(1.0, 12.0, n)

        interp = InterpND(method='akima', points=xcp, x_interp=x, delta_x=0.1)
        y = interp.evaluate_spline(ycp)

        assert_near_equal(y,
                          np.array([ 5.       ,  7.20902005,  9.21276849, 10.81097162, 11.80335574,
                                     12.1278001 , 12.35869145, 12.58588536, 12.81022332, 13.03254681,
                                     13.25369732, 13.47451633, 13.69584534, 13.91852582, 14.14281484,
                                     14.36710105, 14.59128625, 14.81544619, 15.03965664, 15.26399335,
                                     15.48853209, 15.7133486 , 15.93851866, 16.16573502, 16.39927111,
                                     16.63928669, 16.8857123 , 17.1384785 , 17.39751585, 17.66275489,
                                     17.93412619, 18.21156029, 18.49498776, 18.78433915, 19.07954501,
                                     19.38053589, 19.68724235, 19.99959495, 20.31752423, 20.64096076,
                                     20.96983509, 21.37579297, 21.94811407, 22.66809748, 23.51629844,
                                     24.47327219, 25.51957398, 26.63575905, 27.80238264, 29.        ]),
                          tolerance=1e-6)

    def test_interp_spline_akima_derivs(self):

        xcp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        ycp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        n = 5
        x = np.linspace(1.0, 12.0, n)

        interp = InterpND(method='akima', points=xcp, x_interp=x, delta_x=0.1)
        y, dy_dycp = interp.evaluate_spline(ycp, compute_derivative=True)

        assert_near_equal(dy_dycp,
                          np.array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                      0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                                    [-1.86761492e-06,  3.31278014e-02,  1.05874907e+00,
                                     -9.18750000e-02,  0.00000000e+00,  0.00000000e+00],
                                    [ 0.00000000e+00,  0.00000000e+00, -2.10964627e-01,
                                      1.19119941e+00,  2.02602810e-02, -4.95062934e-04],
                                    [ 0.00000000e+00,  0.00000000e+00, -2.64126732e-01,
                                      5.82784977e-01,  6.83151998e-01, -1.81024253e-03],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                                      0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
                          tolerance=1e-6)

    def test_interp_spline_bsplines(self):

        xcp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        ycp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        n = 50
        x = np.linspace(1.0, 12.0, n)

        interp = InterpND(method='bsplines', num_cp=6, x_interp=x)

        y = interp.evaluate_spline(ycp)

        assert_near_equal(y,
                          np.array([ 9.21614583,  9.90911525, 10.52244151, 11.06231159, 11.53491244,
                                     11.94643105, 12.30305438, 12.61096939, 12.87636305, 13.10542234,
                                     13.30433422, 13.47928566, 13.63646363, 13.7820551 , 13.92203064,
                                     14.05954727, 14.19579437, 14.33192094, 14.46907599, 14.60840854,
                                     14.75106758, 14.89820214, 15.05096121, 15.2104938 , 15.37794893,
                                     15.5544756 , 15.74122282, 15.9393396 , 16.14997495, 16.37427787,
                                     16.61339737, 16.86848247, 17.14102103, 17.43486416, 17.75486932,
                                     18.10589772, 18.49281052, 18.92046894, 19.39373414, 19.91746734,
                                     20.4965297 , 21.13578243, 21.8400867 , 22.61430372, 23.46329467,
                                     24.39192074, 25.40504312, 26.507523  , 27.70422156, 29.        ]),
                          tolerance=1e-6)

    def test_table_interp(self):

        # create input param training data, of sizes 25, 5, and 10 points resp.
        p1 = np.linspace(0, 100, 25)
        p2 = np.linspace(-10, 10, 5)
        p3 = np.linspace(0, 1, 10)

        # can use meshgrid to create a 3D array of test data
        P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
        f = np.sqrt(P1) + P2 * P3

        interp = InterpND(method='lagrange3', points=(p1, p2, p3), values=f)

        x = np.array([55.12, -2.14, 0.323])
        f, df_dx = interp.interpolate(x, compute_derivative=True)

        actual = np.array([6.73306794])
        deriv_actual = np.array([[ 0.06734927, 0.323 , -2.14]])

        assert_near_equal(f, actual, tolerance=1e-7)
        assert_near_equal(df_dx, deriv_actual, tolerance=1e-7)

    def test_cs_across_interp(self):
        # The standalone interpolator is used inside of components, so the imaginary part must
        # be carried through all operations to the outputs.
        xcp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        ycp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        n = 50
        x = np.linspace(1.0, 12.0, n)

        ycp = np.array([[5.0 + 1j, 12.0, 14.0, 16.0, 21.0, 29.0],
                        [5.0, 12.0 + 1j, 14.0, 16.0, 21.0, 29.0]])

        for method in SPLINE_METHODS:

            # complex step not supported on scipy methods
            if method.startswith('scipy'):
                continue

            interp = InterpND(method=method, points=xcp, x_interp=x)
            y, dy = interp.evaluate_spline(ycp, compute_derivative=True)

            self.assertTrue(y.dtype == complex)

            if method in ['akima']:
                # Derivs depend on values only for akima.
                self.assertTrue(dy.dtype == complex)

        p1 = np.linspace(0, 100, 25)
        p2 = np.linspace(-10, 10, 5)
        p3 = np.linspace(0, 1, 10)

        # can use meshgrid to create a 3D array of test data
        P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
        f = np.sqrt(P1) + P2 * P3

        x = np.array([[55.12 + 1j, -2.14, 0.323],
                      [55.12, -2.14 + 1j, 0.323],
                      [55.12, -2.14, 0.323 + 1j]])

        for method in TABLE_METHODS:

            # complex step not supported on scipy methods
            if method.startswith('scipy'):
                continue

            if '1D' in method or '2D' in method:
                # These methods are for fixed grids other than 3d.
                continue

            interp = InterpND(method=method, points=(p1, p2, p3), values=f)
            y, dy = interp.interpolate(x, compute_derivative=True)

            self.assertTrue(y.dtype == complex)
            self.assertTrue(dy.dtype == complex)


class TestInterpNDSemiPython(unittest.TestCase):
    """Tests for the standalone semi structured interp."""

    def setUp(self):
        self.interp_configs = {
            "slinear": 2,
            "lagrange2": 3,
            "lagrange3": 4,
            "akima": 4,
        }
        self.interp_methods = self.interp_configs.keys()

        self.tol = {
            "slinear": 5e-2,
            "lagrange2": 5e-2,
            "lagrange3": 1e-4,
            "akima": 1e-3,
        }

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

    def test_2d(self):
        # test interpolated values
        points, values, func, df = self._get_sample_2d()
        np.random.seed(1)
        X, Y = np.meshgrid(*points, indexing='ij')
        X = X.ravel()
        Y = Y.ravel()
        grid = np.array([X, Y]).T

        test_pt = np.random.uniform(0, 3, 2)
        actual = func(*test_pt)
        for method in self.interp_methods:
            interp = InterpNDSemi(grid, values.ravel(), method=method)
            computed = interp.interpolate(test_pt)
            r_err = rel_error(actual, computed)
            assert r_err < self.tol[method]

    def test_minimum_required_gridsize(self):
        for method in self.interp_methods:

            k = self.interp_configs[method] - 1
            x = np.linspace(0, 1, k)
            y = np.linspace(0, 1, k)

            points = [x, y]
            X, Y = np.meshgrid(*points, indexing='ij')
            X = X.ravel()
            Y = Y.ravel()
            values = X + Y
            grid = np.array([X, Y]).T
            with self.assertRaises(ValueError) as cm:
                interp = InterpNDSemi(grid, values, method=method)

            msg = 'There are {} points in a data dimension, but method'.format(k)
            self.assertTrue(str(cm.exception).startswith(msg))

    def test_NaN_exception(self):
        np.random.seed(1234)
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)
        points = [x, y]
        X, Y = np.meshgrid(*points, indexing='ij')
        X = X.ravel()
        Y = Y.ravel()
        grid = np.array([X, Y]).T
        values = np.random.rand(5, 7).ravel()
        interp = InterpNDSemi(grid, values, method='slinear', extrapolate=False)

        with self.assertRaises(OutOfBoundsError) as cm:
            interp.interpolate(np.array([1, np.nan]))

        err = cm.exception

        self.assertEqual(str(err), 'One of the requested xi contains a NaN')
        self.assertEqual(err.idx, 1)
        self.assertTrue(np.isnan(err.value))

    def test_error_messages(self):
        points, values = self._get_sample_4d_large()
        X, Y, Z, W = np.meshgrid(*points, indexing='ij')
        X = X.ravel()
        Y = Y.ravel()
        Z = Z.ravel()
        W = W.ravel()
        grid = np.array([X, Y, Z, W]).T
        values = values.ravel()

        with self.assertRaises(ValueError) as cm:
            interp = InterpNDSemi(grid, values, method='junk')

        msg = ('Interpolation method "junk" is not defined. Valid methods are')
        self.assertTrue(cm.exception.args[0].startswith(msg))

        with self.assertRaises(ValueError) as cm:
            interp = InterpNDSemi(grid, values, method=points)

        msg = ("Argument 'method' should be a string.")
        self.assertTrue(cm.exception.args[0].startswith(msg))

        with self.assertRaises(ValueError) as cm:
            interp = InterpNDSemi(grid, values[:-1], method='slinear')

        msg = ('There are 2016 point arrays, but 2015 values.')
        self.assertEqual(cm.exception.args[0], msg)

        badgrid = deepcopy(grid)
        badgrid[0][0] = -6.0
        with self.assertRaises(ValueError) as cm:
            interp = InterpNDSemi(badgrid, values, method='slinear')

        msg = ('The points in dimension 0 must be strictly ascending.')
        self.assertEqual(cm.exception.args[0], msg)

        with self.assertRaises(KeyError) as cm:
            interp = InterpNDSemi(grid, values, method='slinear', bad_arg=1)

        msg = ("Option 'bad_arg' cannot be set because it has not been declared.")
        self.assertTrue(cm.exception.args[0].endswith(msg))


class TestInterpNDPython(unittest.TestCase):

    """Tests for the non-scipy interpolation algorithms."""

    def setUp(self):
        self.interp_configs = {
            "slinear": 2,
            "cubic": 3,
            "lagrange2": 3,
            "lagrange3": 4,
            "akima": 4,
            "scipy_slinear": 1,
            "scipy_cubic": 3,
            "scipy_quintic": 5,
        }
        self.spline_configs = {
            "slinear": 2,
            "cubic": 3,
            "lagrange2": 3,
            "lagrange3": 4,
            "akima": 4,
            "bsplines": 4,
            "scipy_slinear": 1,
            "scipy_cubic": 3,
            "scipy_quintic": 5,
        }
        self.interp_methods = self.interp_configs.keys()
        self.spline_methods = self.spline_configs.keys()

        self.tol = {
            "slinear": 5e-2,
            "lagrange2": 5e-2,
            "lagrange3": 1e-4,
            "cubic": 1e-4,
            "akima": 1e-3,
            "bsplines": 1e-1,
            "scipy_slinear": 5e-2,
            "scipy_cubic": 1e-4,
            "scipy_quintic": 1e-6,
        }

    def _get_sample_1d(self):
        # test problem with enough points for smooth spline fits
        def f(u):
            return 2.0 * np.cos(u)

        def df(u):
            return -2.0 * np.sin(u)

        # uniformly spaced axis
        u = np.linspace(0, 3, 50)

        points = [u]
        values = f(u)
        return points, values, f, df

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

    def test_minimum_required_gridsize(self):
        for method in self.interp_methods:

            # Scipy does order reduction as needed.
            if method.startswith('scipy'):
                continue

            k = self.interp_configs[method] - 1
            x = np.linspace(0, 1, k)
            y = np.linspace(0, 1, k)
            points = [x, y]
            X, Y = np.meshgrid(*points, indexing='ij')
            values = X + Y
            #self.assertRaises(ValueError, InterpND, points, values, method)
            with self.assertRaises(ValueError) as cm:
                interp = InterpND(method=method, points=points, values=values)

            msg = 'There are {} points in a data dimension, but method'.format(k)
            self.assertTrue(str(cm.exception).startswith(msg))

    def test_spline_single_dim(self):
        # test interpolated values
        points, values, func, df = self._get_sample_1d()
        test_pt = np.array([[0.76], [.33]])
        actual = func(test_pt).flatten()
        for method in self.interp_methods:
            interp = InterpND(method=method, points=points, values=values)
            computed = interp.interpolate(test_pt)
            r_err = rel_error(actual, computed)
            assert r_err < self.tol[method]

    def test_spline_xi1d(self):
        # test interpolated values
        points, values, func, df = self._get_sample_2d()
        np.random.seed(1)
        test_pt = np.random.uniform(0, 3, 2)
        actual = func(*test_pt)
        for method in self.interp_methods:
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
        for method in self.interp_methods:
            k = self.interp_configs[method]
            if method == 'slinear':
                tol = 2
            interp = InterpND(method=method, points=points, values=values,
                              extrapolate=True)
            computed, computed_grad = interp.interpolate(test_pt, compute_derivative=True)
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
        for method in self.interp_methods:
            interp = InterpND(method=method, points=points, values=values)
            computed = interp.interpolate(test_pt)
            r_err = rel_error(actual, computed)
            #print(method, computed, actual, r_err)
            assert r_err < self.tol[method]

    def test_spline_xi3d_akima_delta_x(self):
        points, values, func, df = self. _get_sample_2d()
        np.random.seed(1)
        test_pt = np.random.uniform(0, 3, 6).reshape(3, 2)
        actual = func(*test_pt.T)
        interp = InterpND(method='akima', points=points, values=values, delta_x=0.01)
        computed = interp.interpolate(test_pt)
        r_err = rel_error(actual, computed)
        #print('akima', computed, actual, r_err)
        assert r_err < self.tol['akima']

    def test_spline_deriv_xi1d(self):
        # tests gradient values
        points, values, func, df = self._get_sample_2d()
        np.random.seed(1234)
        test_pt = np.random.uniform(0, 3, 2)
        actual = np.array(df(*test_pt))
        for method in self.interp_methods:
            interp = InterpND(method=method,points=points, values=values)
            computed = interp.gradient(test_pt)
            r_err = rel_error(actual, computed)
            assert r_err < 2.5 * self.tol[method]

            # test that gradients have been cached
            assert_equal_arrays(interp._xi.flatten(), test_pt.flatten())
            assert_equal_arrays(interp._d_dx.flatten(), computed.flatten())

    def test_gradients_returned_by_xi(self):
        # verifies that gradients with respect to xi are returned if cached
        points, values, func, df = self._get_sample_2d()
        np.random.seed(4321)
        for method in self.interp_methods:
            interp = InterpND(method=method, points=points, values=values)
            x = np.array([0.9, 0.1])
            interp._xi = x
            dy = np.array([0.997901, 0.08915])
            interp._d_dx = dy
            assert_near_equal(interp.gradient(x), dy, tolerance=1e-7)

    def test_akima_interpolating_spline(self):
        n_cp = 80
        n_point = 160

        t = np.linspace(0, 3.0*np.pi, n_cp)
        tt = np.linspace(0, 3.0*np.pi, n_point)

        x = np.sin(t)

        # Now, test newer interface for order_reducing spline.

        interp = InterpND(method='akima', points=t, x_interp=tt)
        computed = interp.evaluate_spline(x)

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

        interp = InterpND(method='bsplines', points=t, x_interp=tt)
        computed = interp.evaluate_spline(x)

        x_expected = np.sin(tt)
        delta = computed.flatten() - x_expected

        # Here we test that we don't have crazy interpolation error.
        self.assertLess(max(delta), .15)
        # And that it gets middle points a little better.
        self.assertLess(max(delta[15:-15]), .06)

    def test_interpolating_spline_derivs(self):
        n_cp = 12
        n_point = 21

        t = np.linspace(0, 3.0*np.pi, n_cp)
        tt = np.linspace(0, 3.0*np.pi, n_point)

        x = np.sin(t)

        for method in self.spline_methods:
            interp = InterpND(method=method, points=t, x_interp=tt)
            computed, deriv = interp.evaluate_spline(x, compute_derivative=True)

            x_expected = np.sin(tt)
            delta = computed.flatten() - x_expected

            # Here we test that we don't have crazy interpolation error.
            self.assertLess(max(delta), .25)

    def test_interp_akima_derivs(self):

        xcp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        ycp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        n = 23
        x = np.linspace(1.0, 12.0, n)

        interp = InterpND(method='akima', points=xcp, x_interp=x, delta_x=0.1)
        y, dy_dycp = interp.evaluate_spline(ycp, compute_derivative=True)

        deriv = np.array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                            0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                          [ 3.12508538e-01,  7.81237193e-01, -9.37457312e-02,
                            0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                          [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
                            0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                          [-1.92097534e-05,  7.05028815e-01,  3.39990395e-01,
                           -4.50000000e-02,  0.00000000e+00,  0.00000000e+00],
                          [-1.70753364e-05,  3.80025613e-01,  7.39991462e-01,
                           -1.20000000e-01,  0.00000000e+00,  0.00000000e+00],
                          [-6.40325114e-06,  1.15009605e-01,  1.01999680e+00,
                           -1.35000000e-01,  0.00000000e+00,  0.00000000e+00],
                          [ 0.00000000e+00,  1.11022302e-16,  1.00000000e+00,
                            0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                          [ 0.00000000e+00, -5.62500000e-03,  7.60412946e-01,
                            2.45667949e-01, -5.30632175e-04,  7.47369260e-05],
                          [ 0.00000000e+00, -5.00000000e-03,  5.07767857e-01,
                            4.98447864e-01, -1.41501913e-03,  1.99298469e-04],
                          [ 0.00000000e+00, -1.87500000e-03,  2.51238839e-01,
                            7.52003846e-01, -1.59189652e-03,  2.24210778e-04],
                          [ 0.00000000e+00,  0.00000000e+00,  1.11022302e-16,
                            1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                          [ 0.00000000e+00,  0.00000000e+00, -2.10964627e-01,
                            1.19119941e+00,  2.02602810e-02, -4.95062934e-04],
                          [ 0.00000000e+00,  0.00000000e+00, -3.55003720e-01,
                            1.28195585e+00,  7.41473347e-02, -1.09946322e-03],
                          [ 0.00000000e+00,  0.00000000e+00, -4.35442243e-01,
                            1.27731946e+00,  1.59810592e-01, -1.68780891e-03],
                          [ 0.00000000e+00,  0.00000000e+00, -4.55605159e-01,
                            1.18234038e+00,  2.75399483e-01, -2.13470805e-03],
                          [ 0.00000000e+00,  0.00000000e+00, -4.18817429e-01,
                            1.00206876e+00,  4.19063438e-01, -2.31476868e-03],
                          [ 0.00000000e+00,  0.00000000e+00, -3.28404018e-01,
                            7.41554727e-01,  5.88951889e-01, -2.10259885e-03],
                          [ 0.00000000e+00,  0.00000000e+00, -1.87689887e-01,
                            4.05848427e-01,  7.83214266e-01, -1.37280661e-03],
                          [ 0.00000000e+00,  0.00000000e+00,  2.22044605e-16,
                            0.00000000e+00,  1.00000000e+00,  4.33680869e-19],
                          [ 0.00000000e+00,  0.00000000e+00,  1.18164062e-01,
                           -2.58789062e-01,  1.05371094e+00,  8.69140625e-02],
                          [ 0.00000000e+00,  0.00000000e+00,  1.05034722e-01,
                           -2.50868056e-01,  8.32465278e-01,  3.13368056e-01],
                          [ 0.00000000e+00,  0.00000000e+00,  3.93880208e-02,
                           -1.17513021e-01,  4.44986979e-01,  6.33138021e-01],
                          [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                            0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

        assert_near_equal(deriv, dy_dycp, tolerance=1e-6)

    def test_scipy_auto_reduce_spline_order(self):
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

        interp = InterpND(method='scipy_cubic', points=points, values=values)

        # first dimension (x) should be reduced to k=1 (linear)
        self.assertEqual(interp.table._ki[0], 1)

        # should operate as normal
        x = np.array([0.5, 0, 1001])
        result = interp.interpolate(x)
        assert_near_equal(result, -0.046325695741704434, tolerance=1e-5)

        interp = InterpND(method='scipy_slinear', points=points, values=values)
        value1 = interp.interpolate(x)

        # cycle through different methods that require order reduction
        # in the first dimension
        interp = InterpND(method='scipy_quintic', points=points, values=values)
        value2, grad2 = interp.interpolate(x, compute_derivative=True)
        interp = InterpND(method='scipy_cubic', points=points, values=values)
        value3, grad3 = interp.interpolate(x, compute_derivative=True)

        # values from different methods should be different
        self.assertTrue(value1[0] != value2[0])
        self.assertTrue(value2[0] != value3[0])

    def test_NaN_exception(self):
        np.random.seed(1234)
        x = np.linspace(0, 2, 5)
        y = np.linspace(0, 1, 7)
        values = np.random.rand(5, 7)
        interp = InterpND(method='slinear', points=(x, y), values=values)

        with self.assertRaises(OutOfBoundsError) as cm:
            interp.interpolate(np.array([1, np.nan]))

        err = cm.exception

        self.assertEqual(str(err), 'One of the requested xi contains a NaN')
        self.assertEqual(err.idx, 1)
        self.assertTrue(np.isnan(err.value))
        self.assertEqual(err.lower, 0)
        self.assertEqual(err.upper, 1)

    def test_interp_1Dflat(self):

        x = np.array([0.,1.,2.,3.,4.])
        y = x**2
        f = InterpND(points=x, values=y)

        computed = f.interpolate(np.array([2.5, 3.5]))
        assert_equal_arrays(computed, np.array([6.5, 12.5]))

    def test_interp_1Dflat_list_points(self):

        x = np.array([0.,1.,2.,3.,4.])
        y = x**2
        f = InterpND(points=[x], values=y)

        computed = f.interpolate(np.array([2.5, 3.5]))
        assert_equal_arrays(computed, np.array([6.5, 12.5]))

    def test_interp_1Dflat_list_x(self):

        x = np.array([0.,1.,2.,3.,4.])
        y = x**2
        f = InterpND(points=x, values=y)

        computed = f.interpolate([np.array([2.5]), np.array([3.5])])
        assert_equal_arrays(computed, np.array([6.5, 12.5]))

    def test_error_messages(self):
        points, values = self._get_sample_4d_large()

        with self.assertRaises(ValueError) as cm:
            interp = InterpND(method='junk', points=points, values=values.tolist())

        msg = ('Interpolation method "junk" is not defined. Valid methods are')
        self.assertTrue(str(cm.exception).startswith(msg))

        with self.assertRaises(ValueError) as cm:
            interp = InterpND(method=points, points=points, values=values.tolist())

        msg = ("Argument 'method' should be a string.")
        self.assertTrue(str(cm.exception).startswith(msg))

        with self.assertRaises(ValueError) as cm:
            interp = InterpND(method='slinear', points=points, values=values.tolist()[1])

        msg = ('There are 4 point arrays, but values has 3 dimensions')
        self.assertEqual(str(cm.exception), msg)

        badpoints = deepcopy(points)
        badpoints[0][0] = 55.0
        with self.assertRaises(ValueError) as cm:
            interp = InterpND(method='slinear', points=badpoints, values=values.tolist())

        msg = ('The points in dimension 0 must be strictly ascending')
        self.assertEqual(str(cm.exception), msg)

        badpoints[0] = np.vstack((np.arange(6), np.arange(6)))
        with self.assertRaises(ValueError) as cm:
            interp = InterpND(method='slinear', points=badpoints, values=values.tolist())

        msg = ('The points in dimension 0 must be 1-dimensional')
        self.assertEqual(str(cm.exception), msg)

        badpoints[0] = (np.arange(4))
        with self.assertRaises(ValueError) as cm:
            interp = InterpND(method='slinear', points=badpoints, values=values.tolist())

        msg = ('There are 4 points and 6 values in dimension 0')
        self.assertEqual(str(cm.exception), msg)

        badvalues = np.array(values, dtype=complex)
        with self.assertRaises(ValueError) as cm:
            interp = InterpND(method='scipy_slinear', points=badpoints, values=badvalues.tolist())

        msg = ("Interpolation method 'scipy_slinear' does not support complex points or values.")
        self.assertEqual(str(cm.exception), msg)

        interp = InterpND(method='slinear', points=points, values=values.tolist())
        x = [0.5, 0, 0.5, 0.9]

        with self.assertRaises(KeyError) as cm:
            interp = InterpND(method='slinear', points=points, values=values.tolist(), bad_arg=1)

        msg = ("\"InterpLinear: Option 'bad_arg' cannot be set because it has not been declared.")
        self.assertTrue(str(cm.exception).startswith(msg))

        # Bspline not supported for tables.
        points, values, func, df = self. _get_sample_2d()
        with self.assertRaises(ValueError) as cm:
            interp = InterpND(method='bsplines', points=points, values=values)

        msg = "Method 'bsplines' is not supported for table interpolation."
        self.assertTrue(str(cm.exception).startswith(msg))

        with self.assertRaises(ValueError) as cm:
            interp = InterpND(method='bsplines', points=points)

        msg = "Either 'values' or 'x_interp' must be specified."
        self.assertTrue(str(cm.exception).startswith(msg))

    def test_derivative_hysteresis_bug(self):
        alt = np.array([-1000, 0, 1000], dtype=float)
        rho = np.array([0.00244752, 0.00237717, 0.00230839])

        rho_interp = InterpND(method='slinear', points=alt, values=rho, extrapolate=True)

        x = 0.0
        _, dval1 = rho_interp.interpolate([x], compute_derivative=True)

        x = 0.5
        _, dval2 = rho_interp.interpolate([x], compute_derivative=True)

        x = 0.0
        _, dval3 = rho_interp.interpolate([x], compute_derivative=True)

        x = -0.5
        _, dval4 = rho_interp.interpolate([x], compute_derivative=True)

        x = 0.0
        _, dval5 = rho_interp.interpolate([x], compute_derivative=True)

        assert_near_equal(dval3 - dval1, np.array([[0.0]]))
        assert_near_equal(dval5 - dval1, np.array([[0.0]]))

    def test_cs_across_interpND(self):
        p1 = np.linspace(0, 100, 5)
        p2 = np.linspace(-10, 10, 5)

        # can use meshgrid to create a 2D array of test data
        P1, P2 = np.meshgrid(p1, p2, indexing='ij')
        f_p = np.sqrt(P1) + P2

        nn = 3

        x1 = np.linspace(-2, 101, nn)
        x2 = np.linspace(-10.5, 11, nn)
        X1, X2 = np.meshgrid(x1, x2, indexing='ij')
        x = np.zeros((nn**2, 2))
        x[:, 0] = X1.ravel()
        x[:, 1] = X2.ravel()

        class CompUsesInterp(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('interp', None)

            def setup(self):
                method = self.options['interp']

                self.add_input('p1', p1)
                self.add_input('p2', p2)
                self.add_input('f_p', f_p)
                self.add_input('x', x)

                self.add_output('f', np.ones((nn**2, )))
                self.add_output('df', np.ones((nn**2, 2)))

            def setup_partials(self):
                self.declare_partials('*', '*', method='cs')

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                method = self.options['interp']

                p1 = inputs['p1']
                p2 = inputs['p2']
                f_p = inputs['f_p']
                x = inputs['x']

                interp = InterpND(points=(p1, p2), values=f_p, method=method, extrapolate=True)
                f, df_dx = interp.interpolate(x, compute_derivative=True)

                outputs['f'] = f
                outputs['df'] = df_dx


        for method in self.interp_methods:

            if method.startswith('scipy') or method == 'akima':
                # These aren't supported (yet)
                continue

            prob = om.Problem()
            model = prob.model

            model.add_subsystem('interp', CompUsesInterp(interp=method))

            prob.setup(force_alloc_complex=True)
            prob.run_model()

            derivs = prob.check_partials(method='fd', out_stream=None)
            assert_check_partials(derivs, atol=1e-3, rtol=1e-4)


class TestInterpNDFixedPython(unittest.TestCase):
    """Tests for efficient fixed-grid interpolation."""

    def test_error_messages(self):
        p = np.array([0, 1, 2])
        f = 2.0 * np.array([0, 1, 2])

        with self.assertRaises(ValueError) as cm:
            interp = InterpND(method='1D-akima', points=p, values=f)

        msg = "There are 3 points in a data dimension, but method '1D-akima' requires at least 4 points per dimension."
        self.assertTrue(str(cm.exception).startswith(msg))

        with self.assertRaises(ValueError) as cm:
            interp = InterpND(method='3D-slinear', points=p, values=f)

        msg = "There are 1 dimensions, but method '3D-slinear' only works with a fixed table dimension of 3."
        self.assertTrue(str(cm.exception).startswith(msg))

    def test_trilinear(self):
        # Test fixed 3D-slinear vs general equivalent.

        p1 = np.linspace(0, 100, 25)
        p2 = np.linspace(-10, 10, 15)
        p3 = np.linspace(0, 1, 12)

        # can use meshgrid to create a 3D array of test data
        P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
        f_p = np.sqrt(P1) + P2 * P3

        x1 = np.linspace(-2, 101, 5)
        x2 = np.linspace(-10.5, 11, 5)
        x3 = np.linspace(-0.2, 1.1, 5)
        X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
        x = np.zeros((125, 3))
        x[:, 0] = X1.ravel()
        x[:, 1] = X2.ravel()
        x[:, 2] = X3.ravel()

        interp = InterpND(points=(p1, p2, p3), values=f_p, method='3D-slinear', extrapolate=True)
        f, df_dx = interp.interpolate(x, compute_derivative=True)

        interp_base = InterpND(points=(p1, p2, p3), values=f_p, method='slinear', extrapolate=True)
        f_base, df_dx_base = interp_base.interpolate(x, compute_derivative=True)

        assert_near_equal(f, f_base, 1e-11)
        assert_near_equal(df_dx, df_dx_base, 1e-11)

        # Test non-vectorized.
        for j, x_i in enumerate(x):
            interp = InterpND(points=(p1, p2, p3), values=f_p, method='3D-slinear', extrapolate=True)
            f, df_dx = interp.interpolate(x_i, compute_derivative=True)

            assert_near_equal(f, f_base[j], 1e-11)
            assert_near_equal(df_dx[0], df_dx_base[j, :], 1e-11)

    def test_3Dslinear(self):
        # Test fixed 3D-slinear vs general equivalent.

        p1 = np.linspace(0, 100, 25)
        p2 = np.linspace(-10, 10, 15)
        p3 = np.linspace(0, 1, 12)

        # can use meshgrid to create a 3D array of test data
        P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
        f_p = np.sqrt(P1) + P2 * P3

        x1 = np.linspace(-2, 101, 5)
        x2 = np.linspace(-10.5, 11, 5)
        x3 = np.linspace(-0.2, 1.1, 5)
        X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
        x = np.zeros((125, 3))
        x[:, 0] = X1.ravel()
        x[:, 1] = X2.ravel()
        x[:, 2] = X3.ravel()

        interp = InterpND(points=(p1, p2, p3), values=f_p, method='3D-slinear', extrapolate=True)
        f, df_dx = interp.interpolate(x, compute_derivative=True)

        interp_base = InterpND(points=(p1, p2, p3), values=f_p, method='slinear', extrapolate=True)
        f_base, df_dx_base = interp_base.interpolate(x, compute_derivative=True)

        assert_near_equal(f, f_base, 1e-11)
        assert_near_equal(df_dx, df_dx_base, 1e-11)

        # Test non-vectorized.
        for j, x_i in enumerate(x):
            interp = InterpND(points=(p1, p2, p3), values=f_p, method='3D-slinear', extrapolate=True)
            f, df_dx = interp.interpolate(x_i, compute_derivative=True)

            assert_near_equal(f, f_base[j], 1e-11)
            assert_near_equal(df_dx[0], df_dx_base[j, :], 1e-11)

    def test_2Dslinear(self):
        # Test fixed 3D-slinear vs general equivalent.

        p1 = np.linspace(0, 100, 25)
        p2 = np.linspace(-10, 10, 15)

        # can use meshgrid to create a 3D array of test data
        P1, P2 = np.meshgrid(p1, p2, indexing='ij')
        f_p = np.sqrt(P1) + (0.1 * P2) ** 2

        x1 = np.linspace(-2, 101, 5)
        x2 = np.linspace(-10.5, 11, 5)
        X1, X2 = np.meshgrid(x1, x2, indexing='ij')
        x = np.zeros((25, 2))
        x[:, 0] = X1.ravel()
        x[:, 1] = X2.ravel()

        interp = InterpND(points=(p1, p2), values=f_p, method='2D-slinear', extrapolate=True)
        f, df_dx = interp.interpolate(x, compute_derivative=True)

        interp_base = InterpND(points=(p1, p2), values=f_p, method='slinear', extrapolate=True)
        f_base, df_dx_base = interp_base.interpolate(x, compute_derivative=True)

        assert_near_equal(f, f_base, 1e-11)
        assert_near_equal(df_dx, df_dx_base, 1e-11)

        # Test non-vectorized.
        for j, x_i in enumerate(x):
            interp = InterpND(points=(p1, p2), values=f_p, method='2D-slinear', extrapolate=True)
            f, df_dx = interp.interpolate(x_i, compute_derivative=True)

            assert_near_equal(f, f_base[j], 1e-11)
            assert_near_equal(df_dx[0], df_dx_base[j, :], 1e-11)

    def test_1Dakima(self):
        # Test 1D-akima vs general equivalent.

        p = np.linspace(0, 100, 25)
        f_p = np.cos(p * np.pi * 0.5)
        x = np.linspace(-1, 101, 33)

        interp = InterpND(points=p, values=f_p, method='1D-akima', extrapolate=True)
        f, df_dx = interp.interpolate(x, compute_derivative=True)

        interp_base = InterpND(points=p, values=f_p, method='akima', extrapolate=True)
        f_base, df_dx_base = interp_base.interpolate(x, compute_derivative=True)

        assert_near_equal(f, f_base, 1e-13)
        assert_near_equal(df_dx, df_dx_base.ravel(), 1e-13)

        # Test non-vectorized.
        for j, x_i in enumerate(x):
            interp = InterpND(points=p, values=f_p, method='1D-akima', extrapolate=True)
            f, df_dx = interp.interpolate(x_i, compute_derivative=True)

            assert_near_equal(f, f_base[j], 1e-13)

            # Compare abs error since deriv is near zero in some spots.
            abs_err = np.abs(df_dx[0] -  df_dx_base[j])
            assert_near_equal(abs_err, 0.0, 1e-13)

    def test_1Dslinear(self):
        # Test 1D-akima vs general equivalent.

        p = np.linspace(0, 100, 25)
        f_p = np.cos(p * np.pi * 0.5)
        x = np.linspace(-1, 101, 33) + 0.01

        interp = InterpND(points=p, values=f_p, method='1D-slinear', extrapolate=True)
        f, df_dx = interp.interpolate(x, compute_derivative=True)

        interp_base = InterpND(points=p, values=f_p, method='slinear', extrapolate=True)
        f_base, df_dx_base = interp_base.interpolate(x, compute_derivative=True)

        assert_near_equal(f, f_base, 1e-13)
        assert_near_equal(df_dx, df_dx_base, 1e-13)

        # Test non-vectorized.
        for j, x_i in enumerate(x):
            interp = InterpND(points=p, values=f_p, method='1D-slinear', extrapolate=True)
            f, df_dx = interp.interpolate(x_i, compute_derivative=True)

            assert_near_equal(f, f_base[j], 1e-13)

            # Compare abs error since deriv is near zero in some spots.
            abs_err = np.abs(df_dx[0] -  df_dx_base[j])
            assert_near_equal(abs_err, 0.0, 1e-13)

    def test_1Dlagrange2(self):
        # Test 1Dlagrange2 vs general equivalent.

        p = np.linspace(0, 100, 25)
        f_p = np.cos(p * np.pi * 0.5)
        x = np.linspace(-1, 101, 33) + 0.01

        interp = InterpND(points=p, values=f_p, method='1D-lagrange2', extrapolate=True)
        f, df_dx = interp.interpolate(x, compute_derivative=True)

        interp_base = InterpND(points=p, values=f_p, method='lagrange2', extrapolate=True)
        f_base, df_dx_base = interp_base.interpolate(x, compute_derivative=True)

        assert_near_equal(f, f_base, 1e-13)
        assert_near_equal(df_dx, df_dx_base, 3e-13)

        # Test non-vectorized.
        for j, x_i in enumerate(x):
            interp = InterpND(points=p, values=f_p, method='1D-lagrange2', extrapolate=True)
            f, df_dx = interp.interpolate(x_i, compute_derivative=True)

            assert_near_equal(f, f_base[j], 1e-13)

            # Compare abs error since deriv is near zero in some spots.
            abs_err = np.abs(df_dx[0] -  df_dx_base[j])
            assert_near_equal(abs_err, 0.0, 3e-13)

    def test_2Dlagrange2(self):
        # Test 2D-lagrange2 vs general equivalent.

        p1 = np.linspace(0, 100, 25)
        p2 = np.linspace(-10, 10, 15)

        # can use meshgrid to create a 3D array of test data
        P1, P2 = np.meshgrid(p1, p2, indexing='ij')
        f_p = np.sqrt(P1) + P2

        x1 = np.linspace(-2, 101, 5)
        x2 = np.linspace(-10.5, 11, 5)
        X1, X2 = np.meshgrid(x1, x2, indexing='ij')
        x = np.zeros((25, 2))
        x[:, 0] = X1.ravel()
        x[:, 1] = X2.ravel()

        interp = InterpND(points=(p1, p2), values=f_p, method='2D-lagrange2', extrapolate=True)
        f, df_dx = interp.interpolate(x, compute_derivative=True)

        interp_base = InterpND(points=(p1, p2), values=f_p, method='lagrange2', extrapolate=True)
        f_base, df_dx_base = interp_base.interpolate(x, compute_derivative=True)

        assert_near_equal(f, f_base, 1e-11)
        assert_near_equal(df_dx, df_dx_base, 3e-11)

        # Test non-vectorized.
        for j, x_i in enumerate(x):
            interp = InterpND(points=(p1, p2), values=f_p, method='2D-lagrange2', extrapolate=True)
            f, df_dx = interp.interpolate(x_i, compute_derivative=True)

            assert_near_equal(f, f_base[j], 2e-10)
            assert_near_equal(df_dx[0], df_dx_base[j, :], 2e-10)

    def test_3Dlagrange2(self):
        # Test 3D-lagrange2 vs general equivalent.

        p1 = np.linspace(0, 100, 25)
        p2 = np.linspace(-10, 10, 15)
        p3 = np.linspace(0, 1, 12)

        # can use meshgrid to create a 3D array of test data
        P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
        f_p = np.sqrt(P1) + P2 * P3

        x1 = np.linspace(-2, 101, 5)
        x2 = np.linspace(-10.5, 11, 5)
        x3 = np.linspace(-0.2, 1.1, 5)
        X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
        x = np.zeros((125, 3))
        x[:, 0] = X1.ravel()
        x[:, 1] = X2.ravel()
        x[:, 2] = X3.ravel()

        interp = InterpND(points=(p1, p2, p3), values=f_p, method='3D-lagrange2', extrapolate=True)
        f, df_dx = interp.interpolate(x, compute_derivative=True)

        interp_base = InterpND(points=(p1, p2, p3), values=f_p, method='lagrange2', extrapolate=True)
        f_base, df_dx_base = interp_base.interpolate(x, compute_derivative=True)

        assert_near_equal(f, f_base, 1e-11)
        assert_near_equal(df_dx, df_dx_base, 3e-11)

        # Test non-vectorized.
        for j, x_i in enumerate(x):
            interp = InterpND(points=(p1, p2, p3), values=f_p, method='3D-lagrange2', extrapolate=True)
            f, df_dx = interp.interpolate(x_i, compute_derivative=True)

            assert_near_equal(f, f_base[j], 2e-10)
            assert_near_equal(df_dx[0], df_dx_base[j, :], 2e-10)

    def test_1Dlagrange3(self):
        # Test 1D-lagrange3 vs general equivalent.

        p = np.linspace(0, 100, 25)
        f_p = np.cos(p * np.pi * 0.5)
        x = np.linspace(-1, 101, 33) + 0.01

        interp = InterpND(points=p, values=f_p, method='1D-lagrange3', extrapolate=True)
        f, df_dx = interp.interpolate(x, compute_derivative=True)

        interp_base = InterpND(points=p, values=f_p, method='lagrange3', extrapolate=True)
        f_base, df_dx_base = interp_base.interpolate(x, compute_derivative=True)

        assert_near_equal(f, f_base, 1e-13)
        assert_near_equal(df_dx, df_dx_base, 3e-13)

        # Test non-vectorized.
        for j, x_i in enumerate(x):
            interp = InterpND(points=p, values=f_p, method='1D-lagrange3', extrapolate=True)
            f, df_dx = interp.interpolate(x_i, compute_derivative=True)

            assert_near_equal(f, f_base[j], 1e-13)

            # Compare abs error since deriv is near zero in some spots.
            abs_err = np.abs(df_dx[0] -  df_dx_base[j])
            assert_near_equal(abs_err, 0.0, 3e-13)

    def test_2Dlagrange3(self):
        # Test 2D-lagrange3 vs general equivalent.

        p1 = np.linspace(0, 100, 25)
        p2 = np.linspace(-10, 10, 15)

        # can use meshgrid to create a 3D array of test data
        P1, P2 = np.meshgrid(p1, p2, indexing='ij')
        f_p = np.sqrt(P1) + P2

        x1 = np.linspace(-2, 101, 5)
        x2 = np.linspace(-10.5, 11, 5)
        X1, X2 = np.meshgrid(x1, x2, indexing='ij')
        x = np.zeros((25, 2))
        x[:, 0] = X1.ravel()
        x[:, 1] = X2.ravel()

        interp = InterpND(points=(p1, p2), values=f_p, method='2D-lagrange3', extrapolate=True)
        f, df_dx = interp.interpolate(x, compute_derivative=True)

        interp_base = InterpND(points=(p1, p2), values=f_p, method='lagrange3', extrapolate=True)
        f_base, df_dx_base = interp_base.interpolate(x, compute_derivative=True)

        assert_near_equal(f, f_base, 1e-11)
        assert_near_equal(df_dx, df_dx_base, 3e-11)

        # Test non-vectorized.
        for j, x_i in enumerate(x):
            interp = InterpND(points=(p1, p2), values=f_p, method='2D-lagrange3', extrapolate=True)
            f, df_dx = interp.interpolate(x_i, compute_derivative=True)

            assert_near_equal(f, f_base[j], 2e-10)
            assert_near_equal(df_dx[0], df_dx_base[j, :], 2e-10)

    def test_3Dlagrange3(self):
        # Test 3D-lagrange3 vs general equivalent.

        p1 = np.linspace(0, 100, 25)
        p2 = np.linspace(-10, 10, 15)
        p3 = np.linspace(0, 1, 12)

        # can use meshgrid to create a 3D array of test data
        P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
        f_p = np.sqrt(P1) + P2 * P3

        x1 = np.linspace(-2, 101, 5)
        x2 = np.linspace(-10.5, 11, 5)
        x3 = np.linspace(-0.2, 1.1, 5)
        X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
        x = np.zeros((125, 3))
        x[:, 0] = X1.ravel()
        x[:, 1] = X2.ravel()
        x[:, 2] = X3.ravel()

        interp = InterpND(points=(p1, p2, p3), values=f_p, method='3D-lagrange3', extrapolate=True)
        f, df_dx = interp.interpolate(x, compute_derivative=True)

        interp_base = InterpND(points=(p1, p2, p3), values=f_p, method='lagrange3', extrapolate=True)
        f_base, df_dx_base = interp_base.interpolate(x, compute_derivative=True)

        assert_near_equal(f, f_base, 1e-11)
        assert_near_equal(df_dx, df_dx_base, 3e-11)

        # Test non-vectorized.
        for j, x_i in enumerate(x):
            interp = InterpND(points=(p1, p2, p3), values=f_p, method='3D-lagrange3', extrapolate=True)
            f, df_dx = interp.interpolate(x_i, compute_derivative=True)

            assert_near_equal(f, f_base[j], 2e-10)
            assert_near_equal(df_dx[0], df_dx_base[j, :], 2e-10)

    def test_deprecated_methods(self):

        p1 = np.linspace(0, 100, 5)
        p2 = np.linspace(-10, 10, 3)
        p3 = np.linspace(0, 1, 3)

        # can use meshgrid to create a 3D array of test data
        P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
        f_p = np.sqrt(P1) + P2 * P3

        x1 = np.linspace(-2, 101, 5)
        x2 = np.linspace(-10.5, 11, 5)
        x3 = np.linspace(-0.2, 1.1, 5)
        X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
        x = np.zeros((125, 3))
        x[:, 0] = X1.ravel()
        x[:, 1] = X2.ravel()
        x[:, 2] = X3.ravel()

        msg = "The 'trilinear' method has been renamed to '3D-slinear'."
        with assert_warning(OMDeprecationWarning, msg):
            InterpND(points=(p1, p2, p3), values=f_p, method='trilinear', extrapolate=True)

        msg = "The 'akima1D' method has been renamed to '1D-akima'."
        with assert_warning(OMDeprecationWarning, msg):
            InterpND(points=p1, values=p1, method='akima1D')


if __name__ == '__main__':
    unittest.main()
