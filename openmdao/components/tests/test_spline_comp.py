"""
Unit tests for the spline interpolator component.
"""
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_rel_error
from openmdao.utils.general_utils import printoptions
from openmdao.utils.spline_distributions import SplineDistribution
from openmdao.components.interp_util.interp import InterpND

class SplineTestCase(unittest.TestCase):

    def setUp(self):
        self.x_cp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        self.y_cp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        self.y_cp2 = np.array([1.0, 5.0, 7.0, 8.0, 13.0, 16.0])
        self.n = 50
        self.x = np.linspace(1.0, 12.0, self.n)

        self.prob = om.Problem()

    def test_simple_spline(self):

        comp = om.SplineComp(method='akima', x_cp_val=self.x_cp, x_interp=self.x)
        self.prob.model.add_subsystem('akima1', comp)

        comp.add_spline(y_cp_name='ycp', y_interp_name='y_val', y_cp_val=self.y_cp)

        self.prob.setup(force_alloc_complex=True)
        self.prob.run_model()

    def test_multiple_splines(self):

        comp = om.SplineComp(method='akima', x_cp_val=self.x_cp, x_interp=self.x, x_interp_name='x_val')
        self.prob.model.add_subsystem('akima1', comp)

        comp.add_spline(y_cp_name='ycp1', y_interp_name='y_val1', y_cp_val=self.y_cp)
        comp.add_spline(y_cp_name='ycp2', y_interp_name='y_val2', y_cp_val=self.y_cp2)

        self.prob.setup(force_alloc_complex=True)
        self.prob.run_model()

    def test_akima_interp_options(self):

        akima_option = {'delta_x': 0.1, 'eps': 1e-30}
        comp = om.SplineComp(method='akima', x_cp_val=self.x_cp, x_interp=self.x, x_cp_name='xcp',
                            x_interp_name='x_val', x_units='km', interp_options=akima_option)

        self.prob.model.add_subsystem('atmosphere', comp)

        comp.add_spline(y_cp_name='alt_cp', y_interp_name='alt', y_cp_val=self.y_cp, y_units='kft')

        self.prob.setup(force_alloc_complex=True)
        self.prob.run_model()

    def test_akima_backward_compatibility(self):

        comp = om.SplineComp(method='akima', x_cp_val=self.x_cp, x_interp=self.x,
                             interp_options={'delta_x': 0.1})
        comp.add_spline(y_cp_name='ycp', y_interp_name='y_val', y_cp_val=self.y_cp)

        self.prob.model.add_subsystem('akima1', comp)

        self.prob.setup(force_alloc_complex=True)
        self.prob.run_model()

        # Verification array from AkimaSplineComp
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

        assert_array_almost_equal(akima_y.flatten(), self.prob['akima1.y_val'].flatten())

        derivs = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(derivs, atol=1e-14, rtol=1e-14)

    def test_no_ycp_val(self):

        comp = om.SplineComp(method='akima', x_cp_val=self.x_cp, x_interp=self.x)

        comp.add_spline(y_cp_name='ycp', y_interp_name='y_val')
        self.prob.model.add_subsystem('akima1', comp)

        self.prob.setup(force_alloc_complex=True)
        self.prob.run_model()

    def test_vectorized(self):

        xcp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        ycp = np.array([[5.0, 12.0, 14.0, 16.0, 21.0, 29.0],
                        [7.0, 13.0, 9.0, 6.0, 12.0, 14.0]])
        n = 12
        x = np.linspace(1.0, 12.0, n)

        comp = om.SplineComp(method='akima', vec_size=2, x_cp_val=xcp, x_interp=x,
                             interp_options={'delta_x': 0.1})

        comp.add_spline(y_cp_name='ycp', y_interp_name='y_val', y_cp_val=ycp)
        self.prob.model.add_subsystem('akima1', comp)

        self.prob.setup(force_alloc_complex=True)
        self.prob.run_model()

        y = np.array([[ 5.        , 12.        , 13.01239669, 14.        , 14.99888393,
                        16.        , 17.06891741, 18.26264881, 19.5750558 , 21.        ,
                        24.026042, 29.        ],
                      [ 7.        , 13.        , 11.02673797,  9.        ,  7.09090909,
                        6.        ,  6.73660714,  8.46428571, 10.45982143, 12.        ,
                        13.08035714, 14.        ]])

        assert_array_almost_equal(y.flatten(), self.prob['akima1.y_val'].flatten())

        derivs = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(derivs, atol=1e-14, rtol=1e-14)

    def test_standalone_interp(self):

        interp = InterpND(points=[self.x_cp], values=self.y_cp, interp_method='akima', x_interp=self.x,
                          bounds_error=True)
        y = interp.evaluate_spline(np.expand_dims(self.y_cp, axis=0))

        spline_comp_y_array = np.array([[
            5.        ,  7.21095802,  9.2182764 , 10.8183155 , 11.80743568,
            12.12244898, 12.34693878, 12.57142857, 12.79591837, 13.02040816,
            13.24489796, 13.46938776, 13.69387755, 13.91836735, 14.14285714,
            14.36734694, 14.59183673, 14.81632653, 15.04081633, 15.26530612,
            15.48979592, 15.71428571, 15.93877551, 16.16506444, 16.39785941,
            16.63732612, 16.8833762 , 17.13592126, 17.3948729 , 17.66014276,
            17.93164243, 18.20928355, 18.49297771, 18.78263654, 19.07817165,
            19.37949466, 19.68651717, 19.99915081, 20.31730719, 20.64089793,
            20.96983463, 21.37579297, 21.94811407, 22.66809748, 23.51629844,
            24.47327219, 25.51957398, 26.63575905, 27.80238264, 29.        ]])

        assert_array_almost_equal(spline_comp_y_array.flatten(), y.flatten())

    def test_bspline_interp_basic(self):
        prob = om.Problem()
        model = prob.model

        n_cp = 80
        n_point = 160

        t = np.linspace(0, 3.0*np.pi, n_cp)
        tt = np.linspace(0, 3.0*np.pi, n_point)
        x = np.sin(t)

        model.add_subsystem('px', om.IndepVarComp('x', val=x))

        bspline_options = {'order': 4}
        comp = om.SplineComp(method='bsplines', x_cp_val=t, x_interp=tt, x_cp_name='xcp',
                            x_interp_name='x_val', x_units='km',
                            interp_options=bspline_options)

        prob.model.add_subsystem('interp', comp)

        comp.add_spline(y_cp_name='h_cp', y_interp_name='h', y_cp_val=x, y_units='km')

        model.connect('px.x', 'interp.h_cp')

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        xx = prob['interp.h'].flatten()
        tt = np.linspace(0, 3.0*np.pi, n_point)

        x_expected = np.sin(tt)
        delta = xx - x_expected

        # Here we test that we don't have crazy interpolation error.
        self.assertLess(max(delta), .15)
        # And that it gets middle points a little better.
        self.assertLess(max(delta[15:-15]), .06)

        derivs = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(derivs, atol=1e-14, rtol=1e-14)

    def test_bsplines_vectorized(self):
        prob = om.Problem()
        model = prob.model

        n_cp = 5
        n_point = 10

        t = np.linspace(0, 0.5 * np.pi, n_cp)
        tt = np.linspace(0, 0.5 * np.pi, n_point)
        x = np.empty((2, n_cp))
        x[0, :] = np.sin(t)
        x[1, :] = 2.0 * np.sin(t)

        t_sin = (0.5 * (1.0 + np.sin(-0.5 * np.pi + 2.0 * tt))) * np.pi * 0.5

        model.add_subsystem('px', om.IndepVarComp('x', val=x))
        bspline_options = {'order': 4}
        comp = om.SplineComp(method='bsplines', x_cp_val=t, x_interp=t_sin, x_cp_name='xcp',
                            x_interp_name='x_val', x_units='km', vec_size=2,
                            interp_options=bspline_options)

        prob.model.add_subsystem('interp', comp)

        comp.add_spline(y_cp_name='h_cp', y_interp_name='h', y_cp_val=x, y_units='km')

        model.connect('px.x', 'interp.h_cp')

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        xx = prob['interp.h']

        with printoptions(precision=3, floatmode='fixed'):
            assert_rel_error(self, x[0, :], np.array([
                0., 0.38268343, 0.70710678, 0.92387953, 1.
            ]), 1e-5)
            assert_rel_error(self, x[1, :], 2.0*np.array([
                0., 0.38268343, 0.70710678, 0.92387953, 1.
            ]), 1e-5)

            assert_rel_error(self, xx[0, :], np.array([
                0., 0.06687281, 0.23486869, 0.43286622, 0.6062628,
                0.74821484, 0.86228902, 0.94134389, 0.98587725, 1.
            ]), 1e-5)
            assert_rel_error(self, xx[1, :], 2.0*np.array([
                0., 0.06687281, 0.23486869, 0.43286622, 0.6062628,
                0.74821484, 0.86228902, 0.94134389, 0.98587725, 1.
            ]), 1e-5)


        derivs = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(derivs, atol=1e-14, rtol=1e-14)


class SplineCompFeatureTestCase(unittest.TestCase):

    def test_basic_example(self):
        import numpy as np

        import openmdao.api as om

        xcp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        ycp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        n = 50
        x = np.linspace(1.0, 12.0, n)

        prob = om.Problem()

        akima_option = {'delta_x': 0.1}
        comp = om.SplineComp(method='akima', x_cp_val=xcp, x_interp=x, x_cp_name='xcp',
                             x_interp_name='x_val', interp_options=akima_option)

        prob.model.add_subsystem('akima1', comp)

        comp.add_spline(y_cp_name='ycp', y_interp_name='y_val', y_cp_val=ycp)

        prob.setup(force_alloc_complex=True)
        prob.run_model()

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

        assert_array_almost_equal(akima_y.flatten(), prob['akima1.y_val'].flatten())

    def test_multi_splines(self):

        import numpy as np

        import openmdao.api as om

        x_cp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        y_cp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        y_cp2 = np.array([1.0, 5.0, 7.0, 8.0, 13.0, 16.0])
        n = 50
        x = np.linspace(1.0, 12.0, n)

        prob = om.Problem()

        comp = om.SplineComp(method='akima', x_cp_val=x_cp, x_interp=x, x_interp_name='x_val')
        prob.model.add_subsystem('akima1', comp)

        comp.add_spline(y_cp_name='ycp1', y_interp_name='y_val1', y_cp_val=y_cp)
        comp.add_spline(y_cp_name='ycp2', y_interp_name='y_val2', y_cp_val=y_cp2)

        prob.setup(force_alloc_complex=True)
        prob.run_model()

    def test_spline_distribution_example(self):

        import numpy as np

        import openmdao.api as om

        s_dist = SplineDistribution()
        x_cp = np.linspace(0., 1., 6)
        y_cp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        n = 20
        x = s_dist.sine_distribution(x_cp, n, phase=np.pi)

        prob = om.Problem()

        comp = om.SplineComp(method='akima', x_cp_val=x_cp, x_interp=x)
        prob.model.add_subsystem('akima1', comp)

        comp.add_spline(y_cp_name='ycp', y_interp_name='y_val', y_cp_val=y_cp)

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        akima_y = np.array([[5.         , 5.32381994,  6.28062691 , 7.79410646 , 9.64169506, 11.35166363,
                             12.26525921, 12.99152288, 13.77257256, 14.58710327, 15.41289673, 16.28341046,
                             17.96032258, 20.14140712, 22.31181718, 24.40891577, 26.27368825, 27.74068235,
                             28.67782484, 29.        ]])

        assert_array_almost_equal(akima_y.flatten(), prob['akima1.y_val'].flatten())

    def test_standalone_interp_example(self):

        import numpy as np

        import openmdao.api as om
        from openmdao.components.interp_util.interp import InterpND

        xcp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        ycp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        n = 50
        x = np.linspace(1.0, 12.0, n)

        akima_options = {'delta_x': 0.1}
        interp = InterpND(points=[xcp], values=ycp, interp_method='akima', x_interp=x,
                          bounds_error=True, **akima_options)
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


if __name__ == '__main__':
    unittest.main()