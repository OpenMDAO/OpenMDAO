"""
Unit tests for the spline interpolator component.
"""
import unittest

import numpy as np

import openmdao.api as om
from openmdao.components.spline_comp import SPLINE_METHODS
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.general_utils import printoptions
from openmdao.utils.spline_distributions import cell_centered
from openmdao.components.interp_util.interp import InterpND


class SplineCompTestCase(unittest.TestCase):

    def setUp(self):
        self.x_cp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        self.y_cp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        self.y_cp2 = np.array([1.0, 5.0, 7.0, 8.0, 13.0, 16.0])
        self.n = 50
        self.x = np.linspace(1.0, 12.0, self.n)

        self.prob = om.Problem()

    def test_simple_spline(self):

        comp = om.SplineComp(method='akima', x_cp_val=self.x_cp, x_interp_val=self.x)
        self.prob.model.add_subsystem('akima1', comp)

        comp.add_spline(y_cp_name='ycp', y_interp_name='y_val', y_cp_val=self.y_cp)

        self.prob.setup(force_alloc_complex=True)
        self.prob.run_model()

    def test_multiple_splines(self):

        comp = om.SplineComp(method='akima', x_cp_val=self.x_cp, x_interp_val=self.x)
        self.prob.model.add_subsystem('akima1', comp)

        comp.add_spline(y_cp_name='ycp1', y_interp_name='y_val1', y_cp_val=self.y_cp)
        comp.add_spline(y_cp_name='ycp2', y_interp_name='y_val2', y_cp_val=self.y_cp2)

        self.prob.setup(force_alloc_complex=True)
        self.prob.run_model()

    def test_akima_interp_options(self):

        akima_option = {'delta_x': 0.1, 'eps': 1e-30}
        comp = om.SplineComp(method='akima', x_cp_val=self.x_cp, x_interp_val=self.x,
                             interp_options=akima_option)

        self.prob.model.add_subsystem('atmosphere', comp)

        comp.add_spline(y_cp_name='alt_cp', y_interp_name='alt', y_cp_val=self.y_cp, y_units='kft')

        self.prob.setup(force_alloc_complex=True)
        self.prob.run_model()

    def test_small_akima_spline_bug(self):
        # Fixes a bug that only occure for a 4 point spline.
        prob = om.Problem()

        num_cp = 4
        num_radial = 11
        comp = om.IndepVarComp()
        comp.add_output("chord_cp", units="m", val=np.array([0.1, 0.2, 0.3, 0.15]))
        comp.add_output("theta_cp", units="rad", val=np.array([1.0, 0.8, 0.6, 0.4]))
        prob.model.add_subsystem("inputs_comp", comp, promotes=["*"])

        x_cp = np.linspace(0.0, 1.0, num_cp)
        x_interp = cell_centered(num_radial, start=0.0, end=1.0)
        akima_options = {'delta_x': 0.1}
        comp = om.SplineComp(method='akima', interp_options=akima_options, x_cp_val=x_cp, x_interp_val=x_interp)
        comp.add_spline(y_cp_name='chord_cp', y_interp_name='chord_interp', y_units='m')
        comp.add_spline(y_cp_name='theta_cp', y_interp_name='theta_interp', y_units='rad')
        prob.model.add_subsystem('akima_comp', comp,
                                 promotes_inputs=['chord_cp', 'theta_cp'],
                                 promotes_outputs=['chord_interp', 'theta_interp'])
        prob.setup()

        # Make sure we don't get an exception
        prob.run_model()

    def test_akima_backward_compatibility(self):

        comp = om.SplineComp(method='akima', x_cp_val=self.x_cp, x_interp_val=self.x,
                             interp_options={'delta_x': 0.1})
        comp.add_spline(y_cp_name='ycp', y_interp_name='y_val', y_cp_val=self.y_cp)

        self.prob.model.add_subsystem('akima1', comp)

        self.prob.setup(force_alloc_complex=True)
        self.prob.run_model()

        # Verification array from openmdao 2.x using AkimaSplineComp
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

        assert_near_equal(akima_y.flatten(), self.prob['akima1.y_val'].flatten(), tolerance=1e-8)

        derivs = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(derivs, atol=1e-14, rtol=1e-14)

    def test_no_ycp_val(self):

        comp = om.SplineComp(method='akima', x_cp_val=self.x_cp, x_interp_val=self.x)

        comp.add_spline(y_cp_name='ycp', y_interp_name='y_val')
        self.prob.model.add_subsystem('akima1', comp)

        self.prob.setup(force_alloc_complex=True)
        self.prob.run_model()

    def test_vectorized_akima(self):

        xcp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        ycp = np.array([[5.0, 12.0, 14.0, 16.0, 21.0, 29.0],
                        [7.0, 13.0, 9.0, 6.0, 12.0, 14.0]])
        n = 12
        x = np.linspace(1.0, 12.0, n)

        comp = om.SplineComp(method='akima', vec_size=2, x_cp_val=xcp, x_interp_val=x,
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

        assert_near_equal(y.flatten(), self.prob['akima1.y_val'].flatten(), tolerance=1e-8)

        derivs = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(derivs, atol=1e-14, rtol=1e-14)

    def test_vectorized_all_derivs(self):

        xcp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        ycp = np.array([[5.0, 12.0, 14.0, 16.0, 21.0, 29.0],
                        [7.0, 13.0, 9.0, 6.0, 12.0, 14.0]])
        n = 12
        x = np.linspace(1.0, 12.0, n)

        for method in SPLINE_METHODS:

            prob = om.Problem()

            # These methods have their own test.
            if method in ['akima', 'bsplines']:
                continue

            opts = {}

            comp = om.SplineComp(method=method, vec_size=2, x_cp_val=xcp, x_interp_val=x,
                                 interp_options=opts)

            comp.add_spline(y_cp_name='ycp', y_interp_name='y_val', y_cp_val=ycp)
            prob.model.add_subsystem('interp1', comp)

            prob.setup(force_alloc_complex=True)
            prob.run_model()

            if method.startswith('scipy'):
                derivs = prob.check_partials(out_stream=None)
                assert_check_partials(derivs, atol=1e-7, rtol=1e-7)

            else:
                derivs = prob.check_partials(out_stream=None, method='cs')
                assert_check_partials(derivs, atol=1e-12, rtol=1e-12)

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
        comp = om.SplineComp(method='bsplines', x_interp_val=tt, num_cp=n_cp,
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
        comp = om.SplineComp(method='bsplines', x_interp_val=t_sin, num_cp=n_cp,
                             vec_size=2, interp_options=bspline_options)

        prob.model.add_subsystem('interp', comp)

        comp.add_spline(y_cp_name='h_cp', y_interp_name='h', y_cp_val=x, y_units='km')

        model.connect('px.x', 'interp.h_cp')

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        xx = prob['interp.h']

        with printoptions(precision=3, floatmode='fixed'):
            assert_near_equal(x[0, :], np.array([
                0., 0.38268343, 0.70710678, 0.92387953, 1.
            ]), 1e-5)
            assert_near_equal(x[1, :], 2.0*np.array([
                0., 0.38268343, 0.70710678, 0.92387953, 1.
            ]), 1e-5)

            assert_near_equal(xx[0, :], np.array([
                0., 0.06687281, 0.23486869, 0.43286622, 0.6062628,
                0.74821484, 0.86228902, 0.94134389, 0.98587725, 1.
            ]), 1e-5)
            assert_near_equal(xx[1, :], 2.0*np.array([
                0., 0.06687281, 0.23486869, 0.43286622, 0.6062628,
                0.74821484, 0.86228902, 0.94134389, 0.98587725, 1.
            ]), 1e-5)


        derivs = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(derivs, atol=1e-14, rtol=1e-14)

    def test_bspline_bug(self):
        # Tests a bug fix where the interp_options weren't passed into
        # the bspline interp comp
        bspline_options = {'order': 3}
        comp = om.SplineComp(method='bsplines', num_cp=6, x_interp_val=self.x,
                             interp_options=bspline_options)

        self.prob.model.add_subsystem('atmosphere', comp)

        comp.add_spline(y_cp_name='alt_cp', y_interp_name='alt', y_cp_val=self.y_cp, y_units='kft')

        self.prob.setup(force_alloc_complex=True)

        # If we set the bspline order to 3, then k should internally be 4
        self.assertEqual(comp.interps['alt'].table.k, 4)

    def test_error_messages(self):
        n_cp = 80
        n_point = 160

        t = np.linspace(0, 3.0*np.pi, n_cp)
        tt = np.linspace(0, 3.0*np.pi, n_point)
        x = np.sin(t)

        prob = om.Problem()

        comp = om.SplineComp(method='bsplines', x_interp_val=tt, x_cp_val=t)

        prob.model.add_subsystem('interp', comp)

        comp.add_spline(y_cp_name='h_cp', y_interp_name='h', y_cp_val=x, y_units='km')

        with self.assertRaises(ValueError) as cm:
            prob.setup()

        msg = "SplineComp (interp): 'x_cp_val' is not a valid option when using method 'bsplines'. "
        msg += "Set 'num_cp' instead."
        self.assertEqual(str(cm.exception), msg)

        prob = om.Problem()

        comp = om.SplineComp(method='akima', x_interp_val=tt, num_cp=n_cp, x_cp_val=t)

        prob.model.add_subsystem('interp', comp)

        comp.add_spline(y_cp_name='h_cp', y_interp_name='h', y_cp_val=x, y_units='km')

        with self.assertRaises(ValueError) as cm:
            prob.setup()

        msg = "SplineComp (interp): It is not valid to set both options 'x_cp_val' and 'num_cp'."
        self.assertEqual(str(cm.exception), msg)

        prob = om.Problem()

        comp = om.SplineComp(method='akima', x_interp_val=tt)

        prob.model.add_subsystem('interp', comp)

        comp.add_spline(y_cp_name='h_cp', y_interp_name='h', y_cp_val=x, y_units='km')

        with self.assertRaises(ValueError) as cm:
            prob.setup()

        msg = "SplineComp (interp): Either option 'x_cp_val' or 'num_cp' must be set."
        self.assertEqual(str(cm.exception), msg)

    def test_y_units(self):
        x_cp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        y_cp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])

        n = 50
        x = np.linspace(1.0, 12.0, n)

        prob = om.Problem()
        model = prob.model

        # Set options specific to akima
        akima_option = {'delta_x': 0.1, 'eps': 1e-30}

        comp = om.SplineComp(method='akima', x_cp_val=x_cp, x_interp_val=x,
                             interp_options=akima_option)

        prob.model.add_subsystem('atmosphere', comp)

        comp.add_spline(y_cp_name='alt_cp', y_interp_name='alt', y_cp_val=y_cp, y_units='kft')

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        output = prob.model.list_inputs(units=True)
        self.assertEqual(output[0][1]['units'], 'kft')

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
        comp = om.SplineComp(method='akima', x_cp_val=xcp, x_interp_val=x,
                             interp_options=akima_option)

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

        assert_near_equal(akima_y.flatten(), prob.get_val('akima1.y_val').flatten(), tolerance=1e-8)

    def test_multi_splines(self):

        import numpy as np

        import openmdao.api as om

        x_cp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        y_cp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        y_cp2 = np.array([1.0, 5.0, 7.0, 8.0, 13.0, 16.0])
        n = 50
        x = np.linspace(1.0, 12.0, n)

        prob = om.Problem()

        comp = om.SplineComp(method='akima', x_cp_val=x_cp, x_interp_val=x)
        prob.model.add_subsystem('akima1', comp)

        comp.add_spline(y_cp_name='ycp1', y_interp_name='y_val1', y_cp_val=y_cp)
        comp.add_spline(y_cp_name='ycp2', y_interp_name='y_val2', y_cp_val=y_cp2)

        prob.setup(force_alloc_complex=True)
        prob.run_model()

    def test_spline_distribution_example(self):

        import numpy as np

        import openmdao.api as om
        from openmdao.utils.spline_distributions import sine_distribution

        x_cp = np.linspace(0., 1., 6)
        y_cp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        n = 20
        x = om.sine_distribution(20, start=0, end=1, phase=np.pi)

        prob = om.Problem()

        comp = om.SplineComp(method='akima', x_cp_val=x_cp, x_interp_val=x)
        prob.model.add_subsystem('akima1', comp)

        comp.add_spline(y_cp_name='ycp', y_interp_name='y_val', y_cp_val=y_cp)

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        akima_y = np.array([[5.         , 5.32381994,  6.28062691 , 7.79410646 , 9.64169506, 11.35166363,
                             12.26525921, 12.99152288, 13.77257256, 14.58710327, 15.41289673, 16.28341046,
                             17.96032258, 20.14140712, 22.31181718, 24.40891577, 26.27368825, 27.74068235,
                             28.67782484, 29.        ]])

        assert_near_equal(akima_y.flatten(), prob.get_val('akima1.y_val').flatten(), tolerance=1e-8)

    def test_akima_options(self):
        import numpy as np

        import openmdao.api as om

        x_cp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        y_cp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])

        n = 50
        x = np.linspace(1.0, 12.0, n)

        prob = om.Problem()
        model = prob.model

        # Set options specific to akima
        akima_option = {'delta_x': 0.1, 'eps': 1e-30}

        comp = om.SplineComp(method='akima', x_cp_val=x_cp, x_interp_val=x,
                             interp_options=akima_option)

        prob.model.add_subsystem('atmosphere', comp)

        comp.add_spline(y_cp_name='alt_cp', y_interp_name='alt', y_cp_val=y_cp, y_units='kft')

        prob.setup(force_alloc_complex=True)
        prob.run_model()

    def test_bspline_options(self):
        import numpy as np

        import openmdao.api as om

        prob = om.Problem()
        model = prob.model

        n_cp = 80
        n_point = 160

        t = np.linspace(0, 3.0*np.pi, n_cp)
        tt = np.linspace(0, 3.0*np.pi, n_point)
        x = np.sin(t)

        # Set options specific to bsplines
        bspline_options = {'order': 3}

        comp = om.SplineComp(method='bsplines', x_interp_val=tt, num_cp=n_cp,
                            interp_options=bspline_options)

        prob.model.add_subsystem('interp', comp, promotes_inputs=[('h_cp', 'x')])

        comp.add_spline(y_cp_name='h_cp', y_interp_name='h', y_cp_val=x, y_units=None)

        prob.setup(force_alloc_complex=True)
        prob.set_val('x', x)
        prob.run_model()

    def test_2to3doc_fixed_grid(self):
        ycp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        ncp = len(ycp)
        n = 11

        prob = om.Problem()

        akima_option = {'delta_x': 0.1}
        comp = om.SplineComp(method='akima', num_cp=ncp, x_interp_val=np.linspace(0.0, 1.0, n),
                             interp_options=akima_option)

        prob.model.add_subsystem('comp1', comp)

        comp.add_spline(y_cp_name='chord_cp', y_interp_name='chord', y_cp_val=ycp)

        prob.setup()
        prob.run_model()

        y = np.array([[ 5.        ,  9.4362525 , 12.        , 13.0012475 , 14.        ,
                        14.99875415, 16.        , 17.93874585, 21.        , 24.625    ,
                        29.        ]])

        assert_near_equal(prob['comp1.chord'], y, 1e-6)

    def test_bsplines_2to3doc(self):
        from openmdao.utils.spline_distributions import sine_distribution

        prob = om.Problem()
        model = prob.model

        n_cp = 5
        n_point = 10

        t = np.linspace(0, 0.5 * np.pi, n_cp)
        x = np.empty((2, n_cp))
        x[0, :] = np.sin(t)
        x[1, :] = 2.0 * np.sin(t)

        # In 2.x, the BsplinesComp had a built-in sinusoidal distribution.
        t_sin = sine_distribution(n_point) * np.pi * 0.5

        bspline_options = {'order': 4}
        comp = om.SplineComp(method='bsplines',
                             x_interp_val=t_sin,
                             num_cp=n_cp,
                             vec_size=2,
                             interp_options=bspline_options)

        prob.model.add_subsystem('interp', comp)

        comp.add_spline(y_cp_name='h_cp', y_interp_name='h', y_cp_val=x, y_units='km')

        prob.setup()
        prob.run_model()

        xx = prob['interp.h']

        with printoptions(precision=3, floatmode='fixed'):
            assert_near_equal(x[0, :], np.array([
                0., 0.38268343, 0.70710678, 0.92387953, 1.
            ]), 1e-5)
            assert_near_equal(x[1, :], 2.0*np.array([
                0., 0.38268343, 0.70710678, 0.92387953, 1.
            ]), 1e-5)

            assert_near_equal(xx[0, :], np.array([
                0., 0.06687281, 0.23486869, 0.43286622, 0.6062628,
                0.74821484, 0.86228902, 0.94134389, 0.98587725, 1.
            ]), 1e-5)
            assert_near_equal(xx[1, :], 2.0*np.array([
                0., 0.06687281, 0.23486869, 0.43286622, 0.6062628,
                0.74821484, 0.86228902, 0.94134389, 0.98587725, 1.
            ]), 1e-5)


if __name__ == '__main__':
    unittest.main()
