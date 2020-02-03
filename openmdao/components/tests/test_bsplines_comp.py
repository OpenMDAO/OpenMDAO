"""
Test the B-spline interpolation component.
"""
from __future__ import print_function

import unittest

import numpy as np

try:
    import matplotlib
except ImportError:
    matplotlib = None

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error, assert_warning


class TestBsplinesComp(unittest.TestCase):

    def test_basic(self):
        prob = om.Problem()
        model = prob.model

        n_cp = 80
        n_point = 160

        t = np.linspace(0, 3.0*np.pi, n_cp)
        x = np.sin(t)

        model.add_subsystem('px', om.IndepVarComp('x', val=x))
        model.add_subsystem('interp', om.BsplinesComp(num_control_points=n_cp,
                                                      num_points=n_point,
                                                      in_name='h_cp',
                                                      out_name='h',
                                                      distribution='uniform'))

        model.connect('px.x', 'interp.h_cp')

        prob.setup()
        prob.run_model()

        xx = prob['interp.h'].flatten()
        tt = np.linspace(0, 3.0*np.pi, n_point)

        x_expected = np.sin(tt)
        delta = xx - x_expected

        # Here we test that we don't have crazy interpolation error.
        self.assertLess(max(delta), .15)
        # And that it gets middle points a little better.
        self.assertLess(max(delta[15:-15]), .06)

    def test_units(self):
        n_cp = 5
        n_point = 10

        interp = om.BsplinesComp(num_control_points=n_cp,
                                 num_points=n_point,
                                 in_name='h_cp',
                                 out_name='h',
                                 units='inch')

        prob = om.Problem(model=interp)
        prob.setup()
        prob.run_model()

        # verify that both input and output of the bsplines comp have proper units
        inputs = interp.list_inputs(units=True, out_stream=None)
        self.assertEqual(len(inputs), 1)
        for var, meta in inputs:
            self.assertEqual(meta['units'], 'inch')

        outputs = interp.list_outputs(units=True, out_stream=None)
        self.assertEqual(len(outputs), 1)
        for var, meta in outputs:
            self.assertEqual(meta['units'], 'inch')


@unittest.skipUnless(matplotlib, "Matplotlib is required.")
class TestBsplinesCompFeature(unittest.TestCase):

    def test_basic(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.utils.general_utils import printoptions

        prob = om.Problem()
        model = prob.model

        n_cp = 5
        n_point = 10

        t = np.linspace(0, 0.5*np.pi, n_cp)
        x = np.sin(t)

        model.add_subsystem('px', om.IndepVarComp('x', val=x))
        model.add_subsystem('interp', om.BsplinesComp(num_control_points=n_cp,
                                                      num_points=n_point,
                                                      in_name='h_cp',
                                                      out_name='h'))
        model.connect('px.x', 'interp.h_cp')

        prob.setup()
        prob.run_model()

        xx = prob['interp.h'].flatten()

        with printoptions(precision=3, floatmode='fixed'):
            self.assertEqual('Control Points:', 'Control Points:')
            assert_rel_error(self, x, np.array([
                0., 0.38268343, 0.70710678, 0.92387953, 1.
            ]), 1e-5)

            self.assertEqual('Output Points:', 'Output Points:')
            assert_rel_error(self, xx, np.array([
                0., 0.06687281, 0.23486869, 0.43286622, 0.6062628,
                0.74821484, 0.86228902, 0.94134389, 0.98587725, 1.
            ]), 1e-5)

    def test_vectorized(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.utils.general_utils import printoptions

        prob = om.Problem()
        model = prob.model

        n_cp = 5
        n_point = 10

        t = np.linspace(0, 0.5*np.pi, n_cp)
        x = np.empty((2, n_cp))
        x[0, :] = np.sin(t)
        x[1, :] = 2.0*np.sin(t)

        model.add_subsystem('px', om.IndepVarComp('x', val=x))
        model.add_subsystem('interp', om.BsplinesComp(num_control_points=n_cp,
                                                      num_points=n_point,
                                                      vec_size=2,
                                                      in_name='h_cp',
                                                      out_name='h'))
        model.connect('px.x', 'interp.h_cp')

        prob.setup()
        prob.run_model()

        xx = prob['interp.h']

        with printoptions(precision=3, floatmode='fixed'):
            self.assertEqual('Control Points:', 'Control Points:')
            assert_rel_error(self, x[0, :], np.array([
                0., 0.38268343, 0.70710678, 0.92387953, 1.
            ]), 1e-5)
            assert_rel_error(self, x[1, :], 2.0*np.array([
                0., 0.38268343, 0.70710678, 0.92387953, 1.
            ]), 1e-5)

            self.assertEqual('Output Points:', 'Output Points:')
            assert_rel_error(self, xx[0, :], np.array([
                0., 0.06687281, 0.23486869, 0.43286622, 0.6062628,
                0.74821484, 0.86228902, 0.94134389, 0.98587725, 1.
            ]), 1e-5)
            assert_rel_error(self, xx[1, :], 2.0*np.array([
                0., 0.06687281, 0.23486869, 0.43286622, 0.6062628,
                0.74821484, 0.86228902, 0.94134389, 0.98587725, 1.
            ]), 1e-5)

    def test_bspline_comp_deprecations(self):
        prob = om.Problem()
        model = prob.model

        n_cp = 80
        n_point = 160

        t = np.linspace(0, 3.0*np.pi, n_cp)
        x = np.sin(t)

        model.add_subsystem('px', om.IndepVarComp('x', val=x))

        msg = "'BsplinesComp' has been deprecated. Use 'SplineComp' instead."
        with assert_warning(DeprecationWarning, msg):
            om.BsplinesComp(num_control_points=n_cp, num_points=n_point, in_name='h_cp',
                            out_name='h',
                            distribution='uniform')


@unittest.skipUnless(matplotlib, "Matplotlib is required.")
class TestBsplinesCompFeatureWithPlotting(unittest.TestCase):

    def setUp(self):
        matplotlib.use('Agg')

    def test_distribution_uniform(self):
        import openmdao.api as om

        prob = om.Problem()
        model = prob.model

        n_cp = 20
        n_point = 100

        t = np.linspace(0, 3.0*np.pi, n_cp)
        x = np.sin(t)

        model.add_subsystem('px', om.IndepVarComp('x', val=x))
        model.add_subsystem('interp', om.BsplinesComp(num_control_points=n_cp,
                                                      num_points=n_point,
                                                      in_name='h_cp',
                                                      out_name='h',
                                                      distribution='uniform'))
        model.connect('px.x', 'interp.h_cp')

        prob.setup()
        prob.run_model()

        xx = prob['interp.h'].flatten()
        tt = np.linspace(0, 3.0*np.pi, n_point)

        import matplotlib.pyplot as plt

        plt.plot(tt, xx)
        plt.plot(t, x, "ro")
        plt.xlabel("Distance along Beam")
        plt.ylabel('Design Variable')
        plt.title("Uniform Distribution of Control Points")
        plt.legend(['Variable', 'Control Points'], loc=4)
        plt.grid(True)
        plt.show()

    def test_distribution_sine(self):
        import openmdao.api as om

        prob = om.Problem()
        model = prob.model

        n_cp = 20
        n_point = 100

        tvec = np.linspace(0, 1.0, n_cp)
        t = 3.0 * np.pi * 0.5 * (1.0 + np.sin(-0.5 * np.pi + tvec * np.pi))
        x = np.sin(t)

        model.add_subsystem('px', om.IndepVarComp('x', val=x))
        model.add_subsystem('interp', om.BsplinesComp(num_control_points=n_cp,
                                                      num_points=n_point,
                                                      in_name='h_cp',
                                                      out_name='h',
                                                      distribution='sine'))
        model.connect('px.x', 'interp.h_cp')

        prob.setup()
        prob.run_model()

        xx = prob['interp.h'].flatten()
        ttvec = np.linspace(0, 1.0, n_point)
        tt = 3.0 * np.pi * 0.5 * (1.0 + np.sin(-0.5 * np.pi + ttvec * np.pi))

        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.plot(tt, xx, "b")
        plt.plot(t, x, "ro")
        plt.xlabel("Distance along Beam")
        plt.ylabel('Design Variable')
        plt.title("Sine Distribution of Control Points")
        plt.legend(['Variable', 'Control Points'], loc=4)
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    unittest.main()
