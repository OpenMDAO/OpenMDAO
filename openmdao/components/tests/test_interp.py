"""
Test the B-spline interpolation component.
"""
from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp
from openmdao.components.interp import BsplinesComp
from openmdao.utils.assert_utils import assert_rel_error


class TestBsplinesComp(unittest.TestCase):

    def test_basic(self):
        prob = Problem()
        model = prob.model

        n_cp = 80
        n_point = 160

        t = np.linspace(0, 3.0*np.pi, n_cp)
        x = np.sin(t)

        model.add_subsystem('px', IndepVarComp('x', val=x))
        model.add_subsystem('interp', BsplinesComp(num_control_points=n_cp,
                                                   num_points=n_point,
                                                   in_name='h_cp',
                                                   out_name='h',
                                                   distribution='uniform'))

        model.connect('px.x', 'interp.h_cp')


        prob.setup(check=False)
        prob.run_model()

        xx = prob['interp.h']
        tt = np.linspace(0, 3.0*np.pi, n_point)

        x_expected = np.sin(tt)
        delta = xx - x_expected

        # Here we test that we don't have crazy interpolation error.
        self.assertLess(max(delta), .15)
        # And that it gets middle points a little better.
        self.assertLess(max(delta[15:-15]), .06)


class TestBsplinesCompFeature(unittest.TestCase):

    def setUp(self):
        import matplotlib
        matplotlib.use('Agg')

    def test_basic(self):
        prob = Problem()
        model = prob.model

        n_cp = 5
        n_point = 10

        t = np.linspace(0, 0.5*np.pi, n_cp)
        x = np.sin(t)

        model.add_subsystem('px', IndepVarComp('x', val=x))
        model.add_subsystem('interp', BsplinesComp(num_control_points=n_cp,
                                                   num_points=n_point,
                                                   in_name='h_cp',
                                                   out_name='h'))

        model.connect('px.x', 'interp.h_cp')


        prob.setup(check=False)
        prob.run_model()

        xx = prob['interp.h']

        print('Control Points')
        assert_rel_error(self, x, np.array([0.        , 0.38268343, 0.70710678, 0.92387953, 1.        ]), 1e-5)
        print('Output Points')
        assert_rel_error(self, xx, np.array([0.        , 0.06687281, 0.23486869, 0.43286622, 0.6062628 ,
                                             0.74821484, 0.86228902, 0.94134389, 0.98587725, 1.        ]), 1e-5)

    def test_distribution_uniform(self):
        prob = Problem()
        model = prob.model

        n_cp = 20
        n_point = 100

        t = np.linspace(0, 3.0*np.pi, n_cp)
        x = np.sin(t)

        model.add_subsystem('px', IndepVarComp('x', val=x))
        model.add_subsystem('interp', BsplinesComp(num_control_points=n_cp,
                                                   num_points=n_point,
                                                   in_name='h_cp',
                                                   out_name='h',
                                                   distribution='uniform'))

        model.connect('px.x', 'interp.h_cp')


        prob.setup(check=False)
        prob.run_model()

        xx = prob['interp.h']
        tt = np.linspace(0, 3.0*np.pi, n_point)

        x_expected = np.sin(tt)
        delta = xx - x_expected

        import matplotlib.pyplot as plt

        plt.plot(tt, xx)
        plt.plot(t, x, "ro")
        plt.xlabel("Distance along Beam")
        plt.ylabel('Design Variable')
        plt.title("Uniform Distribution of Control Points")
        plt.legend(['Variable', 'Control Points'], loc=4)
        plt.grid(True)
        plt.show()

        assert_rel_error(self, xx[10], 0.93528587, 1e-4 )

    def test_distribution_sine(self):
        prob = Problem()
        model = prob.model

        n_cp = 20
        n_point = 100

        tvec = np.linspace(0, 1.0, n_cp)
        t = 3.0 * np.pi * 0.5 * (1.0 + np.sin(-0.5 * np.pi + tvec * np.pi))
        x = np.sin(t)

        model.add_subsystem('px', IndepVarComp('x', val=x))
        model.add_subsystem('interp', BsplinesComp(num_control_points=n_cp,
                                                   num_points=n_point,
                                                   in_name='h_cp',
                                                   out_name='h',
                                                   distribution='sine'))

        model.connect('px.x', 'interp.h_cp')


        prob.setup(check=False)
        prob.run_model()

        xx = prob['interp.h']
        ttvec = np.linspace(0, 1.0, n_point)
        tt = 3.0 * np.pi * 0.5 * (1.0 + np.sin(-0.5 * np.pi + ttvec * np.pi))

        x_expected = np.sin(tt)
        delta = xx - x_expected

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

        assert_rel_error(self, xx[10], 0.09568950, 1e-4 )

if __name__ == "__main__":
    unittest.main()
