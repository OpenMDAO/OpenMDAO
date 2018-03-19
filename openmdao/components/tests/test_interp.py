"""
Test the B-spline interpolation component.
"""
from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp
from openmdao.components.interp import BsplinesComp


class TestBSplinesComp(unittest.TestCase):

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


class TestBSplinesCompFeature(unittest.TestCase):

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

        import matplotlib.pyplot as plt

        xx = prob['interp.h']
        tt = np.linspace(0, 3.0*np.pi, n_point)

        plt.plot(tt, xx)
        plt.plot(t, x, "ro")
        plt.show()


if __name__ == "__main__":
    unittest.main()
