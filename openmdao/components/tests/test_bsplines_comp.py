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

from openmdao.api import Problem, IndepVarComp
from openmdao.components.bsplines_comp import BsplinesComp
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

        xx = prob['interp.h'].flatten()
        tt = np.linspace(0, 3.0*np.pi, n_point)

        x_expected = np.sin(tt)
        delta = xx - x_expected

        # Here we test that we don't have crazy interpolation error.
        self.assertLess(max(delta), .15)
        # And that it gets middle points a little better.
        self.assertLess(max(delta[15:-15]), .06)


@unittest.skipUnless(matplotlib, "Matplotlib is required.")
class TestBsplinesCompFeature(unittest.TestCase):

    def setUp(self):
        matplotlib.use('Agg')

    def test_basic(self):
        from openmdao.api import Problem, IndepVarComp
        from openmdao.components.bsplines_comp import BsplinesComp
        from openmdao.utils.general_utils import printoptions

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
        from openmdao.api import Problem, IndepVarComp
        from openmdao.components.bsplines_comp import BsplinesComp
        from openmdao.utils.general_utils import printoptions

        prob = Problem()
        model = prob.model

        n_cp = 5
        n_point = 10

        t = np.linspace(0, 0.5*np.pi, n_cp)
        x = np.empty((2, n_cp))
        x[0, :] = np.sin(t)
        x[1, :] = 2.0*np.sin(t)

        model.add_subsystem('px', IndepVarComp('x', val=x))
        model.add_subsystem('interp', BsplinesComp(num_control_points=n_cp,
                                                   num_points=n_point,
                                                   vec_size=2,
                                                   in_name='h_cp',
                                                   out_name='h'))
        model.connect('px.x', 'interp.h_cp')

        prob.setup(check=False)
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

    def test_distribution_uniform(self):
        from openmdao.api import Problem, IndepVarComp
        from openmdao.components.bsplines_comp import BsplinesComp

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
        from openmdao.api import Problem, IndepVarComp
        from openmdao.components.bsplines_comp import BsplinesComp

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

    def test_units(self):
        from openmdao.api import Problem, IndepVarComp
        from openmdao.components.bsplines_comp import BsplinesComp
        from openmdao.utils.units import convert_units
        from openmdao.utils.general_utils import printoptions

        prob = Problem()
        model = prob.model

        n_cp = 5
        n_point = 10

        t = np.linspace(0, 0.5*np.pi, n_cp)
        x = 100*np.sin(t)

        model.add_subsystem('px', IndepVarComp('x', val=x, units='cm'))
        model.add_subsystem('interp', BsplinesComp(num_control_points=n_cp,
                                                   num_points=n_point,
                                                   in_name='h_cp',
                                                   in_units='inch',
                                                   out_name='h',
                                                   out_units='ft'))
        model.connect('px.x', 'interp.h_cp')

        prob.setup(check=False)
        prob.run_model()

        xx = prob['interp.h'].flatten()

        with printoptions(precision=3, floatmode='fixed'):
            print('x: (cm):', x)
            print('-> inch:', convert_units(x, 'cm', 'inch'))

            print('h_cp:   ', prob['interp.h_cp'].flatten())
            print('-> ft:  ', convert_units(prob['interp.h_cp'].flatten(), 'inch', 'ft'))

            print('h:      ', prob['interp.h'].flatten())

        with printoptions(precision=3, floatmode='fixed'):
            self.assertEqual('Control Points (cm):', 'Control Points (cm):')
            assert_rel_error(self, x, 100*np.sin(t), 1e-5)

            self.assertEqual('Control Points (inches):', 'Control Points (inches):')
            assert_rel_error(self, prob['interp.h_cp'].flatten(),
                             convert_units(x, 'cm', 'inch'), 1e-5)

            self.assertEqual('Output Points (ft):', 'Output Points (ft):')
            assert_rel_error(self, xx[-1], convert_units(x[-1], 'cm', 'ft'), 1e-5)

    def test_units_simple(self):
        from openmdao.api import Problem, IndepVarComp, ExplicitComponent
        from openmdao.utils.units import convert_units

        class UnitsComp(ExplicitComponent):
            def setup(self):
                self.add_input('inp', 0., units='inch')
                self.add_output('outp', 0., units='ft')

                self.declare_partials('*', '*')

            def compute(self, inputs, outputs):
                outputs['outp'] = inputs['inp']

        prob = Problem()

        model = prob.model
        model.add_subsystem('indep', IndepVarComp('x', val=10., units='cm'))
        model.add_subsystem('units', UnitsComp())
        model.connect('indep.x', 'units.inp')

        prob.setup()
        prob.run_model()

        print('indep.x   ', prob['indep.x'],    'cm')
        print('units.inp ', prob['units.inp'],  'inch  ( vs', convert_units(10., 'cm', 'inch'), ')')
        print('units.outp', prob['units.outp'], 'ft    ( vs', convert_units(10., 'cm', 'ft'), ')')


if __name__ == "__main__":
    unittest.main()
