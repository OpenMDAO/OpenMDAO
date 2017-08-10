""" Unit tests for the system interface."""

import unittest
from six import assertRaisesRegex

import warnings

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp
from openmdao.devtools.testutil import assert_rel_error


class TestSystem(unittest.TestCase):

    def test_vector_context_managers(self):
        g1 = Group()
        g1.add_subsystem('Indep', IndepVarComp('a', 5.0), promotes=['a'])
        g2 = g1.add_subsystem('G2', Group(), promotes=['*'])
        g2.add_subsystem('C1', ExecComp('b=2*a'), promotes=['a', 'b'])

        model = Group()
        model.add_subsystem('G1', g1, promotes=['b'])
        model.add_subsystem('Sink', ExecComp('c=2*b'), promotes=['b'])

        p = Problem(model=model)
        p.set_solver_print(level=0)

        # Test pre-setup errors
        with self.assertRaises(Exception) as cm:
            inputs, outputs, residuals = model.get_nonlinear_vectors()
        self.assertEqual(str(cm.exception),
                         "Cannot get vectors because setup has not yet been called.")

        with self.assertRaises(Exception) as cm:
            d_inputs, d_outputs, d_residuals = model.get_linear_vectors('vec')
        self.assertEqual(str(cm.exception),
                         "Cannot get vectors because setup has not yet been called.")

        p.setup()
        p.run_model()

        # Test inputs with original values
        inputs, outputs, residuals = model.get_nonlinear_vectors()
        self.assertEqual(inputs['G1.G2.C1.a'], 5.)

        inputs, outputs, residuals = g1.get_nonlinear_vectors()
        self.assertEqual(inputs['G2.C1.a'], 5.)

        # Test inputs after setting a new value
        inputs, outputs, residuals = g2.get_nonlinear_vectors()
        inputs['C1.a'] = -1.

        inputs, outputs, residuals = model.get_nonlinear_vectors()
        self.assertEqual(inputs['G1.G2.C1.a'], -1.)

        inputs, outputs, residuals = g1.get_nonlinear_vectors()
        self.assertEqual(inputs['G2.C1.a'], -1.)

        # Test outputs with original values
        inputs, outputs, residuals = model.get_nonlinear_vectors()
        self.assertEqual(outputs['G1.G2.C1.b'], 10.)

        inputs, outputs, residuals = g2.get_nonlinear_vectors()

        # Test outputs after setting a new value
        inputs, outputs, residuals = model.get_nonlinear_vectors()
        outputs['G1.G2.C1.b'] = 123.
        self.assertEqual(outputs['G1.G2.C1.b'], 123.)

        inputs, outputs, residuals = g2.get_nonlinear_vectors()
        outputs['C1.b'] = 789.
        self.assertEqual(outputs['C1.b'], 789.)

        # Test residuals
        inputs, outputs, residuals = model.get_nonlinear_vectors()
        residuals['G1.G2.C1.b'] = 99.0
        self.assertEqual(residuals['G1.G2.C1.b'], 99.0)

        # Test linear
        d_inputs, d_outputs, d_residuals = model.get_linear_vectors('linear')
        d_outputs['G1.G2.C1.b'] = 10.
        self.assertEqual(d_outputs['G1.G2.C1.b'], 10.)

        # Test linear with invalid vec_name
        with self.assertRaises(Exception) as cm:
            d_inputs, d_outputs, d_residuals = model.get_linear_vectors('bad_name')
        self.assertEqual(str(cm.exception),
                         "There is no linear vector named %s" % 'bad_name')

    def test_set_checks_shape(self):
        indep = IndepVarComp()
        indep.add_output('a')
        indep.add_output('x', shape=(5, 1))

        g1 = Group()
        g1.add_subsystem('Indep', indep, promotes=['a', 'x'])

        g2 = g1.add_subsystem('G2', Group(), promotes=['*'])
        g2.add_subsystem('C1', ExecComp('b=2*a'), promotes=['a', 'b'])
        g2.add_subsystem('C2', ExecComp('y=2*x',
                                        x=np.zeros((5, 1)),
                                        y=np.zeros((5, 1))),
                                        promotes=['x', 'y'])

        model = Group()
        model.add_subsystem('G1', g1, promotes=['b', 'y'])
        model.add_subsystem('Sink', ExecComp(('c=2*b', 'z=2*y'),
                                             y=np.zeros((5, 1)),
                                             z=np.zeros((5, 1))),
                                             promotes=['b', 'y'])

        p = Problem(model=model)
        p.setup()

        p.set_solver_print(level=0)
        p.run_model()

        msg = "Incompatible shape for '.*': Expected (.*) but got (.*)"

        num_val = -10
        arr_val = -10*np.ones((5, 1))
        bad_val = -10*np.ones((10))

        inputs, outputs, residuals = g2.get_nonlinear_vectors()
        #
        # set input
        #

        # assign array to scalar
        with assertRaisesRegex(self, ValueError, msg):
            inputs['C1.a'] = arr_val

        # assign scalar to array
        inputs['C2.x'] = num_val
        assert_rel_error(self, inputs['C2.x'], arr_val, 1e-10)

        # assign array to array
        inputs['C2.x'] = arr_val
        assert_rel_error(self, inputs['C2.x'], arr_val, 1e-10)

        # assign bad array shape to array
        with assertRaisesRegex(self, ValueError, msg):
            inputs['C2.x'] = bad_val

        # assign list to array
        inputs['C2.x'] = arr_val.tolist()
        assert_rel_error(self, inputs['C2.x'], arr_val, 1e-10)

        # assign bad list shape to array
        with assertRaisesRegex(self, ValueError, msg):
            inputs['C2.x'] = bad_val.tolist()

        #
        # set output
        #

        # assign array to scalar
        with assertRaisesRegex(self, ValueError, msg):
            outputs['C1.b'] = arr_val

        # assign scalar to array
        outputs['C2.y'] = num_val
        assert_rel_error(self, outputs['C2.y'], arr_val, 1e-10)

        # assign array to array
        outputs['C2.y'] = arr_val
        assert_rel_error(self, outputs['C2.y'], arr_val, 1e-10)

        # assign bad array shape to array
        with assertRaisesRegex(self, ValueError, msg):
            outputs['C2.y'] = bad_val

        # assign list to array
        outputs['C2.y'] = arr_val.tolist()
        assert_rel_error(self, outputs['C2.y'], arr_val, 1e-10)

        # assign bad list shape to array
        with assertRaisesRegex(self, ValueError, msg):
            outputs['C2.y'] = bad_val.tolist()

        #
        # set residual
        #

        # assign array to scalar
        with assertRaisesRegex(self, ValueError, msg):
            residuals['C1.b'] = arr_val

        # assign scalar to array
        residuals['C2.y'] = num_val
        assert_rel_error(self, residuals['C2.y'], arr_val, 1e-10)

        # assign array to array
        residuals['C2.y'] = arr_val
        assert_rel_error(self, residuals['C2.y'], arr_val, 1e-10)

        # assign bad array shape to array
        with assertRaisesRegex(self, ValueError, msg):
            residuals['C2.y'] = bad_val

        # assign list to array
        residuals['C2.y'] = arr_val.tolist()
        assert_rel_error(self, residuals['C2.y'], arr_val, 1e-10)

        # assign bad list shape to array
        with assertRaisesRegex(self, ValueError, msg):
            residuals['C2.y'] = bad_val.tolist()

    def test_deprecated_solver_names(self):
        class DummySolver():
            pass

        model = Group()

        # check nl_solver setter
        with warnings.catch_warnings(record=True) as w:
            model.nl_solver = DummySolver()

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message),
                         "The 'nl_solver' attribute provides backwards compatibility "
                         "with OpenMDAO 1.x ; use 'nonlinear_solver' instead.")

        # check nl_solver getter
        with warnings.catch_warnings(record=True) as w:
            solver = model.nl_solver

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message),
                         "The 'nl_solver' attribute provides backwards compatibility "
                         "with OpenMDAO 1.x ; use 'nonlinear_solver' instead.")

        self.assertTrue(isinstance(solver, DummySolver))

        # check ln_solver setter
        with warnings.catch_warnings(record=True) as w:
            model.ln_solver = DummySolver()

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message),
                         "The 'ln_solver' attribute provides backwards compatibility "
                         "with OpenMDAO 1.x ; use 'linear_solver' instead.")

        # check ln_solver getter
        with warnings.catch_warnings(record=True) as w:
            solver = model.ln_solver

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message),
                         "The 'ln_solver' attribute provides backwards compatibility "
                         "with OpenMDAO 1.x ; use 'linear_solver' instead.")

        self.assertTrue(isinstance(solver, DummySolver))


if __name__ == "__main__":
    unittest.main()
