""" Unit tests for the system interface."""

import unittest
from six import assertRaisesRegex

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
        model.suppress_solver_output = True

        # Test pre-setup errors
        with self.assertRaises(Exception) as cm:
            with model.nonlinear_vector_context() as (inputs, outputs, residuals):
                pass
        self.assertEqual(str(cm.exception),
                         "Cannot get vectors because setup has not yet been called.")

        with self.assertRaises(Exception) as cm:
            with model.linear_vector_context('vec') as (d_inputs, d_outputs, d_residuals):
                pass
        self.assertEqual(str(cm.exception),
                         "Cannot get vectors because setup has not yet been called.")

        p.setup()
        p.run_model()

        # Test inputs with original values
        with model.nonlinear_vector_context() as (inputs, outputs, residuals):
            self.assertEqual(inputs['G1.G2.C1.a'], 5.)

        with g1.nonlinear_vector_context() as (inputs, outputs, residuals):
            self.assertEqual(inputs['G2.C1.a'], 5.)

        # Test inputs after setting a new value
        with g2.nonlinear_vector_context() as (inputs, outputs, residuals):
            inputs['C1.a'] = -1.

        with model.nonlinear_vector_context() as (inputs, outputs, residuals):
            self.assertEqual(inputs['G1.G2.C1.a'], -1.)

        with g1.nonlinear_vector_context() as (inputs, outputs, residuals):
            self.assertEqual(inputs['G2.C1.a'], -1.)

        # Test outputs with original values
        with model.nonlinear_vector_context() as (inputs, outputs, residuals):
            self.assertEqual(outputs['G1.G2.C1.b'], 10.)

        with g2.nonlinear_vector_context() as (inputs, outputs, residuals):
            self.assertEqual(outputs['C1.b'], 10.)

        # Test outputs after setting a new value
        with model.nonlinear_vector_context() as (inputs, outputs, residuals):
            outputs['G1.G2.C1.b'] = 123.
            self.assertEqual(outputs['G1.G2.C1.b'], 123.)

        with g2.nonlinear_vector_context() as (inputs, outputs, residuals):
            outputs['C1.b'] = 789.
            self.assertEqual(outputs['C1.b'], 789.)

        # Test residuals
        with model.nonlinear_vector_context() as (inputs, outputs, residuals):
            residuals['G1.G2.C1.b'] = 99.0
            self.assertEqual(residuals['G1.G2.C1.b'], 99.0)

        # Test linear
        with model.linear_vector_context('linear') as (inputs, outputs, residuals):
            outputs['G1.G2.C1.b'] = 10.
            self.assertEqual(outputs['G1.G2.C1.b'], 10.)

        # Test linear with invalid vec_name
        with self.assertRaises(Exception) as cm:
            with model.linear_vector_context('bad_name') as (inputs, outputs, residuals):
                pass
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
        model.add_subsystem('Sink', ExecComp(('c=2*b', 'z=2*y')),
                                             promotes=['b', 'y'])

        p = Problem(model=model)
        p.setup()

        p.model.suppress_solver_output = True
        p.run_model()

        msg = "Incompatible shape for '.*': Expected (.*) but got (.*)"

        num_val = -10
        arr_val = -10*np.ones((5, 1))
        bad_val = -10*np.ones((10))

        with g2.nonlinear_vector_context() as (inputs, outputs, residuals):
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


if __name__ == "__main__":
    unittest.main()
