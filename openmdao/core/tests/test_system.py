import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp

class TestSystem(unittest.TestCase):

    def test_vector_context_managers(self):
        g1 = Group()
        indep = g1.add_subsystem('Indep', IndepVarComp('a', 5.0), promotes=['a'])
        g2 = g1.add_subsystem('G2', Group(), promotes=['*'])
        c1_2 = g2.add_subsystem('C1', ExecComp('b=2*a'), promotes=['a', 'b'])

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


if __name__ == "__main__":
    unittest.main()
