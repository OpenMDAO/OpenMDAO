"""Test the Brent nonlinear solver. """

import unittest

import numpy as np
from scipy.optimize import brentq

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class CompTest(om.ImplicitComponent):

    def setup(self):
        self.add_input('a', val=1.)
        self.add_input('b', val=1.)
        self.add_input('c', val=10.)
        self.add_input('n', val=77.0/27.0)

        self.add_output('x', val=2., lower=0, upper=100)

    def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        n = inputs['n']
        x = outputs['x']

        # Can't take fractional power of negative number
        if x >= 0.0:
            fact = x ** n
        else:
            fact = - (-x) ** n

        residuals['x'] = a * fact + b * x - c


class BracketTestComponent(om.ImplicitComponent):

    def setup(self):
        self.add_input('a', val=.3)
        self.add_input('ap', val=.01)
        self.add_input('lambda_r', val=7.)

        self.add_output('phi', val=0.0)

    def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None):
        a = inputs['a']
        ap = inputs['ap']
        lambda_r = inputs['lambda_r']
        phi = outputs['phi']

        residuals['phi'] = np.sin(phi) / (1 - a) - np.cos(phi) / lambda_r / (1 + ap)

class TestBrentSolver(unittest.TestCase):

    def test_basic_brent_converge(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', CompTest(), promotes=['*'])
        model.nonlinear_solver = om.BrentSolver(
            state_target='x',
            maxiter=100,
            atol=1e-8,
            rtol=1e-8,
        )

        prob.setup()
        prob.set_solver_print(0)

        prob.run_model()

        assert_near_equal(prob.get_val('x')[0], 2.06720359226, 1e-8)

    def test_no_state_var_err(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', CompTest(), promotes=['*'])
        model.nonlinear_solver = om.BrentSolver(
            maxiter=100,
            atol=1e-8,
            rtol=1e-8,
        )
        prob.setup(check=False)

        with self.assertRaises(ValueError) as context:
            prob.final_setup()

        msg = "BrentSolver in <model> <class Group>: 'state_target' option in Brent solver must be specified."
        self.assertEqual(str(context.exception), msg)

    def test_state_var_not_found(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', CompTest(), promotes=['*'])
        model.nonlinear_solver = om.BrentSolver(
            state_target='fake',
            maxiter=100,
            atol=1e-8,
            rtol=1e-8,
        )
        prob.setup(check=False)

        with self.assertRaises(ValueError) as context:
            prob.final_setup()

        msg = "BrentSolver in <model> <class Group>: 'state_target' variable 'fake' not found."
        self.assertEqual(str(context.exception), msg)

    def test_cycle_error(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem(
            'comp1',
            om.ExecComp('x = 0.1 * a'),
            promotes=['*']
        )
        model.add_subsystem(
            'comp2',
            om.ExecComp('a = 0.1 * x'),
            promotes=['*']
        )
        model.nonlinear_solver = om.BrentSolver(
            state_target='x',
        )

        prob.setup()
        prob.set_solver_print(0)

        with self.assertRaises(ValueError) as context:
            prob.run_model()
        msg = "BrentSolver in <model> <class Group>: Brent does not support cycles."
        self.assertEqual(str(context.exception), msg)

    def test_multiple_state_error(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp1', CompTest(), promotes=['*'])
        model.add_subsystem('comp2', CompTest(), promotes_inputs=['*'])
        model.nonlinear_solver = om.BrentSolver(
            state_target='x',
        )

        prob.setup()
        prob.set_solver_print(0)

        with self.assertRaises(ValueError) as context:
            prob.run_model()
        msg = "BrentSolver in <model> <class Group>: Brent can only solve 1 single implicit state."
        self.assertEqual(str(context.exception), msg)

    def test_bad_bounds(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', CompTest(), promotes=['*'])
        model.nonlinear_solver = om.BrentSolver(
            state_target='x',
            lower_bound=30,
            upper_bound=40,
        )

        prob.setup()
        prob.set_solver_print(0)

        # Note: don't check the message because it comes from scipy. Just make sure we are covered.
        with self.assertRaises(ValueError):
            prob.run_model()

    def test_data_pass_bounds(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('lower', om.ExecComp('low = 2*a'), promotes=['*'])
        model.add_subsystem('upper', om.ExecComp('high = 2*b'), promotes=['*'])

        model.add_subsystem('comp', CompTest(), promotes=['*'])
        model.nonlinear_solver = om.BrentSolver(
            state_target='x',
            maxiter=100,
            atol=1e-8,
            rtol=1e-8,
            lower_bound_target='flow',  # bad value for testing error
            lower_bound=-1e-10,  # Assuring that these are ignored.
            upper_bound=1e-10,  # Assuring that these are ignored.
        )

        prob.setup()
        prob.set_solver_print(0)

        with self.assertRaises(ValueError) as context:
            prob.final_setup()

        msg = "BrentSolver in <model> <class Group>: 'lower_bound_target' variable 'flow' not found."
        self.assertEqual(str(context.exception), msg)

        # Now set the correct locations for upper/lower bounds.

        model.nonlinear_solver.options['lower_bound_target'] = 'low'
        model.nonlinear_solver.options['upper_bound_target'] = 'high'

        prob.setup()

        prob.set_val('a', -5.0)
        prob.set_val('b', 55.0)

        prob.run_model()

        assert_near_equal(prob.get_val('x')[0], -3.7451537261581453, 1e-8)

    def test_err_on_non_converge(self):
        # Raise AnalysisError when it fails to converge

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', CompTest(), promotes=['*'])
        model.nonlinear_solver = om.BrentSolver(
            state_target='x',
            maxiter=1,
            err_on_non_converge=True,
        )

        prob.setup()
        prob.set_solver_print(level=0)

        with self.assertRaises(om.AnalysisError) as context:
            prob.run_driver()

        # Note, extra iterations are brackets.
        msg = "Solver 'NL: BRENT' on system '' failed to converge in 3 iterations."
        self.assertEqual(str(context.exception), msg)

    def test_bracket(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', BracketTestComponent(), promotes=['*'])

        eps = 1e-6
        model.nonlinear_solver = om.BrentSolver(
            state_target='phi',
            maxiter=100,
            atol=1e-8,
            rtol=1e-8,
            lower_bound=eps,
            upper_bound=np.pi/2 - eps,
        )

        prob.setup()
        prob.set_solver_print(level=2)
        prob.run_model()

        # manually compute the right answer
        def manual_f(phi, p):
            r = np.sin(phi)/(1-p['a']) - np.cos(phi)/p['lambda_r']/(1+p['ap'])
            return r

        inputs = {
            'a': prob.get_val('a'),
            'ap': prob.get_val('ap'),
            'lambda_r': prob.get_val('lambda_r'),
        }
        phi_star = brentq(manual_f, eps, np.pi/2-eps, args=inputs)

        assert_near_equal(prob.get_val('phi')[0], phi_star, 1e-10)


if __name__ == '__main__':
    unittest.main()