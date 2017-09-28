"""Simple example demonstrating how to implement an implicit component."""
from __future__ import division

import unittest

from six.moves import cStringIO
import numpy as np

from openmdao.api import Problem, Group, ImplicitComponent, IndepVarComp, NewtonSolver, \
                         ScipyIterativeSolver
from openmdao.devtools.testutil import assert_rel_error


# Note: The following class definitions are used in feature docs

class QuadraticComp(ImplicitComponent):
    """
    A Simple Implicit Component representing a Quadratic Equation.

    R(a, b, c, x) = ax^2 + bx + c

    Solution via Quadratic Formula:
    x = (-b + sqrt(b^2 - 4ac)) / 2a
    """

    def setup(self):
        self.add_input('a', val=1.)
        self.add_input('b', val=1.)
        self.add_input('c', val=1.)
        self.add_output('x', val=0.)

        self.declare_partials(of='*', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']
        residuals['x'] = a * x ** 2 + b * x + c

    def solve_nonlinear(self, inputs, outputs):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        outputs['x'] = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)


class QuadraticLinearize(QuadraticComp):

    def linearize(self, inputs, outputs, partials):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']

        partials['x', 'a'] = x ** 2
        partials['x', 'b'] = x
        partials['x', 'c'] = 1.0
        partials['x', 'x'] = 2 * a * x + b

        self.inv_jac = 1.0 / (2 * a * x + b)

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['x'] = self.inv_jac * d_residuals['x']
        elif mode == 'rev':
            d_residuals['x'] = self.inv_jac * d_outputs['x']


class QuadraticJacVec(QuadraticComp):

    def linearize(self, inputs, outputs, partials):
        a = inputs['a']
        b = inputs['b']
        x = outputs['x']
        self.inv_jac = 1.0 / (2 * a * x + b)

    def apply_linear(self, inputs, outputs,
                     d_inputs, d_outputs, d_residuals, mode):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']
        if mode == 'fwd':
            if 'x' in d_residuals:
                if 'x' in d_outputs:
                    d_residuals['x'] += (2 * a * x + b) * d_outputs['x']
                if 'a' in d_inputs:
                    d_residuals['x'] += x ** 2 * d_inputs['a']
                if 'b' in d_inputs:
                    d_residuals['x'] += x * d_inputs['b']
                if 'c' in d_inputs:
                    d_residuals['x'] += d_inputs['c']
        elif mode == 'rev':
            if 'x' in d_residuals:
                if 'x' in d_outputs:
                    d_outputs['x'] += (2 * a * x + b) * d_residuals['x']
                if 'a' in d_inputs:
                    d_inputs['a'] += x ** 2 * d_residuals['x']
                if 'b' in d_inputs:
                    d_inputs['b'] += x * d_residuals['x']
                if 'c' in d_inputs:
                    d_inputs['c'] += d_residuals['x']

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['x'] = self.inv_jac * d_residuals['x']
        elif mode == 'rev':
            d_residuals['x'] = self.inv_jac * d_outputs['x']


class ImplicitCompTestCase(unittest.TestCase):

    def setUp(self):
        group = Group()

        comp1 = group.add_subsystem('comp1', IndepVarComp())
        comp1.add_output('a', 1.0)
        comp1.add_output('b', -4.0)
        comp1.add_output('c', 3.0)

        group.add_subsystem('comp2', QuadraticLinearize())
        group.add_subsystem('comp3', QuadraticJacVec())

        group.connect('comp1.a', 'comp2.a')
        group.connect('comp1.b', 'comp2.b')
        group.connect('comp1.c', 'comp2.c')
        group.connect('comp1.a', 'comp3.a')
        group.connect('comp1.b', 'comp3.b')
        group.connect('comp1.c', 'comp3.c')

        prob = Problem(model=group)
        prob.setup(check=False)

        self.prob = prob

    def test_compute_and_derivs(self):
        prob = self.prob
        prob.run_model()

        assert_rel_error(self, prob['comp2.x'], 3.)
        assert_rel_error(self, prob['comp2.x'], 3.)

        total_derivs = prob.compute_totals(
            wrt=['comp1.a', 'comp1.b', 'comp1.c'],
            of=['comp2.x', 'comp3.x']
        )
        assert_rel_error(self, total_derivs['comp2.x', 'comp1.a'], [[-4.5]])
        assert_rel_error(self, total_derivs['comp2.x', 'comp1.b'], [[-1.5]])
        assert_rel_error(self, total_derivs['comp2.x', 'comp1.c'], [[-0.5]])
        assert_rel_error(self, total_derivs['comp3.x', 'comp1.a'], [[-4.5]])
        assert_rel_error(self, total_derivs['comp3.x', 'comp1.b'], [[-1.5]])
        assert_rel_error(self, total_derivs['comp3.x', 'comp1.c'], [[-0.5]])

    def test_list_inputs_before_run(self):
        msg = "Unable to list inputs until model has been run."
        try:
            self.prob.model.list_inputs()
        except Exception as err:
            self.assertTrue(msg == str(err))
        else:
            self.fail("Exception expected")

    def test_list_outputs_before_run(self):
        msg = "Unable to list outputs until model has been run."
        try:
            self.prob.model.list_outputs()
        except Exception as err:
            self.assertTrue(msg == str(err))
        else:
            self.fail("Exception expected")

    def test_list_residuals_before_run(self):
        msg = "Unable to list residuals until model has been run."
        try:
            self.prob.model.list_residuals()
        except Exception as err:
            self.assertTrue(msg == str(err))
        else:
            self.fail("Exception expected")

    def test_list_inputs(self):
        self.prob.run_model()

        stream = cStringIO()
        inputs = self.prob.model.list_inputs(out_stream=stream)
        self.assertEqual(sorted(inputs), [
            ('comp2.a', [1.]),
            ('comp2.b', [-4.]),
            ('comp2.c', [3.]),
            ('comp3.a', [1.]),
            ('comp3.b', [-4.]),
            ('comp3.c', [3.])
        ])
        text = stream.getvalue()
        self.assertEqual(text.count('comp2.'), 3)
        self.assertEqual(text.count('comp3.'), 3)
        self.assertEqual(text.count('value:'), 6)

    def test_list_explicit_outputs(self):
        self.prob.run_model()

        stream = cStringIO()
        outputs = self.prob.model.list_outputs(implicit=False, out_stream=stream)
        self.assertEqual(sorted(outputs), [
            ('comp1.a', [1.]),
            ('comp1.b', [-4.]),
            ('comp1.c', [3.])
        ])
        text = stream.getvalue()
        self.assertEqual(text.count('comp1.'), 3)
        self.assertEqual(text.count('value:'), 3)
        self.assertEqual(text.count('residual:'), 3)

    def test_list_implicit_outputs(self):
        self.prob.run_model()

        stream = cStringIO()
        states = self.prob.model.list_outputs(explicit=False, out_stream=stream)
        self.assertEqual(sorted(states), [
            ('comp2.x', [3.]),
            ('comp3.x', [3.])
        ])
        text = stream.getvalue()
        self.assertEqual(text.count('comp2.x'), 1)
        self.assertEqual(text.count('comp3.x'), 1)
        self.assertEqual(text.count('value:'), 2)
        self.assertEqual(text.count('residual:'), 2)

    def test_list_residuals(self):
        self.prob.run_model()

        stream = cStringIO()
        resids = self.prob.model.list_residuals(out_stream=stream)
        self.assertEqual(sorted(resids), [
            ('comp1.a', [0.]),
            ('comp1.b', [0.]),
            ('comp1.c', [0.]),
            ('comp2.x', [0.]),
            ('comp3.x', [0.])
        ])
        text = stream.getvalue()
        self.assertEqual(text.count('comp1.'), 3)
        self.assertEqual(text.count('comp2.x'), 1)
        self.assertEqual(text.count('comp3.x'), 1)
        self.assertEqual(text.count('value:'), 5)
        self.assertEqual(text.count('residual:'), 5)

    def test_guess_nonlinear(self):

        class ImpWithInitial(QuadraticLinearize):

            def solve_nonlinear(self, inputs, outputs):
                """ Do nothing. """
                pass

            def guess_nonlinear(self, inputs, outputs, resids):
                # Solution at x=1 and x=3. Default value takes us to the x=1 solution. Here
                # we set it to a value that will take us to the x=3 solution.
                outputs['x'] = 5.0


        group = Group()

        group.add_subsystem('pa', IndepVarComp('a', 1.0))
        group.add_subsystem('pb', IndepVarComp('b', 1.0))
        group.add_subsystem('pc', IndepVarComp('c', 1.0))
        group.add_subsystem('comp2', ImpWithInitial())
        group.connect('pa.a', 'comp2.a')
        group.connect('pb.b', 'comp2.b')
        group.connect('pc.c', 'comp2.c')

        prob = Problem(model=group)
        group.nonlinear_solver = NewtonSolver()
        group.nonlinear_solver.options['solve_subsystems'] = True
        group.nonlinear_solver.options['max_sub_solves'] = 1
        group.linear_solver = ScipyIterativeSolver()

        prob.setup(check=False)

        prob['pa.a'] = 1.
        prob['pb.b'] = -4.
        prob['pc.c'] = 3.

        # Making sure that guess_nonlinear is called early enough to eradicate this.
        prob['comp2.x'] = np.NaN

        prob.run_model()
        assert_rel_error(self, prob['comp2.x'], 3.)

    def test_guess_nonlinear_transfer(self):
        # Test that data is transfered to a component before calling guess_nonlinear.

        class ImpWithInitial(ImplicitComponent):

            def setup(self):
                self.add_input('x', 3.0)
                self.add_output('y', 4.0)

            def solve_nonlinear(self, inputs, outputs):
                """ Do nothing. """
                pass

            def apply_nonlinear(self, inputs, outputs, resids):
                """ Do nothing. """
                pass

            def guess_nonlinear(self, inputs, outputs, resids):
                # Passthrough
                outputs['y'] = inputs['x']


        group = Group()

        group.add_subsystem('px', IndepVarComp('x', 77.0))
        group.add_subsystem('comp1', ImpWithInitial())
        group.add_subsystem('comp2', ImpWithInitial())
        group.connect('px.x', 'comp1.x')
        group.connect('comp1.y', 'comp2.x')

        group.nonlinear_solver = NewtonSolver()
        group.nonlinear_solver.options['maxiter'] = 1

        prob = Problem(model=group)
        prob.set_solver_print(level=0)
        prob.setup(check=False)

        prob.run_model()
        assert_rel_error(self, prob['comp2.y'], 77., 1e-5)

    def test_guess_nonlinear_transfer_subbed(self):
        # Test that data is transfered to a component before calling guess_nonlinear.

        class ImpWithInitial(ImplicitComponent):

            def setup(self):
                self.add_input('x', 3.0)
                self.add_output('y', 4.0)

            def solve_nonlinear(self, inputs, outputs):
                """ Do nothing. """
                pass

            def apply_nonlinear(self, inputs, outputs, resids):
                """ Do nothing. """
                resids['y'] = 1.0e-6
                pass

            def guess_nonlinear(self, inputs, outputs, resids):
                # Passthrough
                outputs['y'] = inputs['x']


        group = Group()
        sub = Group()

        group.add_subsystem('px', IndepVarComp('x', 77.0))
        sub.add_subsystem('comp1', ImpWithInitial())
        sub.add_subsystem('comp2', ImpWithInitial())
        group.connect('px.x', 'sub.comp1.x')
        group.connect('sub.comp1.y', 'sub.comp2.x')

        group.add_subsystem('sub', sub)

        group.nonlinear_solver = NewtonSolver()
        group.nonlinear_solver.options['maxiter'] = 1

        prob = Problem(model=group)
        prob.set_solver_print(level=0)
        prob.setup(check=False)

        prob.run_model()
        assert_rel_error(self, prob['sub.comp2.y'], 77., 1e-5)

    def test_guess_nonlinear_transfer_subbed2(self):
        # Test that data is transfered to a component before calling guess_nonlinear.

        class ImpWithInitial(ImplicitComponent):

            def setup(self):
                self.add_input('x', 3.0)
                self.add_output('y', 4.0)

            def solve_nonlinear(self, inputs, outputs):
                """ Do nothing. """
                pass

            def apply_nonlinear(self, inputs, outputs, resids):
                """ Do nothing. """
                resids['y'] = 1.0e-6
                pass

            def guess_nonlinear(self, inputs, outputs, resids):
                # Passthrough
                outputs['y'] = inputs['x']


        group = Group()
        sub = Group()

        group.add_subsystem('px', IndepVarComp('x', 77.0))
        sub.add_subsystem('comp1', ImpWithInitial())
        sub.add_subsystem('comp2', ImpWithInitial())
        group.connect('px.x', 'sub.comp1.x')
        group.connect('sub.comp1.y', 'sub.comp2.x')

        group.add_subsystem('sub', sub)

        sub.nonlinear_solver = NewtonSolver()
        sub.nonlinear_solver.options['maxiter'] = 1

        prob = Problem(model=group)
        prob.set_solver_print(level=0)
        prob.setup(check=False)

        prob.run_model()
        assert_rel_error(self, prob['sub.comp2.y'], 77., 1e-5)

    def test_guess_nonlinear_feature(self):

        class ImpWithInitial(ImplicitComponent):

            def setup(self):
                self.add_input('a', val=1.)
                self.add_input('b', val=1.)
                self.add_input('c', val=1.)
                self.add_output('x', val=0.)

                self.declare_partials(of='*', wrt='*')

            def apply_nonlinear(self, inputs, outputs, residuals):
                a = inputs['a']
                b = inputs['b']
                c = inputs['c']
                x = outputs['x']
                residuals['x'] = a * x ** 2 + b * x + c

            def solve_nonlinear(self, inputs, outputs):
                a = inputs['a']
                b = inputs['b']
                c = inputs['c']
                outputs['x'] = (-b + (b ** 2 - 4 * a * c) ** 0.5) / 2 / a

            def linearize(self, inputs, outputs, partials):
                a = inputs['a']
                b = inputs['b']
                c = inputs['c']
                x = outputs['x']

                partials['x', 'a'] = x ** 2
                partials['x', 'b'] = x
                partials['x', 'c'] = 1.0
                partials['x', 'x'] = 2 * a * x + b

                self.inv_jac = 1.0 / (2 * a * x + b)

            def solve_nonlinear(self, inputs, outputs):
                """ Do nothing. """
                pass

            def guess_nonlinear(self, inputs, outputs, resids):
                # Solution at 1 and 3. Default value takes us to -1 solution. Here
                # we set it to a value that will tke us to the 3 solution.
                outputs['x'] = 5.0


        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('pa', IndepVarComp('a', 1.0))
        model.add_subsystem('pb', IndepVarComp('b', 1.0))
        model.add_subsystem('pc', IndepVarComp('c', 1.0))
        model.add_subsystem('comp2', ImpWithInitial())
        model.connect('pa.a', 'comp2.a')
        model.connect('pb.b', 'comp2.b')
        model.connect('pc.c', 'comp2.c')

        model.nonlinear_solver = NewtonSolver()
        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 1
        model.linear_solver = ScipyIterativeSolver()

        prob.setup(check=False)

        prob['pa.a'] = 1.
        prob['pb.b'] = -4.
        prob['pc.c'] = 3.

        prob.run_model()

        assert_rel_error(self, prob['comp2.x'], 3.)


class ListFeatureTestCase(unittest.TestCase):

    def setUp(self):
        group = Group()

        comp1 = group.add_subsystem('comp1', IndepVarComp())
        comp1.add_output('a', 1.0)
        comp1.add_output('b', 1.0)
        comp1.add_output('c', 1.0)

        sub = group.add_subsystem('sub', Group())
        sub.add_subsystem('comp2', QuadraticComp())
        sub.add_subsystem('comp3', QuadraticComp())

        group.connect('comp1.a', 'sub.comp2.a')
        group.connect('comp1.b', 'sub.comp2.b')
        group.connect('comp1.c', 'sub.comp2.c')
        group.connect('comp1.a', 'sub.comp3.a')
        group.connect('comp1.b', 'sub.comp3.b')
        group.connect('comp1.c', 'sub.comp3.c')

        global prob  # so we don't need `self.` in feature doc
        prob = Problem(model=group)
        prob.setup()

        prob['comp1.a'] = 1.
        prob['comp1.b'] = -4.
        prob['comp1.c'] = 3.
        prob.run_model()

    def test_list_inputs(self):
        prob.model.list_inputs()

    def test_list_outputs(self):
        prob.model.list_outputs()

    def test_list_explicit_outputs(self):
        prob.model.list_outputs(implicit=False)

    def test_list_implicit_outputs(self):
        prob.model.list_outputs(explicit=False)

    def test_list_residuals(self):
        prob.model.list_residuals()

    def test_list_return_value(self):
        # list inputs
        inputs = prob.model.list_inputs(out_stream=None)
        self.assertEqual(sorted(inputs), [
            ('sub.comp2.a', [1.]),
            ('sub.comp2.b', [-4.]),
            ('sub.comp2.c', [3.]),
            ('sub.comp3.a', [1.]),
            ('sub.comp3.b', [-4.]),
            ('sub.comp3.c', [3.])
        ])

        # list explicit outputs
        outputs = prob.model.list_outputs(implicit=False, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('comp1.a', [1.]),
            ('comp1.b', [-4.]),
            ('comp1.c', [3.])
        ])

        # list residuals
        resids = prob.model.list_residuals(out_stream=None)
        self.assertEqual(sorted(resids), [
            ('comp1.a', [0.]),
            ('comp1.b', [0.]),
            ('comp1.c', [0.]),
            ('sub.comp2.x', [0.]),
            ('sub.comp3.x', [0.])
        ])

    def test_list_no_values(self):
        # list inputs
        inputs = prob.model.list_inputs(values=False)
        self.assertEqual(sorted(inputs), [
            'sub.comp2.a',
            'sub.comp2.b',
            'sub.comp2.c',
            'sub.comp3.a',
            'sub.comp3.b',
            'sub.comp3.c'
        ])

        # list explicit outputs
        outputs = prob.model.list_outputs(implicit=False, values=False)
        self.assertEqual(sorted(outputs), [
            'comp1.a',
            'comp1.b',
            'comp1.c'
        ])

        # list residuals
        resids = prob.model.list_residuals(values=False)
        self.assertEqual(sorted(resids), [
            'comp1.a',
            'comp1.b',
            'comp1.c',
            'sub.comp2.x',
            'sub.comp3.x'
        ])


if __name__ == '__main__':
    unittest.main()
