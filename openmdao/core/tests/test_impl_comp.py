"""Simple example demonstrating how to implement an implicit component."""
from __future__ import division

import unittest

from six.moves import cStringIO
import numpy as np

from openmdao.api import Problem, Group, ImplicitComponent, IndepVarComp, \
    NewtonSolver, ScipyKrylov, AnalysisError
from openmdao.utils.assert_utils import assert_rel_error


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

    def test_list_inputs(self):
        self.prob.run_model()

        stream = cStringIO()
        inputs = self.prob.model.list_inputs(hierarchical=False, out_stream=stream)
        self.assertEqual(sorted(inputs), [
            ('comp2.a', {'value': [1.]}),
            ('comp2.b', {'value': [-4.]}),
            ('comp2.c', {'value': [3.]}),
            ('comp3.a', {'value': [1.]}),
            ('comp3.b', {'value': [-4.]}),
            ('comp3.c', {'value': [3.]})
        ])
        text = stream.getvalue()
        self.assertEqual(text.count('comp2.'), 3)
        self.assertEqual(text.count('comp3.'), 3)
        self.assertEqual(text.count('value'), 1)

    def test_list_inputs_prom_name(self):
        self.prob.run_model()

        stream = cStringIO()
        states = self.prob.model.list_inputs(prom_name=True, shape=True, hierarchical=True,
                                             out_stream=stream)

        text = stream.getvalue()

        self.assertEqual(text.count('comp2.a'), 1)
        self.assertEqual(text.count('comp2.b'), 1)
        self.assertEqual(text.count('comp2.c'), 1)
        self.assertEqual(text.count('comp3.a'), 1)
        self.assertEqual(text.count('comp3.b'), 1)
        self.assertEqual(text.count('comp3.c'), 1)

        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(num_non_empty_lines, 13)

    def test_list_explicit_outputs(self):
        self.prob.run_model()

        stream = cStringIO()
        outputs = self.prob.model.list_outputs(implicit=False, hierarchical=False, out_stream=stream)
        self.assertEqual(sorted(outputs), [
            ('comp1.a', {'value': [1.]}),
            ('comp1.b', {'value': [-4.]}),
            ('comp1.c', {'value': [3.]})
        ])
        text = stream.getvalue()
        self.assertEqual(text.count('comp1.'), 3)
        self.assertEqual(text.count('varname'), 1)
        self.assertEqual(text.count('value'), 1)

    def test_list_implicit_outputs(self):
        self.prob.run_model()

        stream = cStringIO()
        states = self.prob.model.list_outputs(explicit=False, residuals=True,
                                              hierarchical=False, out_stream=stream)
        self.assertTrue(('comp2.x', {'value': [3.], 'resids': [0.]}) in states, msg=None)
        self.assertTrue(('comp3.x', {'value': [3.], 'resids': [0.]}) in states, msg=None)
        text = stream.getvalue()
        self.assertEqual(1, text.count('comp2.x'))
        self.assertEqual(1, text.count('comp3.x'))
        self.assertEqual(1, text.count('value'))
        self.assertEqual(1, text.count('resids'))

    def test_list_outputs_prom_name(self):
        self.prob.run_model()

        stream = cStringIO()
        states = self.prob.model.list_outputs(explicit=False, residuals=True,
                                              prom_name=True, hierarchical=True,
                                              out_stream=stream)

        text = stream.getvalue()
        self.assertEqual(text.count('comp2.x'), 1)
        self.assertEqual(text.count('comp3.x'), 1)
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(num_non_empty_lines, 9)

    def test_list_residuals(self):
        self.prob.run_model()

        stream = cStringIO()
        resids = self.prob.model.list_outputs(values=False, residuals=True, hierarchical=False,
                                              out_stream=stream)
        self.assertEqual(sorted(resids), [
            ('comp1.a', {'resids': [0.]}),
            ('comp1.b', {'resids': [0.]}),
            ('comp1.c', {'resids': [0.]}),
            ('comp2.x', {'resids': [0.]}),
            ('comp3.x', {'resids': [0.]})
        ])
        text = stream.getvalue()
        self.assertEqual(text.count('comp1.'), 3)
        self.assertEqual(text.count('comp2.x'), 1)
        self.assertEqual(text.count('comp3.x'), 1)
        self.assertEqual(text.count('varname'), 2)
        self.assertEqual(text.count('value'), 0)
        self.assertEqual(text.count('resids'), 2)


class ImplicitCompGuessTestCase(unittest.TestCase):

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
        group.linear_solver = ScipyKrylov()

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
        from openmdao.api import Problem, Group, ImplicitComponent, IndepVarComp, NewtonSolver, ScipyKrylov

        class ImpWithInitial(ImplicitComponent):
            """
            An implicit component to solve the quadratic equation: x^2 - 4x + 3
            (solutions at x=1 and x=3)
            """
            def setup(self):
                self.add_input('a', val=1.)
                self.add_input('b', val=-4.)
                self.add_input('c', val=3.)

                self.add_output('x', val=0.)

                self.declare_partials(of='*', wrt='*')

            def apply_nonlinear(self, inputs, outputs, residuals):
                a = inputs['a']
                b = inputs['b']
                c = inputs['c']
                x = outputs['x']
                residuals['x'] = a * x ** 2 + b * x + c

            def linearize(self, inputs, outputs, partials):
                a = inputs['a']
                b = inputs['b']
                c = inputs['c']
                x = outputs['x']

                partials['x', 'a'] = x ** 2
                partials['x', 'b'] = x
                partials['x', 'c'] = 1.0
                partials['x', 'x'] = 2 * a * x + b

            def guess_nonlinear(self, inputs, outputs, resids):
                # Default initial state of zero for x takes us to x=1 solution.
                # Here we set it to a value that will take us to the x=3 solution.
                outputs['x'] = 5.0

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('comp', ImpWithInitial())

        model.nonlinear_solver = NewtonSolver()
        model.linear_solver = ScipyKrylov()

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['comp.x'], 3.)

    def test_guess_nonlinear_inputs_read_only(self):
        class ImpWithInitial(ImplicitComponent):

            def setup(self):
                self.add_input('x', 3.0)
                self.add_output('y', 4.0)

            def guess_nonlinear(self, inputs, outputs, resids):
                # inputs is read_only, should not be allowed
                inputs['x'] = 0.

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

        with self.assertRaises(ValueError) as cm:
            prob.run_model()

        self.assertEqual(str(cm.exception),
                         "Attempt to set value of 'x' in input vector "
                         "when it is read only.")

    def test_guess_nonlinear_inputs_read_only_reset(self):
        class ImpWithInitial(ImplicitComponent):

            def setup(self):
                self.add_input('x', 3.0)
                self.add_output('y', 4.0)

            def guess_nonlinear(self, inputs, outputs, resids):
                raise AnalysisError("It's just a scratch.")

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

        with self.assertRaises(AnalysisError):
            prob.run_model()

        # verify read_only status is reset after AnalysisError
        prob['comp1.x'] = 111.

    def test_guess_nonlinear_resids_read_only(self):
        class ImpWithInitial(ImplicitComponent):

            def setup(self):
                self.add_input('x', 3.0)
                self.add_output('y', 4.0)

            def guess_nonlinear(self, inputs, outputs, resids):
                # inputs is read_only, should not be allowed
                resids['y'] = 0.

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

        with self.assertRaises(ValueError) as cm:
            prob.run_model()

        self.assertEqual(str(cm.exception),
                         "Attempt to set value of 'y' in residual vector "
                         "when it is read only.")


class ImplicitCompReadOnlyTestCase(unittest.TestCase):

    def test_apply_nonlinear_inputs_read_only(self):
        class BadComp(QuadraticComp):
            def apply_nonlinear(self, inputs, outputs, residuals):
                super(BadComp, self).apply_nonlinear(inputs, outputs, residuals)
                inputs['a'] = 0.  # should not be allowed

        prob = Problem()
        prob.model.add_subsystem('bad', BadComp())
        prob.setup()
        prob.run_model()

        # check input vector
        with self.assertRaises(ValueError) as cm:
            prob.model.run_apply_nonlinear()

        self.assertEqual(str(cm.exception),
                         "Attempt to set value of 'a' in input vector "
                         "when it is read only.")

    def test_apply_nonlinear_outputs_read_only(self):
        class BadComp(QuadraticComp):
            def apply_nonlinear(self, inputs, outputs, residuals):
                super(BadComp, self).apply_nonlinear(inputs, outputs, residuals)
                outputs['x'] = 0.  # should not be allowed

        prob = Problem()
        prob.model.add_subsystem('bad', BadComp())
        prob.setup()
        prob.run_model()

        # check output vector
        with self.assertRaises(ValueError) as cm:
            prob.model.run_apply_nonlinear()

        self.assertEqual(str(cm.exception),
                         "Attempt to set value of 'x' in output vector "
                         "when it is read only.")

    def test_apply_nonlinear_read_only_reset(self):
        class BadComp(QuadraticComp):
            def apply_nonlinear(self, inputs, outputs, residuals):
                super(BadComp, self).apply_nonlinear(inputs, outputs, residuals)
                raise AnalysisError("It's just a scratch.")

        prob = Problem()
        prob.model.add_subsystem('bad', BadComp())
        prob.setup()
        prob.run_model()

        with self.assertRaises(AnalysisError):
            prob.model.run_apply_nonlinear()

        # verify read_only status is reset after AnalysisError
        prob['bad.a'] = 111.
        prob['bad.x'] = 111.

    def test_solve_nonlinear_inputs_read_only(self):
        class BadComp(QuadraticComp):
            def solve_nonlinear(self, inputs, outputs):
                super(BadComp, self).solve_nonlinear(inputs, outputs)
                inputs['a'] = 0.  # should not be allowed

        prob = Problem()
        prob.model.add_subsystem('bad', BadComp())
        prob.setup()

        # check input vector
        with self.assertRaises(ValueError) as cm:
            prob.run_model()

        self.assertEqual(str(cm.exception),
                         "Attempt to set value of 'a' in input vector "
                         "when it is read only.")

    def test_solve_nonlinear_inputs_read_only_reset(self):
        class BadComp(QuadraticComp):
            def solve_nonlinear(self, inputs, outputs):
                super(BadComp, self).solve_nonlinear(inputs, outputs)
                raise AnalysisError("It's just a scratch.")

        prob = Problem()
        prob.model.add_subsystem('bad', BadComp())
        prob.setup()

        with self.assertRaises(AnalysisError):
            prob.run_model()

        # verify read_only status is reset after AnalysisError
        prob['bad.a'] = 111.

    def test_linearize_inputs_read_only(self):
        class BadComp(QuadraticLinearize):
            def linearize(self, inputs, outputs, partials):
                super(BadComp, self).linearize(inputs, outputs, partials)
                inputs['a'] = 0.  # should not be allowed

        prob = Problem()
        prob.model.add_subsystem('bad', BadComp())
        prob.setup()
        prob.run_model()

        # check input vector
        with self.assertRaises(ValueError) as cm:
            prob.model.run_linearize()

        self.assertEqual(str(cm.exception),
                         "Attempt to set value of 'a' in input vector "
                         "when it is read only.")

    def test_linearize_outputs_read_only(self):
        class BadComp(QuadraticLinearize):
            def linearize(self, inputs, outputs, partials):
                super(BadComp, self).linearize(inputs, outputs, partials)
                outputs['x'] = 0.  # should not be allowed

        prob = Problem()
        prob.model.add_subsystem('bad', BadComp())
        prob.setup()
        prob.run_model()

        # check input vector
        with self.assertRaises(ValueError) as cm:
            prob.model.run_linearize()

        self.assertEqual(str(cm.exception),
                         "Attempt to set value of 'x' in output vector "
                         "when it is read only.")

    def test_linearize_read_only_reset(self):
        class BadComp(QuadraticLinearize):
            def linearize(self, inputs, outputs, partials):
                super(BadComp, self).linearize(inputs, outputs, partials)
                raise AnalysisError("It's just a scratch.")

        prob = Problem()
        prob.model.add_subsystem('bad', BadComp())
        prob.setup()
        prob.run_model()

        with self.assertRaises(AnalysisError):
            prob.model.run_linearize()

        # verify read_only status is reset after AnalysisError
        prob['bad.a'] = 111.
        prob['bad.x'] = 111.

    def test_apply_linear_inputs_read_only(self):
        class BadComp(QuadraticJacVec):
            def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
                super(BadComp, self).apply_linear(inputs, outputs,
                                                  d_inputs, d_outputs, d_residuals, mode)
                inputs['a'] = 0.  # should not be allowed

        prob = Problem()
        prob.model.add_subsystem('bad', BadComp())
        prob.setup()
        prob.run_model()

        # check input vector
        with self.assertRaises(ValueError) as cm:
            prob.model.run_apply_linear(['linear'], 'fwd')

        self.assertEqual(str(cm.exception),
                         "Attempt to set value of 'a' in input vector "
                         "when it is read only.")

    def test_apply_linear_outputs_read_only(self):
        class BadComp(QuadraticJacVec):
            def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
                super(BadComp, self).apply_linear(inputs, outputs,
                                                  d_inputs, d_outputs, d_residuals, mode)
                outputs['x'] = 0.  # should not be allowed

        prob = Problem()
        prob.model.add_subsystem('bad', BadComp())
        prob.setup()
        prob.run_model()

        # check input vector
        with self.assertRaises(ValueError) as cm:
            prob.model.run_apply_linear(['linear'], 'fwd')

        self.assertEqual(str(cm.exception),
                         "Attempt to set value of 'x' in output vector "
                         "when it is read only.")

    def test_apply_linear_dinputs_read_only(self):
        class BadComp(QuadraticJacVec):
            def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
                super(BadComp, self).apply_linear(inputs, outputs,
                                                  d_inputs, d_outputs, d_residuals, mode)
                d_inputs['a'] = 0.  # should not be allowed

        prob = Problem()
        prob.model.add_subsystem('bad', BadComp())
        prob.setup()
        prob.run_model()

        # check input vector
        with self.assertRaises(ValueError) as cm:
            prob.model.run_apply_linear(['linear'], 'fwd')

        self.assertEqual(str(cm.exception),
                         "Attempt to set value of 'a' in input vector "
                         "when it is read only.")

    def test_apply_linear_doutputs_read_only(self):
        class BadComp(QuadraticJacVec):
            def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
                super(BadComp, self).apply_linear(inputs, outputs,
                                                  d_inputs, d_outputs, d_residuals, mode)
                d_outputs['x'] = 0.  # should not be allowed

        prob = Problem()
        prob.model.add_subsystem('bad', BadComp())
        prob.setup()
        prob.run_model()

        # check input vector
        with self.assertRaises(ValueError) as cm:
            prob.model.run_apply_linear(['linear'], 'fwd')

        self.assertEqual(str(cm.exception),
                         "Attempt to set value of 'x' in output vector "
                         "when it is read only.")

    def test_apply_linear_dresids_read_only(self):
        class BadComp(QuadraticJacVec):
            def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
                super(BadComp, self).apply_linear(inputs, outputs,
                                                  d_inputs, d_outputs, d_residuals, mode)
                d_residuals['x'] = 0.  # should not be allowed

        prob = Problem()
        prob.model.add_subsystem('bad', BadComp())
        prob.setup()
        prob.run_model()

        # check input vector
        with self.assertRaises(ValueError) as cm:
            prob.model.run_apply_linear(['linear'], 'rev')

        self.assertEqual(str(cm.exception),
                         "Attempt to set value of 'x' in residual vector "
                         "when it is read only.")

    def test_apply_linear_read_only_reset(self):
        class BadComp(QuadraticJacVec):
            def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
                super(BadComp, self).apply_linear(inputs, outputs,
                                                  d_inputs, d_outputs, d_residuals, mode)
                raise AnalysisError("It's just a scratch.")

        prob = Problem()
        prob.model.add_subsystem('bad', BadComp())
        prob.setup()
        prob.run_model()

        with self.assertRaises(AnalysisError):
            prob.model.run_apply_linear(['linear'], 'rev')

        # verify read_only status is reset after AnalysisError
        prob['bad.a'] = 111.
        prob['bad.x'] = 111.
        prob.model.bad._vectors['residual']['linear']['x'] = 111.

    def test_solve_linear_doutputs_read_only(self):
        class BadComp(QuadraticJacVec):
            def solve_linear(self, d_outputs, d_residuals, mode):
                super(BadComp, self).solve_linear(d_outputs, d_residuals, mode)
                d_outputs['x'] = 0.  # should not be allowed

        prob = Problem()
        prob.model.add_subsystem('bad', BadComp())
        prob.setup()
        prob.run_model()
        prob.model.run_linearize()

        # check input vector
        with self.assertRaises(ValueError) as cm:
            prob.model.run_solve_linear(['linear'], 'rev')

        self.assertEqual(str(cm.exception),
                         "Attempt to set value of 'x' in output vector "
                         "when it is read only.")

    def test_solve_linear_dresids_read_only(self):
        class BadComp(QuadraticJacVec):
            def solve_linear(self, d_outputs, d_residuals, mode):
                super(BadComp, self).solve_linear(d_outputs, d_residuals, mode)
                d_residuals['x'] = 0.  # should not be allowed

        prob = Problem()
        prob.model.add_subsystem('bad', BadComp())
        prob.setup()
        prob.run_model()
        prob.model.run_linearize()

        # check input vector
        with self.assertRaises(ValueError) as cm:
            prob.model.run_solve_linear(['linear'], 'fwd')

        self.assertEqual(str(cm.exception),
                         "Attempt to set value of 'x' in residual vector "
                         "when it is read only.")

    def test_solve_linear_read_only_reset(self):
        class BadComp(QuadraticJacVec):
            def solve_linear(self, d_outputs, d_residuals, mode):
                super(BadComp, self).solve_linear(d_outputs, d_residuals, mode)
                raise AnalysisError("It's just a scratch.")

        prob = Problem()
        prob.model.add_subsystem('bad', BadComp())
        prob.setup()
        prob.run_model()
        prob.model.run_linearize()

        with self.assertRaises(AnalysisError):
            prob.model.run_solve_linear(['linear'], 'fwd')

        # verify read_only status is reset after AnalysisError
        prob.model.bad._vectors['residual']['linear']['x'] = 111.


class ListFeatureTestCase(unittest.TestCase):

    def setUp(self):
        from openmdao.api import Group, Problem, IndepVarComp
        from openmdao.core.tests.test_impl_comp import QuadraticComp

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

        global prob
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
        prob.model.list_outputs(residuals=True)

    def test_list_prom_names(self):
        prob.model.list_outputs(prom_name=True)

    def test_list_return_value(self):
        inputs = prob.model.list_inputs(out_stream=None)
        self.assertEqual(sorted(inputs), [
            ('sub.comp2.a', {'value': [1.]}),
            ('sub.comp2.b', {'value': [-4.]}),
            ('sub.comp2.c', {'value': [3.]}),
            ('sub.comp3.a', {'value': [1.]}),
            ('sub.comp3.b', {'value': [-4.]}),
            ('sub.comp3.c', {'value': [3.]})
        ])

        # list explicit outputs
        outputs = prob.model.list_outputs(implicit=False, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('comp1.a', {'value': [1.]}),
            ('comp1.b', {'value': [-4.]}),
            ('comp1.c', {'value': [3.]})
        ])

    def test_for_docs_list_no_values(self):
        inputs = prob.model.list_inputs(values=False)
        print(inputs)

        # list only explicit outputs
        outputs = prob.model.list_outputs(implicit=False, values=False)
        print(outputs)

    def test_list_no_values(self):
        inputs = prob.model.list_inputs(values=False)
        self.assertEqual([n[0] for n in sorted(inputs)], [
            'sub.comp2.a',
            'sub.comp2.b',
            'sub.comp2.c',
            'sub.comp3.a',
            'sub.comp3.b',
            'sub.comp3.c'
        ])

        # list only explicit outputs
        outputs = prob.model.list_outputs(implicit=False, values=False)
        self.assertEqual([n[0] for n in sorted(outputs)], [
            'comp1.a',
            'comp1.b',
            'comp1.c'
        ])

    def test_simple_list_vars_options(self):
        from openmdao.api import Group, Problem, IndepVarComp

        class QuadraticComp(ImplicitComponent):
            """
            A Simple Implicit Component representing a Quadratic Equation.

            R(a, b, c, x) = ax^2 + bx + c

            Solution via Quadratic Formula:
            x = (-b + sqrt(b^2 - 4ac)) / 2a
            """

            def setup(self):
                self.add_input('a', val=1., units='ft')
                self.add_input('b', val=1., units='inch')
                self.add_input('c', val=1., units='ft')
                self.add_output('x', val=0.,
                                lower=1.0, upper=100.0,
                                ref=1.1, ref0=2.1,
                                units='inch')

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

        group = Group()

        comp1 = group.add_subsystem('comp1', IndepVarComp())
        comp1.add_output('a', 1.0, units='ft')
        comp1.add_output('b', 1.0, units='inch')
        comp1.add_output('c', 1.0, units='ft')

        sub = group.add_subsystem('sub', Group())
        sub.add_subsystem('comp2', QuadraticComp())
        sub.add_subsystem('comp3', QuadraticComp())

        group.connect('comp1.a', 'sub.comp2.a')
        group.connect('comp1.b', 'sub.comp2.b')
        group.connect('comp1.c', 'sub.comp2.c')

        group.connect('comp1.a', 'sub.comp3.a')
        group.connect('comp1.b', 'sub.comp3.b')
        group.connect('comp1.c', 'sub.comp3.c')

        global prob
        prob = Problem(model=group)
        prob.setup()

        prob['comp1.a'] = 1.
        prob['comp1.b'] = -4.
        prob['comp1.c'] = 3.
        prob.run_model()

        # list_inputs test
        stream = cStringIO()
        inputs = prob.model.list_inputs(values=False, out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(sorted(inputs), [
            ('sub.comp2.a', {}),
            ('sub.comp2.b', {}),
            ('sub.comp2.c', {}),
            ('sub.comp3.a', {}),
            ('sub.comp3.b', {}),
            ('sub.comp3.c', {}),
        ])
        self.assertEqual(1, text.count("6 Input(s) in 'model'"))
        self.assertEqual(1, text.count("top"))
        self.assertEqual(1, text.count("  sub"))
        self.assertEqual(1, text.count("    comp2"))
        self.assertEqual(2, text.count("      a"))
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(num_non_empty_lines, 14)

        # list_outputs tests
        # list implicit outputs
        outputs = prob.model.list_outputs(explicit=False, out_stream=None)
        text = stream.getvalue()
        self.assertEqual(sorted(outputs), [
            ('sub.comp2.x', {'value': [3.]}),
            ('sub.comp3.x', {'value': [3.]})
        ])
        # list explicit outputs
        stream = cStringIO()
        outputs = prob.model.list_outputs(implicit=False, out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('comp1.a', {'value': [1.]}),
            ('comp1.b', {'value': [-4.]}),
            ('comp1.c', {'value': [3.]}),
        ])

    def test_list_residuals_with_tol(self):
        from openmdao.test_suite.components.sellar import SellarImplicitDis1, SellarImplicitDis2
        from openmdao.api import Problem, Group, IndepVarComp, NewtonSolver, ScipyKrylov, LinearBlockGS
        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 1.0))
        model.add_subsystem('d1', SellarImplicitDis1())
        model.add_subsystem('d2', SellarImplicitDis2())
        model.connect('d1.y1', 'd2.y1')
        model.connect('d2.y2', 'd1.y2')

        model.nonlinear_solver = NewtonSolver()
        model.nonlinear_solver.options['maxiter'] = 5
        model.linear_solver = ScipyKrylov()
        model.linear_solver.precon = LinearBlockGS()

        prob.setup(check=False)
        prob.set_solver_print(level=-1)

        prob.run_model()

        outputs = model.list_outputs(residuals_tol=0.01, residuals=True)
        print(outputs)


class CacheUsingComp(ImplicitComponent):
    def setup(self):
        self.cache = {}
        self.lin_sol_count = 0
        self.add_input('x', val=np.ones(10))
        self.add_output('y', val=np.zeros(10))

        self.declare_partials(of='*', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        x = inputs['x']
        y = outputs['y']
        residuals['y'] = x * y ** 2

    def solve_nonlinear(self, inputs, outputs):
        x = inputs['x']
        outputs['y'] = x ** 2 + 1.0

    def linearize(self, inputs, outputs, partials):
        subjac = np.zeros((inputs['x'].size, inputs['x'].size))
        for row, val in enumerate(inputs['x']):
            subjac[row, :] = inputs['x'] * 2.0
        partials['y', 'x'] = subjac
        self.lin_sol_count = 0

    def solve_linear(self, d_outputs, d_residuals, mode):
        # print('                    doutputs', d_outputs['y'])
        # print('dresids', d_residuals['y'])
        # if self.lin_sol_count in self.cache:
        #    print('cache  ', self.cache[self.lin_sol_count])

        fwd = mode == 'fwd'

        if self.lin_sol_count in self.cache:
            if fwd:
                assert(np.all(d_outputs['y'] == self.cache[self.lin_sol_count]))
            else:
                assert(np.all(d_residuals['y'] == self.cache[self.lin_sol_count]))

        if fwd:
            d_outputs['y'] = d_residuals['y'] + 2.
            self.cache[self.lin_sol_count] = d_outputs['y'].copy()
        else:  # rev
            d_residuals['y'] = d_outputs['y'] + 2.
            self.cache[self.lin_sol_count] = d_residuals['y'].copy()

        self.lin_sol_count += 1


class CacheLinSolutionTestCase(unittest.TestCase):
    def test_caching_fwd(self):
        p = Problem()
        p.model.add_subsystem('indeps', IndepVarComp('x', val=np.arange(10, dtype=float)))
        p.model.add_subsystem('C1', CacheUsingComp())
        p.model.connect('indeps.x', 'C1.x')
        p.model.add_design_var('indeps.x', cache_linear_solution=True)
        p.model.add_objective('C1.y')
        p.setup(mode='fwd')
        p.run_model()

        for i in range(10):
            p['indeps.x'] += np.arange(10, dtype=float)
            # run_model always runs setup_driver which resets the cached total jacobian object,
            # so save it here and restore after the run_model.  This is a contrived test.  In
            # real life, we only care about caching linear solutions when we're under run_driver.
            old_tot_jac = p.driver._total_jac
            p.run_model()
            p.driver._total_jac = old_tot_jac
            p.driver._compute_totals(of=['C1.y'], wrt=['indeps.x'])

    def test_caching_rev(self):
        p = Problem()
        p.model.add_subsystem('indeps', IndepVarComp('x', val=np.arange(10, dtype=float)))
        p.model.add_subsystem('C1', CacheUsingComp())
        p.model.connect('indeps.x', 'C1.x')
        p.model.add_design_var('indeps.x')
        p.model.add_objective('C1.y', cache_linear_solution=True)
        p.setup(mode='rev')
        p.run_model()

        for i in range(10):
            p['indeps.x'] += np.arange(10, dtype=float)
            # run_model always runs setup_driver which resets the cached total jacobian object,
            # so save it here and restore after the run_model.  This is a contrived test.  In
            # real life, we only care about caching linear solutions when we're under run_driver.
            old_tot_jac = p.driver._total_jac
            p.run_model()
            p.driver._total_jac = old_tot_jac
            p.driver._compute_totals(of=['C1.y'], wrt=['indeps.x'])


if __name__ == '__main__':
    unittest.main()
