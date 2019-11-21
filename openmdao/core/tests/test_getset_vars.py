"""Test getting/setting variables and subjacs with promoted/relative/absolute names."""

import unittest
from six import assertRaisesRegex

from openmdao.api import Problem, Group, ExecComp, IndepVarComp, DirectSolver


class TestGetSetVariables(unittest.TestCase):

    def test_no_promotion(self):
        """
        Illustrative examples showing how to access variables and subjacs.
        """
        c = ExecComp('y=2*x')

        g = Group()
        g.add_subsystem('c', c)

        model = Group()
        model.add_subsystem('g', g)

        p = Problem(model)
        p.setup()

        # -------------------------------------------------------------------

        # inputs
        p['g.c.x'] = 5.0
        self.assertEqual(p['g.c.x'], 5.0)

        # outputs
        p['g.c.y'] = 5.0
        self.assertEqual(p['g.c.y'], 5.0)

        # Conclude setup but don't run model.
        p.final_setup()

        inputs, outputs, residuals = g.get_nonlinear_vectors()

        # inputs
        inputs['c.x'] = 5.0
        self.assertEqual(inputs['c.x'], 5.0)

        # outputs
        outputs['c.y'] = 5.0
        self.assertEqual(outputs['c.y'], 5.0)

        # Removed part of test where we set values into the jacobian willy-nilly.
        # You can only set declared values now.

    def test_with_promotion(self):
        """
        Illustrative examples showing how to access variables and subjacs.
        """
        c1 = IndepVarComp('x')
        c2 = ExecComp('y=2*x')
        c3 = ExecComp('z=3*x')

        g = Group()
        g.add_subsystem('c1', c1, promotes=['*'])
        g.add_subsystem('c2', c2, promotes=['*'])
        g.add_subsystem('c3', c3, promotes=['*'])

        model = Group()
        model.add_subsystem('g', g, promotes=['*'])

        p = Problem(model)
        p.setup()

        # -------------------------------------------------------------------

        # inputs
        p['g.c2.x'] = 5.0
        self.assertEqual(p['g.c2.x'], 5.0)

        # outputs
        p['g.c2.y'] = 5.0
        self.assertEqual(p['g.c2.y'], 5.0)
        p['y'] = 5.0
        self.assertEqual(p['y'], 5.0)

        # Conclude setup but don't run model.
        p.final_setup()

        inputs, outputs, residuals = g.get_nonlinear_vectors()

        # inputs
        inputs['c2.x'] = 5.0
        self.assertEqual(inputs['c2.x'], 5.0)

        # outputs
        outputs['c2.y'] = 5.0
        self.assertEqual(outputs['c2.y'], 5.0)
        outputs['y'] = 5.0
        self.assertEqual(outputs['y'], 5.0)

        # Removed part of test where we set values into the jacobian willy-nilly. You can only set
        # declared values now.

    def test_no_promotion_errors(self):
        """
        Tests for error-handling for invalid variable names and keys.
        """
        g = Group(assembled_jac_type='dense')
        g.linear_solver = DirectSolver(assemble_jac=True)
        g.add_subsystem('c', ExecComp('y=2*x'))

        p = Problem()
        model = p.model
        model.add_subsystem('g', g)
        p.setup()

        # -------------------------------------------------------------------

        msg = '\'Group (<model>): Variable "{}" not found.\''

        # inputs
        with self.assertRaises(KeyError) as ctx:
            p['x'] = 5.0
            p.final_setup()
        self.assertEqual(str(ctx.exception), msg.format('x'))
        p._initial_condition_cache = {}

        with self.assertRaises(KeyError) as ctx:
            p['x']
        self.assertEqual(str(ctx.exception), msg.format('x'))

        # outputs
        with self.assertRaises(KeyError) as ctx:
            p['y'] = 5.0
            p.final_setup()
        self.assertEqual(str(ctx.exception), msg.format('y'))
        p._initial_condition_cache = {}

        with self.assertRaises(KeyError) as ctx:
            p['y']
        self.assertEqual(str(ctx.exception), msg.format('y'))


        msg = '\'Variable name "{}" not found.\''
        inputs, outputs, residuals = g.get_nonlinear_vectors()

        # inputs
        with assertRaisesRegex(self, KeyError, msg.format('x')):
            inputs['x'] = 5.0
        with assertRaisesRegex(self, KeyError, msg.format('x')):
            inputs['x']
        with assertRaisesRegex(self, KeyError, msg.format('g.c.x')):
            inputs['g.c.x'] = 5.0
        with assertRaisesRegex(self, KeyError, msg.format('g.c.x')):
            inputs['g.c.x']

        # outputs
        with assertRaisesRegex(self, KeyError, msg.format('y')):
            outputs['y'] = 5.0
        with assertRaisesRegex(self, KeyError, msg.format('y')):
            outputs['y']
        with assertRaisesRegex(self, KeyError, msg.format('g.c.y')):
            outputs['g.c.y'] = 5.0
        with assertRaisesRegex(self, KeyError, msg.format('g.c.y')):
            outputs['g.c.y']

        msg = r'Variable name pair \("{}", "{}"\) not found.'
        jac = g.linear_solver._assembled_jac

        # d(output)/d(input)
        with assertRaisesRegex(self, KeyError, msg.format('y', 'x')):
            jac['y', 'x'] = 5.0
        with assertRaisesRegex(self, KeyError, msg.format('y', 'x')):
            jac['y', 'x']
        # allow absolute keys now
        # with assertRaisesRegex(self, KeyError, msg.format('g.c.y', 'g.c.x')):
        #     jac['g.c.y', 'g.c.x'] = 5.0
        # with assertRaisesRegex(self, KeyError, msg.format('g.c.y', 'g.c.x')):
        #     deriv = jac['g.c.y', 'g.c.x']

        # d(output)/d(output)
        with assertRaisesRegex(self, KeyError, msg.format('y', 'y')):
            jac['y', 'y'] = 5.0
        with assertRaisesRegex(self, KeyError, msg.format('y', 'y')):
            jac['y', 'y']
        # allow absoute keys now
        # with assertRaisesRegex(self, KeyError, msg.format('g.c.y', 'g.c.y')):
        #     jac['g.c.y', 'g.c.y'] = 5.0
        # with assertRaisesRegex(self, KeyError, msg.format('g.c.y', 'g.c.y')):
        #     deriv = jac['g.c.y', 'g.c.y']

    def test_with_promotion_errors(self):
        """
        Tests for error-handling for invalid variable names and keys.
        """
        c1 = IndepVarComp('x')
        c2 = ExecComp('y=2*x')
        c3 = ExecComp('z=3*x')

        g = Group(assembled_jac_type='dense')
        g.add_subsystem('c1', c1, promotes=['*'])
        g.add_subsystem('c2', c2, promotes=['*'])
        g.add_subsystem('c3', c3, promotes=['*'])
        g.linear_solver = DirectSolver(assemble_jac=True)

        model = Group()
        model.add_subsystem('g', g, promotes=['*'])

        p = Problem(model)
        p.setup()

        # Conclude setup but don't run model.
        p.final_setup()

        # -------------------------------------------------------------------

        msg1 = 'Variable name "{}" not found.'
        msg2 = "The promoted name x is invalid because it refers to multiple inputs: " \
               "[g.c2.x ,g.c3.x]. Access the value from the connected output variable x instead."

        inputs, outputs, residuals = g.get_nonlinear_vectors()

        # inputs
        with self.assertRaises(Exception) as context:
            inputs['x'] = 5.0
        self.assertEqual(str(context.exception), msg2)
        with self.assertRaises(Exception) as context:
            self.assertEqual(inputs['x'], 5.0)
        self.assertEqual(str(context.exception), msg2)

        with assertRaisesRegex(self, KeyError, msg1.format('g.c2.x')):
            inputs['g.c2.x'] = 5.0
        with assertRaisesRegex(self, KeyError, msg1.format('g.c2.x')):
            self.assertEqual(inputs['g.c2.x'], 5.0)

        # outputs
        with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y')):
            outputs['g.c2.y'] = 5.0
        with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y')):
            self.assertEqual(outputs['g.c2.y'], 5.0)

        msg1 = r'Variable name pair \("{}", "{}"\) not found.'

        jac = g.linear_solver._assembled_jac

        # d(outputs)/d(inputs)
        with self.assertRaises(Exception) as context:
            jac['y', 'x'] = 5.0
        self.assertEqual(str(context.exception), msg2)

        with self.assertRaises(Exception) as context:
            self.assertEqual(jac['y', 'x'], 5.0)
        self.assertEqual(str(context.exception), msg2)

        # absolute keys now allowed
        # with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y', 'g.c2.x')):
        #     jac['g.c2.y', 'g.c2.x'] = 5.0
        # with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y', 'g.c2.x')):
        #     deriv = jac['g.c2.y', 'g.c2.x']

        # d(outputs)/d(outputs)
        # with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y', 'g.c2.y')):
        #     jac['g.c2.y', 'g.c2.y'] = 5.0
        # with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y', 'g.c2.y')):
        #     deriv = jac['g.c2.y', 'g.c2.y']

    def test_nested_promotion_errors(self):
        """
        Tests for error-handling for promoted input variable names.
        """
        c1 = IndepVarComp('x')
        c2 = ExecComp('y=2*x')
        c3 = ExecComp('z=3*x')

        g = Group(assembled_jac_type='dense')
        g.add_subsystem('c2', c2, promotes=['*'])
        g.add_subsystem('c3', c3, promotes=['*'])
        g.linear_solver = DirectSolver(assemble_jac=True)

        model = Group()
        model.add_subsystem('c1', c1, promotes=['*'])
        model.add_subsystem('g', g)

        p = Problem(model)
        p.setup()

        # -------------------------------------------------------------------

        msg1 = "The promoted name g.x is invalid because it refers to multiple inputs: " \
               "[g.c2.x, g.c3.x] that are not connected to an output variable."

        # inputs (g.x is not connected)
        # with assertRaisesRegex(self, RuntimeError, msg1.format('g.x')):
        with self.assertRaises(Exception) as context:
            p['g.x'] = 5.0
            p.final_setup()

        self.assertEqual(str(context.exception), msg1)

        # Repeat test for post final_setup when vectors are allocated.
        p = Problem(model)
        p.setup()
        p.final_setup()

        # -------------------------------------------------------------------

        # inputs (g.x is not connected)
        with self.assertRaises(Exception) as context:
            p['g.x'] = 5.0
            p.final_setup()
        self.assertEqual(str(context.exception), msg1)

        # Start from a clean state again
        p = Problem(model)
        p.setup()

        with self.assertRaises(Exception) as context:
            self.assertEqual(p['g.x'], 5.0)
        self.assertEqual(str(context.exception), msg1)

        msg2 = "The promoted name x is invalid because it refers to multiple inputs: " \
               "[g.c2.x, g.c3.x] that are not connected to an output variable."

        jac = g.linear_solver._assembled_jac
        # d(outputs)/d(inputs)
        with self.assertRaises(Exception) as context:
            jac['y', 'x'] = 5.0
        self.assertEqual(str(context.exception), msg2)

        with self.assertRaises(Exception) as context:
            self.assertEqual(jac['y', 'x'], 5.0)
        self.assertEqual(str(context.exception), msg2)

        # -------------------------------------------------------------------

        # Repeat test for post final_setup when vectors are allocated.
        p = Problem(model)
        p.setup()
        p.final_setup()

        with self.assertRaises(Exception) as context:
            self.assertEqual(p['g.x'], 5.0)
        self.assertEqual(str(context.exception), msg1)

        # d(outputs)/d(inputs)
        with self.assertRaises(Exception) as context:
            jac['y', 'x'] = 5.0
        self.assertEqual(str(context.exception), msg2)

        with self.assertRaises(Exception) as context:
            self.assertEqual(jac['y', 'x'], 5.0)
        self.assertEqual(str(context.exception), msg2)

        # -------------------------------------------------------------------

        msg1 = "The promoted name g.x is invalid because it refers to multiple inputs: " \
               "[g.c2.x ,g.c3.x]. Access the value from the connected output variable x instead."

        # From here, 'g.x' has a valid source.
        model.connect('x', 'g.x')

        p = Problem(model)
        p.setup()

        # inputs (g.x is connected to x)
        p['g.x'] = 5.0
        with self.assertRaises(Exception) as context:
            p.final_setup()
        self.assertEqual(str(context.exception), msg1)

        # Repeat test for post final_setup when vectors are allocated.
        p = Problem(model)
        p.setup()
        p.final_setup()

        # inputs (g.x is connected to x)
        with self.assertRaises(Exception) as context:
            p['g.x'] = 5.0
        self.assertEqual(str(context.exception), msg1)

        # Final test, the getitem
        p = Problem(model)
        p.setup()

        with self.assertRaises(Exception) as context:
            self.assertEqual(p['g.x'], 5.0)
        self.assertEqual(str(context.exception), msg1)

        # d(outputs)/d(inputs)
        with self.assertRaises(Exception) as context:
            jac['y', 'x'] = 5.0
        self.assertEqual(str(context.exception), msg2)

        with self.assertRaises(Exception) as context:
            self.assertEqual(jac['y', 'x'], 5.0)        # Start from a clean state again
        self.assertEqual(str(context.exception), msg2)

        # Repeat test for post final_setup when vectors are allocated.
        p = Problem(model)
        p.setup()
        p.final_setup()

        with self.assertRaises(Exception) as context:
            self.assertEqual(p['g.x'], 5.0)
        self.assertEqual(str(context.exception), msg1)

        # d(outputs)/d(inputs)
        with self.assertRaises(Exception) as context:
            jac['y', 'x'] = 5.0
        self.assertEqual(str(context.exception), msg2)

        with self.assertRaises(Exception) as context:
            self.assertEqual(jac['y', 'x'], 5.0)
        self.assertEqual(str(context.exception), msg2)


class TestGetSetMPI(unittest.TestCase):

    N_PROCS = 2

    def _get_problem(self):
        p = Problem()
        indeps = p.model.add_subsystem('indeps', IndepVarComp())
        indeps.add_output('x', np.ones(5))
        indeps.add_discrete_output('x_desc', 'foo')
        par = p.model.add_subsystem('par', ParallelGroup())

        return p

    def test_get_val(self):
        pass


if __name__ == '__main__':
    unittest.main()
