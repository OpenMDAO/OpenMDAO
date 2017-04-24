"""Test getting/setting variables and subjacs with promoted/relative/absolute names."""

import unittest
from six import assertRaisesRegex

from openmdao.api import Problem, Group, ExecComp, IndepVarComp
from openmdao.devtools.testutil import assert_rel_error


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

        p = Problem(model=model)
        p.setup(check=False)

        # -------------------------------------------------------------------

        # inputs
        p['g.c.x'] = 5.0
        self.assertEqual(p['g.c.x'], 5.0)

        # outputs
        p['g.c.y'] = 5.0
        self.assertEqual(p['g.c.y'], 5.0)

        inputs, outputs, residuals = g.get_nonlinear_vectors()

        # inputs
        inputs['c.x'] = 5.0
        self.assertEqual(inputs['c.x'], 5.0)

        # outputs
        outputs['c.y'] = 5.0
        self.assertEqual(outputs['c.y'], 5.0)

        with g.jacobian_context() as jac:

            # d(output)/d(input)
            jac['c.y', 'c.x'] = 5.0
            self.assertEqual(jac['c.y', 'c.x'], 5.0)

            # d(output)/d(output)
            jac['c.y', 'c.y'] = 5.0
            self.assertEqual(jac['c.y', 'c.y'], 5.0)

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

        p = Problem(model=model)
        p.setup(check=False)

        # -------------------------------------------------------------------

        # inputs
        p['g.c2.x'] = 5.0
        self.assertEqual(p['g.c2.x'], 5.0)

        # outputs
        p['g.c2.y'] = 5.0
        self.assertEqual(p['g.c2.y'], 5.0)
        p['y'] = 5.0
        self.assertEqual(p['y'], 5.0)

        inputs, outputs, residuals = g.get_nonlinear_vectors()

        # inputs
        inputs['c2.x'] = 5.0
        self.assertEqual(inputs['c2.x'], 5.0)

        # outputs
        outputs['c2.y'] = 5.0
        self.assertEqual(outputs['c2.y'], 5.0)
        outputs['y'] = 5.0
        self.assertEqual(outputs['y'], 5.0)

        with g.jacobian_context() as jac:

            # d(outputs)/d(inputs)
            jac['c2.y', 'c2.x'] = 5.0
            self.assertEqual(jac['c2.y', 'c2.x'], 5.0)

            # d(outputs)/d(outputs)
            jac['c2.y', 'c2.y'] = 5.0
            self.assertEqual(jac['c2.y', 'c2.y'], 5.0)
            jac['y', 'y'] = 5.0
            self.assertEqual(jac['y', 'y'], 5.0)

    def test_no_promotion_errors(self):
        """
        Tests for error-handling for invalid variable names and keys.
        """
        c = ExecComp('y=2*x')

        g = Group()
        g.add_subsystem('c', c)

        root = Group()
        root.add_subsystem('g', g)

        p = Problem(model=root)
        p.setup(check=False)

        # -------------------------------------------------------------------

        msg = 'Variable name "{}" not found.'

        # inputs
        with assertRaisesRegex(self, KeyError, msg.format('x')):
            p['x'] = 5.0
        with assertRaisesRegex(self, KeyError, msg.format('x')):
            self.assertEqual(p['x'], 5.0)

        # outputs
        with assertRaisesRegex(self, KeyError, msg.format('y')):
            p['y'] = 5.0
        with assertRaisesRegex(self, KeyError, msg.format('y')):
            self.assertEqual(p['y'], 5.0)

        msg = 'Variable name "{}" not found.'
        inputs, outputs, residuals = g.get_nonlinear_vectors()

        # inputs
        with assertRaisesRegex(self, KeyError, msg.format('x')):
            inputs['x'] = 5.0
        with assertRaisesRegex(self, KeyError, msg.format('x')):
            self.assertEqual(inputs['x'], 5.0)
        with assertRaisesRegex(self, KeyError, msg.format('g.c.x')):
            inputs['g.c.x'] = 5.0
        with assertRaisesRegex(self, KeyError, msg.format('g.c.x')):
            self.assertEqual(inputs['g.c.x'], 5.0)

        # outputs
        with assertRaisesRegex(self, KeyError, msg.format('y')):
            outputs['y'] = 5.0
        with assertRaisesRegex(self, KeyError, msg.format('y')):
            self.assertEqual(outputs['y'], 5.0)
        with assertRaisesRegex(self, KeyError, msg.format('g.c.y')):
            outputs['g.c.y'] = 5.0
        with assertRaisesRegex(self, KeyError, msg.format('g.c.y')):
            self.assertEqual(outputs['g.c.y'], 5.0)

        msg = 'Variable name pair \("{}", "{}"\) not found.'
        with g.jacobian_context() as jac:

            # d(output)/d(input)
            with assertRaisesRegex(self, KeyError, msg.format('y', 'x')):
                jac['y', 'x'] = 5.0
            with assertRaisesRegex(self, KeyError, msg.format('y', 'x')):
                self.assertEqual(jac['y', 'x'], 5.0)
            with assertRaisesRegex(self, KeyError, msg.format('g.c.y', 'g.c.x')):
                jac['g.c.y', 'g.c.x'] = 5.0
            with assertRaisesRegex(self, KeyError, msg.format('g.c.y', 'g.c.x')):
                self.assertEqual(jac['g.c.y', 'g.c.x'], 5.0)

            # d(output)/d(output)
            with assertRaisesRegex(self, KeyError, msg.format('y', 'y')):
                jac['y', 'y'] = 5.0
            with assertRaisesRegex(self, KeyError, msg.format('y', 'y')):
                self.assertEqual(jac['y', 'y'], 5.0)
            with assertRaisesRegex(self, KeyError, msg.format('g.c.y', 'g.c.y')):
                jac['g.c.y', 'g.c.y'] = 5.0
            with assertRaisesRegex(self, KeyError, msg.format('g.c.y', 'g.c.y')):
                self.assertEqual(jac['g.c.y', 'g.c.y'], 5.0)

    def test_with_promotion_errors(self):
        """
        Tests for error-handling for invalid variable names and keys.
        """
        c1 = IndepVarComp('x')
        c2 = ExecComp('y=2*x')
        c3 = ExecComp('z=3*x')

        g = Group()
        g.add_subsystem('c1', c1, promotes=['*'])
        g.add_subsystem('c2', c2, promotes=['*'])
        g.add_subsystem('c3', c3, promotes=['*'])

        root = Group()
        root.add_subsystem('g', g, promotes=['*'])

        p = Problem(model=root)
        p.setup(check=False)

        # -------------------------------------------------------------------

        msg1 = 'Variable name "{}" not found.'
        msg2 = 'The promoted name "{}" is invalid because it is non-unique.'
        inputs, outputs, residuals = g.get_nonlinear_vectors()

        # inputs
        with assertRaisesRegex(self, KeyError, msg2.format('x')):
            inputs['x'] = 5.0
        with assertRaisesRegex(self, KeyError, msg2.format('x')):
            self.assertEqual(inputs['x'], 5.0)
        with assertRaisesRegex(self, KeyError, msg1.format('g.c2.x')):
            inputs['g.c2.x'] = 5.0
        with assertRaisesRegex(self, KeyError, msg1.format('g.c2.x')):
            self.assertEqual(inputs['g.c2.x'], 5.0)

        # outputs
        with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y')):
            outputs['g.c2.y'] = 5.0
        with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y')):
            self.assertEqual(outputs['g.c2.y'], 5.0)

        msg1 = 'Variable name pair \("{}", "{}"\) not found.'
        msg2 = 'The promoted name "{}" is invalid because it is non-unique.'
        with g.jacobian_context() as jac:

            # d(outputs)/d(inputs)
            with assertRaisesRegex(self, KeyError, msg2.format('x')):
                jac['y', 'x'] = 5.0
            with assertRaisesRegex(self, KeyError, msg2.format('x')):
                self.assertEqual(jac['y', 'x'], 5.0)
            with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y', 'g.c2.x')):
                jac['g.c2.y', 'g.c2.x'] = 5.0
            with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y', 'g.c2.x')):
                self.assertEqual(jac['g.c2.y', 'g.c2.x'], 5.0)

            # d(outputs)/d(outputs)
            with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y', 'g.c2.y')):
                jac['g.c2.y', 'g.c2.y'] = 5.0
            with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y', 'g.c2.y')):
                self.assertEqual(jac['g.c2.y', 'g.c2.y'], 5.0)


if __name__ == '__main__':
    unittest.main()
