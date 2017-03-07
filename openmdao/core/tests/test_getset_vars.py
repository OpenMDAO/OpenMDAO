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
        p['g.c.x'] = 1.0
        print(p['g.c.x'])

        # outputs
        p['g.c.y'] = 1.0
        print(p['g.c.y'])

        with g.nonlinear_vector_context() as (inputs, outputs, residuals):

            # inputs
            inputs['c.x'] = 1.0
            print(inputs['c.x'])

            # outputs
            outputs['c.y'] = 1.0
            print(outputs['c.y'])

        with g.jacobian_context() as jac:

            # d(output)/d(input)
            jac['c.y', 'c.x'] = 1.0
            print(jac['c.y', 'c.x'])

            # d(output)/d(output)
            jac['c.y', 'c.y'] = 1.0
            print(jac['c.y', 'c.y'])

    def test_with_promotion(self):
        """
        Illustrative examples showing how to access variables and subjacs.
        """
        c1 = IndepVarComp('x')
        c2 = ExecComp('y=2*x')
        c3 = ExecComp('z=3*x')

        g = Group()
        g.add_subsystem('c1', c1, promotes='*')
        g.add_subsystem('c2', c2, promotes='*')
        g.add_subsystem('c3', c3, promotes='*')

        model = Group()
        model.add_subsystem('g', g, promotes='*')

        p = Problem(model=model)
        p.setup(check=False)

        # -------------------------------------------------------------------

        # inputs
        p['g.c2.x'] = 1.0
        print(p['g.c2.x'])

        # outputs
        p['g.c2.y'] = 1.0
        print(p['g.c2.y'])
        p['y'] = 1.0
        print(p['y'])

        with g.nonlinear_vector_context() as (inputs, outputs, residuals):

            # inputs
            inputs['c2.x'] = 1.0
            print(inputs['c2.x'])

            # outputs
            outputs['c2.y'] = 1.0
            print(outputs['c2.y'])
            outputs['y'] = 1.0
            print(outputs['y'])

        with g.jacobian_context() as jac:

            # d(outputs)/d(inputs)
            jac['c2.y', 'c2.x'] = 1.0
            print(jac['c2.y', 'c2.x'])

            # d(outputs)/d(outputs)
            jac['c2.y', 'c2.y'] = 1.0
            print(jac['c2.y', 'c2.y'])
            jac['y', 'y'] = 1.0
            print(jac['y', 'y'])

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
            p['x'] = 1.0
        with assertRaisesRegex(self, KeyError, msg.format('x')):
            print(p['x'])

        # outputs
        with assertRaisesRegex(self, KeyError, msg.format('y')):
            p['y'] = 1.0
        with assertRaisesRegex(self, KeyError, msg.format('y')):
            print(p['y'])

        msg = 'Variable name "{}" not found.'
        with g.nonlinear_vector_context() as (inputs, outputs, residuals):

            # inputs
            with assertRaisesRegex(self, KeyError, msg.format('x')):
                inputs['x'] = 1.0
            with assertRaisesRegex(self, KeyError, msg.format('x')):
                print(inputs['x'])
            with assertRaisesRegex(self, KeyError, msg.format('g.c.x')):
                inputs['g.c.x'] = 1.0
            with assertRaisesRegex(self, KeyError, msg.format('g.c.x')):
                print(inputs['g.c.x'])

            # outputs
            with assertRaisesRegex(self, KeyError, msg.format('y')):
                outputs['y'] = 1.0
            with assertRaisesRegex(self, KeyError, msg.format('y')):
                print(outputs['y'])
            with assertRaisesRegex(self, KeyError, msg.format('g.c.y')):
                outputs['g.c.y'] = 1.0
            with assertRaisesRegex(self, KeyError, msg.format('g.c.y')):
                print(outputs['g.c.y'])

        msg = 'Variable name pair \("{}", "{}"\) not found.'
        with g.jacobian_context() as jac:

            # d(output)/d(input)
            with assertRaisesRegex(self, KeyError, msg.format('y', 'x')):
                jac['y', 'x'] = 1.0
            with assertRaisesRegex(self, KeyError, msg.format('y', 'x')):
                print(jac['y', 'x'])
            with assertRaisesRegex(self, KeyError, msg.format('g.c.y', 'g.c.x')):
                jac['g.c.y', 'g.c.x'] = 1.0
            with assertRaisesRegex(self, KeyError, msg.format('g.c.y', 'g.c.x')):
                print(jac['g.c.y', 'g.c.x'])

            # d(output)/d(output)
            with assertRaisesRegex(self, KeyError, msg.format('y', 'y')):
                jac['y', 'y'] = 1.0
            with assertRaisesRegex(self, KeyError, msg.format('y', 'y')):
                print(jac['y', 'y'])
            with assertRaisesRegex(self, KeyError, msg.format('g.c.y', 'g.c.y')):
                jac['g.c.y', 'g.c.y'] = 1.0
            with assertRaisesRegex(self, KeyError, msg.format('g.c.y', 'g.c.y')):
                print(jac['g.c.y', 'g.c.y'])

    def test_with_promotion_errors(self):
        """
        Tests for error-handling for invalid variable names and keys.
        """
        c1 = IndepVarComp('x')
        c2 = ExecComp('y=2*x')
        c3 = ExecComp('z=3*x')

        g = Group()
        g.add_subsystem('c1', c1, promotes='*')
        g.add_subsystem('c2', c2, promotes='*')
        g.add_subsystem('c3', c3, promotes='*')

        root = Group()
        root.add_subsystem('g', g, promotes='*')

        p = Problem(model=root)
        p.setup(check=False)

        # -------------------------------------------------------------------

        msg1 = 'Variable name "{}" not found.'
        msg2 = 'The promoted name "{}" is invalid because it is non-unique.'
        with g.nonlinear_vector_context() as (inputs, outputs, residuals):

            # inputs
            with assertRaisesRegex(self, KeyError, msg2.format('x')):
                inputs['x'] = 1.0
            with assertRaisesRegex(self, KeyError, msg2.format('x')):
                print(inputs['x'])
            with assertRaisesRegex(self, KeyError, msg1.format('g.c2.x')):
                inputs['g.c2.x'] = 1.0
            with assertRaisesRegex(self, KeyError, msg1.format('g.c2.x')):
                print(inputs['g.c2.x'])

            # outputs
            with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y')):
                outputs['g.c2.y'] = 1.0
            with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y')):
                print(outputs['g.c2.y'])

        msg1 = 'Variable name pair \("{}", "{}"\) not found.'
        msg2 = 'The promoted name "{}" is invalid because it is non-unique.'
        with g.jacobian_context() as jac:

            # d(outputs)/d(inputs)
            with assertRaisesRegex(self, KeyError, msg2.format('x')):
                jac['y', 'x'] = 1.0
            with assertRaisesRegex(self, KeyError, msg2.format('x')):
                print(jac['y', 'x'])
            with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y', 'g.c2.x')):
                jac['g.c2.y', 'g.c2.x'] = 1.0
            with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y', 'g.c2.x')):
                print(jac['g.c2.y', 'g.c2.x'])

            # d(outputs)/d(outputs)
            with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y', 'g.c2.y')):
                jac['g.c2.y', 'g.c2.y'] = 1.0
            with assertRaisesRegex(self, KeyError, msg1.format('g.c2.y', 'g.c2.y')):
                print(jac['g.c2.y', 'g.c2.y'])


if __name__ == '__main__':
    unittest.main()
