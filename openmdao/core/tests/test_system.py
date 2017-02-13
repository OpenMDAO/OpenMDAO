""" Unit tests for the system interface."""

import unittest
from six import assertRaisesRegex

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp
from openmdao.devtools.testutil import assert_rel_error


class TestSystem(unittest.TestCase):

    def test_get_set(self):
        g1 = Group()
        g1.add_subsystem('Indep', IndepVarComp('a', 5.0), promotes=['a'])
        g2 = g1.add_subsystem('G2', Group(), promotes=['*'])
        g2.add_subsystem('C1', ExecComp('b=2*a'), promotes=['a', 'b'])

        model = Group()
        model.add_subsystem('G1', g1, promotes=['b'])
        model.add_subsystem('Sink', ExecComp('c=2*b'), promotes=['b'])

        p = Problem(model=model)
        model.suppress_solver_output = True

        # test pre-setup errors
        with self.assertRaises(Exception) as cm:
            model.get_input('Sink.b')
        self.assertEqual(str(cm.exception),
                         "Cannot access input 'Sink.b'. Setup has not been called.")
        with self.assertRaises(Exception) as cm:
            model.get_output('Sink.c')
        self.assertEqual(str(cm.exception),
                         "Cannot access output 'Sink.c'. Setup has not been called.")
        with self.assertRaises(Exception) as cm:
            model.get_residual('Sink.c')
        self.assertEqual(str(cm.exception),
                         "Cannot access residual 'Sink.c'. Setup has not been called.")

        p.setup()
        p.run_model()

        self.assertEqual(model.get_input('G1.G2.C1.a'), 5.0)
        self.assertEqual(g1.get_input('G2.C1.a'), 5.0)

        g2.set_input('C1.a', -1.)
        self.assertEqual(model.get_input('G1.G2.C1.a'), -1.)
        self.assertEqual(g1.get_input('G2.C1.a'), -1.)

        self.assertEqual(model.get_output('G1.G2.C1.b'), 10.0)
        self.assertEqual(model.get_output('b'), 10.0)
        self.assertEqual(g2.get_output('C1.b'), 10.0)
        self.assertEqual(g2.get_output('b'), 10.0)

        model.set_output('b', 123.)
        self.assertEqual(model.get_output('G1.G2.C1.b'), 123.0)
        self.assertEqual(model._outputs['G1.G2.C1.b'], 123.0)
        self.assertEqual(model.get_output('b'), 123.0)

        model.set_output('G1.G2.C1.b', 456.)
        self.assertEqual(model.get_output('G1.G2.C1.b'), 456.)
        self.assertEqual(model._outputs['G1.G2.C1.b'], 456.)
        self.assertEqual(model.get_output('b'), 456.)

        g2.set_output('b', 789.)
        self.assertEqual(model.get_output('G1.G2.C1.b'), 789.)
        self.assertEqual(model.get_output('b'), 789.)
        self.assertEqual(g2.get_output('C1.b'), 789.)
        self.assertEqual(g2.get_output('b'), 789.)

        model.set_residual('b', 99.0)
        self.assertEqual(model.get_residual('b'), 99.0)
        self.assertEqual(model.get_residual('G1.G2.C1.b'), 99.0)
        self.assertEqual(model._residuals['G1.G2.C1.b'], 99.0)

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

        prob = Problem(model=model)
        prob.setup()

        msg = "Incompatible shape for assignment. Expected .* but got .*"

        num_val = -10
        arr_val = -10*np.ones((5, 1))
        bad_val = -10*np.ones((10))

        #
        # set input
        #

        # assign array to scalar
        with assertRaisesRegex(self, ValueError, msg):
            g2.set_input('C1.a', arr_val)

        # assign scalar to array
        g2.set_input('C2.x', num_val)
        assert_rel_error(self, model.get_input('G1.G2.C2.x'), arr_val, 1e-10)
        assert_rel_error(self, g1.get_input('G2.C2.x'), arr_val, 1e-10)

        # assign array to array
        g2.set_input('C2.x', arr_val)
        assert_rel_error(self, model.get_input('G1.G2.C2.x'), arr_val, 1e-10)
        assert_rel_error(self, g1.get_input('G2.C2.x'), arr_val, 1e-10)

        # assign bad array shape to array
        with assertRaisesRegex(self, ValueError, msg):
            g2.set_input('C2.x', bad_val)

        # assign list to array
        g2.set_input('C2.x', arr_val.tolist())
        assert_rel_error(self, model.get_input('G1.G2.C2.x'), arr_val, 1e-10)
        assert_rel_error(self, g1.get_input('G2.C2.x'), arr_val, 1e-10)

        # assign bad list shape to array
        with assertRaisesRegex(self, ValueError, msg):
            g2.set_input('C2.x', bad_val.tolist())

        #
        # set_output
        #

        # assign array to scalar
        with assertRaisesRegex(self, ValueError, msg):
            g2.set_output('C1.a', arr_val)

        # assign scalar to array
        g2.set_output('C2.y', num_val)
        assert_rel_error(self, model.get_output('G1.G2.C2.y'), arr_val, 1e-10)
        assert_rel_error(self, g1.get_output('G2.C2.y'), arr_val, 1e-10)

        # assign array to array
        g2.set_output('C2.y', arr_val)
        assert_rel_error(self, model.get_output('G1.G2.C2.y'), arr_val, 1e-10)
        assert_rel_error(self, g1.get_output('G2.C2.y'), arr_val, 1e-10)

        # assign bad array shape to array
        with assertRaisesRegex(self, ValueError, msg):
            g2.set_output('C2.y', bad_val)

        # assign list to array
        g2.set_output('C2.y', arr_val.tolist())
        assert_rel_error(self, model.get_output('G1.G2.C2.y'), arr_val, 1e-10)
        assert_rel_error(self, g1.get_output('G2.C2.y'), arr_val, 1e-10)

        # assign bad list shape to array
        with assertRaisesRegex(self, ValueError, msg):
            g2.set_output('C2.y', bad_val.tolist())

        #
        # set_residual
        #

        # assign array to scalar
        with assertRaisesRegex(self, ValueError, msg):
            model.set_residual('b', arr_val)

        # assign scalar to array
        g2.set_residual('y', num_val)
        assert_rel_error(self, model.get_input('G1.G2.C2.y'), arr_val, 1e-10)
        assert_rel_error(self, g1.get_input('G2.C2.y'), arr_val, 1e-10)

        # assign array to array
        g2.set_residual('y', arr_val)
        assert_rel_error(self, model.get_input('G1.G2.C2.y'), arr_val, 1e-10)
        assert_rel_error(self, g1.get_input('G2.C2.y'), arr_val, 1e-10)

        # assign bad array shape to array
        with assertRaisesRegex(self, ValueError, msg):
            g2.set_residual('y', bad_val)

        # assign list to array
        g2.set_residual('y', arr_val.tolist())
        assert_rel_error(self, model.get_input('G1.G2.C2.y'), arr_val, 1e-10)
        assert_rel_error(self, g1.get_input('G2.C2.y'), arr_val, 1e-10)

        # assign bad list shape to array
        with assertRaisesRegex(self, ValueError, msg):
            g2.set_residual('y', bad_val.tolist())


if __name__ == "__main__":
    unittest.main()
