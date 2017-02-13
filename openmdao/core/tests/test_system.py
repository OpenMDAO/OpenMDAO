import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp

class TestSystem(unittest.TestCase):

    def test_get_set(self):
        g1 = Group()
        indep = g1.add_subsystem('Indep', IndepVarComp('a', 5.0), promotes=['a'])
        g2 = g1.add_subsystem('G2', Group(), promotes=['*'])
        c1_2 = g2.add_subsystem('C1', ExecComp('b=2*a'), promotes=['a', 'b'])

        model = Group()
        model.add_subsystem('G1', g1, promotes=['b'])
        model.add_subsystem('Sink', ExecComp('c=2*b'), promotes=['b'])

        p = Problem(model=model)
        model.suppress_solver_output = True

        # test pre-setup errors
        with self.assertRaises(Exception) as cm:
            model.get_input('Sink.b')
        self.assertEqual(str(cm.exception),
                         ": Cannot access input 'Sink.b'. Setup has not been called.")
        with self.assertRaises(Exception) as cm:
            model.get_output('Sink.c')
        self.assertEqual(str(cm.exception),
                         ": Cannot access output 'Sink.c'. Setup has not been called.")
        with self.assertRaises(Exception) as cm:
            model.get_residual('Sink.c')
        self.assertEqual(str(cm.exception),
                         ": Cannot access residual 'Sink.c'. Setup has not been called.")
        with self.assertRaises(Exception) as cm:
            model.set_input('Sink.b', 1.1)
        self.assertEqual(str(cm.exception),
                         ": Cannot access input 'Sink.b'. Setup has not been called.")
        with self.assertRaises(Exception) as cm:
            model.set_output('Sink.c', 2.2)
        self.assertEqual(str(cm.exception),
                         ": Cannot access output 'Sink.c'. Setup has not been called.")
        with self.assertRaises(Exception) as cm:
            model.set_residual('Sink.c', 3.3)
        self.assertEqual(str(cm.exception),
                         ": Cannot access residual 'Sink.c'. Setup has not been called.")

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

        # test bad varname errors
        with self.assertRaises(KeyError) as cm:
            model.get_input('Sink.bb')
        self.assertEqual(str(cm.exception), '": input \'Sink.bb\' not found."')
        with self.assertRaises(KeyError) as cm:
            model.get_output('Sink.cc')
        self.assertEqual(str(cm.exception), '": output \'Sink.cc\' not found."')
        with self.assertRaises(KeyError) as cm:
            model.get_residual('Sink.cc')
        self.assertEqual(str(cm.exception), '": residual \'Sink.cc\' not found."')

        with self.assertRaises(KeyError) as cm:
            model.set_input('Sink.bb', 2.)
        self.assertEqual(str(cm.exception), '": input \'Sink.bb\' not found."')
        with self.assertRaises(KeyError) as cm:
            model.set_output('Sink.cc', 2.)
        self.assertEqual(str(cm.exception), '": output \'Sink.cc\' not found."')
        with self.assertRaises(KeyError) as cm:
            model.set_residual('Sink.cc', 2.)
        self.assertEqual(str(cm.exception), '": residual \'Sink.cc\' not found."')

if __name__ == "__main__":
    unittest.main()
