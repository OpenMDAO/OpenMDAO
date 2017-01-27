import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp


class SimpleGroup(Group):

    def initialize(self):
        self.add_subsystem('comp1', IndepVarComp('x', 5.0))
        self.add_subsystem('comp2', ExecComp('b=2*a'))
        self.connect('comp1.x', 'comp2.a')


class TestGroup(unittest.TestCase):

    def test_same_sys_name(self):
        """Test error checking for the case where we add two subsystems with the same name."""
        p = Problem(model=Group())
        p.model.add_subsystem('comp1', IndepVarComp('x', 5.0))
        p.model.add_subsystem('comp2', ExecComp('b=2*a'))

        try:
            p.model.add_subsystem('comp2', ExecComp('b=2*a'))
        except Exception as err:
            self.assertEqual(str(err), "Subsystem name 'comp2' is already used.")
        else:
            self.fail('Exception expected.')

    def test_group_simple(self):
        """Simple example for adding subsystems to a group and issuing connections."""
        p = Problem(model=Group())
        p.model.add_subsystem('comp1', IndepVarComp('x', 5.0))
        p.model.add_subsystem('comp2', ExecComp('b=2*a'))
        p.model.connect('comp1.x', 'comp2.a')
        p.setup()

        c1 = p.model.get_subsystem('comp1')
        c2 = p.model.get_subsystem('comp2')
        cx = p.model.get_subsystem('comp')
        self.assertEqual(c1.name, 'comp1')
        self.assertEqual(c2.name, 'comp2')
        self.assertEqual(cx, None)

        p.run_model()
        self.assertEqual(p['comp2.b'], 10.0)

    def test_group_inmethod(self):
        """Example for adding subsystems and connections in the Group implementation."""
        p = Problem(model=SimpleGroup())
        p.setup()

        c1 = p.model.get_subsystem('comp1')
        c2 = p.model.get_subsystem('comp2')
        self.assertEqual(c1.name, 'comp1')
        self.assertEqual(c2.name, 'comp2')

        p.run_model()
        self.assertEqual(p['comp2.b'], 10.0)

    def test_group_promotes(self):
        """Promoting a single variable."""
        p = Problem(model=Group())
        p.model.add_subsystem('comp1', IndepVarComp([
                ('a', 2.0),
                ('x', 5.0),
            ]),
            promotes_outputs='x')
        p.model.add_subsystem('comp2', ExecComp('y=2*x'), promotes_inputs='x')
        p.setup()
        p.run_model()

        self.assertEqual(p['comp1.a'], 2)
        self.assertEqual(p['x'], 5)
        self.assertEqual(p['comp2.y'], 10)

    def test_group_promotes_multiple(self):
        """Promoting multiple variables."""
        p = Problem(model=Group())
        p.model.add_subsystem('comp1', IndepVarComp([
                ('a', 2.0),
                ('x', 5.0),
            ]),
            promotes_outputs=['a', 'x'])
        p.model.add_subsystem('comp2', ExecComp('y=2*x'), promotes_inputs='x')
        p.setup()
        p.run_model()

        self.assertEqual(p['a'], 2)
        self.assertEqual(p['x'], 5)
        self.assertEqual(p['comp2.y'], 10)

    def test_group_promotes_all(self):
        """Promoting all variables with asterisk."""
        p = Problem(model=Group())
        p.model.add_subsystem('comp1', IndepVarComp([
                ('a', 2.0),
                ('x', 5.0),
            ]),
            promotes_outputs='*')
        p.model.add_subsystem('comp2', ExecComp('y=2*x'), promotes_inputs=['x'])
        p.setup()
        p.run_model()

        self.assertEqual(p['a'], 2)
        self.assertEqual(p['x'], 5)
        self.assertEqual(p['comp2.y'], 10)

    def test_group_renames(self):
        """Renaming variables and using implicit connections."""
        raise unittest.SkipTest("The add_subsystem has not yet been updated for renames")
        p = Problem(model=Group())
        p.model.add_subsystem('comp1', IndepVarComp([
                ('a', 2.0),
                ('x', 5.0),
            ]),
            promotes_outputs=['x', ('a', 'q')])
        p.model.add_subsystem('comp2', ExecComp('y=2*x'), promotes_inputs=['x'])
        p.setup()
        p.run_model()

        self.assertEqual(p['q'], 2)
        self.assertEqual(p['x'], 5)
        self.assertEqual(p['comp2.y'], 10)

    def test_group_renames_and_connect(self):
        """Renaming variables and issuing explicit connections."""
        raise unittest.SkipTest("The add_subsystem has not yet been updated for renames")
        p = Problem(model=Group())
        p.model.add_subsystem('comp1', IndepVarComp([
                ('a', 2.0),
                ('x', 5.0),
            ]),
            promotes_outputs=[('x', 'x2'), ('a', 'q')])
        p.model.add_subsystem('comp2', ExecComp('y=2*x'))
        p.model.connect('x2', 'comp2.x')
        p.setup()
        p.run_model()

        self.assertEqual(p['q'], 2)
        self.assertEqual(p['x2'], 5)
        self.assertEqual(p['comp2.x'], 5)
        self.assertEqual(p['comp2.y'], 10)

    def test_group_nested(self):
        """Example of adding subsystems and issuing connections with nested groups."""
        g1 = Group()
        c1_1 = g1.add_subsystem('comp1', IndepVarComp('x', 5.0))
        c1_2 = g1.add_subsystem('comp2', ExecComp('b=2*a'))
        g1.connect('comp1.x', 'comp2.a')

        g2 = Group()
        c2_1 = g2.add_subsystem('comp1', ExecComp('b=2*a'))
        c2_2 = g2.add_subsystem('comp2', ExecComp('b=2*a'))
        g2.connect('comp1.b', 'comp2.a')

        model = Group()
        model.add_subsystem('group1', g1)
        model.add_subsystem('group2', g2)
        model.connect('group1.comp2.b', 'group2.comp1.a')

        p = Problem(model=model)
        p.setup()

        c1_1 = p.model.get_subsystem('group1.comp1')
        c1_2 = p.model.get_subsystem('group1.comp2')
        c2_1 = p.model.get_subsystem('group2.comp1')
        c2_2 = p.model.get_subsystem('group2.comp2')
        self.assertEqual(c1_1.name, 'comp1')
        self.assertEqual(c1_2.name, 'comp2')
        self.assertEqual(c2_1.name, 'comp1')
        self.assertEqual(c2_2.name, 'comp2')

        c1_1 = p.model.get_subsystem('group1').get_subsystem('comp1')
        c1_2 = p.model.get_subsystem('group1').get_subsystem('comp2')
        c2_1 = p.model.get_subsystem('group2').get_subsystem('comp1')
        c2_2 = p.model.get_subsystem('group2').get_subsystem('comp2')
        self.assertEqual(c1_1.name, 'comp1')
        self.assertEqual(c1_2.name, 'comp2')
        self.assertEqual(c2_1.name, 'comp1')
        self.assertEqual(c2_2.name, 'comp2')

        p.run_model()

        self.assertEqual(p['group1.comp1.x'],  5.0)
        self.assertEqual(p['group1.comp2.b'], 10.0)
        self.assertEqual(p['group2.comp1.b'], 20.0)
        self.assertEqual(p['group2.comp2.b'], 40.0)


class TestConnect(unittest.TestCase):

    def setUp(self):
        prob = Problem(Group())

        sub = prob.model.add_subsystem('sub', Group())
        sub.add_subsystem('src', IndepVarComp('x', np.zeros(5,)))
        sub.add_subsystem('tgt', ExecComp('y = x'))
        sub.add_subsystem('cmp', ExecComp('z = x'))

        self.sub = sub
        self.prob = prob

    def test_src_indices_as_int_list(self):
        self.sub.connect('src.x', 'tgt.x', src_indices=[1])

    def test_src_indices_as_int_array(self):
        self.sub.connect('src.x', 'tgt.x', src_indices=np.zeros(1, dtype=int))

    def test_src_indices_as_float_list(self):
        msg = "src_indices must contain integers, but src_indices for " + \
              "connection from 'src.x' to 'tgt.x' contains non-integers."

        with self.assertRaisesRegexp(TypeError, msg):
            self.sub.connect('src.x', 'tgt.x', src_indices=[1.0])

    def test_src_indices_as_float_array(self):
        msg = "src_indices must contain integers, but src_indices for " + \
              "connection from 'src.x' to 'tgt.x' is <class 'numpy.float64'>."

        with self.assertRaisesRegexp(TypeError, msg):
            self.sub.connect('src.x', 'tgt.x', src_indices=np.zeros(1))

    def test_src_indices_as_str(self):
        msg = "src_indices must be an index array, " + \
              "did you mean connect('src.x', [tgt.x, cmp.x])?"

        with self.assertRaisesRegexp(TypeError, msg):
            self.sub.connect('src.x', 'tgt.x', 'cmp.x')

    def test_already_connected(self):
        msg = "Input 'tgt.x' is already connected to 'src.x'."

        self.sub.connect('src.x', 'tgt.x', src_indices=[1])
        with self.assertRaisesRegexp(RuntimeError, msg):
            self.sub.connect('cmp.x', 'tgt.x', src_indices=[1])

    def test_invalid_source(self):
        msg = "Output 'src.z' does not exist for connection " + \
              "in 'sub' from 'src.z' to 'tgt.x'."

        # source and target names can't be checked until setup
        # because initialize_variables is not called until then
        self.sub.connect('src.z', 'tgt.x', src_indices=[1])
        with self.assertRaisesRegexp(NameError, msg):
            self.prob.setup(check=False)

    def test_invalid_target(self):
        msg = "Input 'tgt.z' does not exist for connection " + \
              "in 'sub' from 'src.x' to 'tgt.z'."

        # source and target names can't be checked until setup
        # because initialize_variables is not called until then
        self.sub.connect('src.x', 'tgt.z', src_indices=[1])
        with self.assertRaisesRegexp(NameError, msg):
            self.prob.setup(check=False)

    def test_connect_within_system(self):
        msg = "Input and output are in the same System for connection " + \
              "from 'tgt.y' to 'tgt.x'."

        with self.assertRaisesRegexp(RuntimeError, msg):
            self.sub.connect('tgt.y', 'tgt.x', src_indices=[1])

    def test_connect_within_system_with_promotes(self):
        prob = Problem(Group())

        sub = prob.model.add_subsystem('sub', Group())
        sub.add_subsystem('tgt', ExecComp('y = x'), promotes_outputs=['y'])
        sub.connect('y', 'tgt.x', src_indices=[1])

        msg = "Input and output are in the same System for connection " + \
              "in 'sub' from 'y' to 'tgt.x'."

        with self.assertRaisesRegexp(RuntimeError, msg):
            prob.setup(check=False)

    def test_connect_within_system_with_renames(self):
        prob = Problem(Group())

        sub = prob.model.add_subsystem('sub', Group())
        sub.add_subsystem('tgt', ExecComp('y = x'), renames_outputs={'y': 'y2'})
        sub.connect('y2', 'tgt.x', src_indices=[1])

        msg = "Input and output are in the same System for connection " + \
              "in 'sub' from 'y2' to 'tgt.x'."

        with self.assertRaisesRegexp(RuntimeError, msg):
            prob.setup(check=False)


if __name__ == "__main__":
    unittest.main()
