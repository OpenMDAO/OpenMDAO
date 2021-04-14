import unittest


class TestWarnings(unittest.TestCase):

    def setUp(self):
        """
        Ensure that OpenMDAO warnings are using their default filter action.
        """
        import openmdao.api as om
        om.reset_warnings()

    def test_doc_with_units_warning(self):
        """
        Test nominal UnitsWarning.
        """
        import openmdao.api as om
        import warnings

        class AComp(om.ExplicitComponent):

            def initialize(self):
                pass

            def setup(self):
                self.add_input('a', shape=(10,), units='m')
                self.add_input('x', shape=(10,), units='1/s')
                self.add_input('b', shape=(10,), units='m/s')

                self.add_output('y', shape=(10,), units='m')
                self.add_output('z', shape=(10,), units='m/s')

                self.declare_coloring(wrt='*', form='cs')

            def compute(self, inputs, outputs):
                outputs['y'] = inputs['a'] * inputs['x'] + inputs['b']
                outputs['z'] = inputs['b'] * inputs['x']

        p = om.Problem()

        p.model.add_subsystem('a_comp', AComp())
        p.model.add_subsystem('exec_comp',
                              om.ExecComp('foo = y + z',
                                          y={'shape': (10,)},
                                          z={'shape': (10,)},
                                          foo={'shape': (10,)}))

        p.model.connect('a_comp.y', 'exec_comp.y')
        p.model.connect('a_comp.z', 'exec_comp.z')
        p.driver.declare_coloring()

        p.setup()

        with warnings.catch_warnings(record=True) as w:
            p.setup()
            unit_warnings = [wm for wm in w if wm.category is om.UnitsWarning]
            assert (len(unit_warnings) == 2)

    def test_doc_ignore_units_warning(self):
        """
        Test the ability to ignore UnitsWarning
        """
        import warnings
        import openmdao.api as om

        class AComp(om.ExplicitComponent):

            def initialize(self):
                pass

            def setup(self):
                self.add_input('a', shape=(10,), units='m')
                self.add_input('x', shape=(10,), units='1/s')
                self.add_input('b', shape=(10,), units='m/s')

                self.add_output('y', shape=(10,), units='m')
                self.add_output('z', shape=(10,), units='m/s')

                self.declare_coloring(wrt='*', form='cs')

            def compute(self, inputs, outputs):
                outputs['y'] = inputs['a'] * inputs['x'] + inputs['b']
                outputs['z'] = inputs['b'] * inputs['x']

        p = om.Problem()

        p.model.add_subsystem('a_comp', AComp())
        p.model.add_subsystem('exec_comp',
                              om.ExecComp('foo = y + z',
                                          y={'shape': (10,)},
                                          z={'shape': (10,)},
                                          foo={'shape': (10,)}))

        p.model.connect('a_comp.y', 'exec_comp.y')
        p.model.connect('a_comp.z', 'exec_comp.z')
        p.driver.declare_coloring()

        warnings.filterwarnings('ignore', category=om.UnitsWarning)

        with warnings.catch_warnings(record=True) as w:
            p.setup()
            unit_warnings = [wm for wm in w if wm.category is om.UnitsWarning]
            assert (len(unit_warnings) == 0)

    def test_doc_error_on_openmdao_warning(self):
        """
        Test the ability to raise a UnitWarning to an error.
        """
        import warnings
        import openmdao.api as om

        class AComp(om.ExplicitComponent):

            def initialize(self):
                pass

            def setup(self):
                self.add_input('a', shape=(10,), units='m')
                self.add_input('x', shape=(10,), units='1/s')
                self.add_input('b', shape=(10,), units='m/s')

                self.add_output('y', shape=(10,), units='m')
                self.add_output('z', shape=(10,), units='m/s')

                self.declare_coloring(wrt='*', form='cs')

            def compute(self, inputs, outputs):
                outputs['y'] = inputs['a'] * inputs['x'] + inputs['b']
                outputs['z'] = inputs['b'] * inputs['x']

        p = om.Problem()

        p.model.add_subsystem('a_comp', AComp())
        p.model.add_subsystem('exec_comp',
                              om.ExecComp('foo = y + z',
                                          y={'shape': (10,)},
                                          z={'shape': (10,)},
                                          foo={'shape': (10,)}))

        p.model.connect('a_comp.y', 'exec_comp.y')
        p.model.connect('a_comp.z', 'exec_comp.z')
        p.driver.declare_coloring()

        warnings.filterwarnings('error', category=om.OpenMDAOWarning)

        with self.assertRaises(om.UnitsWarning) as e:
            p.setup()

        expected = "<model> <class Group>: Output 'a_comp.y' with units of 'm' is connected to " \
                   "input 'exec_comp.y' which has no units."

        self.assertEqual(expected, str(e.exception), )
