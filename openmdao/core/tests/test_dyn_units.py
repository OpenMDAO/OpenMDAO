import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_warnings


class DynUnitsComp(om.ExplicitComponent):
    # component whose inputs and outputs have dynamic units
    def __init__(self, n_inputs=1):
        super().__init__()
        self.n_inputs = n_inputs

        for i in range(self.n_inputs):
            self.add_input(f"x{i+1}", units_by_conn=True, copy_units=f"y{i+1}")
            self.add_output(f"y{i+1}", units_by_conn=True, copy_units=f"x{i+1}")

    def compute(self, inputs, outputs):
        for i in range(self.n_inputs):
            outputs[f"y{i+1}"] = 2*inputs[f"x{i+1}"]

class DynUnitsGroupSeries(om.Group):
    # strings together some number of components in series.
    # component type is determined by comp_class
    def __init__(self, n_comps, n_inputs, comp_class):
        super().__init__()
        self.n_comps = n_comps
        self.n_inputs = n_inputs
        self.comp_class = comp_class

        for icmp in range(1, self.n_comps + 1):
            self.add_subsystem(f"C{icmp}", self.comp_class(n_inputs=self.n_inputs))

        for icmp in range(1, self.n_comps):
            for i in range(1, self.n_inputs + 1):
                self.connect(f"C{icmp}.y{i}", f"C{icmp+1}.x{i}")


class DynUnitsGroupConnectedInputs(om.Group):
    # contains some number of components with all of their matching inputs connected.
    # component type is determined by comp_class
    def __init__(self, n_comps, n_inputs, comp_class):
        super().__init__()
        self.n_comps = n_comps
        self.n_inputs = n_inputs
        self.comp_class = comp_class

        for icmp in range(1, self.n_comps + 1):
            self.add_subsystem(f"C{icmp}", self.comp_class(n_inputs=self.n_inputs),
                               promotes_inputs=['*'])


class TestDynUnits(unittest.TestCase):
    def test_simple(self):
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', units='m'))
        indep.add_output('x2', units='ft')
        p.model.add_subsystem('C1', DynUnitsComp(2))
        sink = p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'units_by_conn': True},
                                                  x2={'units_by_conn': True},
                                                  y1={'copy_units': 'x1'},
                                                  y2={'copy_units': 'x2'}))
        p.model.connect('C1.y1', 'sink.x1')
        p.model.connect('C1.y2', 'sink.x2')
        p.model.connect('indep.x1', 'C1.x1')
        p.model.connect('indep.x2', 'C1.x2')
        p.setup()
        p.run_model()
        self.assertEqual(sink._var_abs2meta['input']['sink.x1']['units'], 'm')
        self.assertEqual(sink._var_abs2meta['input']['sink.x2']['units'], 'ft')
        self.assertEqual(sink._var_abs2meta['output']['sink.y1']['units'], 'm')
        self.assertEqual(sink._var_abs2meta['output']['sink.y2']['units'], 'ft')

    def test_simple_compute_units(self):
        class DynUnitsCompDiv(om.ExplicitComponent):
            def setup(self):
                self.add_input('x1', units_by_conn=True)
                self.add_input('x2', units_by_conn=True)
                self.add_output('y', compute_units=lambda dct: dct['x1'] / dct['x2'])

            def compute(self, inputs, outputs):
                outputs['y'] = inputs['x1'] / inputs['x2']

        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', units='m'))
        indep.add_output('x2', units='s')
        C1 = p.model.add_subsystem('C1', DynUnitsCompDiv())  # this uses compute_units
        sink = p.model.add_subsystem('sink', om.ExecComp('y = x * 2.',
                                                  x={'units_by_conn': True, 'copy_units': 'y'},
                                                  y={'units_by_conn': True, 'copy_units': 'x'},))
        p.model.connect('indep.x1', 'C1.x1')
        p.model.connect('indep.x2', 'C1.x2')
        p.model.connect('C1.y', 'sink.x')

        p.setup()
        p.run_model()

        self.assertEqual(C1._var_abs2meta['input']['C1.x1']['units'], 'm')
        self.assertEqual(C1._var_abs2meta['input']['C1.x2']['units'], 's')
        self.assertEqual(C1._var_abs2meta['output']['C1.y']['units'], 'm/s')
        self.assertEqual(sink._var_abs2meta['output']['sink.y']['units'], 'm/s')

    def test_baseline_series(self):
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', units='m'))
        indep.add_output('x2', units='ft')
        p.model.add_subsystem('Gdyn', DynUnitsGroupSeries(3, 2, DynUnitsComp))
        sink = p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'units_by_conn': True, 'copy_units': 'y1'},
                                                  x2={'units_by_conn': True, 'copy_units': 'y2'},
                                                  y1={'units_by_conn': True, 'copy_units': 'x1'},
                                                  y2={'units_by_conn': True, 'copy_units': 'x2'}))
        p.model.connect('Gdyn.C3.y1', 'sink.x1')
        p.model.connect('Gdyn.C3.y2', 'sink.x2')
        p.model.connect('indep.x1', 'Gdyn.C1.x1')
        p.model.connect('indep.x2', 'Gdyn.C1.x2')
        p.setup()
        p.run_model()
        self.assertEqual(sink._var_abs2meta['input']['sink.x1']['units'], 'm')
        self.assertEqual(sink._var_abs2meta['input']['sink.x2']['units'], 'ft')
        self.assertEqual(sink._var_abs2meta['output']['sink.y1']['units'], 'm')
        self.assertEqual(sink._var_abs2meta['output']['sink.y2']['units'], 'ft')

    def test_copy_units_out_out(self):
        # test copy_units from output to output
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', units='m'))
        indep.add_output('x2', units='ft')
        p.model.add_subsystem('Gdyn', DynUnitsGroupSeries(3, 2, DynUnitsComp))
        sink =p.model.add_subsystem('sink', om.ExecComp('y1, y2, y3 = x1*2, x2*2, x1*3',
                                                  x1={'units_by_conn': True},
                                                  x2={'units_by_conn': True},
                                                  y1={'copy_units': 'x1'},
                                                  y2={'copy_units': 'x2'},
                                                  y3={'copy_units': 'y1'}))
        p.model.connect('Gdyn.C3.y1', 'sink.x1')
        p.model.connect('Gdyn.C3.y2', 'sink.x2')
        p.model.connect('indep.x1', 'Gdyn.C1.x1')
        p.model.connect('indep.x2', 'Gdyn.C1.x2')
        p.setup()
        p.run_model()
        self.assertEqual(sink._var_abs2meta['input']['sink.x1']['units'], 'm')
        self.assertEqual(sink._var_abs2meta['input']['sink.x2']['units'], 'ft')
        self.assertEqual(sink._var_abs2meta['output']['sink.y1']['units'], 'm')
        self.assertEqual(sink._var_abs2meta['output']['sink.y2']['units'], 'ft')
        self.assertEqual(sink._var_abs2meta['output']['sink.y3']['units'], 'm')

    def test_copy_units_in_in(self):
        # test copy_units from input to input
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1'))
        indep.add_output('x2')
        p.model.add_subsystem('Gdyn', DynUnitsGroupSeries(3, 2, DynUnitsComp))
        comp = p.model.add_subsystem('comp', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'copy_units': 'y2'},
                                                  x2={'copy_units': 'x1'},
                                                  y1={'units_by_conn': True},
                                                  y2={'units_by_conn': True}))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'units': 'm'},
                                                  x2={'units': 'ft'},
                                                  y1={'units': 'm'},
                                                  y2={'units': 'ft'}))
        p.model.connect('indep.x1', 'Gdyn.C1.x1')
        p.model.connect('indep.x2', 'Gdyn.C1.x2')
        p.model.connect('Gdyn.C3.y1', 'comp.x1')
        p.model.connect('Gdyn.C3.y2', 'comp.x2')
        p.model.connect('comp.y1', 'sink.x1')
        p.model.connect('comp.y2', 'sink.x2')
        p.setup()
        p.run_model()
        self.assertEqual(comp._var_abs2meta['input']['comp.x1']['units'], 'ft')
        self.assertEqual(comp._var_abs2meta['input']['comp.x2']['units'], 'ft')
        self.assertEqual(comp._var_abs2meta['output']['comp.y1']['units'], 'm')
        self.assertEqual(comp._var_abs2meta['output']['comp.y2']['units'], 'ft')

    def test_copy_units_in_in_unresolvable(self):
        # test copy_units from input to input
        p = om.Problem(name='copy_units_in_in_unresolvable')
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', units='m'))
        indep.add_output('x2', units='ft')
        p.model.add_subsystem('comp', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'copy_units': 'x2'},
                                                  x2={'copy_units': 'x1'},
                                                  y1={'units_by_conn': True},
                                                  y2={'units_by_conn': True}))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'units': 'm'},
                                                  x2={'units': 'ft'},
                                                  y1={'units': 'm'},
                                                  y2={'units': 'ft'}))
        p.model.connect('indep.x1', 'comp.x1')
        p.model.connect('indep.x2', 'comp.x2')
        p.model.connect('comp.y1', 'sink.x1')
        p.model.connect('comp.y2', 'sink.x2')
        with self.assertRaises(RuntimeError) as cm:
            p.setup()
            p.final_setup()

        self.assertEqual(cm.exception.args[0],
            "\nCollected errors for problem 'copy_units_in_in_unresolvable':"
            "\n   <model> <class Group>: Failed to resolve units for ['comp.x1', 'comp.x2']. To see the dynamic units dependency graph, do 'openmdao view_dyn_units <your_py_file>'.")

    def test_mismatched_dyn_units(self):
        # this is a source and sink with units, but their units are incompatible
        p = om.Problem(name='mismatched_dyn_units')
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', units='m'))
        indep.add_output('x2', units='s')
        p.model.add_subsystem('comp', om.ExecComp('y1, y2 = x1*.5, x2*.7',
                                                  x1={'units_by_conn': True},
                                                  x2={'units_by_conn': True},
                                                  y1={'compute_units': lambda u: u['x1'] / u['x2']},
                                                  y2={'copy_units': 'x2'}))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'units': 'm'},
                                                  x2={'units': 's'},
                                                  y1={'units': 'm'},
                                                  y2={'units': 's'}))
        p.model.connect('indep.x1', 'comp.x1')
        p.model.connect('indep.x2', 'comp.x2')
        p.model.connect('comp.y1', 'sink.x1')
        p.model.connect('comp.y2', 'sink.x2')
        with self.assertRaises(Exception) as cm:
            p.setup()
            p.final_setup()

        self.assertEqual(str(cm.exception),
           "\nCollected errors for problem 'mismatched_dyn_units':"
           "\n   <model> <class Group>: Output units of 'm/s' for 'comp.y1' are incompatible with input units of 'm' for 'sink.x1'.")

    def test_baseline_conn_inputs(self):
        # this is a source with units and sink without units, with a DynUnitsGroupConnectedInputs between them
        # indep.x? connects to Gdyn.C?.x?
        p = om.Problem()
        indep = p.model.add_subsystem('indep', om.IndepVarComp('x1', units='m'), promotes_outputs=['*'])
        indep.add_output('x2', units='ft')
        Gdyn = p.model.add_subsystem('Gdyn', DynUnitsGroupConnectedInputs(2, 2, DynUnitsComp),
                              promotes_inputs=['*'])
        sink = p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'units_by_conn': True, 'copy_units': 'y1'},
                                                  x2={'units_by_conn': True, 'copy_units': 'y2'},
                                                  y1={'units_by_conn': True, 'copy_units': 'x1'},
                                                  y2={'units_by_conn': True, 'copy_units': 'x2'}))
        p.model.connect('Gdyn.C1.y1', 'sink.x1')
        p.model.connect('Gdyn.C2.y2', 'sink.x2')
        p.setup()
        p.run_model()
        self.assertEqual(sink._var_abs2meta['input']['sink.x1']['units'], 'm')
        self.assertEqual(sink._var_abs2meta['input']['sink.x2']['units'], 'ft')
        self.assertEqual(sink._var_abs2meta['output']['sink.y1']['units'], 'm')
        self.assertEqual(sink._var_abs2meta['output']['sink.y2']['units'], 'ft')
        self.assertEqual(Gdyn.C1._var_abs2meta['output']['Gdyn.C1.y2']['units'], 'ft')  # unconnected dyn shaped output
        self.assertEqual(Gdyn.C2._var_abs2meta['output']['Gdyn.C2.y1']['units'], 'm')  # unconnected dyn shaped output

    def test_cycle_fwd_rev(self):
        # now put the DynUnitsGroupSeries in a cycle (sink.y2 feeds back into Gdyn.C1.x2). Units are known
        # at the sink
        p = om.Problem()
        p.model.add_subsystem('Gdyn', DynUnitsGroupSeries(3,2, DynUnitsComp))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'units': 'm',},
                                                  x2={'units': 'ft',},
                                                  y1={'units': 'm',},
                                                  y2={'units': 'ft',}))
        p.model.connect('Gdyn.C3.y1', 'sink.x1')
        p.model.connect('Gdyn.C3.y2', 'sink.x2')
        p.model.connect('sink.y2', 'Gdyn.C1.x2')
        p.setup()
        p.run_model()
        for n, m in p.model._var_abs2meta['input'].items():
            if n.endswith('1'):
                self.assertEqual(m['units'], 'm')
            elif n.endswith('2'):
                self.assertEqual(m['units'], 'ft')
        for n, m in p.model._var_abs2meta['output'].items():
            if n.endswith('1'):
                self.assertEqual(m['units'], 'm')
            elif n.endswith('2'):
                self.assertEqual(m['units'], 'ft')

    def test_cycle_rev(self):
        # now put the DynUnitsGroupSeries in a cycle (sink.y2 feeds back into Gdyn.C1.x2), but here,
        # only the sink outputs are known and inputs are coming from auto_ivcs.
        p = om.Problem()
        p.model.add_subsystem('Gdyn', DynUnitsGroupSeries(3,2, DynUnitsComp))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'units': 'm'},
                                                  x2={'units': 'ft'},
                                                  y1={'units': 'm'},
                                                  y2={'units': 'ft'}))
        p.model.connect('Gdyn.C3.y1', 'sink.x1')
        p.model.connect('Gdyn.C3.y2', 'sink.x2')
        p.model.connect('sink.y2', 'Gdyn.C1.x2')
        p.setup()
        p.run_model()
        for n, m in p.model._var_abs2meta['input'].items():
            if n.endswith('1'):
                self.assertEqual(m['units'], 'm')
            elif n.endswith('2'):
                self.assertEqual(m['units'], 'ft')
        for n, m in p.model._var_abs2meta['output'].items():
            if n.endswith('1'):
                self.assertEqual(m['units'], 'm')
            elif n.endswith('2'):
                self.assertEqual(m['units'], 'ft')


    def test_cycle_unresolved(self):
        # now put the DynUnitsGroupSeries in a cycle (sink.y2 feeds back into Gdyn.C1.x2), but here,
        # sink.y2 is unsized, so no var in the '2' loop can get resolved.
        p = om.Problem(name='cycle_unresolved')
        p.model.add_subsystem('indep', om.IndepVarComp('x1', units='m'))
        p.model.add_subsystem('Gdyn', DynUnitsGroupSeries(3,2, DynUnitsComp))
        p.model.add_subsystem('sink', om.ExecComp('y1, y2 = x1*2, x2*2',
                                                  x1={'units_by_conn': True, 'copy_units': 'y1'},
                                                  x2={'units_by_conn': True, 'copy_units': 'y2'},
                                                  y1={'units_by_conn': True, 'copy_units': 'x1'},
                                                  y2={'units_by_conn': True, 'copy_units': 'x2'}))
        p.model.connect('Gdyn.C3.y1', 'sink.x1')
        p.model.connect('Gdyn.C3.y2', 'sink.x2')
        p.model.connect('sink.y2', 'Gdyn.C1.x2')
        p.model.connect('indep.x1', 'Gdyn.C1.x1')
        with self.assertRaises(Exception) as cm:
            p.setup()
            p.final_setup()

        self.assertEqual(str(cm.exception),
           "\nCollected errors for problem 'cycle_unresolved':"
           "\n   <model> <class Group>: Failed to resolve units for "
           "['Gdyn.C1.x2', 'Gdyn.C1.y2', 'Gdyn.C2.x2', 'Gdyn.C2.y2', 'Gdyn.C3.x2', 'Gdyn.C3.y2', "
           "'sink.x2', 'sink.y2']. To see the dynamic units dependency graph, do "
           "'openmdao view_dyn_units <your_py_file>'.")

    def test_bad_copy_units_name(self):
        p = om.Problem(name='bad_copy_units_name')
        p.model.add_subsystem('indep', om.IndepVarComp('x1'))
        p.model.add_subsystem('sink', om.ExecComp('y1 = x1*2',
                                                  x1={'units_by_conn': True, 'copy_units': 'y1'},
                                                  y1={'units_by_conn': True, 'copy_units': 'x11'}))
        p.model.connect('indep.x1', 'sink.x1')
        p.setup()

        expected_warnings = (
            (om.OpenMDAOWarning, "<model> <class Group>: 'units_by_conn' was set for unconnected variable 'sink.y1'."),
            (om.OpenMDAOWarning, "<model> <class Group>: Can't copy units of variable 'sink.x11'. Variable doesn't exist or is not continuous.")
        )

        with assert_warnings(expected_warnings):
            p.model._setup_dynamic_properties()

        with self.assertRaises(Exception) as cm:
            p.final_setup()

        self.assertEqual(str(cm.exception),
           "\nCollected errors for problem 'bad_copy_units_name':"
           "\n   <model> <class Group>: Failed to resolve units for ['sink.x1', 'sink.y1']. To see the dynamic units dependency graph, do 'openmdao view_dyn_units <your_py_file>'.")

    def test_unconnected_var_dyn_units(self):
        p = om.Problem(name='unconnected_var_dyn_units')
        p.model.add_subsystem('indep', om.IndepVarComp('x1', units='m'))
        p.model.add_subsystem('sink', om.ExecComp('y1 = x1*2',
                                                  x1={'units_by_conn': True, 'copy_units': 'y1'},
                                                  y1={'units_by_conn': True}))
        p.model.connect('indep.x1', 'sink.x1')
        p.setup()

        expected_warnings = (
            (om.OpenMDAOWarning, "<model> <class Group>: 'units_by_conn' was set for unconnected variable 'sink.y1'."),
        )

        with assert_warnings(expected_warnings):
            p.model._setup_dynamic_properties()

        with self.assertRaises(Exception) as cm:
            p.final_setup()

        self.assertEqual(str(cm.exception),
           "\nCollected errors for problem 'unconnected_var_dyn_units':"
           "\n   <model> <class Group>: Failed to resolve units for ['sink.y1']. To see the dynamic units dependency graph, do 'openmdao view_dyn_units <your_py_file>'.")


class DynPartialsComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', units_by_conn=True, copy_units='y')
        self.add_output('y', units_by_conn=True, copy_units='x')

    def setup_partials(self):
        size = self._get_var_meta('x', 'size')
        self.mat = np.eye(size) * 3.
        rng = np.arange(size)
        self.declare_partials('y', 'x', rows=rng, cols=rng, val=3.0)

    def compute(self, inputs, outputs):
        outputs['y'] = self.mat.dot(inputs['x'])


class TestDynUnitsFeature(unittest.TestCase):
    def test_feature_fwd(self):

        p = om.Problem()
        p.model.add_subsystem('indeps', om.IndepVarComp('x', units='m/s'))
        p.model.add_subsystem('comp', DynPartialsComp())
        p.model.add_subsystem('sink', om.ExecComp('y=x',
                                                  x={'units_by_conn': True},
                                                  y={'copy_units': 'x'}))
        p.model.connect('indeps.x', 'comp.x')
        p.model.connect('comp.y', 'sink.x')
        p.setup()
        p.run_model()
        self.assertEqual(p.model.sink._var_abs2meta['input']['sink.x']['units'], 'm/s')
        self.assertEqual(p.model.sink._var_abs2meta['output']['sink.y']['units'], 'm/s')

    def test_feature_rev(self):

        p = om.Problem()
        p.model.add_subsystem('comp', DynPartialsComp())
        p.model.add_subsystem('sink', om.ExecComp('y=x', x={'units': 'm/s'}, y={'units': 'm/s'}))
        p.model.connect('comp.y', 'sink.x')
        p.setup()
        p.run_model()
        self.assertEqual(p.model.comp._var_abs2meta['input']['comp.x']['units'], 'm/s')
        self.assertEqual(p.model.comp._var_abs2meta['output']['comp.y']['units'], 'm/s')

    def test_feature_middle(self):

        p = om.Problem()
        p.model.add_subsystem('source', om.ExecComp('y=x',
                                                  x={'units': 'm/s'},
                                                  y={'units': 'm/s'}))
        p.model.add_subsystem('comp', om.ExecComp('y=x',
                                                  x={'units_by_conn': True},
                                                  y={'units_by_conn': True}))
        p.model.add_subsystem('sink', om.ExecComp('y=x',
                                                  x={'units': 'm/s'},
                                                  y={'units': 'm/s'}))
        p.model.connect('source.y', 'comp.x')
        p.model.connect('comp.y', 'sink.x')
        p.setup()
        p.run_model()
        self.assertEqual(p.model.comp._var_abs2meta['input']['comp.x']['units'], 'm/s')
        self.assertEqual(p.model.comp._var_abs2meta['output']['comp.y']['units'], 'm/s')


class DynUntComp(om.ExplicitComponent):

    def setup(self):
        self.add_input('x', val=1., units_by_conn=True)
        self.add_output('y', val=1., compute_units=lambda unitsdct: unitsdct['x'])

    def compute(self, inputs, outputs):
        outputs['y'] = 3. * inputs['x']


class PGroup(om.Group):

    def setup(self):
        self.add_subsystem('comp1', DynUntComp(), promotes_inputs=['x'])
        self.add_subsystem('comp2', DynUntComp(), promotes_inputs=['x'])

    def configure(self):
        self.set_input_defaults('x', units='ft')


class TestDynUnitsWithInputConns(unittest.TestCase):
    # this tests the retrieval of units info from a set_input_defaults call during
    # dynamic units determination, which happens *before* group input defaults have
    # been fully processed.
    def test_group_input_defaults(self):
        prob = om.Problem()
        prob.model.add_subsystem('sub', PGroup())

        prob.setup()

        prob.run_model()

        self.assertEqual(prob.model._var_abs2meta['input']['sub.comp1.x']['units'], 'ft')
        self.assertEqual(prob.model._var_abs2meta['input']['sub.comp2.x']['units'], 'ft')

    def test_shape_from_conn_input(self):
        prob = om.Problem()
        sub = prob.model.add_subsystem('sub', om.Group())
        sub.add_subsystem('comp1', om.ExecComp('y=3*x', x={'units_by_conn': True}, y={'copy_units': 'x'}),
                          promotes_inputs=['x'])
        sub.add_subsystem('comp2', om.ExecComp('y=3*x', x={'units': 'ft'}, y={'units': 'ft'}),
                          promotes_inputs=['x'])

        prob.setup()

        prob.run_model()

        self.assertEqual(prob.model._var_abs2meta['input']['sub.comp1.x']['units'], 'ft')
        self.assertEqual(prob.model._var_abs2meta['output']['sub.comp1.y']['units'], 'ft')

    def test_shape_from_conn_input_mismatch_group_inputs(self):
        prob = om.Problem(name='units_from_conn_input_mismatch_group_inputs')
        sub = prob.model.add_subsystem('sub', om.Group())
        sub.add_subsystem('comp1', om.ExecComp('y=3*x', x={'units_by_conn': True}, y={'copy_units': 'x'}),
                          promotes_inputs=['x'])
        sub.add_subsystem('comp2', om.ExecComp('y=3*x', x={'units': 'ft'}, y={'units': 'ft'}),
                          promotes_inputs=['x'])

        # this causes a mismatch error because without it, the auto_ivc will get units of 'ft' since only comp2 has units that are not None,
        # but setting 'x' units to 'km' here results in comp1.x having units of 'km'.  Since comp1.x and comp2.x then have the same numerical
        # values but different units, their 'true' values are mismatched.
        sub.set_input_defaults('x', units='km')

        with self.assertRaises(Exception) as cm:
            prob.setup()
            prob.final_setup()

        # just make sure we still get a clear error msg

        self.assertEqual(cm.exception.args[0],
           "\nCollected errors for problem 'units_from_conn_input_mismatch_group_inputs':"
           "\n   <model> <class Group>: The following inputs, ['sub.comp1.x', 'sub.comp2.x'], promoted to 'sub.x', are connected but their metadata entries ['val'] differ. Call model.set_input_defaults('sub.x', val=?) to remove the ambiguity.")


if __name__ == "__main__":
    unittest.main()
