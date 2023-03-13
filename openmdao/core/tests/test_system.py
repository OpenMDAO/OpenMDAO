""" Unit tests for the system interface."""

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ExplicitComponent
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.utils.om_warnings import OMDeprecationWarning


class TestSystem(unittest.TestCase):

    def test_vector_context_managers(self):
        g1 = Group()
        g1.add_subsystem('Indep', IndepVarComp('a', 5.0), promotes=['a'])
        g2 = g1.add_subsystem('G2', Group(), promotes=['*'])
        g2.add_subsystem('C1', ExecComp('b=2*a'), promotes=['a', 'b'])

        model = Group()
        model.add_subsystem('G1', g1, promotes=['b'])
        model.add_subsystem('Sink', ExecComp('c=2*b'), promotes=['b'])

        p = Problem(model=model)
        p.set_solver_print(level=0)

        # Test pre-setup errors
        with self.assertRaises(Exception) as cm:
            inputs, outputs, residuals = model.get_nonlinear_vectors()
        self.assertEqual(str(cm.exception),
                         "<class Group>: Cannot get vectors because setup has not yet been called.")

        with self.assertRaises(Exception) as cm:
            d_inputs, d_outputs, d_residuals = model.get_linear_vectors()
        self.assertEqual(str(cm.exception),
                         "<class Group>: Cannot get vectors because setup has not yet been called.")

        p.setup()
        p.run_model()

        # Test inputs with original values
        inputs, outputs, residuals = model.get_nonlinear_vectors()
        self.assertEqual(inputs['G1.G2.C1.a'], 5.)

        inputs, outputs, residuals = g1.get_nonlinear_vectors()
        self.assertEqual(inputs['G2.C1.a'], 5.)

        # Test inputs after setting a new value
        inputs, outputs, residuals = g2.get_nonlinear_vectors()
        inputs['C1.a'] = -1.

        inputs, outputs, residuals = model.get_nonlinear_vectors()
        self.assertEqual(inputs['G1.G2.C1.a'], -1.)

        inputs, outputs, residuals = g1.get_nonlinear_vectors()
        self.assertEqual(inputs['G2.C1.a'], -1.)

        # Test outputs with original values
        inputs, outputs, residuals = model.get_nonlinear_vectors()
        self.assertEqual(outputs['G1.G2.C1.b'], 10.)

        inputs, outputs, residuals = g2.get_nonlinear_vectors()

        # Test outputs after setting a new value
        inputs, outputs, residuals = model.get_nonlinear_vectors()
        outputs['G1.G2.C1.b'] = 123.
        self.assertEqual(outputs['G1.G2.C1.b'], 123.)

        inputs, outputs, residuals = g2.get_nonlinear_vectors()
        outputs['C1.b'] = 789.
        self.assertEqual(outputs['C1.b'], 789.)

        # Test residuals
        inputs, outputs, residuals = model.get_nonlinear_vectors()
        residuals['G1.G2.C1.b'] = 99.0
        self.assertEqual(residuals['G1.G2.C1.b'], 99.0)

        # Test linear
        d_inputs, d_outputs, d_residuals = model.get_linear_vectors()
        d_outputs['G1.G2.C1.b'] = 10.
        self.assertEqual(d_outputs['G1.G2.C1.b'], 10.)

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
        model.add_subsystem('Sink', ExecComp(('c=2*b', 'z=2*y'),
                                             y=np.zeros((5, 1)),
                                             z=np.zeros((5, 1))),
                                             promotes=['b', 'y'])

        p = Problem(model=model)
        p.setup()

        p.set_solver_print(level=0)
        p.run_model()

        msg = "'.*' <class Group>: Failed to set value of '.*': could not broadcast input array from shape (.*) into shape (.*)."

        num_val = -10
        arr_val = -10*np.ones((5, 1))
        bad_val = -10*np.ones((10))

        inputs, outputs, residuals = g2.get_nonlinear_vectors()
        #
        # set input
        #

        # assign array to scalar
        with self.assertRaisesRegex(ValueError, msg):
            inputs['C1.a'] = arr_val

        # assign scalar to array
        inputs['C2.x'] = num_val
        assert_near_equal(inputs['C2.x'], arr_val, 1e-10)

        # assign array to array
        inputs['C2.x'] = arr_val
        assert_near_equal(inputs['C2.x'], arr_val, 1e-10)

        # assign bad array shape to array
        with self.assertRaisesRegex(ValueError, msg):
            inputs['C2.x'] = bad_val

        # assign list to array
        inputs['C2.x'] = arr_val.tolist()
        assert_near_equal(inputs['C2.x'], arr_val, 1e-10)

        # assign bad list shape to array
        with self.assertRaisesRegex(ValueError, msg):
            inputs['C2.x'] = bad_val.tolist()

        #
        # set output
        #

        # assign array to scalar
        with self.assertRaisesRegex(ValueError, msg):
            outputs['C1.b'] = arr_val

        # assign scalar to array
        outputs['C2.y'] = num_val
        assert_near_equal(outputs['C2.y'], arr_val, 1e-10)

        # assign array to array
        outputs['C2.y'] = arr_val
        assert_near_equal(outputs['C2.y'], arr_val, 1e-10)

        # assign bad array shape to array
        with self.assertRaisesRegex(ValueError, msg):
            outputs['C2.y'] = bad_val

        # assign list to array
        outputs['C2.y'] = arr_val.tolist()
        assert_near_equal(outputs['C2.y'], arr_val, 1e-10)

        # assign bad list shape to array
        with self.assertRaisesRegex(ValueError, msg):
            outputs['C2.y'] = bad_val.tolist()

        #
        # set residual
        #

        # assign array to scalar
        with self.assertRaisesRegex(ValueError, msg):
            residuals['C1.b'] = arr_val

        # assign scalar to array
        residuals['C2.y'] = num_val
        assert_near_equal(residuals['C2.y'], arr_val, 1e-10)

        # assign array to array
        residuals['C2.y'] = arr_val
        assert_near_equal(residuals['C2.y'], arr_val, 1e-10)

        # assign bad array shape to array
        with self.assertRaisesRegex(ValueError, msg):
            residuals['C2.y'] = bad_val

        # assign list to array
        residuals['C2.y'] = arr_val.tolist()
        assert_near_equal(residuals['C2.y'], arr_val, 1e-10)

        # assign bad list shape to array
        with self.assertRaisesRegex(ValueError, msg):
            residuals['C2.y'] = bad_val.tolist()

    def test_list_inputs_output_with_includes_excludes(self):
        from openmdao.test_suite.scripts.circuit_analysis import Circuit

        p = Problem()
        model = p.model

        model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()
        p.run_model()

        # Inputs with no includes or excludes
        inputs = model.list_inputs(out_stream=None)
        self.assertEqual(len(inputs), 11)

        # Inputs with includes
        inputs = model.list_inputs(includes=['*V_out*'], out_stream=None)
        self.assertEqual(len(inputs), 3)

        # Inputs with includes matching a promoted name
        inputs = model.list_inputs(includes=['*Vg*'], out_stream=None)
        self.assertEqual(len(inputs), 2)

        # Inputs with excludes
        inputs = model.list_inputs(excludes=['*V_out*'], out_stream=None)
        self.assertEqual(len(inputs), 8)

        # Inputs with excludes matching a promoted name
        inputs = model.list_inputs(excludes=['*Vg*'], out_stream=None)
        self.assertEqual(len(inputs), 9)

        # Inputs with includes and excludes
        inputs = model.list_inputs(includes=['*V_out*'], excludes=['*Vg*'], out_stream=None)
        self.assertEqual(len(inputs), 1)

        # Outputs with no includes or excludes. Explicit only
        outputs = model.list_outputs(implicit=False, out_stream=None)
        self.assertEqual(len(outputs), 5)

        # Outputs with includes. Explicit only
        outputs = model.list_outputs(includes=['*I'], implicit=False, out_stream=None)
        self.assertEqual(len(outputs), 4)

        # Outputs with excludes. Explicit only
        outputs = model.list_outputs(excludes=['circuit*'], implicit=False, out_stream=None)
        self.assertEqual(len(outputs), 2)

    def test_list_inputs_outputs_is_indep_is_des_var(self):
        from openmdao.test_suite.components.sellar_feature import SellarMDA

        model = SellarMDA()

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        # model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob = Problem(model)

        prob.setup()
        prob.final_setup()

        indeps = model.list_inputs(is_indep_var=True, out_stream=None)
        self.assertEqual(sorted([name for name, _ in indeps]),
                         ['cycle.d1.x', 'cycle.d1.z', 'cycle.d2.z',
                          'obj_cmp.x', 'obj_cmp.z'])

        desvars = model.list_inputs(is_design_var=True, out_stream=None)
        self.assertEqual(sorted([name for name, _ in desvars]),
                         ['cycle.d1.z', 'cycle.d2.z', 'obj_cmp.z'])

        non_desvars = model.list_inputs(is_design_var=False, out_stream=None)
        self.assertEqual(sorted([name for name, _ in non_desvars]),
                         ['con_cmp1.y1', 'con_cmp2.y2',
                          'cycle.d1.x', 'cycle.d1.y2', 'cycle.d2.y1',
                          'obj_cmp.x', 'obj_cmp.y1', 'obj_cmp.y2'])

        nonDV_indeps = model.list_inputs(is_indep_var=True, is_design_var=False, out_stream=None)
        self.assertEqual(sorted([name for name, _ in nonDV_indeps]),
                         ['cycle.d1.x', 'obj_cmp.x'])

        indeps = model.list_outputs(is_indep_var=True, list_autoivcs=True, out_stream=None)
        self.assertEqual(sorted([name for name, _ in indeps]),
                         ['_auto_ivc.v0', '_auto_ivc.v1'])

        desvars = model.list_outputs(is_design_var=True, list_autoivcs=True, out_stream=None)
        self.assertEqual(sorted([name for name, _ in desvars]),
                         ['_auto_ivc.v0'])

        non_desvars = model.list_outputs(is_design_var=False, list_autoivcs=True, out_stream=None)
        self.assertEqual(sorted([name for name, _ in non_desvars]),
                         ['_auto_ivc.v1', 'con_cmp1.con1', 'con_cmp2.con2',
                          'cycle.d1.y1', 'cycle.d2.y2', 'obj_cmp.obj'])

        nonDV_indeps = model.list_outputs(is_indep_var=True, is_design_var=False, list_autoivcs=True, out_stream=None)
        self.assertEqual(sorted([name for name, _ in nonDV_indeps]),
                         ['_auto_ivc.v1'])

    def test_setup_check_group(self):

        class CustomGroup(Group):

            def setup(self):
                self._custom_setup = True

            def _setup_check(self):
                if not hasattr(self, '_custom_setup'):
                    raise RuntimeError(f"{self.msginfo}: You forget to call super() in setup()")

        class BadGroup(CustomGroup):

            def setup(self):
                # should call super().setup() here
                pass

        p = Problem(model=BadGroup())

        with self.assertRaises(RuntimeError) as cm:
            p.setup()

        self.assertEqual(str(cm.exception), '<model> <class BadGroup>: You forget to call super() in setup()')

    def test_setup_check_component(self):

        class CustomComp(ExplicitComponent):

            def setup(self):
                self._custom_setup = True

            def _setup_check(self):
                if not hasattr(self, '_custom_setup'):
                    raise RuntimeError(f"{self.msginfo}: You forget to call super() in setup()")

        class BadComp(CustomComp):

            def setup(self):
                # should call super().setup() here
                pass

        p = Problem()
        p.model.add_subsystem('comp', BadComp())

        with self.assertRaises(RuntimeError) as cm:
            p.setup()

        self.assertEqual(str(cm.exception), "'comp' <class BadComp>: You forget to call super() in setup()")

    def test_missing_source(self):
        prob = Problem()
        root = prob.model

        root.add_subsystem('initial_comp', ExecComp(['x = 10']), promotes_outputs=['x'])

        prob.setup()

        with self.assertRaises(KeyError) as cm:
            root.get_source('f')

        self.assertEqual(cm.exception.args[0], "<model> <class Group>: source for 'f' not found.")

    def test_list_inputs_before_final_setup(self):
        class SpeedComp(ExplicitComponent):

            def setup(self):
                self.add_input('distance', val=1.0, units='km')
                self.add_input('time', val=1.0, units='h')
                self.add_output('speed', val=1.0, units='km/h')

            def compute(self, inputs, outputs):
                outputs['speed'] = inputs['distance'] / inputs['time']

        prob = Problem()
        prob.model.add_subsystem('c1', SpeedComp(), promotes=['*'])
        prob.model.add_subsystem('c2', ExecComp('f=speed',speed={'units': 'm/s'}), promotes=['*'])

        prob.setup()

        msg = ("Calling `list_inputs` before `final_setup` will only "
              "display the default values of variables and will not show the result of "
              "any `set_val` calls.")

        with assert_warning(UserWarning, msg):
            prob.model.list_inputs(units=True, prom_name=True)

    def test_get_io_metadata(self):
        from openmdao.test_suite.components.sellar_feature import SellarMDA

        prob = Problem()
        prob.model = SellarMDA()

        prob.setup()
        prob.set_solver_print(level=0)

        prob.run_model()

        assert_near_equal(prob.model.get_io_metadata(includes='x'), {
                          'cycle.d1.x': {'copy_shape': None,
                                         'desc': '',
                                         'discrete': False,
                                         'distributed': False,
                                         'global_shape': (1,),
                                         'global_size': 1,
                                         'has_src_indices': False,
                                         'prom_name': 'x',
                                         'shape': (1,),
                                         'shape_by_conn': False,
                                         'size': 1,
                                         'tags': set(),
                                         'units': None},
                          'obj_cmp.x':  {'copy_shape': None,
                                         'desc': '',
                                         'discrete': False,
                                         'distributed': False,
                                         'global_shape': (1,),
                                         'global_size': 1,
                                         'has_src_indices': False,
                                         'prom_name': 'x',
                                         'shape': (1,),
                                         'shape_by_conn': False,
                                         'size': 1,
                                         'tags': set(),
                                         'units': None}
                            })


if __name__ == "__main__":
    unittest.main()
