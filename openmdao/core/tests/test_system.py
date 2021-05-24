""" Unit tests for the system interface."""

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ExplicitComponent
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.warnings import OMDeprecationWarning


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
            d_inputs, d_outputs, d_residuals = model.get_linear_vectors('vec')
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
        d_inputs, d_outputs, d_residuals = model.get_linear_vectors('linear')
        d_outputs['G1.G2.C1.b'] = 10.
        self.assertEqual(d_outputs['G1.G2.C1.b'], 10.)

        # Test linear with invalid vec_name
        with self.assertRaises(Exception) as cm:
            d_inputs, d_outputs, d_residuals = model.get_linear_vectors('bad_name')
        self.assertEqual(str(cm.exception),
                         "<model> <class Group>: There is no linear vector named %s" % 'bad_name')

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

    def test_recording_options_deprecated(self):
        prob = Problem()
        msg = "The recording option, record_model_metadata, on System is deprecated. " \
              "Recording of model metadata will always be done"
        with assert_warning(OMDeprecationWarning, msg):
            prob.model.recording_options['record_model_metadata'] = True

        msg = "The recording option, record_metadata, on System is deprecated. " \
              "Recording of metadata will always be done"
        with assert_warning(OMDeprecationWarning, msg):
            prob.model.recording_options['record_metadata'] = True

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


if __name__ == "__main__":
    unittest.main()
