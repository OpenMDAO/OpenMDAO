"""Test getting/setting variables and subjacs with promoted/relative/absolute names."""

import unittest
import numpy as np
from openmdao.api import Problem, Group, ExecComp, IndepVarComp, DirectSolver, ParallelGroup
from openmdao.utils.mpi import MPI

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


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
        self.assertEqual(str(ctx.exception), msg.format('x'))
        p._initial_condition_cache = {}

        with self.assertRaises(KeyError) as ctx:
            p['x']
        self.assertEqual(str(ctx.exception), msg.format('x'))

        # outputs
        with self.assertRaises(KeyError) as ctx:
            p['y'] = 5.0
        self.assertEqual(str(ctx.exception), msg.format('y'))
        p._initial_condition_cache = {}

        with self.assertRaises(KeyError) as ctx:
            p['y']
        self.assertEqual(str(ctx.exception), msg.format('y'))

        p.final_setup()

        msg = "Group (g): Variable name '{}' not found."
        inputs, outputs, residuals = g.get_nonlinear_vectors()

        # inputs
        for vname in ['x', 'g.c.x']:
            with self.assertRaises(KeyError) as cm:
               inputs[vname] = 5.0
            self.assertEqual(cm.exception.args[0], f"Group (g): Variable name '{vname}' not found.")

            with self.assertRaises(KeyError) as cm:
               inputs[vname]
            self.assertEqual(cm.exception.args[0], f"Group (g): Variable name '{vname}' not found.")


        # outputs
        for vname in ['y', 'g.c.y']:
            with self.assertRaises(KeyError) as cm:
               outputs[vname] = 5.0
            self.assertEqual(cm.exception.args[0], f"Group (g): Variable name '{vname}' not found.")

            with self.assertRaises(KeyError) as cm:
               outputs[vname]
            self.assertEqual(cm.exception.args[0], f"Group (g): Variable name '{vname}' not found.")

        msg = r'Variable name pair \("{}", "{}"\) not found.'
        jac = g.linear_solver._assembled_jac

        # d(output)/d(input)
        with self.assertRaisesRegex(KeyError, msg.format('y', 'x')):
            jac['y', 'x'] = 5.0
        with self.assertRaisesRegex(KeyError, msg.format('y', 'x')):
            jac['y', 'x']
        # allow absolute keys now
        # with self.assertRaisesRegex(KeyError, msg.format('g.c.y', 'g.c.x')):
        #     jac['g.c.y', 'g.c.x'] = 5.0
        # with self.assertRaisesRegex(KeyError, msg.format('g.c.y', 'g.c.x')):
        #     deriv = jac['g.c.y', 'g.c.x']

        # d(output)/d(output)
        with self.assertRaisesRegex(KeyError, msg.format('y', 'y')):
            jac['y', 'y'] = 5.0
        with self.assertRaisesRegex(KeyError, msg.format('y', 'y')):
            jac['y', 'y']
        # allow absoute keys now
        # with self.assertRaisesRegex(KeyError, msg.format('g.c.y', 'g.c.y')):
        #     jac['g.c.y', 'g.c.y'] = 5.0
        # with self.assertRaisesRegex(KeyError, msg.format('g.c.y', 'g.c.y')):
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

        msg1 = "Group (g): Variable name '{}' not found."
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

        with self.assertRaises(KeyError) as cm:
            inputs['g.c2.x'] = 5.0
        self.assertEqual(cm.exception.args[0], msg1.format('g.c2.x'))

        with self.assertRaises(KeyError) as cm:
            inputs['g.c2.x']
        self.assertEqual(cm.exception.args[0], msg1.format('g.c2.x'))

        # outputs
        with self.assertRaises(KeyError) as cm:
            outputs['g.c2.y'] = 5.0
        self.assertEqual(cm.exception.args[0], msg1.format('g.c2.y'))

        with self.assertRaises(KeyError) as cm:
            outputs['g.c2.y']
        self.assertEqual(cm.exception.args[0], msg1.format('g.c2.y'))

        msg1 = r'Variable name pair \("{}", "{}"\) not found.'

        jac = g.linear_solver._assembled_jac

        # d(outputs)/d(inputs)
        with self.assertRaises(Exception) as context:
            jac['y', 'x'] = 5.0
        self.assertEqual(str(context.exception), msg2)

        with self.assertRaises(Exception) as context:
            self.assertEqual(jac['y', 'x'], 5.0)
        self.assertEqual(str(context.exception), msg2)

    def test_serial_multi_src_inds(self):
        p = Problem()
        p.model.add_subsystem('indep', IndepVarComp('x', val=np.ones(10)))
        p.model.add_subsystem('C1', ExecComp('y=x*2.', x=np.zeros(7), y=np.zeros(7)))
        p.model.add_subsystem('C2', ExecComp('y=x*3.', x=np.zeros(3), y=np.zeros(3)))
        p.model.connect('indep.x', 'C1.x', src_indices=list(range(7)))
        p.model.connect('indep.x', 'C2.x', src_indices=list(range(7, 10)))
        p.setup()

        p['C1.x'] = (np.arange(7) + 1.) * 2.
        p['C2.x'] = (np.arange(7,10) + 1.) * 3.

        p.run_model()

        np.testing.assert_allclose(p['indep.x'][:7], (np.arange(7) + 1.) * 2.)
        np.testing.assert_allclose(p['indep.x'][7:10], (np.arange(7,10) + 1.) * 3.)
        np.testing.assert_allclose(p['C1.x'], (np.arange(7) + 1.) * 2.)
        np.testing.assert_allclose(p['C2.x'], (np.arange(7,10) + 1.) * 3.)
        np.testing.assert_allclose(p['C1.y'], (np.arange(7) + 1.) * 4.)
        np.testing.assert_allclose(p['C2.y'], (np.arange(7,10) + 1.) * 9.)

    def test_serial_multi_src_inds_promoted(self):
        p = Problem()
        p.model.add_subsystem('indep', IndepVarComp('x', val=np.ones(10)), promotes=['x'])
        p.model.add_subsystem('C1', ExecComp('y=x*2.',
                                             x={'value': np.zeros(7),
                                                'src_indices': list(range(7))},
                                             y={'value': np.zeros(7)}),
                              promotes=['x'])
        p.model.add_subsystem('C2', ExecComp('y=x*3.',
                                             x={'value': np.zeros(3),
                                                'src_indices': list(range(7,10))},
                                             y={'value': np.zeros(3)}),
                              promotes=['x'])
        p.setup()

        p['C1.x'] = (np.arange(7) + 1.) * 2.
        p['C2.x'] = (np.arange(7,10) + 1.) * 3.

        p.run_model()

        np.testing.assert_allclose(p['indep.x'][:7], (np.arange(7) + 1.) * 2.)
        np.testing.assert_allclose(p['indep.x'][7:10], (np.arange(7,10) + 1.) * 3.)
        np.testing.assert_allclose(p['C1.x'], (np.arange(7) + 1.) * 2.)
        np.testing.assert_allclose(p['C2.x'], (np.arange(7,10) + 1.) * 3.)
        np.testing.assert_allclose(p['C1.y'], (np.arange(7) + 1.) * 4.)
        np.testing.assert_allclose(p['C2.y'], (np.arange(7,10) + 1.) * 9.)

    def test_serial_multi_src_inds_units_promoted(self):
        p = Problem()
        indep = p.model.add_subsystem('indep', IndepVarComp(), promotes=['x'])
        indep.add_output('x', units='inch', val=np.ones(10))
        p.model.add_subsystem('C1', ExecComp('y=x*2.',
                                             x={'value': np.zeros(7),
                                                'units': 'ft',
                                                'src_indices': list(range(7))},
                                             y={'value': np.zeros(7), 'units': 'ft'}),
                              promotes=['x'])
        p.model.add_subsystem('C2', ExecComp('y=x*3.',
                                             x={'value': np.zeros(3),
                                                'units': 'inch',
                                                'src_indices': list(range(7,10))},
                                             y={'value': np.zeros(3), 'units': 'inch'}),
                              promotes=['x'])
        p.setup()

        p['C1.x'] = np.ones(7) * 2.
        p['C2.x'] = np.ones(3) * 3.

        p.run_model()

        np.testing.assert_allclose(p['indep.x'][:7], np.ones(7) * 24.)
        np.testing.assert_allclose(p['indep.x'][7:10], np.ones(3) * 3.)
        np.testing.assert_allclose(p['C1.x'], np.ones(7) * 2.)
        np.testing.assert_allclose(p['C1.y'], np.ones(7) * 4.)
        np.testing.assert_allclose(p['C2.x'], np.ones(3) * 3.)
        np.testing.assert_allclose(p['C2.y'], np.ones(3) * 9.)

    def test_serial_multi_src_inds_units_promoted_no_src(self):
        p = Problem()
        p.model.add_subsystem('C1', ExecComp('y=x*2.',
                                             x={'value': np.zeros(7),
                                                'units': 'ft',
                                                'src_indices': list(range(7))},
                                             y={'value': np.zeros(7), 'units': 'ft'}),
                              promotes=['x'])
        p.model.add_subsystem('C2', ExecComp('y=x*3.',
                                             x={'value': np.zeros(3),
                                                'units': 'inch',
                                                'src_indices': list(range(7, 10))},
                                             y={'value': np.zeros(3), 'units': 'inch'}),
                              promotes=['x'])
        p.model.add_subsystem('C3', ExecComp('y=x*4.',
                                             x={'value': np.zeros(10), 'units': 'mm'},
                                             y={'value': np.zeros(10), 'units': 'mm'}),
                         promotes=['x'])

        with self.assertRaises(RuntimeError) as cm:
            p.setup()

        self.assertEqual(str(cm.exception), "Group (<model>): The following inputs, ['C1.x', 'C2.x', 'C3.x'], promoted to 'x', are connected but the metadata entries ['units'] differ. Call <group>.set_input_defaults('x', units=?), where <group> is the Group named '' to remove the ambiguity.")

    def test_serial_multi_src_inds_units_setval_promoted(self):
        p = Problem()
        indep = p.model.add_subsystem('indep', IndepVarComp(), promotes=['x'])
        indep.add_output('x', units='inch', val=np.ones(10))
        p.model.add_subsystem('C1', ExecComp('y=x*2.',
                                             x={'value': np.zeros(7),
                                                'units': 'ft',
                                                'src_indices': list(range(7))},
                                             y={'value': np.zeros(7), 'units': 'ft'}),
                              promotes=['x'])
        p.model.add_subsystem('C2', ExecComp('y=x*3.',
                                             x={'value': np.zeros(3),
                                                'units': 'inch',
                                                'src_indices': list(range(7,10))},
                                             y={'value': np.zeros(3), 'units': 'inch'}),
                              promotes=['x'])
        p.setup()

        p.set_val('C1.x', np.ones(7) * 24., units='inch')
        p.set_val('C2.x', np.ones(3) * 3., units='inch')

        p.run_model()

        np.testing.assert_allclose(p['indep.x'][:7], np.ones(7) * 24.)
        np.testing.assert_allclose(p['indep.x'][7:10], np.ones(3) * 3.)
        np.testing.assert_allclose(p['C1.x'], np.ones(7) * 2.)
        np.testing.assert_allclose(p['C1.y'], np.ones(7) * 4.)
        np.testing.assert_allclose(p['C2.x'], np.ones(3) * 3.)
        np.testing.assert_allclose(p['C2.y'], np.ones(3) * 9.)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ParTestCase(unittest.TestCase):
    N_PROCS = 2

    def test_par_multi_src_inds(self):
        p = Problem()
        p.model.add_subsystem('indep', IndepVarComp('x', val=np.ones(10)))
        par = p.model.add_subsystem('par', ParallelGroup())
        par.add_subsystem('C1', ExecComp('y=x*2.', x=np.zeros(7), y=np.zeros(7)))
        par.add_subsystem('C2', ExecComp('y=x*3.', x=np.zeros(3), y=np.zeros(3)))
        p.model.connect('indep.x', 'par.C1.x', src_indices=list(range(7)))
        p.model.connect('indep.x', 'par.C2.x', src_indices=list(range(7, 10)))

        p.setup()

        p['indep.x'] = np.concatenate([(np.arange(7) + 1.) * 2., (np.arange(7, 10) + 1.) * 3.])

        p.run_model()

        np.testing.assert_allclose(p['indep.x'][:7], (np.arange(7) + 1.) * 2.)
        np.testing.assert_allclose(p['indep.x'][7:10], (np.arange(7,10) + 1.) * 3.)
        np.testing.assert_allclose(p.get_val('par.C1.x', get_remote=True), (np.arange(7) + 1.) * 2.)
        np.testing.assert_allclose(p.get_val('par.C2.x', get_remote=True), (np.arange(7,10) + 1.) * 3.)
        np.testing.assert_allclose(p.get_val('par.C1.y', get_remote=True), (np.arange(7) + 1.) * 4.)
        np.testing.assert_allclose(p.get_val('par.C2.y', get_remote=True), (np.arange(7,10) + 1.) * 9.)

    @unittest.expectedFailure
    def test_par_multi_src_inds_fail(self):
        p = Problem()
        p.model.add_subsystem('indep', IndepVarComp('x', val=np.ones(10)))
        par = p.model.add_subsystem('par', ParallelGroup())
        par.add_subsystem('C1', ExecComp('y=x*2.', x=np.zeros(7), y=np.zeros(7)))
        par.add_subsystem('C2', ExecComp('y=x*3.', x=np.zeros(3), y=np.zeros(3)))
        p.model.connect('indep.x', 'par.C1.x', src_indices=list(range(7)))
        p.model.connect('indep.x', 'par.C2.x', src_indices=list(range(7, 10)))

        p.setup()

        p['par.C1.x'] = (np.arange(7) + 1.) * 2.
        p['par.C2.x'] = (np.arange(7,10) + 1.) * 3.

        p.run_model()

        np.testing.assert_allclose(p['indep.x'][:7], (np.arange(7) + 1.) * 2.)
        np.testing.assert_allclose(p['indep.x'][7:10], (np.arange(7,10) + 1.) * 3.)
        np.testing.assert_allclose(p['par.C1.x'], (np.arange(7) + 1.) * 2.)
        np.testing.assert_allclose(p['par.C2.x'], (np.arange(7,10) + 1.) * 3.)
        np.testing.assert_allclose(p['par.C1.y'], (np.arange(7) + 1.) * 4.)
        np.testing.assert_allclose(p['par.C2.y'], (np.arange(7,10) + 1.) * 9.)


if __name__ == '__main__':
    unittest.main()
