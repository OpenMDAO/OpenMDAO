"""Test the inputs report. """
import unittest
import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.double_sellar import DoubleSellar
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.visualization.inputs_report.inputs_report import inputs_report


try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

@use_tempdirs
class TestInputsReport(unittest.TestCase):

    def test_inputs_reports(self):
        prob = om.Problem(model=DoubleSellar(), reports=None)
        model = prob.model

        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g1.nonlinear_solver.options['rtol'] = 1.0e-5
        g1.linear_solver = om.DirectSolver(assemble_jac=True)
        g1.options['assembled_jac_type'] = 'dense'

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g2.nonlinear_solver.options['rtol'] = 1.0e-5
        g2.linear_solver = om.DirectSolver(assemble_jac=True)
        g2.options['assembled_jac_type'] = 'dense'

        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov(assemble_jac=True)
        model.options['assembled_jac_type'] = 'dense'

        model.nonlinear_solver.options['solve_subsystems'] = True

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['g1.y1'], 0.64, .00001)
        assert_near_equal(prob['g1.y2'], 0.80, .00001)
        assert_near_equal(prob['g2.y1'], 0.64, .00001)
        assert_near_equal(prob['g2.y2'], 0.80, .00001)

        expected = """| Absolute Name | Input Name | Source Name  | Source is IVC | Source is DV | Units | Shape | Tags | Val    | Min Val | Max Val | Absolute Source |
| :------------ | :--------- | :----------- | :-----------: | :----------: | :---- | :---- | :--- | :----- | ------: | ------: | :-------------- |
| g1.d1.z       | g1.z       | _auto_ivc.v0 |     True      |    False     |       | (2,)  | []   | [0 0]  |       0 |       0 | _auto_ivc.v0    |
| g1.d1.x       | g1.x       | g2.y2        |     False     |    False     |       | (1,)  | []   | [0.8]  |     0.8 |     0.8 | g2.d2.y2        |
| g1.d1.y2      | g1.y2      | g1.y2        |     False     |    False     |       | (1,)  | []   | [0.8]  |     0.8 |     0.8 | g1.d2.y2        |
| g1.d2.z       | g1.z       | _auto_ivc.v0 |     True      |    False     |       | (2,)  | []   | [0 0]  |       0 |       0 | _auto_ivc.v0    |
| g1.d2.y1      | g1.y1      | g1.y1        |     False     |    False     |       | (1,)  | []   | [0.64] |    0.64 |    0.64 | g1.d1.y1        |
| g2.d1.z       | g2.z       | _auto_ivc.v1 |     True      |    False     |       | (2,)  | []   | [0 0]  |       0 |       0 | _auto_ivc.v1    |
| g2.d1.x       | g2.x       | g1.y2        |     False     |    False     |       | (1,)  | []   | [0.8]  |     0.8 |     0.8 | g1.d2.y2        |
| g2.d1.y2      | g2.y2      | g2.y2        |     False     |    False     |       | (1,)  | []   | [0.8]  |     0.8 |     0.8 | g2.d2.y2        |
| g2.d2.z       | g2.z       | _auto_ivc.v1 |     True      |    False     |       | (2,)  | []   | [0 0]  |       0 |       0 | _auto_ivc.v1    |
| g2.d2.y1      | g2.y1      | g2.y1        |     False     |    False     |       | (1,)  | []   | [0.64] |    0.64 |    0.64 | g2.d1.y1        |
"""
        inputs_report(prob, outfile='temp_inputs_report.md', display=True, precision=6, title=None, tablefmt='github')
        with open('temp_inputs_report.md') as f:
            report_content = f.read()
        self.assertEqual(expected, report_content)

    def test_deprecated_flag(self):

        prob = om.Problem(reports=None)
        model = prob.model

        # Kludge to fake something like an IndepVarComp with an 'indep_var' tag on an output.
        class _SimpleIVC(om.ExecComp):

            def setup(self):
                self.add_output('x', val=2.0, tags=['indep_var'])
                self.add_output('y', val=5.0, tags=['openmdao:indep_var'])

            def compute(self, inputs, outputs):
                pass

        model.add_subsystem('c1', _SimpleIVC(), promotes_outputs=['x', 'y'])
        model.add_subsystem('c2', om.ExecComp('z = x + y'), promotes_inputs=['x', 'y'], promotes_outputs=['z'])

        prob.setup()

        prob.run_model()

        z = prob.get_val('z')

        assert_near_equal(7, z)

        expected = """| Absolute Name | Input Name | Source Name | Source is IVC | Source is DV | Units | Shape | Tags | Val | Min Val | Max Val | Absolute Source |
| :------------ | :--------- | :---------- | :-----------: | :----------: | :---- | :---- | :--- | :-- | ------: | ------: | :-------------- |
| c2.x          | x          | x           |     True      |    False     |       | (1,)  | []   | [2] |       2 |       2 | c1.x            |
| c2.y          | y          | y           |     True      |    False     |       | (1,)  | []   | [5] |       5 |       5 | c1.y            |
"""
        with assert_warning(om.OMDeprecationWarning, 'source output x is tagged with the deprecated'
                                                     ' `indep_var` tag. Please change this tag to'
                                                     ' `openmdao:indep_var` as `indep_var` will'
                                                     ' be deprecated in a future release.'):
            inputs_report(prob, outfile='temp_inputs_report.md', display=True, precision=6, title=None, tablefmt='github')
        with open('temp_inputs_report.md') as f:
            report_content = f.read()
        self.assertEqual(expected, report_content)

    def test_zero_size_input(self):
        class TestComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('foo', shape=0)
                self.add_output('bar', shape=0)

            def compute(self, inputs, outputs):
                outputs['bar'] = inputs['foo']

        p = om.Problem()
        p.model.add_subsystem('comp', TestComp())
        p.setup()

        p.set_val('comp.foo', np.array([]))
        p.run_model()

        expected = """| Absolute Name | Input Name | Source Name  | Source is IVC | Source is DV | Units | Shape | Tags | Val | Min Val | Max Val | Absolute Source |
| :------------ | :--------- | :----------- | :-----------: | :----------: | :---- | :---- | :--- | :-- | :------ | :------ | :-------------- |
| comp.foo      | comp.foo   | _auto_ivc.v0 |     True      |    False     |       | (0,)  | []   | []  | []      | []      | _auto_ivc.v0    |
"""

        inputs_report(p, outfile='temp_inputs_report.md', display=True, precision=6, title=None, tablefmt='github')
        with open('temp_inputs_report.md') as f:
            report_content = f.read()
        self.assertEqual(expected, report_content)
