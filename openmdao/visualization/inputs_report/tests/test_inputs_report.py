"""Test the inputs report. """
import unittest
import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.double_sellar import DoubleSellar
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.visualization.inputs_report.inputs_report import inputs_report

from openmdao.utils.mpi import MPI

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


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
@use_tempdirs
class TestInputReportsMPI(unittest.TestCase):
    N_PROCS = 2

    def test_multidim(self):
        # this test would error out before the fix
        class Adder(om.ExplicitComponent):
            def initialize(self):
                self.options.declare('n0')

            def setup(self):
                n0 = self.options['n0']
                self.add_input('x', shape_by_conn=True, distributed=True)
                self.add_output('x_sum', shape=1)

            def compute(self, inputs, outputs):
                outputs['x_sum'] = self.comm.allreduce(np.sum(inputs['x']))


        n0 = (20,2) if MPI.COMM_WORLD.rank ==0 else (8,2)
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc',om.IndepVarComp())
        ivc.add_output('x', val = np.random.random(n0), distributed=True)

        p.model.add_subsystem('adder', Adder(n0=n0))
        p.model.connect('ivc.x','adder.x')

        p.setup()
        p.run_model()

        inputs_report(p, outfile='temp_inputs_report.md', display=True, precision=6, title=None, tablefmt='github')
