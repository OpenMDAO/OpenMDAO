""" Unit test for the SqliteRecorder. """
import errno
import os
import unittest
from io import StringIO
import numpy as np

import sqlite3

from shutil import rmtree
from tempfile import mkdtemp

import openmdao.api as om
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.assert_utils import assert_no_warning

from openmdao.test_suite.components.ae_tests import AEComp
from openmdao.test_suite.components.sellar import SellarDerivatives, SellarDerivativesGrouped, \
    SellarProblem, SellarStateConnection, SellarProblemWithArrays, SellarDis1, SellarDis2
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.paraboloid_problem import ParaboloidProblem
from openmdao.solvers.linesearch.tests.test_backtracking import ImplCompTwoStates

from openmdao.recorders.tests.sqlite_recorder_test_utils import assertMetadataRecorded, \
    assertDriverIterDataRecorded, assertSystemIterDataRecorded, assertSolverIterDataRecorded, \
    assertViewerDataRecorded, assertSystemMetadataIdsRecorded, assertSystemIterCoordsRecorded, \
    assertDriverDerivDataRecorded, assertProblemDerivDataRecorded

from openmdao.recorders.tests.recorder_test_utils import run_driver
from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_equal_arrays
from openmdao.utils.general_utils import determine_adder_scaler, remove_whitespace
from openmdao.utils.testing_utils import use_tempdirs

# check that pyoptsparse is installed. if it is, try to use SLSQP.
OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class Cycle(om.Group):

    def setup(self):
        self.add_subsystem('d1', SellarDis1())
        self.add_subsystem('d2', SellarDis2())
        self.connect('d1.y1', 'd2.y1')

        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.options['maxiter'] = 20
        self.linear_solver = om.DirectSolver()

        # paths are relative, not absolute like for Driver and Problem
        self.nonlinear_solver.recording_options['includes'] = ['d1*']
        self.nonlinear_solver.recording_options['excludes'] = ['*z']

class SellarMDAConnect(om.Group):

    def setup(self):
        indeps = self.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', 1.0)
        indeps.add_output('z', np.array([5.0, 2.0]))

        self.add_subsystem('cycle', Cycle())

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                  z=np.array([0.0, 0.0]), x=0.0))

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'))
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'))

        self.connect('indeps.x', ['cycle.d1.x', 'obj_cmp.x'])
        self.connect('indeps.z', ['cycle.d1.z', 'cycle.d2.z', 'obj_cmp.z'])
        self.connect('cycle.d1.y1', ['obj_cmp.y1', 'con_cmp1.y1'])
        self.connect('cycle.d2.y2', ['obj_cmp.y2', 'con_cmp2.y2'])


@use_tempdirs
class TestSqliteRecorder(unittest.TestCase):

    def setUp(self):
        self.filename = "sqlite_test"
        self.recorder = om.SqliteRecorder(self.filename, record_viewer_data=False)

        self.eps = 1e-3

    def test_only_desvars_recorded(self):
        prob = SellarProblem()

        driver = prob.driver
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = False
        driver.recording_options['record_constraints'] = False
        driver.recording_options['includes'] = []
        driver.add_recorder(self.recorder)

        prob.setup()
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0, )]
        expected_outputs = {"x": [1.0, ], "z": [5.0, 2.0]}

        expected_data = ((coordinate, (t0, t1), expected_outputs, None, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_add_recorder_after_setup(self):
        prob = SellarProblem()

        driver = prob.driver
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = False
        driver.recording_options['record_constraints'] = False
        driver.recording_options['includes'] = []

        driver.add_recorder(self.recorder)

        prob.setup()
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0, )]
        expected_outputs = {"x": [1.0, ], "z": [5.0, 2.0]}

        expected_data = ((coordinate, (t0, t1), expected_outputs, None, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_only_objectives_recorded(self):
        prob = SellarProblem()

        driver = prob.driver
        driver.recording_options['record_desvars'] = False
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = False
        driver.recording_options['includes'] = []
        driver.add_recorder(self.recorder)

        prob.setup()
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0, )]
        expected_objectives = {"obj_cmp.obj": [28.58830817, ]}
        expected_outputs = expected_objectives

        expected_data = ((coordinate, (t0, t1), expected_outputs, None, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_only_constraints_recorded(self):
        prob = SellarProblem()

        driver = prob.driver
        driver.recording_options['record_desvars'] = False
        driver.recording_options['record_objectives'] = False
        driver.recording_options['record_constraints'] = True
        driver.recording_options['includes'] = []
        driver.add_recorder(self.recorder)

        prob.setup()
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0, )]
        expected_constraints = {
            "con_cmp1.con1": [-22.42830237, ],
            "con_cmp2.con2": [-11.94151185, ],
        }
        expected_outputs = expected_constraints

        expected_data = ((coordinate, (t0, t1), expected_outputs, None, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_simple_driver_recording(self):
        prob = ParaboloidProblem()

        driver = prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['record_derivatives'] = True
        driver.recording_options['includes'] = ['*']
        driver.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'ScipyOptimize_SLSQP', (4, )]

        expected_desvars = {"p1.x": [7.16706813], "p2.y": [-7.83293187]}
        expected_objectives = {"comp.f_xy": [-27.0833]}
        expected_constraints = {"con.c": [-15.0]}

        expected_outputs = expected_desvars
        expected_outputs.update(expected_objectives)
        expected_outputs.update(expected_constraints)

        expected_inputs = {
            "con.x": 7.1666667,
            "comp.y": -7.83333333,
            "comp.x": 7.1666667,
            "con.y": -7.8333333
        }

        expected_data = ((coordinate, (t0, t1), expected_outputs, expected_inputs, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

        expected_derivs = {
            "comp.f_xy!p1.x": np.array([[0.50120438]]),
            "comp.f_xy!p2.y": np.array([[-0.49879562]]),
            "con.c!p1.x": np.array([[-1.0]]),
            "con.c!p2.y": np.array([[1.0]])
        }

        expected_data = ((coordinate, (t0, t1), expected_derivs),)
        assertDriverDerivDataRecorded(self, expected_data, self.eps)

    def test_driver_recording_ndarray_var_settings(self):
        prob = SellarProblemWithArrays()

        driver = prob.driver
        driver.recording_options['record_desvars'] = False
        driver.recording_options['record_objectives'] = False
        driver.recording_options['record_constraints'] = True
        driver.recording_options['includes'] = []
        driver.add_recorder(self.recorder)

        prob.setup()
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0, )]
        expected_constraints = {
            "con_cmp1.con1": [-22.42830237, ],
            "con_cmp2.con2": [-11.94151185, ],
        }
        expected_outputs = expected_constraints

        expected_data = ((coordinate, (t0, t1), expected_outputs, None, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SLSQP")
    def test_simple_driver_recording_pyoptsparse(self):
        prob = ParaboloidProblem()

        driver = prob.driver = pyOptSparseDriver(optimizer='SLSQP')
        driver.options['print_results'] = False
        driver.opt_settings['ACC'] = 1e-9
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['record_derivatives'] = True
        driver.recording_options['includes'] = ['*']
        driver.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'pyOptSparse_SLSQP', (3, )]

        expected_desvars = {"p1.x": [7.16706813], "p2.y": [-7.83293187]}
        expected_objectives = {"comp.f_xy": [-27.0833]}
        expected_constraints = {"con.c": [-15.0]}

        expected_outputs = expected_desvars
        expected_outputs.update(expected_objectives)
        expected_outputs.update(expected_constraints)

        expected_inputs = {
            "con.x": 7.1666667,
            "comp.y": -7.83333333,
            "comp.x": 7.1666667,
            "con.y": -7.8333333
        }

        expected_data = ((coordinate, (t0, t1), expected_outputs, expected_inputs, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

        expected_derivs = {
            "comp.f_xy!p1.x": np.array([[0.50120438]]),
            "comp.f_xy!p2.y": np.array([[-0.49879562]]),
            "con.c!p1.x": np.array([[-1.0]]),
            "con.c!p2.y": np.array([[1.0]])
        }

        expected_data = ((coordinate, (t0, t1), expected_derivs),)
        assertDriverDerivDataRecorded(self, expected_data, self.eps)

    def test_double_run_driver_option_overwrite(self):
        prob = ParaboloidProblem()

        driver = prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)

        prob.model.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()

        cr = om.CaseReader(self.filename)

        self.assertTrue(cr._system_options['root']['component_options']['assembled_jac_type'], 'csc')

        # New option and re-run of run_driver
        prob.model.options['assembled_jac_type'] = 'dense'
        prob.setup()
        prob.run_driver()

        cr = om.CaseReader(self.filename)
        self.assertTrue(cr._system_options['root_1']['component_options']['assembled_jac_type'], 'dense')

        stream = StringIO()

        cr.list_model_options(out_stream=stream)

        text = stream.getvalue().split('\n')

        expected = [
            "Run Number: 0",
            "    Subsystem: root",
            "        assembled_jac_type : csc",
            "Run Number: 1",
            "    Subsystem: root",
            "        assembled_jac_type : dense"
        ]

        for i, line in enumerate(expected):
            if line and not line.startswith('-'):
                self.assertEqual(remove_whitespace(text[i]), remove_whitespace(line))

        stream = StringIO()

        cr.list_model_options(run_counter=1, out_stream=stream)

        text = stream.getvalue().split('\n')

        expected = [
            "Run Number: 1",
            "    Subsystem: root",
            "        assembled_jac_type : dense"
        ]

        for i, line in enumerate(expected):
            if line and not line.startswith('-'):
                self.assertEqual(remove_whitespace(text[i]), remove_whitespace(line))

    def test_double_run_model_option_overwrite(self):
        prob = ParaboloidProblem()

        driver = prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)

        prob.model.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        prob.run_model()

        cr = om.CaseReader(self.filename)

        self.assertTrue(cr._system_options['root']['component_options']['assembled_jac_type'], 'csc')

        # New option and re-run of run_driver
        prob.model.options['assembled_jac_type'] = 'dense'
        prob.setup()
        prob.run_model()

        cr = om.CaseReader(self.filename)
        self.assertTrue(cr._system_options['root_1']['component_options']['assembled_jac_type'], 'dense')

        stream = StringIO()

        cr.list_model_options(out_stream=stream)

        text = stream.getvalue().split('\n')

        expected = [
            "Run Number: 0",
            "    Subsystem: root",
            "        assembled_jac_type : csc",
            "Run Number: 1",
            "    Subsystem: root",
            "        assembled_jac_type : dense"
        ]

        for i, line in enumerate(expected):
            if line and not line.startswith('-'):
                self.assertEqual(remove_whitespace(text[i]), remove_whitespace(line))

        stream = StringIO()

        cr.list_model_options(run_counter=1, out_stream=stream)

        text = stream.getvalue().split('\n')

        expected = [
            "Run Number: 1",
            "    Subsystem: root",
            "        assembled_jac_type : dense"
        ]

        for i, line in enumerate(expected):
            if line and not line.startswith('-'):
                self.assertEqual(remove_whitespace(text[i]), remove_whitespace(line))

    def test_simple_driver_recording_with_prefix(self):
        prob = ParaboloidProblem()

        driver = prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['record_derivatives'] = True
        driver.recording_options['includes'] = ['*']
        driver.add_recorder(self.recorder)
        prob.model.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        run1_t0, run1_t1 = run_driver(prob, case_prefix='Run1')
        run2_t0, run2_t1 = run_driver(prob, case_prefix='Run2')
        prob.cleanup()

        run1_coord = [0, 'ScipyOptimize_SLSQP', (4, )]  # 1st run, 5 iterations
        run2_coord = [0, 'ScipyOptimize_SLSQP', (0, )]  # 2nd run, 1 iteration

        expected_desvars = {"p1.x": [7.16706813], "p2.y": [-7.83293187]}
        expected_objectives = {"comp.f_xy": [-27.0833]}
        expected_constraints = {"con.c": [-15.0]}

        expected_outputs = expected_desvars
        expected_outputs.update(expected_objectives)
        expected_outputs.update(expected_constraints)

        expected_inputs = {
            "con.x": 7.1666667,
            "comp.y": -7.83333333,
            "comp.x": 7.1666667,
            "con.y": -7.8333333
        }

        expected_data = (
            (run1_coord, (run1_t0, run1_t1), expected_outputs, expected_inputs, None),
        )
        assertDriverIterDataRecorded(self, expected_data, self.eps, prefix='Run1')

        expected_data = (
            (run2_coord, (run2_t0, run2_t1), expected_outputs, expected_inputs, None),
        )
        assertDriverIterDataRecorded(self, expected_data, self.eps, prefix='Run2')

        expected_derivs = {
            "comp.f_xy!p1.x": np.array([[0.50120438]]),
            "comp.f_xy!p2.y": np.array([[-0.49879562]]),
            "con.c!p1.x": np.array([[-1.0]]),
            "con.c!p2.y": np.array([[1.0]])
        }

        expected_data = (
            (run1_coord, (run1_t0, run1_t1), expected_derivs),
        )
        assertDriverDerivDataRecorded(self, expected_data, self.eps, prefix='Run1')

    def test_driver_everything_recorded_by_default(self):
        prob = ParaboloidProblem()

        driver = prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)
        driver.add_recorder(self.recorder)
        driver.recording_options['includes'] = ['*']

        prob.setup()
        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'ScipyOptimize_SLSQP', (3, )]

        expected_desvars = {"p1.x": [7.16706813, ], "p2.y": [-7.83293187]}
        expected_objectives = {"comp.f_xy": [-27.0833]}
        expected_constraints = {"con.c": [-15.0]}

        expected_inputs = {
            "con.x": 7.1666667,
            "comp.y": -7.83333333,
            "comp.x": 7.1666667,
            "con.y": -7.8333333
        }

        expected_outputs = expected_desvars
        expected_outputs.update(expected_objectives)
        expected_outputs.update(expected_constraints)

        expected_data = ((coordinate, (t0, t1), expected_outputs, expected_inputs, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_driver_records_metadata(self):
        prob = SellarProblem()

        recorder = om.SqliteRecorder(self.filename)

        driver = prob.driver
        driver.recording_options['includes'] = ["p1.x"]
        driver.add_recorder(recorder)

        prob.setup()
        prob.final_setup()  # Conclude setup but don't run model.
        prob.cleanup()

        prom2abs = {
            'input': {
                'z': ['d1.z', 'd2.z', 'obj_cmp.z'],
                'x': ['d1.x', 'obj_cmp.x'],
                'y2': ['d1.y2', 'obj_cmp.y2', 'con_cmp2.y2'],
                'y1': ['d2.y1', 'obj_cmp.y1', 'con_cmp1.y1']
            },
            'output': {
                '_auto_ivc.v0': ['_auto_ivc.v0'],
                '_auto_ivc.v1': ['_auto_ivc.v1'],
                'y1': ['d1.y1'],
                'y2': ['d2.y2'],
                'obj': ['obj_cmp.obj'],
                'con1': ['con_cmp1.con1'],
                'con2': ['con_cmp2.con2']
            }
        }

        abs2prom = {
            'input': {
                'd1.z': 'z',
                'd1.x': 'x',
                'd1.y2': 'y2',
                'd2.z': 'z',
                'd2.y1': 'y1',
                'obj_cmp.x': 'x',
                'obj_cmp.y1': 'y1',
                'obj_cmp.y2': 'y2',
                'obj_cmp.z': 'z',
                'con_cmp1.y1': 'y1',
                'con_cmp2.y2': 'y2'
            },
            'output': {
                '_auto_ivc.v0': '_auto_ivc.v0',
                '_auto_ivc.v1': '_auto_ivc.v1',
                'd1.y1': 'y1',
                'd2.y2': 'y2',
                'obj_cmp.obj': 'obj',
                'con_cmp1.con1': 'con1',
                'con_cmp2.con2': 'con2'
            }
        }

        assertMetadataRecorded(self, prom2abs, abs2prom)
        expected_problem_metadata = {
            'connections_list_length': 11,
            'tree_children_length': 6,
            'abs2prom': abs2prom,
        }
        assertViewerDataRecorded(self, expected_problem_metadata)

    def test_deprecated_option(self):
        # check that deprecated options are recorded but no warning is issued
        from openmdao.core.driver import Driver
        class MyDriver(Driver):
            def _declare_options(self):
                # Deprecated option
                self.options.declare('user_teriminate_signal', default=None, desc='Oops.',
                                     deprecation="The option was misspelled and is deprecated.")

        prob = om.Problem(driver=MyDriver())
        prob.driver.add_recorder(om.SqliteRecorder(self.filename))

        prob.setup()
        with assert_no_warning(DeprecationWarning):
            prob.final_setup()
        prob.cleanup()

        expected_problem_metadata = {
            'connections_list_length': 0,
            'tree_children_length': 0,
            'abs2prom': {}
        }
        data = assertViewerDataRecorded(self, expected_problem_metadata)
        self.assertTrue('user_teriminate_signal' in data['driver']['options'],
                        'Deprecated key not found in recorded options')

    def test_system_record_model_metadata(self):
        # first check to see if recorded recursively, which is the default
        prob = om.Problem(model=SellarDerivatives())
        prob.setup()

        recorder = om.SqliteRecorder("cases.sql")
        prob.model.add_recorder(recorder)

        prob.set_solver_print(level=0)
        prob.run_model()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")
        # Quick check to see that keys and values were recorded
        for key in ['root', '_auto_ivc', 'd1', 'd2', 'obj_cmp', 'con_cmp1', 'con_cmp2']:
            self.assertTrue(key in cr._system_options.keys())

        value = cr._system_options['root']['component_options']['assembled_jac_type']
        self.assertEqual(value, 'csc')  # quick check only. Too much to check exhaustively

    def test_record_system_options(self):
        # Regardless what object the case recorder is attached to, system options
        #  should be recorded for all systems in the model

        expected_system_options_keys = ['root', '_auto_ivc', 'd1', 'd2', 'obj_cmp', 'con_cmp1',
                                        'con_cmp2']

        # Recorder on Driver
        prob = om.Problem(model=SellarDerivatives())
        prob.setup()
        recorder = om.SqliteRecorder("cases_driver.sql")
        prob.driver.add_recorder(recorder)
        prob.set_solver_print(level=0)
        prob.run_model()
        prob.cleanup()
        cr = om.CaseReader("cases_driver.sql")
        # Quick check to see that keys and values were recorded
        for key in expected_system_options_keys:
            self.assertTrue(key in cr._system_options.keys())
        value = cr._system_options['root']['component_options']['assembled_jac_type']
        self.assertEqual('csc', value)  # quick check only. Too much to check exhaustively

        # Recorder on Problem
        prob = om.Problem(model=SellarDerivatives())
        prob.setup()
        recorder = om.SqliteRecorder("cases_problem.sql")
        prob.add_recorder(recorder)
        prob.set_solver_print(level=0)
        prob.run_model()
        prob.cleanup()
        cr = om.CaseReader("cases_problem.sql")
        # Quick check to see that keys and values were recorded
        for key in expected_system_options_keys:
            self.assertTrue(key in cr._system_options.keys())
        value = cr._system_options['root']['component_options']['assembled_jac_type']
        self.assertEqual(value, 'csc')  # quick check only. Too much to check exhaustively

        # Recorder on a subsystem
        prob = om.Problem(model=SellarDerivatives())
        prob.setup()
        recorder = om.SqliteRecorder("cases_subsystem.sql")
        prob.model.d1.add_recorder(recorder)
        prob.set_solver_print(level=0)
        prob.run_model()
        prob.cleanup()
        cr = om.CaseReader("cases_subsystem.sql")
        # Quick check to see that keys and values were recorded
        for key in expected_system_options_keys:
            self.assertTrue(key in cr._system_options.keys())
        value = cr._system_options['root']['component_options']['assembled_jac_type']
        self.assertEqual(value, 'csc')  # quick check only. Too much to check exhaustively

        # Recorder on a solver
        prob = om.Problem(model=SellarDerivatives())
        prob.setup()
        recorder = om.SqliteRecorder("cases_solver.sql")
        prob.model.nonlinear_solver.add_recorder(recorder)
        prob.set_solver_print(level=0)
        prob.run_model()
        prob.cleanup()
        cr = om.CaseReader("cases_solver.sql")
        # Quick check to see that keys and values were recorded
        for key in expected_system_options_keys:
            self.assertTrue(key in cr._system_options.keys())
        value = cr._system_options['root']['component_options']['assembled_jac_type']
        self.assertEqual(value, 'csc')  # quick check only. Too much to check exhaustively

    def test_warning__system_options_overwriting(self):

        prob = ParaboloidProblem()
        prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)
        prob.add_recorder(self.recorder)
        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()
        prob.record('final')

        prob.setup()
        msg = "The model is being run again, if the options or scaling of any components " \
              "has changed then only their new values will be recorded."

        with assert_warning(UserWarning, msg):
            prob.run_driver()

    def test_without_n2_data(self):
        prob = SellarProblem()

        recorder = om.SqliteRecorder(self.filename, record_viewer_data=False)

        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.final_setup()  # Conclude setup but don't run model.
        prob.cleanup()

        assertViewerDataRecorded(self, None)

    def test_record_system(self):
        prob = SellarProblem()
        prob.setup()

        model = prob.model
        model.recording_options['record_inputs'] = True
        model.recording_options['record_outputs'] = True
        model.recording_options['record_residuals'] = True
        model.add_recorder(self.recorder)

        model.nonlinear_solver.options['use_apply_nonlinear'] = True

        d1 = model.d1  # SellarDis1withDerivatives, an ExplicitComp
        d1.recording_options['record_inputs'] = True
        d1.recording_options['record_outputs'] = True
        d1.recording_options['record_residuals'] = True
        d1.add_recorder(self.recorder)

        obj_cmp = model.obj_cmp  # an ExecComp
        obj_cmp.recording_options['record_inputs'] = True
        obj_cmp.recording_options['record_outputs'] = True
        obj_cmp.recording_options['record_residuals'] = True
        obj_cmp.add_recorder(self.recorder)

        t0, t1 = run_driver(prob)

        expected_data = [
            # data from 'd1'
            [
                # coords
                [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ),
                 'NonlinearBlockGS', (6, ), 'd1._solve_nonlinear', (6, )],
                # timestamps
                (t0, t1),
                # inputs
                {"d1.y2": [12.05848815], "d1.z": [5.0, 2.0], "d1.x": [1.0, ]},
                # outputs
                {"d1.y1": [25.58830237]},
                # residuals
                {"d1.y1": [0.0]}
            ],

            # data from 'obj_cmp'
            [
                # coords
                [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ),
                 'NonlinearBlockGS', (6, ), 'obj_cmp._solve_nonlinear', (6, )],
                # timestamps
                (t0, t1),
                # inputs
                {"obj_cmp.z": [5.0, 2.0],
                 "obj_cmp.y1": [25.58830236],
                 "obj_cmp.x": [1.0, ],
                 "obj_cmp.y2": [12.05857185]},
                # outputs
                {"obj_cmp.obj": [28.58830816]},
                # residuals
                {"obj_cmp.obj": [0.0]}
            ],
        ]
        assertSystemIterDataRecorded(self, expected_data, self.eps)

        # run again with prefix, only changes should be iter count reset and timestamps
        t0, t1 = run_driver(prob, case_prefix='Run#2')
        prob.cleanup()

        expected_data[0][0] = [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ),
                               'NonlinearBlockGS', (0, ), 'd1._solve_nonlinear', (0, )]
        expected_data[0][1] = (t0, t1)

        expected_data[1][0] = [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ),
                               'NonlinearBlockGS', (0, ), 'obj_cmp._solve_nonlinear', (0, )]
        expected_data[1][1] = (t0, t1)

        assertSystemIterDataRecorded(self, expected_data, self.eps, prefix='Run#2')

    def test_includes(self):
        prob = ParaboloidProblem()

        driver = prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)

        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['includes'] = ['*']
        driver.recording_options['excludes'] = ['y*']
        driver.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)

        prob.cleanup()

        coordinate = [0, 'ScipyOptimize_SLSQP', (3, )]

        expected_desvars = {"p1.x": prob["p1.x"], "p2.y": prob["p2.y"]}
        expected_objectives = {"comp.f_xy": prob['comp.f_xy']}
        expected_constraints = {"con.c": prob['con.c']}

        expected_responses = expected_objectives.copy()
        expected_responses.update(expected_constraints)

        expected_outputs = expected_desvars
        expected_outputs.update(expected_objectives)
        expected_outputs.update(expected_constraints)

        expected_inputs = {
            "con.x": 7.1666667,
            "comp.y": -7.83333333,
            "comp.x": 7.1666667,
            "con.y": -7.8333333
        }

        expected_data = ((coordinate, (t0, t1), expected_outputs, expected_inputs, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_includes_post_setup(self):
        prob = ParaboloidProblem()

        driver = prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)

        prob.setup()

        # Set up recorder after intitial setup.
        driver.add_recorder(self.recorder)
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['includes'] = ['*']
        driver.recording_options['excludes'] = ['y*']

        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'ScipyOptimize_SLSQP', (3, )]

        expected_desvars = {"p1.x": prob["p1.x"], "p2.y": prob["p2.y"]}
        expected_objectives = {"comp.f_xy": prob['comp.f_xy']}
        expected_constraints = {"con.c": prob['con.c']}

        expected_outputs = expected_desvars
        expected_outputs.update(expected_objectives)
        expected_outputs.update(expected_constraints)

        expected_inputs = {
            "con.x": 7.1666667,
            "comp.y": -7.83333333,
            "comp.x": 7.1666667,
            "con.y": -7.8333333
        }

        expected_data = ((coordinate, (t0, t1), expected_outputs, expected_inputs, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_record_system_with_hierarchy(self):
        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=om.NonlinearRunOnce)
        prob.setup(mode='rev')

        model = prob.model
        model.recording_options['record_inputs'] = True
        model.recording_options['record_outputs'] = True
        model.recording_options['record_residuals'] = True
        model.add_recorder(self.recorder)

        model.mda.nonlinear_solver.options['use_apply_nonlinear'] = True

        d1 = model.mda.d1
        d1.recording_options['record_inputs'] = True
        d1.recording_options['record_outputs'] = True
        d1.recording_options['record_residuals'] = True
        d1.add_recorder(self.recorder)

        d2 = model.mda.d2
        d2.recording_options['record_inputs'] = True
        d2.recording_options['record_outputs'] = True
        d2.recording_options['record_residuals'] = True
        d2.add_recorder(self.recorder)

        prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)

        t0, t1 = run_driver(prob)
        prob.cleanup()

        #
        # check data for 'd1'
        #
        coordinate = [
            0,
            'ScipyOptimize_SLSQP', (1, ),
            'root._solve_nonlinear', (1, ),
            'NLRunOnce', (0, ),
            'mda._solve_nonlinear', (1, ),
            'NonlinearBlockGS', (0,),
            'mda.d1._solve_nonlinear', (7, )
        ]

        expected_inputs = {
            "mda.d1.z": [5.0, 2.0],
            "mda.d1.x": [1.0, ],
            "mda.d1.y2": [12.0584865, ],
        }
        expected_outputs = {"mda.d1.y1": [25.5883027, ], }
        expected_residuals = {"mda.d1.y1": [0.0, ], }

        expected_data = (
            (coordinate, (t0, t1), expected_inputs, expected_outputs, expected_residuals),
        )
        assertSystemIterDataRecorded(self, expected_data, self.eps)

        #
        # check data for 'd2'
        #
        coordinate = [
            0,
            'ScipyOptimize_SLSQP', (1, ),
            'root._solve_nonlinear', (1, ),
            'NLRunOnce', (0, ),
            'mda._solve_nonlinear', (1, ),
            'NonlinearBlockGS', (0,),
            'mda.d2._solve_nonlinear', (7, )
        ]

        expected_inputs = None
        expected_outputs = {"pz.z": [2.8640616, 0.825643, ], }
        expected_residuals = {"pz.z": [0.0, 0.0], }
        expected_inputs = {
            "mda.d2.z": [5.0, 2.0],
            "mda.d2.y1": [25.5883027, ],
        }
        expected_outputs = {"mda.d2.y2": [12.0584865, ], }
        expected_residuals = {"mda.d2.y2": [0.0, ], }

        expected_data = (
            (coordinate, (t0, t1), expected_inputs, expected_outputs, expected_residuals),
        )
        assertSystemIterDataRecorded(self, expected_data, self.eps)

    def test_record_solver(self):
        prob = SellarProblem()
        prob.setup()

        nl = prob.model.nonlinear_solver
        nl.options['use_apply_nonlinear'] = True
        nl.recording_options['record_abs_error'] = True
        nl.recording_options['record_rel_error'] = True
        nl.recording_options['record_solver_residuals'] = True
        nl.add_recorder(self.recorder)

        t0, t1 = run_driver(prob)

        coordinate = [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ), 'NonlinearBlockGS', (6, )]

        expected_abs_error = 1.318802844707e-10
        expected_rel_error = 3.62990740e-12

        expected_solver_output = {
            "con_cmp1.con1": [-22.42830237],
            "d1.y1": [25.58830237],
            "con_cmp2.con2": [-11.941511849],
            "_auto_ivc.v0": [5.0, 2.0],
            "obj_cmp.obj": [28.588308165],
            "d2.y2": [12.058488150],
            "_auto_ivc.v1": [1.0]
        }

        expected_solver_residuals = {
            "con_cmp1.con1": [0.0],
            "d1.y1": [-1.318802844707534e-10],
            "con_cmp2.con2": [0.0],
            "_auto_ivc.v0": [0.0, 0.0],
            "obj_cmp.obj": [0.0],
            "d2.y2": [0.0],
            "_auto_ivc.v1": [0.0]
        }

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps)

        #
        # run again with a prefix, iter_counts should be reset
        #
        t0, t1 = run_driver(prob, case_prefix='run_again')
        prob.cleanup()

        coordinate = [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ), 'NonlinearBlockGS', (0, )]

        expected_abs_error = 2.60769184e-12
        expected_rel_error = 0.01977317421405673

        expected_solver_residuals = {
            "con_cmp1.con1": [0.0],
            "d1.y1": [2.60769184e-12],
            "con_cmp2.con2": [0.0],
            "_auto_ivc.v0": [0.0, 0.0],
            "obj_cmp.obj": [0.0],
            "d2.y2": [0.0],
            "_auto_ivc.v1": [0.0]
        }

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps, prefix='run_again')

    def test_record_solver_includes_excludes(self):
        prob = om.Problem()

        prob.model = SellarMDAConnect()

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-8

        prob.set_solver_print(level=0)

        prob.model.add_design_var('indeps.x', lower=0, upper=10)
        prob.model.add_design_var('indeps.z', lower=0, upper=10)
        prob.model.add_objective('obj_cmp.obj')
        prob.model.add_constraint('con_cmp1.con1', upper=0)
        prob.model.add_constraint('con_cmp2.con2', upper=0)

        prob.setup()

        nl = prob.model._get_subsystem('cycle').nonlinear_solver
        nl.add_recorder(self.recorder)

        prob['indeps.x'] = 2.
        prob['indeps.z'] = [-1., -1.]

        prob.run_driver()

        cr = om.CaseReader(self.filename)
        solver_cases = cr.list_cases('root.cycle.nonlinear_solver')

        # Test values from cases
        last_case = cr.get_case(solver_cases[-1])

        self.assertEqual(sorted(last_case.inputs.keys()), ['d1.x', 'd1.y2'])
        self.assertEqual(sorted(last_case.outputs.keys()), ['d1.y1'])

        rec = om.SqliteRecorder(os.path.join(self.tempdir, "gleep.sql"), record_viewer_data=False)
        nl.add_recorder(rec)

        nl.recording_options['includes'] = ['*']
        nl.recording_options['excludes'] = []
        prob.setup()


        # Make sure default includes and excludes still works
        prob = om.Problem()

        prob.model = SellarMDAConnect()

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-8

        prob.set_solver_print(level=0)

        prob.model.add_design_var('indeps.x', lower=0, upper=10)
        prob.model.add_design_var('indeps.z', lower=0, upper=10)
        prob.model.add_objective('obj_cmp.obj')
        prob.model.add_constraint('con_cmp1.con1', upper=0)
        prob.model.add_constraint('con_cmp2.con2', upper=0)

        prob.setup()

        nl = prob.model._get_subsystem('cycle').nonlinear_solver
        # Default includes and excludes
        nl.recording_options['includes'] = ['*']
        nl.recording_options['excludes'] = []

        filename = "sqlite2"
        recorder = om.SqliteRecorder(filename, record_viewer_data=False)
        nl.add_recorder(recorder)

        prob['indeps.x'] = 2.
        prob['indeps.z'] = [-1., -1.]

        prob.run_driver()

        cr = om.CaseReader(filename)
        solver_cases = cr.list_cases('root.cycle.nonlinear_solver')

        # Test values from cases
        last_case = cr.get_case(solver_cases[-1])

        self.assertEqual(sorted(last_case.inputs.keys()),
                         ['d1.x', 'd1.y2', 'd1.z', 'd2.y1', 'd2.z'])
        self.assertEqual(sorted(last_case.outputs.keys()), ['d1.y1', 'd2.y2'])

    def test_record_line_search_armijo_goldstein(self):
        prob = om.Problem()
        prob.model.add_subsystem('px', om.IndepVarComp('x', 1.0))
        prob.model.add_subsystem('comp', ImplCompTwoStates())
        prob.model.connect('px.x', 'comp.x')

        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        prob.model.nonlinear_solver.options['maxiter'] = 10
        prob.model.linear_solver = om.ScipyKrylov()

        ls = prob.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='vector')
        ls.add_recorder(self.recorder)
        ls.options['c'] = .1

        prob.setup()

        prob['px.x'] = 2.0
        prob['comp.y'] = 0.
        prob['comp.z'] = 1.6

        t0, t1 = run_driver(prob)

        prob.cleanup()

        coordinate = [
            0,
            'Driver', (0,),
            'root._solve_nonlinear', (0,),
            'NewtonSolver', (2,),
            'ArmijoGoldsteinLS', (2,)
        ]

        expected_abs_error = 3.2882366094914777
        expected_rel_error = 0.9999999999999998

        expected_solver_output = {
            "comp.z": [1.5],
            "comp.y": [1.75],
            "px.x": [2.0],
            }

        expected_solver_residuals = None

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps)

    def test_record_line_search_bounds_enforce(self):
        prob = SellarProblem()
        prob.setup()

        model = prob.model
        model.linear_solver = om.ScipyKrylov()

        nl = model.nonlinear_solver = om.NewtonSolver()
        nl.options['solve_subsystems'] = True
        nl.options['max_sub_solves'] = 4

        ls = nl.linesearch = om.BoundsEnforceLS(bound_enforcement='vector')
        ls.add_recorder(self.recorder)

        t0, t1 = run_driver(prob)

        prob.cleanup()

        coordinate = [
            0,
            'Driver', (0,),
            'root._solve_nonlinear', (0,),
            'NewtonSolver', (1,),
            'BoundsEnforceLS', (0,)
        ]

        expected_abs_error = 7.02783609310096e-10
        expected_rel_error = 8.078674883382422e-07

        expected_solver_output = {
            "con_cmp1.con1": [-22.42830237],
            "d1.y1": [25.58830237],
            "con_cmp2.con2": [-11.941511849],
            "_auto_ivc.v0": [5.0, 2.0],
            "obj_cmp.obj": [28.588308165],
            "d2.y2": [12.058488150],
            "_auto_ivc.v1": [1.0]
        }

        expected_solver_residuals = None

        expected_solver_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                                 expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_solver_data, self.eps)

    def test_record_pop_bug(self):
        prob = SellarProblem()
        model = prob.model

        model.add_subsystem('ae', AEComp())
        model.connect('y1', 'ae.x')
        prob.setup()

        model.linear_solver = om.ScipyKrylov()

        nl = model.nonlinear_solver = om.NewtonSolver()
        nl.options['solve_subsystems'] = True
        nl.options['max_sub_solves'] = 4

        ls = nl.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='vector')
        model.add_recorder(self.recorder)

        try:
            t0, t1 = run_driver(prob)
        except om.AnalysisError:
            pass

        self.assertTrue(len(prob._recording_iter.stack) == 0)

    def test_record_solver_nonlinear_block_gs(self):
        prob = SellarProblem(linear_solver=om.LinearBlockGS,
                             nonlinear_solver=om.NonlinearBlockGS)
        prob.setup()

        prob.model.nonlinear_solver.add_recorder(self.recorder)
        prob.model.nonlinear_solver.recording_options['record_solver_residuals'] = True
        prob.model.nonlinear_solver.options['use_apply_nonlinear'] = True

        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NonlinearBlockGS', (6, )]
        expected_abs_error = 1.31880284470753394998e-10
        expected_rel_error = 3.6299074030587596e-12

        expected_solver_output = {
            '_auto_ivc.v1': [1.],
            '_auto_ivc.v0': [5., 2.],
            'd1.y1': [25.58830237],
            'd2.y2': [12.05848815],
            'obj_cmp.obj': [28.58830817],
            'con_cmp1.con1': [-22.42830237],
            'con_cmp2.con2': [-11.94151185]
        }

        expected_solver_residuals = {
            '_auto_ivc.v1': [0.],
            '_auto_ivc.v0': [0., 0.],
            'd1.y1': [-1.31880284e-10],
            'd2.y2': [0.],
            'obj_cmp.obj': [0.],
            'con_cmp1.con1': [0.],
            'con_cmp2.con2': [0.]
        }

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps)

    def test_record_solver_nonlinear_block_jac(self):
        prob = SellarProblem(linear_solver=om.LinearBlockGS, nonlinear_solver=om.NonlinearBlockJac)
        prob.setup()

        prob.model.nonlinear_solver.add_recorder(self.recorder)

        prob.set_solver_print(-1)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NonlinearBlockJac', (9,)]

        expected_abs_error = 7.234027587097439e-07
        expected_rel_error = 1.991112651729199e-08

        expected_solver_output = {
            '_auto_ivc.v1': [1.],
            '_auto_ivc.v0': [5., 2.],
            'd1.y1': [25.58830237],
            'd2.y2': [12.05848815],
            'obj_cmp.obj': [28.58830817],
            'con_cmp1.con1': [-22.42830237],
            'con_cmp2.con2': [-11.94151185]
        }

        expected_solver_residuals = None

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps)

    def test_record_solver_nonlinear_newton(self):
        prob = SellarProblem(linear_solver=om.LinearBlockGS,
                             nonlinear_solver=om.NewtonSolver(solve_subsystems=False))
        prob.setup()

        prob.model.nonlinear_solver.add_recorder(self.recorder)

        prob.set_solver_print(-1)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (2,)]

        expected_abs_error = 2.1677810075550974e-10
        expected_rel_error = 5.966657077752565e-12

        expected_solver_output = {
            '_auto_ivc.v1': [1.],
            '_auto_ivc.v0': [5., 2.],
            'd1.y1': [25.58830237],
            'd2.y2': [12.05848815],
            'obj_cmp.obj': [28.58830817],
            'con_cmp1.con1': [-22.42830237],
            'con_cmp2.con2': [-11.94151185]
        }

        expected_solver_residuals = None

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps)

    def test_record_solver_nonlinear_nonlinear_run_once(self):
        prob = SellarProblem(nonlinear_solver=om.NonlinearRunOnce)
        prob.setup()

        prob.model.nonlinear_solver.add_recorder(self.recorder)

        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        # No norms so no expected norms
        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NLRunOnce', (0,)]

        expected_abs_error = None
        expected_rel_error = None

        expected_solver_output = {
            '_auto_ivc.v1': [1.],
            '_auto_ivc.v0': [5., 2.],
            'd1.y1': [27.8],
            'd2.y2': [12.27257053],
            'obj_cmp.obj': [30.80000468],
            'con_cmp1.con1': [-24.64],
            'con_cmp2.con2': [-11.72742947]
        }

        expected_solver_residuals = None

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps)

    def test_record_solver_linear(self):
        # adding a recorder to a linear solver should raise an error
        prob = SellarProblem()
        prob.setup()

        nl = prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

        linear_solvers = [
            om.DirectSolver, om.ScipyKrylov, om.PETScKrylov,
            om.LinearBlockGS, om.LinearRunOnce, om.LinearBlockJac
        ]

        for solver in linear_solvers:
            try:
                ln = nl.linear_solver = solver()
            except RuntimeError as err:
                if str(err).endswith('PETSc is not available.'):
                    continue
                else:
                    raise err

            with self.assertRaises(RuntimeError) as cm:
                ln.add_recorder(self.recorder)

            self.assertEqual(str(cm.exception), 'Recording is not supported on Linear Solvers.')

    def test_record_driver_system_solver(self):
        # Test what happens when all three types are recorded: Driver, System, and Solver

        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=om.NonlinearRunOnce)
        prob.setup(mode='rev')

        driver = prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)

        #
        # Add recorders
        #

        # Driver
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.add_recorder(self.recorder)

        # System
        cc = prob.model.con_cmp1
        cc.recording_options['record_inputs'] = True
        cc.recording_options['record_outputs'] = True
        cc.recording_options['record_residuals'] = True
        cc.add_recorder(self.recorder)

        # Solver
        nl = prob.model.mda.nonlinear_solver = om.NonlinearBlockGS()
        nl.recording_options['record_abs_error'] = True
        nl.recording_options['record_rel_error'] = True
        nl.recording_options['record_solver_residuals'] = True
        nl.add_recorder(self.recorder)

        t0, t1 = run_driver(prob)

        prob.cleanup()

        #
        # Driver recording test
        #
        coordinate = [0, 'ScipyOptimize_SLSQP', (6, )]

        expected_desvars = {
            "z": prob['z'],
            "x": prob['x']
        }
        expected_objectives = {
            "obj_cmp.obj": prob['obj_cmp.obj']
        }
        expected_constraints = {
            "con_cmp1.con1": prob['con_cmp1.con1'],
            "con_cmp2.con2": prob['con_cmp2.con2']
        }

        expected_outputs = expected_desvars
        expected_outputs.update(expected_objectives)
        expected_outputs.update(expected_constraints)

        expected_driver_data = ((coordinate, (t0, t1), expected_outputs, None, None),)
        assertDriverIterDataRecorded(self, expected_driver_data, self.eps)

        #
        # System recording test
        #
        coordinate = [0, 'ScipyOptimize_SLSQP', (2, ), 'root._solve_nonlinear', (2, ), 'NLRunOnce', (0, ),
                      'con_cmp1._solve_nonlinear', (2, )]

        expected_inputs = {"con_cmp1.y1": [7.73433739], }
        expected_outputs = {"con_cmp1.con1": [-4.57433739], }
        expected_residuals = {"con_cmp1.con1": [0.0], }

        expected_system_data = (
            (coordinate, (t0, t1), expected_inputs, expected_outputs, expected_residuals),
        )
        assertSystemIterDataRecorded(self, expected_system_data, self.eps)

        #
        # Solver recording test
        #
        coordinate = [0, 'ScipyOptimize_SLSQP', (6, ), 'root._solve_nonlinear', (6, ), 'NLRunOnce', (0, ),
                      'mda._solve_nonlinear', (6, ), 'NonlinearBlockGS', (4, )]

        expected_abs_error = 0.0,
        expected_rel_error = 0.0,

        expected_solver_output = {
            "mda.d2.y2": [3.75610187],
            "mda.d1.y1": [3.16],
        }

        expected_solver_residuals = {
            "mda.d2.y2": [0.0],
            "mda.d1.y1": [0.0],
        }

        expected_solver_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                                 expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_solver_data, self.eps)

    def test_global_counter(self):
        # The case recorder maintains a global counter across all recordings

        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=om.NonlinearRunOnce)
        prob.setup(mode='rev')

        driver = prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)

        # Add recorders for Driver, System, Solver
        driver.add_recorder(self.recorder)
        prob.model.add_recorder(self.recorder)
        prob.model.mda.nonlinear_solver.add_recorder(self.recorder)

        t0, t1 = run_driver(prob)

        prob.cleanup()

        # get global counter values from driver, system, and solver recording
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        cur.execute("SELECT counter FROM driver_iterations")
        counters_driver = set(i[0] for i in cur.fetchall())
        cur.execute("SELECT counter FROM system_iterations")
        counters_system = set(i[0] for i in cur.fetchall())
        cur.execute("SELECT counter FROM solver_iterations")
        counters_solver = set(i[0] for i in cur.fetchall())
        cur.execute("SELECT COUNT(rowid) FROM global_iterations")
        global_iterations_records = cur.fetchone()[0]
        con.close()

        # Check to see that they make sense
        self.assertEqual(self.recorder._counter, global_iterations_records)
        self.assertEqual(self.recorder._counter, len(counters_driver) + len(counters_system) +
                         len(counters_solver))
        self.assertTrue(counters_driver.isdisjoint(counters_system))
        self.assertTrue(counters_driver.isdisjoint(counters_solver))
        self.assertTrue(counters_system.isdisjoint(counters_solver))

    def test_implicit_component(self):
        from openmdao.core.tests.test_impl_comp import QuadraticLinearize, QuadraticJacVec

        indeps = om.IndepVarComp()
        indeps.add_output('a', 1.0)
        indeps.add_output('b', 1.0)
        indeps.add_output('c', 1.0)

        group = om.Group()
        group.add_subsystem('comp1', indeps)
        group.add_subsystem('comp2', QuadraticLinearize())
        group.add_subsystem('comp3', QuadraticJacVec())
        group.connect('comp1.a', 'comp2.a')
        group.connect('comp1.b', 'comp2.b')
        group.connect('comp1.c', 'comp2.c')
        group.connect('comp1.a', 'comp3.a')
        group.connect('comp1.b', 'comp3.b')
        group.connect('comp1.c', 'comp3.c')

        prob = om.Problem(model=group)
        prob.setup()

        prob['comp1.a'] = 1.
        prob['comp1.b'] = -4.
        prob['comp1.c'] = 3.

        comp2 = prob.model.comp2  # ImplicitComponent
        comp2.add_recorder(self.recorder)

        t0, t1 = run_driver(prob)

        prob.cleanup()

        coordinate = [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ), 'NLRunOnce', (0, ),
                      'comp2._solve_nonlinear', (0, )]

        expected_inputs = {
            "comp2.a": [1.0, ],
            "comp2.b": [-4.0, ],
            "comp2.c": [3.0, ],
        }
        expected_outputs = {"comp2.x": [3.0, ], }
        expected_residuals = {"comp2.x": [0.0, ], }

        expected_data = (
            (coordinate, (t0, t1), expected_inputs, expected_outputs, expected_residuals),
        )
        assertSystemIterDataRecorded(self, expected_data, self.eps)

    def test_multidimensional_arrays(self):
        # component TestExplCompArray, put in a model and run it; its outputs are multi-d-arrays.
        from openmdao.test_suite.components.expl_comp_array import TestExplCompArray
        comp = TestExplCompArray(thickness=1.)
        prob = om.Problem(comp).setup()

        prob['lengths'] = 3.
        prob['widths'] = 2.

        comp.add_recorder(self.recorder)
        comp.recording_options['record_inputs'] = True
        comp.recording_options['record_outputs'] = True
        comp.recording_options['record_residuals'] = True

        t0, t1 = run_driver(prob)

        prob.cleanup()

        # coordinate = rank0:._solve_nonlinear | 0
        coordinate = [0, 'Driver', (0,), '._solve_nonlinear', (0,)]

        expected_inputs = {
            'lengths': [[3., 3.], [3., 3.]],
            'widths': [[2., 2.], [2., 2.]],
        }

        expected_outputs = {
            'total_volume': [24.],
            'areas': [[6., 6.], [6., 6.]],
        }

        expected_residuals = {
            'total_volume': [0.],
            'areas': [[0., 0.], [0., 0.]],
        }

        expected_data = (
            (coordinate, (t0, t1), expected_inputs, expected_outputs, expected_residuals),
        )
        assertSystemIterDataRecorded(self, expected_data, self.eps)

    def test_record_system_recursively(self):
        # Test adding recorders to all Systems using the recurse option to add_recorder

        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=om.NonlinearRunOnce)
        prob.setup(mode='rev')

        # Need to do recursive adding of recorders AFTER setup
        prob.model.add_recorder(self.recorder, recurse=True)

        prob.model.mda.nonlinear_solver.options['use_apply_nonlinear'] = True

        prob.run_model()
        prob.cleanup()

        # Just make sure all Systems had some metadata recorded
        assertSystemMetadataIdsRecorded(self, [
            'root',
            '_auto_ivc',
            'mda',
            'mda.d1',
            'mda.d2',
            'obj_cmp',
            'con_cmp1',
            'con_cmp2'
        ])

        # Make sure all the Systems are recorded
        assertSystemIterCoordsRecorded(self, [
            'rank0:root._solve_nonlinear|0',
            'rank0:root._solve_nonlinear|0|NLRunOnce|0|con_cmp1._solve_nonlinear|0',
            'rank0:root._solve_nonlinear|0|NLRunOnce|0|con_cmp2._solve_nonlinear|0',
            'rank0:root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0',
            'rank0:root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|0|mda.d1._solve_nonlinear|0',
            'rank0:root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|0|mda.d2._solve_nonlinear|0',
            'rank0:root._solve_nonlinear|0|NLRunOnce|0|obj_cmp._solve_nonlinear|0',
            'rank0:root._solve_nonlinear|0|NLRunOnce|0|_auto_ivc._solve_nonlinear|0',
        ])

    def test_record_system_with_prefix(self):
        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=om.NonlinearRunOnce)
        prob.setup(mode='rev')

        prob.model.mda.nonlinear_solver.options['use_apply_nonlinear'] = True

        prob.model.add_recorder(self.recorder, recurse=True)

        prob.run_model('Run1')
        prob.run_model('Run2')
        prob.cleanup()

        # Just make sure all Systems had some metadata recorded
        assertSystemMetadataIdsRecorded(self, [
            'root',
            '_auto_ivc',
            'mda',
            'mda.d1',
            'mda.d2',
            'obj_cmp',
            'con_cmp1',
            'con_cmp2'
        ])

        # Make sure all the Systems are recorded at least once
        assertSystemIterCoordsRecorded(self, [
            'Run1_rank0:root._solve_nonlinear|0',
            'Run1_rank0:root._solve_nonlinear|0|NLRunOnce|0|con_cmp1._solve_nonlinear|0',
            'Run1_rank0:root._solve_nonlinear|0|NLRunOnce|0|con_cmp2._solve_nonlinear|0',
            'Run1_rank0:root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0',
            'Run1_rank0:root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|0|mda.d1._solve_nonlinear|0',
            'Run1_rank0:root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|0|mda.d2._solve_nonlinear|0',
            'Run1_rank0:root._solve_nonlinear|0|NLRunOnce|0|obj_cmp._solve_nonlinear|0',
            'Run1_rank0:root._solve_nonlinear|0|NLRunOnce|0|_auto_ivc._solve_nonlinear|0',

            'Run2_rank0:root._solve_nonlinear|0',
            'Run2_rank0:root._solve_nonlinear|0|NLRunOnce|0|con_cmp1._solve_nonlinear|0',
            'Run2_rank0:root._solve_nonlinear|0|NLRunOnce|0|con_cmp2._solve_nonlinear|0',
            'Run2_rank0:root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0',
            'Run2_rank0:root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|0|mda.d1._solve_nonlinear|0',
            'Run2_rank0:root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|0|mda.d2._solve_nonlinear|0',
            'Run2_rank0:root._solve_nonlinear|0|NLRunOnce|0|obj_cmp._solve_nonlinear|0',
            'Run2_rank0:root._solve_nonlinear|0|NLRunOnce|0|_auto_ivc._solve_nonlinear|0',
        ])

    def test_driver_recording_with_system_vars(self):
        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=om.NonlinearRunOnce)

        driver = prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['record_inputs'] = False
        driver.recording_options['includes'] = ['y2']
        driver.add_recorder(self.recorder)

        prob.setup()

        t0, t1 = run_driver(prob)

        prob.cleanup()

        # Driver recording test
        coordinate = [0, 'ScipyOptimize_SLSQP', (6, )]

        expected_desvars = {
            "z": prob['z'],
            "x": prob['x']
        }
        expected_objectives = {
            "obj_cmp.obj": prob['obj_cmp.obj']
        }

        expected_constraints = {
            "con_cmp1.con1": prob['con_cmp1.con1'],
            "con_cmp2.con2": prob['con_cmp2.con2']
        }

        expected_sysincludes = {
            'mda.d2.y2': prob['mda.d2.y2']
        }

        expected_outputs = expected_desvars
        expected_outputs.update(expected_objectives)
        expected_outputs.update(expected_constraints)
        expected_outputs.update(expected_sysincludes)

        expected_data = ((coordinate, (t0, t1), expected_outputs, None, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_recorder_file_already_exists_no_append(self):
        prob = SellarProblem()

        driver = prob.driver
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = False
        driver.recording_options['record_constraints'] = False
        driver.recording_options['includes'] = []
        driver.add_recorder(self.recorder)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        # Open up a new instance of the recorder but with the same filename
        prob = SellarProblem()

        driver = prob.driver
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = False
        driver.recording_options['record_constraints'] = False
        driver.recording_options['includes'] = []
        driver.add_recorder(om.SqliteRecorder(self.filename))

        prob.setup()
        t0, t1 = run_driver(prob)
        prob.cleanup()

        # Do a simple test to see if recording second time was OK
        coordinate = [0, 'Driver', (0, )]

        expected_outputs = {"x": [1.0, ], "z": [5.0, 2.0]}

        expected_data = ((coordinate, (t0, t1), expected_outputs, None, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_recorder_cleanup(self):
        def assert_closed(self, recorder):
            try:
                recorder.connection.execute("SELECT * FROM metadata;")
            except sqlite3.ProgrammingError as err:
                self.assertEqual(str(err), 'Cannot operate on a closed database.')
            else:
                self.fail('SqliteRecorder database was not closed.')

        prob = SellarProblem(SellarStateConnection)
        prob.setup()

        driver = prob.driver
        system = prob.model.sub
        solver = prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS()

        # create 3 different recorders
        driver_recorder = om.SqliteRecorder('driver_cases.sql')
        system_recorder = om.SqliteRecorder('system_cases.sql')
        solver_recorder = om.SqliteRecorder('solver_cases.sql')

        # add recorders
        driver.add_recorder(driver_recorder)
        system.add_recorder(system_recorder)
        solver.add_recorder(solver_recorder)

        # run
        prob.run_driver()

        # check recorders have been added
        self.assertTrue(driver._rec_mgr._recorders == [driver_recorder])
        self.assertTrue(system._rec_mgr._recorders == [system_recorder])
        self.assertTrue(solver._rec_mgr._recorders == [solver_recorder])

        # check recorders are active
        self.assertTrue(driver_recorder.connection is not None)
        self.assertTrue(system_recorder.connection is not None)
        self.assertTrue(solver_recorder.connection is not None)

        # cleanup
        prob.cleanup()

        # check recorders are closed
        assert_closed(self, driver_recorder)
        assert_closed(self, system_recorder)
        assert_closed(self, solver_recorder)

        # check recorders are removed
        self.assertFalse(driver._rec_mgr.has_recorders())
        self.assertFalse(system._rec_mgr.has_recorders())
        self.assertFalse(solver._rec_mgr.has_recorders())

    def test_problem_record_no_voi(self):
        prob = om.Problem(model=SellarDerivatives())

        prob.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()

        prob.record('final')
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        problem_cases = cr.list_cases('problem')
        self.assertEqual(len(problem_cases), 1)

        final_case = cr.get_case('final')

        # we didn't declare any VOIs
        desvars = final_case.get_design_vars()
        objectives = final_case.get_objectives()
        constraints = final_case.get_constraints()

        self.assertEqual(len(desvars), 0)
        self.assertEqual(len(objectives), 0)
        self.assertEqual(len(constraints), 0)

        # by default we should get all outputs
        self.assertEqual(set(final_case.outputs.keys()),
                         {'con1', 'con2', 'obj', 'x', 'y1', 'y2', 'z'})

    def test_problem_record_iteration_deprecated(self):
        prob = om.Problem(model=SellarDerivatives())

        prob.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()

        msg = "'Problem.record_iteration' has been deprecated. Use 'Problem.record' instead."

        with assert_warning(DeprecationWarning, msg):
            prob.record_iteration('final')
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        # Just do some simple tests to make sure things were recorded
        problem_cases = cr.list_cases('problem')
        self.assertEqual(len(problem_cases), 1)

        final_case = cr.get_case('final')

        # by default we should get all outputs
        self.assertEqual(set(final_case.outputs.keys()),
                         {'con1', 'con2', 'obj', 'x', 'y1', 'y2', 'z'})

    def test_problem_record_with_options(self):
        prob = om.Problem(model=SellarDerivatives())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.recording_options['record_objectives'] = False
        prob.recording_options['record_constraints'] = False
        prob.recording_options['record_desvars'] = False

        prob.setup()
        prob.run_driver()

        prob.record('final')
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        problem_cases = cr.list_cases('problem')
        self.assertEqual(len(problem_cases), 1)

        final_case = cr.get_case('final')

        desvars = final_case.get_design_vars()
        objectives = final_case.get_objectives()
        constraints = final_case.get_constraints()

        self.assertEqual(len(desvars), 2)
        self.assertEqual(len(objectives), 1)
        self.assertEqual(len(constraints), 2)

        # includes all outputs (default) minus the VOIs, which we have excluded
        self.assertEqual(set(final_case.outputs.keys()), {'x', 'y1', 'z', 'con1', 'y2', 'obj', 'con2'})

    def test_problem_record_inputs_outputs_residuals(self):
        prob = om.Problem(model=SellarDerivatives())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.add_recorder(om.SqliteRecorder("cases.sql"))
        prob.recording_options['includes'] = ['*']
        prob.recording_options['record_inputs'] = True
        prob.recording_options['record_outputs'] = True
        prob.recording_options['record_residuals'] = True

        prob.setup()
        prob.run_driver()

        prob.record('final')
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        problem_cases = cr.list_cases('problem')
        self.assertEqual(len(problem_cases), 1)

        final_case = cr.get_case('final')

        desvars = final_case.get_design_vars()
        objectives = final_case.get_objectives()
        constraints = final_case.get_constraints()

        self.assertEqual(len(desvars), 2)
        self.assertEqual(len(objectives), 1)
        self.assertEqual(len(constraints), 2)

        # includes all outputs (default) minus the VOIs, which we have excluded
        self.assertEqual(set(final_case.outputs.keys()), {'con2', 'z', 'con1', 'y1', 'x', 'y2', 'obj'})
        self.assertEqual(set(final_case.inputs.keys()), {'y1', 'x', 'y2', 'z'})
        self.assertEqual(set(final_case.residuals.keys()), {'con2', 'z', 'con1', 'y1', 'x', 'y2', 'obj'})
        self.assertAlmostEqual(final_case.inputs['d2.y1'][0], 25.58830236987513)
        self.assertAlmostEqual(final_case.outputs['con2'][0], -11.94151184938868)
        self.assertAlmostEqual(final_case.residuals['con2'][0], -1.3036682844358438e-11)
        # self.assertAlmostEqual(final_case.outputs['circuit.R1.I'][0], 0.09908047)

    def test_problem_record_inputs(self):

        # By default you should not get any inputs recorded
        prob = ParaboloidProblem()
        prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)
        prob.add_recorder(self.recorder)
        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()
        prob.record('case1')
        cr = om.CaseReader(self.filename)
        final_case = cr.get_case('case1')
        self.assertEqual(final_case.residuals, None)

        # Turn on inputs recording for Problem
        prob.recording_options['record_inputs'] = True
        prob.setup()
        prob.run_driver()
        prob.record('case2')
        cr = om.CaseReader(self.filename)
        final_case = cr.get_case('case2')
        self.assertEqual(set(final_case.inputs.keys()), {'y', 'x'})
        self.assertAlmostEqual(final_case.inputs['comp.y'][0], -7.833333333333334)

        # Default is includes = ['*'] and excludes = []

        # Run again with excludes.
        prob.recording_options['excludes'] = ['*y']
        prob.setup()
        prob.run_driver()
        prob.record('case3')
        cr = om.CaseReader(self.filename)
        final_case = cr.get_case('case3')
        self.assertEqual(set(final_case.inputs.keys()), {'x'})

        # Run again with includes.
        prob.recording_options['excludes'] = []
        prob.recording_options['includes'] = ['*y']
        prob.setup()
        prob.run_driver()
        prob.record('case4')
        cr = om.CaseReader(self.filename)
        final_case = cr.get_case('case4')
        self.assertEqual(set(final_case.inputs.keys()), {'y'})

        # run again with record_residuals = False
        prob.recording_options['includes'] = ['*']
        prob.recording_options['excludes'] = []
        prob.recording_options['record_residuals'] = False
        prob.setup()
        prob.run_driver()
        prob.record('case5')
        prob.cleanup()
        cr = om.CaseReader(self.filename)
        final_case = cr.get_case('case5')
        self.assertEqual(final_case.residuals, None)

    def test_problem_record_outputs(self):

        prob = ParaboloidProblem()
        prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)
        prob.recording_options['record_residuals'] = True
        # driver.recording_options['includes'] = ['*']
        # driver.recording_options['excludes'] = ['y*']
        prob.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()

        prob.record('final')
        cr = om.CaseReader(self.filename)
        final_case = cr.get_case('final')
        self.assertEqual(set(final_case.residuals.keys()), {'f_xy', 'y', 'x', 'c'})
        self.assertAlmostEqual(final_case.residuals['f_xy'][0], 0.0)

        # run again with includes and excludes
        prob.recording_options['excludes'] = ['f*']
        prob.recording_options['includes'] = ['x*']
        prob.setup()
        prob.run_driver()
        prob.record('final2')
        cr = om.CaseReader(self.filename)
        final_case = cr.get_case('final2')
        self.assertEqual(set(final_case.residuals.keys()), {'x'})

        # run again with record_residuals = False
        prob.recording_options['includes'] = ['*']
        prob.recording_options['excludes'] = []
        prob.recording_options['record_residuals'] = False
        prob.setup()
        prob.run_driver()
        prob.record('final3')
        prob.cleanup()
        cr = om.CaseReader(self.filename)
        final_case = cr.get_case('final3')
        self.assertEqual(final_case.residuals, None)

    def test_problem_record_residuals(self):

        prob = ParaboloidProblem()
        prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)
        prob.recording_options['record_residuals'] = True
        # driver.recording_options['includes'] = ['*']
        # driver.recording_options['excludes'] = ['y*']
        prob.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()

        prob.record('final')
        cr = om.CaseReader(self.filename)
        final_case = cr.get_case('final')
        self.assertEqual(set(final_case.residuals.keys()), {'f_xy', 'y', 'x', 'c'})
        self.assertAlmostEqual(final_case.residuals['f_xy'][0], 0.0)

        # run again with includes and excludes
        prob.recording_options['excludes'] = ['f*']
        prob.recording_options['includes'] = ['x*']
        prob.setup()
        prob.run_driver()
        prob.record('final2')
        cr = om.CaseReader(self.filename)
        final_case = cr.get_case('final2')
        self.assertEqual(set(final_case.residuals.keys()), {'x'})

        # run again with record_residuals = False
        prob.recording_options['includes'] = ['*']
        prob.recording_options['excludes'] = []
        prob.recording_options['record_residuals'] = False
        prob.setup()
        prob.run_driver()
        prob.record('final3')
        prob.cleanup()
        cr = om.CaseReader(self.filename)
        final_case = cr.get_case('final3')
        self.assertEqual(final_case.residuals, None)

    def test_problem_record_solver_data(self):
        prob = SellarProblem()
        prob.setup()

        recorder = om.SqliteRecorder("cases.sql")
        prob.add_recorder(recorder)
        prob.recording_options['includes'] = ['*']
        prob.recording_options['record_abs_error'] = True
        prob.recording_options['record_rel_error'] = True
        prob.recording_options['record_residuals'] = True

        # Just for comparison, see what values you get from recording
        #  the top level solver
        nl = prob.model.nonlinear_solver
        nl.options['use_apply_nonlinear'] = True
        nl.add_recorder(recorder)
        nl.recording_options['record_abs_error'] = True
        nl.recording_options['record_rel_error'] = True
        nl.recording_options['record_solver_residuals'] = True

        prob.run_driver()

        prob.record('final')
        prob.cleanup()

        # get the cases from the problem and solver recording
        cr = om.CaseReader("cases.sql")
        final_case = cr.get_case('final')
        root_solver_cases = cr.list_cases('root.nonlinear_solver', recurse=False)
        last_root_solver_case = cr.get_case(root_solver_cases[-1])

        # Check the errors both the value from the problem recording and
        #   make sure it is the same as the solver last case
        self.assertAlmostEqual(final_case.abs_err, 0.0)
        self.assertAlmostEqual(final_case.rel_err, 0.0)
        self.assertEqual(final_case.abs_err, last_root_solver_case.abs_err)
        self.assertEqual(final_case.rel_err, last_root_solver_case.rel_err)

        # check the residuals are the same from the problem and solver recording
        model_residuals = final_case.residuals
        solver_residuals = last_root_solver_case.residuals
        for key in model_residuals.keys():
            assert_equal_arrays(model_residuals[key], solver_residuals[key] )

    def test_driver_record_outputs(self):

        prob = ParaboloidProblem()
        driver = prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)
        driver.recording_options['record_desvars'] = False
        driver.recording_options['record_objectives'] = False
        driver.recording_options['record_constraints'] = False
        driver.recording_options['record_inputs'] = False
        driver.recording_options['record_outputs'] = True
        driver.recording_options['record_residuals'] = False
        driver.recording_options['includes'] = ['*']
        driver.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)

        coordinate = [0, 'ScipyOptimize_SLSQP', (4, )]

        expected_outputs = {
            "p2.y": -7.83333333,
            "con.c": -15.,
            "p1.x": 7.16666667,
            "comp.f_xy": -27.08333333
        }

        expected_data = ((coordinate, (t0, t1), expected_outputs, None, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

        # run again with includes and excludes
        self.recorder.delete_recordings()

        driver.recording_options['excludes'] = ['f*']
        driver.recording_options['includes'] = ['x*']
        prob.setup()
        t0, t1 = run_driver(prob)

        expected_outputs = {
            "p1.x": 7.16666667,
        }
        expected_data = ((coordinate, (t0, t1), expected_outputs, None, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)


        # run again with record_residuals = False
        self.recorder.delete_recordings()
        driver.recording_options['includes'] = ['*']
        driver.recording_options['excludes'] = []
        driver.recording_options['record_outputs'] = False
        prob.setup()
        t0, t1 = run_driver(prob)
        prob.cleanup()

        expected_data = ((coordinate, (t0, t1), None, None, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_driver_record_residuals(self):

        prob = ParaboloidProblem()
        driver = prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)
        driver.recording_options['record_desvars'] = False
        driver.recording_options['record_objectives'] = False
        driver.recording_options['record_constraints'] = False
        driver.recording_options['record_inputs'] = False
        driver.recording_options['record_outputs'] = False
        driver.recording_options['record_residuals'] = True
        driver.recording_options['includes'] = ['*']
        driver.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)

        coordinate = [0, 'ScipyOptimize_SLSQP', (4, )]

        expected_residuals = {
            "p2.y": 0.0,
            "con.c": 0.0,
            "p1.x": 0.0,
            "comp.f_xy": 0.0
        }

        expected_data = ((coordinate, (t0, t1), None, None, expected_residuals),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)


        # run again with includes and excludes
        self.recorder.delete_recordings()

        driver.recording_options['excludes'] = ['f*']
        driver.recording_options['includes'] = ['x*']
        prob.setup()
        t0, t1 = run_driver(prob)

        expected_residuals = {
            "p1.x": 0.0,
        }
        expected_data = ((coordinate, (t0, t1), None, None, expected_residuals),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)


        # run again with record_residuals = False
        self.recorder.delete_recordings()
        driver.recording_options['includes'] = ['*']
        driver.recording_options['excludes'] = []
        driver.recording_options['record_residuals'] = False
        prob.setup()
        t0, t1 = run_driver(prob)
        prob.cleanup()

        expected_data = ((coordinate, (t0, t1), None, None, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_problem_record_options_includes(self):
        prob = om.Problem(model=SellarDerivatives())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.recording_options['includes'] = []

        prob.setup()
        prob.run_driver()

        prob.record('final')
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        problem_cases = cr.list_cases('problem')
        self.assertEqual(len(problem_cases), 1)

        final_case = cr.get_case('final')

        desvars = final_case.get_design_vars()
        objectives = final_case.get_objectives()
        constraints = final_case.get_constraints()

        self.assertEqual(len(desvars), 2)
        self.assertEqual(len(objectives), 1)
        self.assertEqual(len(constraints), 2)

        # includes no outputs except the the VOIs that are recorded by default
        self.assertEqual(set(final_case.outputs.keys()),
                         {'con1', 'con2', 'obj', 'x', 'z'})

    def test_problem_recording_derivatives(self):
        prob = ParaboloidProblem()

        prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)
        prob.recording_options['record_derivatives'] = True
        prob.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)
        case_name = "state1"
        prob.record(case_name)
        prob.cleanup()

        expected_derivs = {
            "comp.f_xy!p1.x": np.array([[0.5]]),
            "comp.f_xy!p2.y": np.array([[-0.5]]),
            "con.c!p1.x": np.array([[-1.0]]),
            "con.c!p2.y": np.array([[1.0]])
        }

        expected_data = ((case_name, (t0, t1), expected_derivs),)
        assertProblemDerivDataRecorded(self, expected_data, self.eps)

    def test_problem_recording_derivatives_option_false(self):
        prob = ParaboloidProblem()

        prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)
        # By default the option record_derivatives is False
        prob.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)
        case_name = "state1"
        prob.record(case_name)
        prob.cleanup()

        expected_derivs = None
        expected_data = ((case_name, (t0, t1), expected_derivs),)
        assertProblemDerivDataRecorded(self, expected_data, self.eps)

    def test_problem_recording_derivatives_no_voi(self):

        prob = om.Problem(model=SellarDerivatives())

        prob.recording_options['record_derivatives'] = True
        prob.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)

        case_name = "state1"
        prob.record(case_name)

        prob.cleanup()

        cr = om.CaseReader(self.filename)

        problem_cases = cr.list_cases('problem')
        self.assertEqual(len(problem_cases), 1)

        # No desvars or responses given so cannot compute total derivs
        expected_derivs = None

        expected_data = ((case_name, (t0, t1), expected_derivs),)
        assertProblemDerivDataRecorded(self, expected_data, self.eps)

    def test_simple_paraboloid_scaled_desvars(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_constraints'] = True

        recorder = om.SqliteRecorder("cases.sql")
        prob.driver.add_recorder(recorder)

        ref = 5.0
        ref0 = -5.0
        model.add_design_var('x', lower=-50.0, upper=50.0, ref=ref, ref0=ref0)
        model.add_design_var('y', lower=-50.0, upper=50.0, ref=ref, ref0=ref0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup(check=False, mode='fwd')

        prob.set_solver_print(level=0)
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        # Test values from one case, the last case
        driver_cases = cr.list_cases('driver')

        last_case = cr.get_case(driver_cases[-1])

        des_vars = last_case.get_design_vars(scaled=False)
        unscaled_x = des_vars['x'][0]
        unscaled_y = des_vars['y'][0]

        des_vars = last_case.get_design_vars(scaled=True)
        scaled_x = des_vars['x'][0]
        scaled_y = des_vars['y'][0]

        adder, scaler = determine_adder_scaler(ref0, ref, None, None)
        self.assertAlmostEqual((unscaled_x + adder) * scaler, scaled_x, places=12)
        self.assertAlmostEqual((unscaled_y + adder) * scaler, scaled_y, places=12)

    def test_problem_record_before_final_setup(self):
        prob = om.Problem()
        prob.add_recorder(self.recorder)
        prob.setup()

        with self.assertRaises(RuntimeError) as cm:
            prob.record('initial')

        self.assertEqual(str(cm.exception),
                         "Problem: Problem.record() cannot be called before "
                         "`Problem.run_model()`, `Problem.run_driver()`, or "
                         "`Problem.final_setup()`.")

        prob.cleanup()


@use_tempdirs
class TestFeatureSqliteRecorder(unittest.TestCase):

    def test_feature_simple_driver_recording(self):
        import openmdao.api as om
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = om.Problem()

        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        driver = prob.driver = om.ScipyOptimizeDriver()
        driver.options['optimizer'] = 'SLSQP'
        driver.options['tol'] = 1e-9

        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True

        case_recorder_filename = 'cases.sql'

        recorder = om.SqliteRecorder(case_recorder_filename)
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(case_recorder_filename)
        case = cr.get_case('rank0:ScipyOptimize_SLSQP|4')

        assert_near_equal(case.outputs['x'], 7.16666667, 1e-6)
        assert_near_equal(case.outputs['y'], -7.83333333, 1e-6)

    def test_feature_problem_metadata(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = om.Problem(SellarDerivatives())

        recorder = om.SqliteRecorder("cases.sql")
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        # access list of connections stored in metadata
        connections = sorted(cr.problem_metadata['connections_list'], key=lambda x: (x['tgt'], x['src']))

        self.assertEqual('\n'.join([conn['src']+'\t'+conn['tgt'] for conn in connections]),
                         '\n'.join(["d1.y1\tcon_cmp1.y1",
                                    "d2.y2\tcon_cmp2.y2",
                                    "_auto_ivc.v1\td1.x",
                                    "d2.y2\td1.y2",
                                    "_auto_ivc.v0\td1.z",
                                    "d1.y1\td2.y1",
                                    "_auto_ivc.v0\td2.z",
                                    "_auto_ivc.v1\tobj_cmp.x",
                                    "d1.y1\tobj_cmp.y1",
                                    "d2.y2\tobj_cmp.y2",
                                    "_auto_ivc.v0\tobj_cmp.z"]))

        # access the model tree stored in metadata
        self.assertEqual(set(cr.problem_metadata['tree'].keys()),
                         {'name', 'type', 'class', 'expressions', 'component_type',
                          'subsystem_type', 'is_parallel', 'linear_solver', 'linear_solver_options',
                          'nonlinear_solver', 'nonlinear_solver_options', 'children', 'options'})

        self.assertEqual(cr.problem_metadata['tree']['name'], 'root')
        self.assertEqual(sorted([child["name"] for child in cr.problem_metadata['tree']["children"]]),
                         ['_auto_ivc', 'con_cmp1', 'con_cmp2', 'd1', 'd2', 'obj_cmp'])

    def test_feature_problem_metadata_with_driver_information(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = om.Problem(SellarDerivatives())
        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        # DOE
        driver = prob.driver = om.DOEDriver(om.UniformGenerator())
        recorder = om.SqliteRecorder("cases.sql")
        prob.driver.add_recorder(recorder)
        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")
        metadata = cr.problem_metadata['driver']
        self.assertEqual(set(metadata.keys()), {'name', 'type', 'options', 'opt_settings'})
        self.assertEqual(metadata['name'], 'DOEDriver')
        self.assertEqual(metadata['type'], 'doe')
        self.assertEqual(metadata['options'], {'debug_print': [], 'generator': 'UniformGenerator',
                                               'run_parallel': False, 'procs_per_model': 1})

        # Optimization
        driver = prob.driver = om.ScipyOptimizeDriver()
        recorder = om.SqliteRecorder("cases.sql")
        driver.options['optimizer'] = 'SLSQP'
        driver.options['tol'] = 1e-3
        driver.opt_settings['maxiter'] = 1000
        prob.driver.add_recorder(recorder)
        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")
        metadata = cr.problem_metadata['driver']
        self.assertEqual(set(metadata.keys()), {'name', 'type', 'options', 'opt_settings'})
        self.assertEqual(metadata['name'], 'ScipyOptimizeDriver')
        self.assertEqual(metadata['type'], 'optimization')
        self.assertEqual(metadata['options'], {"debug_print": [], "optimizer": "SLSQP",
                                               "tol": 1e-03, "maxiter": 200, "disp": True})
        self.assertEqual(metadata['opt_settings'], {"maxiter": 1000})

    def test_feature_solver_metadata(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = om.Problem(model=SellarDerivatives())
        prob.setup()

        # create recorder
        recorder = om.SqliteRecorder("cases.sql")

        # add recorder to the nonlinear solver for the model
        prob.model.nonlinear_solver = om.NonlinearBlockGS()
        prob.model.nonlinear_solver.add_recorder(recorder)

        # add recorder to the nonlinear solver for Component 'd1'
        d1 = prob.model.d1
        d1.nonlinear_solver = om.NonlinearBlockGS()
        d1.nonlinear_solver.options['maxiter'] = 5
        d1.nonlinear_solver.add_recorder(recorder)

        prob.run_model()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        metadata = cr.solver_metadata

        self.assertEqual(sorted(metadata.keys()), [
            'd1.NonlinearBlockGS', 'root.NonlinearBlockGS'
        ])
        self.assertEqual(metadata['d1.NonlinearBlockGS']['solver_options']['maxiter'], 5)
        self.assertEqual(metadata['root.NonlinearBlockGS']['solver_options']['maxiter'], 10)

    def test_feature_recording_system_options(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = om.Problem(model=SellarDerivatives())

        prob.setup()

        # create recorder and attach to driver and model
        recorder = om.SqliteRecorder("cases.sql")
        prob.driver.add_recorder(recorder)
        prob.model.add_recorder(recorder)

        prob.run_model()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        # options for all the systems in the model
        options = cr.list_model_options()

        self.assertEqual(sorted(options.keys()),
                         sorted(['root']))

        # options for system 'root'
        self.assertEqual(options['root']['ln_maxiter'], None)

    def test_feature_system_recording_options(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = om.Problem(model=SellarDerivatives())
        prob.setup()

        recorder = om.SqliteRecorder("cases.sql")

        obj_cmp = prob.model.obj_cmp
        obj_cmp.add_recorder(recorder)
        obj_cmp.recording_options['includes'] = ['*']
        obj_cmp.recording_options['excludes'] = ['obj_cmp.x']

        prob.model.nonlinear_solver.options['use_apply_nonlinear'] = True

        prob.run_model()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        system_cases = cr.list_cases('root.obj_cmp')

        self.assertEqual(len(system_cases), prob.model._nonlinear_solver._iter_count)

        case = cr.get_case(system_cases[0])

        self.assertEqual(sorted(case.inputs.keys()), ['y1', 'y2', 'z'])

    def test_feature_basic_case_recording(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar_feature import SellarMDAWithUnits
        import numpy as np

        # build the model
        prob = om.Problem(model=SellarMDAWithUnits())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        # setup the optimization
        driver = prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        # Create a recorder variable
        recorder = om.SqliteRecorder('cases.sql')
        # Attach a recorder to the problem
        prob.add_recorder(recorder)

        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()
        prob.record("after_run_driver")

        # Instantiate your CaseReader
        cr = om.CaseReader("cases.sql")
        # Isolate "problem" as your source
        driver_cases = cr.list_cases('problem')
        # Get the first case from the recorder
        case = cr.get_case('after_run_driver')

        # These options will give outputs as the model sees them
        # Gets value but will not convert units
        const = case['con1']

        # get_val can convert your result's units if desired
        const_K = case.get_val("con1", units='K')

        assert_near_equal(const, -1.68550507e-10, 1e-3)
        assert_near_equal(const_K, 273.15, 1e-3)

        # list_outputs will list your model's outputs and return a list of them too
        print(case.list_outputs())

        # This code below will find all the objectives, design variables, and constraints that the
        # problem source contains
        objectives = case.get_objectives()
        design_vars = case.get_design_vars()
        constraints = case.get_constraints()

        assert_near_equal(objectives['obj'], 3.18339395, 1e-4)
        assert_near_equal(design_vars['x'], 0., 1e-4)
        assert_near_equal(constraints['con1'], -1.68550507e-10, 1e-4)

    def test_feature_driver_options(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        import numpy as np

        prob = om.Problem(model=SellarDerivatives())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        driver = prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9)
        driver.recording_options['includes'] = ['*']
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_inputs'] = True
        driver.recording_options['record_outputs'] = True
        driver.recording_options['record_residuals'] = True

        recorder = om.SqliteRecorder("cases.sql")
        driver.add_recorder(recorder)

        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        driver_cases = cr.list_cases('driver')

        last_case = cr.get_case(driver_cases[-1])

        objectives = last_case.get_objectives()
        design_vars = last_case.get_design_vars()
        constraints = last_case.get_constraints()

        assert_near_equal(objectives['obj'], prob['obj'], 1e-1)
        assert_near_equal(design_vars['x'], prob['x'], 1e-1)
        assert_near_equal(constraints['con1'], prob['con1'], 1e-1)
        assert_near_equal(last_case.inputs['obj_cmp.x'], prob['x'])
        assert_near_equal(last_case.outputs['z'], prob['z'])
        assert_near_equal(last_case.residuals['obj'], 0.0, tolerance = 1e-10)

    def test_feature_solver_options(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = om.Problem(model=SellarDerivatives())
        prob.setup()

        recorder = om.SqliteRecorder("cases.sql")

        solver = prob.model.nonlinear_solver
        solver.add_recorder(recorder)
        solver.recording_options['record_abs_error'] = True
        solver.options['use_apply_nonlinear'] = True

        prob.run_model()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        solver_cases = cr.list_cases('root.nonlinear_solver')

        self.assertEqual(len(solver_cases), solver._iter_count)

        case = cr.get_case(solver_cases[0])

        self.assertAlmostEqual(case.abs_err, 2.2545141)

    def test_feature_circuit_with_recorder(self):
        import openmdao.api as om
        from openmdao.test_suite.scripts.circuit_analysis import Resistor, Diode, Node

        class Circuit(om.Group):

            def setup(self):
                self.add_subsystem('n1', Node(n_in=1, n_out=2),
                                   promotes_inputs=[('I_in:0', 'I_in')])
                self.add_subsystem('n2', Node())  # leaving defaults

                self.add_subsystem('R1', Resistor(R=100.), promotes_inputs=[('V_out', 'Vg')])
                self.add_subsystem('R2', Resistor(R=10000.))
                self.add_subsystem('D1', Diode(), promotes_inputs=[('V_out', 'Vg')])

                self.connect('n1.V', ['R1.V_in', 'R2.V_in'])
                self.connect('R1.I', 'n1.I_out:0')
                self.connect('R2.I', 'n1.I_out:1')

                self.connect('n2.V', ['R2.V_out', 'D1.V_in'])
                self.connect('R2.I', 'n2.I_in:0')
                self.connect('D1.I', 'n2.I_out:0')

                self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
                self.nonlinear_solver.options['iprint'] = 2
                self.nonlinear_solver.options['maxiter'] = 20
                self.linear_solver = om.DirectSolver()

        prob = om.Problem()

        model = prob.model
        model.add_subsystem('ground', om.IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', om.IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        recorder = om.SqliteRecorder("cases.sql")
        prob.driver.add_recorder(recorder)
        prob.driver.recording_options['includes'] = ['*']
        prob.setup()

        # set some initial guesses
        prob['circuit.n1.V'] = 10.
        prob['circuit.n2.V'] = 1.

        prob.run_driver()

        prob.cleanup()

        # create the case reader
        cr = om.CaseReader("cases.sql")

        # grab the data recorded in the first driver iteration
        driver_cases = cr.list_cases('driver')
        first_driver_case = cr.get_case(driver_cases[0])

        self.assertAlmostEqual(first_driver_case.inputs['circuit.R1.V_in'][0], 9.90804734)
        self.assertAlmostEqual(first_driver_case.outputs['circuit.R1.I'][0], 0.09908047)

    def test_feature_load_system_case_for_restart(self):
        #######################################################################
        # Do the initial optimization run
        #######################################################################
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        import numpy as np

        prob = om.Problem(model=SellarDerivatives())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        recorder = om.SqliteRecorder('cases.sql')
        model.add_recorder(recorder)

        model.recording_options['record_inputs'] = True
        model.recording_options['record_outputs'] = True
        model.recording_options['record_residuals'] = True
        model.recording_options['options_excludes'] = ['*']

        driver = prob.driver = om.ScipyOptimizeDriver()
        driver.options['optimizer'] = 'SLSQP'
        driver.options['tol'] = 1e-9
        driver.options['disp'] = False

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        #######################################################################
        # Assume that the optimization given above failed before it finished.
        # To debug the problem, we can run the script again, but this time using
        # the last recorded case as a starting point.
        #######################################################################
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = om.Problem(model=SellarDerivatives())
        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        model.recording_options['record_inputs'] = True
        model.recording_options['record_outputs'] = True
        model.recording_options['record_residuals'] = True
        model.recording_options['options_excludes'] = ['*']

        prob.driver = om.ScipyOptimizeDriver()
        driver = prob.driver
        driver.options['optimizer'] = 'SLSQP'
        driver.options['tol'] = 1e-9
        driver.options['disp'] = False

        prob.setup()

        cr = om.CaseReader('cases.sql')

        # Load the last case written
        last_case = cr.get_case(-1)
        prob.load_case(last_case)

        prob.run_driver()
        prob.cleanup()

    def test_feature_record_with_prefix(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = om.Problem(model=SellarDerivatives())
        prob.setup()

        recorder = om.SqliteRecorder("cases.sql", record_viewer_data=False)
        prob.model.add_recorder(recorder)
        prob.driver.add_recorder(recorder)

        prob.set_solver_print(0)

        prob.run_model(case_prefix='Model_Run1')
        prob.run_driver(case_prefix='Driver_Run1')

        prob.run_model('Model_Run2')
        prob.run_driver('Driver_Run2')

        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        # all cases recorded by the root system
        model_cases = cr.list_cases('root', recurse=False)
        self.assertEqual('\n'.join(model_cases), '\n'.join([
            'Model_Run1_rank0:root._solve_nonlinear|0',
            'Driver_Run1_rank0:Driver|0|root._solve_nonlinear|0',
            'Model_Run2_rank0:root._solve_nonlinear|0',
            'Driver_Run2_rank0:Driver|0|root._solve_nonlinear|0'
        ]))

        # all cases recorded by the driver
        driver_cases = cr.list_cases('driver', recurse=False)
        self.assertEqual('\n'.join(driver_cases), '\n'.join([
            'Driver_Run1_rank0:Driver|0',
            'Driver_Run2_rank0:Driver|0'
        ]))

    def test_feature_problem_record(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        import numpy as np

        prob = om.Problem(model=SellarDerivatives())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9)

        prob.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.recording_options['includes'] = []
        prob.recording_options['record_objectives'] = True
        prob.recording_options['record_constraints'] = True
        prob.recording_options['record_desvars'] = True

        prob.setup()
        prob.run_driver()
        prob.record('final')
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        # get list of cases recorded on problem
        problem_cases = cr.list_cases('problem')
        self.assertEqual(problem_cases, ['final'])

        # get list of output variables recorded on problem
        problem_vars = cr.list_source_vars('problem', out_stream=None)
        self.assertEqual(sorted(problem_vars['outputs']), ['con1', 'con2', 'obj', 'x', 'z'])

        # get the recorded case and check values
        case = cr.get_case('final')

        objectives = case.get_objectives()
        design_vars = case.get_design_vars()
        constraints = case.get_constraints()

        assert_near_equal(objectives['obj'], 3.18, 1e-1)

        assert_near_equal(design_vars, case.get_design_vars(), 1e-1)
        assert_near_equal(constraints, case.get_constraints(), 1e-1)

    def test_scaling_multiple_calls(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivatives

        import numpy as np

        scaler = 2.

        prob = om.Problem(model=SellarDerivatives())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj', scaler=scaler)
        model.add_constraint('con1', upper=0.0, scaler=scaler)
        model.add_constraint('con2', upper=0.0, scaler=scaler)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9)

        prob.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.recording_options['includes'] = []
        prob.recording_options['record_objectives'] = True
        prob.recording_options['record_constraints'] = True
        prob.recording_options['record_desvars'] = True

        prob.setup()
        prob.run_driver()
        prob.record('final')
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        # get list of cases recorded on problem
        problem_cases = cr.list_cases('problem')
        self.assertEqual(problem_cases, ['final'])

        # get list of output variables recorded on problem
        problem_vars = cr.list_source_vars('problem', out_stream=None)
        self.assertEqual(sorted(problem_vars['outputs']), ['con1', 'con2', 'obj', 'x', 'z'])

        # get the recorded case and check values
        case = cr.get_case('final')

        objectives = case.get_objectives()
        design_vars = case.get_design_vars()
        constraints = case.get_constraints()

        # Methods are called a second time
        assert_near_equal(objectives['obj'], case.get_objectives()['obj'], 1e-1)
        assert_near_equal(design_vars, case.get_design_vars(), 1e-1)
        assert_near_equal(constraints, case.get_constraints(), 1e-1)

    def test_recorder_resetup(self):
        vec_size = 7
        prob = om.Problem(model=om.Group())

        class _TestSys(om.Group):

            def initialize(self):
                self.options.declare('vec_size', types=int)

            def setup(self):
                nn = self.options['vec_size']

                ivc = self.add_subsystem('ivc', subsys=om.IndepVarComp(), promotes_outputs=['*'])
                ivc.add_output('x', shape=(nn,), units=None)

                self.add_subsystem('mag',
                                   subsys=om.ExecComp('y=x**2',
                                                      y={'shape': (nn,)},
                                                      x={'shape': (nn,)}),
                                   promotes_inputs=['*'], promotes_outputs=['*'])

                self.add_subsystem('sum',
                                   subsys=om.ExecComp('z=sum(y)',
                                                      y={'shape': (nn,)},
                                                      z={'shape': (1,)}),
                                   promotes_inputs=['*'], promotes_outputs=['*'])

                self.add_design_var('x', lower=0, upper=100)
                self.add_objective('z')

        test_sys = prob.model.add_subsystem('test_sys', subsys=_TestSys(vec_size=vec_size))

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9)

        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.driver.recording_options['includes'] = ['*y*']
        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['record_desvars'] = True

        prob.setup()

        prob.set_val('test_sys.x', np.random.rand(vec_size))

        prob.run_driver()

        y0 = prob.get_val('test_sys.y')

        test_sys.options['vec_size'] = 10

        prob.setup()

        prob.set_val('test_sys.x', np.random.rand(test_sys.options['vec_size']))

        prob.run_driver()

        y1 = prob.get_val('test_sys.y')

        case = om.CaseReader('cases.sql').get_case(-1)

        y_recorded = case.get_val('test_sys.y')

        assert_near_equal(y_recorded, y1)


class TestFeatureAdvancedExample(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar_feature import SellarMDAWithUnits
        import numpy as np
        import matplotlib.pyplot as plt

        # build the model
        prob = om.Problem(model=SellarMDAWithUnits())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        # setup the optimization
        driver = prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        # Here we show how to attach recorders to each of the four objects; problem, driver, solver, and system
        # Create a recorder variable
        recorder = om.SqliteRecorder('cases.sql')
        # Attach a recorder to the problem
        prob.add_recorder(recorder)
        # Attach a recorder to the driver
        driver.add_recorder(recorder)

        prob.setup()

        # To attach a recorder to the system, you need to call it after `setup` so the model hierarchy has been generated
        obj_cmp = prob.model.obj_cmp
        obj_cmp.add_recorder(recorder)
        # Attach a recorder to the solver
        model.cycle.add_recorder(recorder)

        prob.set_solver_print(0)
        prob.run_driver()
        prob.record("final_state")
        prob.cleanup()

    def test_feature_system_recorder(self):
        import openmdao.api as om

        # Instantiate your CaseReader
        cr = om.CaseReader("cases.sql")

        system_cases = cr.list_cases('root.obj_cmp')

        # Number of cases in the optimization
        num_cases = len(system_cases)
        print("Number of cases:", num_cases)

        # Get the keys of all the inputs to the obj_func
        case = cr.get_case(system_cases[0])
        self.assertEqual(list(case.inputs.keys()), ['x', 'y1', 'y2', 'z'])

        for i in range(num_cases):
            case = cr.get_case(system_cases[i])
            print(case['y1'])

    def test_feature_solver_recorder(self):
        import openmdao.api as om

        # Instantiate your CaseReader
        cr = om.CaseReader("cases.sql")

        solver_cases = cr.list_cases('root.cycle')

        num_cases = len(solver_cases)
        print("Number of cases:", num_cases)

        case = cr.get_case(solver_cases[3])
        assert_near_equal(case['y1'], 4.17430704, 1e-8)
        assert_near_equal(case['y2'], 4.28622419, 1e-8)

    def test_feature_driver_recorder(self):
        import openmdao.api as om

        # Instantiate your CaseReader
        cr = om.CaseReader("cases.sql")

        driver_cases = cr.list_cases('driver')

        last_case = cr.get_case(driver_cases[-1])

        objectives = last_case.get_objectives()
        design_vars = last_case.get_design_vars()
        constraints = last_case.get_constraints()

        assert_near_equal(objectives['obj'], 3.18339395, 1e-8)
        assert_near_equal(design_vars['x'], 0., 1e-8)
        assert_near_equal(design_vars['z'][0], 1.97763888, 1e-8)
        assert_near_equal(design_vars['z'][1], 1.25035459e-15, 1e-8)
        assert_near_equal(constraints['con1'], -1.68550507e-10, 1e-8)
        assert_near_equal(constraints['con2'], -20.24472223, 1e-8)


    def test_feature_problem_recorder(self):
        import openmdao.api as om

        # Instantiate your CaseReader
        cr = om.CaseReader("cases.sql")

        # get list of cases recorded on problem
        problem_cases = cr.list_cases('problem')
        self.assertEqual(problem_cases, ['final_state'])

        # get list of output variables recorded on problem
        problem_vars = cr.list_source_vars('problem', out_stream=None)
        self.assertEqual(sorted(problem_vars['outputs']),
                         ['con1', 'con2', 'obj', 'x', 'y1', 'y2', 'z'])

        # get the recorded case and check values
        case = cr.get_case('final_state')

        objectives = case.get_objectives()
        design_vars = case.get_design_vars()
        constraints = case.get_constraints()

        assert_near_equal(objectives['obj'], 3.18339395, 1e-8)

    def test_feature_plot_des_vars(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import openmdao.api as om

        # Instantiate your CaseReader
        cr = om.CaseReader("cases.sql")
        driver_cases = cr.list_cases('driver')

        dv_x_values = []
        dv_z_values = []
        for i in range(len(driver_cases)):
            last_case = cr.get_case(driver_cases[i])
            design_vars = last_case.get_design_vars()
            if design_vars:
                dv_x_values.append(design_vars['x'])
                dv_z_values.append(design_vars['z'])

        # Below is a short script to see the path the design variables took to convergence

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
        ax1.plot(np.arange(len(dv_x_values)), np.array(dv_x_values))

        ax1.set(xlabel='Iterations', ylabel='Design Var: X', title='Optimization History')
        ax1.grid()

        ax2.plot(np.arange(len(dv_z_values)), np.array(dv_z_values))

        ax2.set(xlabel='Iterations', ylabel='Design Var: Z', title='Optimization History')
        ax2.grid()
        # There are two lines in the right plot because "Z" contains two variables that are being
        # optimized


@use_tempdirs
class TestFeatureBasicRecording(unittest.TestCase):

    def setUp(self):
        self.record_cases()

    def record_cases(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar_feature import SellarMDA

        import numpy as np

        # create our Sellar problem
        prob = om.Problem(model=SellarMDA())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.driver = om.ScipyOptimizeDriver(disp=False)

        # create a case recorder
        recorder = om.SqliteRecorder('cases.sql')

        # add the recorder to the driver so driver iterations will be recorded
        prob.driver.add_recorder(recorder)

        # add the recorder to the problem so we can manually save a case
        prob.add_recorder(recorder)

        # perform setup and run the problem
        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()

        # record the final state of the problem
        prob.record('final')

        # clean up and shut down
        prob.cleanup()

    def test_read_cases(self):
        import openmdao.api as om

        # open database of previously saved cases
        cr = om.CaseReader("cases.sql")

        # get a list of cases that were recorded by the driver
        driver_cases = cr.list_cases('driver')

        self.assertEqual(len(driver_cases), 11)

        # get the first driver case and inspect the variables of interest
        case = cr.get_case(driver_cases[0])

        objectives = case.get_objectives()
        design_vars = case.get_design_vars()
        constraints = case.get_constraints()

        assert_near_equal(objectives['obj'], 28.58, 1e-1)

        assert_near_equal(design_vars, case.get_design_vars(), 1e-1)
        assert_near_equal(constraints, case.get_constraints(), 1e-1)

        # get a list of cases that we manually recorded
        self.assertEqual(cr.list_cases('problem'), ['final'])

        # get the final case and inspect the variables of interest
        case = cr.get_case('final')

        objectives = case.get_objectives()
        design_vars = case.get_design_vars()
        constraints = case.get_constraints()

        assert_near_equal(objectives['obj'], 3.18, 1e-1)

        assert_near_equal(design_vars, case.get_design_vars(), 1e-1)
        assert_near_equal(constraints, case.get_constraints(), 1e-1)


if __name__ == "__main__":
    unittest.main()
