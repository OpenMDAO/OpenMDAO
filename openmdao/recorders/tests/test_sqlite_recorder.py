""" Unit test for the SqliteRecorder. """
import errno
import os
import sqlite3
import time
import unittest
import numpy as np

from shutil import rmtree
from six import iteritems, PY2, PY3
from tempfile import mkdtemp

from openmdao.api import BoundsEnforceLS, NonlinearBlockGS, ArmijoGoldsteinLS, NonlinearBlockJac,\
            NewtonSolver, NonLinearRunOnce, SqliteRecorder, Group, IndepVarComp, ExecComp, \
            DirectSolver, ScipyIterativeSolver, PetscKSP, LinearBlockGS, LinearRunOnce, \
            LinearBlockJac

from openmdao.core.problem import Problem
from openmdao.devtools.testutil import assert_rel_error
from openmdao.utils.record_util import format_iteration_coordinate
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.recorders.sqlite_recorder import format_version, blob_to_array
from openmdao.recorders.recording_iteration_stack import recording_iteration
from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, \
    SellarDis2withDerivatives
from openmdao.test_suite.components.paraboloid import Paraboloid
from sqlite_recorder_test_utils import _assertDriverIterationDataRecorded
from recorder_test_utils import run_driver

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

if PY2:
    import cPickle as pickle
if PY3:
    import pickle

# check that pyoptsparse is installed. if it is, try to use SLSQP.
OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
    optimizers = {'pyoptsparse': pyOptSparseDriver}

def run_driver(problem):
    t0 = time.time()
    problem.run_driver()
    t1 = time.time()

    return t0, t1

def _assertSystemIterationDataRecorded(test, db_cur, expected, tolerance):
    """
        Expected can be from multiple cases.
    """

    # iterate through the cases
    for coord, (t0, t1), inputs_expected, outputs_expected, residuals_expected in expected:
        iter_coord = format_iteration_coordinate(coord)

        # from the database, get the actual data recorded
        db_cur.execute("SELECT * FROM system_iterations WHERE "
                       "iteration_coordinate=:iteration_coordinate",
                       {"iteration_coordinate": iter_coord})
        row_actual = db_cur.fetchone()
        test.assertTrue(row_actual, 'System iterations table does not contain the requested iteration coordinate: "{}"'.format(iter_coord))

        counter, global_counter, iteration_coordinate, timestamp, success, msg, inputs_blob, \
            outputs_blob, residuals_blob = row_actual

        inputs_actual = blob_to_array(inputs_blob)
        outputs_actual = blob_to_array(outputs_blob)
        residuals_actual = blob_to_array(residuals_blob)

        # Does the timestamp make sense?
        test.assertTrue(t0 <= timestamp and timestamp <= t1)

        test.assertEqual(success, 1)
        test.assertEqual(msg, '')

        for vartype, actual, expected in (
            ('inputs', inputs_actual, inputs_expected),
            ('outputs', outputs_actual, outputs_expected),
            ('residuals', residuals_actual, residuals_expected),
        ):

            if expected is None:
                test.assertEqual(actual, np.array(None, dtype=object))
            else:
                # Check to see if the number of values in actual and expected match
                test.assertEqual(len(actual[0]), len(expected))
                for key, value in iteritems(expected):
                    # Check to see if the keys in the actual and expected match
                    test.assertTrue(key in actual[0].dtype.names,
                                    '{} variable not found in actual data '
                                    'from recorder'.format(key))
                    # Check to see if the values in actual and expected match
                    assert_rel_error(test, actual[0][key], expected[key], tolerance)
        return


def _assertSolverIterationDataRecorded(test, db_cur, expected, tolerance):
    """
        Expected can be from multiple cases.
    """

    # iterate through the cases
    for coord, (t0, t1), expected_abs_error, expected_rel_error, expected_output, \
            expected_solver_residuals in expected:

        iter_coord = format_iteration_coordinate(coord)

        # from the database, get the actual data recorded
        db_cur.execute("SELECT * FROM solver_iterations WHERE iteration_coordinate=:iteration_coordinate",
                       {"iteration_coordinate": iter_coord})
        row_actual = db_cur.fetchone()
        test.assertTrue(row_actual, 'Solver iterations table does not contain the requested iteration coordinate: "{}"'.format(iter_coord))

        counter, global_counter, iteration_coordinate, timestamp, success, msg, abs_err, rel_err, \
            output_blob, residuals_blob = row_actual

        output_actual = blob_to_array(output_blob)
        residuals_actual = blob_to_array(residuals_blob)
        # Does the timestamp make sense?
        test.assertTrue(t0 <= timestamp and timestamp <= t1, 'timestamp should be between when the model '
                                                             'started and stopped')

        test.assertEqual(success, 1)
        test.assertEqual(msg, '')
        if expected_abs_error:
            test.assertTrue(abs_err, 'Expected absolute error but none recorded')
            assert_rel_error(test, abs_err, expected_abs_error, tolerance)
        if expected_rel_error:
            test.assertTrue(rel_err, 'Expected relative error but none recorded')
            assert_rel_error(test, rel_err, expected_rel_error, tolerance)

        for vartype, actual, expected in (
                ('outputs', output_actual, expected_output),
                ('residuals', residuals_actual, expected_solver_residuals),
        ):

            if expected is None:
                test.assertEqual(actual, np.array(None, dtype=object))
            else:
                # Check to see if the number of values in actual and expected match
                test.assertEqual(len(actual[0]), len(expected))
                for key, value in iteritems(expected):
                    # Check to see if the keys in the actual and expected match
                    test.assertTrue(key in actual[0].dtype.names, '{} variable not found in actual '
                                                                  'data from recorder'.format(key))
                    # Check to see if the values in actual and expected match
                    assert_rel_error(test, actual[0][key], expected[key], tolerance)
        return


def _assertMetadataRecorded(test, db_cur):

    db_cur.execute("SELECT format_version FROM metadata")
    row = db_cur.fetchone()

    format_version_actual = row[0]
    format_version_expected = format_version

    # this always gets recorded
    test.assertEqual(format_version_actual, format_version_expected)

    return


def _assertDriverMetadataRecorded(test, db_cur, expected):

    db_cur.execute("SELECT model_viewer_data FROM driver_metadata")
    row = db_cur.fetchone()

    if expected is None:
        test.assertEqual(None, row)
        return

    if PY2:
        model_viewer_data = pickle.loads(str(row[0]))
    if PY3:
        model_viewer_data = pickle.loads(row[0])

    test.assertTrue(isinstance(model_viewer_data, dict))

    test.assertEqual(2, len(model_viewer_data))

    test.assertTrue(isinstance(model_viewer_data['connections_list'], list))

    test.assertEqual(expected['connections_list_length'],
                     len(model_viewer_data['connections_list']))

    test.assertEqual(expected['tree_length'], len(model_viewer_data['tree']))

    tr = model_viewer_data['tree']
    test.assertEqual(set(['name', 'type', 'subsystem_type', 'children']), set(tr.keys()))
    test.assertEqual(expected['tree_children_length'], len(model_viewer_data['tree']['children']))

    cl = model_viewer_data['connections_list']
    for c in cl:
        test.assertTrue(set(c.keys()).issubset(set(['src', 'tgt', 'cycle_arrows'])))

    return


class TestSqliteRecorder(unittest.TestCase):
    """
    Features
    --------
    CaseRecorder
    """
    def setUp(self):
        recording_iteration.stack = []
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "sqlite_test")
        self.recorder = SqliteRecorder(self.filename)
        # print(self.filename)  # comment out to make filename printout go away.
        self.eps = 1e-5

    def tearDown(self):
        # return  # comment out to allow db file to be removed.
        try:
            rmtree(self.dir)
            pass
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def assertDriverIterationDataRecorded(self, expected, tolerance):
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        _assertDriverIterationDataRecorded(self, cur, expected, tolerance)
        con.close()

    def assertSystemIterationDataRecorded(self, expected, tolerance):
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        _assertSystemIterationDataRecorded(self, cur, expected, tolerance)
        con.close()

    def assertSystemIterationCoordinatesRecorded(self, iteration_coordinates):
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        for iteration_coordinate in iteration_coordinates:
            cur.execute("SELECT * FROM system_iterations WHERE "
                           "iteration_coordinate=:iteration_coordinate",
                           {"iteration_coordinate": iteration_coordinate})
            row_actual = cur.fetchone()
            self.assertTrue(row_actual,
                'System iterations table does not contain the requested iteration coordinate: "{}"'.\
                            format(iteration_coordinate))

    def assertSolverIterationDataRecorded(self, expected, tolerance):
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        _assertSolverIterationDataRecorded(self, cur, expected, tolerance)
        con.close()

    def assertMetadataRecorded(self):
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        _assertMetadataRecorded(self, cur)
        con.close()

    def assertDriverMetadataRecorded(self, expected_driver_metadata):
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        _assertDriverMetadataRecorded(self, cur, expected_driver_metadata)
        con.close()

    def assertSystemMetadataIdsRecorded(self, ids):
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        for id in ids:
            cur.execute("SELECT * FROM system_metadata WHERE "
                           "id=:id",
                           {"id": id})
            row_actual = cur.fetchone()
            self.assertTrue(row_actual,
                'System metadata table does not contain the requested id: "{}"'.format(id))

    def setup_sellar_model(self):
        self.prob = Problem()

        model = self.prob.model = Group()
        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])
        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])
        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                            z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])
        self.prob.model.nonlinear_solver = NonlinearBlockGS()

        self.prob.model.add_design_var('x', lower=-100, upper=100)
        self.prob.model.add_design_var('z', lower=-100, upper=100)
        self.prob.model.add_objective('obj')
        self.prob.model.add_constraint('con1')
        self.prob.model.add_constraint('con2')

    def setup_sellar_grouped_model(self):
        self.prob = Problem()

        model = self.prob.model = Group()

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        mda = model.add_subsystem('mda', Group(), promotes=['x', 'z', 'y1', 'y2'])
        mda.linear_solver = ScipyIterativeSolver()
        mda.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        mda.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                            z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        mda.nonlinear_solver = NonlinearBlockGS()
        model.linear_solver = ScipyIterativeSolver()

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

    def test_only_desvars_recorded(self):

        self.setup_sellar_model()

        self.recorder.options['record_desvars'] = True
        self.recorder.options['record_responses'] = False
        self.recorder.options['record_objectives'] = False
        self.recorder.options['record_constraints'] = False
        self.prob.driver.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        coordinate = [0, 'Driver', (0, )]

        expected_desvars = {
                            "px.x": [1.0, ],
                            "pz.z": [5.0, 2.0]
                           }

        self.assertDriverIterationDataRecorded(((coordinate, (t0, t1), expected_desvars,
                                           None, None, None),), self.eps)

    def test_add_recorder_after_setup(self):

        self.setup_sellar_model()

        self.recorder.options['record_desvars'] = True
        self.recorder.options['record_responses'] = False
        self.recorder.options['record_objectives'] = False
        self.recorder.options['record_constraints'] = False

        self.prob.setup(check=False)

        self.prob.driver.add_recorder(self.recorder)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        coordinate = [0, 'Driver', (0, )]

        expected_desvars = {
                            "px.x": [1.0, ],
                            "pz.z": [5.0, 2.0]
                           }

        self.assertDriverIterationDataRecorded(((coordinate, (t0, t1), expected_desvars,
                                           None, None, None),), self.eps)

    def test_only_objectives_recorded(self):

        self.setup_sellar_model()

        self.recorder.options['record_desvars'] = False
        self.recorder.options['record_responses'] = False
        self.recorder.options['record_objectives'] = True
        self.recorder.options['record_constraints'] = False
        self.prob.driver.add_recorder(self.recorder)
        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        coordinate = [0, 'Driver', (0, )]

        expected_objectives = {"obj_cmp.obj": [28.58830817, ]}

        self.assertDriverIterationDataRecorded(((coordinate, (t0, t1), None, None,
                                           expected_objectives, None),), self.eps)

    def test_only_constraints_recorded(self):

        self.setup_sellar_model()

        self.recorder.options['record_desvars'] = False
        self.recorder.options['record_responses'] = False
        self.recorder.options['record_objectives'] = False
        self.recorder.options['record_constraints'] = True
        self.prob.driver.add_recorder(self.recorder)
        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        coordinate = [0, 'Driver', (0, )]

        expected_constraints = {
                            "con_cmp1.con1": [-22.42830237, ],
                            "con_cmp2.con2": [-11.94151185, ],
                            }

        self.assertDriverIterationDataRecorded(((coordinate, (t0, t1), None, None, None,
                                           expected_constraints), ), self.eps)

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed" )
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP" )
    def test_simple_driver_recording(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        model.suppress_solver_output = True

        prob.driver = pyOptSparseDriver()

        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_desvars'] = True
        self.recorder.options['record_responses'] = True
        self.recorder.options['record_objectives'] = True
        self.recorder.options['record_constraints'] = True

        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)
        prob.setup(check=False)

        t0, t1 = run_driver(prob)

        prob.cleanup()

        coordinate = [0, 'SLSQP', (3, )]

        expected_desvars = {
                            "p1.x": [7.16706813, ],
                            "p2.y": [-7.83293187, ]
                           }

        expected_objectives = {"comp.f_xy": [-27.0833, ], }

        expected_constraints = {"con.c": [-15.0, ], }

        self.assertDriverIterationDataRecorded(((coordinate, (t0, t1), expected_desvars, None,
                                           expected_objectives, expected_constraints),), self.eps)

    def test_driver_records_metadata(self):
        self.setup_sellar_model()

        self.recorder.options['includes'] = ["p1.x"]
        self.recorder.options['record_metadata'] = True
        self.prob.driver.add_recorder(self.recorder)
        self.prob.setup(check=False)

        # Conclude setup but don't run model.
        self.prob.final_setup()

        self.prob.cleanup()

        self.assertMetadataRecorded()
        expected_driver_metadata = {
            'connections_list_length': 11,
            'tree_length': 4,
            'tree_children_length': 7,
        }
        self.assertDriverMetadataRecorded(expected_driver_metadata)

    def test_driver_doesnt_record_metadata(self):

        self.setup_sellar_model()

        self.recorder.options['record_metadata'] = False
        self.prob.driver.add_recorder(self.recorder)
        self.prob.setup(check=False)

        self.prob.cleanup()

        self.assertMetadataRecorded()
        expected_driver_metadata = None
        self.assertDriverMetadataRecorded(expected_driver_metadata)

    @unittest.skipIf(PETScVector is None or os.environ.get("TRAVIS"),
                     "PETSc is required." if PETScVector is None
                     else "Unreliable on Travis CI.")
    def test_record_system(self):
        self.setup_sellar_model()

        self.recorder.options['record_inputs'] = True
        self.recorder.options['record_outputs'] = True
        self.recorder.options['record_residuals'] = True
        self.recorder.options['record_metadata'] = True

        self.prob.model.add_recorder(self.recorder)

        d1 = self.prob.model.get_subsystem('d1')  # instance of SellarDis1withDerivatives, a Group
        d1.add_recorder(self.recorder)

        obj_cmp = self.prob.model.get_subsystem('obj_cmp')  # an ExecComp
        obj_cmp.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        coordinate = [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ),
                      'NonlinearBlockGS', (6, ), 'd1._solve_nonlinear', (6, )]
        expected_inputs = {
                            "d1.y2": [12.05848815],
                            "d1.z": [5.0, 2.0],
                            "d1.x": [1.0, ],
                          }
        expected_outputs = {"d1.y1": [25.58830237, ], }
        expected_residuals = {"d1.y1": [0.0, ], }
        self.assertSystemIterationDataRecorded(((coordinate, (t0, t1), expected_inputs,
                                                 expected_outputs, expected_residuals),), self.eps)

        coordinate = [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ),
                      'NonlinearBlockGS', (6, ), 'obj_cmp._solve_nonlinear', (6, )]
        expected_inputs = {
                            "obj_cmp.z": [5.0, 2.0],
                            "obj_cmp.y1": [25.58830236, ],
                            "obj_cmp.x": [1.0, ],
                            "obj_cmp.y2": [12.05857185, ],
                          }
        expected_outputs = {"obj_cmp.obj": [28.58830816, ], }
        expected_residuals = {"obj_cmp.obj": [0.0, ], }
        self.assertSystemIterationDataRecorded(((coordinate, (t0, t1), expected_inputs,
                                                 expected_outputs, expected_residuals),), self.eps)

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed" )
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP" )
    def test_includes(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        model.suppress_solver_output = True

        prob.driver = pyOptSparseDriver()

        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_desvars'] = True
        self.recorder.options['record_responses'] = True
        self.recorder.options['record_objectives'] = True
        self.recorder.options['record_constraints'] = True
        self.recorder.options['includes'] = ['*']
        self.recorder.options['excludes'] = ['p2*']

        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.opt_settings['ACC'] = 1e-9

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup(check=False)
        t0, t1 = run_driver(prob)

        prob.cleanup()

        coordinate = [0, 'SLSQP', (5, )]

        expected_desvars = {"p1.x": prob["p1.x"]}

        expected_objectives = {"comp.f_xy": prob['comp.f_xy'], }

        expected_constraints = {"con.c": prob['con.c'], }

        self.assertDriverIterationDataRecorded(((coordinate, (t0, t1), expected_desvars, None,
                                           expected_objectives, expected_constraints), ), self.eps)

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed" )
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP" )
    def test_includes_post_setup(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        model.suppress_solver_output = True

        prob.driver = pyOptSparseDriver()

        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.opt_settings['ACC'] = 1e-9

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup(check=False)

        # Set up recorder after intitial setup.
        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_desvars'] = True
        self.recorder.options['record_responses'] = True
        self.recorder.options['record_objectives'] = True
        self.recorder.options['record_constraints'] = True
        self.recorder.options['includes'] = ['*']
        self.recorder.options['excludes'] = ['p2*']

        t0, t1 = run_driver(prob)

        prob.cleanup()

        coordinate = [0, 'SLSQP', (5, )]

        expected_desvars = {"p1.x": prob["p1.x"]}

        expected_objectives = {"comp.f_xy": prob['comp.f_xy'], }

        expected_constraints = {"con.c": prob['con.c'], }

        self.assertDriverIterationDataRecorded(((coordinate, (t0, t1), expected_desvars, None,
                                                 expected_objectives, expected_constraints), ), self.eps)

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed" )
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP" )
    def test_record_system_with_hierarchy(self):
        self.setup_sellar_grouped_model()

        self.prob.driver = pyOptSparseDriver()
        self.prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            self.prob.driver.opt_settings['ACC'] = 1e-9

        self.recorder.options['record_inputs'] = True
        self.recorder.options['record_outputs'] = True
        self.recorder.options['record_residuals'] = True
        self.recorder.options['record_metadata'] = True

        self.prob.model.add_recorder(self.recorder)

        pz = self.prob.model.get_subsystem('pz')  # IndepVarComp which is an ExplicitComponent
        pz.add_recorder(self.recorder)

        mda = self.prob.model.get_subsystem('mda')  # Group
        d1 = mda.get_subsystem('d1')
        d1.add_recorder(self.recorder)

        self.prob.setup(check=False, mode='rev')

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        coordinate = [0, 'SLSQP', (0, ), 'root._solve_nonlinear', (0, ), 'NLRunOnce', (0, ),
                      'mda._solve_nonlinear', (0, ), 'NonlinearBlockGS', (4,), 'mda.d1._solve_nonlinear', (4, )]
        # Coord: rank0:SLSQP | 0 | NLRunOnce | 0 | NonlinearBlockGS | 4 | mda.d1._solve_nonlinear | 4

        expected_inputs = {
                            "mda.d1.z": [5.0, 2.0],
                            "mda.d1.x": [1.0, ],
                            "mda.d1.y2": [12.0584865, ],
                          }
        expected_outputs = {"mda.d1.y1": [25.5883027, ], }
        expected_residuals = {"mda.d1.y1": [0.0, ], }

        self.assertSystemIterationDataRecorded(((coordinate, (t0, t1), expected_inputs,
                                                 expected_outputs, expected_residuals),), self.eps)

        coordinate = [0, 'SLSQP', (2, ), 'root._solve_nonlinear', (2, ), 'NLRunOnce', (0, ),
                      'pz._solve_nonlinear', (2, )]

        expected_inputs = None
        expected_outputs = {"pz.z": [2.8640616, 0.825643, ], }
        expected_residuals = {"pz.z": [0.0, 0.0], }
        self.assertSystemIterationDataRecorded(((coordinate, (t0, t1), expected_inputs, expected_outputs,
                                                 expected_residuals), ), self.eps)

    def test_record_solver(self):
        self.setup_sellar_model()

        self.recorder.options['record_abs_error'] = True
        self.recorder.options['record_rel_error'] = True
        self.recorder.options['record_solver_output'] = True
        self.recorder.options['record_solver_residuals'] = True
        self.prob.model._nonlinear_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        coordinate = [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ), 'NonlinearBlockGS', (6, )]

        expected_abs_error = 1.31880284470753394998e-10

        expected_rel_error = 3.6299074030587596e-12

        expected_solver_output = {
            "con_cmp1.con1": [-22.42830237000701],
            "d1.y1": [25.58830237000701],
            "con_cmp2.con2": [-11.941511849375644],
            "pz.z": [5.0, 2.0],
            "obj_cmp.obj": [28.588308165163074],
            "d2.y2": [12.058488150624356],
            "px.x": [1.0]
        }

        expected_solver_residuals = {
            "con_cmp1.con1": [0.0],
            "d1.y1": [1.318802844707534e-10],
            "con_cmp2.con2": [0.0],
            "pz.z": [0.0, 0.0],
            "obj_cmp.obj": [0.0],
            "d2.y2": [0.0],
            "px.x": [0.0]
        }

        self.assertSolverIterationDataRecorded(((coordinate, (t0, t1), expected_abs_error,
                                                 expected_rel_error, expected_solver_output,
                                                 expected_solver_residuals),), self.eps)

    def test_record_line_search_armijo_goldstein(self):
        self.setup_sellar_model()

        model = self.prob.model
        model.nonlinear_solver = NewtonSolver()
        model.linear_solver = ScipyIterativeSolver()

        model._nonlinear_solver.options['solve_subsystems'] = True
        model._nonlinear_solver.options['max_sub_solves'] = 4
        ls = model._nonlinear_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='vector')

        # This is pretty bogus, but it ensures that we get a few LS iterations.
        ls.options['c'] = 100.0
        ls.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (3,), 'ArmijoGoldsteinLS', (4,)]
        expected_abs_error = 3.49773898733e-9
        expected_rel_error = expected_abs_error / 2.9086436370499857e-08
        expected_solver_output = None
        expected_solver_residuals = None

        self.assertSolverIterationDataRecorded(((coordinate, (t0, t1), expected_abs_error,
                                                 expected_rel_error, expected_solver_output,
                                                 expected_solver_residuals),), self.eps)

    def test_record_line_search_bounds_enforce(self):
        self.setup_sellar_model()

        model = self.prob.model
        model.nonlinear_solver = NewtonSolver()
        model.linear_solver = ScipyIterativeSolver()

        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 4
        ls = model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='vector')

        ls.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (1,), 'BoundsEnforceLS', (0,)]
        expected_abs_error = 7.02783609310096e-10
        expected_rel_error = 8.078674883382422e-07
        expected_solver_output = None
        expected_solver_residuals = None

        self.assertSolverIterationDataRecorded(((coordinate, (t0, t1), expected_abs_error,
                                                 expected_rel_error, expected_solver_output,
                                                 expected_solver_residuals),), self.eps)

    def test_record_solver_nonlinear_block_gs(self):
        self.setup_sellar_model()

        self.prob.model.nonlinear_solver = NonlinearBlockGS()
        self.prob.model.nonlinear_solver.add_recorder(self.recorder)

        self.recorder.options['record_solver_output'] = True
        self.recorder.options['record_solver_residuals'] = True

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NonlinearBlockGS', (6, )]
        expected_abs_error = 1.31880284470753394998e-10
        expected_rel_error = 3.6299074030587596e-12

        expected_solver_output = {
            'px.x': [1.],
            'pz.z': [5., 2.],
            'd1.y1': [25.58830237],
            'd2.y2': [12.05848815],
            'obj_cmp.obj': [28.58830817],
            'con_cmp1.con1': [-22.42830237],
            'con_cmp2.con2': [-11.94151185]
        }

        expected_solver_residuals = {
            'px.x': [-0.],
            'pz.z': [-0., -0.],
            'd1.y1': [1.31880284e-10],
            'd2.y2': [0.],
            'obj_cmp.obj': [0.],
            'con_cmp1.con1': [0.],
            'con_cmp2.con2': [0.]
        }

        self.assertSolverIterationDataRecorded(((coordinate, (t0, t1), expected_abs_error,
                                                  expected_rel_error, expected_solver_output,
                                                  expected_solver_residuals),), self.eps)

    def test_record_solver_nonlinear_block_jac(self):
        self.setup_sellar_model()

        self.prob.model.nonlinear_solver = NonlinearBlockJac()
        self.prob.model.nonlinear_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NonlinearBlockJac', (9,)]

        expected_abs_error = 7.234027587097439e-07
        expected_rel_error = 1.991112651729199e-08
        expected_solver_residuals = None
        expected_solver_output = None

        self.assertSolverIterationDataRecorded(((coordinate, (t0, t1), expected_abs_error,
                                                 expected_rel_error, expected_solver_output,
                                                 expected_solver_residuals),), self.eps)

    def test_record_solver_nonlinear_newton(self):
        self.setup_sellar_model()

        self.prob.model.nonlinear_solver = NewtonSolver()
        self.prob.model.nonlinear_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (9,)]

        expected_abs_error = 5.041402548755789e-06
        expected_rel_error = 1.3876088080160474e-07
        expected_solver_residuals = None
        expected_solver_output = None

        self.assertSolverIterationDataRecorded(((coordinate, (t0, t1), expected_abs_error,
                                                 expected_rel_error, expected_solver_output,
                                                 expected_solver_residuals),), self.eps)

    def test_record_solver_nonlinear_nonlinear_run_once(self):
        self.setup_sellar_model()

        self.prob.model.nonlinear_solver = NonLinearRunOnce()
        self.prob.model.nonlinear_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        # No norms so no expected norms
        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NLRunOnce', (0,)]
        expected_abs_error = None
        expected_rel_error = None
        expected_solver_residuals = None
        expected_solver_output = None

        self.assertSolverIterationDataRecorded(((coordinate, (t0, t1), expected_abs_error,
                                                 expected_rel_error, expected_solver_output,
                                                 expected_solver_residuals),), self.eps)

    def test_record_solver_linear_direct_solver(self):

        self.setup_sellar_model()

        self.prob.model.nonlinear_solver = NewtonSolver()
        # used for analytic derivatives
        self.prob.model.nonlinear_solver.linear_solver = DirectSolver()

        self.recorder.options['record_abs_error'] = True
        self.recorder.options['record_rel_error'] = True
        self.recorder.options['record_solver_output'] = True
        self.recorder.options['record_solver_residuals'] = True
        self.prob.model.nonlinear_solver.linear_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)
        t0, t1 = run_driver(self.prob)

        # No norms so no expected norms
        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (2,), 'DirectSolver', (0,)]

        expected_abs_error = None
        expected_rel_error = None

        expected_solver_output = {
            'px.x': [0.],
            'pz.z': [0.0, 0.00000000e+00],
            'd1.y1': [0.00045069],
            'd2.y2': [-0.00225346],
            'obj_cmp.obj': [0.00045646],
            'con_cmp1.con1': [-0.00045069],
            'con_cmp2.con2': [-0.00225346]
        }

        expected_solver_residuals = {
            'px.x': [0.],
            'pz.z': [-0., -0.],
            'd1.y1': [0.0],
            'd2.y2': [-0.00229801],
            'obj_cmp.obj': [5.75455956e-06],
            'con_cmp1.con1': [-0.],
            'con_cmp2.con2': [-0.]
        }

        self.assertSolverIterationDataRecorded(((coordinate, (t0, t1), expected_abs_error,
                                                 expected_rel_error, expected_solver_output,
                                                 expected_solver_residuals),), self.eps)

    def test_record_solver_linear_scipy_iterative_solver(self):
        self.setup_sellar_model()

        self.prob.model.nonlinear_solver = NewtonSolver()
        # used for analytic derivatives
        self.prob.model.nonlinear_solver.linear_solver = ScipyIterativeSolver()

        self.recorder.options['record_abs_error'] = True
        self.recorder.options['record_rel_error'] = True
        self.recorder.options['record_solver_output'] = True
        self.recorder.options['record_solver_residuals'] = True
        self.prob.model.nonlinear_solver.linear_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)
        t0, t1 = run_driver(self.prob)

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (2,), 'ScipyIterativeSolver', (1,)]
        expected_abs_error = 0.0
        expected_rel_error = 0.0

        expected_solver_output = {
            'px.x': [0.],
            'pz.z': [0., 0.],
            'd1.y1': [0.0],
            'd2.y2': [-0.41168147],
            'obj_cmp.obj': [-0.48667678],
            'con_cmp1.con1': [0.770496],
            'con_cmp2.con2': [-2.70578793e-06]
        }

        expected_solver_residuals = {
            'px.x': [0.],
            'pz.z': [0., 0.],
            'd1.y1': [-0.08233575],
            'd2.y2': [-0.41168152],
            'obj_cmp.obj': [-0.4866797],
            'con_cmp1.con1': [0.77049654],
            'con_cmp2.con2': [0.41167877]
        }

        self.assertSolverIterationDataRecorded(((coordinate, (t0, t1), expected_abs_error,
                                                 expected_rel_error, expected_solver_output,
                                                 expected_solver_residuals),), self.eps)

    @unittest.skipIf(PETScVector is None or os.environ.get("TRAVIS"),
                     "PETSc is required." if PETScVector is None
                     else "Unreliable on Travis CI.")
    def test_record_solver_linear_petsc_ksp(self):
        self.setup_sellar_model()

        self.prob.model.nonlinear_solver = NewtonSolver()
        # used for analytic derivatives
        self.prob.model.nonlinear_solver.linear_solver = PetscKSP()

        self.recorder.options['record_abs_error'] = True
        self.recorder.options['record_rel_error'] = True
        self.recorder.options['record_solver_output'] = True
        self.recorder.options['record_solver_residuals'] = True
        self.prob.model.nonlinear_solver.linear_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)
        t0, t1 = run_driver(self.prob)

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (2,), 'PetscKSP', (3,)]
        expected_abs_error = 0.0
        expected_rel_error = 0.0

        expected_solver_output = {
            'px.x': [0.],
            'pz.z': [0., 0.],
            'd1.y1': [-7.86357118e-07],
            'd2.y2': [0.00177091],
            'obj_cmp.obj': [0.70719095],
            'con_cmp1.con1': [-0.70702038],
            'con_cmp2.con2': [3.93178559e-06]
        }

        expected_solver_residuals = {
            'px.x': [0.],
            'pz.z': [0., 0.],
            'd1.y1': [0.00035339623935061308],
            'd2.y2': [0.00177099],
            'obj_cmp.obj': [0.70719175],
            'con_cmp1.con1': [-0.70702117],
            'con_cmp2.con2': [-0.00176698]
        }
        self.assertSolverIterationDataRecorded(((coordinate, (t0, t1), expected_abs_error,
                                                 expected_rel_error, expected_solver_output,
                                                 expected_solver_residuals),), self.eps)

    def test_record_solver_linear_block_gs(self):

        # raise unittest.SkipTest("Linear Solver recording not working yet")
        self.setup_sellar_model()

        self.prob.model.nonlinear_solver = NewtonSolver()
        # used for analytic derivatives
        self.prob.model.nonlinear_solver.linear_solver = LinearBlockGS()

        self.recorder.options['record_abs_error'] = True
        self.recorder.options['record_rel_error'] = True
        self.recorder.options['record_solver_output'] = True
        self.recorder.options['record_solver_residuals'] = True
        self.prob.model.nonlinear_solver.linear_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)
        t0, t1 = run_driver(self.prob)

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (2,), 'LinearBlockGS', (6,)]
        expected_abs_error = 9.109083208861876e-11
        expected_rel_error = 9.114367543620551e-12

        expected_solver_output = {
            'px.x': [-0.],
            'pz.z': [-0., -0.],
            'd1.y1': [0.00045069],
            'd2.y2': [-0.00225346],
            'obj_cmp.obj': [0.00045646],
            'con_cmp1.con1': [-0.00045069],
            'con_cmp2.con2': [-0.00225346]
        }

        expected_solver_residuals = {
            'px.x': [0.],
            'pz.z': [0., 0.],
            'd1.y1': [9.10908321e-11],
            'd2.y2': [0.],
            'obj_cmp.obj': [-2.03287907e-20],
            'con_cmp1.con1': [0.],
            'con_cmp2.con2': [0.]
        }

        self.assertSolverIterationDataRecorded(((coordinate, (t0, t1), expected_abs_error,
                                                 expected_rel_error, expected_solver_output,
                                                 expected_solver_residuals),), self.eps)

    def test_record_solver_linear_linear_run_once(self):

        # raise unittest.SkipTest("Linear Solver recording not working yet")
        self.setup_sellar_model()

        self.prob.model.nonlinear_solver = NewtonSolver()
        # used for analytic derivatives
        self.prob.model.nonlinear_solver.linear_solver = LinearRunOnce()

        self.recorder.options['record_abs_error'] = True
        self.recorder.options['record_rel_error'] = True
        self.recorder.options['record_solver_output'] = True
        self.recorder.options['record_solver_residuals'] = True
        self.prob.model.nonlinear_solver.linear_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)
        t0, t1 = run_driver(self.prob)

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (9,), 'LinearRunOnce', (0,)]
        expected_abs_error = None
        expected_rel_error = None

        expected_solver_output = {
            'px.x': [-0.],
            'pz.z': [-0., -0.],
            'd1.y1': [-4.15366975e-05],
            'd2.y2': [-4.10568454e-06],
            'obj_cmp.obj': [-4.15366737e-05],
            'con_cmp1.con1': [4.15366975e-05],
            'con_cmp2.con2': [-4.10568454e-06]
        }

        expected_solver_residuals = {
            'px.x': [-0.],
            'pz.z': [0., 0.],
            'd1.y1': [-4.15366975e-05],
            'd2.y2': [4.10564051e-06],
            'obj_cmp.obj': [-4.15366737e-05],
            'con_cmp1.con1': [-4.15366975e-05],
            'con_cmp2.con2': [-4.10568454e-06]
        }

        self.assertSolverIterationDataRecorded(((coordinate, (t0, t1), expected_abs_error,
                                                 expected_rel_error, expected_solver_output,
                                                 expected_solver_residuals),), self.eps)

    def test_record_solver_linear_block_jac(self):
        # raise unittest.SkipTest("Linear Solver recording not working yet")
        self.setup_sellar_model()

        self.prob.model.nonlinear_solver = NewtonSolver()
        # used for analytic derivatives
        self.prob.model.nonlinear_solver.linear_solver = LinearBlockJac()

        self.recorder.options['record_abs_error'] = True
        self.recorder.options['record_rel_error'] = True
        self.recorder.options['record_solver_output'] = True
        self.recorder.options['record_solver_residuals'] = True
        self.prob.model.nonlinear_solver.linear_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)
        t0, t1 = run_driver(self.prob)

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (3,), 'LinearBlockJac', (9,)]
        expected_abs_error = 9.947388408259769e-11
        expected_rel_error = 4.330301334141486e-08

        expected_solver_output = {
            'px.x': [-0.],
            'pz.z': [-0., -0.],
            'd1.y1': [4.55485639e-09],
            'd2.y2': [-2.27783334e-08],
            'obj_cmp.obj': [-2.28447051e-07],
            'con_cmp1.con1': [2.28461863e-07],
            'con_cmp2.con2': [-2.27742837e-08]
        }

        expected_solver_residuals = {
            'px.x': [-0.],
            'pz.z': [0., 0.],
            'd1.y1': [-2.84055951e-16],
            'd2.y2': [6.93561782e-12],
            'obj_cmp.obj': [7.01674811e-11],
            'con_cmp1.con1': [-7.01674811e-11],
            'con_cmp2.con2': [1.42027975e-15]
        }

        self.assertSolverIterationDataRecorded(((coordinate, (t0, t1), expected_abs_error,
                                                 expected_rel_error, expected_solver_output,
                                                 expected_solver_residuals),), self.eps)

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed" )
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP" )
    def test_record_driver_system_solver(self):

        # Test what happens when all three types are recorded:
        #    Driver, System, and Solver

        self.setup_sellar_grouped_model()

        self.prob.driver = pyOptSparseDriver()
        self.prob.driver.options['optimizer'] = OPTIMIZER
        self.prob.driver.opt_settings['ACC'] = 1e-9

        self.recorder.options['record_metadata'] = True

        # Add recorders
        # Driver
        self.prob.driver.add_recorder(self.recorder)
        # System
        pz = self.prob.model.get_subsystem('pz')  # IndepVarComp which is an ExplicitComponent
        pz.add_recorder(self.recorder)
        # Solver
        mda = self.prob.model.get_subsystem('mda')
        mda.nonlinear_solver.add_recorder(self.recorder)

        # Driver
        self.recorder.options['record_desvars'] = True
        self.recorder.options['record_responses'] = True
        self.recorder.options['record_objectives'] = True
        self.recorder.options['record_constraints'] = True

        # System
        self.recorder.options['record_inputs'] = True
        self.recorder.options['record_outputs'] = True
        self.recorder.options['record_residuals'] = True

        # Solver
        self.recorder.options['record_abs_error'] = True
        self.recorder.options['record_rel_error'] = True
        self.recorder.options['record_solver_output'] = True
        self.recorder.options['record_solver_residuals'] = True

        self.prob.setup(check=False, mode='rev')
        t0, t1 = run_driver(self.prob)
        self.prob.cleanup()

        # Driver recording test
        coordinate = [0, 'SLSQP', (7, )]

        expected_desvars = {
                            "pz.z": self.prob['pz.z'],
                            "px.x": self.prob['px.x']
        }

        expected_objectives = {"obj_cmp.obj": self.prob['obj_cmp.obj'], }

        expected_constraints = {
                                 "con_cmp1.con1": self.prob['con_cmp1.con1'],
                                 "con_cmp2.con2": self.prob['con_cmp2.con2'],
        }

        self.assertDriverIterationDataRecorded(((coordinate, (t0, t1), expected_desvars, None,
                                           expected_objectives, expected_constraints),), self.eps)

        # System recording test
        coordinate = [0, 'SLSQP', (2, ), 'root._solve_nonlinear', (2, ), 'NLRunOnce', (0, ),
                      'pz._solve_nonlinear', (2, )]

        expected_inputs = None
        expected_outputs = {"pz.z": [2.8640616, 0.825643, ], }
        expected_residuals = {"pz.z": [0.0, 0.0], }
        self.assertSystemIterationDataRecorded(((coordinate, (t0, t1), expected_inputs, expected_outputs,
                                                 expected_residuals), ), self.eps)

        # Solver recording test
        coordinate = [0, 'SLSQP', (6, ), 'root._solve_nonlinear', (6, ), 'NLRunOnce', (0, ),
                      'mda._solve_nonlinear', (6, ), 'NonlinearBlockGS', (4, )]

        expected_abs_error = 0.0,

        expected_rel_error = 0.0,

        expected_solver_output = {
            "mda.d2.y2": [3.75527777],
            "mda.d1.y1": [3.16],
        }

        expected_solver_residuals = {
            "mda.d2.y2": [0.0],
            "mda.d1.y1": [0.0],
        }

        self.assertSolverIterationDataRecorded(((coordinate, (t0, t1), expected_abs_error,
                                                 expected_rel_error, expected_solver_output,
                                                 expected_solver_residuals),), self.eps)

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed" )
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP" )
    def test_global_counter(self):

        # The case recorder maintains a global counter across all recordings

        self.setup_sellar_grouped_model()

        self.prob.driver = pyOptSparseDriver()
        self.prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            self.prob.driver.opt_settings['ACC'] = 1e-2  # to speed the test up
            self.prob.driver.opt_settings['ACC'] = 1e-9

        self.recorder.options['record_inputs'] = True
        self.recorder.options['record_outputs'] = True
        self.recorder.options['record_residuals'] = True
        self.recorder.options['record_metadata'] = True

        # Add recorders for Driver, System, Solver
        self.prob.driver.add_recorder(self.recorder)
        self.prob.model.add_recorder(self.recorder)
        mda = self.prob.model.get_subsystem('mda')
        mda.nonlinear_solver.add_recorder(self.recorder)

        self.prob.setup(check=False, mode='rev')
        t0, t1 = run_driver(self.prob)
        self.prob.cleanup()

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
        group = Group()
        group.add_subsystem('comp1', IndepVarComp([('a', 1.0), ('b', 1.0), ('c', 1.0)]))
        group.add_subsystem('comp2', QuadraticLinearize())
        group.add_subsystem('comp3', QuadraticJacVec())
        group.connect('comp1.a', 'comp2.a')
        group.connect('comp1.b', 'comp2.b')
        group.connect('comp1.c', 'comp2.c')
        group.connect('comp1.a', 'comp3.a')
        group.connect('comp1.b', 'comp3.b')
        group.connect('comp1.c', 'comp3.c')

        prob = Problem(model=group)
        prob.setup(check=False)

        prob['comp1.a'] = 1.
        prob['comp1.b'] = -4.
        prob['comp1.c'] = 3.

        comp2 = prob.model.get_subsystem('comp2')  # ImplicitComponent

        self.recorder.options['record_metadata'] = False

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
        self.assertSystemIterationDataRecorded(((coordinate, (t0, t1), expected_inputs,
                                                 expected_outputs, expected_residuals),), self.eps)

    def test_multidimensional_arrays(self):
        # component TestExplCompArray, put in a model and run it; its outputs are multi-d-arrays.
        from openmdao.test_suite.components.expl_comp_array import TestExplCompArray
        comp = TestExplCompArray(thickness=1.)
        prob = Problem(comp).setup(check=False)

        prob['lengths'] = 3.
        prob['widths'] = 2.

        comp.add_recorder(self.recorder)
        self.recorder.options['record_inputs'] = True
        self.recorder.options['record_outputs'] = True
        self.recorder.options['record_residuals'] = True
        self.recorder.options['record_metadata'] = False

        t0, t1 = run_driver(prob)

        prob.run_model()

        # coordinate = rank0:._solve_nonlinear | 0
        coordinate = [0, 'Driver', (0,), '._solve_nonlinear', (0,)]

        expected_inputs = {
            'lengths': [[3.,  3.], [3., 3.]],
            'widths': [[2.,  2.], [2., 2.]],
        }

        expected_outputs = {
            'total_volume': [24.],
            'areas': [[6., 6.], [6., 6.]],
        }

        expected_residuals = {
            'total_volume': [0.],
            'areas': [[0., 0.], [0., 0.]],
        }

        self.assertSystemIterationDataRecorded(((coordinate, (t0, t1), expected_inputs,
                                                 expected_outputs, expected_residuals),), self.eps)

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed" )
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP" )
    def test_record_system_recursively(self):
        # Test adding recorders to all Systems using the recurse option
        #    to add_recorder

        self.setup_sellar_grouped_model()

        self.prob.driver = pyOptSparseDriver()
        self.prob.driver.options['optimizer'] = OPTIMIZER
        self.prob.driver.opt_settings['ACC'] = 1e-9

        self.recorder.options['record_metadata'] = True

        # Add recorder to model and all subsystems
        self.recorder.options['record_inputs'] = True
        self.recorder.options['record_outputs'] = True
        self.recorder.options['record_residuals'] = True

        self.prob.setup(check=False, mode='rev')

        # Need to do recursive adding of recorders AFTER setup
        self.prob.model.add_recorder(self.recorder, recurse=True)

        t0, t1 = run_driver(self.prob)
        self.prob.cleanup()

        # Just make sure all Systems had some metadata recorded
        self.assertSystemMetadataIdsRecorded(
            [
                'root',
                'px',
                'pz',
                'mda',
                'mda.d1',
                'mda.d2',
                'obj_cmp',
                'con_cmp1',
                'con_cmp2'
            ]
        )

        # Make sure all the Systems are recorded at least once
        self.assertSystemIterationCoordinatesRecorded(
            [
            'rank0:SLSQP|0|root._solve_nonlinear|0',
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0|con_cmp1._solve_nonlinear|0',
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0|con_cmp2._solve_nonlinear|0',
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0',
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|0|mda.d1._solve_nonlinear|0',
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|0|mda.d2._solve_nonlinear|0',
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0|obj_cmp._solve_nonlinear|0',
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0|px._solve_nonlinear|0',
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0|pz._solve_nonlinear|0',
            ]
        )

    def test_recorder_file_already_exists_no_append(self):

        self.setup_sellar_model()

        self.recorder.options['record_metadata'] = True
        self.recorder.options['record_desvars'] = True
        self.recorder.options['record_responses'] = False
        self.recorder.options['record_objectives'] = False
        self.recorder.options['record_constraints'] = False
        self.prob.driver.add_recorder(self.recorder)

        self.prob.setup(check=False)
        self.prob.run_driver()
        self.prob.cleanup()

        # Open up a new instance of the recorder but with the same filename
        self.setup_sellar_model()
        recorder = SqliteRecorder(self.filename)
        recorder.options['record_metadata'] = True
        recorder.options['record_desvars'] = True
        recorder.options['record_responses'] = False
        recorder.options['record_objectives'] = False
        recorder.options['record_constraints'] = False
        self.prob.driver.add_recorder(recorder)

        self.prob.setup(check=False)
        t0, t1 = run_driver(self.prob)
        self.prob.cleanup()

        # Do a simple test to see if recording second time was OK
        coordinate = [0, 'Driver', (0, )]

        expected_desvars = {
                            "px.x": [1.0, ],
                            "pz.z": [5.0, 2.0]
                           }

        self.assertDriverIterationDataRecorded(((coordinate, (t0, t1), expected_desvars,
                                           None, None, None),), self.eps)

if __name__ == "__main__":
    unittest.main()
