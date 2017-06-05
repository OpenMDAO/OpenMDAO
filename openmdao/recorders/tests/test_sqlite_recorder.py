""" Unit test for the SqliteRecorder. """

import cPickle
import errno
import os
from shutil import rmtree
from six import iteritems
import sqlite3
from tempfile import mkdtemp
import time
import unittest

import numpy as np

from openmdao.api import SqliteRecorder, Group, IndepVarComp, ExecComp
from openmdao.api import BoundsEnforceLS, NonlinearBlockGS, ArmijoGoldsteinLS, NonlinearBlockJac, NewtonSolver, \
                            NLRunOnce
from openmdao.api import DirectSolver, ScipyIterativeSolver, PetscKSP, LinearBlockGS, LNRunOnce, LinearBlockJac

from openmdao.core.problem import Problem
from openmdao.devtools.testutil import assert_rel_error
from openmdao.utils.record_util import format_iteration_coordinate
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.recorders.sqlite_recorder import format_version, blob_to_array
from openmdao.test_suite.components.sellar import SellarDerivatives, SellarDerivativesGrouped
from openmdao.test_suite.components.double_sellar import DoubleSellar
from openmdao.test_suite.components.paraboloid import Paraboloid

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')
if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver

optimizers = {'pyoptsparse': pyOptSparseDriver}
# optimizers = {'scipy': ScipyOptimizer, }


def run_driver(problem):
    t0 = time.time()
    problem.run_driver()
    t1 = time.time()

    return t0, t1


def _assertIterationDataRecorded(test, db_cur, expected, tolerance):
    """
        expected can be from multiple cases
    """

    # iterate through the cases
    for coord, (t0, t1), desvars_expected, responses_expected, objectives_expected, constraints_expected in expected:
        iter_coord = format_iteration_coordinate(coord)

        # from the database, get the actual data recorded
        db_cur.execute("SELECT * FROM driver_iterations WHERE iteration_coordinate=:iteration_coordinate", {"iteration_coordinate": iter_coord})
        row_actual  = db_cur.fetchone()

        counter, global_counter, iteration_coordinate, timestamp, success, msg, desvars_blob, responses_blob, objectives_blob, constraints_blob = row_actual

        desvars_actual = blob_to_array(desvars_blob)
        responses_actual = blob_to_array(responses_blob)
        objectives_actual = blob_to_array(objectives_blob)
        constraints_actual = blob_to_array(constraints_blob)

        # Does the timestamp make sense?
        test.assertTrue( t0 <= timestamp and timestamp <= t1)

        test.assertEqual(success, 1)
        test.assertEqual(msg, '')

        for vartype, actual, expected in (
            ('desvars', desvars_actual, desvars_expected),
            ('responses', responses_actual, responses_expected),
            ('objectives', objectives_actual, objectives_expected),
            ('constraints', constraints_actual, constraints_expected),
            ):

            if expected is None:
                test.assertEqual(actual, np.array(None, dtype=object))
            else:
                # Check to see if the number of values in actual and expected match
                test.assertEqual(len(actual[0]), len(expected))
                for key, value in iteritems(expected):
                    # Check to see if the keys in the actual and expected match
                    test.assertTrue(key in actual[0].dtype.names, '{} variable not found in actual data from recorder'.format(key))
                    # Check to see if the values in actual and expected match
                    assert_rel_error(test, actual[0][key], expected[key], tolerance)
        return

def _assertSystemIterationDataRecorded(test, db_cur, expected, tolerance):
    """
        expected can be from multiple cases
    """

    # iterate through the cases
    for coord, (t0, t1), inputs_expected, outputs_expected, residuals_expected in expected:
        iter_coord = format_iteration_coordinate(coord)

        # from the database, get the actual data recorded
        db_cur.execute("SELECT * FROM system_iterations WHERE iteration_coordinate=:iteration_coordinate", {"iteration_coordinate": iter_coord})
        row_actual = db_cur.fetchone()

        counter, global_counter, iteration_coordinate, timestamp, success, msg, inputs_blob, outputs_blob, residuals_blob = row_actual

        inputs_actual = blob_to_array(inputs_blob)
        outputs_actual = blob_to_array(outputs_blob)
        residuals_actual = blob_to_array(residuals_blob)

        # Does the timestamp make sense?
        test.assertTrue( t0 <= timestamp and timestamp <= t1)

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
                    test.assertTrue(key in actual[0].dtype.names, '{} variable not found in actual data from recorder'.format(key))
                    # Check to see if the values in actual and expected match
                    assert_rel_error(test, actual[0][key], expected[key], tolerance)
        return

def _assertSolverIterationDataRecorded(test, db_cur, expected, tolerance):
    """
        expected can be from multiple cases
    """

    # iterate through the cases
    for coord, (t0, t1), expected_abs_error, expected_rel_error, expected_output, \
        expected_solver_residuals in expected:

        iter_coord = format_iteration_coordinate(coord)

        # from the database, get the actual data recorded
        db_cur.execute("SELECT * FROM solver_iterations WHERE iteration_coordinate=:iteration_coordinate", {"iteration_coordinate": iter_coord})
        row_actual = db_cur.fetchone()
        # db_cur.execute("SELECT * FROM solver_iterations")
        # rows = db_cur.fetchall()
        test.assertTrue(row_actual, 'Solver iterations table is empty. Should contain at least one record')

        counter, global_counter, iteration_coordinate, timestamp, success, msg, abs_err, rel_err, output_blob, residuals_blob = row_actual

        output_actual = blob_to_array(output_blob)
        residuals_actual = blob_to_array(residuals_blob)
        # Does the timestamp make sense?
        test.assertTrue( t0 <= timestamp and timestamp <= t1, 'timestamp should be between when the model started and stopped')

        test.assertEqual(success, 1)
        test.assertEqual(msg, '')
        if expected_abs_error:
            test.assertTrue( abs_err, 'Expected absolute error but none recorded')
            assert_rel_error(test, abs_err, expected_abs_error, tolerance)
        if expected_rel_error:
            test.assertTrue( rel_err, 'Expected relative error but none recorded')
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
                    test.assertTrue(key in actual[0].dtype.names, '{} variable not found in actual data from recorder'.format(key))
                    # Check to see if the values in actual and expected match
                    assert_rel_error(test, actual[0][key], expected[key], tolerance)
        return

def _assertSolverIterationDataRecordedBasic(test, db_cur):
    """
        Just make sure something was recorded for the solver
    """
    db_cur.execute("SELECT * FROM solver_iterations")
    row_actual = db_cur.fetchone()
    test.assertTrue(row_actual, 'Solver iterations table is empty. Should contain at least one record')

def _assertMetadataRecorded(test, db_cur):

    db_cur.execute("SELECT format_version FROM metadata")
    row  = db_cur.fetchone()

    format_version_actual = row[0]
    format_version_expected = format_version

    # this always gets recorded
    test.assertEqual( format_version_actual, format_version_expected)

    return

def _assertDriverMetadataRecorded(test, db_cur, expected):

    db_cur.execute("SELECT model_viewer_data FROM driver_metadata")
    row = db_cur.fetchone()

    if expected is None:
        test.assertEqual(None,row)
        return

    model_viewer_data = cPickle.loads(str(row[0]))

    test.assertTrue(isinstance(model_viewer_data, dict ) )

    test.assertEqual(2, len( model_viewer_data))

    test.assertTrue(isinstance(model_viewer_data['connections_list'], list ) )

    test.assertEqual(expected['connections_list_length'], len( model_viewer_data['connections_list']))
    test.assertEqual(expected['tree_length'], len( model_viewer_data['tree']))
    tr = model_viewer_data['tree']
    test.assertEqual(set(['name', 'type', 'subsystem_type', 'children']), set(tr.keys()))
    test.assertEqual(expected['tree_children_length'], len( model_viewer_data['tree']['children']))

    cl = model_viewer_data['connections_list']
    for c in cl:
        test.assertTrue(set(c.keys()).issubset(set(['src', 'tgt', 'cycle_arrows'])))

    return


class TestSqliteRecorder(unittest.TestCase):
    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "sqlite_test")
        self.recorder = SqliteRecorder(self.filename)
        self.eps = 1e-5

    def tearDown(self):
        print('self.filename', self.filename)
        return
        try:
            rmtree(self.dir)
            pass
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def assertIterationDataRecorded(self, expected, tolerance):
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        _assertIterationDataRecorded(self, cur, expected, tolerance)
        con.close()

    def assertSystemIterationDataRecorded(self, expected, tolerance):
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        _assertSystemIterationDataRecorded(self, cur, expected, tolerance)
        con.close()

    def assertSolverIterationDataRecorded(self, expected, tolerance):
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        _assertSolverIterationDataRecorded(self, cur, expected, tolerance)
        con.close()

    def assertSolverIterationDataRecordedBasic(self):
        '''Just want to make sure something was recorded'''
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        _assertSolverIterationDataRecordedBasic(self, cur)
        con.close()

    def assertMetadataRecorded(self ):
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        _assertMetadataRecorded(self, cur)
        con.close()

    def assertDriverMetadataRecorded(self, expected_driver_metadata ):
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        _assertDriverMetadataRecorded(self, cur, expected_driver_metadata)
        con.close()

    def setup_sellar_model(self):
        self.prob = Problem()

        self.prob.model = SellarDerivatives()
        self.prob.model.nl_solver = NonlinearBlockGS()

        self.prob.model.add_design_var('x', lower=-100, upper=100)
        self.prob.model.add_design_var('z', lower=-100, upper=100)
        self.prob.model.add_objective('obj')
        self.prob.model.add_constraint('con1')
        self.prob.model.add_constraint('con2')

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

        coordinate = [0, 'Driver', (1, )]

        expected_desvars = {"px.x": [1.0,],
                            "pz.z": [5.0, 2.0]
                            }

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_desvars, None, None, None),), self.eps)

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

        coordinate = [0, 'Driver', (1, )]

        expected_objectives = {"obj_cmp.obj": [28.58830817,]}

        self.assertIterationDataRecorded(((coordinate, (t0, t1), None, None, expected_objectives, None),), self.eps)

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

        coordinate = [0, 'Driver', (1, )]

        expected_constraints = {
                            "con_cmp1.con1": [-22.42830237,],
                            "con_cmp2.con2": [-11.94151185,],
                            }

        self.assertIterationDataRecorded(((coordinate, (t0, t1), None, None, None, expected_constraints),), self.eps)

    # TODO_RECORDERS - need to add tests for recording options for recording Systems and Solvers

    def test_simple_driver_recording(self):

        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

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
        prob.driver.options['print_results'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)
        prob.setup(check=False)

        t0, t1 = run_driver(prob)

        prob.cleanup()

        coordinate = [0, 'SLSQP', (4, )]

        expected_desvars = {"p1.x": [7.16666666166666666666667,],
                            "p2.y": [-7.833333333333334,]
                            }

        expected_objectives = {"comp.f_xy": [-27.0833,],
                            }

        expected_constraints = {"con.c": [-15.0,],
                            }

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_desvars, None, expected_objectives, expected_constraints),), self.eps)

    def test_driver_records_metadata(self):
        self.setup_sellar_model()

        self.recorder.options['includes'] = ["p1.x"]
        self.recorder.options['record_metadata'] = True
        self.prob.driver.add_recorder(self.recorder)
        self.prob.setup(check=False)

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

    def test_record_system(self):
        self.setup_sellar_model()

        self.recorder.options['record_inputs'] = True
        self.recorder.options['record_outputs'] = True
        self.recorder.options['record_residuals'] = True
        self.recorder.options['record_metadata'] = True

        self.prob.model.add_recorder(self.recorder)

        d1 = self.prob.model.get_subsystem('d1')  # an instance of SellarDis1withDerivatives which is a Group
        d1.add_recorder(self.recorder)

        obj_cmp = self.prob.model.get_subsystem('obj_cmp') # an ExecComp
        obj_cmp.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        # TODO_RECORDERS - need to check the recording of the d1 also
        coordinate = [0, 'obj_cmp', (6, )]

        expected_inputs = {
                            "obj_cmp.z": [5.0, 2.0],
                            "obj_cmp.y1": [25.58914915,],
                            "obj_cmp.x": [1.0,],
                            "obj_cmp.y2": [12.05857185,],
                            }

        expected_outputs = {"obj_cmp.obj": [28.58915495,],
                            }

        expected_residuals = {"obj_cmp.obj": [0.0,],
                            }
        self.assertSystemIterationDataRecorded(((coordinate, (t0, t1), expected_inputs, expected_outputs,
                                                 expected_residuals),), self.eps)

    def test_includes(self):
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

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
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup(check=False)
        t0, t1 = run_driver(prob)

        prob.cleanup()

        coordinate = [0, 'SLSQP', (4, )]

        expected_desvars = {"p1.x": [7.16666666166666666666667, ]
                            }

        expected_objectives = {"comp.f_xy": [-27.0833,],
                            }

        expected_constraints = {"con.c": [-15.0,],
                            }

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_desvars, None, expected_objectives, expected_constraints),), self.eps)

    def test_record_system_with_hierarchy(self):
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

        prob = Problem()
        model = prob.model = SellarDerivativesGrouped()

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = True

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        self.recorder.options['record_inputs'] = True
        self.recorder.options['record_outputs'] = True
        self.recorder.options['record_residuals'] = True
        self.recorder.options['record_metadata'] = True

        prob.model.add_recorder(self.recorder)

        pz = prob.model.get_subsystem('pz') # IndepVarComp which is an ExplicitComponent
        pz.add_recorder(self.recorder)

        mda = prob.model.get_subsystem('mda') # Group
        d1 = mda.get_subsystem('d1')
        d1.add_recorder(self.recorder)

        prob.setup(check=False, mode='rev')

        t0, t1 = run_driver(prob)

        prob.cleanup()

        #TODO_RECORDERS - need to test values !!!

    def test_record_solver(self):
        self.setup_sellar_model()

        self.recorder.options['record_abs_error'] = True
        self.recorder.options['record_rel_error'] = True
        self.recorder.options['record_solver_output'] = True
        self.recorder.options['record_solver_residuals'] = True
        self.prob.model.nl_solver = NonlinearBlockGS()
        self.prob.model.nl_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        coordinate = [0, 'NonlinearBlockGS', (7, )]

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

        self.assertSolverIterationDataRecorded(((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                                                 expected_solver_output, expected_solver_residuals),), self.eps)

    def test_record_line_search_armijo_goldstein(self):
        self.setup_sellar_model()

        model = self.prob.model
        model.nl_solver = NewtonSolver()
        model.ln_solver = ScipyIterativeSolver()

        model.nl_solver.options['solve_subsystems'] = True
        model.nl_solver.options['max_sub_solves'] = 4
        ls = model.nl_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='vector')

        # This is pretty bogus, but it ensures that we get a few LS iterations.
        ls.options['c'] = 100.0
        ls.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        # TODO_RECORDERS - should really check to see that we get more than just one record as this test does
        self.assertSolverIterationDataRecordedBasic()

    def test_record_line_search_bounds_enforce(self):
        self.setup_sellar_model()

        model = self.prob.model
        model.nl_solver = NewtonSolver()
        model.ln_solver = ScipyIterativeSolver()

        model.nl_solver.options['solve_subsystems'] = True
        model.nl_solver.options['max_sub_solves'] = 4
        ls = model.nl_solver.linesearch = BoundsEnforceLS(bound_enforcement='vector')

        ls.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        # TODO_RECORDERS - should really check to see that we get more than just one record as this test does
        self.assertSolverIterationDataRecordedBasic()


    def test_record_solver_nonlinear_block_gs(self):
        self.setup_sellar_model()

        self.prob.model.nl_solver = NonlinearBlockGS()
        self.prob.model.nl_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        self.assertSolverIterationDataRecordedBasic()

    def test_record_solver_nonlinear_block_jac(self):
        self.setup_sellar_model()

        self.prob.model.nl_solver = NonlinearBlockJac()
        self.prob.model.nl_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        self.assertSolverIterationDataRecordedBasic()

    def test_record_solver_nonlinear_newton(self):
        self.setup_sellar_model()

        self.prob.model.nl_solver = NewtonSolver()
        self.prob.model.nl_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        self.assertSolverIterationDataRecordedBasic()

    def test_record_solver_nonlinear_nl_run_once(self):
        self.setup_sellar_model()

        self.prob.model.nl_solver = NLRunOnce()
        self.prob.model.nl_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        #### No norms so no expected norms
        self.assertSolverIterationDataRecordedBasic()

    def test_record_solver_linear_direct_solver(self):

        # raise unittest.SkipTest("Linear Solver recording not working yet")

        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.nl_solver = NewtonSolver()
        # used for analytic derivatives
        model.nl_solver.ln_solver = DirectSolver()
        self.recorder.options['record_abs_error'] = True
        self.recorder.options['record_rel_error'] = True
        self.recorder.options['record_solver_output'] = True
        self.recorder.options['record_solver_residuals'] = True
        model.nl_solver.ln_solver.add_recorder(self.recorder)

        prob.setup()
        t0, t1 = run_driver(prob)

        #### No norms so no expected norms

        # TODO_RECORDERS - need to be more thorough
        self.assertSolverIterationDataRecordedBasic()

    def test_record_solver_linear_scipy_iterative_solver(self):

        # raise unittest.SkipTest("Linear Solver recording not working yet")

        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.nl_solver = NewtonSolver()
        # used for analytic derivatives
        model.nl_solver.ln_solver = ScipyIterativeSolver()
        self.recorder.options['record_abs_error'] = True
        self.recorder.options['record_rel_error'] = True
        self.recorder.options['record_solver_output'] = True
        self.recorder.options['record_solver_residuals'] = True
        model.nl_solver.ln_solver.add_recorder(self.recorder)

        prob.setup()
        t0, t1 = run_driver(prob)

        # TODO_RECORDERS - need to be more thorough
        self.assertSolverIterationDataRecordedBasic()

    def test_record_solver_linear_petsc_ksp(self):

        # raise unittest.SkipTest("Linear Solver recording not working yet")

        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.nl_solver = NewtonSolver()
        # used for analytic derivatives
        model.nl_solver.ln_solver = PetscKSP()
        self.recorder.options['record_abs_error'] = True
        self.recorder.options['record_rel_error'] = True
        self.recorder.options['record_solver_output'] = True
        self.recorder.options['record_solver_residuals'] = True
        model.nl_solver.ln_solver.add_recorder(self.recorder)

        prob.setup()
        t0, t1 = run_driver(prob)

        # TODO_RECORDERS - need to be more thorough
        self.assertSolverIterationDataRecordedBasic()

    def test_record_solver_linear_block_gs(self):

        # raise unittest.SkipTest("Linear Solver recording not working yet")

        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.nl_solver = NewtonSolver()
        # used for analytic derivatives
        model.nl_solver.ln_solver = LinearBlockGS()
        self.recorder.options['record_abs_error'] = True
        self.recorder.options['record_rel_error'] = True
        self.recorder.options['record_solver_output'] = True
        self.recorder.options['record_solver_residuals'] = True
        model.nl_solver.ln_solver.add_recorder(self.recorder)

        prob.setup()
        t0, t1 = run_driver(prob)

        # TODO_RECORDERS - need to be more thorough
        self.assertSolverIterationDataRecordedBasic()

    def test_record_solver_linear_ln_run_once(self):

        # raise unittest.SkipTest("Linear Solver recording not working yet")

        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.nl_solver = NewtonSolver()
        # used for analytic derivatives
        model.nl_solver.ln_solver = LNRunOnce()
        self.recorder.options['record_abs_error'] = True
        self.recorder.options['record_rel_error'] = True
        self.recorder.options['record_solver_output'] = True
        self.recorder.options['record_solver_residuals'] = True
        model.nl_solver.ln_solver.add_recorder(self.recorder)

        prob.setup()
        t0, t1 = run_driver(prob)

        #### No norms so no expected norms

        # TODO_RECORDERS - need to be more thorough
        self.assertSolverIterationDataRecordedBasic()

    def test_record_solver_linear_block_jac(self):

        # raise unittest.SkipTest("Linear Solver recording not working yet")

        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.nl_solver = NewtonSolver()
        # used for analytic derivatives
        model.nl_solver.ln_solver = LinearBlockJac()
        self.recorder.options['record_abs_error'] = True
        self.recorder.options['record_rel_error'] = True
        self.recorder.options['record_solver_output'] = True
        self.recorder.options['record_solver_residuals'] = True
        model.nl_solver.ln_solver.add_recorder(self.recorder)

        prob.setup()
        t0, t1 = run_driver(prob)

        # TODO_RECORDERS - need to be more thorough
        self.assertSolverIterationDataRecordedBasic()

    def test_global_counter(self):
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

        prob = Problem()
        model = prob.model = SellarDerivativesGrouped()

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-2 # to speed the test up
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = True

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        self.recorder.options['record_inputs'] = True
        self.recorder.options['record_outputs'] = True
        self.recorder.options['record_residuals'] = True
        self.recorder.options['record_metadata'] = True

        # Add recorders for Driver, System, Solver
        prob.driver.add_recorder(self.recorder)
        prob.model.add_recorder(self.recorder)
        mda = prob.model.get_subsystem('mda')
        mda.nl_solver.add_recorder(self.recorder)

        prob.setup(check=False, mode='rev')
        t0, t1 = run_driver(prob)
        prob.cleanup()

        # get global counter values from driver, system, and solver recording
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        cur.execute("SELECT counter FROM driver_iterations")
        counters_driver = set( i[0] for i in cur.fetchall() )
        cur.execute("SELECT counter FROM system_iterations")
        counters_system = set( i[0] for i in cur.fetchall() )
        cur.execute("SELECT counter FROM solver_iterations")
        counters_solver = set( i[0] for i in cur.fetchall() )
        cur.execute("SELECT COUNT(rowid) FROM global_iterations")
        global_iterations_records = cur.fetchone()[0]
        con.close()

        # Check to see that they make sense
        self.assertEqual(self.recorder._counter, global_iterations_records )
        self.assertEqual(self.recorder._counter, len(counters_driver) + len(counters_system) + len(counters_solver) )
        self.assertTrue(counters_driver.isdisjoint(counters_system))
        self.assertTrue(counters_driver.isdisjoint(counters_solver))
        self.assertTrue(counters_system.isdisjoint(counters_solver))

    def test_implicit_component(self):

        prob = Problem()
        model = prob.model = DoubleSellar()

        g1 = model.get_subsystem('g1')
        g1.nl_solver = NewtonSolver()
        g1.nl_solver.options['rtol'] = 1.0e-5
        g1.ln_solver = DirectSolver()

        g2 = model.get_subsystem('g2')
        g2.nl_solver = NewtonSolver()
        g2.nl_solver.options['rtol'] = 1.0e-5
        g2.ln_solver = DirectSolver()

        model.nl_solver = NewtonSolver()
        model.ln_solver = ScipyIterativeSolver()

        model.nl_solver.options['solve_subsystems'] = True

        prob.setup()
        t0, t1 = run_driver(prob)
        prob.cleanup()



if __name__ == "__main__":
    unittest.main()
