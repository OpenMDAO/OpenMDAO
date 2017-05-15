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

from openmdao.api import NonlinearBlockGS, SqliteRecorder, Group, IndepVarComp, ExecComp
from openmdao.core.problem import Problem
from openmdao.devtools.testutil import assert_rel_error
from openmdao.utils.record_util import format_iteration_coordinate
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.recorders.sqlite_recorder import format_version, blob_to_array
from openmdao.test_suite.components.sellar import SellarDerivatives, SellarDerivativesGrouped
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.groups.parallel_groups import ConvergeDiverge

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')
if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver

optimizers = {'pyoptsparse': pyOptSparseDriver}
# optimizers = {'scipy': ScipyOptimizer,


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

        counter, iteration_coordinate, timestamp, success, msg, desvars_blob, responses_blob, objectives_blob, constraints_blob = row_actual

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

        counter, iteration_coordinate, timestamp, success, msg, inputs_blob, outputs_blob, residuals_blob = row_actual

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

        d1 = self.prob.model.get_subsystem('d1')    # an instance of SellarDis1withDerivatives
        d1.add_recorder(self.recorder)

        obj_cmp = self.prob.model.get_subsystem('obj_cmp') # an ExecComp
        obj_cmp.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        coordinate = [0, 'obj_cmp', (6, )]

        expected_inputs = {
                            "obj_cmp.z": [5.0, 2.0],
                            "obj_cmp.y1": [25.54548589,],
                            "obj_cmp.x": [1.0,],
                            "obj_cmp.y2": [12.05425424,],
                            }

        expected_outputs = {"obj_cmp.obj": [28.54549171,],
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

        pz = prob.model.get_subsystem('pz')
        pz.add_recorder(self.recorder)

        mda = prob.model.get_subsystem('mda')
        d1 = mda.get_subsystem('d1')
        d1.add_recorder(self.recorder)

        prob.setup(check=False, mode='rev')
        prob.run_driver()
        prob.cleanup()

    def qqq_test_record_solver(self):
        self.setup_sellar_model()

        self.recorder.options['record_abs_error'] = True
        self.recorder.options['record_rel_error'] = True
        self.recorder.options['record_output'] = True
        self.recorder.options['record_solver_residuals'] = True
        self.prob.model.nl_solver = NonlinearBlockGS()
        self.prob.model._nl_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        coordinate = [0, 'obj_cmp', (6, )]

        expected_abs_error = {

                            }

        expected_rel_error = {
                            }

        expected_solver_residuals = {
                            }

        expected_ouput = {
                    }

        self.assertSystemIterationDataRecorded(((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                                                 expected_solver_residuals, expected_ouput),), self.eps)
if __name__ == "__main__":
    unittest.main()
