""" Unit test for the SqliteRecorder. """

import errno
import os
import unittest
from shutil import rmtree
from tempfile import mkdtemp
import time

import sqlite3
import numpy as np



def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler).
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    """
    Utility function for numpy structured arrays.
    """
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)


from openmdao.devtools.testutil import assert_rel_error


from six import iteritems, iterkeys

from collections import OrderedDict

from sqlitedict import SqliteDict

import numpy as np
from numpy.testing import assert_allclose

from openmdao.core.problem import Problem
from openmdao.api import SqliteRecorder
from openmdao.utils.record_util import format_iteration_coordinate

from openmdao.recorders.sqlite_recorder import format_version

from openmdao.test_suite.components.sellar import SellarDerivatives
import warnings


from openmdao.api import Group, IndepVarComp, ExecComp
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.expl_comp_array import TestExplCompArrayDense
from openmdao.utils.general_utils import set_pyoptsparse_opt

from openmdao.test_suite.groups.parallel_groups import ConvergeDiverge

from openmdao.api import NonlinearBlockGS




# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
# OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')

# if OPTIMIZER:
#     from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


# Test that pyoptsparse SLSQP is a viable option
try:
    import pyoptsparse.pySLSQP.slsqp as slsqp
except ImportError:
    slsqp = None

try:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
except ImportError:
    pyOptSparseDriver = None

optimizers = {'pyoptsparse': pyOptSparseDriver}
# optimizers = {'scipy': ScipyOptimizer,


def run_driver(problem):
    t0 = time.time()
    problem.run_driver()
    t1 = time.time()

    return t0, t1


def _assertIterationDataRecorded(test, db_cur, expected, tolerance):
    '''
        expected can be from multiple cases
    '''
    sentinel = object()
    # keys = list(iterkeys(db))

    # Check to see if we have the right number of cases

    # need the number of records in the table
    db_cur.execute("SELECT Count(*) FROM driver_iterations")
    row = db_cur.fetchone()
    num_cases_in_db = row[0]

    test.assertEquals(num_cases_in_db, len(expected))

    # iterate through the cases
    for coord, (t0, t1), desvars_expected, responses_expected, objectives_expected, constraints_expected in expected:
        iter_coord = format_iteration_coordinate(coord)

        # from the database, get the actual data recorded 


        db_cur.execute("SELECT * FROM driver_iterations WHERE iteration_coordinate=:iteration_coordinate", {"iteration_coordinate": iter_coord})        
        row_actual  = db_cur.fetchone()
        counter, iteration_coordinate, timestamp, success, msg, desvars_actual, responses_array, objectives_array, constraints_array = row_actual

        # Does the timestamp make sense?
        test.assertTrue( t0 <= timestamp and timestamp <= t1)

        test.assertEqual(success, 1)
        test.assertEqual(msg, '')


        # actual_group = db[iter_coord]
        # groupings = {
        #     "timestamp":  None,
        #     "Parameters": params,
        #     "Unknowns":   unknowns,
        #     "Residuals":  resids,
        #     'success': 1,
        #     'msg': '',
        # }

        if desvars_expected is None:
            test.assertIsNone(desvars_actual)
        else:
            # Check to see if the number of values in actual and expected match
            test.assertEqual(len(desvars_actual[0]), len(desvars_expected))
            # 
            # Check to see if the keys in the actual and expected match
            #  .dtype.names
            # Check to see if the values in actual and expected match
            for key, value in iteritems(desvars_expected):
                test.assertTrue(key in desvars_actual[0].dtype.names)
                assert_rel_error(test, desvars_actual[0][key], desvars_expected[key], tolerance)

        if responses_expected is None:
            test.assertIsNone(responses_array)
        else:
            for key, value in iteritems(responses_expected):
                assert_rel_error(test, responses_array[0][key], responses_expected[key], tolerance)

        if objectives_expected is None:
            test.assertIsNone(objectives_array)
        else:
            for key, value in iteritems(objectives_expected):
                assert_rel_error(test, objectives_array[0][key], objectives_expected[key], tolerance)

        if constraints_expected is None:
            test.assertIsNone(constraints_array)
        else:
            for key, value in iteritems(constraints_expected):
                assert_rel_error(test, constraints_array[0][key], constraints_expected[key], tolerance)

        return #TODO_RECORDERS remove this


        for label, values in iteritems(groupings):
            actual = actual_group.get(label, None)
            if isinstance(values, int):
                test.assertEqual(actual, values)
            else:
                # If len(actual) == len(expected) and actual <= expected, then
                # actual == expected.
                test.assertEqual(len(actual), len(values))

                for key, val in values:
                    found_val = actual.get(key, sentinel)

                    if found_val is sentinel:
                        test.fail("Did not find key '{0}'".format(key))

                    try:
                        assert_rel_error(test, found_val, val, tolerance)
                    except TypeError:
                        test.assertEqual(val, found_val)

class TestSqliteRecorder(unittest.TestCase):
    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "sqlite_test")
        # self.tablename_metadata = 'metadata'
        self.tablename_iterations = 'iterations'
        # self.tablename_derivs = 'derivs'
        self.recorder = SqliteRecorder(self.filename)
        print(self.filename)
        self.eps = 1e-5

    def tearDown(self):
        return
        try:
            # rmtree(self.dir)
            pass
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def assertIterationDataRecorded(self, expected, tolerance):
        con = sqlite3.connect(self.filename, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = con.cursor()
        _assertIterationDataRecorded(self, cur, expected, tolerance)
        con.close()



    def setup_sellar_model2(self):
        self.prob = Problem()

        self.prob.model = SellarDerivatives()
        self.prob.model.nl_solver = NonlinearBlockGS()

        self.prob.model.add_design_var('x', lower=-100, upper=100)
        self.prob.model.add_design_var('z', lower=-100, upper=100)
        self.prob.model.add_objective('obj')
        self.prob.model.add_constraint('con1')
        self.prob.model.add_constraint('con2')

        self.prob.setup(check=False)

    def setup_sellar_model(self):
        self.prob = Problem()
        self.prob.model = model = SellarDerivatives()

        optimizer = 'pyoptsparse'
        self.prob.driver = optimizers[optimizer]()

        self.prob.model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                   upper=np.array([10.0, 10.0]))
        self.prob.model.add_design_var('x', lower=0.0, upper=10.0)
        self.prob.model.add_objective('obj')
        self.prob.model.add_constraint('con1', upper=0.0)
        self.prob.model.add_constraint('con2', upper=0.0)
        self.prob.model.suppress_solver_output = True

        self.prob.setup(check=False)

    def test_only_desvars_recorded(self):

        self.setup_sellar_model2()

        self.recorder.options['record_desvars'] = True
        self.recorder.options['record_responses'] = False
        self.recorder.options['record_objectives'] = False
        self.recorder.options['record_constraints'] = False
        self.prob.driver.add_recorder(self.recorder)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()  # closes recorders TODO_RECORDER: need to implement a cleanup

        coordinate = [0, 'Driver', (1, )]

        expected_desvars = {"px.x": [1.0,],
                            "pz.z": [5.0, 2.0]
                            }

        self.assertIterationDataRecorded(((coordinate, (t0, t1), expected_desvars, None, None, None),), self.eps)


    def test_only_objectives_recorded(self):

        self.setup_sellar_model2()

        self.recorder.options['record_desvars'] = False
        self.recorder.options['record_responses'] = False
        self.recorder.options['record_objectives'] = True
        self.recorder.options['record_constraints'] = False
        self.prob.driver.add_recorder(self.recorder)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()  # closes recorders TODO_RECORDER: need to implement a cleanup

        coordinate = [0, 'Driver', (1, )]

        expected_objectives = {"obj_cmp.obj": [28.58830817,],  # {'obj_cmp.obj': array([ 28.58830817])}
                            }

        self.assertIterationDataRecorded(((coordinate, (t0, t1), None, None, expected_objectives, None),), self.eps)

    def test_only_constraints_recorded(self):

        self.setup_sellar_model2()

        self.recorder.options['record_desvars'] = False
        self.recorder.options['record_responses'] = False
        self.recorder.options['record_objectives'] = False
        self.recorder.options['record_constraints'] = True
        self.prob.driver.add_recorder(self.recorder)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()  # closes recorders TODO_RECORDER: need to implement a cleanup

        coordinate = [0, 'Driver', (1, )]

        expected_constraints = {
                            "con_cmp1.con1": [-22.42830237,],
                            "con_cmp2.con2": [-11.94151185,],  
                            }

        self.assertIterationDataRecorded(((coordinate, (t0, t1), None, None, None, expected_constraints),), self.eps)



    def qqqtest_basic(self):
        prob = Problem()
        prob.model = model = SellarDerivatives()

        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_inputs'] = True
        self.recorder.options['record_params'] = True

        model.add_design_var('z')
        model.add_objective('obj')
        model.add_constraint('con1', lower=0)
        model.suppress_solver_output = True

        prob.setup(check=False)
        prob.run_driver()

        prob.cleanup()  # closes recorders TODO_RECORDER: need to implement a cleanup

    def qqqtest_simple_driver_recording(self):
        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        model.suppress_solver_output = True

        prob.driver = pyOptSparseDriver()

        prob.driver.add_recorder(self.recorder)
        self.recorder.options['record_responses'] = True
        self.recorder.options['record_objectives'] = True
        self.recorder.options['record_constraints'] = True

        prob.driver.options['optimizer'] = OPTIMIZER
        # prob.driver.options['optimizer'] = 'SLSQP'
        # # prob.driver.options['optimizer'] = 'CONMIN'
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup(check=False)
        prob.run_driver()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

        from openmdao.recorders.sqlite_reader import SqliteCaseReader
        scr = SqliteCaseReader(self.filename)


