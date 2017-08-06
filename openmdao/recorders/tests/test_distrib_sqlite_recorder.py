import errno
import os
from six import iteritems
import sqlite3
import time
import unittest
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np


from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp, SqliteRecorder
from openmdao.devtools.testutil import assert_rel_error
from openmdao.recorders.sqlite_recorder import blob_to_array
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.utils.mpi import MPI
from openmdao.utils.record_util import format_iteration_coordinate

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


def run_driver(problem):
    t0 = time.time()
    problem.run_driver()
    t1 = time.time()
    return t0, t1


def _assertDriverIterationDataRecorded(test, db_cur, expected, tolerance):
    """
        Expected can be from multiple cases.
    """
    # iterate through the cases
    for coord, (t0, t1), desvars_expected, responses_expected, objectives_expected, \
            constraints_expected in expected:
        iter_coord = format_iteration_coordinate(coord)

        # from the database, get the actual data recorded
        db_cur.execute("SELECT * FROM driver_iterations WHERE "
                       "iteration_coordinate=:iteration_coordinate",
                       {"iteration_coordinate": iter_coord})
        row_actual = db_cur.fetchone()

        test.assertTrue(row_actual,
            'Driver iterations table does not contain the requested iteration coordinate: "{}"'.format(iter_coord))


        counter, global_counter, iteration_coordinate, timestamp, success, msg, desvars_blob,\
            responses_blob, objectives_blob, constraints_blob = row_actual

        desvars_actual = blob_to_array(desvars_blob)
        responses_actual = blob_to_array(responses_blob)
        objectives_actual = blob_to_array(objectives_blob)
        constraints_actual = blob_to_array(constraints_blob)

        # Does the timestamp make sense?
        test.assertTrue(t0 <= timestamp and timestamp <= t1)

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
                    test.assertTrue(key in actual[0].dtype.names,
                                    '{} variable not found in actual data'
                                    ' from recorder'.format(key))
                    # Check to see if the values in actual and expected match
                    assert_rel_error(test, actual[0][key], expected[key], tolerance)
        return


class DistributedAdder(ExplicitComponent):
    """
    Distributes the work of adding 10 to every item in the param vector
    """

    def __init__(self, size):
        super(DistributedAdder, self).__init__()

        self.local_size = self.size = size

    def get_req_procs(self):
        """
        min/max number of procs that this component can use
        """
        return 1, self.size

    def setup(self):
        """
        specify the local sizes of the variables and which specific indices this specific
        distributed component will handle. Indices do NOT need to be sequential or
        contiguous!
        """
        comm = self.comm
        rank = comm.rank

        # NOTE: evenly_distrib_idxs is a helper function to split the array
        #       up as evenly as possible
        sizes, offsets = evenly_distrib_idxs(comm.size, self.size)
        local_size, local_offset = sizes[rank], offsets[rank]
        self.local_size = local_size

        start = local_offset
        end = local_offset + local_size

        self.add_input('x', val=np.zeros(local_size, float),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_output('y', val=np.zeros(local_size, float))

    def compute(self, inputs, outputs):

        # NOTE: Each process will get just its local part of the vector
        # print('process {0:d}: {1}'.format(self.comm.rank, params['x'].shape))

        outputs['y'] = inputs['x'] + 10.


class Summer(ExplicitComponent):
    """
    Agreggation component that collects all the values from the distributed
    vector addition and computes a total
    """

    def __init__(self, size):
        super(Summer, self).__init__()
        self.size = size

    def setup(self):
        # NOTE: this component depends on the full y array, so OpenMDAO
        #      will automatically gather all the values for it
        self.add_input('y', val=np.zeros(self.size))
        self.add_output('sum', 0.0, shape=1)

    def compute(self, inputs, outputs):
        outputs['sum'] = np.sum(inputs['y'])


@unittest.skipIf(PETScVector is None or os.environ.get("TRAVIS"),
                 "PETSc is required." if PETScVector is None
                 else "Unreliable on Travis CI.")
class DistributedRecorderTest(unittest.TestCase):

    N_PROCS = 2

    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "sqlite_test")
        self.recorder = SqliteRecorder(self.filename)
        self.eps = 1e-5

    def tearDown(self):
        try:
            rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def assertDriverIterationDataRecorded(self, expected, tolerance):
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        _assertDriverIterationDataRecorded(self, cur, expected, tolerance)
        con.close()

    def test_distrib_record_system(self):
        prob = Problem()
        prob.model = Group()

        try:
            prob.model.add_recorder(self.recorder)
        except RuntimeError as err:
            self.assertEqual(str(err), "Recording of Systems when running parallel code is not supported yet")
        else:
            self.fail('RuntimeError expected.')

    def test_distrib_record_solver(self):
        prob = Problem()
        prob.model = Group()
        try:
            prob.model.nonlinear_solver.add_recorder(self.recorder)
        except RuntimeError as err:
            self.assertEqual(str(err), "Recording of Solvers when running parallel code is not supported yet")
        else:
            self.fail('RuntimeError expected.')

    def test_distrib_record_driver(self):
        size = 100  # how many items in the array

        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('des_vars', IndepVarComp('x', np.ones(size)), promotes=['x'])
        prob.model.add_subsystem('plus', DistributedAdder(size), promotes=['x', 'y'])
        prob.model.add_subsystem('summer', Summer(size), promotes=['y', 'sum'])
        self.recorder.options['record_desvars'] = True
        self.recorder.options['record_responses'] = True
        self.recorder.options['record_objectives'] = True
        self.recorder.options['record_constraints'] = True
        prob.driver.add_recorder(self.recorder)

        prob.model.add_design_var('x')
        prob.model.add_objective('sum')

        prob.setup(vector_class=PETScVector, check=False)

        prob['x'] = np.ones(size)

        t0, t1 = run_driver(prob)
        prob.cleanup()

        if prob.comm.rank == 0:
            coordinate = [0, 'Driver', (0,)]

            expected_desvars = {
                "des_vars.x": prob['des_vars.x'],
            }

            expected_objectives = {
                "summer.sum": prob['summer.sum'],
            }

            self.assertDriverIterationDataRecorded(((coordinate, (t0, t1), expected_desvars, None,
                                                     expected_objectives, None),), self.eps)


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
