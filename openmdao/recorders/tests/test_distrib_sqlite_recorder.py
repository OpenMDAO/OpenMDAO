import errno
import os
import sqlite3
import unittest
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np

from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp, SqliteRecorder
from openmdao.utils.array_utils import evenly_distrib_idxs
from sqlite_recorder_test_utils import _assertDriverIterationDataRecorded
from recorder_test_utils import run_driver

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

class DistributedAdder(ExplicitComponent):
    """
    Distributes the work of adding 10 to every item in the param vector
    """

    def __init__(self, size):
        super(DistributedAdder, self).__init__()
        self.distributed = True

        self.local_size = self.size = size

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

        #NOTE: Each process will get just its local part of the vector
        #print('process {0:d}: {1}'.format(self.comm.rank, params['x'].shape))

        outputs['y'] = inputs['x'] + 10.


class Summer(ExplicitComponent):
    """
    Aggregation component that collects all the values from the distributed
    vector addition and computes a total
    """

    def __init__(self, size):
        super(Summer, self).__init__()
        self.size = size

    def setup(self):
        #NOTE: this component depends on the full y array, so OpenMDAO
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
