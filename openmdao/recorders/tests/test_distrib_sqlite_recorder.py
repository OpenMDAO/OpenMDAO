import os

import unittest

from shutil import rmtree
from tempfile import mkdtemp

from openmdao.api import Problem, Group, SqliteRecorder

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

@unittest.skipIf(PETScVector is None or os.environ.get("TRAVIS"),
                 "PETSc is required." if PETScVector is None
                 else "Unreliable on Travis CI.")
class DistributedRecorderTest(unittest.TestCase):

    N_PROCS = 2

    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "sqlite_test")
        self.recorder = SqliteRecorder(self.filename)

    def tearDown(self):
        try:
            rmtree(self.dir)
            pass
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

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


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
