"""Test N2 with MPI and more than one process."""

import os
import unittest
import openmdao.api as om
from openmdao.visualization.n2_viewer.n2_viewer import n2
from openmdao.utils.mpi import MPI

try:
    from openmdao.parallel_api import PETScVector
except ImportError:
    PETScVector = None

# set DEBUG to True if you want to view the generated HTML file
DEBUG = False
OUTFILE = "n2test.html"


def clean_outfile(outFile):
    if not DEBUG:
        try:
            os.remove(outFile)
        except OSError:
            # Don't want the test to fail if the test file is already removed
            pass


# Test model derived from one contributed by Anil Yildirim at U-Mich MDO lab
class myComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('x2', distributed=True)
        self.add_output('x3', shape=MPI.COMM_WORLD.rank, distributed=True)

    def compute(self, inputs, outputs):
        outputs['x3'] = inputs['x2'] + 1


class Top(om.Group):
    def setup(self):

        indep_var = self.add_subsystem('indep_var', om.IndepVarComp())
        myc = self.add_subsystem('myComp', myComp())

        indep_var.add_output('x1', 2.0)

        self.connect('indep_var.x1', 'myComp.x2')


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class N2ParallelTestCase(unittest.TestCase):
    N_PROCS = 2

    def setUp(self):
        self.p = om.Problem()
        self.p.model = Top()
        self.p.setup()

    def test_n2_parallel(self):
        """
        Verify that allgather() is called from all ranks and doesn't sit there blocking.
        """
        om.n2(self.p, show_browser=False, outfile=OUTFILE)
        MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0:
            self.assertTrue(os.path.exists(OUTFILE), msg=f"{OUTFILE} not found")

    def tearDown(self):
        clean_outfile(OUTFILE)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class MultiN2ParallelTestCase(unittest.TestCase):
    N_PROCS = 2

    def setUp(self):
        # Create 2 communicators and make a problem on each one
        self.globalComm = MPI.COMM_WORLD
        self.localComm = self.globalComm.Split(MPI.COMM_WORLD.rank)

        self.p = om.Problem(comm=self.localComm)
        self.p.model = Top()
        self.p.setup()

        self.outFile = f"n2test_{self.globalComm.rank}.html"

    def test_n2_parallel(self):
        """
        Verify that an N2 file is generated for the problems on both comms
        """
        om.n2(self.p, show_browser=False, outfile=self.outFile)
        MPI.COMM_WORLD.barrier()
        if self.localComm.rank == 0:
            self.assertTrue(os.path.exists(self.outFile), msg=f"{self.outFile} not found")

    def tearDown(self):
        clean_outfile(self.outFile)


if __name__ == "__main__":
    unittest.main()
