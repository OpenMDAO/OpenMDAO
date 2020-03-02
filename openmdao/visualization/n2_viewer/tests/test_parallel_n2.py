"""Test N2 with MPI and more than one process."""

import os
import unittest
import openmdao.api as om
from openmdao.visualization.n2_viewer.n2_viewer import _get_viewer_data, n2
from openmdao.utils.mpi import MPI

try:
    from openmdao.parallel_api import PETScVector
except ImportError:
    PETScVector = None

# set DEBUG to True if you want to view the generated HTML file
DEBUG = False
OUTFILE = 'n2test.html'

# Test model derived from one contributed by Anil Yildirim at U-Mich MDO lab
class myComp(om.ExplicitComponent):
    def initialize(self):
        self.options['distributed'] = True

    def setup(self):
        self.add_input('x2')
        self.add_output('x3', shape=MPI.COMM_WORLD.rank)

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

    def tearDown(self):
        if not DEBUG:
            try:
                os.remove(OUTFILE)
            except:
                # Don't want the test to fail if the test file is already removed
                pass

if __name__ == "__main__":
    unittest.main()
