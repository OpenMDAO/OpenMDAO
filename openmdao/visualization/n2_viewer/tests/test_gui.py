"""Test N2 GUI with multiple models using Node.js."""

from __future__ import division, print_function

import os
import unittest
import openmdao.api as om
# from openmdao.visualization.n2_viewer.n2_viewer import _get_viewer_data, n2
# from openmdao.utils.mpi import MPI

# set DEBUG to True if you want to view the generated HTML file
GUI_TEST_SUBDIR = 'gui_test_models'

# @unittest.skipUnless(MPI, "MPI is required to test N2 parallel generation.")
class N2ParallelTestCase(unittest.TestCase):
    N_PROCS = 2

    def setUp(self):
        self.parentDir = os.path.dirname(os.path.realpath(__file__))
        self.modelDir = self.parentDir + '/' + GUI_TEST_SUBDIR
        self.models = filter(lambda x: x.endswith('.py'), os.listdir(self.modelDir))
        
    def test_n2_common(self):
        """
        Perform a set of generic tests against all models in GUI_TEST_SUBDIR.
        """
        modelList = ','.join(map(str, self.models))
        print(modelList)


    def tearDown(self):
        pass
        # if not DEBUG:
        #    try:
        #        os.remove(OUTFILE)
        #    except:
        #        # Don't want the test to fail if the test file is already removed
        #        pass

if __name__ == "__main__":
    unittest.main()
