import unittest

import types

from openmdao.api import Problem
from openmdao.test_suite.components.sellar import SellarNoDerivatives

from openmdao.devtools import iprof_mem


@unittest.skip("interactive test, not to be run with test suite")
class TestProfileMemory(unittest.TestCase):

    def test_sellar(self):
        prob = Problem(SellarNoDerivatives()).setup()

        iprof_mem.setup()

        # check that the callback has been registered as expected
        self.assertTrue(iprof_mem._registered)
        self.assertTrue(isinstance(iprof_mem._trace_memory, types.FunctionType))
        self.assertTrue(isinstance(iprof_mem.mem_usage, types.FunctionType))

        # can't check output (it comes at system exit)
        # just make sure no exceptions are raised
        iprof_mem.start()
        prob.run_model()
        iprof_mem.stop()

        # expect output similar to the following:
        #
        # Memory (MB)   Calls  File:Line:Function
        # ---------------------------------------
        #     0.1406        1  path/to/openmdao/vectors/default_vector.py:251:_initialize_views
        #     0.1406        1  path/to/openmdao/vectors/vector.py:115:__init__
        #     0.2812        2  path/to/openmdao/core/system.py:1109:_setup_bounds
        #      0.332        1  path/to/openmdao/core/problem.py:424:run_model
        #      0.332        1  path/to/openmdao/core/system.py:728:_final_setup
        #      0.332        1  path/to/openmdao/core/problem.py:613:final_setup
        #     0.5742        3  path/to/openmdao/core/system.py:1065:_setup_vectors
        # ---------------------------------------
        # Memory (MB)   Calls  File:Line:Function


if __name__ == "__main__":
    unittest.main()
