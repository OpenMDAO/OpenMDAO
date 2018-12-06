import unittest

import types
import os
import sys
import tempfile
import shutil
import subprocess

from openmdao.api import Problem
from openmdao.test_suite.components.sellar import SellarNoDerivatives

from openmdao.devtools import iprof_mem


@unittest.skip("interactive test, not to be run with test suite")
class TestProfileMemory(unittest.TestCase):

    def test_sellar(self):
        prob = Problem(SellarNoDerivatives()).setup()

        with iprof_mem.memtrace(min_mem=0.1):
            prob.run_model()

        # expect output similar to the following:
        # 0.11  (435 calls)  </Users/banaylor/dev/blue/openmdao/utils/name_maps.py:124>.name2abs_name
        # 0.11  (14 calls)  ExplicitComponent._solve_nonlinear:(IndepVarComp)
        # 0.11  (7 calls)  NonlinearRunOnce.solve
        # 0.11  (150 calls)  Vector.__contains__:(DefaultVector)
        # 0.12  (7 calls)  Group._solve_nonlinear
        # 0.13  (1 calls)  Driver._update_voi_meta
        # 0.14  (2 calls)  DefaultTransfer._setup_transfers
        # 0.16  (1 calls)  NonlinearBlockGS._iter_initialize
        # 0.16  (1 calls)  NonlinearSolver._iter_initialize:(NonlinearBlockGS)
        # 0.19  (24 calls)  ExplicitComponent._apply_nonlinear:(ExecComp)
        # 0.20  (1 calls)  System._setup_vectors:(SellarNoDerivatives)
        # 0.25  (105 calls)  _IODict.__getitem__
        # 0.26  (80 calls)  Vector.__init__:(DefaultVector)
        # 0.26  (21 calls)  ExplicitComponent._solve_nonlinear:(ExecComp)
        # 0.34  (45 calls)  ExecComp.compute
        # 0.39  (8 calls)  NonlinearSolver._run_apply:(NonlinearBlockGS)
        # 0.39  (8 calls)  Group._apply_nonlinear:(SellarNoDerivatives)
        # 0.57  (7 calls)  NonlinearBlockGS._single_iteration
        # 0.59  (1 calls)  System._final_setup:(SellarNoDerivatives)
        # 0.75  (1 calls)  Problem.final_setup
        # 1.07  (1 calls)  NonlinearSolver.solve:(NonlinearBlockGS)
        # 1.07  (1 calls)  Solver._run_iterator:(NonlinearBlockGS)
        # 1.07  (1 calls)  System.run_solve_nonlinear:(SellarNoDerivatives)
        # 1.07  (1 calls)  Group._solve_nonlinear:(SellarNoDerivatives)
        # 1.83  (1 calls)  Problem.run_model


class TestCmdlineMemory(unittest.TestCase):
    def setUp(self):
        try:
            import psutil
        except ImportError:
            raise unittest.SkipTest("psutil is not installed")

        self.tstfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mem_model.py')
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='TestDOEDriver-')
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    def _run_command(self, cmd):
        try:
            output = subprocess.check_output(cmd.split()).decode('utf-8', 'ignore')
        except subprocess.CalledProcessError as err:
            msg = "Running command '{}' failed. " + \
                  "Output was: \n{}".format(cmd, err.output.decode('utf-8'))
            self.fail(msg)

    def test_mem(self):
        self._run_command('openmdao mem %s' % self.tstfile)
        self._run_command('openmdao mempost mem_trace.raw')

    def test_mem_tree(self):
        self._run_command('openmdao mem -t %s' % self.tstfile)
        self._run_command('openmdao mempost -t mem_trace.raw')


if __name__ == "__main__":
    unittest.main()
