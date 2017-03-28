""" Unit test for the SqliteRecorder. """

import errno
import os
import unittest
from shutil import rmtree
from tempfile import mkdtemp
import time

from six import iteritems, iterkeys

from collections import OrderedDict

from sqlitedict import SqliteDict

import numpy as np
from numpy.testing import assert_allclose

from openmdao.api import Problem, SqliteRecorder
# from openmdao.util.record_util import format_iteration_coordinate

from openmdao.recorders.sqlite_recorder import format_version

from openmdao.test_suite.components.sellar import SellarDerivatives



class TestSqliteRecorder(unittest.TestCase):
    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "sqlite_test")
        # self.tablename_metadata = 'metadata'
        self.tablename_iterations = 'iterations'
        # self.tablename_derivs = 'derivs'
        self.recorder = SqliteRecorder(self.filename)
        print(self.filename)
        # self.eps = 1e-5

    def tearDown(self):
        return
        try:
            rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def test_basic(self):
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




