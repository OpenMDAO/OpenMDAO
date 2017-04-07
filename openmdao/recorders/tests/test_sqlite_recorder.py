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
import warnings


from openmdao.api import Problem, Group, IndepVarComp, ExecComp
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.expl_comp_array import TestExplCompArrayDense
from openmdao.utils.general_utils import set_pyoptsparse_opt




# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class TestSqliteRecorder(unittest.TestCase):
    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "sqlite_test")
        # self.tablename_metadata = 'metadata'
        self.tablename_iterations = 'iterations'
        # self.tablename_derivs = 'derivs'
        self.recorder = SqliteRecorder(self.filename)
        #print(self.filename)
        # self.eps = 1e-5

    def tearDown(self):
        return
        try:
            # rmtree(self.dir)
            pass
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

    def test_simple_driver_recording(self):
        # raise unittest.SkipTest("drivers not implemented yet")
        prob = Problem()
        model = prob.model = Group()


        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        model.suppress_solver_output = True

        # prob.driver = ScipyOpt()
        # prob.driver.options['method'] = 'slsqp'


        prob.driver = pyOptSparseDriver()

        prob.driver.add_recorder(self.recorder)

        prob.driver.options['optimizer'] = OPTIMIZER
        # prob.driver.options['optimizer'] = 'SLSQP'
        # # prob.driver.options['optimizer'] = 'CONMIN'
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup(check=False)
        prob.run_driver()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)


