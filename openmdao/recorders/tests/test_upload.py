"""
Unit tests for upload_data.
"""
import errno
import os
import time
import unittest
import numpy as np
import requests_mock
import json
import base64

from shutil import rmtree
from six import iteritems, PY2, PY3
from tempfile import mkdtemp

from openmdao.api import BoundsEnforceLS, NonlinearBlockGS, ArmijoGoldsteinLS, NonlinearBlockJac,\
            NewtonSolver, NonLinearRunOnce, WebRecorder, Group, IndepVarComp, ExecComp, \
            DirectSolver, ScipyIterativeSolver, PetscKSP, LinearBlockGS, LinearRunOnce, \
            LinearBlockJac, SqliteRecorder, upload

from openmdao.core.problem import Problem
from openmdao.devtools.testutil import assert_rel_error
from openmdao.utils.record_util import format_iteration_coordinate
from openmdao.recorders.recording_iteration_stack import recording_iteration
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.recorders.web_recorder import format_version
from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, \
    SellarDis2withDerivatives
from openmdao.test_suite.components.paraboloid import Paraboloid

if PY2:
    import cPickle as pickle
if PY3:
    import pickle

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')
if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
    optimizers = {'pyoptsparse': pyOptSparseDriver}

def run_driver(problem):
    t0 = time.time()
    problem.run_driver()
    t1 = time.time()

    return t0, t1

@requests_mock.Mocker()
class TestDataUploader(unittest.TestCase):
    _endpoint_base = 'http://www.openmdao.org/visualization/case'
    _default_case_id = '123456'
    _accepted_token = 'test'
    recorded_metadata = False
    recorded_driver_iteration = False
    recorded_global_iteration = False
    recorded_system_metadata = False
    recorded_system_iteration = False
    recorded_solver_metadata = False
    recorded_solver_iterations = False
    driver_data = None
    driver_iteration_data = None
    gloabl_iteration_data = None
    system_metadata = None
    system_iterations = None
    solver_metadata = None
    solver_iterations = None

    def setUp(self):
        # if OPT is None:
        #     raise unittest.SkipTest("pyoptsparse is not installed")

        # if OPTIMIZER is None:
        #     raise unittest.SkipTest("pyoptsparse is not providing SLSQP")

        recording_iteration.stack = []
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "sqlite_test")
        self.recorder = SqliteRecorder(self.filename)
        # print(self.filename)  # comment out to make filename printout go away.
        self.eps = 1e-5

    def tearDown(self):
        # return  # comment out to allow db file to be removed.
        try:
            rmtree(self.dir)
            pass
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def assert_array_close(self, test_val, comp_set):
        values_arr = [t for t in comp_set if t['name'] == test_val['name']]
        if len(values_arr) != 1:
            self.assertTrue(False, 'Expected to find a value with a unique name in the comp_set,\
             but found 0 or more than 1 instead')
            return
        np.testing.assert_almost_equal(test_val['values'], values_arr[0]['values'], decimal=5)

    def setup_sellar_model(self):
        self.prob = Problem()

        model = self.prob.model = Group()
        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])
        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])
        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                                                promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])
        self.prob.model.nonlinear_solver = NonlinearBlockGS()

        self.prob.model.add_design_var('x', lower=-100, upper=100)
        self.prob.model.add_design_var('z', lower=-100, upper=100)
        self.prob.model.add_objective('obj')
        self.prob.model.add_constraint('con1')
        self.prob.model.add_constraint('con2')

    def setup_sellar_grouped_model(self):
        self.prob = Problem()

        model = self.prob.model = Group()

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        mda = model.add_subsystem('mda', Group(), promotes=['x', 'z', 'y1', 'y2'])
        mda.linear_solver = ScipyIterativeSolver()
        mda.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        mda.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0, y1=0.0,
                                                y2=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        mda.nonlinear_solver = NonlinearBlockGS()
        model.linear_solver = ScipyIterativeSolver()

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

    def setup_endpoints(self, m):
        m.post(self._endpoint_base, json=self.check_header, status_code=200)
        m.post(self._endpoint_base + '/' + self._default_case_id + '/global_iterations',
               json=self.check_global_iteration)
        m.post(self._endpoint_base + '/' + self._default_case_id + '/driver_metadata',
               json=self.check_driver)
        m.post(self._endpoint_base + '/' + self._default_case_id + '/driver_iterations',
               json=self.check_driver_iteration)
        m.post(self._endpoint_base + '/' + self._default_case_id + '/system_metadata',
               json=self.check_system_metadata)
        m.post(self._endpoint_base + '/' + self._default_case_id + '/system_iterations',
               json=self.check_system_iteration)
        m.post(self._endpoint_base + '/' + self._default_case_id + '/solver_metadata',
               json=self.check_solver_metadata)
        m.post(self._endpoint_base + '/' + self._default_case_id + '/solver_iterations',
               json=self.check_solver_iterations)

    def check_header(self, request, context):
        if request.headers['token'] == self._accepted_token:
            return {
                'case_id': self._default_case_id,
                'status': 'Success'
            }
        else:
            return {
                'case_id': '-1',
                'status': 'Failed',
                'reasoning': 'Bad token'
            }

    def check_driver(self, request, context):
        self.recorded_metadata = True
        self.driver_data = request.body
        return {'status': 'Success'}

    def check_driver_iteration(self, request, context):
        self.recorded_driver_iteration = True
        self.driver_iteration_data = request.body
        return {'status': 'Success'}

    def check_global_iteration(self, request, context):
        self.recorded_global_iteration = True
        self.global_iteration_data = request.body
        return {'status': 'Success'}

    def check_system_metadata(self, request, context):
        self.recorded_system_metadata = True
        self.system_metadata = request.body

    def check_system_iteration(self, request, context):
        self.recorded_system_iteration = True
        self.system_iterations = request.body

    def check_solver_metadata(self, request, context):
        self.recorded_solver_metadata = True
        self.solver_metadata = request.body

    def check_solver_iterations(self, request, context):
        self.recorded_solver_iterations = True
        self.solver_iterations = request.body

    def test_driver_records_metadata(self, m):
        self.setup_endpoints(m)

        self.setup_sellar_model()

        self.recorder.options['includes'] = ["p1.x"]
        self.recorder.options['record_metadata'] = True
        self.prob.driver.add_recorder(self.recorder)
        self.prob.setup(check=False)

        # Need this since we aren't running the model.
        self.prob.final_setup()

        self.prob.cleanup()

        upload(self.filename, self._accepted_token, suppress_output=True)
        self.assertTrue(self.recorded_metadata)
        self.recorded_metadata = False

        driv_data = json.loads(self.driver_data)
        self.driver_data = None

        driv_id = driv_data['id']
        model_data = json.loads(driv_data['model_viewer_data'])
        connections = model_data['connections_list']
        self.assertEqual(driv_id, 'Driver')
        self.assertEqual(len(connections), 11)

    def test_driver_doesnt_record_metadata(self, m):
        self.setup_endpoints(m)

        self.setup_sellar_model()

        self.recorder.options['record_metadata'] = False
        self.prob.driver.add_recorder(self.recorder)
        self.prob.setup(check=False)

        self.prob.cleanup()
        upload(self.filename, self._accepted_token, suppress_output=True)

        driv_data = json.loads(self.driver_data)
        self.assertEqual(driv_data['model_viewer_data'], '{}')

    def test_only_desvars_recorded(self, m):
        self.setup_endpoints(m)

        self.setup_sellar_model()

        self.recorder.options['record_desvars'] = True
        self.recorder.options['record_responses'] = False
        self.recorder.options['record_objectives'] = False
        self.recorder.options['record_constraints'] = False
        self.prob.driver.add_recorder(self.recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()
        upload(self.filename, self._accepted_token)

        driver_iteration_data = json.loads(self.driver_iteration_data)
        self.driver_iteration_data = None
        self.assertTrue({'name': 'px.x', 'values': [1.0]} in driver_iteration_data['desvars'])
        self.assertTrue({'name': 'pz.z', 'values': [5.0, 2.0]} in driver_iteration_data['desvars'])
        self.assertEqual(driver_iteration_data['responses'], [])
        self.assertEqual(driver_iteration_data['objectives'], [])
        self.assertEqual(driver_iteration_data['constraints'], [])

if __name__ == "__main__":
    unittest.main()
