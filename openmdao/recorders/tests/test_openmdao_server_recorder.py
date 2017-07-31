""" Unit test for the OpenMDAOServerRecorder. """
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
            NewtonSolver, NonLinearRunOnce, OpenMDAOServerRecorder, Group, IndepVarComp, ExecComp, \
            DirectSolver, ScipyIterativeSolver, PetscKSP, LinearBlockGS, LinearRunOnce, \
            LinearBlockJac

from openmdao.core.problem import Problem
from openmdao.devtools.testutil import assert_rel_error
from openmdao.utils.record_util import format_iteration_coordinate
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.recorders.openmdao_server_recorder import format_version
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
class TestServerRecorder(unittest.TestCase):
    _endpoint_base = 'http://207.38.86.50:18403/case'
    _default_case_id = '123456'
    _accepted_token = 'test'
    recorded_metadata          = False
    recorded_driver_iteration  = False
    recorded_global_iteration  = False
    recorded_system_metadata   = False
    recorded_system_iteration  = False
    recorded_solver_metadata   = False
    recorded_solver_iterations = False
    driver_data           = None
    driver_iteration_data = None
    gloabl_iteration_data = None
    system_metadata       = None
    system_iterations     = None
    solver_metadata       = None
    solver_iterations     = None

    def setUp(self):
        super(TestServerRecorder, self).setUp()

    def assert_array_close(self, test_val, comp_set):
        values_arr = [t for t in comp_set if t['name'] == test_val['name']]
        if(len(values_arr) != 1):
            self.assertTrue(False, 'Expected to find a value with a unique name in the comp_set, but found 0 or more than 1 instead')
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
                            z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
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
        m.post(self._endpoint_base + '/' + self._default_case_id + '/global_iterations', json=self.check_global_iteration)
        m.post(self._endpoint_base + '/' + self._default_case_id + '/driver_metadata', json=self.check_driver)
        m.post(self._endpoint_base + '/' + self._default_case_id + '/driver_iterations', json=self.check_driver_iteration)
        m.post(self._endpoint_base + '/' + self._default_case_id + '/system_metadata', json=self.check_system_metadata)
        m.post(self._endpoint_base + '/' + self._default_case_id + '/system_iterations', json=self.check_system_iteration)
        m.post(self._endpoint_base + '/' + self._default_case_id + '/solver_metadata', json=self.check_solver_metadata)
        m.post(self._endpoint_base + '/' + self._default_case_id + '/solver_iterations', json=self.check_solver_iterations)

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

    def test_get_case_success(self, m):
        self.setup_endpoints(m)
        recorder = OpenMDAOServerRecorder(self._accepted_token)
        self.assertEqual(recorder._case_id, self._default_case_id)

    def test_get_case_fail(self, m):
        self.setup_endpoints(m)
        recorder = OpenMDAOServerRecorder('')
        self.assertEqual(recorder._case_id, '-1')

    def test_driver_records_metadata(self, m):
        self.setup_endpoints(m)
        recorder = OpenMDAOServerRecorder(self._accepted_token)

        self.setup_sellar_model()

        recorder.options['includes'] = ["p1.x"]
        recorder.options['record_metadata'] = True
        self.prob.driver.add_recorder(recorder)
        self.prob.setup(check=False)

        self.prob.cleanup()
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

        recorder = OpenMDAOServerRecorder(self._accepted_token)
        recorder.options['record_metadata'] = False
        self.prob.driver.add_recorder(recorder)
        self.prob.setup(check=False)

        self.prob.cleanup()

        self.assertFalse(self.recorded_metadata)
        self.assertEqual(self.driver_data, None)

    def test_only_desvars_recorded(self, m):
        self.setup_endpoints(m)

        recorder = OpenMDAOServerRecorder(self._accepted_token)

        self.setup_sellar_model()

        recorder.options['record_desvars'] = True
        recorder.options['record_responses'] = False
        recorder.options['record_objectives'] = False
        recorder.options['record_constraints'] = False
        self.prob.driver.add_recorder(recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()  

        expected_desvars = {
                            "px.x": [1.0, ],
                            "pz.z": [5.0, 2.0]
                           }

        driver_iteration_data = json.loads(self.driver_iteration_data)
        self.driver_iteration_data = None
        self.assertTrue({'name': 'px.x', 'values': [1.0]} in driver_iteration_data['desvars'])
        self.assertTrue({'name': 'pz.z', 'values': [5.0, 2.0]} in driver_iteration_data['desvars'])
        self.assertEqual(driver_iteration_data['responses'], [])
        self.assertEqual(driver_iteration_data['objectives'], [])
        self.assertEqual(driver_iteration_data['constraints'], [])

    def test_only_objectives_recorded(self, m):
        self.setup_endpoints(m)
        recorder = OpenMDAOServerRecorder(self._accepted_token)

        self.setup_sellar_model()

        recorder.options['record_desvars'] = False
        recorder.options['record_responses'] = False
        recorder.options['record_objectives'] = True
        recorder.options['record_constraints'] = False
        self.prob.driver.add_recorder(recorder)
        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        driver_iteration_data = json.loads(self.driver_iteration_data)
        self.assertAlmostEqual(driver_iteration_data['objectives'][0]['values'][0], 28.5883082)
        self.assertEqual(driver_iteration_data['objectives'][0]['name'], 'obj_cmp.obj')
        self.assertEqual(driver_iteration_data['desvars'], [])
        self.assertEqual(driver_iteration_data['responses'], [])
        self.assertEqual(driver_iteration_data['constraints'], [])

    def test_only_constraints_recorded(self, m):
        self.setup_endpoints(m)
        recorder = OpenMDAOServerRecorder(self._accepted_token)

        self.setup_sellar_model()

        recorder.options['record_desvars'] = False
        recorder.options['record_responses'] = False
        recorder.options['record_objectives'] = False
        recorder.options['record_constraints'] = True
        self.prob.driver.add_recorder(recorder)
        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        driver_iteration_data = json.loads(self.driver_iteration_data)
        if driver_iteration_data['constraints'][0]['name'] == 'con_cmp1.con1':
            self.assertAlmostEqual(driver_iteration_data['constraints'][0]['values'][0], -22.42830237)
            self.assertAlmostEqual(driver_iteration_data['constraints'][1]['values'][0], -11.94151185)
            self.assertEqual(driver_iteration_data['constraints'][1]['name'], 'con_cmp2.con2')
            self.assertEqual(driver_iteration_data['constraints'][0]['name'], 'con_cmp1.con1')
        elif driver_iteration_data['constraints'][0]['name'] == 'con_cmp2.con2':
            self.assertAlmostEqual(driver_iteration_data['constraints'][1]['values'][0], -22.42830237)
            self.assertAlmostEqual(driver_iteration_data['constraints'][0]['values'][0], -11.94151185)
            self.assertEqual(driver_iteration_data['constraints'][0]['name'], 'con_cmp2.con2')
            self.assertEqual(driver_iteration_data['constraints'][1]['name'], 'con_cmp1.con1')
        else:
            self.assertTrue(False, 'Driver iteration data did not contain the expected names for constraints')

        self.assertEqual(driver_iteration_data['desvars'], [])
        self.assertEqual(driver_iteration_data['objectives'], [])
        self.assertEqual(driver_iteration_data['responses'], [])

    def test_record_system(self, m):
        self.setup_endpoints(m)
        recorder = OpenMDAOServerRecorder(self._accepted_token)

        self.setup_sellar_model()

        recorder.options['record_inputs'] = True
        recorder.options['record_outputs'] = True
        recorder.options['record_residuals'] = True
        recorder.options['record_metadata'] = True

        self.prob.model.add_recorder(recorder)

        d1 = self.prob.model.get_subsystem('d1')  # instance of SellarDis1withDerivatives, a Group
        d1.add_recorder(recorder)

        obj_cmp = self.prob.model.get_subsystem('obj_cmp')  # an ExecComp
        obj_cmp.add_recorder(recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        coordinate = [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ),
                      'NonlinearBlockGS', (6, ), 'd1._solve_nonlinear', (6, )]
        expected_inputs = {
                            "d1.y2": [12.05848815],
                            "d1.z": [5.0, 2.0],
                            "d1.x": [1.0, ],
                          }
        expected_outputs = {"d1.y1": [25.58830237, ], }
        expected_residuals = {"d1.y1": [0.0, ], }

        system_metadata = json.loads(self.system_metadata)
        system_iterations = json.loads(self.system_iterations)
        scaling_facts_raw = system_metadata['scaling_factors']
        scaling_facts_ascii = scaling_facts_raw.encode('ascii')
        scaling_facts_base64 = base64.decodebytes(scaling_facts_ascii)
        scaling_facts = pickle.loads(scaling_facts_base64)
        system_metadata['scaling_factors'] = scaling_facts

        inputs = [
            {'name': 'd1.z', 'values': [5.0, 2.0]},
            {'name': 'd1.x', 'values': [1.0]},
            {'name': 'd2.z', 'values': [5.0, 2.0]},
            {'name': 'd1.y2', 'values': [12.05848815]}
        ]

        outputs = [
            {'name': 'd1.y1', 'values': [25.58830237]}
        ]

        residuals = [
            {'name': 'd1.y1', 'values': [0.0]}
        ]

        for i in inputs:
            self.assert_array_close(i, system_iterations['inputs'])
        for o in outputs:
            self.assert_array_close(o, system_iterations['outputs'])
        for r in residuals:
            self.assert_array_close(r, system_iterations['residuals'])

        inputs = [ 
            {'name': 'con_cmp2.y2', 'values': [12.058488150624356]},
            {'name': 'obj_cmp.y1', 'values': [25.58830237000701]},
            {'name': 'obj_cmp.x', 'values': [1.0]},
            {'name': 'obj_cmp.z', 'values': [5.0, 2.0]}
        ]

        outputs = [
            {'name': 'obj_cmp.obj', 'values': [28.58830816]}
        ]

        residuals = [
            {'name': 'obj_cmp.obj', 'values': [0.0]}
        ]
        
        for i in inputs:
            self.assert_array_close(i, system_iterations['inputs'])
        for o in outputs:
            self.assert_array_close(o, system_iterations['outputs'])
        for r in residuals:
            self.assert_array_close(r, system_iterations['residuals'])

    def test_simple_driver_recording(self, m):
        self.setup_endpoints(m)
        recorder = OpenMDAOServerRecorder(self._accepted_token)

        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        model.suppress_solver_output = True

        prob.driver = pyOptSparseDriver()

        prob.driver.add_recorder(recorder)
        recorder.options['record_desvars'] = True
        recorder.options['record_responses'] = True
        recorder.options['record_objectives'] = True
        recorder.options['record_constraints'] = True

        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)
        prob.setup(check=False)

        t0, t1 = run_driver(prob)

        prob.cleanup()

        driver_iteration_data = json.loads(self.driver_iteration_data)

        expected_desvars = [
                {'name': 'p1.x', 'values': [7.1666666]},
                {'name': 'p2.y', 'values': [-7.8333333]}
            ]

        expected_objectives = [
            {'name': 'comp.f_xy', 'values': [-27.083333]}
        ]

        expected_constraints = [
            {'name': 'con.c', 'values': [-15.0]}
        ]

        for d in expected_desvars:
            self.assert_array_close(d, driver_iteration_data['desvars'])

        for o in expected_objectives:
            self.assert_array_close(o, driver_iteration_data['objectives'])

        for c in expected_constraints:
            self.assert_array_close(c, driver_iteration_data['constraints'])

    def test_record_solver(self, m):
        self.setup_endpoints(m)
        recorder = OpenMDAOServerRecorder(self._accepted_token)
        
        self.setup_sellar_model()

        recorder.options['record_abs_error'] = True
        recorder.options['record_rel_error'] = True
        recorder.options['record_solver_output'] = True
        recorder.options['record_solver_residuals'] = True
        self.prob.model._nonlinear_solver.add_recorder(recorder)

        self.prob.setup(check=False)

        t0, t1 = run_driver(self.prob)

        self.prob.cleanup()

        coordinate = [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ), 'NonlinearBlockGS', (6, )]

        expected_solver_output = {
            "con_cmp1.con1": [-22.42830237000701],
            "d1.y1": [25.58830237000701],
            "con_cmp2.con2": [-11.941511849375644],
            "pz.z": [5.0, 2.0],
            "obj_cmp.obj": [28.588308165163074],
            "d2.y2": [12.058488150624356],
            "px.x": [1.0]
        }

        expected_solver_residuals = {
            "con_cmp1.con1": [0.0],
            "d1.y1": [1.318802844707534e-10],
            "con_cmp2.con2": [0.0],
            "pz.z": [0.0, 0.0],
            "obj_cmp.obj": [0.0],
            "d2.y2": [0.0],
            "px.x": [0.0]
        }

        solver_iteration = json.loads(self.solver_iterations)
        solver_metadata = json.loads(self.solver_metadata)
        print("iteration: " + str(solver_iteration))
        print("")
        print("metadata: " + str(solver_metadata))

        self.assertAlmostEqual(solver_iteration['abs_err'], 1.31880284470753394998e-10)
        self.assertAlmostEqual(solver_iteration['rel_error'], 3.6299074030587596e-12)
        

    # def test_includes(self, m):
    #     self.setup_endpoints(m)
    #     recorder = OpenMDAOServerRecorder(self._accepted_token)

    #     if OPT is None:
    #         raise unittest.SkipTest("pyoptsparse is not installed")

    #     if OPTIMIZER is None:
    #         raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

    #     prob = Problem()
    #     model = prob.model = Group()

    #     model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
    #     model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
    #     model.add_subsystem('comp', Paraboloid(), promotes=['*'])
    #     model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

    #     model.suppress_solver_output = True

    #     prob.driver = pyOptSparseDriver()

    #     prob.driver.add_recorder(recorder)
    #     recorder.options['record_desvars'] = True
    #     recorder.options['record_responses'] = True
    #     recorder.options['record_objectives'] = True
    #     recorder.options['record_constraints'] = True
    #     recorder.options['includes'] = ['*']
    #     recorder.options['excludes'] = ['p2*']

    #     prob.driver.options['optimizer'] = OPTIMIZER
    #     prob.driver.opt_settings['ACC'] = 1e-9

    #     model.add_design_var('x', lower=-50.0, upper=50.0)
    #     model.add_design_var('y', lower=-50.0, upper=50.0)
    #     model.add_objective('f_xy')
    #     model.add_constraint('c', upper=-15.0)

    #     prob.setup(check=False)
    #     t0, t1 = run_driver(prob)

    #     prob.cleanup()

    #     coordinate = [0, 'SLSQP', (5, )]

    #     expected_desvars = [
    #         {'name': 'p1.x', 'values': prob['p1.x']}
    #     ]

    #     expected_objectives = [
    #         {'name': 'comp.f_xy', 'values': prob['comp.f_xy']}
    #     ]

    #     expected_constraints = [
    #         {'name': 'con.c', 'values': prob['con.c']}
    #     ]

    #     system_iterations = json.loads(self.system_iterations)

        # for d in expected_desvars:
        #     self.assert_array_close(d, system_iterations['inputs'])
        # for o in expected_objectives:
        #     self.assert_array_close(o, system_iterations['outputs'])
        # for c in expected_constraints:
        #     self.assert_array_close(c, system_iterations['residuals'])

if __name__ == "__main__":
    unittest.main()
