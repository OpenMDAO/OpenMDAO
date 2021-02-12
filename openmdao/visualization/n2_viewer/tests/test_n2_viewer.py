"""Unit Tests for n2_viewer"""
import unittest

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.visualization.n2_viewer.n2_viewer import _get_viewer_data
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped, SellarProblem


@use_tempdirs
class TestN2Viewer(unittest.TestCase):

    def setUp(self):

        self.filename = "cases.sql"
        self.recorder = om.SqliteRecorder(self.filename)

        self.driver_case = "rank0:DOEDriver_PlackettBurman|3"
        self.problem_case = "rank0:DOEDriver_PlackettBurman|3|root._solve_nonlinear|3"

    def test_driver_case(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = om.DOEDriver(om.PlackettBurmanGenerator())
        prob.driver.add_recorder(self.recorder)
        prob.driver.recording_options['includes'] = ['*']

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        data_dict = _get_viewer_data(self.filename, case_id=self.driver_case)

        vals = data_dict['tree']['children'][2]['children']
        x_val = vals[0]['value']
        y_val = vals[1]['value']
        f_xy_val = vals[2]['value']

        self.assertEqual(x_val, np.array([1.]))
        self.assertEqual(y_val, np.array([1.]))
        self.assertEqual(f_xy_val, np.array([27.]))

    def test_driver_case_na(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = om.DOEDriver(om.PlackettBurmanGenerator())
        prob.driver.add_recorder(self.recorder)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        data_dict = _get_viewer_data(self.filename, case_id=self.driver_case)

        vals = data_dict['tree']['children'][2]['children']
        x_val = vals[0]['value']
        y_val = vals[1]['value']
        f_xy_val = vals[2]['value']

        self.assertEqual(x_val, "N/A")
        self.assertEqual(y_val, "N/A")
        self.assertEqual(f_xy_val, np.array([27.]))

    def test_index_number(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = om.DOEDriver(om.PlackettBurmanGenerator())
        prob.driver.add_recorder(self.recorder)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        data_dict = _get_viewer_data(self.filename, case_id=3)

        vals = data_dict['tree']['children'][2]['children']
        x_val = vals[0]['value']
        y_val = vals[1]['value']
        f_xy_val = vals[2]['value']

        self.assertEqual(x_val, "N/A")
        self.assertEqual(y_val, "N/A")
        self.assertEqual(f_xy_val, np.array([27.]))

    def test_root_case(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        recorder = om.SqliteRecorder("cases.sql")
        prob.driver = om.DOEDriver(om.PlackettBurmanGenerator())
        prob.driver.add_recorder(recorder)
        prob.model.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        data_dict = _get_viewer_data("cases.sql",
                                    case_id=self.problem_case)

        vals = data_dict['tree']['children'][2]['children']
        x_val = vals[0]['value']
        y_val = vals[1]['value']
        f_xy_val = vals[2]['value']

        self.assertEqual(x_val, np.array([1.]))
        self.assertEqual(y_val, np.array([1.]))
        self.assertEqual(f_xy_val, np.array([27.]))

    def test_driver_root_index_case(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        recorder = om.SqliteRecorder("cases.sql")
        prob.driver = om.DOEDriver(om.PlackettBurmanGenerator())
        prob.driver.add_recorder(recorder)
        prob.model.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        data_dict = _get_viewer_data("cases.sql", case_id=6)

        vals = data_dict['tree']['children'][2]['children']
        x_val = vals[0]['value']
        y_val = vals[1]['value']
        f_xy_val = vals[2]['value']

        self.assertEqual(x_val, np.array([1.]))
        self.assertEqual(y_val, np.array([1.]))
        self.assertEqual(f_xy_val, np.array([27.]))

    def test_root_index_case(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        recorder = om.SqliteRecorder("cases.sql")
        prob.driver = om.DOEDriver(om.PlackettBurmanGenerator())
        prob.model.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        data_dict = _get_viewer_data("cases.sql", case_id=3)

        vals = data_dict['tree']['children'][2]['children']
        x_val = vals[0]['value']
        y_val = vals[1]['value']
        f_xy_val = vals[2]['value']

        self.assertEqual(x_val, np.array([1.]))
        self.assertEqual(y_val, np.array([1.]))
        self.assertEqual(f_xy_val, np.array([27.]))

    def test_auto_ivc_case(self):
        prob = SellarProblem(SellarDerivativesGrouped)

        prob.driver = om.ScipyOptimizeDriver(tol=1e-9, disp=False)
        prob.driver.add_recorder(self.recorder)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)
        first_case = cr.list_cases()[0]

        data_dict = _get_viewer_data(self.filename, case_id=first_case)

        vals = data_dict['tree']['children'][0]['children']
        ivc_0_val = vals[0]['value']
        ivc_1_val = vals[1]['value']

        self.assertEqual(ivc_0_val, [5., 2])
        self.assertEqual(ivc_1_val, np.array([1.]))

    def test_problem_case(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = om.DOEDriver(om.PlackettBurmanGenerator())
        prob.add_recorder(self.recorder)
        prob.recording_options['record_inputs'] = True

        prob.setup()
        prob.run_driver()
        prob.record('final')
        prob.cleanup()

        data_dict = _get_viewer_data(self.filename, case_id='final')

        vals = data_dict['tree']['children'][2]['children']
        x_val = vals[0]['value']
        y_val = vals[1]['value']
        f_xy_val = vals[2]['value']

        self.assertEqual(x_val, np.array([1.]))
        self.assertEqual(y_val, np.array([1.]))
        self.assertEqual(f_xy_val, np.array([27.]))