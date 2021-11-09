import unittest

import numpy as np

try:
    import bokeh
except ImportError:
    bokeh = None

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
class TestCaseViewer(unittest.TestCase):

    def setUp(self):
        self.filename = "test_case.sql"
        self.recorder = om.SqliteRecorder(self.filename)

    def test_case_viewer_data(self):
        prob = om.Problem()

        prob.model.add_subsystem('paraboloid', om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('paraboloid.x', lower=-50, upper=50)
        prob.model.add_design_var('paraboloid.y', lower=-50, upper=50)
        prob.model.add_objective('paraboloid.f')

        prob.driver.add_recorder(self.recorder)

        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['record_desvars'] = True

        prob.setup()

        prob.set_val('paraboloid.x', 3.0)
        prob.set_val('paraboloid.y', -4.0)

        prob.run_driver()

        cv = om.CaseViewer(self.filename, testing=True)

        circ_data = cv.circle_data.data

        # Test non vectorized data
        self.assertEqual(circ_data['x_vals'], [3.0])
        self.assertEqual(circ_data['y_vals'], [0.0])
        self.assertEqual(circ_data['cases'], ['rank0:ScipyOptimize_SLSQP|0'])
        self.assertEqual(circ_data['color'], ['#1f77b4'])

        # Test x and y value changes, case iterations and case iteration with min/max option
        cv.io_select_x.value = 'paraboloid.f'
        cv.io_select_y.value = 'Case Iterations'
        cv.case_iter_select.value = "Min/Max"
        cv.case_select.value = ['0', '1', '2', '3', '4']

        circ_data = cv.circle_data.data

        assert_near_equal(circ_data['x_vals'], [-15.0, -15.0, -27.0, -26.217711652253072,
                                                -27.32518474221697])
        self.assertEqual(circ_data['y_vals'], [0, 1, 2, 3, 4])
        self.assertEqual(circ_data['cases'], ['rank0:ScipyOptimize_SLSQP|0',
                                              'rank0:ScipyOptimize_SLSQP|1',
                                              'rank0:ScipyOptimize_SLSQP|2',
                                              'rank0:ScipyOptimize_SLSQP|3',
                                              'rank0:ScipyOptimize_SLSQP|4'])
        self.assertEqual(circ_data['color'], ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78',
                                              '#2ca02c'])

    def test_vectorized_case_data(self):

        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExecComp('v = v0 - 9.81 * t',
                                                        v={'shape': (100,), 'val': np.ones(100)},
                                                        t={'shape': (100,), 'val': np.linspace(0, 10, 100)},
                                                        v0={'shape': (1,), 'val':1.0}),
                                      promotes_inputs=['v0', 't'],
                                      promotes_outputs=['v'])

        C2 = prob.model.add_subsystem('C2', om.ExecComp('d = ((v0 + v) / 2) * t',
                                                        t={'shape': (100,), 'val': np.linspace(0, 10, 100)},
                                                        d={'shape': (100,), 'val': np.ones(100)},
                                                        v={'shape': (100,), 'val': np.ones(100)},
                                                        v0={'shape': (1,), 'val':1.0}),
                                      promotes_inputs=['v0', 'v', 't'],
                                      promotes_outputs=['d'])

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_objective('d', index=-1, scaler=-1)
        prob.model.add_design_var('v0', lower=0, upper=200)
        prob.model.add_constraint('v', indices=[-1], equals=0)

        prob.driver.add_recorder(self.recorder)
        prob.add_recorder(self.recorder)

        prob.driver.recording_options['includes'] = []
        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['record_inputs'] = True

        prob.setup()
        prob.set_val('v0', 100)
        prob.run_driver()

        cv = om.CaseViewer(self.filename, testing=True)

        cv.io_select_x.value = 'v'
        cv.io_select_y.value = 'd'
        cv.case_select.value = ['0', '1', '2']
        multi_line_data = cv.multi_line_data.data

        self.assertEqual(np.array(multi_line_data['x_vals']).shape, (3, 100))
        self.assertEqual(np.array(multi_line_data['y_vals']).shape, (3, 100))

    def test_warning_notes(self):
        prob = om.Problem()
        C1 = prob.model.add_subsystem('C1', om.ExecComp('v = v0 - 9.81 * t',
                                                        v={'shape': (100,), 'val': np.ones(100)},
                                                        t={'shape': (100,), 'val': np.linspace(0, 10, 100)},
                                                        v0={'shape': (1,), 'val':1.0}),
                                      promotes_inputs=['v0', 't'],
                                      promotes_outputs=['v'])

        C2 = prob.model.add_subsystem('C2', om.ExecComp('d = ((v0 + v) / 2) * t',
                                                        t={'shape': (100,), 'val': np.linspace(0, 10, 100)},
                                                        d={'shape': (100,), 'val': np.ones(100)},
                                                        v={'shape': (100,), 'val': np.ones(100)},
                                                        v0={'shape': (1,), 'val':1.0}),
                                      promotes_inputs=['v0', 'v', 't'],
                                      promotes_outputs=['d'])

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_objective('d', index=-1, scaler=-1)
        prob.model.add_design_var('v0', lower=0, upper=200)
        prob.model.add_constraint('v', indices=[-1], equals=0)

        prob.driver.add_recorder(self.recorder)
        prob.add_recorder(self.recorder)

        prob.driver.recording_options['includes'] = []
        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['record_inputs'] = True

        prob.setup()
        prob.set_val('v0', 100)
        prob.run_driver()

        cv = om.CaseViewer(self.filename, testing=True)

        cv._case_plot_calc(np.array([[0., 0, 0]]), np.array([[1, 2, 3]]))

        self.assertEqual(cv.warning_box.text, "NOTE: One or more variables are 0 arrays. "
                         "Select a different case or variable")

        # Test x and y value changes, case iterations and case iteration with min/max option
        cv.io_select_x.value = 'Number of Points'
        cv.io_select_y.value = 'Case Iterations'
        self.assertEqual(cv.warning_box.text, "NOTE: Cannot compare Number of Points to Case "
                         "Iterations")

        cv = om.CaseViewer(self.filename, testing=True)
        cv.io_select_x.value = 'Number of Points'
        cv.io_select_y.value = 'Case Iterations'

    def test_zero_array_warning(self):

        prob = om.Problem()

        C1 = prob.model.add_subsystem('C1', om.ExecComp('v = v0 - 9.81 * t',
                                                        v={'shape': (100,), 'val': np.ones(100)},
                                                        t={'shape': (100,), 'val': np.linspace(0, 0, 100)},
                                                        v0={'shape': (1,), 'val':1.0}),
                                        promotes_inputs=['v0', 't'],
                                        promotes_outputs=['v'])

        C2 = prob.model.add_subsystem('C2', om.ExecComp('d = ((v0 + v) / 2) * t',
                                                        t={'shape': (100,), 'val': np.linspace(0, 0, 100)},
                                                        d={'shape': (100,), 'val': np.zeros(100)},
                                                        v={'shape': (100,), 'val': np.zeros(100)},
                                                        v0={'shape': (1,), 'val':1.0}),
                                        promotes_inputs=['v0', 'v', 't'],
                                        promotes_outputs=['d'])

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_objective('d', index=-1, scaler=-1)
        prob.model.add_design_var('v0', lower=0, upper=0)
        prob.model.add_constraint('v', indices=[-1], equals=0)

        prob.driver.add_recorder(self.recorder)
        prob.add_recorder(self.recorder)

        prob.driver.recording_options['includes'] = []
        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['record_inputs'] = True

        prob.setup()
        prob.set_val('v0', 100)
        prob.run_driver()

        cv = om.CaseViewer(self.filename, testing=True)

        cv.io_select_x.value = 'd'
        cv.io_select_y.value = 'd'

        self.assertEqual(cv.warning_box.text, "NOTE: Both X and Y values contain zeros for values, "
                                              "unable to plot")

        cv._line_color_list(np.ones(266))
        self.assertEqual(cv.warning_box.text, "NOTE: Cannot compare more than 256 cases")

