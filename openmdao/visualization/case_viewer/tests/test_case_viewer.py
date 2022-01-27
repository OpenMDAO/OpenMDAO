import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

try:
    import bokeh
except ModuleNotFoundError:
    bokeh = None

@use_tempdirs
@unittest.skipIf(bokeh is None, "Bokeh is required")
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
        prob.add_recorder(self.recorder)

        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['record_desvars'] = True

        prob.setup()

        prob.set_val('paraboloid.x', 3.0)
        prob.set_val('paraboloid.y', -4.0)

        prob.run_driver()
        prob.record("after_run_driver")

        # Check opening values
        cv = om.CaseViewer(self.filename)

        circ_data = cv.circle_data.data

        # Test non vectorized data
        self.assertEqual(circ_data['x_vals'], [3.0])
        self.assertEqual(circ_data['y_vals'], [0.0])
        self.assertEqual(circ_data['cases'], ['rank0:ScipyOptimize_SLSQP|0'])
        self.assertEqual(circ_data['color'], ['#1f77b4'])

        # Test X and Y value changes, case iterations on X and case iteration with min/max option on Y
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

       # Test Y and X value changes, case iterations on X and case iteration with min/max option on Y
        cv.io_select_x.value = 'Case Iterations'
        cv.io_select_y.value = 'paraboloid.f'
        cv.case_iter_select.value = "Min/Max"
        cv.case_select.value = ['0', '1', '2', '3', '4']

        circ_data = cv.circle_data.data

        assert_near_equal(circ_data['y_vals'], [-15.0, -15.0, -27.0, -26.217711652253072,
                                                -27.32518474221697])
        self.assertEqual(circ_data['x_vals'], [0, 1, 2, 3, 4])
        self.assertEqual(circ_data['cases'], ['rank0:ScipyOptimize_SLSQP|0',
                                              'rank0:ScipyOptimize_SLSQP|1',
                                              'rank0:ScipyOptimize_SLSQP|2',
                                              'rank0:ScipyOptimize_SLSQP|3',
                                              'rank0:ScipyOptimize_SLSQP|4'])
        self.assertEqual(circ_data['color'], ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78',
                                              '#2ca02c'])

        # Test change of source
        cv = om.CaseViewer(self.filename)

        cv.source_select.value = 'problem'
        circ_data = cv.circle_data.data

        assert_near_equal(circ_data['x_vals'], [6.666666666666667])
        self.assertEqual(circ_data['y_vals'], [0.])
        self.assertEqual(circ_data['color'], ['#1f77b4'])
        self.assertEqual(circ_data['cases'], ['rank0:ScipyOptimize_SLSQP|0'])

        # Test single Problem case iteration error
        cv = om.CaseViewer(self.filename)

        cv.source_select.value = 'problem'
        cv.case_select.value = ['0']
        cv.io_select_y.value = 'Case Iterations'

        self.assertEqual(cv.warning_box.text, "Case Iterations needs 2 or more cases to function")

        # Test single Problem case iteration error (Reversed)
        cv = om.CaseViewer(self.filename)

        cv.source_select.value = 'problem'
        cv.case_select.value = ['0']
        cv.io_select_x.value = 'Case Iterations'

        self.assertEqual(cv.warning_box.text, "Case Iterations needs 2 or more cases to function")

        # Test change of X values and check y value filtering
        cv = om.CaseViewer(self.filename)

        cv.io_select_x.value = "paraboloid.f"
        cv.source_select.value = 'driver'

        self.assertEqual(cv.io_select_y.options, ['paraboloid.f', 'paraboloid.x', 'paraboloid.y',
                                                  'Variable Array Index', 'Case Iterations'])

        cv = om.CaseViewer(self.filename)
        cv.io_select_x.value = "Variable Array Index"
        cv.source_select.value = 'driver'

        self.assertEqual(cv.io_select_y.options['outputs'], ['paraboloid.f', 'paraboloid.x', 'paraboloid.y'])
        self.assertEqual(cv.io_select_y.options['Other'], ['Variable Array Index', 'Case Iterations'])

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

        # Test X (v) and Y (num_points) opening values
        cv = om.CaseViewer(self.filename)

        cv.io_select_x.value = 'v'
        cv.io_select_y.value = 'Variable Array Index'
        cv.case_select.value = ['0', '1', '2']

        multi_line_data = cv.multi_line_data.data

        self.assertEqual(np.array(multi_line_data['x_vals']).shape, (3, 100))
        # Getting a cross section to check each line's 1st value
        assert_near_equal(np.array(multi_line_data['x_vals'])[:, 1],
                            np.array([99.00909091, 99.00909091, 97.10909091]), tolerance=1e-6)

        self.assertEqual(np.array(multi_line_data['y_vals']).shape, (3, 100))
        self.assertEqual(multi_line_data['y_vals'][0], list(np.arange(0, 100, 1.)))
        self.assertEqual(multi_line_data['color'], ['#1f77b4', '#aec7e8', '#ff7f0e'])
        self.assertEqual(multi_line_data['cases'], ['rank0:ScipyOptimize_SLSQP|0',
                                                    'rank0:ScipyOptimize_SLSQP|1',
                                                    'rank0:ScipyOptimize_SLSQP|2'])

        # Test Y (v) and x (num_points) opening values.
        cv = om.CaseViewer(self.filename)

        cv.io_select_y.value = 'v'
        cv.io_select_x.value = 'Variable Array Index'
        cv.case_select.value = ['0', '1', '2']

        multi_line_data = cv.multi_line_data.data

        self.assertEqual(np.array(multi_line_data['y_vals']).shape, (3, 100))
        assert_near_equal(np.array(multi_line_data['y_vals'])[:, 1],
                            np.array([99.00909091, 99.00909091, 97.10909091]), tolerance=1e-6)

        self.assertEqual(np.array(multi_line_data['x_vals']).shape, (3, 100))
        self.assertEqual(multi_line_data['x_vals'][0], list(np.arange(0, 100, 1.)))

        # Test Case Iterations (Min/Max)
        cv = om.CaseViewer(self.filename)

        cv.io_select_y.value = 'd'
        cv.io_select_x.value = 'Case Iterations'
        cv.case_select.value = ['0', '1', '2']

        multi_line_data = cv.multi_line_data.data

        self.assertEqual(np.array(multi_line_data['x_vals']).shape, (3, 100))

        self.assertEqual(np.array(multi_line_data['y_vals']).shape, (3, 100))
        assert_near_equal(np.array(multi_line_data['y_vals'])[:, 0:3],
                          np.array([[ 0.        , 10.05096419, 20.00183655],
                                    [ 0.        , 10.05096419, 20.00183655],
                                    [ 0.        ,  9.859045  , 19.61799816]]), tolerance=1e-6)

        # Test Case Iterations (Min/Max) reversed
        cv = om.CaseViewer(self.filename)

        cv.io_select_y.value = 'Case Iterations'
        cv.io_select_x.value = 'd'
        cv.case_select.value = ['0', '1', '2']

        multi_line_data = cv.multi_line_data.data

        self.assertEqual(np.array(multi_line_data['y_vals']).shape, (3, 100))

        self.assertEqual(np.array(multi_line_data['x_vals']).shape, (3, 100))
        assert_near_equal(np.array(multi_line_data['x_vals'])[:, 0:3],
                          np.array([[ 0.        , 10.05096419, 20.00183655],
                                    [ 0.        , 10.05096419, 20.00183655],
                                    [ 0.        ,  9.859045  , 19.61799816]]), tolerance=1e-6)


        # Test Case Iterations (Norm)
        cv = om.CaseViewer(self.filename)

        cv.io_select_y.value = 'd'
        cv.io_select_x.value = 'Case Iterations'
        cv.case_select.value = ['0', '1', '2']
        cv.case_iter_select.value = "Norm"

        multi_line_data = cv.multi_line_data.data

        self.assertEqual(multi_line_data['x_vals'][0], [0, 1, 2])
        self.assertEqual(multi_line_data['y_vals'][0], [3689.681336358587, 3689.681336358587, 3580.985941990675])

        # Test Case Iterations (Norm) reversed
        cv = om.CaseViewer(self.filename)

        cv.io_select_x.value = 'd'
        cv.io_select_y.value = 'Case Iterations'
        cv.case_select.value = ['0', '1', '2']
        cv.case_iter_select.value = "Norm"

        multi_line_data = cv.multi_line_data.data

        self.assertEqual(multi_line_data['y_vals'][0], [0, 1, 2])
        self.assertEqual(multi_line_data['x_vals'][0], [3689.681336358587, 3689.681336358587,
                                                        3580.985941990675])

        # Test Case Iterations (Vector Lines)
        cv = om.CaseViewer(self.filename)

        cv.io_select_x.value = 'd'
        cv.io_select_y.value = 'Case Iterations'
        cv.case_select.value = ['0', '1', '2']
        cv.case_iter_select.value = "Vector Lines"

        multi_line_data = cv.multi_line_data.data

        self.assertEqual(np.array(multi_line_data['x_vals']).shape, (100, 3))
        assert_near_equal(multi_line_data['x_vals'][1], [10.050964187327823, 10.050964187327823,
                                                         9.859044995408633], tolerance=1e-6)

        self.assertEqual(np.array(multi_line_data['y_vals']).shape, (100, 3))
        assert_near_equal(multi_line_data['y_vals'][1], [0, 1, 2])

        # Test Case Iterations (Vector Lines) reversed
        cv = om.CaseViewer(self.filename)

        cv.io_select_y.value = 'd'
        cv.io_select_x.value = 'Case Iterations'
        cv.case_select.value = ['0', '1', '2']
        cv.case_iter_select.value = "Vector Lines"

        multi_line_data = cv.multi_line_data.data

        self.assertEqual(np.array(multi_line_data['y_vals']).shape, (100, 3))
        assert_near_equal(multi_line_data['y_vals'][1], [10.050964187327823, 10.050964187327823,
                                                         9.859044995408633], tolerance=1e-6)

        self.assertEqual(np.array(multi_line_data['x_vals']).shape, (100, 3))
        assert_near_equal(multi_line_data['x_vals'][1], [0, 1, 2])

        # Test X and Y variable selection
        cv = om.CaseViewer(self.filename)

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

        cv = om.CaseViewer(self.filename)

        cv._case_plot_calc(np.array([[0., 0, 0]]), np.array([[1, 2, 3]]))

        self.assertEqual(cv.warning_box.text, "NOTE: One or more variables are 0 arrays. "
                         "Select a different case or variable")

        # Test x and y value changes, case iterations and case iteration with min/max option
        cv.io_select_x.value = 'Variable Array Index'
        cv.io_select_y.value = 'Case Iterations'
        self.assertEqual(cv.warning_box.text, "NOTE: Cannot compare Variable Array Index to Case "
                         "Iterations")

        cv = om.CaseViewer(self.filename)
        cv.io_select_x.value = 'Variable Array Index'
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

        cv = om.CaseViewer(self.filename)

        cv.io_select_x.value = 'd'
        cv.io_select_y.value = 'd'

        self.assertEqual(cv.warning_box.text, "NOTE: Both X and Y values contain zeros for values, "
                                              "unable to plot")

        cv._line_color_list(np.ones(266))
        self.assertEqual(cv.warning_box.text, "NOTE: Cannot compare more than 256 cases")

