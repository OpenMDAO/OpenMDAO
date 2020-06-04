""" Unit tests for unstructured metamodels in view_mm. """
import unittest
import os

import numpy as np
from numpy.testing import assert_almost_equal

try:
    import bokeh
    from openmdao.visualization.meta_model_viewer.meta_model_visualization import MetaModelVisualization
except ImportError:
    bokeh = None

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

@unittest.skipUnless(bokeh, "Bokeh is required")
class UnstructuredMetaModelCompTests(unittest.TestCase):

    csv_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'known_data_point_files')

    def setUp(self):
        self.mm = mm = om.MetaModelUnStructuredComp()

        filename = os.path.join(self.csv_dir, 'unstructured_data_points.csv')

        # Training Data
        x_train1 = np.genfromtxt(
            filename, delimiter=',', usecols=0)
        x_train2 = np.genfromtxt(
            filename, delimiter=',', usecols=1)
        x_train3 = np.genfromtxt(
            filename, delimiter=',', usecols=2)
        y = np.sin(x_train1 * x_train2 * x_train3)

        # Inputs
        mm.add_input('input_1', 0., training_data=x_train1)
        mm.add_input('input_2', 0., training_data=x_train2)
        mm.add_input('input_3', 0., training_data=x_train3)

        # Outputs
        mm.add_output('output_1', 0., training_data=y)

        # Surrogate Model
        mm.options['default_surrogate'] = om.KrigingSurrogate()

        prob = om.Problem()
        prob.model.add_subsystem('mm', mm)
        prob.setup()
        prob.final_setup()

    def test_missing_training_data_in_parameter(self):

        # Model
        interp = om.MetaModelUnStructuredComp()

        # Training Data
        x_train = np.linspace(0, 10, 20)
        y_train = np.linspace(0, 20, 20)

        # Inputs
        interp.add_input('simple_x', 0., training_data=x_train)
        interp.add_input('sin_x', 0.)

        #Outputs
        interp.add_output('cos_x', 0., training_data=.5*np.cos(y_train))

        # Surrogate Model
        interp.options['default_surrogate'] = om.ResponseSurface()

        prob = om.Problem()
        prob.model.add_subsystem('interp', interp)
        prob.setup()

        with self.assertRaises(Exception) as context:
            viz = MetaModelVisualization(interp)

        msg = "No training data present for one or more parameters"
        self.assertTrue(msg in str(context.exception))

    def test_single_input_parameter(self):

        # Model
        interp = om.MetaModelUnStructuredComp()

        # Training Data
        x_train = np.linspace(0, 10, 20)
        y_train = np.linspace(0, 20, 20)

        # Inputs
        interp.add_input('simple_x', 0., training_data=x_train)

        #Outputs
        interp.add_output('cos_x', 0., training_data=.5*np.cos(y_train))

        # Surrogate Model
        interp.options['default_surrogate'] = om.ResponseSurface()

        prob = om.Problem()
        prob.model.add_subsystem('interp', interp)
        prob.setup()

        with self.assertRaises(Exception) as context:
            viz = MetaModelVisualization(interp)

        msg = 'Must have more than one input value'
        self.assertTrue(msg in str(context.exception))


    def test_training_point_array_width(self):

        # Model
        interp = om.MetaModelUnStructuredComp()

        # Training Data
        x_train = np.linspace(0, 10, 20)
        y_train = np.linspace(0, 20, 20)

        # Inputs
        interp.add_input('x', 0., training_data=x_train)
        interp.add_input('y', 0., training_data=x_train)

        #Outputs
        interp.add_output('cos_x', 0., training_data=.5*np.cos(y_train))

        # Surrogate Model
        interp.options['default_surrogate'] = om.ResponseSurface()

        prob = om.Problem()
        prob.model.add_subsystem('interp', interp)
        prob.setup()

        viz = MetaModelVisualization(interp)
        training_points_output = viz._unstructured_training_points()

        self.assertTrue(training_points_output.shape[1] == 2)

    def test_training_point_array_for_nan_values(self):

        # Model
        interp = om.MetaModelUnStructuredComp()

        # Training Data
        x_train = np.linspace(0, 10, 20)
        y_train = np.linspace(0, 20, 20)

        # Inputs
        interp.add_input('x', 0., training_data=x_train)
        interp.add_input('y', 0., training_data=x_train)

        #Outputs
        interp.add_output('cos_x', 0., training_data=.5*np.cos(y_train))

        # Surrogate Model
        interp.options['default_surrogate'] = om.ResponseSurface()

        prob = om.Problem()
        prob.model.add_subsystem('interp', interp)
        prob.setup()

        viz = MetaModelVisualization(interp)
        training_points_output = viz._unstructured_training_points()

        for i in range(0, 2):
            self.assertFalse(np.any(np.isnan(training_points_output[:, i])))

    def test_make_predictions(self):

        # Model
        interp = om.MetaModelUnStructuredComp()

        # Training Data
        x_train = np.linspace(0, 10, 20)
        y_train = np.linspace(10, 20, 20)

        # Inputs
        interp.add_input('simple_x', 0., training_data=x_train)
        interp.add_input('sin_x', 0., training_data=x_train)

        #Outputs
        interp.add_output('cos_x', 0., training_data=.5*np.cos(y_train))

        # Surrogate Model
        interp.options['default_surrogate'] = om.ResponseSurface()

        prob = om.Problem()
        prob.model.add_subsystem('interp', interp)
        prob.setup()

        viz = MetaModelVisualization(interp)
        resolution = 50
        data = dict({'simple_x': np.array([np.random.rand(resolution**2, 1)]),
                     'sin_x': np.array([np.random.rand(resolution**2, 1)])})
        pred_array = viz._make_predictions(data)

        self.assertTrue(pred_array.shape == (resolution**2, 1))

    def test_working_response_surface(self):
        # Model
        interp = om.MetaModelUnStructuredComp()

        # Training Data
        x_train1 = np.linspace(0, 10, 20)
        x_train2 = np.linspace(0, 20, 20)
        x_train3 = np.linspace(0, 30, 20)
        x_train4 = np.linspace(0, 40, 20)
        y_train = np.linspace(10, 20, 20)

        # Inputs
        interp.add_input('input_1', 0., training_data=x_train1)
        interp.add_input('input_2', 0., training_data=x_train2)
        interp.add_input('input_3', 0., training_data=x_train3)
        interp.add_input('input_4', 0., training_data=x_train4)

        # Outputs
        interp.add_output('output_1', 0., training_data=.5 * np.cos(y_train))
        interp.add_output('output_2', 0., training_data=.5 * np.sin(y_train))

        # Surrogate Model
        interp.options['default_surrogate'] = om.ResponseSurface()

        prob = om.Problem()
        prob.model.add_subsystem('interp', interp)
        prob.setup()
        prob.final_setup()

    def test_not_top_level_prob(self):
        # Model
        interp = om.MetaModelUnStructuredComp()

        # Training Data
        x_train1 = np.linspace(0, 10, 20)
        x_train2 = np.linspace(0, 20, 20)
        x_train3 = np.linspace(0, 30, 20)
        x_train4 = np.linspace(0, 40, 20)
        y_train = np.linspace(10, 20, 20)

        # Inputs
        interp.add_input('input_1', 0., training_data=x_train1)
        interp.add_input('input_2', 0., training_data=x_train2)
        interp.add_input('input_3', 0., training_data=x_train3)
        interp.add_input('input_4', 0., training_data=x_train4)

        # Outputs
        interp.add_output('output_1', 0., training_data=.5 * np.cos(y_train))
        interp.add_output('output_2', 0., training_data=.5 * np.sin(y_train))

        # Surrogate Model
        interp.options['default_surrogate'] = om.ResponseSurface()

        prob = om.Problem(model=interp)
        prob.setup()
        prob.final_setup()


    def test_in_between_training_points_right(self):

        filename = os.path.join(self.csv_dir, 'unstructured_test_points_right.csv')

        known_points = np.genfromtxt(
            filename, delimiter=',', skip_header=1)
        known_points = np.delete(known_points, 2, 1)
        known_points = known_points[known_points[:,1].argsort()[::-1]]

        adjusted_points = MetaModelVisualization(self.mm)
        adjusted_points.input_point_list = [1.619333019591837, 0.01423411, 0.02435233]
        adjusted_points.dist_range = 0.75

        new_points = adjusted_points._unstructured_training_points(compute_distance=True, source='right')
        new_points = np.delete(new_points, 2, 1)
        new_points = new_points[new_points[:,1].argsort()[::-1]]

        assert_almost_equal(known_points, new_points, decimal=4)

    def test_in_between_training_points_bottom(self):

        filename = os.path.join(self.csv_dir, 'unstructured_test_points_bottom.csv')

        known_points = np.genfromtxt(
            filename, delimiter=',', skip_header=1)
        known_points = np.delete(known_points, 2, 1)
        known_points = known_points[known_points[:,0].argsort()[::-1]]

        adjusted_points = MetaModelVisualization(self.mm)
        adjusted_points.input_point_list = [0.04203304, 2.043553874897959, 0.02435233]
        adjusted_points.dist_range = 0.75

        new_points = adjusted_points._unstructured_training_points(compute_distance=True, source='bottom')
        new_points = np.delete(new_points, 2, 1)
        new_points = new_points[new_points[:,0].argsort()[::-1]]

        assert_almost_equal(known_points, new_points, decimal=4)

    def test_alpha_transparency(self):

        adjusted_points = MetaModelVisualization(self.mm)
        adjusted_points.input_point_list = [0.04203304, 2.043553874897959, 0.02435233]
        adjusted_points.dist_range = 0.75

        known_points_right = np.array([0.86567989, 0.78348589, 0.639777  , 0.62940986, 0.62587447,
                                        0.56152752, 0.50987878, 0.49376071, 0.48388914, 0.46630121,
                                        0.4640111 , 0.45980945, 0.41160175, 0.39350227, 0.39026132,
                                        0.38265725, 0.38192591, 0.36878805, 0.34231505, 0.32336551,
                                        0.30383287, 0.29137243, 0.28730608, 0.2857517 , 0.26159429,
                                        0.26053377, 0.22521007, 0.18561353, 0.17247835, 0.13633153,
                                        0.10337783, 0.10021358, 0.08660427, 0.08069668, 0.06691626,
                                        0.05474111, 0.04688808, 0.04670316, 0.01284643, 0.00901992,
                                        0.00842553, 0.00386693])

        known_points_bottom = np.array([7.65526230e-01, 7.38928632e-01, 7.26005432e-01, 7.15400571e-01,
                                        6.49085815e-01, 6.40394477e-01, 6.28033520e-01, 6.13040213e-01,
                                        5.83063203e-01, 4.39487058e-01, 3.68150531e-01, 3.43219760e-01,
                                        3.23457593e-01, 2.99935268e-01, 2.52810393e-01, 2.44806774e-01,
                                        2.43471983e-01, 2.36658494e-01, 2.33785648e-01, 2.22517218e-01,
                                        2.08587699e-01, 1.99369532e-01, 1.95640614e-01, 1.80272528e-01,
                                        1.74355451e-01, 1.52993016e-01, 1.28729050e-01, 1.28385003e-01,
                                        1.28254220e-01, 1.05787985e-01, 1.01550282e-01, 8.44650788e-02,
                                        6.39578812e-02, 3.34477398e-02, 1.97405267e-02, 6.40957590e-04])

        right_points = adjusted_points._unstructured_training_points(compute_distance=True, source='right')
        right_plot = adjusted_points._right_plot()
        right_transparency = adjusted_points.right_alphas

        bottom_points = adjusted_points._unstructured_training_points(compute_distance=True, source='bottom')
        bottom_plot = adjusted_points._bottom_plot()
        bottom_transparency = adjusted_points.bottom_alphas

        assert_near_equal(right_transparency, known_points_right, 1.1e-02)
        assert_near_equal(bottom_transparency, known_points_bottom, 1.6e-02)



if __name__ == '__main__':
    unittest.main()
