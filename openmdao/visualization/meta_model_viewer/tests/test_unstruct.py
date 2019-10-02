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

        adjusted_points = MetaModelVisualization(self.mm)
        adjusted_points.input_point_list = [1.619333019591837, 0.01423411, 0.02435233]
        adjusted_points.dist_range = 0.75

        new_points = adjusted_points._unstructured_training_points(compute_distance=True, source='right')

        # Make sure that arrays equal each other to the 6th decimal place
        assert_almost_equal(known_points, new_points, decimal=4)

    def test_in_between_training_points_bottom(self):

        filename = os.path.join(self.csv_dir, 'unstructured_test_points_bottom.csv')

        known_points = np.genfromtxt(
            filename, delimiter=',', skip_header=1)

        adjusted_points = MetaModelVisualization(self.mm)
        adjusted_points.input_point_list = [0.04203304, 2.043553874897959, 0.02435233]
        adjusted_points.dist_range = 0.75

        new_points = adjusted_points._unstructured_training_points(compute_distance=True, source='bottom')

        # Make sure that arrays equal each other to the 6th decimal place
        assert_almost_equal(known_points, new_points, decimal=4)

    def test_alpha_transparency(self):

        adjusted_points = MetaModelVisualization(self.mm)
        adjusted_points.input_point_list = [0.04203304, 2.043553874897959, 0.02435233]
        adjusted_points.dist_range = 0.75

        known_points_right = np.array([0.86567999, 0.78348615, 0.63977714, 0.62941009, 0.62587479,
                                    0.56152768, 0.50987912, 0.49376081, 0.48388929, 0.46630135,
                                    0.46401112, 0.4598095 , 0.41160183, 0.39350238, 0.39026133,
                                    0.38265734, 0.38192589, 0.36878817, 0.34231522, 0.32336566,
                                    0.30383293, 0.29137251, 0.28730615, 0.28575177, 0.26159435,
                                    0.26053391, 0.2252101 , 0.18561353, 0.17247848, 0.13633159,
                                    0.10337788, 0.10021363, 0.08660427, 0.08069673, 0.06691631,
                                    0.0547412 , 0.04688812, 0.04670322, 0.01284644, 0.00901997,
                                    0.00842558, 0.00386698])

        known_points_bottom = np.array([0.765526225, 0.738928771, 0.726005596, 0.715400685,
                                        0.649085936, 0.640394534, 0.628033739, 0.613040224,
                                        0.583063429, 0.439487175, 0.368150584, 0.343219846,
                                        0.323457648, 0.299935318, 0.252810468, 0.244806901,
                                        0.243472059, 0.236658572, 0.233785726, 0.222517297,
                                        0.208587812, 0.199369605, 0.195640700, 0.180272582,
                                        0.174355523, 0.152993107, 0.128729120, 0.128385050,
                                        0.128254288, 0.105788052, 0.101550328, 0.0844650938,
                                        0.0639579453, 0.0334477759, 0.0197405899, 0.000640993538])

        right_points = adjusted_points._unstructured_training_points(compute_distance=True, source='right')
        right_plot = adjusted_points._right_plot()
        right_transparency = adjusted_points.right_alphas

        bottom_points = adjusted_points._unstructured_training_points(compute_distance=True, source='bottom')
        bottom_plot = adjusted_points._bottom_plot()
        bottom_transparency = adjusted_points.bottom_alphas

        # Make sure that arrays equal each other to the 6th decimal place
        assert_almost_equal(known_points_right, right_transparency, decimal=5)
        assert_almost_equal(known_points_bottom, bottom_transparency, decimal=5)


if __name__ == '__main__':
    unittest.main()
