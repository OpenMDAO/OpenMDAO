import unittest

import numpy as np
import openmdao.api as om
from openmdao.visualization.meta_model_viewer.meta_model_visualization import MetaModelVisualization


class UnstructuredMetaModelCompTests(unittest.TestCase):

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
        training_points_output = viz._training_points()

        self.assertTrue(training_points_output.shape[1] == 5)

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
        training_points_output = viz._training_points()

        for i in range(0, 4):
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




if __name__ == '__main__':
    unittest.main()
