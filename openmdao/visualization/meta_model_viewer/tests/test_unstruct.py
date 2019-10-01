import unittest

import numpy as np

try:
    import bokeh
    from openmdao.visualization.meta_model_viewer.meta_model_visualization import MetaModelVisualization
except ImportError:
    bokeh = None

import openmdao.api as om

@unittest.skipUnless(bokeh, "Bokeh is required")
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


if __name__ == '__main__':
    unittest.main()
