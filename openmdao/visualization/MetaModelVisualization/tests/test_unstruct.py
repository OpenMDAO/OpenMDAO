import unittest

import numpy as np
import openmdao.api as om
from main import UnstructuredMetaModelVisualization


class UnstructuredMetaModelCompTests(unittest.TestCase):

    def test_missing_training_data_in_parameter(self):
        
        # Model
        interp = om.MetaModelUnStructuredComp()

        # Training Data
        x_train = np.linspace(0,10,20)
        y_train = np.linspace(0,20,20)

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
            viz = UnstructuredMetaModelVisualization(prob, interp)

        msg = "No training data present for one or more parameters"
        self.assertTrue(msg in str(context.exception))

    def test_single_input_parameter(self):
        
        # Model
        interp = om.MetaModelUnStructuredComp()

        # Training Data
        x_train = np.linspace(0,10,20)
        y_train = np.linspace(0,20,20)

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
            viz = UnstructuredMetaModelVisualization(prob, interp)

        msg = 'Must have more than one input value'
        self.assertTrue(msg in str(context.exception))

if __name__ == '__main__':
    unittest.main()