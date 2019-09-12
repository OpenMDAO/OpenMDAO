"""Example script for meta model viewer."""
import numpy as np
import openmdao.api as om
from openmdao.visualization.meta_model_viewer.meta_model_visualization import MetaModelVisualization

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

# Surrogate Model
interp.options['default_surrogate'] = om.ResponseSurface()

prob = om.Problem()
prob.model.add_subsystem('interp', interp)
prob.setup()
prob.final_setup()
