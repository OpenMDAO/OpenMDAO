import numpy as np
from math import pi
import openmdao.api as om

# Model
interp = om.MetaModelUnStructuredComp()

# Training Data
x_train1 = np.random.uniform(0, pi, 100)
x_train2 = np.random.uniform(0, pi, 100)
x_train3 = np.random.uniform(0, pi, 100)
y = np.sin(x_train1 * x_train2 * x_train3)

# Inputs
interp.add_input('input_1', 0., training_data=x_train1)
interp.add_input('input_2', 0., training_data=x_train2)
interp.add_input('input_3', 0., training_data=x_train3)

# Outputs
interp.add_output('output_1', 0., training_data=y)

# Surrogate Model
interp.options['default_surrogate'] = om.KrigingSurrogate()

prob = om.Problem()
prob.model.add_subsystem('interp', interp)
prob.setup()
prob.final_setup()