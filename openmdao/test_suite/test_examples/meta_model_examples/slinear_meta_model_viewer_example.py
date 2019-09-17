import numpy as np
import openmdao.api as om
from openmdao.visualization.meta_model_viewer.meta_model_visualization import MetaModelVisualization

# Create regular grid interpolator instance
xor_interp = om.MetaModelStructuredComp(method='slinear')

# set up inputs and outputs
xor_interp.add_input('x', 0.0, training_data=np.array([0.0, 1.0]), units=None)
xor_interp.add_input('y', 1.0, training_data=np.array([0.0, 1.0]), units=None)

xor_interp.add_output('xor', 1.0, training_data=np.array([[0.0, 1.0], [1.0, 0.0]]), units=None)

# Set up the OpenMDAO model
prob = om.Problem()
prob.model.add_subsystem('comp', xor_interp, promotes=["*"])
prob.setup()

# Now test out a 'fuzzy' XOR
prob['x'] = 0.9
prob['y'] = 0.001242

prob.run_model()

computed = prob['xor']
actual = 0.8990064

print(computed)


# we can verify all gradients by checking against finite-difference
prob.check_partials(compact_print=True)



