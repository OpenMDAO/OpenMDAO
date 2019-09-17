import numpy as np
import openmdao.api as om
from openmdao.visualization.meta_model_viewer.meta_model_visualization import MetaModelVisualization

# create input param training data, of sizes 25, 5, and 10 points resp.
p1 = np.linspace(0, 200, 25)
p2 = np.linspace(-10, 10, 5)
p3 = np.linspace(0, 1, 10)

# can use meshgrid to create a 3D array of test data
P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
f = np.sqrt(P1) + P2 * P3

# verify the shape matches the order and size of the input params
print(f.shape)

# Create regular grid interpolator instance
interp = om.MetaModelStructuredComp(method='cubic')
interp.add_input('p1', 0.5, training_data=p1)
interp.add_input('p2', 0.0, training_data=p2)
interp.add_input('p3', 3.14, training_data=p3)

interp.add_output('f', 0.0, training_data=f)

# Set up the OpenMDAO model
model = om.Group()
model.add_subsystem('comp', interp, promotes=["*"])
prob = om.Problem(model)
prob.setup()

# set inputs
prob['p1'] = 55.12
prob['p2'] = -2.14
prob['p3'] = 0.323

prob.run_model()

computed = prob['f']
actual = 6.73306472
viz = MetaModelVisualization(interp)
# we can verify all gradients by checking against finite-difference
prob.check_partials(compact_print=True)
