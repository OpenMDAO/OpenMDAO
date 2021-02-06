""" Unit tests for the problem interface."""
import numpy as np
import openmdao.api as om
# from openmdao.visualization.n2_viewer.n2_viewer import n2
# Whether to pop up a browser window for each N2
DEBUG_BROWSER = False
# set DEBUG_FILES to True if you want to view the generated HTML file(s)
DEBUG_FILES = False
"""Check against the scipy solver."""
SIZE = 10
model = om.Group()
x = np.array([1, 2, -3])
A = np.array([[5.0, -3.0, 2.0], [1.0, 7.0, -4.0], [1.0, 0.0, 8.0]])
x = np.random.rand(SIZE) * 1e20
A = np.random.rand(SIZE, SIZE)
b = A.dot(x)
model.add_subsystem('p1', om.IndepVarComp('A', A))
model.add_subsystem('p2', om.IndepVarComp('b', b))

lingrp = model.add_subsystem('lingrp', om.Group(), promotes=['*'])
lingrp.add_subsystem('lin', om.LinearSystemComp(size=SIZE))
model.connect('p1.A', 'lin.A')
model.connect('p2.b', 'lin.b')
prob = om.Problem(model)
prob.setup()
prob.final_setup()

lingrp.linear_solver = om.ScipyKrylov()
prob.setup()


