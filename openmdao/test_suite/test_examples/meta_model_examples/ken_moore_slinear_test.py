from openmdao.test_suite.components.branin import Branin
from openmdao.utils.general_utils import pad_name
from time import time

import numpy as np
import matplotlib.pyplot as plt

import openmdao.api as om
from openmdao.visualization.meta_model_viewer.meta_model_visualization import MetaModelVisualization

num_train = 23
num_eval = 155

def train_branin_2d(num_train=10, num_eval=100):

    x0_min, x0_max = -5.0, 10.0
    x1_min, x1_max = 0.0, 15.0
    x0_train = np.linspace(x0_min, x0_max, num_train)
    x1_train = np.linspace(x1_min, x1_max, num_train)
    grid = (x0_train, x1_train)
    x0_eval = np.linspace(x0_min, x0_max, num_eval)
    x1_eval = np.linspace(x1_min, x1_max, num_eval)
    training_data = np.empty((num_train, num_train))
    actual = np.empty((num_eval, num_eval))

    prob = om.Problem()
    ivc = om.IndepVarComp()
    ivc.add_output('x0', 0.0)
    ivc.add_output('x1', 0.0)

    prob.model.add_subsystem('p', ivc, promotes=['*'])
    prob.model.add_subsystem('comp', Branin(), promotes=['*'])


    prob.setup()
    save = prob._post_setup_func
    prob._post_setup_func = None
    prob.final_setup()
    prob._post_setup_func = save

    for i, x0, in enumerate(x0_train):
        for j, x1, in enumerate(x1_train):
            prob['x0'] = x0
            prob['x1'] = x1
            prob.run_model()
            training_data[i, j] = prob['f']

    for i, x0, in enumerate(x0_eval):
        for j, x1, in enumerate(x1_eval):
            prob['x0'] = x0
            prob['x1'] = x1
            prob.run_model()
            actual[i, j] = prob['f']

    return grid, training_data, actual



# def run_branin_2d(grid, training_data, num_eval=100, method='slinear'):
train = train_branin_2d()
grid_result0 = train[0][0]
grid_result1 = train[0][1]
t_data = train[1]

x0_min, x0_max = -5.0, 10.0
x1_min, x1_max = 0.0, 15.0
x0_eval = np.linspace(x0_min, x0_max, num_eval)
x1_eval = np.linspace(x1_min, x1_max, num_eval)
approx = np.empty((num_eval, num_eval))

prob = om.Problem()
ivc = om.IndepVarComp()
ivc.add_output('x0', 0.0)
ivc.add_output('x1', 0.0)

prob.model.add_subsystem('p', ivc, promotes=['*'])
mm = prob.model.add_subsystem('mm', om.MetaModelStructuredComp(method='slinear'),
                                promotes=['x0', 'x1'])
mm.add_input('x0', 0.0, grid_result0)
mm.add_input('x1', 0.0, grid_result1)
mm.add_output('f', 0.0, t_data)

prob.setup()
prob.final_setup()

viz = MetaModelVisualization(mm)