""" Benchmark the structured metamodel interpolants against some standard problems.
"""
from time import time

import numpy as np
import matplotlib.pyplot as plt

import openmdao.api as om
# from openmdao.components.meta_model_structured_comp import ALL_METHODS
from openmdao.test_suite.components.branin import Branin
from openmdao.utils.general_utils import pad_name


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


def run_branin_2d(grid, training_data, num_eval=100, method='slinear'):

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
    mm = prob.model.add_subsystem('mm', om.MetaModelStructuredComp(method=method),
                                  promotes=['x0', 'x1'])
    mm.add_input('x0', 0.0, grid[0])
    mm.add_input('x1', 0.0, grid[1])
    mm.add_output('f', 0.0, training_data)

    prob.setup()

    t0 = time()
    for i, x0, in enumerate(x0_eval):
        for j, x1, in enumerate(x1_eval):
            prob['x0'] = x0
            prob['x1'] = x1
            prob.run_model()
            approx[i, j] = prob['mm.f']

    return approx, time() - t0

def train_branin_1d(num_train=10, num_eval=100):

    x0_min, x0_max = -5.0, 10.0
    x0_train = np.linspace(x0_min, x0_max, num_train)
    grid = (x0_train)
    x0_eval = np.linspace(x0_min, x0_max, num_eval)
    training_data = np.empty((num_train, ))
    actual = np.empty((num_eval, ))

    prob = om.Problem()
    ivc = om.IndepVarComp()
    ivc.add_output('x0', 0.0)

    prob.model.add_subsystem('p', ivc, promotes=['*'])
    prob.model.add_subsystem('comp', Branin(), promotes=['*'])

    prob.setup()

    for i, x0, in enumerate(x0_train):
        prob['x0'] = x0
        prob.run_model()
        training_data[i] = prob['f']

    for i, x0, in enumerate(x0_eval):
        prob['x0'] = x0
        prob.run_model()
        actual[i] = prob['f']

    return grid, training_data, actual


def run_branin_1d(grid, training_data, num_eval=100, method='slinear'):

    x0_min, x0_max = -5.0, 10.0
    x0_eval = np.linspace(x0_min, x0_max, num_eval)
    approx = np.empty((num_eval, ))

    prob = om.Problem()
    ivc = om.IndepVarComp()
    ivc.add_output('x0', 0.0)

    prob.model.add_subsystem('p', ivc, promotes=['*'])
    mm = prob.model.add_subsystem('mm', om.MetaModelStructuredComp(method=method),
                                  promotes=['x0'])
    mm.add_input('x0', 0.0, grid)
    mm.add_output('f', 0.0, training_data)

    prob.setup()

    t0 = time()
    for i, x0, in enumerate(x0_eval):
        prob['x0'] = x0
        prob.run_model()
        approx[i] = prob['mm.f']

    return approx, time() - t0

if __name__ == "__main__":

    num_train = 23
    num_eval = 155
    plot = False
    # Can be norm_error, max_error, time
    sort_by = 'norm_error'
    case = 'branin_2D'

    case_dict = {'branin_2D': (train_branin_2d, run_branin_2d),
                 'branin_1D': (train_branin_1d, run_branin_1d),
                 }

    f_train, f_eval = case_dict[case]

    grid, training_data, actual = f_train(num_train=num_train, num_eval=num_eval)

    methods = []
    error_norms = []
    error_inf_norms = []
    times = []

    ALL_METHODS = ('cubic', 'slinear', 'lagrange2', 'lagrange3', 'akima',
               'scipy_cubic', 'scipy_slinear', 'scipy_quintic')
    for j, method in enumerate(ALL_METHODS):
        approx, dt = f_eval(grid, training_data, num_eval=num_eval, method=method)
        rel_error = np.abs((approx - actual) / np.max(np.abs(actual)))
        error_norm = np.linalg.norm(rel_error) / num_eval
        error_inf_norm = np.max(rel_error)

        print(method, 'done')

        methods.append(method)
        error_norms.append(error_norm)
        error_inf_norms.append(error_inf_norm)
        times.append(dt)

        # only do this for 1D.
        if plot:
            plt.figure(j)
            x = np.linspace(0, 1, num_eval)
            plt.plot(x, actual)
            plt.plot(x, approx, 'r')
            plt.show()

    if sort_by == 'norm_error':
        sorted_idx = np.argsort(error_norms)
    elif sort_by == 'max_error':
        sorted_idx = np.argsort(error_inf_norms)
    elif sort_by == 'time':
        sorted_idx = np.argsort(times)

    print('\n')
    print('Problem:', case)
    print('Num train = ', num_train, ", Num eval = ", num_eval)
    print('  Sorted by', sort_by, '\n')
    print("Method          Norm Error       Max Error       Time")
    print('-' * 60)

    for idx in reversed(sorted_idx):
        method = pad_name(methods[idx], 15)
        error_norm = pad_name(str(np.round(error_norms[idx], 8)), 15)
        error_inf_norm = pad_name(str(np.round(error_inf_norms[idx], 8)), 15)
        dt = pad_name(str(np.round(times[idx], 8)), 15)

        print(method, error_norm, error_inf_norm, dt)


    print('done')