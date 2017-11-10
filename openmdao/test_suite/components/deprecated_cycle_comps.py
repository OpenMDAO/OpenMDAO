"""Deprecated Components for use in `CycleGroup`. For details, see `CycleGroup`."""
from __future__ import division, print_function

from six.moves import range

import numpy as np

import unittest

from openmdao.components.deprecated_component import Component

from openmdao.test_suite.components.cycle_comps import \
    _compute_A, _compute_dA, array_idx


class DeprecatedCycleComp(Component):

    def _inputs_to_vector(self, inputs):
        var_shape = self.metadata['var_shape']
        num_var = self.metadata['num_var']
        size = np.prod(var_shape)
        x = np.zeros(num_var * size)
        for i in range(num_var):
            x_i = inputs[self._cycle_names['x'].format(i)].flat
            x[size * i:size * (i + 1)] = x_i

        return x

    def _vector_to_outputs(self, vec, outputs):
        var_shape = self.metadata['var_shape']
        num_var = self.metadata['num_var']
        size = np.prod(var_shape)
        for i in range(num_var):
            y_i = vec[size * i:size * (i + 1)].reshape(var_shape)
            outputs[self._cycle_names['y'].format(i)] = y_i

    def __str__(self):
        return 'Deprecated Cycle Component'

    def __init__(self, **kwargs):
        super(DeprecatedCycleComp, self).__init__()

        self.metadata.declare('jacobian_type', default='matvec',
                              values=['matvec', 'dense', 'sparse-coo', 'sparse-csr',
                                      'sparse-csc'],
                              desc='method of assembling derivatives')
        self.metadata.declare('partial_type', default='array',
                              values=['array', 'sparse', 'aij'],
                              desc='type of partial derivatives')
        self.metadata.declare('num_var', types=int, default=1,
                              desc='Number of variables per component')
        self.metadata.declare('var_shape', types=tuple, default=(3,),
                              desc='Shape of each variable')
        self.metadata.declare('index', types=int,
                              desc='Index of the component. Used for testing implicit connections')
        self.metadata.declare('connection_type', default='explicit',
                              values=['explicit', 'implicit'],
                              desc='How to connect variables.')
        self.metadata.declare('num_comp', types=int, default=2,
                              desc='Total number of components')

        if 'finite_difference' in kwargs:
            if kwargs.get('finite_difference'):
                raise unittest.SkipTest('Finite Difference not supported.')
            del kwargs['finite_difference']

        self.metadata.update(kwargs)

        if self.metadata['jacobian_type'] == 'matvec':
            self.apply_linear = self.jac_vec

        if self.metadata['jacobian_type'] not in ['matvec', 'dense']:
            raise unittest.SkipTest('only matvec and dense jacobians are supported')
        if self.metadata['partial_type'] != 'array':
            raise unittest.SkipTest('only array partials are supported')

        self._cycle_names = {}

        if self.metadata['connection_type'] == 'implicit':
            idx = self.metadata['index']
            self._cycle_names['x'] = 'x_{}_{{}}'.format(idx)
            self._cycle_names['y'] = 'x_{}_{{}}'.format(idx + 1)
            self._cycle_names['theta'] = 'theta_{}'.format(idx)
            self._cycle_names['theta_out'] = 'theta_{}'.format(idx + 1)
            num_var = self.metadata['num_var']
            self._cycle_promotes_in = [self._cycle_names['x'].format(i) for i in range(num_var)]
            self._cycle_promotes_out = [self._cycle_names['y'].format(i) for i in range(num_var)]
            self._cycle_promotes_in.append(self._cycle_names['theta'])
            self._cycle_promotes_out.append(self._cycle_names['theta_out'])
        else:
            self._cycle_names['x'] = 'x_{}'
            self._cycle_names['y'] = 'y_{}'
            self._cycle_names['theta'] = 'theta'
            self._cycle_names['theta_out'] = 'theta_out'
            self._cycle_promotes_in = self._cycle_promotes_out = []



        self.angle_param = 'theta'

        self.num_var = self.metadata['num_var']
        self.var_shape = self.metadata['var_shape']
        self.size = self.num_var * np.prod(self.var_shape)

        for i in range(self.num_var):
            self.add_param(self._cycle_names['x'].format(i), shape=self.var_shape)
            self.add_output(self._cycle_names['y'].format(i), shape=self.var_shape)

        self.add_param(self._cycle_names['theta'], val=1.)
        self.add_output(self._cycle_names['theta_out'], shape=(1,))

    def _init_parameterized(self):
        pass

    def setup(self):
        self.declare_partials(of='*', wrt='*')

    def solve_nonlinear(self, inputs, outputs, residuals):
        theta = inputs[self._cycle_names['theta']]
        A = _compute_A(self.size, theta)
        x = self._inputs_to_vector(inputs)
        y = A.dot(x)
        self._vector_to_outputs(y, outputs)
        outputs[self._cycle_names['theta_out']] = theta

    def jac_vec(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        # Turned into apply_linear for matvec tests.
        if self.metadata['jacobian_type'] == 'matvec':
            angle_param = self._cycle_names[self.angle_param]
            x = self._inputs_to_vector(inputs)
            angle = inputs[angle_param]
            A = _compute_A(self.size, angle)
            dA = _compute_dA(self.size, angle)

            var_shape = self.metadata['var_shape']
            var_size = np.prod(var_shape)
            num_var = self.metadata['num_var']
            x_name = self._cycle_names['x']
            y_name = self._cycle_names['y']
            theta_name = self._cycle_names['theta']
            theta_out_name = self._cycle_names['theta_out']

            if mode == 'fwd':
                for j in range(num_var):
                    x_j = x_name.format(j)
                    if x_j in d_inputs:
                        dx = d_inputs[x_j].flat[:]
                        for i in range(num_var):
                            y_i = y_name.format(i)
                            if y_i in d_residuals:
                                Aij = A[array_idx(i, var_size), array_idx(j, var_size)]
                                d_residuals[y_i] += Aij.dot(dx).reshape(var_shape)

                if theta_name in d_inputs and theta_out_name in d_residuals:
                    dtheta = d_inputs[theta_name]
                    d_residuals[theta_out_name] += dtheta

                if angle_param in d_inputs:
                    dangle = d_inputs[angle_param]
                    dy_dangle = (dA.dot(x)) * dangle
                    for i in range(num_var):
                        y_i = y_name.format(i)
                        if y_i in d_residuals:
                            d_residuals[y_i] += dy_dangle[array_idx(i, var_size)].reshape(var_shape)

            elif mode == 'rev':
                for i in range(num_var):
                    y_i = y_name.format(i)
                    if y_i in d_residuals:
                        dy_i = d_residuals[y_i].flat[:]
                        for j in range(num_var):
                            x_j = x_name.format(j)
                            if x_j in d_inputs:
                                Aij = A[array_idx(i, var_size), array_idx(j, var_size)]
                                d_inputs[x_j] += Aij.T.dot(dy_i).reshape(var_shape)
                            if angle_param in d_inputs:
                                dAij = dA[array_idx(i, var_size), array_idx(j, var_size)]
                                x_j_vec = inputs[x_j].flat[:]
                                d_inputs[angle_param] += x_j_vec.T.dot(dAij.T.dot(dy_i))

                if theta_out_name in d_residuals and theta_name in d_inputs:
                    dtheta_out = d_residuals[theta_out_name]
                    d_inputs[theta_name] += dtheta_out

    def make_jacobian_entry(self, A, pd_type):
        if pd_type == 'array':
            return A

        raise ValueError('Unknown partial_type: {}'.format(pd_type))

    def _array2kwargs(self, arr, pd_type):
        jac = self.make_jacobian_entry(arr, pd_type)
        return {'val': jac}

    def linearize(self, inputs, outputs, resids):
        if self.metadata['jacobian_type'] != 'matvec':
            partials = {}

            angle_param = self._cycle_names[self.angle_param]
            angle = inputs[angle_param]
            num_var = self.num_var
            var_shape = self.var_shape
            var_size = np.prod(var_shape)
            x = self._inputs_to_vector(inputs)
            size = self.size
            A = _compute_A(size, angle)
            dA = _compute_dA(size, angle)
            dA_x = np.atleast_2d(dA.dot(x)).T
            pd_type = self.metadata['partial_type']
            dtheta = np.array([[1.]])

            y_name = self._cycle_names['y']
            x_name = self._cycle_names['x']

            for out_idx in range(num_var):
                out_var = y_name.format(out_idx)
                for in_idx in range(num_var):
                    in_var = x_name.format(in_idx)
                    Aij = A[array_idx(out_idx, var_size), array_idx(in_idx, var_size)]
                    J_y_x = self.make_jacobian_entry(Aij, pd_type)
                    J_y_angle = self.make_jacobian_entry(dA_x[array_idx(out_idx, var_size)],
                                                         pd_type)

                    partials[out_var, in_var] = J_y_x
                    partials[out_var, angle_param] = J_y_angle

            theta_out = self._cycle_names['theta_out']
            theta = self._cycle_names['theta']
            partials[theta_out, theta] = self.make_jacobian_entry(dtheta, pd_type)

            return partials


class DeprecatedFirstComp(DeprecatedCycleComp):
    def __str__(self):
        return 'Deprecated Cycle Component - First'

    def __init__(self, **kwargs):
        super(DeprecatedFirstComp, self).__init__(**kwargs)
        self.add_param('psi', val=1.)
        self.angle_param = 'psi'
        self._cycle_names['psi'] = 'psi'

    def solve_nonlinear(self, inputs, outputs, residuals):
        theta = inputs[self._cycle_names['theta']]
        psi = inputs[self._cycle_names['psi']]
        A = _compute_A(self.size, psi)
        y = A.dot(np.ones(self.size))
        self._vector_to_outputs(y, outputs)
        outputs[self._cycle_names['theta_out']] = theta


class DeprecatedLastComp(DeprecatedFirstComp):
    def __str__(self):
        return 'Deprecated Cycle Component - Last'

    def __init__(self, **kwargs):
        super(DeprecatedLastComp, self).__init__(**kwargs)
        self.add_output('x_norm2', shape=(1,))
        self._n = 1

    def solve_nonlinear(self, inputs, outputs, residuals):
        theta = inputs[self._cycle_names['theta']]
        psi = inputs[self._cycle_names['psi']]
        k = self.metadata['num_comp']
        x = self._inputs_to_vector(inputs)

        outputs['x_norm2'] = 0.5*np.dot(x,x)

        # theta_out has 1/2 the error as theta does to the correct angle.
        outputs[self._cycle_names['theta_out']] = theta / 2 + (self._n * 2 * np.pi - psi) / (2 * k - 2)

    def linearize(self, inputs, outputs, resids):
        if self.metadata['jacobian_type'] != 'matvec':
            partials = {}

            pd_type = self.metadata['partial_type']
            for i in range(self.metadata['num_var']):
                in_var = self._cycle_names['x'].format(i)
                partials['x_norm2', in_var] = self.make_jacobian_entry(inputs[in_var].flat[:],
                                                                       pd_type)

            k = self.metadata['num_comp']
            theta_out = self._cycle_names['theta_out']
            theta = self._cycle_names['theta']
            partials[theta_out, theta] = self.make_jacobian_entry(np.array([.5]), pd_type)
            partials[theta_out, self._cycle_names['psi']] = \
                self.make_jacobian_entry(np.array([-1/(2*k-2)]), pd_type)

            return partials

    def jac_vec(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        # Turned into apply_linear for matvec tests.
        if self.metadata['jacobian_type'] == 'matvec':
            k = self.metadata['num_comp']
            num_var = self.metadata['num_var']
            theta_out = self._cycle_names['theta_out']
            theta = self._cycle_names['theta']
            psi = self._cycle_names['psi']

            if mode == 'fwd':
                if theta_out in d_residuals:
                    if theta in d_inputs:
                        d_residuals[theta_out] += 0.5 * d_inputs[theta]
                    if psi in d_inputs:
                        d_residuals[theta_out] += -d_inputs[psi] / (2 * k - 2)
                for i in range(num_var):
                    in_var = self._cycle_names['x'].format(i)
                    if in_var in d_inputs and 'x_norm2' in d_residuals:
                        d_residuals['x_norm2'] += np.dot(inputs[in_var].flat, d_inputs[in_var].flat)

            elif mode == 'rev':
                if 'x_norm2' in d_residuals:
                    dxnorm = d_residuals['x_norm2']
                    for i in range(num_var):
                        x_i_name = self._cycle_names['x'].format(i)
                        if x_i_name in d_inputs:
                            d_inputs[x_i_name] += inputs[x_i_name] * dxnorm

                if theta_out in d_residuals:
                    dtheta_out = d_residuals[theta_out]
                    if theta in d_inputs:
                        d_inputs[theta] += .5*dtheta_out
                    if psi in d_inputs:
                        d_inputs[psi] += -dtheta_out/(2*k-2)
