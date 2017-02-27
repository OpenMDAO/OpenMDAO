"""Components for use in `CycleGroup`. For details, see `CycleGroup`."""
from __future__ import division, print_function

import numpy as np
import scipy.sparse as sparse

from openmdao.components.deprecated_component import Component as DeprecatedComponent


PSI = 1.

_vec_terms = {}


def _solve_nonlinear_vector_terms(system_size):
    # Try/Except pattern is much faster than if key in ... if the key is present (which it will be
    # outside of the first invocation).
    try:
        return _vec_terms[system_size]
    except KeyError:
        u = np.zeros(system_size)
        u[[0, -1]] = np.sqrt(2)/2

        v = np.zeros(system_size)
        v[1:-1] = 1 / np.sqrt(system_size - 2)

        cross_terms = np.outer(v, u) - np.outer(u, v)
        same_terms = np.outer(u, u) + np.outer(v, v)

        _vec_terms[system_size] = u, v, cross_terms, same_terms

        return u, v, cross_terms, same_terms


def _solve_nonlinear_A(system_size, theta):
    u, v, cross_terms, same_terms = _solve_nonlinear_vector_terms(system_size)
    return (np.eye(system_size)
            + np.sin(theta) * cross_terms
            + (np.cos(theta) - 1) * same_terms)


def _solve_nonlinear_dA(system_size, theta):
    u, v, cross_terms, same_terms = _solve_nonlinear_vector_terms(system_size)
    return np.cos(theta) * cross_terms - np.sin(theta) * same_terms


def array_idx(i, var_size):
    return slice(i * var_size, (i + 1) * var_size)


class DeprecatedCycleComp(DeprecatedComponent):

    def _params_to_vector(self, params):
        var_shape = self.metadata['var_shape']
        num_var = self.metadata['num_var']
        size = np.prod(var_shape)
        x = np.zeros(num_var * size)
        for i in range(num_var):
            x_i = params[self._cycle_names['x'].format(i)].flat
            x[size * i:size * (i + 1)] = x_i

        return x

    def _vector_to_unknowns(self, vec, unknowns):
        var_shape = self.metadata['var_shape']
        num_var = self.metadata['num_var']
        size = np.prod(var_shape)
        for i in range(num_var):
            y_i = vec[size * i:size * (i + 1)].reshape(var_shape)
            unknowns[self._cycle_names['y'].format(i)] = y_i

    def __str__(self):
        return 'Explicit Cycle Component'

    def __init__(self, **kwargs):
        super(DeprecatedCycleComp, self).__init__(**kwargs)
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

        # def initialize(self):
        self.metadata.declare('jacobian_type', value='matvec',
                              values=['matvec', 'dense', 'sparse-coo', 'sparse-csr'],
                              desc='method of assembling derivatives')
        self.metadata.declare('partial_type', value='array',
                              values=['array', 'sparse', 'aij'],
                              desc='type of partial derivatives')
        self.metadata.declare('num_var', type_=int, value=1,
                              desc='Number of variables per component')
        self.metadata.declare('var_shape', type_=tuple, value=(3,),
                              desc='Shape of each variable')
        self.metadata.declare('index', type_=int,
                              desc='Index of the component. Used for testing implicit connections')
        self.metadata.declare('connection_type', type_=str, value='explicit',
                              values=['explicit', 'implicit'],
                              desc='How to connect variables.')

        self.angle_param = 'theta'

        # def initialize_variables(self):
        self.num_var = self.metadata['num_var']
        self.var_shape = self.metadata['var_shape']
        self.size = self.num_var * np.prod(self.var_shape)

        for i in range(self.num_var):
            self.add_input(self._cycle_names['x'].format(i), shape=self.var_shape)
            self.add_output(self._cycle_names['y'].format(i), shape=self.var_shape)

        self.add_input(self._cycle_names['theta'], val=1.)
        self.add_output(self._cycle_names['theta_out'], shape=(1,))

    def solve_nonlinear(self, params, unknowns, residuals):
        theta = params[self._cycle_names['theta']]
        A = _solve_nonlinear_A(self.size, theta)
        x = self._params_to_vector(params)
        y = A.dot(x)
        self._vector_to_unknowns(y, unknowns)
        unknowns[self._cycle_names['theta_out']] = theta

    def apply_linear(self, params, unknowns, d_params, d_unknowns, d_residuals, mode):
        if self.metadata['jacobian_type'] == 'matvec':
            angle_param = self._cycle_names[self.angle_param]
            x = self._params_to_vector(params)
            angle = params[angle_param]
            A = _solve_nonlinear_A(self.size, angle)
            dA = _solve_nonlinear_dA(self.size, angle)

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
                    if x_j in d_params:
                        dx = d_params[x_j].flat[:]
                        for i in range(num_var):
                            y_i = y_name.format(i)
                            if y_i in d_unknowns:
                                Aij = A[array_idx(i, var_size), array_idx(j, var_size)]
                                d_unknowns[y_i] += Aij.dot(dx).reshape(var_shape)

                if theta_name in d_params and theta_out_name in d_unknowns:
                    dtheta = d_params[theta_name]
                    d_unknowns[theta_out_name] += dtheta

                if angle_param in d_params:
                    dangle = d_params[angle_param]
                    dy_dangle = (dA.dot(x)) * dangle
                    for i in range(num_var):
                        y_i = y_name.format(i)
                        if y_i in d_unknowns:
                            d_unknowns[y_i] += dy_dangle[array_idx(i, var_size)].reshape(var_shape)

            elif mode == 'rev':
                for i in range(num_var):
                    y_i = y_name.format(i)
                    if y_i in d_unknowns:
                        dy_i = d_unknowns[y_i].flat[:]
                        for j in range(num_var):
                            x_j = x_name.format(j)
                            if x_j in d_params:
                                Aij = A[array_idx(i, var_size), array_idx(j, var_size)]
                                d_params[x_j] += Aij.T.dot(dy_i).reshape(var_shape)
                            if angle_param in d_params:
                                dAij = dA[array_idx(i, var_size), array_idx(j, var_size)]
                                x_j_vec = params[x_j].flat[:]
                                d_params[angle_param] += x_j_vec.T.dot(dAij.T.dot(dy_i))

                if theta_out_name in d_unknowns and theta_name in d_params:
                    dtheta_out = d_unknowns[theta_out_name]
                    d_params[theta_name] += dtheta_out

    def make_jacobian_entry(self, A, pd_type):
        if pd_type == 'array':
            return A
        if pd_type == 'sparse':
            return sparse.csr_matrix(A)
        if pd_type == 'aij':
            data = []
            rows = []
            cols = []
            A = np.atleast_2d(A)
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if np.abs(A[i, j]) > 1e-15:
                        data.append(A[i, j])
                        rows.append(i)
                        cols.append(j)
            return [np.array(data), np.array(rows), np.array(cols)]

        raise ValueError('Unknown partial_type: {}'.format(pd_type))

    def _array2kwargs(self, arr, pd_type):
        jac = self.make_jacobian_entry(arr, pd_type)
        if pd_type == 'aij':
            return {'val': jac[0], 'rows': jac[1], 'cols': jac[2]}
        else:
            return {'val': jac}

    def initialize_partials(self):
        pd_type = self.metadata['partial_type']
        if self.metadata['jacobian_type'] != 'matvec' and pd_type != 'array':
            num_var = self.num_var
            var_shape = self.var_shape
            var_size = np.prod(var_shape)
            A = np.ones((self.size, self.size))
            dA_x = np.ones((self.size, 1))
            dtheta = np.array([[1.]])
            angle_param = self._cycle_names[self.angle_param]

            # if our subjacs are not dense, we must assign values here that
            # match their type (data values don't matter, only structure).
            # Otherwise, we assume they are dense and we'll get an error later
            # when we assign a subjac with a type that doesn't match.
            for out_idx in range(num_var):
                out_var = self._cycle_names['y'].format(out_idx)
                for in_idx in range(num_var):
                    in_var = self._cycle_names['x'].format(in_idx)
                    Aij = A[array_idx(out_idx, var_size), array_idx(in_idx, var_size)]

                    self.declare_partials(out_var, in_var,
                                          **self._array2kwargs(Aij, pd_type))
                    self.declare_partials(out_var, angle_param,
                                          **self._array2kwargs(dA_x[array_idx(out_idx, var_size)],
                                                               pd_type))

            self.declare_partials(self._cycle_names['theta_out'], self._cycle_names['theta'],
                                  **self._array2kwargs(dtheta, pd_type))

    def solve_nonlinear_partial_derivs(self, params, unknowns, partials):
        if self.metadata['jacobian_type'] != 'matvec':
            angle_param = self._cycle_names[self.angle_param]
            angle = params[angle_param]
            num_var = self.num_var
            var_shape = self.var_shape
            var_size = np.prod(var_shape)
            x = self._params_to_vector(params)
            size = self.size
            A = _solve_nonlinear_A(size, angle)
            dA = _solve_nonlinear_dA(size, angle)
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


class DeprecatedFirstComp(DeprecatedCycleComp):
    def __str__(self):
        return 'Explicit Cycle Component - First'

    def __init__(self):
        self.add_input('psi', val=1.)
        self.angle_param = 'psi'
        self._cycle_names['psi'] = 'psi'
        super(DeprecatedFirstComp, self).initialize_variables()

    def solve_nonlinear(self, params, unknowns, residuals):
        theta = params[self._cycle_names['theta']]
        psi = params[self._cycle_names['psi']]
        A = _solve_nonlinear_A(self.size, psi)
        y = A.dot(np.ones(self.size))
        self._vector_to_unknowns(y, unknowns)
        unknowns[self._cycle_names['theta_out']] = theta


class DeprecatedLastComp(DeprecatedFirstComp):
    def __str__(self):
        return 'Explicit Cycle Component - Last'

    def __init__(self):
        self.add_output('x_norm2', shape=(1,))
        self._n = 1
        super(DeprecatedLastComp, self).initialize_variables()

    def solve_nonlinear(self, params, unknowns, residuals):
        theta = params[self._cycle_names['theta']]
        psi = params[self._cycle_names['psi']]
        k = self.metadata['num_comp']
        x = self._params_to_vector(params)

        unknowns['x_norm2'] = 0.5*np.dot(x,x)

        # theta_out has 1/2 the error as theta does to the correct angle.
        unknowns[self._cycle_names['theta_out']] = theta / 2 + (self._n * 2 * np.pi - psi) / (2 * k - 2)

    def initialize_partials(self):
        super(DeprecatedLastComp, self).initialize_partials()

        pd_type = self.metadata['partial_type']
        if self.metadata['jacobian_type'] != 'matvec' and pd_type != 'array':
            x = np.ones(self.var_shape)
            for i in range(self.metadata['num_var']):
                in_var = self._cycle_names['x'].format(i)
                self.declare_partials('x_norm2', in_var,
                                      **self._array2kwargs(x.flatten(), pd_type))

            self.declare_partials(self._cycle_names['theta_out'], self._cycle_names['psi'],
                                  **self._array2kwargs(np.array([1.]), pd_type))

    def solve_nonlinear_partial_derivs(self, params, unknowns, partials):
        if self.metadata['jacobian_type'] != 'matvec':
            pd_type = self.metadata['partial_type']
            for i in range(self.metadata['num_var']):
                in_var = self._cycle_names['x'].format(i)
                partials['x_norm2', in_var] = self.make_jacobian_entry(params[in_var].flat[:],
                                                                       pd_type)

            k = self.metadata['num_comp']
            theta_out = self._cycle_names['theta_out']
            theta = self._cycle_names['theta']
            partials[theta_out, theta] = self.make_jacobian_entry(np.array([.5]), pd_type)
            partials[theta_out, self._cycle_names['psi']] = \
                self.make_jacobian_entry(np.array([-1/(2*k-2)]), pd_type)

    def apply_linear(self, params, unknowns, d_params, d_unknowns, d_residuals, mode):
        if self.metadata['jacobian_type'] == 'matvec':
            k = self.metadata['num_comp']
            num_var = self.metadata['num_var']
            theta_out = self._cycle_names['theta_out']
            theta = self._cycle_names['theta']
            psi = self._cycle_names['psi']

            if mode == 'fwd':
                if theta_out in d_unknowns:
                    if theta in d_params:
                        d_unknowns[theta_out] += 0.5 * d_params[theta]
                    if psi in d_params:
                        d_unknowns[theta_out] += -d_params[psi] / (2 * k - 2)
                for i in range(num_var):
                    in_var = self._cycle_names['x'].format(i)
                    if in_var in d_params and 'x_norm2' in d_unknowns:
                        d_unknowns['x_norm2'] += np.dot(params[in_var].flat, d_params[in_var].flat)

            elif mode == 'rev':
                if 'x_norm2' in d_unknowns:
                    dxnorm = d_unknowns['x_norm2']
                    for i in range(num_var):
                        x_i_name = self._cycle_names['x'].format(i)
                        if x_i_name in d_params:
                            d_params[x_i_name] += params[x_i_name] * dxnorm

                if theta_out in d_unknowns:
                    dtheta_out = d_unknowns[theta_out]
                    if theta in d_params:
                        d_params[theta] += .5*dtheta_out
                    if psi in d_params:
                        d_params[psi] += -dtheta_out/(2*k-2)
