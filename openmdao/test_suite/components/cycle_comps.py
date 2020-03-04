"""Components for use in `CycleGroup`. For details, see `CycleGroup`."""
import numpy as np
import scipy.sparse as sparse

import unittest

from openmdao.core.explicitcomponent import ExplicitComponent


PSI = 1.

_vec_terms = {}


def _compute_vector_terms(system_size):
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


def _compute_A(system_size, theta):
    u, v, cross_terms, same_terms = _compute_vector_terms(system_size)
    return (np.eye(system_size)
            + np.sin(theta) * cross_terms
            + (np.cos(theta) - 1) * same_terms)


def _compute_dA(system_size, theta):
    u, v, cross_terms, same_terms = _compute_vector_terms(system_size)
    return np.cos(theta) * cross_terms - np.sin(theta) * same_terms


def array_idx(i, var_size):
    return slice(i * var_size, (i + 1) * var_size)


class ExplicitCycleComp(ExplicitComponent):

    def _inputs_to_vector(self, inputs):
        var_shape = self.options['var_shape']
        num_var = self.options['num_var']
        size = np.prod(var_shape)
        x = np.zeros(num_var * size)
        for i in range(num_var):
            x_i = inputs[self._cycle_names['x'].format(i)].flat
            x[size * i:size * (i + 1)] = x_i

        return x

    def _vector_to_outputs(self, vec, outputs):
        var_shape = self.options['var_shape']
        num_var = self.options['num_var']
        size = np.prod(var_shape)
        for i in range(num_var):
            y_i = vec[size * i:size * (i + 1)].reshape(var_shape)
            outputs[self._cycle_names['y'].format(i)] = y_i

    def __str__(self):
        return 'Explicit Cycle Component'

    def initialize(self):
        self.options.declare('jacobian_type', default='matvec',
                              values=['matvec', 'dense', 'sparse-csc'],
                              desc='method of assembling derivatives')
        self.options.declare('partial_type', default='array',
                              values=['array', 'sparse', 'aij'],
                              desc='type of partial derivatives')
        self.options.declare('num_var', types=int, default=1,
                              desc='Number of variables per component')
        self.options.declare('var_shape', types=tuple, default=(3,),
                              desc='Shape of each variable')
        self.options.declare('index', types=int,
                              desc='Index of the component. Used for testing implicit connections')
        self.options.declare('connection_type', default='explicit',
                              values=['explicit', 'implicit'],
                              desc='How to connect variables.')
        self.options.declare('partial_method', default='exact',
                              values=('exact', 'fd', 'cs'),
                              desc='How derivatives should be solved (exact, fd, or cs)')
        self.options.declare('num_comp', types=int, default=2,
                              desc='Total number of components')

        self.angle_param = 'theta'

        self._cycle_names = {}

    def _init_parameterized(self):
        self.num_var = self.options['num_var']
        self.var_shape = self.options['var_shape']
        self.size = self.num_var * np.prod(self.var_shape)

        if self.options['jacobian_type'] == 'matvec':
            self.compute_jacvec_product = self.jacvec_product

        if self.options['connection_type'] == 'implicit':
            idx = self.options['index']
            self._cycle_names['x'] = 'x_{}_{{}}'.format(idx)
            self._cycle_names['y'] = 'x_{}_{{}}'.format(idx + 1)
            self._cycle_names['theta'] = 'theta_{}'.format(idx)
            self._cycle_names['theta_out'] = 'theta_{}'.format(idx + 1)
            num_var = self.options['num_var']
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

    def setup(self):
        for i in range(self.num_var):
            self.add_input(self._cycle_names['x'].format(i), shape=self.var_shape)
            self.add_output(self._cycle_names['y'].format(i), shape=self.var_shape)

        self.add_input(self._cycle_names['theta'], val=1.)
        self.add_output(self._cycle_names['theta_out'], shape=(1,))

        # Setup partials

        pd_type = self.options['partial_type']

        if self.options['partial_method'] != 'exact':
            if self.options['jacobian_type'] == 'matvec':
                raise unittest.SkipTest('not testing FD and matvec')
            if pd_type != 'array':
                raise unittest.SkipTest('only dense FD supported')
            self.declare_partials('*', '*', method=self.options['partial_method'])

        elif self.options['jacobian_type'] != 'matvec' and pd_type != 'array':
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
                                          **self._array2kwargs(Aij, pd_type, 'exact'))
                    self.declare_partials(out_var, angle_param,
                                          **self._array2kwargs(dA_x[array_idx(out_idx, var_size)],
                                                               pd_type, 'exact'))

            self.declare_partials(self._cycle_names['theta_out'], self._cycle_names['theta'],
                                  **self._array2kwargs(dtheta, pd_type, 'exact'))

        else:
            # Declare everything
            self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        theta = inputs[self._cycle_names['theta']]
        A = _compute_A(self.size, theta)
        x = self._inputs_to_vector(inputs)
        y = A.dot(x)
        self._vector_to_outputs(y, outputs)
        outputs[self._cycle_names['theta_out']] = theta

    def jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        angle_param = self._cycle_names[self.angle_param]
        x = self._inputs_to_vector(inputs)
        angle = inputs[angle_param]
        A = _compute_A(self.size, angle)
        dA = _compute_dA(self.size, angle)

        var_shape = self.options['var_shape']
        var_size = np.prod(var_shape)
        num_var = self.options['num_var']
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
                        if y_i in d_outputs:
                            Aij = A[array_idx(i, var_size), array_idx(j, var_size)]
                            d_outputs[y_i] += Aij.dot(dx).reshape(var_shape)

            if theta_name in d_inputs and theta_out_name in d_outputs:
                dtheta = d_inputs[theta_name]
                d_outputs[theta_out_name] += dtheta

            if angle_param in d_inputs:
                dangle = d_inputs[angle_param]
                dy_dangle = (dA.dot(x)) * dangle
                for i in range(num_var):
                    y_i = y_name.format(i)
                    if y_i in d_outputs:
                        d_outputs[y_i] += dy_dangle[array_idx(i, var_size)].reshape(var_shape)

        elif mode == 'rev':
            for i in range(num_var):
                y_i = y_name.format(i)
                if y_i in d_outputs:
                    dy_i = d_outputs[y_i].flat[:]
                    for j in range(num_var):
                        x_j = x_name.format(j)
                        if x_j in d_inputs:
                            Aij = A[array_idx(i, var_size), array_idx(j, var_size)]
                            d_inputs[x_j] += Aij.T.dot(dy_i).reshape(var_shape)
                        if angle_param in d_inputs:
                            dAij = dA[array_idx(i, var_size), array_idx(j, var_size)]
                            x_j_vec = inputs[x_j].flat[:]
                            d_inputs[angle_param] += x_j_vec.T.dot(dAij.T.dot(dy_i))

            if theta_out_name in d_outputs and theta_name in d_inputs:
                dtheta_out = d_outputs[theta_out_name]
                d_inputs[theta_name] += dtheta_out

    def make_jacobian_entry(self, A, pd_type):
        if pd_type == 'aij':
            return self.make_sub_jacobian(A, pd_type)[0]
        return self.make_sub_jacobian(A, pd_type)

    def make_sub_jacobian(self, A, pd_type):
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

    def _array2kwargs(self, arr, pd_type, method):
        jac = self.make_sub_jacobian(arr, pd_type)
        if pd_type == 'aij':
            return {'val': jac[0], 'rows': jac[1], 'cols': jac[2], 'method': method}
        else:
            return {'val': jac, 'method': method}

    def compute_partials(self, inputs, partials):
        if self.options['jacobian_type'] != 'matvec' and self.options['partial_method'] == 'exact':
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
            pd_type = self.options['partial_type']
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


class ExplicitFirstComp(ExplicitCycleComp):
    def __str__(self):
        return 'Explicit Cycle Component - First'

    def setup(self):
        self.add_input('psi', val=1.)
        self.angle_param = 'psi'
        self._cycle_names['psi'] = 'psi'
        super(ExplicitFirstComp, self).setup()

    def compute(self, inputs, outputs):
        theta = inputs[self._cycle_names['theta']]
        psi = inputs[self._cycle_names['psi']]
        A = _compute_A(self.size, psi)
        y = A.dot(np.ones(self.size))
        self._vector_to_outputs(y, outputs)
        outputs[self._cycle_names['theta_out']] = theta


class ExplicitLastComp(ExplicitFirstComp):
    def __str__(self):
        return 'Explicit Cycle Component - Last'

    def setup(self):
        super(ExplicitLastComp, self).setup()

        self.add_output('x_norm2', shape=(1,))
        self._n = 1

        # Setup partials

        pd_type = self.options['partial_type']
        method = self.options['partial_method']
        if self.options['jacobian_type'] != 'matvec' and pd_type != 'array':
            x = np.ones(self.var_shape)
            for i in range(self.options['num_var']):
                in_var = self._cycle_names['x'].format(i)
                self.declare_partials('x_norm2', in_var,
                                      **self._array2kwargs(x.flatten(), pd_type, method))

            self.declare_partials(self._cycle_names['theta_out'], self._cycle_names['psi'],
                                  **self._array2kwargs(np.array([1.]), pd_type, method))

    def compute(self, inputs, outputs):
        theta = inputs[self._cycle_names['theta']]
        psi = inputs[self._cycle_names['psi']]
        k = self.options['num_comp']
        x = self._inputs_to_vector(inputs)

        outputs['x_norm2'] = 0.5*np.dot(x,x)

        # theta_out has 1/2 the error as theta does to the correct angle.
        outputs[self._cycle_names['theta_out']] = theta / 2 + (self._n * 2 * np.pi - psi) / (2 * k - 2)

    def compute_partials(self, inputs, partials):
        if self.options['jacobian_type'] != 'matvec' and self.options['partial_method'] == 'exact':
            pd_type = self.options['partial_type']
            for i in range(self.options['num_var']):
                in_var = self._cycle_names['x'].format(i)
                partials['x_norm2', in_var] = self.make_jacobian_entry(inputs[in_var].flat[:],
                                                                       pd_type)

            k = self.options['num_comp']
            theta_out = self._cycle_names['theta_out']
            theta = self._cycle_names['theta']
            partials[theta_out, theta] = self.make_jacobian_entry(np.array([.5]), pd_type)
            partials[theta_out, self._cycle_names['psi']] = \
                self.make_jacobian_entry(np.array([-1/(2*k-2)]), pd_type)

    def jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if self.options['jacobian_type'] == 'matvec':
            k = self.options['num_comp']
            num_var = self.options['num_var']
            theta_out = self._cycle_names['theta_out']
            theta = self._cycle_names['theta']
            psi = self._cycle_names['psi']

            if mode == 'fwd':
                if theta_out in d_outputs:
                    if theta in d_inputs:
                        d_outputs[theta_out] += 0.5 * d_inputs[theta]
                    if psi in d_inputs:
                        d_outputs[theta_out] += -d_inputs[psi] / (2 * k - 2)
                for i in range(num_var):
                    in_var = self._cycle_names['x'].format(i)
                    if in_var in d_inputs and 'x_norm2' in d_outputs:
                        d_outputs['x_norm2'] += np.dot(inputs[in_var].flat, d_inputs[in_var].flat)

            elif mode == 'rev':
                if 'x_norm2' in d_outputs:
                    dxnorm = d_outputs['x_norm2']
                    for i in range(num_var):
                        x_i_name = self._cycle_names['x'].format(i)
                        if x_i_name in d_inputs:
                            d_inputs[x_i_name] += inputs[x_i_name] * dxnorm

                if theta_out in d_outputs:
                    dtheta_out = d_outputs[theta_out]
                    if theta in d_inputs:
                        d_inputs[theta] += .5*dtheta_out
                    if psi in d_inputs:
                        d_inputs[psi] += -dtheta_out/(2*k-2)
