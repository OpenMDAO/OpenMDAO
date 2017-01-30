"""Components for use in `CycleGroup`."""
from __future__ import division, print_function

import numpy as np
import scipy.sparse as sparse

from openmdao.api import ExplicitComponent


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

def _inputs_to_vector(inputs, num_var, var_shape):
    size = np.prod(var_shape)
    x = np.zeros(num_var * size)
    for i in range(num_var):
        x_i = inputs['x_{}'.format(i)].flat
        x[size*i:size*(i+1)] = x_i

    return x


def _vector_to_outputs(vec, outputs, num_var, var_shape):
    size = np.prod(var_shape)
    for i in range(num_var):
        y_i = vec[size * i:size * (i + 1)].reshape(var_shape)
        outputs['y_{}'.format(i)] = y_i

class ExplicitCycleComp(ExplicitComponent):
    def __str__(self):
        return 'Explicit Cycle Component'

    def initialize(self):
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

        self.angle_param = 'theta'

    def initialize_variables(self):
        self.num_var = self.metadata['num_var']
        self.var_shape = self.metadata['var_shape']
        self.size = self.num_var * np.prod(self.var_shape)

        for i in range(self.num_var):
            self.add_input('x_{}'.format(i), shape=self.var_shape)
            self.add_output('y_{}'.format(i), shape=self.var_shape)

        self.add_input('theta', value=1.)
        self.add_output('theta_out', shape=(1,))

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        A = _compute_A(self.size, theta)
        x = _inputs_to_vector(inputs, self.num_var, self.var_shape)
        y = A.dot(x)
        _vector_to_outputs(y, outputs, self.num_var, self.var_shape)
        outputs['theta_out'] = theta

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        if self.metadata['jacobian_type'] == 'matvec':
            angle_param = self.angle_param
            x = _inputs_to_vector(inputs, self.num_var, self.var_shape)
            angle = inputs[angle_param]
            A = _compute_A(self.size, angle)
            dA = _compute_dA(self.size, angle)
            if mode == 'fwd':
                if 'x' in d_inputs and 'y' in d_outputs:
                    dx = d_inputs['x']
                    d_outputs['y'] += A.dot(dx)

                if 'theta' in d_inputs and 'theta_out' in d_outputs:
                    dtheta = d_inputs['theta']
                    d_outputs['theta_out'] += dtheta

                if angle_param in d_inputs and 'y' in d_outputs:
                    dangle = d_inputs[angle_param]
                    d_outputs['y'] += (dA.dot(x)) * dangle

            elif mode == 'rev':
                if 'y' in d_outputs:
                    dy = d_outputs['y']
                    if 'x' in d_inputs:
                        d_inputs['x'] += A.T.dot(dy)
                    if angle_param in d_inputs:
                        d_inputs[angle_param] += x.T.dot(dA.T.dot(dy))

                if 'theta_out' in d_outputs and 'theta' in d_inputs:
                    dtheta_out = d_outputs['theta_out']
                    d_inputs['theta'] += dtheta_out

    def compute_jacobian(self, inputs, outputs, jacobian):
        if self.metadata['jacobian_type'] != 'matvec':
            angle_param = self.angle_param
            angle = inputs[angle_param]
            num_var = self.num_var
            var_shape = self.var_shape
            var_size = np.prod(var_shape)
            x = _inputs_to_vector(inputs, num_var, var_shape)
            size = self.size
            A = _compute_A(size, angle)
            dA = _compute_dA(size, angle)
            dA_x = np.atleast_2d(dA.dot(x)).T
            pd_type = self.metadata['partial_type']
            dtheta = np.array([[1.]])

            def array_idx(i):
                return slice(i * var_size, (i + 1) * var_size)

            def make_jacobian_entry(A, pd_type):
                if pd_type == 'array':
                    return A
                if pd_type == 'sparse':
                    return sparse.csr_matrix(A)
                if pd_type == 'aij':
                    data = []
                    rows = []
                    cols = []
                    for i in range(A.shape[0]):
                        for j in range(A.shape[1]):
                            if np.abs(A[i, j]) > 1e-15:
                                data.append(A[i, j])
                                rows.append(i)
                                cols.append(j)
                    return np.array(data), np.array(rows), np.array(cols)

            for out_idx in range(num_var):
                out_var = 'y_{0}'.format(out_idx)
                for in_idx in range(num_var):
                    in_var = 'x_{0}'.format(in_idx)

                    J_y_x = make_jacobian_entry(A[array_idx(out_idx), array_idx(in_idx)], pd_type)
                    J_y_angle = make_jacobian_entry(dA_x[array_idx(out_idx)], pd_type)

                    jacobian[out_var, in_var] = J_y_x
                    jacobian[out_var, angle_param] = J_y_angle

            jacobian['theta_out', 'theta'] = make_jacobian_entry(dtheta, pd_type)


class ExplicitFirstComp(ExplicitCycleComp):
    def __str__(self):
        return 'Explicit Cycle Component - First'

    def initialize_variables(self):
        super(ExplicitFirstComp, self).initialize_variables()
        self.add_input('psi', value=1.)
        self.angle_param = 'psi'

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        psi = inputs['psi']
        A = _compute_A(self.size, psi)
        y = A.dot(np.ones(self.size))
        _vector_to_outputs(y, outputs, self.num_var, self.var_shape)
        outputs['theta_out'] = theta


class ExplicitLastComp(ExplicitFirstComp):
    def __str__(self):
        return 'Explicit Cycle Component - Last'

    def initialize_variables(self):
        super(ExplicitLastComp, self).initialize_variables()
        self.add_output('x_norm2', shape=(1,))
        self._n = 1

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        psi = inputs['psi']
        k = self.metadata['num_comp']
        x = _inputs_to_vector(inputs, self.metadata['num_var'], self.metadata['var_shape'])

        outputs['x_norm2'] = 0.5*np.dot(x,x)

        # theta_out has 1/2 the error as theta does to the correct angle.
        outputs['theta_out'] = theta/2 + (self._n * 2 * np.pi - psi) / (2*k - 2)

    def compute_jacobian(self, inputs, outputs, jacobian):
        if self.metadata['jacobian_type'] != 'matvec':

            for i in range(self.metadata['num_var']):
                in_var = 'x_{0}'.format(i)
                jacobian['x_norm2', in_var] = inputs[in_var].flat[:]

            k = self.metadata['num_comp']
            jacobian['theta_out', 'theta'] = np.array([.5])
            jacobian['theta_out', 'psi'] = np.array([-1/(2*k-2)])

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        if self.metadata['jacobian_type'] == 'matvec':
            k = self.metadata['num_comp']
            num_var = self.metadata['num_var']
            var_shape = self.metadata['var_shape']
            if mode == 'fwd':
                if 'theta_out' in d_outputs:
                    if 'theta' in d_inputs :
                        d_outputs['theta_out'] += 0.5 * d_inputs['theta']
                    if 'psi' in d_inputs:
                        d_outputs['theta_out'] += -d_inputs['psi'] / (2 * k - 2)
                for i in range(num_var):
                    in_var = 'x_{0}'.format(i)
                    if in_var in d_inputs and 'x_norm2' in d_outputs:
                        d_outputs['x_norm2'] += np.dot(inputs[in_var].flat, d_inputs[in_var].flat)

            elif mode == 'rev':
                if 'x' in d_inputs:
                    if 'x_norm2' in d_outputs:
                        dxnorm = d_outputs['x_norm2']
                        for i in range(num_var):
                            in_var = 'x_{0}'.format(i)
                            d_inputs['x'] += inputs[in_var] * dxnorm

                if 'theta_out' in d_outputs:
                    dtheta_out = d_outputs['theta_out']
                    if 'theta' in d_inputs:
                        d_inputs['theta'] += .5*dtheta_out
                    if 'psi' in d_inputs:
                        d_inputs['psi'] += -dtheta_out/(2*k-2)