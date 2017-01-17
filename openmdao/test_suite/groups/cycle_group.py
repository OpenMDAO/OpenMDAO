"""Contains test groups for cycles with easily verified values/derivatives."""
from __future__ import print_function, division
from openmdao.api import ExplicitComponent, Group, IndepVarComp
import numpy as np
from functools import lru_cache
import scipy.sparse as sparse
from six.moves import range


@lru_cache(maxsize=2)
def _compute_vector_terms(N):
    u = np.zeros(N)
    u[[0, -1]] = np.sqrt(2)/2

    v = np.zeros(N)
    v[1:-1] = 1 / np.sqrt(N - 2)

    cross_terms = np.outer(v, u) - np.outer(u, v)
    same_terms = np.outer(u, u) + np.outer(v, v)

    return u, v, cross_terms, same_terms


@lru_cache()
def _compute_A(N, theta):
    u, v, cross_terms, same_terms = _compute_vector_terms(N)
    return (np.eye(N)
            + np.sin(theta) * cross_terms
            + (np.cos(theta) - 1) * same_terms)


@lru_cache()
def _compute_dA(N, theta):
    u, v, cross_terms, same_terms = _compute_vector_terms(N)
    return (np.cos(theta) * cross_terms
            - np.sin(theta) * same_terms)


def _dense_to_aij(A):
    data = []
    rows = []
    cols = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.abs(A[i, j]) > 1e-15:
                data.append(A[i, j])
                rows.append(i)
                cols.append(j)
    return (data, rows, cols)


def _cycle_comp_jacobian(component, inputs, outputs, jacobian, angle_param):
    if component.metadata['jacobian_type'] != 'matvec':
        N = component.N
        angle = inputs[angle_param]
        x = inputs['x']
        A = _compute_A(N, angle)
        dA = _compute_dA(N, angle)
        dA_x = dA.dot(x)
        dA_x.shape = [3, 1]
        pd_type = component.metadata['partial_type']
        dtheta = np.array([1.])
        if pd_type == 'array':
            J_y_x = A
            J_y_angle = dA_x
            J_theta = dtheta
        elif pd_type == 'sparse':
            J_y_x = sparse.csr_matrix(A)
            J_y_angle = sparse.csr_matrix(dA_x)
            J_theta = sparse.csr_matrix(dtheta)
        elif pd_type == 'aij':
            J_y_x = _dense_to_aij(A)
            J_y_angle = _dense_to_aij(dA_x)
            J_theta = _dense_to_aij(dtheta)
        else:
            raise ValueError('Unknown partial_type: {}'.format(pd_type))

        jacobian['y', 'x'] = J_y_x
        jacobian['y', angle_param] = J_y_angle
        jacobian['theta_out', 'theta'] = J_theta


def _cycle_comp_jacvec(component, inputs, outputs, d_inputs, d_outputs, mode, angle_param):
    if component.metadata['jacobian_type'] == 'matvec':
        if 'y' in d_outputs and 'theta_out' in d_outputs and \
                        'x' in d_inputs and 'theta' in d_inputs and angle_param in d_inputs:
            x = inputs['x']
            angle = inputs[angle_param]
            A = _compute_A(component.N, angle)
            dA = _compute_dA(component.N, angle)
            if mode == 'fwd':
                dx = d_inputs['x']
                dtheta = d_inputs['theta']
                dangle = d_inputs[angle_param]

                d_outputs['y'] += A.dot(dx) + (dA.dot(x)) * dangle
                d_outputs['theta_out'] += dtheta

            elif mode == 'rev':
                dy = d_outputs['y']
                dtheta_out = d_outputs['theta_out']

                d_inputs['x'] += A.T.dot(dy)
                d_inputs['theta'] += dtheta_out
                d_inputs[angle_param] += x.T.dot(dA.T.dot(dy))


class ExplicitCycleComp(ExplicitComponent):
    def __str__(self):
        return 'Explicit Cycle Component'

    def initialize(self):
        self.metadata.declare('variable_length', type_=int, value=3,
                              desc='Size of the system used within the cycle.')
        self.metadata.declare('jacobian_type', value='matvec',
                              values=['matvec', 'dense', 'sparse-coo',
                                      'sparse-csr'],
                              desc='method of assembling derivatives')
        self.metadata.declare('partial_type', value='array',
                              values=['array', 'sparse', 'aij'],
                              desc='type of partial derivatives')

    def initialize_variables(self):
        self.N = N = self.metadata['variable_length']

        if N < 3:
            raise ValueError('Error: variable_length must be >= 3')

        self.add_input('x', shape=(N,))
        self.add_input('theta', value=1.)

        self.add_output('theta_out', shape=(1,))
        self.add_output('y', shape=(N,))

        self._u, self._v, self._cross_terms, self._same_terms = _compute_vector_terms(N)

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        A = _compute_A(self.N, theta)
        outputs['y'] = A.dot(inputs['x'])
        outputs['theta_out'] = theta

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        _cycle_comp_jacvec(self, inputs, outputs, d_inputs, d_outputs, mode, 'theta')

    def compute_jacobian(self, inputs, outputs, jacobian):
        _cycle_comp_jacobian(self, inputs, outputs, jacobian, 'theta')


class ExplicitFirstComp(ExplicitComponent):
    def __str__(self):
        return 'Explicit Cycle Component - First'

    def initialize(self):
        self.metadata.declare('variable_length', type_=int, value=3,
                              desc='Size of the system used within the cycle.')
        self.metadata.declare('jacobian_type', value='matvec',
                              values=['matvec', 'dense', 'sparse-coo',
                                      'sparse-csr'],
                              desc='method of assembling derivatives')
        self.metadata.declare('partial_type', value='array',
                              values=['array', 'sparse', 'aij'],
                              desc='type of partial derivatives')

    def initialize_variables(self):
        self.N = N = self.metadata['variable_length']

        if N < 3:
            raise ValueError('Error: variable_length must be >= 3')

        self.add_input('x', shape=(N,))
        self.add_input('theta', value=1.)

        self.add_output('theta_out', shape=(1,))
        self.add_output('y', shape=(N,))

        self._u, self._v, self._cross_terms, self._same_terms = _compute_vector_terms(N)

        self.add_input('psi', value=1.)

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        psi = inputs['psi']
        A = _compute_A(self.N, psi)
        outputs['y'] = A.dot(inputs['x'])
        outputs['theta_out'] = theta

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        _cycle_comp_jacvec(self, inputs, outputs, d_inputs, d_outputs, mode, 'psi')

    def compute_jacobian(self, inputs, outputs, jacobian):
        _cycle_comp_jacobian(self, inputs, outputs, jacobian, 'psi')


class ExplicitLastComp(ExplicitComponent):
    def __str__(self):
        return 'Explicit Cycle Component - Last'

    def initialize(self):
        self.metadata.declare('variable_length', type_=int, value=3,
                              desc='Size of the system used within the cycle.')
        self.metadata.declare('jacobian_type', value='matvec',
                              values=['matvec', 'dense', 'sparse-coo',
                                      'sparse-csr'],
                              desc='method of assembling derivatives')
        self.metadata.declare('partial_type', value='array',
                              values=['array', 'sparse', 'aij'],
                              desc='type of partial derivatives')

    def initialize_variables(self):
        self.N = N = self.metadata['variable_length']

        if N < 3:
            raise ValueError('Error: variable_length must be >= 3')

        self.add_input('x', shape=(N,))
        self.add_input('theta', value=1.)
        self.add_input('psi', value=1.)

        self.add_output('theta_out', shape=(1,))
        self.add_output('x_sum', shape=(1,))

        self._u, self._v, self._cross_terms, self._same_terms = _compute_vector_terms(N)

        self._n = 0

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        psi = inputs['psi']
        k = self.metadata['num_comp']

        outputs['x_sum'] = np.sum(inputs['x'])

        # theta_out has 1/2 the error as theta does to the correct angle.
        outputs['theta_out'] = theta/2 + (self._n * 2 * np.pi - psi) / (2*k - 2)

    def compute_jacobian(self, inputs, outputs, jacobian):
        if self.metadata['jacobian_type'] != 'matvec':
            jacobian['x_sum', 'x'] = np.ones(self.N)

            k = self.metadata['num_comp']
            jacobian['theta_out', 'theta'] = np.array([.5])
            jacobian['theta_out', 'psi'] = np.array([-1/(2*k-2)])

    def compute_jacvec_product(self, inputs, outputs,
                               d_inputs, d_outputs, mode):

        if self.metadata['jacobian_type'] == 'matvec':
            if 'x_sum' in d_outputs and 'theta_out' in d_outputs and \
                            'x' in d_inputs and 'theta' in d_inputs and 'psi' in d_inputs:
                k = self.metadata['num_comp']
                if mode == 'fwd':
                    dx = d_inputs['x']
                    dtheta = d_inputs['theta']
                    dpsi = d_inputs['psi']

                    d_outputs['x_sum'] += np.sum(dx)
                    d_outputs['theta_out'] += np.array([.5*dtheta - dpsi/(2*k-2)])
                elif mode == 'rev':
                    dxsum = d_outputs['x_sum']
                    dtheta_out = d_outputs['theta_out']

                    d_inputs['x'] += np.ones(self.N) * dxsum
                    d_inputs['theta'] += .5*dtheta_out
                    d_inputs['psi'] += -dtheta_out/(2*k-2)


class CycleGroup(Group):
    """Group with a cycle. Derivatives and values are known."""

    def _initialize_metadata(self):
        self.metadata.declare('num_comp', type_=int, value=2,
                              desc='Total number of components')
        self.metadata.declare('variable_length', type_=int, value=10,
                              desc='Size of the underlying systems.')
        self.metadata.declare('connection_type', type_=str, value='explicit',
                              values=['explicit', 'implicit'],
                              desc='How to connect variables.')
        self.metadata.declare('component_class', type_=str, value='explicit',
                              values=['explicit'],
                              desc='Component class to instantiate')
        self.metadata.declare('jacobian_type', value='matvec',
                              values=['matvec', 'dense', 'sparse-coo',
                                      'sparse-csr'],
                              desc='method of assembling derivatives')
        self.metadata.declare('partial_type', value='array',
                              values=['array', 'sparse', 'aij'],
                              desc='type of partial derivatives')

    def initialize(self):
        self._initialize_metadata()

        num_comp = self.metadata['num_comp']
        if num_comp < 2:
            raise ValueError('Number of components must be at least 2.')

        self.N = N = self.metadata['variable_length']
        if N < 3:
            raise ValueError('Variable length must be at least 3.')

        connection_type = self.metadata['connection_type']

        comp_class = self.metadata['component_class']

        if comp_class == 'explicit':
            first_class = ExplicitFirstComp
            middle_class = ExplicitCycleComp
            last_class = ExplicitLastComp
        else:
            raise ValueError('Should not happen.')

        self._generate_components(connection_type, first_class, middle_class, last_class, num_comp)

    def _generate_components(self, connection_type, first_class, middle_class, last_class,
                             num_comp):
        first_name = 'first'
        last_name = 'last'
        comp_args = {
            'variable_length': self.N,
            'jacobian_type': self.metadata['jacobian_type'],
            'partial_type': self.metadata['partial_type']
        }

        self.add_subsystem('psi_comp', IndepVarComp('psi', 1.))
        self.add_subsystem('x0_comp', IndepVarComp('x', np.ones(self.N)))

        self._add_middle_cycle(connection_type, first_class, first_name, 0, comp_args)
        prev_name = first_name
        idx = 0

        connection_variables = (
            ('y', 'x'),
            ('theta_out', 'theta')
        )

        for idx in range(1, num_comp - 1):
            current_name = 'middle_{0}'.format(idx)
            self._add_middle_cycle(connection_type, middle_class, current_name, idx, comp_args)

            if connection_type == 'explicit':
                self._explicit_connections(prev_name, current_name, connection_variables)

            prev_name = current_name
        if connection_type == 'explicit':
            self.add_subsystem(last_name, last_class(**comp_args))

            self._explicit_connections(prev_name, last_name, connection_variables)
            self._explicit_connections(last_name, first_name,[('theta_out', 'theta')])

        elif connection_type == 'implicit':
            renames_inputs = {
                'x': 'x_{}'.format(idx + 1),
                'theta': 'theta_{0}'.format(idx + 1)
            }
            renames_outputs = {
                'theta_out': 'theta_0'
            }
            self.add_subsystem(last_name, last_class(**comp_args),
                               renames_inputs=renames_inputs,
                               renames_outputs=renames_outputs)

        self.connect('psi_comp.psi', first_name + '.psi')
        self.connect('psi_comp.psi', last_name + '.psi')
        self.connect('x0_comp.x', first_name + '.x')

    def _explicit_connections(self, prev_name, current_name, vars):
        for out_var, in_var in vars:
            self.connect(
                '{0}.{1}'.format(prev_name, out_var),
                '{0}.{1}'.format(current_name, in_var)
            )

    def _add_middle_cycle(self, connection_type, comp_class, comp_name, index, comp_args):
        if connection_type == 'explicit':
            self.add_subsystem(comp_name, comp_class(**comp_args))
        elif connection_type == 'implicit':
            renames_inputs = {
                'theta': 'theta_{0}'.format(index),
                'x': 'x_{0}'.format(index),
            }
            renames_outputs = {
                'theta_out': 'theta_{0}'.format(index+1),
                'y': 'x_{0}'.format(index+1),
            }

            self.add_subsystem(comp_name, comp_class(**comp_args),
                               renames_inputs=renames_inputs,
                               renames_outputs=renames_outputs)


