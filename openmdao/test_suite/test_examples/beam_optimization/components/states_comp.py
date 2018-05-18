from __future__ import division
from six.moves import range

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from openmdao.api import ImplicitComponent


class StatesComp(ImplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('force_vector', types=np.ndarray)

    def setup(self):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes + 2

        self.add_input('K_local', shape=(num_elements, 4, 4))
        self.add_output('d', shape=size)

        self.K = np.zeros((size, size))

        cols = np.arange(16*num_elements)
        rows = np.repeat(np.arange(4), 4)
        rows = np.tile(rows, num_elements) + np.repeat(np.arange(num_elements), 16) * 2

        self.declare_partials('d', 'K_local', rows=rows, cols=cols)
        self.declare_partials('d', 'd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        force_vector = np.concatenate([self.options['force_vector'], np.zeros(2)])

        self.K = self.assemble_dense_K(inputs)
        residuals['d'] = np.dot(self.K, outputs['d']) - force_vector

    def solve_nonlinear(self, inputs, outputs):
        force_vector = np.concatenate([self.options['force_vector'], np.zeros(2)])

        self.K = self.assemble_dense_K(inputs)
        self.lu = lu_factor(self.K)

        outputs['d'] = lu_solve(self.lu, force_vector)

    def linearize(self, inputs, outputs, partials):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes + 2

        self.K = self.assemble_dense_K(inputs)
        self.lu = lu_factor(self.K)

        i_elem = np.tile(np.arange(4), 4)
        i_d = np.tile(i_elem, num_elements) + np.repeat(np.arange(num_elements), 16) * 2

        partials['d', 'K_local'] = outputs['d'][i_d]

        partials['d', 'd'] = self.K

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['d'] = lu_solve(self.lu, d_residuals['d'], trans=0)
        else:
            d_residuals['d'] = lu_solve(self.lu, d_outputs['d'], trans=1)

    def assemble_dense_K(self, inputs):
        """
        Assemble the stiffness matrix.

        Returns
        -------
        ndarray
            Stiffness matrix as dense ndarray.
        """
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        K = self.K
        K[:] = 0.0

        for ind in range(num_elements):
            ind1_ = 2 * ind
            ind2_ = 2 * ind + 4

            K[ind1_:ind2_, ind1_:ind2_] += inputs['K_local'][ind, :, :]

        K[2 * num_nodes + 0, 0] = 1.0
        K[2 * num_nodes + 1, 1] = 1.0
        K[0, 2 * num_nodes + 0] = 1.0
        K[1, 2 * num_nodes + 1] = 1.0

        return K


class MultiStatesComp(ImplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('force_vector', types=np.ndarray)
        self.options.declare('num_rhs', types=int)

    def setup(self):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes + 2
        num_rhs = self.options['num_rhs']

        self.add_input('K_local', shape=(num_elements, 4, 4))
        for j in range(num_rhs):
            self.add_output('d_%d' % j, shape=size)

        self.K = np.zeros((size, size))

        cols = np.arange(16*num_elements)
        rows = np.repeat(np.arange(4), 4)
        rows = np.tile(rows, num_elements) + np.repeat(np.arange(num_elements), 16) * 2

        for j in range(num_rhs):
            disp = 'd_%d' % j

            self.declare_partials(disp, 'K_local', rows=rows, cols=cols)
            self.declare_partials(disp, disp)

    def apply_nonlinear(self, inputs, outputs, residuals):
        num_rhs = self.options['num_rhs']

        self.K = self.assemble_dense_K(inputs)
        for j in range(num_rhs):
            force_vector = np.concatenate([self.options['force_vector'][:, j], np.zeros(2)])
            residuals['d_%d' % j] = np.dot(self.K, outputs['d_%d' % j]) - force_vector

    def solve_nonlinear(self, inputs, outputs):
        num_rhs = self.options['num_rhs']

        self.K = self.assemble_dense_K(inputs)
        self.lu = lu_factor(self.K)

        for j in range(num_rhs):

            force_vector = np.concatenate([self.options['force_vector'][:, j], np.zeros(2)])
            outputs['d_%d' % j] = lu_solve(self.lu, force_vector)

    def linearize(self, inputs, outputs, partials):
        num_rhs = self.options['num_rhs']
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes + 2

        self.K = self.assemble_dense_K(inputs)
        self.lu = lu_factor(self.K)

        i_elem = np.tile(np.arange(4), 4)
        i_d = np.tile(i_elem, num_elements) + np.repeat(np.arange(num_elements), 16) * 2

        for j in range(num_rhs):
            disp = 'd_%d' % j
            partials[disp, 'K_local'] = outputs[disp][i_d]
            partials[disp, disp] = self.K

    def solve_linear(self, d_outputs, d_residuals, mode):
        num_rhs = self.options['num_rhs']

        for j in range(num_rhs):
            disp = 'd_%d' % j

            if mode == 'fwd':
                d_outputs[disp] = lu_solve(self.lu, d_residuals[disp], trans=0)
            else:
                d_residuals[disp] = lu_solve(self.lu, d_outputs[disp], trans=1)

    def assemble_dense_K(self, inputs):
        """
        Assemble the stiffness matrix.

        Returns
        -------
        ndarray
            Stiffness matrix as dense ndarray.
        """
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        K = self.K
        K[:] = 0.0

        for ind in range(num_elements):
            ind1_ = 2 * ind
            ind2_ = 2 * ind + 4

            K[ind1_:ind2_, ind1_:ind2_] += inputs['K_local'][ind, :, :]

        K[2 * num_nodes + 0, 0] = 1.0
        K[2 * num_nodes + 1, 1] = 1.0
        K[0, 2 * num_nodes + 0] = 1.0
        K[1, 2 * num_nodes + 1] = 1.0

        return K