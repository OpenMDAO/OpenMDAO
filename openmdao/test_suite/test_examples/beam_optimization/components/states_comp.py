from __future__ import division
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from openmdao.api import ImplicitComponent


class StatesComp(ImplicitComponent):

    def initialize(self):
        self.metadata.declare('num_elements', types=int)
        self.metadata.declare('force_vector', types=np.ndarray)

    def setup(self):
        num_elements = self.metadata['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes + 2

        self.add_input('K', shape=(size, size))
        self.add_output('d', shape=size)

        rows = np.outer(np.arange(size), np.ones(size, int)).flatten()
        cols = np.arange(size ** 2)
        self.declare_partials('d', 'K', rows=rows, cols=cols)

        self.declare_partials('d', 'd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        force_vector = np.concatenate([self.metadata['force_vector'], np.zeros(2)])

        residuals['d'] = np.dot(inputs['K'], outputs['d']) - force_vector

    def solve_nonlinear(self, inputs, outputs):
        force_vector = np.concatenate([self.metadata['force_vector'], np.zeros(2)])

        self.lu = lu_factor(inputs['K'])

        outputs['d'] = lu_solve(self.lu, force_vector)

    def linearize(self, inputs, outputs, partials):
        num_elements = self.metadata['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes + 2

        self.lu = lu_factor(inputs['K'])

        partials['d', 'K'] = np.outer(np.ones(size), outputs['d']).flatten()
        partials['d', 'd'] = inputs['K']

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['d'] = lu_solve(self.lu, d_residuals['d'], trans=0)
        else:
            d_residuals['d'] = lu_solve(self.lu, d_outputs['d'], trans=1)
