"""Define the implicit test component (array)."""
from __future__ import division, print_function

import numpy as np
import scipy.sparse

from openmdao.api import ImplicitComponent


class TestImplCompArray(ImplicitComponent):

    def __init__(self, **kwargs):
        super(TestImplCompArray, self).__init__(**kwargs)

        self.metadata['mtx'] = np.array([
            [0.99, 0.01],
            [0.01, 0.99],
        ])

    def initialize_variables(self):
        self.add_input('rhs', val=np.ones(2))
        self.add_output('x', val=np.zeros(2))

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['x'] = self.metadata['mtx'].dot(outputs['x']) - inputs['rhs']

    def solve_nonlinear(self, inputs, outputs):
        outputs['x'] = np.linalg.solve(self.metadata['mtx'], inputs['rhs'])


class TestImplCompArrayDense(TestImplCompArray):

    def linearize(self, inputs, outputs, jacobian):
        jacobian['x', 'x'] = self.metadata['mtx']
        jacobian['x', 'rhs'] = -np.eye(2)


class TestImplCompArraySpmtx(TestImplCompArray):

    def linearize(self, inputs, outputs, jacobian):
        ones = np.ones(2)
        inds = np.arange(2)

        jacobian['x', 'x'] = scipy.sparse.csr_matrix(self.metadata['mtx'])
        jacobian['x', 'rhs'] = scipy.sparse.csr_matrix((-ones, (inds, inds)))


class TestImplCompArraySparse(TestImplCompArray):

    def linearize(self, inputs, outputs, jacobian):
        ones = np.ones(2)
        inds = np.arange(2)

        jacobian['x', 'x'] = (self.metadata['mtx'].flatten(),
                              np.arange(4), np.arange(4))
        jacobian['x', 'rhs'] = (-ones, inds, inds)


class TestImplCompArrayMatVec(TestImplCompArray):

    def linearize(self, inputs, outputs, jacobian):
        pass

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals,
                     mode):
        if mode == 'fwd':

            d_residuals['x'] += self.metadata['mtx'].dot(d_outputs['x'])
            d_residuals['x'] += -d_inputs['rhs']

        else:

            d_outputs['x'] += self.metadata['mtx'].dot(d_residuals['x'])
            d_inputs['rhs'] += -d_residuals['x']