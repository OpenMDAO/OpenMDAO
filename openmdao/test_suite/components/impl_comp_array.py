"""Define the implicit test component (array)."""

import numpy as np
import scipy.sparse

import openmdao.api as om


class TestImplCompArray(om.ImplicitComponent):

    def initialize(self):
        self.mtx = np.array([
            [0.99, 0.01],
            [0.01, 0.99],
        ])

    def setup(self):
        self.add_input('rhs', val=np.ones(2))
        self.add_output('x', val=np.zeros(2))

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['x'] = self.mtx.dot(outputs['x']) - inputs['rhs']

    def solve_nonlinear(self, inputs, outputs):
        outputs['x'] = np.linalg.solve(self.mtx, inputs['rhs'])


class TestImplCompArrayDense(TestImplCompArray):

    def linearize(self, inputs, outputs, jacobian):
        jacobian['x', 'x'] = self.mtx
        jacobian['x', 'rhs'] = -np.eye(2)


class TestImplCompArraySpmtx(TestImplCompArray):

    def linearize(self, inputs, outputs, jacobian):
        ones = np.ones(2)
        inds = np.arange(2)

        jacobian['x', 'x'] = scipy.sparse.csr_matrix(self.mtx)
        jacobian['x', 'rhs'] = scipy.sparse.csr_matrix((-ones, (inds, inds)))


class TestImplCompArraySparse(TestImplCompArray):

    def linearize(self, inputs, outputs, jacobian):
        ones = np.ones(2)
        inds = np.arange(2)

        jacobian['x', 'x'] = (self.mtx.flatten(),
                              np.arange(4), np.arange(4))
        jacobian['x', 'rhs'] = (-ones, inds, inds)


class TestImplCompArrayMatVec(TestImplCompArray):

    def linearize(self, inputs, outputs, jacobian):
        pass

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals,
                     mode):

        if mode == 'fwd':
            d_residuals['x'] += self.mtx.dot(d_outputs['x'])
            d_residuals['x'] += -d_inputs['rhs']

        else:
            d_outputs['x'] += self.mtx.dot(d_residuals['x'])
            d_inputs['rhs'] += -d_residuals['x']
