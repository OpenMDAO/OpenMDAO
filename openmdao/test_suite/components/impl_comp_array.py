"""Define the implicit test component (array)."""
from __future__ import division, print_function

import numpy
import scipy.sparse

from openmdao.api import ImplicitComponent


class TestImplCompArray(ImplicitComponent):

    def __init__(self, **kwargs):
        super(TestImplCompArray, self).__init__(**kwargs)

        self.metadata['mtx'] = numpy.array([
            [0.99, 0.01],
            [0.01, 0.99],
        ])

    def initialize_variables(self):
        self.add_input('rhs', val=numpy.ones(2))
        self.add_output('x', val=numpy.zeros(2))

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['x'] = self.metadata['mtx'].dot(outputs['x']) - inputs['rhs']

    def solve_nonlinear(self, inputs, outputs):
        outputs['x'] = numpy.linalg.solve(self.metadata['mtx'], inputs['rhs'])


class TestImplCompArrayDense(TestImplCompArray):

    def linearize(self, inputs, outputs, jacobian):
        jacobian['x', 'x'] = self.metadata['mtx']
        jacobian['x', 'rhs'] = -numpy.eye(2)


class TestImplCompArraySpmtx(TestImplCompArray):

    def linearize(self, inputs, outputs, jacobian):
        ones = numpy.ones(2)
        inds = numpy.arange(2)

        jacobian['x', 'x'] = scipy.sparse.csr_matrix(self.metadata['mtx'])
        jacobian['x', 'rhs'] = scipy.sparse.csr_matrix((-ones, (inds, inds)))


class TestImplCompArraySparse(TestImplCompArray):

    def linearize(self, inputs, outputs, jacobian):
        ones = numpy.ones(2)
        inds = numpy.arange(2)

        jacobian['x', 'x'] = (self.metadata['mtx'].flatten(),
                              numpy.arange(4), numpy.arange(4))
        jacobian['x', 'rhs'] = (-ones, inds, inds)
