"""Define the implicit test component (simple)."""
from __future__ import division, print_function

import numpy
import scipy.sparse
import scipy.optimize

from openmdao.api import ImplicitComponent


class TestImplCompSimple(ImplicitComponent):

    def initialize_variables(self):
        self.add_input('a', val=1.)
        self.add_output('x', val=0.)

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['x'] = numpy.exp(outputs['x']) - \
            inputs['a']**2 * outputs['x']**2

    def solve_nonlinear(self, inputs, outputs):
        def func(x):
            return numpy.exp(x) - inputs['a']**2 * x**2

        outputs['x'] = scipy.optimize.brentq(func, -1./inputs['a'], 0)


class TestImplCompSimpleDense(TestImplCompSimple):

    def linearize(self, inputs, outputs, jacobian):
        jacobian['x', 'x'] = numpy.exp(outputs['x']) - \
            2 * inputs['a']**2 * outputs['x']
        jacobian['x', 'a'] = -2 * inputs['a'] * outputs['x']**2


class TestImplCompSimpleSpmtx(TestImplCompSimple):

    def linearize(self, inputs, outputs, jacobian):
        jacobian['x', 'x'] = scipy.sparse.csr_matrix((
            numpy.exp(outputs['x']) - 2 * inputs['a']**2 * outputs['x'],
            (0, 0)))
        jacobian['x', 'a'] = scipy.sparse.csr_matrix((
            -2 * inputs['a'] * outputs['x']**2,
            (0, 0)))


class TestImplCompSimpleSparse(TestImplCompSimple):

    def linearize(self, inputs, outputs, jacobian):
        jacobian['x', 'x'] = (
            numpy.exp(outputs['x']) - 2 * inputs['a']**2 * outputs['x'], 0, 0)
        jacobian['x', 'a'] = (
            -2 * inputs['a'] * outputs['x']**2, 0, 0)
