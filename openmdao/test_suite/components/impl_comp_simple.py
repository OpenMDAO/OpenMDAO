"""Define the implicit test component (simple)."""
import numpy as np
import scipy.sparse
import scipy.optimize

import openmdao.api as om


class TestImplCompSimple(om.ImplicitComponent):

    def setup(self):
        self.add_input('a', val=1.)
        self.add_output('x', val=0.)

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['x'] = np.exp(outputs['x']) - \
            inputs['a']**2 * outputs['x']**2

    def solve_nonlinear(self, inputs, outputs):
        def func(x):
            return np.exp(x) - inputs['a']**2 * x**2

        outputs['x'] = scipy.optimize.brentq(func, -1./inputs['a'], 0)


class TestImplCompSimpleDense(TestImplCompSimple):

    def setup(self):
        super(TestImplCompSimpleDense, self).setup()
        self.declare_partials(of='*', wrt='*')

    def linearize(self, inputs, outputs, jacobian):
        jacobian['x', 'x'] = np.exp(outputs['x']) - \
            2 * inputs['a']**2 * outputs['x']
        jacobian['x', 'a'] = -2 * inputs['a'] * outputs['x']**2


class TestImplCompSimpleSpmtx(TestImplCompSimple):

    def setup(self):
        super(TestImplCompSimpleSpmtx, self).setup()
        self.declare_partials(of='*', wrt='*')

    def linearize(self, inputs, outputs, jacobian):
        jacobian['x', 'x'] = scipy.sparse.csr_matrix((
            np.exp(outputs['x']) - 2 * inputs['a']**2 * outputs['x'],
            (0, 0)
        ))
        jacobian['x', 'a'] = scipy.sparse.csr_matrix((
            -2 * inputs['a'] * outputs['x']**2,
            (0, 0)
        ))


class TestImplCompSimpleSparse(TestImplCompSimple):

    def setup(self):
        super(TestImplCompSimpleSparse, self).setup()
        self.declare_partials(of='*', wrt='*')

    def linearize(self, inputs, outputs, jacobian):
        jacobian['x', 'x'] = (
            np.exp(outputs['x']) - 2 * inputs['a']**2 * outputs['x'], 0, 0)
        jacobian['x', 'a'] = (
            -2 * inputs['a'] * outputs['x']**2, 0, 0)
