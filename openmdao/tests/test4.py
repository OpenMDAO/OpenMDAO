from __future__ import division
import numpy
import unittest
import scipy.sparse.linalg

from openmdao.api import Problem, ImplicitComponent, Group, PETScVector
from openmdao.solvers.ln_scipy import ScipyIterativeSolver


class CompA(ImplicitComponent):

    def initialize_variables(self):
        self.add_input('b')
        self.add_output('a')

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['a'] = 1 * outputs['a'] + 3 * inputs['b']

    def apply_linear(self, inputs, outputs,
                     d_inputs, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_residuals['a'] = 0
            if 'a' in d_outputs:
                d_residuals['a'] += 1 * d_outputs['a']
            if 'a' in d_outputs and 'b' in d_inputs:
                d_residuals['a'] += 3 * d_inputs['b']
        if mode == 'rev':
            d_outputs['a'] = 0
            d_inputs['b'] = 0
            if 'a' in d_outputs:
                d_outputs['a'] += 1 * d_residuals['a']
            if 'a' in d_outputs and 'b' in d_inputs:
                d_inputs['b'] += 3 * d_residuals['a']


class CompB(ImplicitComponent):

    def initialize_variables(self):
        self.add_input('a')
        self.add_output('b')

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['b'] = 1 * outputs['b'] + 2 * inputs['a']

    def apply_linear(self, inputs, outputs,
                     d_inputs, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_residuals['b'] = 0
            if 'b' in d_outputs:
                d_residuals['b'] += 1 * d_outputs['b']
            if 'b' in d_outputs and 'a' in d_inputs:
                d_residuals['b'] += 2 * d_inputs['a']
        if mode == 'rev':
            d_outputs['b'] = 0
            d_inputs['a'] = 0
            if 'b' in d_outputs:
                d_outputs['b'] += 1 * d_residuals['b']
            if 'b' in d_outputs and 'a' in d_inputs:
                d_inputs['a'] += 2 * d_residuals['b']


class GroupG(Group):

    def initialize(self):
        self.add_subsystem('CA', CompA(), promotes=['*'])
        self.add_subsystem('CB', CompB(), promotes=['*'])


class Test(unittest.TestCase):

    def setUp(self):
        group = GroupG()
        self.p = Problem(group)

        gmres = scipy.sparse.linalg.gmres
        self.p.root._solvers_linear = ScipyIterativeSolver(options={'solver':gmres})
        self.p.setup()

    def assertEqualArrays(self, a, b):
        self.assertTrue(numpy.linalg.norm(a-b) < 1e-15)

    def test_apply_linear(self):
        root = self.p.root

        root.set_solver_print(False)
        root._solve_nonlinear()

        root._vectors['output'][''].set_const(1.0)
        root._apply_linear([''], 'fwd')
        output = root._vectors['residual']['']._data[0]
        self.assertEqualArrays(output, [4, 3])

        root._vectors['residual'][''].set_const(1.0)
        root._apply_linear([''], 'rev')
        output = root._vectors['output']['']._data[0]
        self.assertEqualArrays(output, [3, 4])

    def test_solve_linear(self):
        root = self.p.root

        root.set_solver_print(False)
        root._solve_nonlinear()

        root._vectors['residual'][''].set_const(1.0)
        root._solve_linear([''], 'fwd')
        output = root._vectors['output']['']._data[0]
        self.assertEqualArrays(output, [0.4, 0.2])

        root._vectors['output'][''].set_const(1.0)
        root._solve_linear([''], 'rev')
        output = root._vectors['residual']['']._data[0]
        self.assertEqualArrays(output, [0.2, 0.4])


if __name__ == '__main__':
    unittest.main()
