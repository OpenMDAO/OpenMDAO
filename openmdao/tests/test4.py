from __future__ import division
import numpy
import unittest
import scipy.sparse.linalg

from openmdao.api import Problem, ImplicitComponent, Group
from openmdao.api import ScipyIterativeSolver, LinearBlockJac, LinearBlockGS
from openmdao.api import view_model

class CompA(ImplicitComponent):

    def initialize_variables(self):
        self.add_input('b')
        self.add_output('a')

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['a'] = 6 * outputs['a'] + 1 * inputs['b']

    def apply_linear(self, inputs, outputs,
                     d_inputs, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            if 'a' in d_residuals:
                if 'a' in d_outputs:
                    d_residuals['a'] += 6 * d_outputs['a']
                if 'b' in d_inputs:
                    d_residuals['a'] += 1 * d_inputs['b']
        if mode == 'rev':
            if 'a' in d_residuals:
                if 'a' in d_outputs:
                    d_outputs['a'] += 6 * d_residuals['a']
                if 'b' in d_inputs:
                    d_inputs['b'] += 1 * d_residuals['a']

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['a'] = 1./6. * d_residuals['a']
        elif mode == 'rev':
            d_residuals['a'] = 1./6. * d_outputs['a']


class CompB(ImplicitComponent):

    def initialize_variables(self):
        self.add_input('a')
        self.add_output('b')

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['b'] = 2 * outputs['b'] + 1 * inputs['a']

    def apply_linear(self, inputs, outputs,
                     d_inputs, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            if 'b' in d_residuals:
                if 'b' in d_outputs:
                    d_residuals['b'] += 2 * d_outputs['b']
                if 'a' in d_inputs:
                    d_residuals['b'] += 1 * d_inputs['a']
        if mode == 'rev':
            if 'b' in d_residuals:
                if 'b' in d_outputs:
                    d_outputs['b'] += 2 * d_residuals['b']
                if 'a' in d_inputs:
                    d_inputs['a'] += 1 * d_residuals['b']

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['b'] = 1./2. * d_residuals['b']
        elif mode == 'rev':
            d_residuals['b'] = 1./2. * d_outputs['b']


class GroupG(Group):

    def initialize(self):
        self.add_subsystem('CA', CompA(), promotes=['*'])
        self.add_subsystem('CB', CompB(), promotes=['*'])


class Test(unittest.TestCase):

    def setUp(self):
        group = GroupG()
        self.p = Problem(group)

        gmres = scipy.sparse.linalg.gmres
        self.p.setup(check=False)
        self.p.root.ln_solver = LinearBlockGS()

        #view_model(self.p, show_browser=False)

    def assertEqualArrays(self, a, b, tol=1e-3):
        self.assertTrue(numpy.linalg.norm(a-b) < tol)

    def test_apply_linear(self):
        root = self.p.root

        root.suppress_solver_output = True
        #root._solve_nonlinear()

        root._vectors['output'][''].set_const(1.0)
        root._apply_linear([''], 'fwd')
        output = root._vectors['residual']['']._data[0]
        self.assertEqualArrays(output, [7, 3])

        root._vectors['residual'][''].set_const(1.0)
        root._apply_linear([''], 'rev')
        output = root._vectors['output']['']._data[0]
        self.assertEqualArrays(output, [7, 3])

    def test_solve_linear(self):
        root = self.p.root

        root.suppress_solver_output = True
        #root._solve_nonlinear()

        root._vectors['residual'][''].set_const(11.0)
        root._vectors['output'][''].set_const(0.0)
        root._solve_linear([''], 'fwd')
        output = root._vectors['output']['']._data[0]
        self.assertEqualArrays(output, [1, 5])

        root._vectors['output'][''].set_const(11.0)
        root._vectors['residual'][''].set_const(0.0)
        root._solve_linear([''], 'rev')
        output = root._vectors['residual']['']._data[0]
        self.assertEqualArrays(output, [1, 5])


if __name__ == '__main__':
    unittest.main()
