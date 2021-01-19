import numpy as np
import unittest
import scipy.sparse.linalg

from openmdao.api import Problem, ImplicitComponent, Group
from openmdao.api import LinearBlockGS
from openmdao.utils.assert_utils import assert_near_equal


class CompA(ImplicitComponent):

    def setup(self):
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

    def setup(self):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_subsystem('CA', CompA(), promotes=['*'])
        self.add_subsystem('CB', CompB(), promotes=['*'])


class Test(unittest.TestCase):

    def setUp(self):
        group = GroupG()
        self.p = Problem(group)

        self.p.model.linear_solver = LinearBlockGS()
        self.p.setup()

        # Conclude setup but don't run model.
        self.p.final_setup()

        #n2(self.p, show_browser=False)

    def test_apply_linear(self):
        root = self.p.model
        self.p.set_solver_print(level=0)

        d_inputs, d_outputs, d_residuals = root.get_linear_vectors()

        d_outputs.set_val(1.0)
        root.run_apply_linear(['linear'], 'fwd')
        output = d_residuals._data
        assert_near_equal(output, [7, 3])

        d_residuals.set_val(1.0)
        root.run_apply_linear(['linear'], 'rev')
        output = d_outputs._data
        assert_near_equal(output, [7, 3])

    def test_solve_linear(self):
        root = self.p.model
        self.p.set_solver_print(level=0)

        d_inputs, d_outputs, d_residuals = root.get_linear_vectors()

        d_residuals.set_val(11.0)
        d_outputs.set_val(0.0)
        root.run_solve_linear(['linear'], 'fwd')
        output = d_outputs._data
        assert_near_equal(output, [1, 5], 1e-10)

        d_outputs.set_val(11.0)
        d_residuals.set_val(0.0)
        root.run_solve_linear(['linear'], 'rev')
        output = d_residuals._data
        assert_near_equal(output, [1, 5], 1e-10)


if __name__ == '__main__':
    unittest.main()
