"""Simple example demonstrating how to implement an implicit component."""
from __future__ import division

from six.moves import cStringIO

import unittest

from openmdao.api import Problem, Group, ImplicitComponent, IndepVarComp
from openmdao.devtools.testutil import assert_rel_error


class TestImplCompSimpleCompute(ImplicitComponent):

    def initialize_variables(self):
        self.add_input('a', val=1.)
        self.add_input('b', val=1.)
        self.add_input('c', val=1.)
        self.add_output('x', val=0.)

    def apply_nonlinear(self, inputs, outputs, residuals):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']
        residuals['x'] = a * x ** 2 + b * x + c

    def solve_nonlinear(self, inputs, outputs):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        outputs['x'] = (-b + (b ** 2 - 4 * a * c) ** 0.5) / 2 / a


class TestImplCompSimpleLinearize(TestImplCompSimpleCompute):

    def linearize(self, inputs, outputs, partials):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']

        partials['x', 'a'] = x ** 2
        partials['x', 'b'] = x
        partials['x', 'c'] = 1.0
        partials['x', 'x'] = 2 * a * x + b

        self.inv_jac = 1.0 / (2 * a * x + b)

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['x'] = self.inv_jac * d_residuals['x']
        elif mode == 'rev':
            d_residuals['x'] = self.inv_jac * d_outputs['x']


class TestImplCompSimpleJacVec(TestImplCompSimpleCompute):

    def linearize(self, inputs, outputs, partials):
        a = inputs['a']
        b = inputs['b']
        x = outputs['x']
        self.inv_jac = 1.0 / (2 * a * x + b)

    def apply_linear(self, inputs, outputs,
                     d_inputs, d_outputs, d_residuals, mode):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']
        if mode == 'fwd':
            if 'x' in d_residuals:
                if 'x' in d_outputs:
                    d_residuals['x'] += (2 * a * x + b) * d_outputs['x']
                if 'a' in d_inputs:
                    d_residuals['x'] += x ** 2 * d_inputs['a']
                if 'b' in d_inputs:
                    d_residuals['x'] += x * d_inputs['b']
                if 'c' in d_inputs:
                    d_residuals['x'] += d_inputs['c']
        elif mode == 'rev':
            if 'x' in d_residuals:
                if 'x' in d_outputs:
                    d_outputs['x'] += (2 * a * x + b) * d_residuals['x']
                if 'a' in d_inputs:
                    d_inputs['a'] += x ** 2 * d_residuals['x']
                if 'b' in d_inputs:
                    d_inputs['b'] += x * d_residuals['x']
                if 'c' in d_inputs:
                    d_inputs['c'] += d_residuals['x']

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['x'] = self.inv_jac * d_residuals['x']
        elif mode == 'rev':
            d_residuals['x'] = self.inv_jac * d_outputs['x']


class TestImplCompSimple(unittest.TestCase):

    def test_compute(self):
        group = Group()
        group.add_subsystem('comp1', IndepVarComp([('a', 1.0), ('b', 1.0), ('c', 1.0)]))
        group.add_subsystem('comp2', TestImplCompSimpleLinearize())
        group.add_subsystem('comp3', TestImplCompSimpleJacVec())
        group.connect('comp1.a', 'comp2.a')
        group.connect('comp1.b', 'comp2.b')
        group.connect('comp1.c', 'comp2.c')
        group.connect('comp1.a', 'comp3.a')
        group.connect('comp1.b', 'comp3.b')
        group.connect('comp1.c', 'comp3.c')

        prob = Problem(model=group)
        prob.setup(check=False)

        prob['comp1.a'] = 1.
        prob['comp1.b'] = -4.
        prob['comp1.c'] = 3.
        prob.run_model()
        assert_rel_error(self, prob['comp2.x'], 3.)
        assert_rel_error(self, prob['comp2.x'], 3.)

        total_derivs = prob.compute_total_derivs(
            wrt=['comp1.a', 'comp1.b', 'comp1.c'],
            of=['comp2.x', 'comp3.x']
        )
        assert_rel_error(self, total_derivs['comp2.x', 'comp1.a'], -4.5)
        assert_rel_error(self, total_derivs['comp2.x', 'comp1.b'], -1.5)
        assert_rel_error(self, total_derivs['comp2.x', 'comp1.c'], -0.5)
        assert_rel_error(self, total_derivs['comp3.x', 'comp1.a'], -4.5)
        assert_rel_error(self, total_derivs['comp3.x', 'comp1.b'], -1.5)
        assert_rel_error(self, total_derivs['comp3.x', 'comp1.c'], -0.5)

        # Piggyback testing of list_states

        stream = cStringIO()
        prob.model.list_states(stream=stream)
        content = stream.getvalue()

        self.assertTrue('comp2.x' in content)
        self.assertTrue('comp3.x' in content)

if __name__ == '__main__':
    unittest.main()
