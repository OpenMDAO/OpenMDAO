from __future__ import division, print_function

import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

#size = 5

#class MyImplicitComp(om.ImplicitComponent):


    #def setup(self):
        #self.nonlinear_solver = om.NewtonSolver()
        #self.linear_solver = om.DirectSolver()

        #self.add_input('a', val=np.ones(size))
        #self.add_input('b', val=np.ones(size))
        #self.add_input('c', val=np.ones(size))

        #self.add_output('x', val=np.ones(size))
        #self.add_output('y', val=np.ones(size))
        #self.add_output('z', val=np.ones(size))

        #arange = np.arange(size, dtype=int)

        #self.declare_partials('x', 'a', rows=arange, cols=arange)
        #self.declare_partials('x', 'b', rows=arange, cols=arange, val=1.0)
        #self.declare_partials('y', 'b', rows=arange, cols=arange, val=3.0)
        #self.declare_partials('y', 'c', rows=arange, cols=arange, val=1.0)
        #self.declare_partials('z', 'c', rows=arange, cols=arange, val=5.0)

        #for v in ('x', 'y', 'z'):
            #self.declare_partials(v, v, rows=arange, cols=arange, val=1.0)

    #def apply_nonlinear(self, inputs, outputs, residuals):
        #residuals['x'] = inputs['a'] ** 2 * 2. + inputs['b']
        #residuals['y'] = inputs['b'] ** 2 * 3. + inputs['c']
        #residuals['z'] = inputs['c'] ** 2 * 5.

    #def linearize(self, inputs, outputs, partials):
        #partials['x', 'a'] = inputs['a'] * 4.0
        #partials['y', 'b'] = inputs['b'] * 6.0
        #partials['z', 'c'] = inputs['a'] * 10.0


class QuadraticComp(om.ImplicitComponent):
    """
    A Simple Implicit Component representing a Quadratic Equation.

    R(a, b, c, x) = ax^2 + bx + c

    Solution via Quadratic Formula:
    x = (-b + sqrt(b^2 - 4ac)) / 2a
    """

    def setup(self):
        self.add_input('a', val=1.)
        self.add_input('b', val=1.)
        self.add_input('c', val=1.)
        self.add_output('x', val=0.)

        self.declare_partials(of='x', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        a = inputs['a']
        b = inputs['b']
        c = inputs['c']
        x = outputs['x']
        residuals['x'] = a * x ** 2 + b * x + c

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


class NondistribDirectCompTestCase(unittest.TestCase):
    N_PROCS = 2

    def test_direct(self):
        p = om.Problem()
        
        p.model.add_subsystem('comp', QuadraticComp())

        p.setup(force_alloc_complex=True)
        p.run_model()
        
        partials = p.check_partials(includes=['comp'], method='cs', out_stream=None)
        assert_check_partials(partials)




if __name__ == '__main__':
    unittest.main()
