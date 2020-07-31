"""Test for a feature doc showing how to use cache_linear_solution"""
from distutils.version import LooseVersion
import unittest
from copy import deepcopy
from io import StringIO


import numpy as np
import scipy
from scipy.sparse.linalg import gmres

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class CacheLinearTestCase(unittest.TestCase):

    def test_feature_cache_linear(self):

        from distutils.version import LooseVersion
        import numpy as np
        import scipy
        from scipy.sparse.linalg import gmres

        import openmdao.api as om


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
                self.add_output('states', val=[0,0])

                self.declare_partials(of='*', wrt='*')

            def apply_nonlinear(self, inputs, outputs, residuals):
                a = inputs['a']
                b = inputs['b']
                c = inputs['c']
                x = outputs['states'][0]
                y = outputs['states'][1]

                residuals['states'][0] = a * x ** 2 + b * x + c
                residuals['states'][1] = a * y + b

            def solve_nonlinear(self, inputs, outputs):
                a = inputs['a']
                b = inputs['b']
                c = inputs['c']
                outputs['states'][0] = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
                outputs['states'][1] = -b/a

            def linearize(self, inputs, outputs, partials):
                a = inputs['a'][0]
                b = inputs['b'][0]
                c = inputs['c'][0]
                x = outputs['states'][0]
                y = outputs['states'][1]

                partials['states', 'a'] = [[x**2],[y]]
                partials['states', 'b'] = [[x],[1]]
                partials['states', 'c'] = [[1.0],[0]]
                partials['states', 'states'] = [[2*a*x+b, 0],[0, a]]

                self.state_jac = np.array([[2*a*x+b, 0],[0, a]])

            def solve_linear(self, d_outputs, d_residuals, mode):

                if mode == 'fwd':
                    print("incoming initial guess", d_outputs['states'])
                    if LooseVersion(scipy.__version__) < LooseVersion("1.1"):
                        d_outputs['states'] = gmres(self.state_jac, d_residuals['states'], x0=d_outputs['states'])[0]
                    else:
                        d_outputs['states'] = gmres(self.state_jac, d_residuals['states'], x0=d_outputs['states'], atol='legacy')[0]
                elif mode == 'rev':
                    if LooseVersion(scipy.__version__) < LooseVersion("1.1"):
                        d_residuals['states'] = gmres(self.state_jac, d_outputs['states'], x0=d_residuals['states'])[0]
                    else:
                        d_residuals['states'] = gmres(self.state_jac, d_outputs['states'], x0=d_residuals['states'], atol='legacy')[0]

        p = om.Problem()
        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'

        p.model.set_input_defaults('b', val=4.)
        p.model.add_subsystem('quad', QuadraticComp(), promotes_inputs=['a', 'b', 'c'], promotes_outputs=['states'])
        p.model.add_subsystem('obj', om.ExecComp('y = (x[1]-x[0])**2', x=np.ones(2)))
        p.model.connect('states', 'obj.x')

        p.model.add_design_var('a', cache_linear_solution=True)
        p.model.add_constraint('states', upper=10)
        p.model.add_objective('obj.y')

        p.setup(mode='fwd')
        p.run_driver()

        print(p['a'], p['b'], p['c'])
        print(p['states'])
        assert_near_equal(p['obj.y'], 0.25029766, 1e-3)


if __name__ == "__main__":
    unittest.main()

