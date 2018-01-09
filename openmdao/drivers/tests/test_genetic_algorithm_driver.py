""" Unit tests for the SimpleGADriver Driver."""

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver
from openmdao.utils.assert_utils import assert_rel_error


class TestSimpleGA(unittest.TestCase):

    def test_simple_test_func(self):

        class MyComp(ExplicitComponent):

            def setup(self):
                self.add_input('x', np.zeros((2, )))

                self.add_output('a', 0.0)
                self.add_output('b', 0.0)
                self.add_output('c', 0.0)
                self.add_output('d', 0.0)

            def compute(self, inputs, outputs):
                x = inputs['x']

                outputs['a'] = (2*x[0] - 3*x[1])**2
                outputs['b'] = 18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2
                outputs['c'] = (x[0] + x[1] + 1)**2
                outputs['d'] = 19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2


        prob = Problem()
        prob.model = model = Group()

        model.add_subsystem('px', IndepVarComp('x', np.zeros((2, ))))
        model.add_subsystem('comp', MyComp())
        model.add_subsystem('obj', ExecComp('f=(30 + a*b)*(1 + c*d)'))

        model.connect('px.x', 'comp.x')
        model.connect('comp.a', 'obj.a')
        model.connect('comp.b', 'obj.b')
        model.connect('comp.c', 'obj.c')
        model.connect('comp.d', 'obj.d')

        model.add_design_var('px.x', lower=0.0, upper=1.0)
        model.add_objective('obj.f')

        prob.driver = SimpleGADriver()
        prob.setup(check=False)

        prob.run_driver()

        # Solution: xopt = [0.2857, -0.8571], fopt = 23.2933
        assert_rel_error(self, prob['obj.f'], 23.2933, 1e-5)
        assert_rel_error(self, prob['px.x'][0], 0.2857, 1e-5)
        assert_rel_error(self, prob['px.x'][1], -0.8571, 1e-5)

if __name__ == "__main__":
    unittest.main()
