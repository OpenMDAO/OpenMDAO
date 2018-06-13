"""Test the Broyden nonlinear solver. """
from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, LinearRunOnce, ImplicitComponent, IndepVarComp
from openmdao.solvers.nonlinear.broyden import BroydenSolver
from openmdao.test_suite.components.sellar import SellarStateConnection
from openmdao.utils.assert_utils import assert_rel_error


class VectorEquation(ImplicitComponent):
    """Equation with 5 states in a single vector. Should converge to x=[0,0,0,0,0]"""

    def setup(self):
        self.add_input('c', 0.01)
        self.add_output('x', np.ones((5, )))

    def apply_nonlinear(self, inputs, outputs, residuals):
        c = inputs['c']
        x = outputs['x']

        d = np.array([3, 2, 1.5, 1, 0.5])

        residuals['x'] = -d*x - c*x**3


class MixedEquation(ImplicitComponent):
    """Equation with 5 states split between 3 vars. Should converge to x=[0,0,0,0,0]"""

    def setup(self):
        self.add_input('c', 0.01)
        self.add_output('x12', np.ones((2, )))
        self.add_output('x3', 1.0)
        self.add_output('x45', np.ones((2, )))

    def apply_nonlinear(self, inputs, outputs, residuals):
        c = inputs['c']
        x = np.empty((5, ))
        x[:2] = outputs['x12']
        x[2] = outputs['x3']
        x[3:] = outputs['x45']

        d = np.array([3, 2, 1.5, 1, 0.5])

        res = -d*x - c*x**3
        residuals['x12'] = res[:2]
        residuals['x3'] = res[2]
        residuals['x45'] = res[3:]


class TestBryoden(unittest.TestCase):

    def test_simple_Sellar(self):
        # Test top level Sellar (i.e., not grouped).

        prob = Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=BroydenSolver(),
                                                   linear_solver=LinearRunOnce())

        prob.setup(check=False)

        prob.set_solver_print(level=0)
        model.nonlinear_solver.options['state_vars'] = ['state_eq.y2_command']

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['state_eq.y2_command'], 12.05848819, .00001)

    def test_vector(self):
        # Testing Broyden on a 5 state single vector case.

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('c', 0.01))
        model.add_subsystem('vec', VectorEquation())

        model.connect('p1.c', 'vec.c')

        model.nonlinear_solver = BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['vec.x']
        model.nonlinear_solver.options['maxiter'] = 15

        prob.setup(check=False)

        prob.run_model()

        assert_rel_error(self, prob['vec.x'], np.zeros((5, )), 1e-6)

    def test_mixed(self):
        # Testing Broyden on a 5 state case split among 3 vars.

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('c', 0.01))
        model.add_subsystem('mixed', MixedEquation())

        model.connect('p1.c', 'mixed.c')

        model.nonlinear_solver = BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['mixed.x12', 'mixed.x3', 'mixed.x45']
        model.nonlinear_solver.options['maxiter'] = 15

        prob.setup(check=False)

        prob.run_model()

        assert_rel_error(self, prob['mixed.x12'], np.zeros((2, )), 1e-6)
        assert_rel_error(self, prob['mixed.x3'], 0.0, 1e-6)
        assert_rel_error(self, prob['mixed.x45'], np.zeros((2, )), 1e-6)

    def test_mixed_promoted_vars(self):
        # Testing Broyden on a 5 state case split among 3 vars.

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('c', 0.01))
        model.add_subsystem('mixed', MixedEquation(), promotes_outputs=['x12', 'x3', 'x45'])

        model.connect('p1.c', 'mixed.c')

        model.nonlinear_solver = BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['x12', 'x3', 'x45']
        model.nonlinear_solver.options['maxiter'] = 15

        prob.setup(check=False)

        prob.run_model()

        assert_rel_error(self, prob['x12'], np.zeros((2, )), 1e-6)
        assert_rel_error(self, prob['x3'], 0.0, 1e-6)
        assert_rel_error(self, prob['x45'], np.zeros((2, )), 1e-6)

    def test_error_not_state(self):
        # Testing Broyden on a 5 state case split among 3 vars.

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('c', 0.01))
        model.add_subsystem('mixed', VectorEquation())

        model.connect('p1.c', 'mixed.c')

        model.nonlinear_solver = BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['p1.c']

        prob.setup(check=False)

        prob.run_model()

if __name__ == "__main__":
    unittest.main()