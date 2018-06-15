"""Test the Broyden nonlinear solver. """
from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, LinearRunOnce, ImplicitComponent, IndepVarComp, DirectSolver
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

        self.declare_partials(of=['x12', 'x3', 'x45'], wrt='c')
        self.declare_partials(of='x12', wrt='x12', rows=np.arange(2), cols=np.arange(2),
                              val=-np.array([3.0, 2]))
        self.declare_partials(of='x3', wrt='x3', rows=np.arange(1), cols=np.arange(1),
                              val=-np.array([1.5]))
        self.declare_partials(of='x45', wrt='x45', rows=np.arange(2), cols=np.arange(2),
                              val=-np.array([1, 0.5]))

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

    def linearize(self, inputs, outputs, jacobian):
        c = inputs['c']
        x = np.empty((5, ))
        x12 = outputs['x12']
        x3 = outputs['x3']
        x45 = outputs['x45']

        jacobian['x12', 'c'] = -3.0 * x12**2
        jacobian['x3', 'c'] = -3.0 * x3**2
        jacobian['x45', 'c'] = -3.0 * x45**2


class ForBroydenResetJac(ImplicitComponent):
    """
    Pretty simple equation, but initial value will jump around for the first several
    iterations to force recomputation of jacobian.
    """
    def setup(self):
        self.add_input('x', 1.0)
        self.add_output('y', 1.0)

        self.declare_partials(of='y', wrt=['x', 'y'])

    def apply_nonlinear(self, inputs, outputs, residuals):
        x = inputs['x']
        y = outputs['y']

        vals = np.array([3.0, -2.5, -50.7, 140.0, -2500.5, 3.0])
        val = vals[min(len(vals) - 1, self.iter_count)]

        residuals['y'] = val*y**2 - x

    def linearize(self, inputs, outputs, jacobian):
        x = inputs['x']
        y = outputs['y']

        jacobian['y', 'x'] = -1.0

        jacobian['y', 'y'] = 6.0 * y


class TestBryoden(unittest.TestCase):

    def test_simple_sellar(self):
        # Test top level Sellar (i.e., not grouped).

        prob = Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=BroydenSolver(),
                                                   linear_solver=LinearRunOnce())

        prob.setup(check=False)

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

    def test_mixed_jacobian(self):
        # Testing Broyden on a 5 state case split among 3 vars.

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('c', 0.01))
        model.add_subsystem('mixed', MixedEquation())

        model.connect('p1.c', 'mixed.c')

        model.nonlinear_solver = BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['mixed.x12', 'mixed.x3', 'mixed.x45']
        model.nonlinear_solver.options['maxiter'] = 15
        model.nonlinear_solver.options['compute_jacobian'] = True
        model.nonlinear_solver.linear_solver = DirectSolver()

        prob.setup(check=False)

        prob.run_model()

        assert_rel_error(self, prob['mixed.x12'], np.zeros((2, )), 1e-6)
        assert_rel_error(self, prob['mixed.x3'], 0.0, 1e-6)
        assert_rel_error(self, prob['mixed.x45'], np.zeros((2, )), 1e-6)

        # Normally takes about 13 iters, but takes around 4 if you calculate an initial
        # Jacobian.
        self.assertTrue(model.nonlinear_solver._iter_count < 6)

    def test_simple_sellar_jacobian(self):
        # Test top level Sellar (i.e., not grouped).

        prob = Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=BroydenSolver(),
                                                   linear_solver=LinearRunOnce())

        prob.setup(check=False)

        model.nonlinear_solver.options['state_vars'] = ['state_eq.y2_command']
        model.nonlinear_solver.options['compute_jacobian'] = True
        model.nonlinear_solver.linear_solver = DirectSolver()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['state_eq.y2_command'], 12.05848819, .00001)

        # Normally takes about 4 iters, but takes around 3 if you calculate an initial
        # Jacobian.
        self.assertTrue(model.nonlinear_solver._iter_count < 4)

    def test_jacobian_update(self):
        # Test top level Sellar (i.e., not grouped).

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 0.01))
        model.add_subsystem('comp', ForBroydenResetJac())

        model.connect('p1.x', 'comp.x')

        model.nonlinear_solver = BroydenSolver()
        model.nonlinear_solver.options['state_vars'] = ['comp.y']
        model.nonlinear_solver.options['maxiter'] = 30
        model.nonlinear_solver.options['compute_jacobian'] = True
        model.nonlinear_solver.options['max_converge_failures'] = 200
        model.nonlinear_solver.options['diverge_limit'] = np.inf
        model.nonlinear_solver.linear_solver = DirectSolver()

        prob.setup(check=False)

        prob.set_solver_print(level=2)
        prob.run_model()

        assert_rel_error(self, prob['comp.y'], 0.05773503, .00001)

if __name__ == "__main__":
    unittest.main()