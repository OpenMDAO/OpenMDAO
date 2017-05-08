""" Test for the Backktracking Line Search"""

import numpy as np
import unittest

from six.moves import range

from openmdao.api import Problem, Group, IndepVarComp, DirectSolver
from openmdao.devtools.testutil import assert_rel_error
from openmdao.solvers.ls_backtracking import ArmijoGoldsteinLS, BoundsEnforceLS
from openmdao.solvers.nl_newton import NewtonSolver
from openmdao.solvers.ln_scipy import ScipyIterativeSolver
from openmdao.test_suite.components.double_sellar import DoubleSellar
from openmdao.test_suite.components.implicit_newton_linesearch \
    import ImplCompOneState, ImplCompTwoStates, ImplCompTwoStatesArrays


class TestArmejoGoldsteinBounds(unittest.TestCase):

    def setUp(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.connect('px.x', 'comp.x')

        top.model.nl_solver = NewtonSolver()
        top.model.nl_solver.options['maxiter'] = 10
        top.model.ln_solver = ScipyIterativeSolver()

        top.setup(check=False)

        self.top = top

    def test_nolinesearch(self):
        top = self.top

        # Run without a line search at x=2.0
        top['px.x'] = 2.0
        top.run_model()
        assert_rel_error(self, top['comp.y'], 4.666666, 1e-4)
        assert_rel_error(self, top['comp.z'], 1.333333, 1e-4)

        # Run without a line search at x=0.5
        top['px.x'] = 0.5
        top.run_model()
        assert_rel_error(self, top['comp.y'], 5.833333, 1e-4)
        assert_rel_error(self, top['comp.z'], 2.666666, 1e-4)

    def test_linesearch_bounds_vector(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='vector')
        ls.options['maxiter'] = 10
        ls.options['alpha'] = 1.0

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bound: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 1.6
        top.run_model()
        assert_rel_error(self, top['comp.z'], 1.5, 1e-8)

        # Test upper bound: should go to the upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.0
        top['comp.z'] = 2.4
        top.run_model()
        assert_rel_error(self, top['comp.z'], 2.5, 1e-8)

    def test_linesearch_bounds_wall(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='wall')
        ls.options['maxiter'] = 10
        ls.options['alpha'] = 10.0

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bound: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 1.6
        top.run_model()
        assert_rel_error(self, top['comp.z'], 1.5, 1e-8)

        # Test upper bound: should go to the upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.0
        top['comp.z'] = 2.4
        top.run_model()
        assert_rel_error(self, top['comp.z'], 2.5, 1e-8)

    def test_linesearch_bounds_scalar(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='scalar')
        ls.options['maxiter'] = 10
        ls.options['alpha'] = 10.0

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bound: should stop just short of the lower bound
        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 1.6
        top.run_model()
        self.assertTrue(1.5 <= top['comp.z'] <= 1.6)

        # Test lower bound: should stop just short of the upper bound
        top['px.x'] = 0.5
        top['comp.y'] = 0.0
        top['comp.z'] = 2.4
        top.run_model()
        self.assertTrue(2.4 <= top['comp.z'] <= 2.5)


class TestBoundsEnforceLSArrayBounds(unittest.TestCase):

    def setUp(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3,1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nl_solver = NewtonSolver()
        top.model.nl_solver.options['maxiter'] = 10
        top.model.ln_solver = ScipyIterativeSolver()

        top.set_solver_print(level=0)
        top.setup(check=False)

        self.top = top
        self.ub = np.array([2.6, 2.5, 2.65])

    def test_linesearch_vector_bound_enforcement(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = BoundsEnforceLS(bound_enforcement='vector')

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], [1.5], 1e-8)

        # Test upper bounds: should go to the minimum upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], [2.5], 1e-8)

    def test_linesearch_wall_bound_enforcement_wall(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = BoundsEnforceLS(bound_enforcement='wall')

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], [1.5], 1e-8)

        # Test upper bounds: should go to the upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], [self.ub[ind]], 1e-8)

    def test_linesearch_wall_bound_enforcement_scalar(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = BoundsEnforceLS(bound_enforcement='scalar')

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bounds: should stop just short of the lower bound
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()
        for ind in range(3):
            self.assertTrue(1.5 <= top['comp.z'][ind] <= 1.6)

        # Test upper bounds: should stop just short of the minimum upper bound
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()
        for ind in range(3):
            self.assertTrue(2.4 <= top['comp.z'][ind] <= self.ub[ind])


class TestArmijoGoldsteinLSArrayBounds(unittest.TestCase):

    def setUp(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3,1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nl_solver = NewtonSolver()
        top.model.nl_solver.options['maxiter'] = 10
        top.model.ln_solver = ScipyIterativeSolver()

        top.set_solver_print(level=0)
        top.setup(check=False)

        self.top = top
        self.ub = np.array([2.6, 2.5, 2.65])

    def test_linesearch_vector_bound_enforcement(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='vector')
        ls.options['c'] = .1

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], [1.5], 1e-8)

        # Test upper bounds: should go to the minimum upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], [2.5], 1e-8)

    def test_linesearch_wall_bound_enforcement_wall(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='wall')

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], [1.5], 1e-8)

        # Test upper bounds: should go to the upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], [self.ub[ind]], 1e-8)

    def test_linesearch_wall_bound_enforcement_scalar(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='scalar')

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bounds: should stop just short of the lower bound
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()
        for ind in range(3):
            self.assertTrue(1.5 <= top['comp.z'][ind] <= 1.6)

        # Test upper bounds: should stop just short of the minimum upper bound
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()
        for ind in range(3):
            self.assertTrue(2.4 <= top['comp.z'][ind] <= self.ub[ind])

    def test_with_subsolves(self):
        prob = Problem()
        model = prob.model = DoubleSellar()

        g1 = model.get_subsystem('g1')
        g1.nl_solver = NewtonSolver()
        g1.nl_solver.options['rtol'] = 1.0e-5
        g1.ln_solver = DirectSolver()

        g2 = model.get_subsystem('g2')
        g2.nl_solver = NewtonSolver()
        g2.nl_solver.options['rtol'] = 1.0e-5
        g2.ln_solver = DirectSolver()

        model.nl_solver = NewtonSolver()
        model.ln_solver = ScipyIterativeSolver()

        model.nl_solver.options['solve_subsystems'] = True
        model.nl_solver.options['max_sub_solves'] = 4
        ls = model.nl_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='vector')

        # This is pretty bogus, but it ensures that we get a few LS iterations.
        ls.options['c'] = 100.0

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['g1.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g1.y2'], 0.80, .00001)
        assert_rel_error(self, prob['g2.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g2.y2'], 0.80, .00001)


class TestFeatureLineSearch(unittest.TestCase):

    def test_feature_specification(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.connect('px.x', 'comp.x')

        top.model.nl_solver = NewtonSolver()
        top.model.nl_solver.options['maxiter'] = 10
        top.model.ln_solver = ScipyIterativeSolver()

        ls = top.model.nl_solver.linesearch = ArmijoGoldsteinLS()
        ls.options['maxiter'] = 10

        top.setup(check=False)

        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 1.6
        top.run_model()
        assert_rel_error(self, top['comp.z'], 1.5, 1e-8)

    def test_feature_boundscheck_basic(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3,1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nl_solver = NewtonSolver()
        top.model.nl_solver.options['maxiter'] = 10
        top.model.ln_solver = ScipyIterativeSolver()

        ls = top.model.nl_solver.linesearch = BoundsEnforceLS()

        top.setup(check=False)

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()

        assert_rel_error(self, top['comp.z'][0], [1.5], 1e-8)
        assert_rel_error(self, top['comp.z'][1], [1.5], 1e-8)
        assert_rel_error(self, top['comp.z'][2], [1.5], 1e-8)

    def test_feature_boundscheck_vector(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3,1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nl_solver = NewtonSolver()
        top.model.nl_solver.options['maxiter'] = 10
        top.model.ln_solver = ScipyIterativeSolver()

        ls = top.model.nl_solver.linesearch = BoundsEnforceLS()
        ls.options['bound_enforcement'] = 'vector'

        top.setup(check=False)

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()

        assert_rel_error(self, top['comp.z'][0], [1.5], 1e-8)
        assert_rel_error(self, top['comp.z'][1], [1.5], 1e-8)
        assert_rel_error(self, top['comp.z'][2], [1.5], 1e-8)

    def test_feature_boundscheck_wall(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3,1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nl_solver = NewtonSolver()
        top.model.nl_solver.options['maxiter'] = 10
        top.model.ln_solver = ScipyIterativeSolver()

        ls = top.model.nl_solver.linesearch = BoundsEnforceLS()
        ls.options['bound_enforcement'] = 'wall'

        top.setup(check=False)

        # Test upper bounds: should go to the upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()

        assert_rel_error(self, top['comp.z'][0], [2.6], 1e-8)
        assert_rel_error(self, top['comp.z'][1], [2.5], 1e-8)
        assert_rel_error(self, top['comp.z'][2], [2.65], 1e-8)

    def test_feature_boundscheck_scalar(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3,1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nl_solver = NewtonSolver()
        top.model.nl_solver.options['maxiter'] = 10
        top.model.ln_solver = ScipyIterativeSolver()

        ls = top.model.nl_solver.linesearch = BoundsEnforceLS()
        ls.options['bound_enforcement'] = 'scalar'

        top.setup(check=False)
        top.run_model()

        # Test lower bounds: should stop just short of the lower bound
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()

        print(top['comp.z'][0])
        print(top['comp.z'][1])
        print(top['comp.z'][2])


if __name__ == "__main__":
    unittest.main()
