""" Test for the Backktracking Line Search"""

import numpy as np
import unittest

from six.moves import range

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.solvers.ls_backtracking import BacktrackingLineSearch, ArmijoGoldstein, \
                                             BoundsCheck
from openmdao.solvers.nl_newton import NewtonSolver
from openmdao.solvers.ln_scipy import ScipyIterativeSolver
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.implicit_newton_linesearch \
    import ImplCompOneState, ImplCompTwoStates, ImplCompTwoStatesArrays


class TestBacktrackingLineSearch(unittest.TestCase):

    def test_options(self):
        """Verify that the BacktrackingLineSearch options are declared."""

        group = Group()
        group.nl_solver = BacktrackingLineSearch()

        assert(group.nl_solver.options['bound_enforcement'] == 'vector')
        assert(group.nl_solver.options['rho'] == 0.5)
        assert(group.nl_solver.options['alpha'] == 1.0)

    def test_newton_with_backtracking(self):

        top = Problem()
        root = top.model = Group()
        root.add_subsystem('comp', ImplCompOneState())
        root.add_subsystem('p', IndepVarComp('x', 1.2278849186466743))
        root.connect('p.x', 'comp.x')

        root.nl_solver = NewtonSolver()
        root.ln_solver = ScipyIterativeSolver()
        ls = root.nl_solver.linesearch = BacktrackingLineSearch(rtol=0.9)
        ls.options['maxiter'] = 100
        ls.options['alpha'] = 10.0

        top.setup(check=False)
        top['comp.y'] = 1.0
        top.run_model()

        # This tests that Newton can converge with the line search
        assert_rel_error(self, top['comp.y'], .3968459, .0001)


class TestBacktrackingLineSearchBounds(unittest.TestCase):

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

        ls = top.model.nl_solver.linesearch = BacktrackingLineSearch(rtol=0.9,
                                                                     bound_enforcement='vector')
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

    def test_linesearch_bounds_wall(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = BacktrackingLineSearch(rtol=0.9,
                                                                     bound_enforcement='wall')
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

        ls = top.model.nl_solver.linesearch = BacktrackingLineSearch(rtol=0.9,
                                                                     bound_enforcement='scalar')
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


class TestBacktrackingLineSearchArrayBounds(unittest.TestCase):

    def setUp(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3,1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nl_solver = NewtonSolver()
        top.model.nl_solver.options['maxiter'] = 10
        top.model.ln_solver = ScipyIterativeSolver()

        top.setup(check=False)

        self.top = top
        self.ub = np.array([2.6, 2.5, 2.65])

    def test_nolinesearch(self):
        top = self.top

        # Run without a line search at x=2.0
        top['px.x'] = 2.0
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.y'][ind], 4.666666, 1e-4)
            assert_rel_error(self, top['comp.z'][ind], 1.333333, 1e-4)

        # Run without a line search at x=0.5
        top['px.x'] = 0.5
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.y'][ind], 5.833333, 1e-4)
            assert_rel_error(self, top['comp.z'][ind], 2.666666, 1e-4)

    def test_linesearch_vector_bound_enforcement(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = BacktrackingLineSearch(rtol=0.9,
                                                                     bound_enforcement='vector')
        ls.options['maxiter'] = 10
        ls.options['alpha'] = 10.0

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], 1.5, 1e-8)

        # Test upper bounds: should go to the minimum upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], 2.5, 1e-8)

    def test_linesearch_wall_bound_enforcement_wall(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = BacktrackingLineSearch(rtol=0.9,
                                                                     bound_enforcement='wall')
        ls.options['maxiter'] = 10
        ls.options['alpha'] = 10.0

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], 1.5, 1e-8)

        # Test upper bounds: should go to the upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], self.ub[ind], 1e-8)

    def test_linesearch_wall_bound_enforcement_scalar(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = BacktrackingLineSearch(rtol=0.9,
                                                                     bound_enforcement='scalar')
        ls.options['maxiter'] = 10
        ls.options['alpha'] = 10.0

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


class TestBoundsCheckLineSearchArrayBounds(unittest.TestCase):

    def setUp(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3,1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nl_solver = NewtonSolver()
        top.model.nl_solver.options['maxiter'] = 10
        top.model.ln_solver = ScipyIterativeSolver()

        top.model.suppress_solver_output = True
        top.setup(check=False)

        self.top = top
        self.ub = np.array([2.6, 2.5, 2.65])

    def test_linesearch_vector_bound_enforcement(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = BoundsCheck(bound_enforcement='vector')

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], 1.5, 1e-8)

        # Test upper bounds: should go to the minimum upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], 2.5, 1e-8)

    def test_linesearch_wall_bound_enforcement_wall(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = BoundsCheck(bound_enforcement='wall')

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], 1.5, 1e-8)

        # Test upper bounds: should go to the upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], self.ub[ind], 1e-8)

    def test_linesearch_wall_bound_enforcement_scalar(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = BoundsCheck(bound_enforcement='scalar')

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


class TestArmijoGoldsteinLineSearchArrayBounds(unittest.TestCase):

    def setUp(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3,1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nl_solver = NewtonSolver()
        top.model.nl_solver.options['maxiter'] = 10
        top.model.ln_solver = ScipyIterativeSolver()

        top.model.suppress_solver_output = True
        top.setup(check=False)

        self.top = top
        self.ub = np.array([2.6, 2.5, 2.65])

    def test_linesearch_vector_bound_enforcement(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = ArmijoGoldstein(bound_enforcement='vector')
        ls.options['c'] = .1

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], 1.5, 1e-8)

        # Test upper bounds: should go to the minimum upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], 2.5, 1e-8)

    def test_linesearch_wall_bound_enforcement_wall(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = ArmijoGoldstein(bound_enforcement='wall')

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], 1.5, 1e-8)

        # Test upper bounds: should go to the upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()
        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], self.ub[ind], 1e-8)

    def test_linesearch_wall_bound_enforcement_scalar(self):
        top = self.top

        ls = top.model.nl_solver.linesearch = ArmijoGoldstein(bound_enforcement='scalar')

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


class TestFeatureBacktrackingLineSearch(unittest.TestCase):

    def test_feature_specification(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.connect('px.x', 'comp.x')

        top.model.nl_solver = NewtonSolver()
        top.model.nl_solver.options['maxiter'] = 10
        top.model.ln_solver = ScipyIterativeSolver()

        ls = top.model.nl_solver.linesearch = BacktrackingLineSearch()
        ls.options['maxiter'] = 10
        ls.options['alpha'] = 10.0
        ls.options['rtol'] = 0.9

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

        ls = top.model.nl_solver.linesearch = BoundsCheck()

        top.setup(check=False)

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()

        assert_rel_error(self, top['comp.z'][0], 1.5, 1e-8)
        assert_rel_error(self, top['comp.z'][1], 1.5, 1e-8)
        assert_rel_error(self, top['comp.z'][2], 1.5, 1e-8)

    def test_feature_boundscheck_vector(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3,1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nl_solver = NewtonSolver()
        top.model.nl_solver.options['maxiter'] = 10
        top.model.ln_solver = ScipyIterativeSolver()

        ls = top.model.nl_solver.linesearch = BoundsCheck()
        ls.options['bound_enforcement'] = 'vector'

        top.setup(check=False)

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()

        assert_rel_error(self, top['comp.z'][0], 1.5, 1e-8)
        assert_rel_error(self, top['comp.z'][1], 1.5, 1e-8)
        assert_rel_error(self, top['comp.z'][2], 1.5, 1e-8)

    def test_feature_boundscheck_wall(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3,1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nl_solver = NewtonSolver()
        top.model.nl_solver.options['maxiter'] = 10
        top.model.ln_solver = ScipyIterativeSolver()

        ls = top.model.nl_solver.linesearch = BoundsCheck()
        ls.options['bound_enforcement'] = 'wall'

        top.setup(check=False)

        # Test upper bounds: should go to the upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()

        assert_rel_error(self, top['comp.z'][0], 2.6, 1e-8)
        assert_rel_error(self, top['comp.z'][1], 2.5, 1e-8)
        assert_rel_error(self, top['comp.z'][2], 2.65, 1e-8)

    def test_feature_boundscheck_scalar(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3,1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nl_solver = NewtonSolver()
        top.model.nl_solver.options['maxiter'] = 10
        top.model.ln_solver = ScipyIterativeSolver()

        ls = top.model.nl_solver.linesearch = BoundsCheck()
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
