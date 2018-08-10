""" Test for the Backtracking Line Search"""

import sys
import unittest

from six.moves import cStringIO as StringIO
from six.moves import range

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, DirectSolver, AnalysisError, \
    ExplicitComponent, ImplicitComponent
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.solvers.linesearch.backtracking import ArmijoGoldsteinLS, BoundsEnforceLS
from openmdao.solvers.nonlinear.newton import NewtonSolver
from openmdao.solvers.linear.scipy_iter_solver import ScipyKrylov
from openmdao.test_suite.components.double_sellar import DoubleSellar
from openmdao.test_suite.components.implicit_newton_linesearch \
    import ImplCompTwoStates, ImplCompTwoStatesArrays


class TestArmijoGoldsteinBounds(unittest.TestCase):

    def setUp(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = ScipyKrylov()

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

        ls = top.model.nonlinear_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='vector')
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

        ls = top.model.nonlinear_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='wall')
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

        ls = top.model.nonlinear_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='scalar')
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


class ParaboloidAE(ExplicitComponent):
    """ Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
    This version raises an analysis error if x < 2.0
    The AE in ParaboloidAE stands for AnalysisError."""

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
        Optimal solution (minimum): x = 6.6667; y = -7.3333
        """
        x = inputs['x']
        y = inputs['y']

        if x < 1.75:
            raise AnalysisError('Try Again.')

        outputs['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

    def compute_partials(self, inputs, partials):
        """ Jacobian for our paraboloid."""
        x = inputs['x']
        y = inputs['y']

        partials['f_xy', 'x'] = 2.0*x - 6.0 + y
        partials['f_xy', 'y'] = 2.0*y + 8.0 + x


class TestAnalysisErrorExplicit(unittest.TestCase):

    def setUp(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.add_subsystem('par', ParaboloidAE())
        top.model.connect('px.x', 'comp.x')
        top.model.connect('comp.z', 'par.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 1
        top.model.linear_solver = ScipyKrylov()

        ls = top.model.nonlinear_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='vector')
        ls.options['maxiter'] = 10
        ls.options['alpha'] = 1.0
        top.set_solver_print(level=0)

        self.top = top
        self.ls = ls

    def test_retry(self):
        # Test the behavior with the switch turned on.
        top = self.top
        top.setup(check=False)

        # Test lower bound: should go as far as it can without going past 1.75 and triggering an
        # AnalysisError. It doesn't do a great job, so ends up at 1.8 instead of 1.75
        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 2.1
        top.run_model()
        assert_rel_error(self, top['comp.z'], 1.8, 1e-8)

    def test_no_retry(self):
        # Test the behavior with the switch turned off.
        self.ls.options['retry_on_analysis_error'] = False

        top = self.top
        top.setup(check=False)

        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 2.1

        with self.assertRaises(AnalysisError) as context:
            top.run_model()

        self.assertEqual(str(context.exception), 'Try Again.')


class ImplCompTwoStatesAE(ImplicitComponent):

    def setup(self):
        self.add_input('x', 0.5)
        self.add_output('y', 0.0)
        self.add_output('z', 2.0, lower=1.5, upper=2.5)

        self.maxiter = 10
        self.atol = 1.0e-12

        self.declare_partials(of='*', wrt='*')

        self.counter = 0

    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Don't solve; just calculate the residual.
        """

        x = inputs['x']
        y = outputs['y']
        z = outputs['z']

        residuals['y'] = y - x - 2.0*z
        residuals['z'] = x*z + z - 4.0

        self.counter += 1
        if self.counter > 5 and self.counter < 11:
            raise AnalysisError('catch me')

    def linearize(self, inputs, outputs, jac):
        """
        Analytical derivatives.
        """

        # Output equation
        jac[('y', 'x')] = -1.0
        jac[('y', 'y')] = 1.0
        jac[('y', 'z')] = -2.0

        # State equation
        jac[('z', 'z')] = -inputs['x'] + 1.0
        jac[('z', 'x')] = -outputs['z']


class ImplCompTwoStatesGuess(ImplCompTwoStatesAE):

    def guess_nonlinear(self, inputs, outputs, residuals):
        outputs['z'] = 3.0


class TestAnalysisErrorImplicit(unittest.TestCase):

    def test_deep_analysis_error_iprint(self):

        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', 7.0))

        sub = top.model.add_subsystem('sub', Group())
        sub.add_subsystem('comp', ImplCompTwoStatesAE())

        top.model.connect('px.x', 'sub.comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 2
        top.model.nonlinear_solver.options['solve_subsystems'] = True
        top.model.linear_solver = ScipyKrylov()

        sub.nonlinear_solver = NewtonSolver()
        sub.nonlinear_solver.options['maxiter'] = 2
        sub.linear_solver = ScipyKrylov()

        ls = top.model.nonlinear_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='wall')
        ls.options['maxiter'] = 5
        ls.options['alpha'] = 10.0
        ls.options['retry_on_analysis_error'] = True
        ls.options['c'] = 10000.0

        top.setup(check=False)
        top.set_solver_print(level=2)

        stdout = sys.stdout
        strout = StringIO()

        sys.stdout = strout
        try:
            top.run_model()
        finally:
            sys.stdout = stdout

        output = strout.getvalue().split('\n')
        self.assertTrue(output[26].startswith('|  LS: AG 3'))

    def test_read_only_bug(self):
        # this tests for a bug in which guess_nonlinear failed due to the output
        # vector being left in a read only state after the AnalysisError

        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', 7.0))

        sub = top.model.add_subsystem('sub', Group())
        sub.add_subsystem('comp', ImplCompTwoStatesGuess())

        top.model.connect('px.x', 'sub.comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 2
        top.model.nonlinear_solver.options['solve_subsystems'] = True
        top.model.linear_solver = ScipyKrylov()

        sub.nonlinear_solver = NewtonSolver()
        sub.nonlinear_solver.options['maxiter'] = 2
        sub.linear_solver = ScipyKrylov()

        ls = top.model.nonlinear_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='wall')
        ls.options['maxiter'] = 5
        ls.options['alpha'] = 10.0
        ls.options['retry_on_analysis_error'] = True
        ls.options['c'] = 10000.0

        top.setup(check=False)
        top.set_solver_print(level=2)

        stdout = sys.stdout
        strout = StringIO()

        sys.stdout = strout
        try:
            top.run_model()
        finally:
            sys.stdout = stdout

        output = strout.getvalue().split('\n')
        self.assertTrue(output[26].startswith('|  LS: AG 3'))


class TestBoundsEnforceLSArrayBounds(unittest.TestCase):

    def setUp(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = ScipyKrylov()

        top.set_solver_print(level=0)
        top.setup(check=False)

        self.top = top
        self.ub = np.array([2.6, 2.5, 2.65])

    def test_linesearch_vector_bound_enforcement(self):
        top = self.top

        ls = top.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='vector')
        ls.options['print_bound_enforce'] = True

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

        stdout = sys.stdout
        strout = StringIO()

        sys.stdout = strout
        try:
            top.run_model()
        finally:
            sys.stdout = stdout

        txt = strout.getvalue()

        self.assertTrue("'comp.z' exceeds upper bound" in txt)

        for ind in range(3):
            assert_rel_error(self, top['comp.z'][ind], [2.5], 1e-8)

    def test_linesearch_wall_bound_enforcement_wall(self):
        top = self.top

        top.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='wall')

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

        top.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='scalar')

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

    def test_error_handling(self):
        # Make sure the debug_print doesn't bomb out.

        class Bad(ExplicitComponent):

            def setup(self):
                self.add_input('x', val=0.0)
                self.add_input('y', val=0.0)

                self.add_output('f_xy', val=0.0, upper=1.0)

                self.declare_partials(of='*', wrt='*')
                self.count = 0

            def compute(self, inputs, outputs):
                if self.count < 1:
                    x = inputs['x']
                    y = inputs['y']
                    outputs['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0
                else:
                    outputs['f_xy'] = np.inf

                self.count += 1

            def compute_partials(self, inputs, partials):
                x = inputs['x']
                y = inputs['y']

                partials['f_xy', 'x'] = 2.0*x - 6.0 + y
                partials['f_xy', 'y'] = 2.0*y + 8.0 + x

        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.add_subsystem('par', Bad())
        top.model.connect('px.x', 'comp.x')
        top.model.connect('comp.z', 'par.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 3

        top.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='vector')
        top.set_solver_print(level=0)

        top.setup(check=False)

        # Make sure we don't raise an error when we reach the final debug print.
        top.run_model()

    def test_undeclared_options(self):
        # Test that using options that should not exist in class, cause an
        # error if they are set when instantiating BoundsEnforceLS.
        # atol, rtol, maxiter, and err_on_maxiter are not used in BoundsEnforceLS

        with self.assertRaises(KeyError) as context:
            BoundsEnforceLS(bound_enforcement='scalar', atol=1.0)

        self.assertEqual(str(context.exception), "\"Key 'atol' cannot be set because it "
                                                 "has not been declared.\"")

        with self.assertRaises(KeyError) as context:
            BoundsEnforceLS(bound_enforcement='scalar', rtol=2.0)

        self.assertEqual(str(context.exception), "\"Key 'rtol' cannot be set because it "
                                                 "has not been declared.\"")

        with self.assertRaises(KeyError) as context:
            BoundsEnforceLS(bound_enforcement='scalar', maxiter=1)

        self.assertEqual(str(context.exception), "\"Key 'maxiter' cannot be set because it "
                                                 "has not been declared.\"")

        with self.assertRaises(KeyError) as context:
            BoundsEnforceLS(bound_enforcement='scalar', err_on_maxiter=True)

        self.assertEqual(str(context.exception), "\"Key 'err_on_maxiter' cannot be set because it "
                                                 "has not been declared.\"")


class TestArmijoGoldsteinLSArrayBounds(unittest.TestCase):

    def setUp(self):
        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = ScipyKrylov()

        top.set_solver_print(level=0)
        top.setup(check=False)

        self.top = top
        self.ub = np.array([2.6, 2.5, 2.65])

    def test_linesearch_vector_bound_enforcement(self):
        top = self.top

        ls = top.model.nonlinear_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='vector')
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

        top.model.nonlinear_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='wall')

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

        top.model.nonlinear_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='scalar')

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

        g1 = model.g1
        g1.nonlinear_solver = NewtonSolver()
        g1.nonlinear_solver.options['rtol'] = 1.0e-5
        g1.linear_solver = DirectSolver()

        g2 = model.g2
        g2.nonlinear_solver = NewtonSolver()
        g2.nonlinear_solver.options['rtol'] = 1.0e-5
        g2.linear_solver = DirectSolver()

        model.nonlinear_solver = NewtonSolver()
        model.linear_solver = ScipyKrylov()

        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 4
        ls = model.nonlinear_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='vector')

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
        from openmdao.api import Problem, Group, IndepVarComp, NewtonSolver, ScipyKrylov, ArmijoGoldsteinLS
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStates

        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = ScipyKrylov()

        ls = top.model.nonlinear_solver.linesearch = ArmijoGoldsteinLS()
        ls.options['maxiter'] = 10

        top.setup(check=False)

        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 1.6
        top.run_model()
        assert_rel_error(self, top['comp.z'], 1.5, 1e-8)

    def test_feature_boundsenforcels_basic(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NewtonSolver, ScipyKrylov, BoundsEnforceLS
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = ScipyKrylov()

        top.model.nonlinear_solver.linesearch = BoundsEnforceLS()

        top.setup(check=False)

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()

        assert_rel_error(self, top['comp.z'][0], [1.5], 1e-8)
        assert_rel_error(self, top['comp.z'][1], [1.5], 1e-8)
        assert_rel_error(self, top['comp.z'][2], [1.5], 1e-8)

    def test_feature_armijogoldsteinls_basic(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NewtonSolver, ScipyKrylov, ArmijoGoldsteinLS
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = ScipyKrylov()

        top.model.nonlinear_solver.linesearch = ArmijoGoldsteinLS()

        top.setup(check=False)

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()

        assert_rel_error(self, top['comp.z'][0], [1.5], 1e-8)
        assert_rel_error(self, top['comp.z'][1], [1.5], 1e-8)
        assert_rel_error(self, top['comp.z'][2], [1.5], 1e-8)

    def test_feature_boundscheck_basic(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NewtonSolver, ScipyKrylov, BoundsEnforceLS
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = ScipyKrylov()

        top.model.nonlinear_solver.linesearch = BoundsEnforceLS()

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
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NewtonSolver, ScipyKrylov, BoundsEnforceLS
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = ScipyKrylov()

        ls = top.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='vector')
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
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NewtonSolver, ScipyKrylov, BoundsEnforceLS
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = ScipyKrylov()

        ls = top.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='wall')
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
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NewtonSolver, ScipyKrylov, BoundsEnforceLS
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = ScipyKrylov()

        ls = top.model.nonlinear_solver.linesearch = BoundsEnforceLS(bound_enforcement='scalar')
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

    def test_feature_print_bound_enforce(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NewtonSolver, ScipyKrylov, BoundsEnforceLS
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        newt = top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 2
        top.model.linear_solver = ScipyKrylov()

        ls = newt.linesearch = BoundsEnforceLS(bound_enforcement='vector')
        ls.options['print_bound_enforce'] = True

        top.set_solver_print(level=2)
        top.setup()

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()

        assert_rel_error(self, top['comp.z'][0], [1.5], 1e-8)
        assert_rel_error(self, top['comp.z'][1], [1.5], 1e-8)
        assert_rel_error(self, top['comp.z'][2], [1.5], 1e-8)

    def test_feature_armijo_boundscheck_vector(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NewtonSolver, ScipyKrylov, ArmijoGoldsteinLS
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = ScipyKrylov()

        ls = top.model.nonlinear_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='vector')
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

    def test_feature_armijo_boundscheck_wall(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NewtonSolver, ScipyKrylov, ArmijoGoldsteinLS
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = ScipyKrylov()

        ls = top.model.nonlinear_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='wall')
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

    def test_feature_armijo_boundscheck_scalar(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NewtonSolver, ScipyKrylov, ArmijoGoldsteinLS
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = ScipyKrylov()

        ls = top.model.nonlinear_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='scalar')
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

    def test_feature_armijo_print_bound_enforce(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NewtonSolver, ScipyKrylov, ArmijoGoldsteinLS
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = Problem()
        top.model = Group()
        top.model.add_subsystem('px', IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        newt = top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 2
        top.model.linear_solver = ScipyKrylov()

        ls = newt.linesearch = ArmijoGoldsteinLS()
        ls.options['print_bound_enforce'] = True

        top.set_solver_print(level=2)
        top.setup()

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()

        assert_rel_error(self, top['comp.z'][0], [1.5], 1e-8)
        assert_rel_error(self, top['comp.z'][1], [1.5], 1e-8)
        assert_rel_error(self, top['comp.z'][2], [1.5], 1e-8)


if __name__ == "__main__":
    unittest.main()
