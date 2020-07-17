""" Test for the Backtracking Line Search"""

import sys
import unittest
from math import atan

from io import StringIO

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.double_sellar import DoubleSellar
from openmdao.test_suite.components.implicit_newton_linesearch \
    import ImplCompTwoStates, ImplCompTwoStatesArrays
from openmdao.test_suite.components.sellar import SellarDis1, SellarDis2withDerivatives
from openmdao.utils.assert_utils import assert_near_equal


class TestArmejoGoldsteinBounds(unittest.TestCase):

    def setUp(self):
        top = om.Problem()
        top.model.add_subsystem('px', om.IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = om.ScipyKrylov()

        top.setup()

        self.top = top

    def test_nolinesearch(self):
        top = self.top
        top.model.nonlinear_solver.linesearch = None

        # Run without a line search at x=2.0
        top['px.x'] = 2.0
        top.run_model()
        assert_near_equal(top['comp.y'], 4.666666, 1e-4)
        assert_near_equal(top['comp.z'], 1.333333, 1e-4)

        # Run without a line search at x=0.5
        top['px.x'] = 0.5
        top.run_model()
        assert_near_equal(top['comp.y'], 5.833333, 1e-4)
        assert_near_equal(top['comp.z'], 2.666666, 1e-4)

    def test_linesearch_bounds_vector(self):
        top = self.top

        ls = top.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='vector')
        ls.options['maxiter'] = 10
        ls.options['alpha'] = 1.0

        # Setup again because we assigned a new linesearch
        top.setup()

        # Test lower bound: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 1.6
        top.run_model()
        assert_near_equal(top['comp.z'], 1.5, 1e-8)

        # Test upper bound: should go to the upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.0
        top['comp.z'] = 2.4
        top.run_model()
        assert_near_equal(top['comp.z'], 2.5, 1e-8)

    def test_linesearch_bounds_wall(self):
        top = self.top

        ls = top.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='wall')
        ls.options['maxiter'] = 10
        ls.options['alpha'] = 10.0

        # Setup again because we assigned a new linesearch
        top.setup()

        # Test lower bound: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 1.6
        top.run_model()
        assert_near_equal(top['comp.z'], 1.5, 1e-8)

        # Test upper bound: should go to the upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.0
        top['comp.z'] = 2.4
        top.run_model()
        assert_near_equal(top['comp.z'], 2.5, 1e-8)

    def test_linesearch_bounds_scalar(self):
        top = self.top

        ls = top.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='scalar')
        ls.options['maxiter'] = 10
        ls.options['alpha'] = 1.0

        # Setup again because we assigned a new linesearch
        top.setup()

        # Test lower bound: should stop just short of the lower bound
        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 1.6
        top.run_model()
        self.assertGreaterEqual(top['comp.z'], 1.5)
        self.assertLessEqual(top['comp.z'], 1.6)

        # Test lower bound: should stop just short of the upper bound
        top['px.x'] = 0.5
        top['comp.y'] = 0.0
        top['comp.z'] = 2.4
        top.run_model()
        self.assertGreaterEqual(top['comp.z'], 2.5)
        self.assertLessEqual(top['comp.z'], 2.5)

    def test_bound_enforce_print_bug(self):
        # Error during print if bound was one-sided.

        class OneSidedBounds(ImplCompTwoStatesArrays):

            def setup(self):
                self.add_input('x', np.zeros((3, 1)))
                self.add_output('y', np.zeros((3, 1)))
                self.add_output('z', 2.0*np.ones((3, 1)),
                    upper=np.array([2.6, 2.5, 2.65]).reshape((3,1)))

                self.maxiter = 10
                self.atol = 1.0e-12

                self.declare_partials(of='*', wrt='*')

        top = om.Problem()
        top.model.add_subsystem('px', om.IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', OneSidedBounds())
        top.model.connect('px.x', 'comp.x')

        newt = top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 2
        top.model.linear_solver = om.ScipyKrylov()

        ls = newt.linesearch = om.ArmijoGoldsteinLS()
        ls.options['print_bound_enforce'] = True

        top.set_solver_print(level=2)
        top.setup()

        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6

        # Should run without an exception being raised.
        top.run_model()


class ParaboloidAE(om.ExplicitComponent):
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
            raise om.AnalysisError('Try Again.')

        outputs['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

    def compute_partials(self, inputs, partials):
        """ Jacobian for our paraboloid."""
        x = inputs['x']
        y = inputs['y']

        partials['f_xy', 'x'] = 2.0*x - 6.0 + y
        partials['f_xy', 'y'] = 2.0*y + 8.0 + x


class TestAnalysisErrorExplicit(unittest.TestCase):

    def setUp(self):
        top = om.Problem()
        top.model.add_subsystem('px', om.IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.add_subsystem('par', ParaboloidAE())
        top.model.connect('px.x', 'comp.x')
        top.model.connect('comp.z', 'par.x')

        top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 1
        top.model.linear_solver = om.ScipyKrylov()

        ls = top.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='vector')
        ls.options['maxiter'] = 10
        ls.options['alpha'] = 1.0
        top.set_solver_print(level=0)

        self.top = top
        self.ls = ls

    def test_retry(self):
        # Test the behavior with the switch turned on.
        top = self.top
        top.setup()
        self.ls.options['retry_on_analysis_error'] = True

        # Test lower bound: should go as far as it can without going past 1.75 and triggering an
        # AnalysisError. It doesn't do a great job, so ends up at 1.8 instead of 1.75
        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 2.1
        top.run_model()
        assert_near_equal(top['comp.z'], 1.8, 1e-8)

    def test_no_retry(self):
        # Test the behavior with the switch turned off.
        self.ls.options['retry_on_analysis_error'] = False

        top = self.top
        top.setup()

        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 2.1

        with self.assertRaises(om.AnalysisError) as context:
            top.run_model()

        self.assertEqual(str(context.exception), 'Try Again.')


class ImplCompTwoStatesAE(om.ImplicitComponent):

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
        self.upper = 1
        self.lower = 0

        x = inputs['x']
        y = outputs['y']
        z = outputs['z']

        residuals['y'] = y - x - 2.0*z
        residuals['z'] = x*z + z - 4.0

        self.counter += 1
        if self.counter > self.lower and self.counter < self.upper:
            raise om.AnalysisError('catch me')

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

        top = om.Problem()
        top.model.add_subsystem('px', om.IndepVarComp('x', 7.0))

        sub = top.model.add_subsystem('sub', om.Group())
        sub.add_subsystem('comp', ImplCompTwoStatesAE())
        sub.upper = 5
        sub.lower = 11

        top.model.connect('px.x', 'sub.comp.x')

        top.model.nonlinear_solver = om.NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 2
        top.model.nonlinear_solver.options['solve_subsystems'] = True
        top.model.nonlinear_solver.linesearch = None
        top.model.linear_solver = om.ScipyKrylov()

        sub.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        sub.nonlinear_solver.options['maxiter'] = 2
        sub.nonlinear_solver.linesearch = None
        sub.linear_solver = om.ScipyKrylov()

        ls = top.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='wall')
        ls.options['maxiter'] = 5
        ls.options['alpha'] = 10.0
        ls.options['retry_on_analysis_error'] = True
        ls.options['c'] = 1.0

        top.setup()
        top.set_solver_print(level=2)

        stdout = sys.stdout
        strout = StringIO()

        sys.stdout = strout
        try:
            top.run_model()
        finally:
            sys.stdout = stdout

        output = strout.getvalue().split('\n')

        correct = False
        for line in output:
            # make sure a line starting with this string is present in stdout
            if line.startswith('|  LS: AG 3'):
                correct = True
                break
        self.assertTrue(correct, msg='Expected line search output not found in stdout')

    def test_read_only_bug(self):
        # this tests for a bug in which guess_nonlinear failed due to the output
        # vector being left in a read only state after the AnalysisError

        top = om.Problem()
        top.model.add_subsystem('px', om.IndepVarComp('x', 7.0))

        sub = top.model.add_subsystem('sub', om.Group())
        sub.add_subsystem('comp', ImplCompTwoStatesAE())
        sub.upper = 20
        sub.lower = 25

        top.model.connect('px.x', 'sub.comp.x')

        top.model.nonlinear_solver = om.NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 2
        top.model.nonlinear_solver.options['solve_subsystems'] = True
        top.model.nonlinear_solver.linesearch = None
        top.model.linear_solver = om.ScipyKrylov()

        sub.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        sub.nonlinear_solver.options['maxiter'] = 2
        sub.nonlinear_solver.linesearch = None
        sub.linear_solver = om.ScipyKrylov()

        ls = top.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='wall')
        ls.options['maxiter'] = 5
        ls.options['alpha'] = 10.0
        ls.options['retry_on_analysis_error'] = True
        ls.options['c'] = 1.0

        top.setup()
        top.set_solver_print(level=2)

        stdout = sys.stdout
        strout = StringIO()

        sys.stdout = strout
        try:
            top.run_model()
        finally:
            sys.stdout = stdout

        output = strout.getvalue().split('\n')

        correct = False
        for line in output:
            # make sure a line starting with this string is present in stdout
            if line.startswith('|  LS: AG 3'):
                correct = True
                break
        self.assertTrue(correct, msg='Expected line search output not found in stdout')


class TestBoundsEnforceLSArrayBounds(unittest.TestCase):

    def setUp(self):
        top = om.Problem()
        top.model.add_subsystem('px', om.IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = om.ScipyKrylov()

        top.set_solver_print(level=0)
        top.setup()

        self.top = top
        self.ub = np.array([2.6, 2.5, 2.65])

    def test_linesearch_vector_bound_enforcement(self):
        top = self.top

        ls = top.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement='vector')
        ls.options['print_bound_enforce'] = True

        # Setup again because we assigned a new linesearch
        top.setup()

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()
        for ind in range(3):
            assert_near_equal(top['comp.z'][ind], [1.5], 1e-8)

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
            assert_near_equal(top['comp.z'][ind], [2.5], 1e-8)

    def test_linesearch_wall_bound_enforcement_wall(self):
        top = self.top

        top.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement='wall')

        # Setup again because we assigned a new linesearch
        top.setup()

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()
        for ind in range(3):
            assert_near_equal(top['comp.z'][ind], [1.5], 1e-8)

        # Test upper bounds: should go to the upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()
        for ind in range(3):
            assert_near_equal(top['comp.z'][ind], [self.ub[ind]], 1e-8)

    def test_linesearch_wall_bound_enforcement_scalar(self):
        top = self.top

        top.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement='scalar')

        # Setup again because we assigned a new linesearch
        top.setup()

        # Test lower bounds: should stop just short of the lower bound
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()
        for ind in range(3):
            self.assertGreaterEqual(top['comp.z'][ind], 1.5)
            self.assertLessEqual(top['comp.z'][ind], 1.6)

        # Test upper bounds: should stop just short of the minimum upper bound
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()
        for ind in range(3):
            self.assertTrue(2.4 <= top['comp.z'][ind] <= self.ub[ind])

    def test_error_handling(self):
        # Make sure the debug_print doesn't bomb out.

        class Bad(om.ExplicitComponent):

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

        top = om.Problem()
        top.model.add_subsystem('px', om.IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.add_subsystem('par', Bad())
        top.model.connect('px.x', 'comp.x')
        top.model.connect('comp.z', 'par.x')

        top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 3

        top.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement='vector')
        top.set_solver_print(level=0)

        top.setup()

        # Make sure we don't raise an error when we reach the final debug print.
        top.run_model()

    def test_undeclared_options(self):
        # Test that using options that should not exist in class, cause an
        # error if they are set when instantiating BoundsEnforceLS.
        # atol, rtol, maxiter, and err_on_non_converge are not used in BoundsEnforceLS

        with self.assertRaises(KeyError) as context:
            om.BoundsEnforceLS(bound_enforcement='scalar', atol=1.0)

        self.assertEqual(str(context.exception), "\"BoundsEnforceLS: Option 'atol' cannot be set because it "
                                                 "has not been declared.\"")

        with self.assertRaises(KeyError) as context:
            om.BoundsEnforceLS(bound_enforcement='scalar', rtol=2.0)

        self.assertEqual(str(context.exception), "\"BoundsEnforceLS: Option 'rtol' cannot be set because it "
                                                 "has not been declared.\"")

        with self.assertRaises(KeyError) as context:
            om.BoundsEnforceLS(bound_enforcement='scalar', maxiter=1)

        self.assertEqual(str(context.exception), "\"BoundsEnforceLS: Option 'maxiter' cannot be set because it "
                                                 "has not been declared.\"")

        with self.assertRaises(KeyError) as context:
            om.BoundsEnforceLS(bound_enforcement='scalar', err_on_non_converge=True)

        self.assertEqual(str(context.exception), "\"BoundsEnforceLS: Option 'err_on_non_converge' cannot be set because it "
                                                 "has not been declared.\"")


class SellarDis1withDerivativesMod(SellarDis1):
    # Version of Sellar discipline 1 with a slightly incorrect x derivative.
    # This will still solve, but will require some backtracking at times.

    def _do_declares(self):
        self.declare_partials(of='*', wrt='*')

    def compute_partials(self, inputs, partials):
        partials['y1', 'y2'] = -0.2
        partials['y1', 'z'] = np.array([[2.0 * inputs['z'][0], 1.0]])
        partials['y1', 'x'] = 1.5


class SubSellarMod(om.Group):

    def __init__(self, units=None, scaling=None, **kwargs):
        super(SubSellarMod, self).__init__(**kwargs)

        self.add_subsystem('d1', SellarDis1withDerivativesMod(units=units, scaling=scaling),
                           promotes=['x', 'z', 'y1', 'y2'])
        self.add_subsystem('d2', SellarDis2withDerivatives(units=units, scaling=scaling),
                           promotes=['z', 'y1', 'y2'])


class DoubleSellarMod(om.Group):

    def __init__(self, units=None, scaling=None, **kwargs):
        super(DoubleSellarMod, self).__init__(**kwargs)

        self.add_subsystem('g1', SubSellarMod(units=units, scaling=scaling))
        self.add_subsystem('g2', SubSellarMod(units=units, scaling=scaling))

        self.connect('g1.y2', 'g2.x')
        self.connect('g2.y2', 'g1.x')

        # Converge the outer loop with Gauss Seidel, with a looser tolerance.
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        self.linear_solver = om.DirectSolver()


class TestArmijoGoldsteinLSArrayBounds(unittest.TestCase):

    def setUp(self):
        top = om.Problem()
        top.model.add_subsystem('px', om.IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = om.ScipyKrylov()

        top.set_solver_print(level=0)
        top.setup()

        self.top = top
        self.ub = np.array([2.6, 2.5, 2.65])

    def test_linesearch_vector_bound_enforcement(self):
        top = self.top

        ls = top.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='vector')
        ls.options['c'] = .1

        # Setup again because we assigned a new linesearch
        top.setup()

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()
        for ind in range(3):
            assert_near_equal(top['comp.z'][ind], [1.5], 1e-8)

        # Test upper bounds: should go to the minimum upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()
        for ind in range(3):
            assert_near_equal(top['comp.z'][ind], [2.5], 1e-8)

    def test_linesearch_wall_bound_enforcement_wall(self):
        top = self.top

        top.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='wall')

        # Setup again because we assigned a new linesearch
        top.setup()

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()
        for ind in range(3):
            assert_near_equal(top['comp.z'][ind], [1.5], 1e-8)

        # Test upper bounds: should go to the upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.
        top['comp.z'] = 2.4
        top.run_model()
        for ind in range(3):
            assert_near_equal(top['comp.z'][ind], [self.ub[ind]], 1e-8)

    def test_linesearch_wall_bound_enforcement_scalar(self):
        top = self.top

        top.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='scalar')

        # Setup again because we assigned a new linesearch
        top.setup()

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

        prob = om.Problem()
        model = prob.model = DoubleSellarMod()

        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g1.nonlinear_solver.options['rtol'] = 1.0e-5
        g1.linear_solver = om.DirectSolver()

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        g2.nonlinear_solver.options['rtol'] = 1.0e-5
        g2.linear_solver = om.DirectSolver()

        model.nonlinear_solver = om.NewtonSolver()
        model.linear_solver = om.ScipyKrylov()

        model.nonlinear_solver.options['solve_subsystems'] = True
        model.nonlinear_solver.options['max_sub_solves'] = 4
        ls = model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='vector')

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['g1.y1'], 0.64, .00001)
        assert_near_equal(prob['g1.y2'], 0.80, .00001)
        assert_near_equal(prob['g2.y1'], 0.64, .00001)
        assert_near_equal(prob['g2.y2'], 0.80, .00001)


class CompAtan(om.ImplicitComponent):
    """
    A simple implicit component with the following equation:

    F(x, y) = 33.0 * atan(y-20)**2 + x

    x is an input, y is the state to be solved.
    for x = -100, y should be 19.68734033

    This equation poses a challenge because a guess that is far from the solution yields large
    gradients and divergence. Additionally, the jacobian becomes singular at y = 20. To address
    this, a lower and upper bound are added on y so that a solver with a BoundsEnforceLS does not
    allow it to stray into problematic regions.
    """

    def setup(self):
        self.add_input('x', 1.0)
        self.add_output('y', 1.0, lower=1.0, upper=19.9)

        self.declare_partials(of='y', wrt='x')
        self.declare_partials(of='y', wrt='y')

    def apply_nonlinear(self, inputs, outputs, residuals):
        x = inputs['x']
        y = outputs['y']

        residuals['y'] = (33.0 * atan(y-20.0))**2 + x

    def linearize(self, inputs, outputs, jacobian):
        x = inputs['x']
        y = outputs['y']

        jacobian['y', 'y'] = 2178.0*atan(y-20.0) / (y**2 - 40.0*y + 401.0)
        jacobian['y', 'x'] = 1.0


class TestFeatureLineSearch(unittest.TestCase):

    def test_feature_specification(self):
        import openmdao.api as om
        from openmdao.solvers.linesearch.tests.test_backtracking import CompAtan

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', CompAtan(), promotes_inputs=['x'])

        prob.setup()

        prob.set_val('x', -100.0)

        # Initial value for the state:
        prob.set_val('comp.y', 12.0)

        # You can change the om.NewtonSolver settings after setup is called
        newton = prob.model.nonlinear_solver = om.NewtonSolver()
        prob.model.linear_solver = om.DirectSolver()
        newton.options['iprint'] = 2
        newton.options['rtol'] = 1e-8
        newton.options['solve_subsystems'] = True

        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options['iprint'] = 2

        prob.run_model()

        assert_near_equal(prob.get_val('comp.y'), 19.68734033, 1e-6)

    def test_feature_boundsenforcels_basic(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = om.Problem()
        top.model.add_subsystem('px', om.IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = om.ScipyKrylov()

        top.model.nonlinear_solver.linesearch = om.BoundsEnforceLS()

        top.setup()

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()

        for ind in range(3):
            assert_near_equal(top['comp.z'][ind], [1.5], 1e-8)

    def test_feature_armijogoldsteinls_basic(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = om.Problem()
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays(), promotes_inputs=['x'])

        top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = om.ScipyKrylov()

        top.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

        top.setup()
        top.set_val('x', np.array([2., 2, 2]).reshape(3, 1))
        # Test lower bounds: should go to the lower bound and stall
        top.set_val('comp.y', 0.)
        top.set_val('comp.z', 1.6)
        top.run_model()

        for ind in range(3):
            assert_near_equal(top.get_val('comp.z', indices=ind), [1.5], 1e-8)

    def test_feature_boundscheck_basic(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = om.Problem()
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays(), promotes_inputs=['x'])

        top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = om.ScipyKrylov()

        top.model.nonlinear_solver.linesearch = om.BoundsEnforceLS()

        top.setup()
        top.set_val('x', np.array([2., 2, 2]).reshape(3, 1))

        # Test lower bounds: should go to the lower bound and stall
        top.set_val('comp.y', 0.)
        top.set_val('comp.z', 1.6)
        top.run_model()

        for ind in range(3):
            assert_near_equal(top.get_val('comp.z', indices=ind), [1.5], 1e-8)

    def test_feature_boundscheck_vector(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = om.Problem()
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays(), promotes_inputs=['x'])

        top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = om.ScipyKrylov()

        top.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement='vector')

        top.setup()
        top.set_val('x', np.array([2., 2, 2]).reshape(3, 1))

        # Test lower bounds: should go to the lower bound and stall
        top.set_val('comp.y', 0.)
        top.set_val('comp.z', 1.6)
        top.run_model()

        for ind in range(3):
            assert_near_equal(top.get_val('comp.z', indices=ind), [1.5], 1e-8)

    def test_feature_boundscheck_wall(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = om.Problem()
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays(), promotes_inputs=['x'])

        top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = om.ScipyKrylov()

        top.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement='wall')

        top.setup()
        top.set_val('x', np.array([0.5, 0.5, 0.5]).reshape(3, 1))

        # Test upper bounds: should go to the upper bound and stall
        top.set_val('comp.y', 0.)
        top.set_val('comp.z', 2.4)
        top.run_model()

        assert_near_equal(top.get_val('comp.z', indices=0), [2.6], 1e-8)
        assert_near_equal(top.get_val('comp.z', indices=1), [2.5], 1e-8)
        assert_near_equal(top.get_val('comp.z', indices=2), [2.65], 1e-8)

    def test_feature_boundscheck_scalar(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = om.Problem()
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays(), promotes_inputs=['x'])

        top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = om.ScipyKrylov()

        top.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(bound_enforcement='scalar')

        top.setup()
        top.set_val('x', np.array([2., 2, 2]).reshape(3, 1))
        top.run_model()

        # Test lower bounds: should stop just short of the lower bound
        top.set_val('comp.y', 0.)
        top.set_val('comp.z', 1.6)
        top.run_model()

    def test_feature_print_bound_enforce(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = om.Problem()
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays(), promotes_inputs=['x'])

        newt = top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 2
        top.model.linear_solver = om.ScipyKrylov()

        ls = newt.linesearch = om.BoundsEnforceLS(bound_enforcement='vector')
        ls.options['print_bound_enforce'] = True

        top.set_solver_print(level=2)


        top.setup()
        top.set_val('x', np.array([2., 2, 2]).reshape(3, 1))

        # Test lower bounds: should go to the lower bound and stall
        top.set_val('comp.y', 0.)
        top.set_val('comp.z', 1.6)
        top.run_model()

        for ind in range(3):
            assert_near_equal(top.get_val('comp.z', indices=ind), [1.5], 1e-8)

    def test_feature_armijo_boundscheck_vector(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = om.Problem()
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays(), promotes_inputs=['x'])

        top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = om.ScipyKrylov()

        top.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='vector')

        top.setup()

        top.set_val('x', np.array([2., 2, 2]).reshape(3, 1))

        # Test lower bounds: should go to the lower bound and stall
        top.set_val('comp.y', 0.)
        top.set_val('comp.z', 1.6)
        top.run_model()

        for ind in range(3):
            assert_near_equal(top.get_val('comp.z', indices=ind), [1.5], 1e-8)

    def test_feature_armijo_boundscheck_wall(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = om.Problem()
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays(), promotes_inputs=['x'])

        top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = om.ScipyKrylov()

        top.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='wall')

        top.setup()

        top.set_val('x', np.array([0.5, 0.5, 0.5]).reshape(3, 1))

        # Test upper bounds: should go to the upper bound and stall
        top.set_val('comp.y', 0.)
        top.set_val('comp.z', 2.4)
        top.run_model()

        assert_near_equal(top.get_val('comp.z', indices=0), [2.6], 1e-8)
        assert_near_equal(top.get_val('comp.z', indices=1), [2.5], 1e-8)
        assert_near_equal(top.get_val('comp.z', indices=2), [2.65], 1e-8)

    def test_feature_armijo_boundscheck_scalar(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = om.Problem()
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays(), promotes_inputs=['x'])

        top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = om.ScipyKrylov()

        ls = top.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='scalar')

        top.setup()
        top.set_val('x', np.array([2., 2, 2]).reshape(3, 1))
        top.run_model()

        # Test lower bounds: should stop just short of the lower bound
        top.set_val('comp.y', 0.)
        top.set_val('comp.z', 1.6)
        top.run_model()

    def test_feature_armijo_print_bound_enforce(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = om.Problem()
        top.model.add_subsystem('px', om.IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        newt = top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 2
        top.model.linear_solver = om.ScipyKrylov()

        ls = newt.linesearch = om.ArmijoGoldsteinLS()
        ls.options['print_bound_enforce'] = True

        top.set_solver_print(level=2)
        top.setup()

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()

        for ind in range(3):
            assert_near_equal(top['comp.z'][ind], [1.5], 1e-8)

    def test_feature_goldstein(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStatesArrays

        top = om.Problem()
        top.model.add_subsystem('px', om.IndepVarComp('x', np.ones((3, 1))))
        top.model.add_subsystem('comp', ImplCompTwoStatesArrays())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = om.ScipyKrylov()

        ls = top.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='vector')
        ls.options['method'] = 'Goldstein'

        top.setup()

        # Test lower bounds: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.
        top['comp.z'] = 1.6
        top.run_model()

        for ind in range(3):
            assert_near_equal(top['comp.z'][ind], [1.5], 1e-8)


if __name__ == "__main__":
    unittest.main()
