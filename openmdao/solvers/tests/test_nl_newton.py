"""Test the Newton nonlinear solver. """

import unittest

import numpy as np

from openmdao.api import Group, Problem, IndepVarComp, LinearBlockGS, \
    NewtonSolver, ExecComp, ScipyIterativeSolver, ImplicitComponent, \
    DirectSolver, DenseJacobian
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.double_sellar import DoubleSellar
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped, \
     SellarNoDerivatives, SellarDerivatives, \
     SellarStateConnection


class TestNewton(unittest.TestCase):

    def test_specify_newton_ln_solver_in_system(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.nl_solver = NewtonSolver()
        # used for analytic derivatives
        model.ln_solver = DirectSolver()

        prob.setup()

        self.assertIsInstance(model.nl_solver.ln_solver, DirectSolver)

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_feature_newton_basic(self):
        """ Feature test for slotting a Newton solver and using it to solve
        Sellar.
        """
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nl_solver = NewtonSolver()

        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_sellar_grouped(self):
        # Tests basic Newton solution on Sellar in a subgroup

        prob = Problem()
        prob.model = SellarDerivativesGrouped()
        mda = prob.model.get_subsystem('mda')
        mda.nl_solver = NewtonSolver()

        prob.setup(check=False)
        prob.model.suppress_solver_output = True
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nl_solver._iter_count, 8)

    def test_sellar(self):
        # Just tests Newton on Sellar with FD derivs.

        raise unittest.SkipTest("FD not implemented yet")

        prob = Problem()
        prob.model = SellarNoDerivatives()
        prob.model.nl_solver = NewtonSolver()

        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nl_solver.iter_count, 8)

    def test_sellar_analysis_error(self):
        # Make sure analysis error is raised.

        raise unittest.SkipTest("AnalysisError not implemented yet")

        prob = Problem()
        prob.model = SellarNoDerivatives()
        prob.model.nl_solver = Newton()
        prob.model.nl_solver.options['err_on_maxiter'] = True
        prob.model.nl_solver.options['maxiter'] = 2

        prob.setup(check=False)

        try:
            prob.run_model()
        except AnalysisError as err:
            self.assertEqual(str(err), "Solve in '': Newton FAILED to converge after 2 iterations")
        else:
            self.fail("expected AnalysisError")

    def test_sellar_derivs(self):
        # Test top level Sellar (i.e., not grouped).
        # Also, piggybacked testing that makes sure we only call apply_nonlinear
        # on the head component behind the cycle break.

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nl_solver = NewtonSolver()
        prob.model.ln_solver = LinearBlockGS()

        prob.setup(check=False)
        prob.model.suppress_solver_output = False
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nl_solver._iter_count, 8)

        ## Make sure we only call apply_linear on 'heads'
        #nd1 = prob.model.d1.execution_count
        #nd2 = prob.model.d2.execution_count
        #if prob.model.d1._run_apply == True:
            #self.assertEqual(nd1, 2*nd2)
        #else:
            #self.assertEqual(2*nd1, nd2)

    def test_sellar_derivs_with_Lin_GS(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nl_solver = NewtonSolver()

        prob.setup(check=False)
        prob.model.suppress_solver_output = True
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nl_solver._iter_count, 8)

    def test_sellar_state_connection(self):
        # Sellar model closes loop with state connection instead of a cycle.

        prob = Problem()
        prob.model = SellarStateConnection()
        prob.model.nl_solver = NewtonSolver()
        prob.model.suppress_solver_output = True
        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['state_eq.y2_command'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nl_solver._iter_count, 8)

    def test_sellar_state_connection_fd_system(self):
        # Sellar model closes loop with state connection instead of a cycle.
        # This test is just fd.

        raise unittest.SkipTest("FD not implemented yet")

        prob = Problem()
        prob.model = SellarStateConnection()
        prob.model.nl_solver = NewtonSolver()

        # TODO - Specify FD for group.
        # prob.model.deriv_options['type'] = 'fd'

        prob.setup(check=False)
        prob.model.suppress_solver_output = True
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['state_eq.y2_command'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nl_solver._iter_count, 6)

    def test_sellar_specify_linear_solver(self):

        prob = Problem()
        prob.model = SellarStateConnection()
        prob.model.nl_solver = NewtonSolver()

        # Use bad settings for this one so that problem doesn't converge.
        # That way, we test that we are really using Newton's Lin Solver
        # instead.
        prob.model.ln_solver = ScipyIterativeSolver()
        prob.model.ln_solver.options['maxiter'] = 1

        # The good solver
        prob.model.nl_solver.ln_solver = ScipyIterativeSolver()

        prob.model.suppress_solver_output = True
        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['state_eq.y2_command'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nl_solver._iter_count, 8)
        self.assertEqual(prob.model.ln_solver._iter_count, 0)
        self.assertGreater(prob.model.nl_solver.ln_solver._iter_count, 0)

    def test_sellar_specify_linear_direct_solver(self):

        prob = Problem()
        prob.model = SellarStateConnection()
        prob.model.nl_solver = NewtonSolver()

        # Use bad settings for this one so that problem doesn't converge.
        # That way, we test that we are really using Newton's Lin Solver
        # instead.
        prob.model.ln_solver = ScipyIterativeSolver()
        prob.model.ln_solver.options['maxiter'] = 1

        # The good solver
        prob.model.nl_solver.ln_solver = DirectSolver()

        prob.model.suppress_solver_output = True
        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['state_eq.y2_command'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nl_solver._iter_count, 8)
        self.assertEqual(prob.model.ln_solver._iter_count, 0)

    def test_implicit_utol(self):
        # We are setup for reach utol termination condition quite quickly.

        raise unittest.SkipTest("solver utol not implemented yet")

        class CubicImplicit(ImplicitComponent):
            """ A Simple Implicit Component.
            f(x) = x**3 + 3x**2 -6x +18
            """

            def __init__(self):
                super(CubicImplicit, self).__init__()

                # Params
                self.add_input('x', 0.0)

                # States
                self.add_output('z', 0.0)

            def compute(self, inputs, outputs):
                pass

            def apply_nonlinear(self, inputs, outputs, resids):
                """ Don't solve; just calculate the residual."""

                x = inputs['x']
                z = outputs['z']

                resids['z'] = (z**3 + 3.0*z**2 - 6.0*z + x)*1e15

            def linearize(self, inputs, outputs, partials):
                """Analytical derivatives."""

                # x = inputs['x']
                z = outputs['z']

                # State equation
                partials[('z', 'z')] = (3.0*z**2 + 6.0*z - 6.0)*1e15
                partials[('z', 'x')] = 1.0*1e15

        prob = Problem()
        root = prob.model = Group()
        root.add_subsystem('p1', IndepVarComp('x', 17.4))
        root.add_subsystem('comp', CubicImplicit())
        root.connect('p1.x', 'comp.x')

        prob.model.nl_solver = NewtonSolver()
        prob.model.ln_solver = ScipyIterativeSolver()

        prob.setup(check=False)
        prob['comp.z'] = -4.93191510182

        prob.model.suppress_solver_output = True
        prob.run_model()

        assert_rel_error(self, prob['comp.z'], -4.93191510182, .00001)
        self.assertLessEqual(prob.model.nl_solver._iter_count, 4,
                             msg='Should get there pretty quick because of utol.')

    def test_solve_subsystems_basic(self):
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

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['g1.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g1.y2'], 0.80, .00001)
        assert_rel_error(self, prob['g2.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g2.y2'], 0.80, .00001)

    def test_solve_subsystems_assembled_jac_top(self):
        prob = Problem()
        model = prob.model = DoubleSellar()
        model.jacobian = DenseJacobian()

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

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['g1.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g1.y2'], 0.80, .00001)
        assert_rel_error(self, prob['g2.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g2.y2'], 0.80, .00001)

    def test_solve_subsystems_assembled_jac_subgroup(self):
        prob = Problem()
        model = prob.model = DoubleSellar()

        g1 = model.get_subsystem('g1')
        g1.nl_solver = NewtonSolver()
        g1.nl_solver.options['rtol'] = 1.0e-5
        g1.ln_solver = DirectSolver()
        g1.jacobian = DenseJacobian()

        g2 = model.get_subsystem('g2')
        g2.nl_solver = NewtonSolver()
        g2.nl_solver.options['rtol'] = 1.0e-5
        g2.ln_solver = DirectSolver()

        model.nl_solver = NewtonSolver()
        model.ln_solver = ScipyIterativeSolver()

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['g1.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g1.y2'], 0.80, .00001)
        assert_rel_error(self, prob['g2.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g2.y2'], 0.80, .00001)

    def test_solve_subsystems_internals(self):
        # Here we test that this feature is doing what it should do by counting the
        # number of calls in various places.

        class CountNewton(NewtonSolver):
            """ This version of Newton also counts how many times it runs in total."""

            def __init__(self, **kwargs):
                super(CountNewton, self).__init__(**kwargs)
                self.total_count = 0

            def _iter_execute(self):
                super(CountNewton, self)._iter_execute()
                self.total_count += 1

        class CountDS(DirectSolver):
            """ This version of Newton also counts how many times it linearizes"""

            def __init__(self, **kwargs):
                super(DirectSolver, self).__init__(**kwargs)
                self.lin_count = 0

            def _linearize(self):
                super(CountDS, self)._linearize()
                self.lin_count += 1

        prob = Problem()
        model = prob.model = DoubleSellar()

        # each SubSellar group converges itself
        g1 = model.get_subsystem('g1')
        g1.nl_solver = CountNewton()
        g1.nl_solver.options['rtol'] = 1.0e-5
        g1.ln_solver = CountDS()  # used for derivatives

        g2 = model.get_subsystem('g2')
        g2.nl_solver = CountNewton()
        g2.nl_solver.options['rtol'] = 1.0e-5
        g2.ln_solver = DirectSolver()

        # Converge the outer loop with Gauss Seidel, with a looser tolerance.
        model.nl_solver = NewtonSolver()
        model.ln_solver = ScipyIterativeSolver()

        # Enfore behavior: max_sub_solves = 0 means we run once during init

        model.nl_solver.options['maxiter'] = 5
        model.nl_solver.options['solve_subsystems'] = True
        model.nl_solver.options['max_sub_solves'] = 0
        prob.model.suppress_solver_output = True

        prob.setup()
        prob.run_model()

        # Verifying subsolvers ran
        self.assertEqual(g1.nl_solver.total_count, 2)
        self.assertEqual(g2.nl_solver.total_count, 2)
        self.assertEqual(g1.ln_solver.lin_count, 3)

        prob = Problem()
        model = prob.model = DoubleSellar()

        # each SubSellar group converges itself
        g1 = model.get_subsystem('g1')
        g1.nl_solver = CountNewton()
        g1.nl_solver.options['rtol'] = 1.0e-5
        g1.ln_solver = CountDS()  # used for derivatives

        g2 = model.get_subsystem('g2')
        g2.nl_solver = CountNewton()
        g2.nl_solver.options['rtol'] = 1.0e-5
        g2.ln_solver = DirectSolver()

        # Converge the outer loop with Gauss Seidel, with a looser tolerance.
        model.nl_solver = NewtonSolver()
        model.ln_solver = ScipyIterativeSolver()

        # Enforce Behavior: baseline

        model.nl_solver.options['maxiter'] = 5
        model.nl_solver.options['solve_subsystems'] = True
        model.nl_solver.options['max_sub_solves'] = 5
        prob.model.suppress_solver_output = True

        prob.setup()
        prob.run_model()

        # Verifying subsolvers ran
        self.assertEqual(g1.nl_solver.total_count, 5)
        self.assertEqual(g2.nl_solver.total_count, 5)
        self.assertEqual(g1.ln_solver.lin_count, 8)

        prob = Problem()
        model = prob.model = DoubleSellar()

        # each SubSellar group converges itself
        g1 = model.get_subsystem('g1')
        g1.nl_solver = CountNewton()
        g1.nl_solver.options['rtol'] = 1.0e-5
        g1.ln_solver = CountDS()  # used for derivatives

        g2 = model.get_subsystem('g2')
        g2.nl_solver = CountNewton()
        g2.nl_solver.options['rtol'] = 1.0e-5
        g2.ln_solver = DirectSolver()

        # Converge the outer loop with Gauss Seidel, with a looser tolerance.
        model.nl_solver = NewtonSolver()
        model.ln_solver = ScipyIterativeSolver()

        # Enfore behavior: max_sub_solves = 1 means we run during init and first iteration of iter_execute

        model.nl_solver.options['maxiter'] = 5
        model.nl_solver.options['solve_subsystems'] = True
        model.nl_solver.options['max_sub_solves'] = 1
        prob.model.suppress_solver_output = True

        prob.setup()
        prob.run_model()

        # Verifying subsolvers ran
        self.assertEqual(g1.nl_solver.total_count, 4)
        self.assertEqual(g2.nl_solver.total_count, 4)
        self.assertEqual(g1.ln_solver.lin_count, 6)

    def test_maxiter_one(self):
        # Fix bug when maxiter was set to 1.
        # This bug caused linearize to run before apply in this case.

        class ImpComp(ImplicitComponent):

            def initialize_variables(self):
                self.add_input('a', val=1.)
                self.add_output('x', val=0.)
                self.applied = False

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['x'] = np.exp(outputs['x']) - \
                    inputs['a']**2 * outputs['x']**2
                self.applied = True

            def solve_nonlinear(self, inputs, outputs):
                pass

            def linearize(self, inputs, outputs, jacobian):
                jacobian['x', 'x'] = np.exp(outputs['x']) - \
                    2 * inputs['a']**2 * outputs['x']
                jacobian['x', 'a'] = -2 * inputs['a'] * outputs['x']**2

                if not self.applied:
                    raise RuntimeError("Bug! Linearize called before Apply!")

        prob = Problem()
        root = prob.model = Group()
        root.add_subsystem('p1', IndepVarComp('a', 1.0))
        root.add_subsystem('comp', ImpComp())
        root.connect('p1.a', 'comp.a')

        root.nl_solver = NewtonSolver()
        root.nl_solver.options['maxiter'] = 1
        root.suppress_solver_output = True

        prob.setup()
        prob.run_model()


class TestNewtonFeatures(unittest.TestCase):

    def test_feature_basic(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        nlgbs = prob.model.nl_solver = NewtonSolver()

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_feature_maxiter(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        nlgbs = prob.model.nl_solver = NewtonSolver()

        nlgbs.options['maxiter'] = 2

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.5878516779, .00001)
        assert_rel_error(self, prob['y2'], 12.0607416105, .00001)

    def test_feature_rtol(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        nlgbs = prob.model.nl_solver = NewtonSolver()

        nlgbs.options['rtol'] = 1e-3

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.5878516779, .00001)
        assert_rel_error(self, prob['y2'], 12.0607416105, .00001)

    def test_feature_atol(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        nlgbs = prob.model.nl_solver = NewtonSolver()

        nlgbs.options['atol'] = 1e-4

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.5882856302, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_feature_ln_solver(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        nlgbs = prob.model.nl_solver = NewtonSolver()
        prob.ln_solver = DirectSolver()

        prob.setup()

        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_feature_max_sub_solves(self):
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
        model.nl_solver.options['max_sub_solves'] = 0

        prob.setup()
        prob.run_model()

if __name__ == "__main__":
    unittest.main()
