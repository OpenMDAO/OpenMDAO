""" Unit test for the Newton nonlinear solver. """

import unittest
from six import iteritems

import numpy as np

from openmdao.api import Group, Problem, IndepVarComp, LinearBlockGS, \
    NewtonSolver, ExecComp, ScipyIterativeSolver, ImplicitComponent
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped, \
     SellarNoDerivatives, SellarDerivatives, \
     SellarStateConnection


class TestNewton(unittest.TestCase):

    def test_sellar_grouped(self):
        # Tests basic Newton solution on Sellar in a subgroup

        prob = Problem()
        prob.model = SellarDerivativesGrouped()
        mda = [s for s in prob.model._subsystems_allprocs if s.name == 'mda'][0]
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

        prob.setup(check=False)
        prob.model.suppress_solver_output = True
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
        prob.model.ln_solver = LinearBlockGS()
        prob.model.ln_solver.options['maxiter'] = 2

        prob.setup(check=False)
        prob.model.suppress_solver_output = True
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        ## Make sure we aren't iterating like crazy
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

        ## Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nl_solver._iter_count, 8)

    def test_sellar_state_connection_fd_system(self):
        # Sellar model closes loop with state connection instead of a cycle.
        # This test is just fd.

        raise unittest.SkipTest("FD not implemented yet")

        prob = Problem()
        prob.model = SellarStateConnection()
        prob.model.nl_solver = NewtonSolver()

        # TODO - Specify FD for group.
        #prob.model.deriv_options['type'] = 'fd'

        prob.setup(check=False)
        prob.model.suppress_solver_output = True
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['state_eq.y2_command'], 12.05848819, .00001)

        ## Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nl_solver._iter_count, 6)

    def test_sellar_specify_linear_solver(self):

        raise unittest.SkipTest("BUG: cannot specify subsolver")

        prob = Problem()
        prob.model = SellarStateConnection()
        prob.model.nl_solver = NewtonSolver()

        # Use bad settings for this one so that problem doesn't converge.
        # That way, we test that we are really using Newton's Lin Solver
        # instead.
        prob.model.ln_solver = ScipyIterativeSolver()
        prob.model.ln_solver.options['maxiter'] = 1

        # The good solver
        prob.model.nl_solver.options['subsolvers']['linear'] = ScipyIterativeSolver()

        prob.model.suppress_solver_output = True
        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['state_eq.y2_command'], 12.05848819, .00001)

        ## Make sure we aren't iterating like crazy
        self.assertLess(prob.model.nl_solver._iter_count, 8)
        self.assertEqual(prob.model.ln_solver._iter_count, 0)
        self.assertGreater(prob.model.nl_solver.options['subsolvers']['linear']._iter_count, 0)

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

            def compute(self, params, unknowns):
                pass

            def apply_nonlinear(self, params, unknowns, resids):
                """ Don't solve; just calculate the residual."""

                x = params['x']
                z = unknowns['z']

                resids['z'] = (z**3 + 3.0*z**2 - 6.0*z + x)*1e15

            def linearize(self, params, unknowns, J):
                """Analytical derivatives."""

                x = params['x']
                z = unknowns['z']

                # State equation
                J[('z', 'z')] = (3.0*z**2 + 6.0*z - 6.0)*1e15
                J[('z', 'x')] = 1.0*1e15

        prob = Problem()
        root = prob.model = Group()
        root.add_subsystem('comp', CubicImplicit())
        root.add_subsystem('p1', IndepVarComp('x', 17.4))
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


if __name__ == "__main__":
    unittest.main()
