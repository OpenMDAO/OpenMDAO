"""Test the ScipyIterativeSolver linear solver class."""

from __future__ import division, print_function

import unittest

from openmdao.core.group import Group
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.problem import Problem
from openmdao.devtools.testutil import assert_rel_error
from openmdao.solvers.ln_bgs import LinearBlockGS
from openmdao.solvers.ln_scipy import ScipyIterativeSolver, gmres
from openmdao.solvers.nl_newton import NewtonSolver
from openmdao.solvers.tests.linear_test_base import LinearSolverTests
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimpleDense
from openmdao.test_suite.components.sellar import SellarDerivatives
from openmdao.test_suite.groups.implicit_group import TestImplicitGroup


class TestScipyIterativeSolver(LinearSolverTests.LinearSolverTestCase):

    ln_solver_class = ScipyIterativeSolver

    def test_options(self):
        """Verify that the SciPy solver specific options are declared."""

        group = Group()
        group.ln_solver = ScipyIterativeSolver()

        assert(group.ln_solver.options['solver'] == gmres)

    def test_solve_linear_scipy(self):
        """Solve implicit system with ScipyIterativeSolver."""

        group = TestImplicitGroup(lnSolverClass=ScipyIterativeSolver)

        p = Problem(group)
        p.setup(check=False)
        p.set_solver_print(level=0)

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        group.run_solve_linear(['linear'], 'fwd')
        output = d_outputs._data
        assert_rel_error(self, output[1], group.expected_solution[0], 1e-15)
        assert_rel_error(self, output[5], group.expected_solution[1], 1e-15)

        # reverse
        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        group.run_solve_linear(['linear'], 'rev')
        output = d_residuals._data
        assert_rel_error(self, output[1], group.expected_solution[0], 1e-15)
        assert_rel_error(self, output[5], group.expected_solution[1], 1e-15)

    def test_solve_linear_scipy_maxiter(self):
        """Verify that ScipyIterativeSolver abides by the 'maxiter' option."""

        group = TestImplicitGroup(lnSolverClass=ScipyIterativeSolver)
        group.ln_solver.options['maxiter'] = 2

        p = Problem(group)
        p.setup(check=False)
        p.set_solver_print(level=0)

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        group.run_solve_linear(['linear'], 'fwd')

        self.assertTrue(group.ln_solver._iter_count == 2)

        # reverse
        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        group.run_solve_linear(['linear'], 'rev')

        self.assertTrue(group.ln_solver._iter_count == 2)

    def test_solve_on_subsystem(self):
        """solve an implicit system with GMRES attached to a subsystem"""

        p = Problem()
        model = p.model = Group()
        dv = model.add_subsystem('des_vars', IndepVarComp())
        # just need a dummy variable so the sizes don't match between root and g1
        dv.add_output('dummy', val=1.0, shape=10)

        g1 = model.add_subsystem('g1', TestImplicitGroup(lnSolverClass=ScipyIterativeSolver))

        p.model.ln_solver.options['maxiter'] = 1
        p.setup(check=False)

        p.set_solver_print(level=0)

        # forward
        d_inputs, d_outputs, d_residuals = g1.get_linear_vectors()

        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        g1._solve_linear(['linear'], 'fwd')

        output = d_outputs._data
        # The empty first entry in _data is due to the dummy
        #     variable being in a different variable set not owned by g1
        assert_rel_error(self, output[1], g1.expected_solution[0], 1e-15)
        assert_rel_error(self, output[5], g1.expected_solution[1], 1e-15)

        # reverse
        d_inputs, d_outputs, d_residuals = g1.get_linear_vectors()

        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        g1.ln_solver._linearize()
        g1._solve_linear(['linear'], 'rev')

        output = d_residuals._data
        assert_rel_error(self, output[1], g1.expected_solution[0], 3e-15)
        assert_rel_error(self, output[5], g1.expected_solution[1], 3e-15)


class TestScipyIterativeSolverFeature(unittest.TestCase):

    def test_feature_simple(self):
        """Tests feature for adding a Scipy GMRES solver and calculating the
        derivatives."""
        # Tests derivatives on a simple comp that defines compute_jacvec.
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('x_param', IndepVarComp('length', 3.0),
                            promotes=['length'])
        model.add_subsystem('mycomp', TestExplCompSimpleDense(),
                            promotes=['length', 'width', 'area'])

        model.ln_solver = ScipyIterativeSolver()
        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='fwd')
        prob['width'] = 2.0
        prob.run_model()

        of = ['area']
        wrt = ['length']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['area', 'length'][0][0], 2.0, 1e-6)

    def test_specify_solver(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = ScipyIterativeSolver()

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)

    def test_feature_maxiter(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = ScipyIterativeSolver()
        model.ln_solver.options['maxiter'] = 3

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 0.0, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 0.0, .00001)

    def test_feature_atol(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = ScipyIterativeSolver()
        model.ln_solver.options['atol'] = 1.0e-20

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001055699, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448533563, .00001)

    def test_specify_precon(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nl_solver = NewtonSolver()
        prob.model.ln_sollver = ScipyIterativeSolver()

        prob.model.ln_solver.precon = LinearBlockGS()
        prob.model.ln_solver.precon.options['maxiter'] = 2

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

if __name__ == "__main__":
    unittest.main()
