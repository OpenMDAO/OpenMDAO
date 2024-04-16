"""Test the ScipyKrylov linear solver class."""

import unittest

import numpy as np
import scipy

from packaging.version import Version

import openmdao.api as om
from openmdao.solvers.linear.tests.linear_test_base import LinearSolverTests
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimpleDense
from openmdao.test_suite.components.misc_components import Comp4LinearCacheTest
from openmdao.test_suite.components.quad_implicit import QuadraticComp
from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives
from openmdao.test_suite.groups.implicit_group import TestImplicitGroup
from openmdao.utils.assert_utils import assert_near_equal, assert_check_totals


# use this to fake out the TestImplicitGroup so it'll use the solver we want.
def krylov_factory(solver):
    def f(junk=None):
        return om.ScipyKrylov(solver=solver)
    return f


class TestScipyKrylov(LinearSolverTests.LinearSolverTestCase):

    linear_solver_name = 'gmres'
    linear_solver_class = krylov_factory('gmres')

    def test_options(self):
        """Verify that the SciPy solver specific options are declared."""

        group = om.Group()
        group.linear_solver = self.linear_solver_class()

        self.assertEqual(group.linear_solver.options['solver'], self.linear_solver_name)

    def test_solve_linear_scipy(self):
        """Solve implicit system with ScipyKrylov."""
        group = TestImplicitGroup(lnSolverClass=lambda : om.ScipyKrylov(solver=self.linear_solver_name))

        p = om.Problem(group)
        p.setup()
        p.set_solver_print(level=0)

        # Conclude setup but don't run model.
        p.final_setup()

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_val(1.0)
        d_outputs.set_val(0.0)
        group.run_solve_linear('fwd')
        output = d_outputs.asarray()
        assert_near_equal(output, group.expected_solution, 1e-15)

        # reverse
        d_outputs.set_val(1.0)
        d_residuals.set_val(0.0)
        group.run_solve_linear('rev')
        output = d_residuals.asarray()
        assert_near_equal(output, group.expected_solution, 1e-15)

    def test_solve_linear_scipy_maxiter(self):
        """Verify that ScipyKrylov abides by the 'maxiter' option."""

        group = TestImplicitGroup(lnSolverClass=self.linear_solver_class)
        group.linear_solver.options['maxiter'] = 2

        p = om.Problem(group)
        p.setup()
        p.set_solver_print(level=0)

        # Conclude setup but don't run model.
        p.final_setup()

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_val(1.0)
        d_outputs.set_val(0.0)
        group.run_solve_linear('fwd')

        self.assertTrue(group.linear_solver._iter_count == 2)

        # reverse
        d_outputs.set_val(1.0)
        d_residuals.set_val(0.0)
        group.run_solve_linear('rev')

        self.assertTrue(group.linear_solver._iter_count == 2)

    def test_solve_on_subsystem(self):
        """solve an implicit system with GMRES attached to a subsystem"""

        p = om.Problem()
        model = p.model
        dv = model.add_subsystem('des_vars', om.IndepVarComp())
        # just need a dummy variable so the sizes don't match between root and g1
        dv.add_output('dummy', val=1.0, shape=10)

        grp = TestImplicitGroup(lnSolverClass=self.linear_solver_class)
        g1 = model.add_subsystem('g1', grp)

        p.setup()

        p.set_solver_print(level=0)

        # Conclude setup but don't run model.
        p.final_setup()

        # forward
        d_inputs, d_outputs, d_residuals = g1.get_linear_vectors()

        d_residuals.set_val(1.0)
        d_outputs.set_val(0.0)
        g1.run_solve_linear('fwd')

        output = d_outputs.asarray()
        assert_near_equal(output, g1.expected_solution, 1e-15)

        # reverse
        d_inputs, d_outputs, d_residuals = g1.get_linear_vectors()

        d_outputs.set_val(1.0)
        d_residuals.set_val(0.0)
        g1.linear_solver._linearize()
        g1.run_solve_linear('rev')

        output = d_residuals.asarray()
        assert_near_equal(output, g1.expected_solution, 3e-15)

    @unittest.skipIf(Version(scipy.__version__) == Version('1.12.0'), "Bug in scipy 1.12.0, see Issue #19948")
    def test_linear_solution_cache_fwd(self):
        # Test derivatives across a converged Sellar model. When caching
        # is performed, the second solve takes less iterations than the
        # first one.

        # Forward mode

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('d1', Comp4LinearCacheTest(), promotes=['x', 'y'])

        model.nonlinear_solver = om.NonlinearBlockGS()
        model.linear_solver = om.ScipyKrylov()

        model.add_design_var('x', cache_linear_solution=True)
        model.add_objective('y', cache_linear_solution=True)

        prob.setup(mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        J = prob.driver._compute_totals(of=['y'], wrt=['x'], return_format='flat_dict')
        icount1 = prob.model.linear_solver._iter_count
        J = prob.driver._compute_totals(of=['y'], wrt=['x'], return_format='flat_dict')
        icount2 = prob.model.linear_solver._iter_count

        # Should take less iterations when starting from previous solution.
        self.assertTrue(icount2 < icount1,
                        f"Second solve ran for {icount2} iterations, which should have been less than "
                        f"the first solve, which ran for {icount1} iterations.")

    def test_linear_solution_cache_rev(self):
        # Test derivatives across a converged Sellar model. When caching
        # is performed, the second solve takes less iterations than the
        # first one.

        # Reverse mode

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('d1', Comp4LinearCacheTest(), promotes=['x', 'y'])

        model.nonlinear_solver = om.NonlinearBlockGS()
        model.linear_solver = om.ScipyKrylov()

        model.add_design_var('x', cache_linear_solution=True)
        model.add_objective('y', cache_linear_solution=True)

        prob.setup(mode='rev')
        prob.set_solver_print(level=0)
        prob.run_model()

        J = prob.driver._compute_totals(of=['y'], wrt=['x'], return_format='flat_dict')
        icount1 = prob.model.linear_solver._iter_count
        J = prob.driver._compute_totals(of=['y'], wrt=['x'], return_format='flat_dict')
        icount2 = prob.model.linear_solver._iter_count

        # Should take less iterations when starting from previous solution.
        self.assertTrue(icount2 < icount1,
                        f"Second solve ran for {icount2} iterations, which should have been less than "
                        f"the first solve, which ran for {icount1} iterations.")


class TestScipyKrylovFeature(unittest.TestCase):

    def test_feature_simple(self):
        """Tests feature for adding a Scipy GMRES solver and calculating the
        derivatives."""

        # Tests derivatives on a simple comp that defines compute_jacvec.
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('x_param', om.IndepVarComp('length', 3.0),
                            promotes=['length'])
        model.add_subsystem('mycomp', TestExplCompSimpleDense(),
                            promotes=['length', 'width', 'area'])

        model.linear_solver = om.ScipyKrylov()
        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='fwd')
        prob['width'] = 2.0
        prob.run_model()

        of = ['area']
        wrt = ['length']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_near_equal(J['area', 'length'][0][0], 2.0, 1e-6)

    def test_specify_solver(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = om.NonlinearBlockGS()

        model.linear_solver = om.ScipyKrylov()

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_near_equal(J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_near_equal(J['obj', 'z'][0][1], 1.78448534, .00001)

    def test_feature_maxiter(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = om.NonlinearBlockGS()

        model.linear_solver = om.ScipyKrylov()
        model.linear_solver.options['maxiter'] = 3
        model.linear_solver.options['err_on_non_converge'] = True

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        wrt = ['z']
        of = ['obj']

        with self.assertRaises(om.AnalysisError) as cm:
            J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')

        msg = "Solver 'LN: SCIPY' on system '' failed to converge in 3 iterations."
        self.assertEqual(str(cm.exception), msg)

    def test_feature_atol(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = om.NonlinearBlockGS()

        model.linear_solver = om.ScipyKrylov()
        model.linear_solver.options['atol'] = 1.0e-20

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_near_equal(J['obj', 'z'][0][0], 9.61001055699, .00001)
        assert_near_equal(J['obj', 'z'][0][1], 1.78448533563, .00001)

    def test_specify_precon(self):

        prob = om.Problem()
        model = prob.model

        sub1 = model.add_subsystem('sub1', om.Group())
        sub1.add_subsystem('q1', QuadraticComp())
        sub1.add_subsystem('z1', om.ExecComp('y = -6.0 + .01 * x'))
        sub2 = model.add_subsystem('sub2', om.Group())
        sub2.add_subsystem('q2', QuadraticComp())
        sub2.add_subsystem('z2', om.ExecComp('y = -6.0 + .01 * x'))

        model.connect('sub1.q1.x', 'sub1.z1.x')
        model.connect('sub1.z1.y', 'sub2.q2.c')
        model.connect('sub2.q2.x', 'sub2.z2.x')
        model.connect('sub2.z2.y', 'sub1.q1.c')

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        model.linear_solver = om.ScipyKrylov()

        prob.setup()

        model.sub1.linear_solver = om.DirectSolver()
        model.sub2.linear_solver = om.DirectSolver()

        model.linear_solver.precon = om.LinearBlockGS()

        prob.set_solver_print(level=2)
        prob.run_model()

        assert_near_equal(prob.get_val('sub1.q1.x'), 1.996, .0001)
        assert_near_equal(prob.get_val('sub2.q2.x'), 1.996, .0001)


class TestCaching(unittest.TestCase):

    def test_caching(self):
        prob = om.Problem()
        prob.driver = om.ScipyOptimizeDriver()
        model = prob.model

        model.add_subsystem('ivc', om.IndepVarComp('x', 1.0))
        G = model.add_subsystem('G', om.Group())
        G.add_subsystem('comp', om.ExecComp('y = 2.*x'))
        G.linear_solver = om.ScipyKrylov(iprint=2, assemble_jac=True,
                                         rhs_checking={'verbose': True, 'collect_stats': True})
        G.linear_solver.precon = om.LinearBlockGS(iprint=-1)
        model.add_subsystem('comp2', om.ExecComp('y = 3.*x'))
        model.add_subsystem('comp3', om.ExecComp('y = 4.*x'))

        model.connect('ivc.x', 'G.comp.x')
        model.connect('G.comp.y', ['comp2.x', 'comp3.x'])

        model.add_design_var('ivc.x')
        model.add_objective('comp2.y')
        model.add_constraint('G.comp.y', upper=10.0)

        prob.setup(mode='rev')

        prob.set_val('ivc.x', .3)
        prob.run_model()

        J = prob.compute_totals()

        self.assertEqual(G.linear_solver._lin_rhs_checker._stats['parhits'], 1)

        assert_check_totals(prob.check_totals(out_stream=None))

if __name__ == "__main__":
    unittest.main()
