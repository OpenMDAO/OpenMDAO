"""Test the PetsKSP linear solver class."""

import unittest

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimpleDense
from openmdao.test_suite.components.misc_components import Comp4LinearCacheTest
from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from openmdao.test_suite.groups.implicit_group import TestImplicitGroup

from openmdao.utils.assert_utils import assert_near_equal, assert_warning


@unittest.skipUnless(PETScVector is not None, "PETSc is required.")
class TestPETScKrylov(unittest.TestCase):

    def test_options(self):
        """Verify that the PETScKrylov specific options are declared."""

        group = om.Group()
        group.linear_solver = om.PETScKrylov()

        assert(group.linear_solver.options['ksp_type'] == 'fgmres')

    def test_solve_linear_ksp_default(self):
        """Solve implicit system with PETScKrylov using default method."""

        group = TestImplicitGroup(lnSolverClass=om.PETScKrylov)

        p = om.Problem(group)
        p.setup()
        p.set_solver_print(level=0)

        # Conclude setup but don't run model.
        p.final_setup()

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_val(1.0)
        d_outputs.set_val(0.0)
        group.run_solve_linear(['linear'], 'fwd')

        output = d_outputs.asarray()
        assert_near_equal(output, group.expected_solution, 1e-15)

        # reverse
        d_outputs.set_val(1.0)
        d_residuals.set_val(0.0)
        group.run_solve_linear(['linear'], 'rev')

        output = d_residuals.asarray()
        assert_near_equal(output, group.expected_solution, 1e-15)

    def test_solve_linear_ksp_gmres(self):
        """Solve implicit system with PETScKrylov using 'gmres' method."""

        group = TestImplicitGroup(lnSolverClass=om.PETScKrylov)
        group.linear_solver.options['ksp_type'] = 'gmres'

        p = om.Problem(group)
        p.setup()
        p.set_solver_print(level=0)

        # Conclude setup but don't run model.
        p.final_setup()

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_val(1.0)
        d_outputs.set_val(0.0)
        group.run_solve_linear(['linear'], 'fwd')

        output = d_outputs.asarray()
        assert_near_equal(output, group.expected_solution, 1e-15)

        # reverse
        d_outputs.set_val(1.0)
        d_residuals.set_val(0.0)
        group.run_solve_linear(['linear'], 'rev')

        output = d_residuals.asarray()
        assert_near_equal(output, group.expected_solution, 1e-15)

    def test_solve_linear_ksp_maxiter(self):
        """Verify that PETScKrylov abides by the 'maxiter' option."""

        group = TestImplicitGroup(lnSolverClass=om.PETScKrylov)
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
        group.run_solve_linear(['linear'], 'fwd')

        self.assertTrue(group.linear_solver._iter_count == 3)

        # reverse
        d_outputs.set_val(1.0)
        d_residuals.set_val(0.0)
        group.run_solve_linear(['linear'], 'rev')

        self.assertTrue(group.linear_solver._iter_count == 3)

    def test_solve_linear_ksp_precon(self):
        """Solve implicit system with PETScKrylov using a preconditioner."""

        group = TestImplicitGroup(lnSolverClass=om.PETScKrylov)
        precon = group.linear_solver.precon = om.LinearBlockGS()

        p = om.Problem(group)
        p.setup()
        p.set_solver_print(level=0)

        # Conclude setup but don't run model.
        p.final_setup()

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_val(1.0)
        d_outputs.set_val(0.0)
        group.run_solve_linear(['linear'], 'fwd')

        output = d_outputs.asarray()
        assert_near_equal(output, group.expected_solution, 1e-15)

        self.assertTrue(precon._iter_count > 0)

        # reverse
        d_outputs.set_val(1.0)
        d_residuals.set_val(0.0)
        group.run_solve_linear(['linear'], 'rev')

        output = d_residuals.asarray()
        assert_near_equal(output, group.expected_solution, 3e-15)

        self.assertTrue(precon._iter_count > 0)

        # test the direct solver and make sure KSP correctly recurses for _linearize
        precon = group.linear_solver.precon = om.DirectSolver(assemble_jac=False)
        p.setup()

        # Conclude setup but don't run model.
        p.final_setup()

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_val(1.0)
        d_outputs.set_val(0.0)
        group.linear_solver._linearize()
        group.run_solve_linear(['linear'], 'fwd')

        output = d_outputs.asarray()
        assert_near_equal(output, group.expected_solution, 1e-15)

        # reverse
        d_outputs.set_val(1.0)
        d_residuals.set_val(0.0)
        group.linear_solver._linearize()
        group.run_solve_linear(['linear'], 'rev')

        output = d_residuals.asarray()
        assert_near_equal(output, group.expected_solution, 3e-15)

    def test_solve_linear_ksp_precon_left(self):
        """Solve implicit system with PETScKrylov using a preconditioner."""

        group = TestImplicitGroup(lnSolverClass=om.PETScKrylov)
        precon = group.linear_solver.precon = om.DirectSolver(assemble_jac=False)
        group.linear_solver.options['precon_side'] = 'left'
        group.linear_solver.options['ksp_type'] = 'richardson'

        p = om.Problem(group)
        p.setup()
        p.set_solver_print(level=0)

        # Conclude setup but don't run model.
        p.final_setup()

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_val(1.0)
        d_outputs.set_val(0.0)
        group.run_linearize()
        group.run_solve_linear(['linear'], 'fwd')

        output = d_outputs.asarray()
        assert_near_equal(output, group.expected_solution, 1e-15)

        # reverse
        d_outputs.set_val(1.0)
        d_residuals.set_val(0.0)
        group.run_linearize()
        group.run_solve_linear(['linear'], 'rev')

        output = d_residuals.asarray()
        assert_near_equal(output, group.expected_solution, 3e-15)

        # test the direct solver and make sure KSP correctly recurses for _linearize
        precon = group.linear_solver.precon = om.DirectSolver(assemble_jac=False)
        group.linear_solver.options['precon_side'] = 'left'
        group.linear_solver.options['ksp_type'] = 'richardson'

        p.setup()

        # Conclude setup but don't run model.
        p.final_setup()

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_val(1.0)
        d_outputs.set_val(0.0)
        group.linear_solver._linearize()
        group.run_solve_linear(['linear'], 'fwd')

        output = d_outputs.asarray()
        assert_near_equal(output, group.expected_solution, 1e-15)

        # reverse
        d_outputs.set_val(1.0)
        d_residuals.set_val(0.0)
        group.linear_solver._linearize()
        group.run_solve_linear(['linear'], 'rev')

        output = d_residuals.asarray()
        assert_near_equal(output, group.expected_solution, 3e-15)

    def test_solve_on_subsystem(self):
        """solve an implicit system with KSP attached anywhere but the root"""

        p = om.Problem()
        model = p.model

        dv = model.add_subsystem('des_vars', om.IndepVarComp())
        # just need a dummy variable so the sizes don't match between root and g1
        dv.add_output('dummy', val=1.0, shape=10)

        g1 = model.add_subsystem('g1', TestImplicitGroup(lnSolverClass=om.PETScKrylov))

        p.setup()

        p.set_solver_print(level=0)

        # Conclude setup but don't run model.
        p.final_setup()

        # forward
        d_inputs, d_outputs, d_residuals = g1.get_linear_vectors()
        d_residuals.set_val(1.0)
        d_outputs.set_val(0.0)
        g1.run_solve_linear(['linear'], 'fwd')

        output = d_outputs.asarray()
        assert_near_equal(output, g1.expected_solution, 1e-15)

        # reverse
        d_inputs, d_outputs, d_residuals = g1.get_linear_vectors()

        d_outputs.set_val(1.0)
        d_residuals.set_val(0.0)
        g1.linear_solver._linearize()
        g1.run_solve_linear(['linear'], 'rev')

        output = d_residuals.asarray()
        assert_near_equal(output, g1.expected_solution, 3e-15)

    def test_linear_solution_cache(self):
        # Test derivatives across a converged Sellar model. When caching
        # is performed, the second solve takes less iterations than the
        # first one.

        # Forward mode

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('d1', Comp4LinearCacheTest(), promotes=['x', 'y'])

        model.nonlinear_solver = om.NonlinearBlockGS()
        model.linear_solver = om.PETScKrylov()

        model.add_design_var('x', cache_linear_solution=True)
        model.add_objective('y', cache_linear_solution=True)

        prob.setup(mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        J = prob.driver._compute_totals(of=['y'], wrt=['x'], use_abs_names=False,
                                        return_format='flat_dict')
        icount1 = prob.model.linear_solver._iter_count
        J = prob.driver._compute_totals(of=['y'], wrt=['x'], use_abs_names=False,
                                        return_format='flat_dict')
        icount2 = prob.model.linear_solver._iter_count

        # Should take less iterations when starting from previous solution.
        self.assertTrue(icount2 < icount1)

        # Reverse mode

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('d1', Comp4LinearCacheTest(), promotes=['x', 'y'])

        model.nonlinear_solver = om.NonlinearBlockGS()
        model.linear_solver = om.PETScKrylov()

        model.add_design_var('x', cache_linear_solution=True)
        model.add_objective('y', cache_linear_solution=True)

        prob.setup(mode='rev')
        prob.set_solver_print(level=0)
        prob.run_model()

        J = prob.driver._compute_totals(of=['y'], wrt=['x'], use_abs_names=False,
                                        return_format='flat_dict')
        icount1 = prob.model.linear_solver._iter_count
        J = prob.driver._compute_totals(of=['y'], wrt=['x'], use_abs_names=False,
                                        return_format='flat_dict')
        icount2 = prob.model.linear_solver._iter_count

        # Should take less iterations when starting from previous solution.
        self.assertTrue(icount2 < icount1)

    def test_error_under_cs(self):
        """Verify that PETScKrylov abides by the 'maxiter' option."""
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        model.linear_solver = om.PETScKrylov()

        model.approx_totals(method='cs')

        prob.setup(mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        with self.assertRaises(RuntimeError) as cm:
            J = prob.compute_totals(of=['obj'], wrt=['z'])

        msg = 'PETScKrylov in <model> <class Group>: PETScKrylov solver is not supported under complex step.'
        self.assertEqual(str(cm.exception), msg)


@unittest.skipUnless(PETScVector, "PETSc is required.")
class TestPETScKrylovSolverFeature(unittest.TestCase):

    def test_specify_solver(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

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

        model.linear_solver = om.PETScKrylov()

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_near_equal(J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_near_equal(J['obj', 'z'][0][1], 1.78448534, .00001)

    def test_specify_ksp_type(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, \
             SellarDis2withDerivatives

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

        model.linear_solver = om.PETScKrylov()
        model.linear_solver.options['ksp_type'] = 'gmres'

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
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

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

        model.linear_solver = om.PETScKrylov()
        model.linear_solver.options['maxiter'] = 3

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_near_equal(J['obj', 'z'][0][0], 4.93218027, .00001)
        assert_near_equal(J['obj', 'z'][0][1], 1.73406455, .00001)

    def test_feature_atol(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

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

        model.linear_solver = om.PETScKrylov()
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

    def test_feature_rtol(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

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

        model.linear_solver = om.PETScKrylov()
        model.linear_solver.options['rtol'] = 1.0e-20

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
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, \
             SellarDis2withDerivatives

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        model.linear_solver = om.PETScKrylov()

        model.linear_solver.precon = om.LinearBlockGS()
        model.linear_solver.precon.options['maxiter'] = 2

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

    def test_specify_precon_left(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, \
             SellarDis2withDerivatives

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        model.linear_solver = om.PETScKrylov()

        model.linear_solver.precon = om.DirectSolver()
        model.linear_solver.options['precon_side'] = 'left'
        model.linear_solver.options['ksp_type'] = 'richardson'

        prob.setup()

        prob.set_val('x', 1.)
        prob.set_val('z', np.array([5.0, 2.0]))

        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 25.58830273, .00001)
        assert_near_equal(prob.get_val('y2'), 12.05848819, .00001)

if __name__ == "__main__":
    unittest.main()
