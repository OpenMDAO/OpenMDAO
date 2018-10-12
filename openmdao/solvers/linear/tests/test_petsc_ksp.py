"""Test the PetsKSP linear solver class."""

from __future__ import division, print_function

import unittest
import warnings

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, PETScKrylov, LinearBlockGS, DirectSolver, \
     ExecComp, NewtonSolver, NonlinearBlockGS, PetscKSP
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimpleDense
from openmdao.test_suite.components.misc_components import Comp4LinearCacheTest
from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from openmdao.test_suite.groups.implicit_group import TestImplicitGroup

from openmdao.utils.assert_utils import assert_rel_error


@unittest.skipUnless(PETScVector, "PETSc is required.")
class TestPETScKrylov(unittest.TestCase):

    def test_options(self):
        """Verify that the PETScKrylov specific options are declared."""

        group = Group()
        group.linear_solver = PETScKrylov()

        assert(group.linear_solver.options['ksp_type'] == 'fgmres')

    def test_solve_linear_ksp_default(self):
        """Solve implicit system with PETScKrylov using default method."""

        # use PetscKSP here to check for deprecation warning and verify that the deprecated
        # class still gets the right answer without duplicating this test.
        with warnings.catch_warnings(record=True) as w:
            group = TestImplicitGroup(lnSolverClass=PetscKSP)

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message), "PetscKSP is deprecated.  Use PETScKrylov instead.")

        p = Problem(group)
        p.setup(check=False)
        p.set_solver_print(level=0)

        # Conclude setup but don't run model.
        p.final_setup()

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        group.run_solve_linear(['linear'], 'fwd')

        output = d_outputs._data
        assert_rel_error(self, output, group.expected_solution, 1e-15)

        # reverse
        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        group.run_solve_linear(['linear'], 'rev')

        output = d_residuals._data
        assert_rel_error(self, output, group.expected_solution, 1e-15)

    def test_solve_linear_ksp_gmres(self):
        """Solve implicit system with PETScKrylov using 'gmres' method."""

        group = TestImplicitGroup(lnSolverClass=PETScKrylov)
        group.linear_solver.options['ksp_type'] = 'gmres'

        p = Problem(group)
        p.setup(check=False)
        p.set_solver_print(level=0)

        # Conclude setup but don't run model.
        p.final_setup()

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        group.run_solve_linear(['linear'], 'fwd')

        output = d_outputs._data
        assert_rel_error(self, output, group.expected_solution, 1e-15)

        # reverse
        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        group.run_solve_linear(['linear'], 'rev')

        output = d_residuals._data
        assert_rel_error(self, output, group.expected_solution, 1e-15)

    def test_solve_linear_ksp_maxiter(self):
        """Verify that PETScKrylov abides by the 'maxiter' option."""

        group = TestImplicitGroup(lnSolverClass=PETScKrylov)
        group.linear_solver.options['maxiter'] = 2

        p = Problem(group)
        p.setup(check=False)
        p.set_solver_print(level=0)

        # Conclude setup but don't run model.
        p.final_setup()

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        group.run_solve_linear(['linear'], 'fwd')

        self.assertTrue(group.linear_solver._iter_count == 3)

        # reverse
        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        group.run_solve_linear(['linear'], 'rev')

        self.assertTrue(group.linear_solver._iter_count == 3)

    def test_solve_linear_ksp_precon(self):
        """Solve implicit system with PETScKrylov using a preconditioner."""

        group = TestImplicitGroup(lnSolverClass=PETScKrylov)
        precon = group.linear_solver.precon = LinearBlockGS()

        p = Problem(group)
        p.setup(check=False)
        p.set_solver_print(level=0)

        # Conclude setup but don't run model.
        p.final_setup()

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        group.run_solve_linear(['linear'], 'fwd')

        output = d_outputs._data
        assert_rel_error(self, output, group.expected_solution, 1e-15)

        self.assertTrue(precon._iter_count > 0)

        # reverse
        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        group.run_solve_linear(['linear'], 'rev')

        output = d_residuals._data
        assert_rel_error(self, output, group.expected_solution, 3e-15)

        self.assertTrue(precon._iter_count > 0)

        # test the direct solver and make sure KSP correctly recurses for _linearize
        precon = group.linear_solver.precon = DirectSolver()
        p.setup(check=False)

        # Conclude setup but don't run model.
        p.final_setup()

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        group.linear_solver._linearize()
        group.run_solve_linear(['linear'], 'fwd')

        output = d_outputs._data
        assert_rel_error(self, output, group.expected_solution, 1e-15)

        # reverse
        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        group.linear_solver._linearize()
        group.run_solve_linear(['linear'], 'rev')

        output = d_residuals._data
        assert_rel_error(self, output, group.expected_solution, 3e-15)

    def test_solve_linear_ksp_precon_left(self):
        """Solve implicit system with PETScKrylov using a preconditioner."""

        group = TestImplicitGroup(lnSolverClass=PETScKrylov)
        precon = group.linear_solver.precon = DirectSolver()
        group.linear_solver.options['precon_side'] = 'left'
        group.linear_solver.options['ksp_type'] = 'richardson'

        p = Problem(group)
        p.setup(check=False)
        p.set_solver_print(level=0)

        # Conclude setup but don't run model.
        p.final_setup()

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        group.run_linearize()
        group.run_solve_linear(['linear'], 'fwd')

        output = d_outputs._data
        assert_rel_error(self, output, group.expected_solution, 1e-15)

        # reverse
        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        group.run_linearize()
        group.run_solve_linear(['linear'], 'rev')

        output = d_residuals._data
        assert_rel_error(self, output, group.expected_solution, 3e-15)

        # test the direct solver and make sure KSP correctly recurses for _linearize
        precon = group.linear_solver.precon = DirectSolver()
        group.linear_solver.options['precon_side'] = 'left'
        group.linear_solver.options['ksp_type'] = 'richardson'

        p.setup(check=False)

        # Conclude setup but don't run model.
        p.final_setup()

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        group.linear_solver._linearize()
        group.run_solve_linear(['linear'], 'fwd')

        output = d_outputs._data
        assert_rel_error(self, output, group.expected_solution, 1e-15)

        # reverse
        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        group.linear_solver._linearize()
        group.run_solve_linear(['linear'], 'rev')

        output = d_residuals._data
        assert_rel_error(self, output, group.expected_solution, 3e-15)

    def test_preconditioner_deprecation(self):

        group = TestImplicitGroup(lnSolverClass=PETScKrylov)

        msg = "The 'preconditioner' property provides backwards compatibility " \
            + "with OpenMDAO <= 1.x ; use 'precon' instead."

        # check deprecation on setter
        with warnings.catch_warnings(record=True) as w:
            precon = group.linear_solver.preconditioner = LinearBlockGS()

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message), msg)

        # check deprecation on getter
        with warnings.catch_warnings(record=True) as w:
            pre = group.linear_solver.preconditioner

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message), msg)

    def test_solve_on_subsystem(self):
        """solve an implicit system with KSP attached anywhere but the root"""

        p = Problem()
        model = p.model

        dv = model.add_subsystem('des_vars', IndepVarComp())
        # just need a dummy variable so the sizes don't match between root and g1
        dv.add_output('dummy', val=1.0, shape=10)

        g1 = model.add_subsystem('g1', TestImplicitGroup(lnSolverClass=PETScKrylov))

        p.setup(check=False)

        p.set_solver_print(level=0)

        # Conclude setup but don't run model.
        p.final_setup()

        # forward
        d_inputs, d_outputs, d_residuals = g1.get_linear_vectors()
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        g1.run_solve_linear(['linear'], 'fwd')

        output = d_outputs._data
        assert_rel_error(self, output, g1.expected_solution, 1e-15)

        # reverse
        d_inputs, d_outputs, d_residuals = g1.get_linear_vectors()

        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        g1.linear_solver._linearize()
        g1.run_solve_linear(['linear'], 'rev')

        output = d_residuals._data
        assert_rel_error(self, output, g1.expected_solution, 3e-15)

    def test_linear_solution_cache(self):
        # Test derivatives across a converged Sellar model. When caching
        # is performed, the second solve takes less iterations than the
        # first one.

        # Forward mode

        prob = Problem()
        model = prob.model

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('d1', Comp4LinearCacheTest(), promotes=['x', 'y'])

        model.nonlinear_solver = NonlinearBlockGS()
        model.linear_solver = PETScKrylov()

        model.add_design_var('x', cache_linear_solution=True)
        model.add_objective('y', cache_linear_solution=True)

        prob.setup(mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        J = prob.driver._compute_totals(of=['y'], wrt=['x'], global_names=False,
                                        return_format='flat_dict')
        icount1 = prob.model.linear_solver._iter_count
        J = prob.driver._compute_totals(of=['y'], wrt=['x'], global_names=False,
                                        return_format='flat_dict')
        icount2 = prob.model.linear_solver._iter_count

        # Should take less iterations when starting from previous solution.
        self.assertTrue(icount2 < icount1)

        # Reverse mode

        prob = Problem()
        model = prob.model

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('d1', Comp4LinearCacheTest(), promotes=['x', 'y'])

        model.nonlinear_solver = NonlinearBlockGS()
        model.linear_solver = PETScKrylov()

        model.add_design_var('x', cache_linear_solution=True)
        model.add_objective('y', cache_linear_solution=True)

        prob.setup(mode='rev')
        prob.set_solver_print(level=0)
        prob.run_model()

        J = prob.driver._compute_totals(of=['y'], wrt=['x'], global_names=False,
                                        return_format='flat_dict')
        icount1 = prob.model.linear_solver._iter_count
        J = prob.driver._compute_totals(of=['y'], wrt=['x'], global_names=False,
                                        return_format='flat_dict')
        icount2 = prob.model.linear_solver._iter_count

        # Should take less iterations when starting from previous solution.
        self.assertTrue(icount2 < icount1)

    def test_error_under_cs(self):
        """Verify that PETScKrylov abides by the 'maxiter' option."""
        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = NewtonSolver()
        model.linear_solver = PETScKrylov()

        model.approx_totals(method='cs')

        prob.setup(mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        with self.assertRaises(RuntimeError) as cm:
            J = prob.compute_totals(of=['obj'], wrt=['z'])

        msg = 'PETScKrylov solver is not supported under complex step.'
        self.assertEqual(str(cm.exception), msg)


@unittest.skipUnless(PETScVector, "PETSc is required.")
class TestPETScKrylovSolverFeature(unittest.TestCase):

    def test_specify_solver(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NonlinearBlockGS, PETScKrylov, ExecComp
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = NonlinearBlockGS()

        model.linear_solver = PETScKrylov()

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)

    def test_specify_ksp_type(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NonlinearBlockGS, PETScKrylov, \
             ExecComp
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, \
             SellarDis2withDerivatives

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = NonlinearBlockGS()

        model.linear_solver = PETScKrylov()
        model.linear_solver.options['ksp_type'] = 'gmres'

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)

    def test_feature_maxiter(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NonlinearBlockGS, PETScKrylov, ExecComp
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = NonlinearBlockGS()

        model.linear_solver = PETScKrylov()
        model.linear_solver.options['maxiter'] = 3

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 4.93218027, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.73406455, .00001)

    def test_feature_atol(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NonlinearBlockGS, PETScKrylov, ExecComp
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = NonlinearBlockGS()

        model.linear_solver = PETScKrylov()
        model.linear_solver.options['atol'] = 1.0e-20

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001055699, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448533563, .00001)

    def test_feature_rtol(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, NonlinearBlockGS, PETScKrylov, ExecComp
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = NonlinearBlockGS()

        model.linear_solver = PETScKrylov()
        model.linear_solver.options['rtol'] = 1.0e-20

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001055699, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448533563, .00001)

    def test_specify_precon(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, LinearBlockGS, PETScKrylov, \
             NewtonSolver, ExecComp
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, \
             SellarDis2withDerivatives

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = NewtonSolver()
        model.linear_solver = PETScKrylov()

        model.linear_solver.precon = LinearBlockGS()
        model.linear_solver.precon.options['maxiter'] = 2

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_specify_precon_left(self):
        import numpy as np

        from openmdao.api import Problem, Group, IndepVarComp, DirectSolver, PETScKrylov, \
             NewtonSolver, ExecComp
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, \
             SellarDis2withDerivatives

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = NewtonSolver()
        model.linear_solver = PETScKrylov()

        model.linear_solver.precon = DirectSolver()
        model.linear_solver.options['precon_side'] = 'left'
        model.linear_solver.options['ksp_type'] = 'richardson'

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

if __name__ == "__main__":
    unittest.main()
