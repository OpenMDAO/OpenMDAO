"""Test the PetsKSP linear solver class."""

from __future__ import division, print_function

import unittest


from openmdao.solvers.ln_petsc_ksp import PetscKSP
from openmdao.solvers.ln_bgs import LinearBlockGS
from openmdao.solvers.ln_direct import DirectSolver

from openmdao.core.problem import Problem
from openmdao.core.group import Group
from openmdao.core.indepvarcomp import IndepVarComp

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from openmdao.test_suite.groups.implicit_group import TestImplicitGroup

from openmdao.devtools.testutil import assert_rel_error


@unittest.skipUnless(PETScVector, "PETSc is required.")
class TestPetscKSP(unittest.TestCase):

    def test_options(self):
        """Verify that the PetscKSP specific options are declared."""

        group = Group()
        group.ln_solver = PetscKSP()

        assert(group.ln_solver.options['ksp_type'] == 'fgmres')

    def test_solve_linear_ksp_default(self):
        """Solve implicit system with PetscKSP using default method."""

        group = TestImplicitGroup(lnSolverClass=PetscKSP)

        p = Problem(group)
        p.setup(vector_class=PETScVector, check=False)
        p.model.suppress_solver_output = True

        # forward
        group._vectors['residual']['linear'].set_const(1.0)
        group._vectors['output']['linear'].set_const(0.0)
        group._solve_linear(['linear'], 'fwd')

        output = group._vectors['output']['linear']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 1e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 1e-15)

        # reverse
        group._vectors['output']['linear'].set_const(1.0)
        group._vectors['residual']['linear'].set_const(0.0)
        group._solve_linear(['linear'], 'rev')

        output = group._vectors['residual']['linear']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 1e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 1e-15)

    def test_solve_linear_ksp_gmres(self):
        """Solve implicit system with PetscKSP using 'gmres' method."""

        group = TestImplicitGroup(lnSolverClass=PetscKSP, use_varsets=False)
        group.ln_solver.options['ksp_type'] = 'gmres'

        p = Problem(group)
        p.setup(vector_class=PETScVector, check=False)
        p.model.suppress_solver_output = True

        # forward
        group._vectors['residual']['linear'].set_const(1.0)
        group._vectors['output']['linear'].set_const(0.0)
        group._solve_linear(['linear'], 'fwd')

        output = group._vectors['output']['linear']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 1e-15)

        # reverse
        group._vectors['output']['linear'].set_const(1.0)
        group._vectors['residual']['linear'].set_const(0.0)
        group._solve_linear(['linear'], 'rev')

        output = group._vectors['residual']['linear']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 1e-15)

    def test_solve_linear_ksp_maxiter(self):
        """Verify that PetscKSP abides by the 'maxiter' option."""

        group = TestImplicitGroup(lnSolverClass=PetscKSP)
        group.ln_solver.options['maxiter'] = 2

        p = Problem(group)
        p.setup(vector_class=PETScVector, check=False)
        p.model.suppress_solver_output = True

        # forward
        group._vectors['residual']['linear'].set_const(1.0)
        group._vectors['output']['linear'].set_const(0.0)
        group._solve_linear(['linear'], 'fwd')

        self.assertTrue(group.ln_solver._iter_count == 3)

        # reverse
        group._vectors['output']['linear'].set_const(1.0)
        group._vectors['residual']['linear'].set_const(0.0)
        group._solve_linear(['linear'], 'rev')

        self.assertTrue(group.ln_solver._iter_count == 3)

    def test_solve_linear_ksp_precon(self):
        """Solve implicit system with PetscKSP using a preconditioner."""

        group = TestImplicitGroup(lnSolverClass=PetscKSP)
        precon = group.ln_solver.precon = LinearBlockGS()

        p = Problem(group)
        p.setup(vector_class=PETScVector, check=False)
        p.model.suppress_solver_output = True

        # forward
        group._vectors['residual']['linear'].set_const(1.0)
        group._vectors['output']['linear'].set_const(0.0)
        group._solve_linear(['linear'], 'fwd')

        output = group._vectors['output']['linear']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 1e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 1e-15)

        self.assertTrue(precon._iter_count > 0)

        # reverse
        group._vectors['output']['linear'].set_const(1.0)
        group._vectors['residual']['linear'].set_const(0.0)
        group._solve_linear(['linear'], 'rev')

        output = group._vectors['residual']['linear']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 3e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 3e-15)

        self.assertTrue(precon._iter_count > 0)

        # test the direct solver and make sure KSP correctly recurses for _linearize
        precon = group.ln_solver.precon = DirectSolver()
        p.setup(vector_class=PETScVector, check=False)
        p.model.suppress_solver_output = True

        # forward
        group._vectors['residual']['linear'].set_const(1.0)
        group._vectors['output']['linear'].set_const(0.0)
        group.ln_solver._linearize()
        group._solve_linear(['linear'], 'fwd')

        output = group._vectors['output']['linear']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 1e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 1e-15)

        # reverse
        group._vectors['output']['linear'].set_const(1.0)
        group._vectors['residual']['linear'].set_const(0.0)
        group.ln_solver._linearize()
        group._solve_linear(['linear'], 'rev')

        output = group._vectors['residual']['linear']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 3e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 3e-15)

    def test_solve_on_subsystem(self):
        """solve an implicit system with KSP attached anywhere but the root"""

        p = Problem()
        model = p.model = Group()
        dv = model.add_subsystem('des_vars', IndepVarComp())
        # just need a dummy variable so the sizes don't match between root and g1
        dv.add_output('dummy', val=1.0, shape=10)

        g1 = model.add_subsystem('g1', TestImplicitGroup(lnSolverClass=PetscKSP))

        p.model.ln_solver.options['maxiter'] = 1
        p.setup(vector_class=PETScVector, check=False)

        p.model.suppress_solver_output = True

        # forward
        with g1.linear_vector_context() as (d_inputs, d_outputs, d_residuals):
            d_residuals.set_const(1.0)
            d_outputs.set_const(0.0)
            g1._solve_linear(['linear'], 'fwd')

            output = d_outputs._data
            # The empty first entry in _data is due to the dummy
            #     variable being in a different variable set not owned by g1
            assert_rel_error(self, output[1], g1.expected_solution[0], 1e-15)
            assert_rel_error(self, output[2], g1.expected_solution[1], 1e-15)

        # reverse
        with g1.linear_vector_context() as (d_inputs, d_outputs, d_residuals):
            d_outputs.set_const(1.0)
            d_residuals.set_const(0.0)
            g1.ln_solver._linearize()
            g1._solve_linear(['linear'], 'rev')

            output = d_residuals._data
            assert_rel_error(self, output[1], g1.expected_solution[0], 3e-15)
            assert_rel_error(self, output[2], g1.expected_solution[1], 3e-15)


if __name__ == "__main__":
    unittest.main()
