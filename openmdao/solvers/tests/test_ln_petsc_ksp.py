"""Test the PetsKSP linear solver class."""

from __future__ import division, print_function

import unittest


from openmdao.solvers.ln_petsc_ksp import PetscKSP
from openmdao.solvers.ln_bgs import LinearBlockGS

from openmdao.core.problem import Problem

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from openmdao.test_suite.groups.implicit_group import TestImplicitGroup

from openmdao.devtools.testutil import assert_rel_error


class TestPetscKSP(unittest.TestCase):
    def setUp(self):
        if PETScVector is None:
            raise unittest.SkipTest("PETSc is required.")

    def test_solve_linear_ksp_default(self):
        """Solve implicit system with PetscKSP using default method."""

        group = TestImplicitGroup(lnSolverClass=PetscKSP)

        p = Problem(group)
        p.setup(vector_class=PETScVector, check=False)
        p.root.suppress_solver_output = True

        # forward
        group._vectors['residual'][''].set_const(1.0)
        group._vectors['output'][''].set_const(0.0)
        group._solve_linear([''], 'fwd')

        output = group._vectors['output']['']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 1e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 1e-15)

        # reverse
        group._vectors['output'][''].set_const(1.0)
        group._vectors['residual'][''].set_const(0.0)
        group._solve_linear([''], 'rev')

        output = group._vectors['residual']['']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 1e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 1e-15)

    def test_solve_linear_ksp_gmres(self):
        """Solve implicit system with PetscKSP using 'gmres' method."""

        group = TestImplicitGroup(lnSolverClass=PetscKSP, use_varsets=False)
        group.ln_solver.options['ksp_type'] = 'gmres'

        p = Problem(group)
        p.setup(vector_class=PETScVector, check=False)
        p.root.suppress_solver_output = True

        # forward
        group._vectors['residual'][''].set_const(1.0)
        group._vectors['output'][''].set_const(0.0)
        group._solve_linear([''], 'fwd')

        output = group._vectors['output']['']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 1e-15)

        # reverse
        group._vectors['output'][''].set_const(1.0)
        group._vectors['residual'][''].set_const(0.0)
        group._solve_linear([''], 'rev')

        output = group._vectors['residual']['']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 1e-15)

    def test_solve_linear_ksp_maxiter(self):
        """Verify that PetscKSP abides by the 'maxiter' option."""

        group = TestImplicitGroup(lnSolverClass=PetscKSP)
        group.ln_solver.options['maxiter'] = 2

        p = Problem(group)
        p.setup(vector_class=PETScVector, check=False)
        p.root.suppress_solver_output = True

        # forward
        group._vectors['residual'][''].set_const(1.0)
        group._vectors['output'][''].set_const(0.0)
        group._solve_linear([''], 'fwd')

        self.assertTrue(group.ln_solver._iter_count == 3)

        # reverse
        group._vectors['output'][''].set_const(1.0)
        group._vectors['residual'][''].set_const(0.0)
        group._solve_linear([''], 'rev')

        self.assertTrue(group.ln_solver._iter_count == 3)

    def test_solve_linear_ksp_precon(self):
        """Solve implicit system with PetscKSP using a preconditioner."""

        group = TestImplicitGroup(lnSolverClass=PetscKSP)
        precon = group.ln_solver.set_subsolver('precon', LinearBlockGS())

        p = Problem(group)
        p.setup(vector_class=PETScVector, check=False)
        p.root.suppress_solver_output = True

        # forward
        group._vectors['residual'][''].set_const(1.0)
        group._vectors['output'][''].set_const(0.0)
        group._solve_linear([''], 'fwd')

        output = group._vectors['output']['']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 1e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 1e-15)

        self.assertTrue(precon._iter_count > 0)

        # reverse
        group._vectors['output'][''].set_const(1.0)
        group._vectors['residual'][''].set_const(0.0)
        group._solve_linear([''], 'rev')

        output = group._vectors['residual']['']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 3e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 3e-15)

        self.assertTrue(precon._iter_count > 0)


if __name__ == "__main__":
    unittest.main()
