"""Test the PetsKSP linear solver class."""

from __future__ import division, print_function

import unittest


from openmdao.solvers.ln_petsc_ksp import PetscKSP

from openmdao.api import Problem
from openmdao.api import LinearBlockGS, LinearBlockJac

from openmdao.vectors.petsc_vector import PETScVector

from openmdao.test_suite.groups.implicit_group import TestImplicitGroup
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped

from openmdao.devtools.testutil import assert_rel_error


class TestPetscKSP(unittest.TestCase):

    def test_solve_linear_ksp_default(self):
        """Solve implicit system with PetscKSP using default method."""

        group = TestImplicitGroup(lnSolverClass=PetscKSP)

        p = Problem(group)
        p.setup(VectorClass=PETScVector)

        # forward
        group._vectors['residual'][''].set_const(1.0)
        group._vectors['output'][''].set_const(0.0)
        group._solve_linear([''], 'fwd')
        output = group._vectors['output']['']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 2e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 2e-15)

        # reverse
        group._vectors['output'][''].set_const(1.0)
        group._vectors['residual'][''].set_const(0.0)
        group._solve_linear([''], 'rev')
        output = group._vectors['residual']['']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 2e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 2e-15)

    def test_solve_linear_ksp_gmres(self):
        """Solve implicit system with PetscKSP using 'gmres' method."""

        group = TestImplicitGroup(lnSolverClass=PetscKSP)
        group.ln_solver.options['ksp_type'] = 'gmres'

        p = Problem(group)
        p.setup(VectorClass=PETScVector)

        # forward
        group._vectors['residual'][''].set_const(1.0)
        group._vectors['output'][''].set_const(0.0)
        group._solve_linear([''], 'fwd')
        output = group._vectors['output']['']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 2e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 2e-15)

        # reverse
        group._vectors['output'][''].set_const(1.0)
        group._vectors['residual'][''].set_const(0.0)
        group._solve_linear([''], 'rev')
        output = group._vectors['residual']['']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 2e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 2e-15)

    def test_solve_linear_ksp_maxiter(self):
        """Verify that PetscKSP abides by the 'maxiter' option."""

        group = TestImplicitGroup(lnSolverClass=PetscKSP)
        group.ln_solver.options['maxiter'] = 2

        p = Problem(group)
        p.setup(VectorClass=PETScVector)

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
        group.ln_solver.set_subsolver('precon', LinearBlockGS())

        p = Problem(group)
        p.setup(VectorClass=PETScVector)

        # forward
        group._vectors['residual'][''].set_const(1.0)
        group._vectors['output'][''].set_const(0.0)
        group._solve_linear([''], 'fwd')
        output = group._vectors['output']['']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 2e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 2e-15)

        # reverse
        group._vectors['output'][''].set_const(1.0)
        group._vectors['residual'][''].set_const(0.0)
        group._solve_linear([''], 'rev')
        output = group._vectors['residual']['']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 2e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 2e-15)

    def test_sellar_derivs_grouped_precon(self):
        """Solve Sellar problem with PetscKSP using a preconditioner."""

        group = SellarDerivativesGrouped()
        group.ln_solver = PetscKSP()
        group.ln_solver.set_subsolver('precon', LinearBlockGS())

        # set mda linear solver to LinearBlockJac with maxiter of 2
        for subsys in group._subsystems_allprocs:
            if subsys.name == 'mda':
                subsys.ln_solver = LinearBlockJac()
                subsys.ln_solver.options['maxiter'] = 2

        p = Problem(group)
        p.setup(VectorClass=PETScVector)
        p.run()

        # just make sure we are at the right answer
        assert_rel_error(self, p['y1'], 25.58830273, .00001)
        assert_rel_error(self, p['y2'], 12.05848819, .00001)

    def test_sellar_derivs_grouped_precon_mda(self):
        """Solve Sellar MDA sub-problem with PetscKSP using a preconditioner."""

        group = SellarDerivativesGrouped()

        # set mda linear solver to PetscKSP with preconditioner
        for subsys in group._subsystems_allprocs:
            if subsys.name == 'mda':
                subsys.ln_solver = PetscKSP()
                subsys.ln_solver.set_subsolver('precon', LinearBlockGS())

        p = Problem(group)
        p.setup(VectorClass=PETScVector)
        p.run()

        # just make sure we are at the right answer
        assert_rel_error(self, p['y1'], 25.58830273, .00001)
        assert_rel_error(self, p['y2'], 12.05848819, .00001)


if __name__ == "__main__":
    unittest.main()
