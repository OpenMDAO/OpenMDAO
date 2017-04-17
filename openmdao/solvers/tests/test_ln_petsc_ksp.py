"""Test the PetsKSP linear solver class."""

from __future__ import division, print_function

import unittest


from openmdao.core.problem import Problem
from openmdao.core.group import Group
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.solvers.ln_petsc_ksp import PetscKSP
from openmdao.solvers.ln_bgs import LinearBlockGS
from openmdao.solvers.ln_direct import DirectSolver
from openmdao.solvers.nl_newton import NewtonSolver
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimpleDense
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped, \
     SellarDerivatives

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

    def test_solve_linear_ksp_gmres(self):
        """Solve implicit system with PetscKSP using 'gmres' method."""

        group = TestImplicitGroup(lnSolverClass=PetscKSP, use_varsets=False)
        group.ln_solver.options['ksp_type'] = 'gmres'

        p = Problem(group)
        p.setup(vector_class=PETScVector, check=False)
        p.model.suppress_solver_output = True

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        group.run_solve_linear(['linear'], 'fwd')

        output = d_outputs._data
        assert_rel_error(self, output[0], group.expected_solution[0], 1e-15)

        # reverse
        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        group.run_solve_linear(['linear'], 'rev')

        output = d_residuals._data
        assert_rel_error(self, output[0], group.expected_solution[0], 1e-15)

    def test_solve_linear_ksp_maxiter(self):
        """Verify that PetscKSP abides by the 'maxiter' option."""

        group = TestImplicitGroup(lnSolverClass=PetscKSP)
        group.ln_solver.options['maxiter'] = 2

        p = Problem(group)
        p.setup(vector_class=PETScVector, check=False)
        p.model.suppress_solver_output = True

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        group.run_solve_linear(['linear'], 'fwd')

        self.assertTrue(group.ln_solver._iter_count == 3)

        # reverse
        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        group.run_solve_linear(['linear'], 'rev')

        self.assertTrue(group.ln_solver._iter_count == 3)

    def test_solve_linear_ksp_precon(self):
        """Solve implicit system with PetscKSP using a preconditioner."""

        group = TestImplicitGroup(lnSolverClass=PetscKSP)
        precon = group.ln_solver.precon = LinearBlockGS()

        p = Problem(group)
        p.setup(vector_class=PETScVector, check=False)
        p.model.suppress_solver_output = True

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        group.run_solve_linear(['linear'], 'fwd')

        output = d_outputs._data
        assert_rel_error(self, output[1], group.expected_solution[0], 1e-15)
        assert_rel_error(self, output[5], group.expected_solution[1], 1e-15)

        self.assertTrue(precon._iter_count > 0)

        # reverse
        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        group.run_solve_linear(['linear'], 'rev')

        output = d_residuals._data
        assert_rel_error(self, output[1], group.expected_solution[0], 3e-15)
        assert_rel_error(self, output[5], group.expected_solution[1], 3e-15)

        self.assertTrue(precon._iter_count > 0)

        # test the direct solver and make sure KSP correctly recurses for _linearize
        precon = group.ln_solver.precon = DirectSolver()
        p.setup(vector_class=PETScVector, check=False)

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        group.ln_solver._linearize()
        group.run_solve_linear(['linear'], 'fwd')

        output = d_outputs._data
        assert_rel_error(self, output[1], group.expected_solution[0], 1e-15)
        assert_rel_error(self, output[5], group.expected_solution[1], 1e-15)

        # reverse
        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        group.ln_solver._linearize()
        group.run_solve_linear(['linear'], 'rev')

        output = d_residuals._data
        assert_rel_error(self, output[1], group.expected_solution[0], 3e-15)
        assert_rel_error(self, output[5], group.expected_solution[1], 3e-15)

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


@unittest.skipUnless(PETScVector, "PETSc is required.")
class TestPetscKSPSolverFeature(unittest.TestCase):

    def test_specify_solver(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = PetscKSP()

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)

    def test_specify_ksp_type(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = PetscKSP()
        model.ln_solver.options['ksp_type'] = 'gmres'

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

        model.ln_solver = PetscKSP()
        model.ln_solver.options['maxiter'] = 3

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.2654054431, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.87246623559, .00001)

    def test_feature_atol(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = PetscKSP()
        model.ln_solver.options['atol'] = 1.0e-20

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001055699, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448533563, .00001)

    def test_feature_rtol(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = PetscKSP()
        model.ln_solver.options['rtol'] = 1.0e-20

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
        prob.model.ln_sollver = PetscKSP()

        prob.model.ln_solver.precon = LinearBlockGS()
        prob.model.ln_solver.precon.options['maxiter'] = 2

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

if __name__ == "__main__":
    unittest.main()
