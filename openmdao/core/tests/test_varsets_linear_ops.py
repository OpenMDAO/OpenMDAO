
import unittest

from openmdao.core.problem import Problem
from openmdao.solvers.linear.scipy_iter_solver import ScipyKrylov
from openmdao.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS

from openmdao.test_suite.groups.implicit_group import TestImplicitGroup

from openmdao.utils.assert_utils import assert_rel_error


class TestVarSets(unittest.TestCase):
    def setUp(self):
        group = TestImplicitGroup(lnSolverClass=ScipyKrylov,
                                  nlSolverClass=NonlinearBlockGS,
                                  use_varsets=True)

        p = Problem(group)
        p.set_solver_print(level=0)
        p.setup(check=False)
        p.final_setup()
        self.p = p

        # now create the same problem with no varsets
        group = TestImplicitGroup(lnSolverClass=ScipyKrylov,
                                  nlSolverClass=NonlinearBlockGS,
                                  use_varsets=False)

        p = Problem(group)
        p.set_solver_print(level=0)
        p.setup(check=False)
        p.final_setup()
        self.p_no_varsets = p

    def test_apply_linear(self):
        # apply linear, 'fwd' with varsets
        root = self.p.model
        d_inputs, d_outputs, d_residuals = root.get_linear_vectors()

        d_outputs.set_const(1.0)
        root.run_apply_linear(['linear'], 'fwd')

        output = d_residuals._data
        assert_rel_error(self, output[1], [4, 5, 4, 5], 1e-15)
        assert_rel_error(self, output[5], [3, 6, 3, 6], 1e-15)
        data = d_residuals.get_data()

        # apply linear, 'fwd' with no varsets, number should be the same, but reordered
        root = self.p_no_varsets.model
        d_inputs, d_outputs, d_residuals = root.get_linear_vectors()

        d_outputs.set_const(1.0)
        root.run_apply_linear(['linear'], 'fwd')

        output_novs = d_residuals._data
        expected = data
        assert_rel_error(self, output_novs[0], expected, 1e-15)

        # apply linear, 'rev' with varsets
        root = self.p.model
        d_inputs, d_outputs, d_residuals = root.get_linear_vectors()

        d_residuals.set_const(1.0)
        root.run_apply_linear(['linear'], 'rev')

        output = d_outputs._data
        assert_rel_error(self, output[1], [4, 5, 4, 5], 1e-15)
        assert_rel_error(self, output[5], [3, 6, 3, 6], 1e-15)
        data = d_outputs.get_data()

        # apply linear, 'rev' with no varsets, number should be the same, but reordered
        root = self.p_no_varsets.model
        d_inputs, d_outputs, d_residuals = root.get_linear_vectors()

        d_residuals.set_const(1.0)
        root.run_apply_linear(['linear'], 'rev')

        output_novs = d_outputs._data
        expected = data
        assert_rel_error(self, output_novs[0], expected, 1e-15)

    def test_solve_linear(self):
        # solve linear, 'fwd' with varsets
        root = self.p.model
        d_inputs, d_outputs, d_residuals = root.get_linear_vectors()

        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        root.run_solve_linear(['linear'], 'fwd')

        output = d_outputs._data
        assert_rel_error(self, output[1], root.expected_solution[0], 1e-15)
        assert_rel_error(self, output[5], root.expected_solution[1], 1e-15)
        data = d_outputs.get_data()

        # solve linear, 'fwd' with no varsets for comparison
        root = self.p_no_varsets.model
        d_inputs, d_outputs, d_residuals = root.get_linear_vectors()

        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)

        root.run_solve_linear(['linear'], 'fwd')
        output_novs = d_outputs._data
        expected = data
        assert_rel_error(self, output_novs[0], expected, 1e-15)

        # solve linear, 'rev' with varsets
        root = self.p.model
        d_inputs, d_outputs, d_residuals = root.get_linear_vectors()

        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        root.run_solve_linear(['linear'], 'rev')

        output = d_residuals._data
        assert_rel_error(self, output[1], root.expected_solution[0], 1e-15)
        assert_rel_error(self, output[5], root.expected_solution[1], 1e-15)
        data = d_residuals.get_data()

        # solve linear, 'rev' with no varsets for comparison
        root = self.p_no_varsets.model
        d_inputs, d_outputs, d_residuals = root.get_linear_vectors()

        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        root.run_solve_linear(['linear'], 'rev')

        output_novs = d_residuals._data
        expected = data
        assert_rel_error(self, output_novs[0], expected, 1e-15)


if __name__ == "__main__":
    unittest.main()
