"""Test the ScipyIterativeSolver linear solver class."""

from __future__ import division, print_function

import unittest

from openmdao.solvers.ln_scipy import ScipyIterativeSolver

from openmdao.core.problem import Problem

from openmdao.test_suite.groups.implicit_group import TestImplicitGroup

from openmdao.devtools.testutil import assert_rel_error


class TestScipyIterativeSolver(unittest.TestCase):

    def test_solve_linear_scipy(self):
        """Solve implicit system with ScipyIterativeSolver."""

        group = TestImplicitGroup(lnSolverClass=ScipyIterativeSolver)

        p = Problem(group)
        p.setup(check=False)
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

    def test_solve_linear_scipy_maxiter(self):
        """Verify that ScipyIterativeSolver abides by the 'maxiter' option."""

        group = TestImplicitGroup(lnSolverClass=ScipyIterativeSolver)
        group.ln_solver.options['maxiter'] = 2

        p = Problem(group)
        p.setup(check=False)
        p.root.suppress_solver_output = True

        # forward
        group._vectors['residual'][''].set_const(1.0)
        group._vectors['output'][''].set_const(0.0)
        group._solve_linear([''], 'fwd')

        self.assertTrue(group.ln_solver._iter_count == 2)

        # reverse
        group._vectors['output'][''].set_const(1.0)
        group._vectors['residual'][''].set_const(0.0)
        group._solve_linear([''], 'rev')

        self.assertTrue(group.ln_solver._iter_count == 2)


if __name__ == "__main__":
    unittest.main()
