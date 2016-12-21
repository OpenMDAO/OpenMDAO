"""Test the DirectSolver linear solver class."""

from __future__ import division, print_function

import unittest

from openmdao.solvers.ln_direct import DirectSolver

from openmdao.core.problem import Problem

from openmdao.test_suite.groups.implicit_group import TestImplicitGroup

from openmdao.devtools.testutil import assert_rel_error


class TestDirectSolver(unittest.TestCase):

    def test_solve_linear_direct_default(self):
        """Solve implicit system with DirectSolver.
        """

        group = TestImplicitGroup(lnSolverClass=DirectSolver)

        p = Problem(group)
        p.setup(check=False)

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

    def test_solve_linear_direct_LU(self):
        """Solve implicit system with DirectSolver using 'LU' method.
        """

        group = TestImplicitGroup(lnSolverClass=DirectSolver)
        group.ln_solver.options['method'] = 'LU'

        p = Problem(group)
        p.setup(check=False)

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


if __name__ == "__main__":
    unittest.main()
