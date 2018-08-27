from __future__ import division, print_function

import unittest
from six import iterkeys

from openmdao.utils.assert_utils import assert_rel_error
from openmdao.solvers.linear.direct import DirectSolver
from openmdao.test_suite.parametric_suite import parametric_suite


class ParameterizedTestCasesSubset(unittest.TestCase):
    N_PROCS = 2

    @parametric_suite(
        assembled_jac=[False],
        jacobian_type=['dense'],
        partial_type=['array'], #, 'aij'],
        partial_method=['fd'], #, 'cs'],
        num_var=[3], #, 4],
        var_shape=[(2, 3)], #, (2,)],
        connection_type=['explicit'],
        run_by_default=True,
    )
    def test_subset(self, param_instance):
        param_instance.linear_solver_class = DirectSolver
        param_instance.linear_solver_options = {}  # defaults not valid for DirectSolver

        param_instance.setup()
        problem = param_instance.problem
        model = problem.model

        expected_values = model.expected_values
        if expected_values:
            actual = {key: problem[key] for key in iterkeys(expected_values)}
            assert_rel_error(self, actual, expected_values, 1e-4)

        expected_totals = model.expected_totals
        if expected_totals:
            # Forward Derivatives Check
            totals = param_instance.compute_totals('fwd')
            assert_rel_error(self, totals, expected_totals, 1e-4)

            # Reverse Derivatives Check
            totals = param_instance.compute_totals('rev')
            assert_rel_error(self, totals, expected_totals, 1e-4)


if __name__ == '__main__':
    unittest.main()
