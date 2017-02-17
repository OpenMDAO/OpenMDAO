"""General tests to demonstrate the parametric suite"""
from __future__ import print_function, division

import unittest
from six import iterkeys

from openmdao.test_suite.parametric_suite import parametric_suite
from openmdao.devtools.testutil import assert_rel_error


class ParameterizedTestCases(unittest.TestCase):
    """Demonstration of parametric testing using the full test suite."""

    @parametric_suite('*')
    def test_openmdao(self, test):
        test.setup()
        problem = test.problem

        root = problem.model

        expected_values = root.expected_values
        if expected_values:
            actual = {key: problem[key] for key in iterkeys(expected_values)}
            assert_rel_error(self, actual, expected_values, 1e-8)

        expected_totals = root.expected_totals
        if expected_totals:
            # Forward Derivatives Check
            totals = test.compute_totals('fwd')
            assert_rel_error(self, totals, expected_totals, 1e-8)

            # Reverse Derivatives Check
            totals = test.compute_totals('rev')
            assert_rel_error(self, totals, expected_totals, 1e-8)


class ParameterizedTestCasesSubset(unittest.TestCase):
    """Duplicating some testing to demonstrate filters and default running."""
    
    @parametric_suite(jacobian_type='*',
                      num_comp=[2, 5, 10],
                      partial_type='aij',
                      run_by_default=True)
    def test_subset(self, param_instance):
        param_instance.setup()
        problem = param_instance.problem
        model = problem.model

        expected_values = model.expected_values
        if expected_values:
            actual = {key: problem[key] for key in iterkeys(expected_values)}
            assert_rel_error(self, actual, expected_values, 1e-8)

        expected_totals = model.expected_totals
        if expected_totals:
            # Forward Derivatives Check
            totals = param_instance.compute_totals('fwd')
            assert_rel_error(self, totals, expected_totals, 1e-8)

            # Reverse Derivatives Check
            totals = param_instance.compute_totals('rev')
            assert_rel_error(self, totals, expected_totals, 1e-8)

if __name__ == '__main__':
    unittest.main()
