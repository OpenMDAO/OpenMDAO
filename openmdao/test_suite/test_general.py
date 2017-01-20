"""General tests to demonstrate the parametric suite"""
from __future__ import print_function, division

import unittest
from nose_parameterized import parameterized
from openmdao.test_suite.parametric_suite import full_test_suite, test_suite
from openmdao.devtools.testutil import assert_rel_error
from six import iterkeys


def _test_name(testcase_func, param_num, params):
    return '_'.join(('test', params.args[0].name))


def _test_name2(testcase_func, param_num, params):
    return '_'.join(('test2', params.args[0].name))


class ParameterizedTestCases(unittest.TestCase):
    """The TestCase that actually runs all of the cases inherits from this."""

    @parameterized.expand(full_test_suite(),
                          testcase_func_name=_test_name)
    def test_openmdao(self, test):
        test.setup()
        problem = test.problem

        root = problem.root

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
    """Duplicating some testing to demonstrate filters."""
    @parameterized.expand(test_suite(jacobian_type='*', num_comp=[2, 5, 10], partial_type='aij'),
                          testcase_func_name=_test_name2)
    def test_subset(self, test):
        test.setup()
        problem = test.problem
        root = problem.root

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