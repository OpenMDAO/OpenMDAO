"""General tests to demonstrate the parametric suite. Possible arguments are given below (defaults).
To test more than one option, pass in an Iterable of requested options.

All Parametric Groups
---------------------
'group_type': Controls which type of ParametricGroups to test. Will test all groups if not specified
'local_vector_class': One of ['default', 'petsc'], which local vector class to use for the problem. ('default')
'assembled_jac': bool. If an assembled jacobian should be used. (True)
'jacobian_type': One of ['matvec', 'dense', 'sparse-csc']. How the Jacobians are used.
                 Controls the type of AssembledJacobian. ('matvec')
                    - 'matvec': Uses compute_jacvec_product.
                    - 'dense': Uses an ndarray.
                    - 'sparse-csc': Uses a Compressed Sparse Col sparse format.

CycleGroup ('group_type': 'cycle')
----------------------------------
'connection_type': One of ['implicit', 'explicit']. If connections are done explicitly or through
                   promotions ('implicit').
'partial_type': One of ['array', 'sparse', 'aij']. How the component partial derivatives are
                specified ('array').
                    - 'array': Uses an ndarray.
                    - 'sparse': Uses the Scipy CSR sparse format.
                    - 'aij': Uses the [values, rows, cols] format.
'partial_method': str. How derivatives should be solved.
            Approximated with finite differences, (fd, cs) OR
            solved for analytically, (exact).
'num_comp': int. Number of components to use. Must be at least 2. (2)
'num_var': int. Number of variables to use per component. Must be at least 1. (3)
'var_shape': tuple(int). Shape to use for each variable. (2, 3).
"""

import unittest

from openmdao.test_suite.parametric_suite import parametric_suite
from openmdao.utils.assert_utils import assert_near_equal


class ParameterizedTestCases(unittest.TestCase):
    """Demonstration of parametric testing using the full test suite."""

    @parametric_suite('*')
    def test_openmdao(self, test):
        test.setup()
        problem = test.problem

        root = problem.model

        expected_values = root.expected_values
        if expected_values:
            actual = {key: problem[key] for key in expected_values}
            assert_near_equal(actual, expected_values, 1e-8)

        error_bound = 1e-4 if root.options['partial_method'] != 'exact' else 1e-8

        expected_totals = root.expected_totals
        if expected_totals:
            # Forward Derivatives Check
            totals = test.compute_totals('fwd')
            assert_near_equal(totals, expected_totals, error_bound)

            # Reverse Derivatives Check
            totals = test.compute_totals('rev')
            assert_near_equal(totals, expected_totals, error_bound)


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
            actual = {key: problem[key] for key in expected_values}
            assert_near_equal(actual, expected_values, 1e-8)

        expected_totals = model.expected_totals
        if expected_totals:
            # Reverse Derivatives Check
            totals = param_instance.compute_totals('rev')
            assert_near_equal(totals, expected_totals, 1e-8)

            # Forward Derivatives Check
            totals = param_instance.compute_totals('fwd')
            assert_near_equal(totals, expected_totals, 1e-8)

            # Reverse Derivatives Check
            totals = param_instance.compute_totals('rev')
            assert_near_equal(totals, expected_totals, 1e-8)


if __name__ == '__main__':
    unittest.main()
