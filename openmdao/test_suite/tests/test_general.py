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

        error_bound = 1e-4 if root.metadata['finite_difference'] else 1e-8

        expected_totals = root.expected_totals
        if expected_totals:
            # Forward Derivatives Check
            totals = test.compute_totals('fwd')
            assert_rel_error(self, totals, expected_totals, error_bound)

            # Reverse Derivatives Check
            totals = test.compute_totals('rev')
            assert_rel_error(self, totals, expected_totals, error_bound)


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


from openmdao.devtools.problem_viewer.problem_viewer import view_model


class DeprecatedTestCases(unittest.TestCase):
    """Duplicating some testing to demonstrate filters and default running."""

    # self.default_params = {
    #     'vector_class': ['default', 'petsc'],
    #     'global_jac': [True, False],
    #     'jacobian_type': ['matvec', 'dense', 'sparse-coo', 'sparse-csr'],
    # }

    # self.default_params.update({
    #     'component_class': ['explicit'],
    #     'connection_type': ['implicit', 'explicit'],
    #     'partial_type': ['array', 'sparse', 'aij'],
    #     'num_comp': [3, 2],
    #     'num_var': [3, 1],
    #     'var_shape': [(2, 3), (3,)],
    # })

    @parametric_suite(vector_class='default',
                      global_jac=True,
                      # jacobian_type='matvec',
                      jacobian_type='dense',
                      component_class='deprecated',
                      # component_class='explicit',
                      connection_type='explicit',
                      partial_type='array',
                      num_comp=[2],
                      num_var=[2],
                      var_shape=[(3,)],
                      run_by_default=True)
    def test_subset(self, param_instance):
        param_instance.setup()
        problem = param_instance.problem
        model = problem.model

        for typ in ['input', 'output']:
            print('======', typ, '======')
            for name in sorted(model._var_allprocs_names[typ]):
                print('%16s' % name, '=', problem[name])

        # view_model(problem)

        expected_values = model.expected_values
        from pprint import pprint
        pprint(expected_values)
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
