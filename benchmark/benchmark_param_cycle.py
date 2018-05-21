
import unittest

from openmdao.api import Problem, Group, NewtonSolver, ScipyKrylov, NonlinearBlockGS, \
    LinearBlockGS, DirectSolver
from openmdao.test_suite.parametric_suite import ParameterizedInstance
from openmdao.utils.assert_utils import assert_rel_error


def _build(solver_class=NewtonSolver, linear_solver_class=ScipyKrylov,
           solver_options=None, linear_solver_options=None, **options):
    suite = ParameterizedInstance('cycle', **options)
    suite.solver_class = solver_class
    if solver_options is not None:
        suite.solver_options = solver_options
    if linear_solver_options is not None:
        suite.linear_solver_options = linear_solver_options
    suite.linear_solver_class = linear_solver_class
    suite.setup()
    return suite


def _check_results(testcase, test, error_bound):
    problem = test.problem
    root = problem.model

    expected_values = root.expected_values
    if expected_values:
        actual = {key: problem[key] for key in expected_values}
        assert_rel_error(testcase, actual, expected_values, 1e-8)

    expected_totals = root.expected_totals
    if expected_totals:
        # Forward Derivatives Check
        totals = test.compute_totals('fwd')
        assert_rel_error(testcase, totals, expected_totals, error_bound)

        # Reverse Derivatives Check
        totals = test.compute_totals('rev')
        assert_rel_error(testcase, totals, expected_totals, error_bound)


class BM(unittest.TestCase):

    def benchmark_comp200_var5_nlbs_lbgs(self):
        suite = _build(
            solver_class=NonlinearBlockGS, linear_solver_class=LinearBlockGS,
            assembled_jac=False,
            jacobian_type='dense',
            connection_type='explicit',
            partial_type='array',
            finite_difference=False,
            num_var=5, num_comp=200,
            var_shape=(3,)
        )
        suite.problem.run_driver()
        _check_results(self, suite, error_bound=1e-7)

    def benchmark_comp200_var5_newton_lings(self):
        suite = _build(
            solver_class=NewtonSolver, linear_solver_class=LinearBlockGS,
            solver_options={'maxiter': 20},
            linear_solver_options={'maxiter': 200, 'atol': 1e-10, 'rtol': 1e-10},
            assembled_jac=False,
            jacobian_type='dense',
            connection_type='explicit',
            partial_type='array',
            finite_difference=False,
            num_var=5, num_comp=200,
            var_shape=(3,)
        )
        suite.problem.run_driver()
        _check_results(self, suite, error_bound=1e-7)

    def benchmark_comp200_var5_newton_direct_assembled(self):
        suite = _build(
            solver_class=NewtonSolver, linear_solver_class=DirectSolver,
            linear_solver_options={},  # defaults not valid for DirectSolver
            assembled_jac=True,
            jacobian_type='dense',
            connection_type='explicit',
            partial_type='array',
            finite_difference=False,
            num_var=5, num_comp=200,
            var_shape=(3,)
        )
        suite.problem.run_driver()
        _check_results(self, suite, error_bound=1e-7)

    def benchmark_comp50_var5_newton_direct_assembled_fd(self):
        suite = _build(
            solver_class=NewtonSolver, linear_solver_class=DirectSolver,
            linear_solver_options={},  # defaults not valid for DirectSolver
            assembled_jac=True,
            jacobian_type='dense',
            connection_type='explicit',
            partial_type='array',
            finite_difference=True,
            num_var=5, num_comp=50,
            var_shape=(3,)
        )
        suite.problem.run_driver()
        _check_results(self, suite, error_bound=1e-5)


if __name__ == '__main__':
    tname = 'benchmark_comp200_var5_newton_direct_assembled_fd'
    getattr(BM(tname), tname)()
