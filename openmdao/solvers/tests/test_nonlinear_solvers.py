"""Test the NonlinearBlockGS solver."""
from __future__ import division, print_function
import unittest

from openmdao.solvers.nl_bgs import NonlinearBlockGS
from openmdao.solvers.nl_bjac import NonlinearBlockJac
from openmdao.solvers.nl_btlinesearch import BacktrackingLineSearch
from openmdao.solvers.nl_newton import NewtonSolver

from openmdao.test_suite.test_general import full_test_suite, _nice_name
import itertools
from nose_parameterized import parameterized
from six import iteritems

from openmdao.solvers.ln_scipy import ScipyIterativeSolver


SOLVERS_AND_OPTIONS = (
    (NonlinearBlockGS, {}),
    (NonlinearBlockJac, {}),
    (NewtonSolver, {'subsolvers':{'linesearch':BacktrackingLineSearch()}}),
    (NewtonSolver, {'subsolvers':{'linear': ScipyIterativeSolver(maxiter=100,)}}),
)


def make_tests():
    for solver_class, opts in SOLVERS_AND_OPTIONS:
        for test_problem, in full_test_suite():
            test_problem.solver_class = solver_class
            test_problem.solver_options = opts
            yield (test_problem, solver_class, opts)

make_tests.__test__ = False


def _test_name(testcase_func, param_num, params):
    test_problem = params.args[0]
    solver_class = params.args[1]
    solver_options = params.args[2]
    return '_'.join(('test',
                     solver_class.__name__,
                     _nice_name(solver_options),
                     test_problem.name))


class NonlinearSolversTest(unittest.TestCase):
    @parameterized.expand(make_tests(),
                          testcase_func_name=_test_name)
    def test_nonlinear_solver(self, test, *args):

        fail, rele, abse = test.run()
        if fail:
            self.fail('Problem run failed: re %f ; ae %f' % (rele, abse))

        self.assertAlmostEqual(test.apply_linear_test(mode='fwd'), 0)
        self.assertAlmostEqual(test.apply_linear_test(mode='rev'), 0)
        self.assertAlmostEqual(test.solve_linear_test(mode='fwd'), 0)
        self.assertAlmostEqual(test.solve_linear_test(mode='rev'), 0)