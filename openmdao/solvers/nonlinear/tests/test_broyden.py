"""Test the Broyden nonlinear solver. """
from __future__ import print_function

import unittest

from openmdao.api import Problem, LinearRunOnce
from openmdao.solvers.nonlinear.broyden import BroydenSolver
from openmdao.test_suite.components.sellar import SellarStateConnection
from openmdao.utils.assert_utils import assert_rel_error


class TestBryoden(unittest.TestCase):

    def test_simple_Sellar(self):
        # Test top level Sellar (i.e., not grouped).

        prob = Problem()
        model = prob.model = SellarStateConnection(nonlinear_solver=BroydenSolver(),
                                                   linear_solver=LinearRunOnce())

        prob.setup(check=False)

        prob.set_solver_print(level=0)
        model.nonlinear_solver.options['state_vars'] = ['state_eq.y2_command']

        prob.set_solver_print(level=2, depth=1e99, type_='all')
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['state_eq.y2_command'], 12.05848819, .00001)


if __name__ == "__main__":
    unittest.main()