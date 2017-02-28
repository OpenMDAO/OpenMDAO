"""Test the NLRunOnce linear solver class."""

import unittest

from openmdao.api import Problem, ScipyIterativeSolver
from openmdao.devtools.testutil import assert_rel_error
from openmdao.solvers.nl_runonce import NLRunOnce
from openmdao.test_suite.groups.parallel_groups import ConvergeDivergeGroups


class TestNLRunOnceSolver(unittest.TestCase):

    def test_converge_diverge_groups(self):
        # Test derivatives for converge-diverge-groups topology.
        prob = Problem()
        prob.model = ConvergeDivergeGroups()
        prob.model.ln_solver = ScipyIterativeSolver()
        prob.model.suppress_solver_output = True

        prob.model.nl_solver = NLRunOnce()
        g1 = prob.model.get_subsystem('g1')
        g2 = g1.get_subsystem('g2')
        g3 = prob.model.get_subsystem('g3')
        g1.nl_solver = NLRunOnce()
        g2.nl_solver = NLRunOnce()
        g3.nl_solver = NLRunOnce()

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        # Make sure value is fine.
        assert_rel_error(self, prob['c7.y1'], -102.7, 1e-6)

if __name__ == "__main__":
    unittest.main()
