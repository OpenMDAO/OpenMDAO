"""Test the parallel groups."""

from __future__ import division, print_function

import unittest

from openmdao.core.problem import Problem

from openmdao.vectors.petsc_vector import PETScVector

from openmdao.test_suite.groups.parallel_groups import \
    FanOutGrouped, FanInGrouped, Diamond, ConvergeDiverge

from openmdao.devtools.testutil import assert_rel_error


class TestParallelGroups(unittest.TestCase):

    N_PROCS = 2

    def test_fan_out_grouped(self):

        prob = Problem(FanOutGrouped())
        prob.setup(VectorClass=PETScVector)
        prob.run()

        assert_rel_error(self, prob['c2.y'], -6.0, 1e-6)
        assert_rel_error(self, prob['c3.y'], 15.0, 1e-6)

    def test_fan_in_grouped(self):

        prob = Problem()
        prob.root = FanInGrouped()
        prob.setup(check=False)
        prob.run()

        print(prob['c3.y'])
        assert_rel_error(self, prob['c3.y'], 29.0, 1e-6)

    def test_diamond(self):

        prob = Problem()
        prob.root = Diamond()
        prob.setup(check=False)
        prob.run()

        print(prob['c4.y1'])
        print(prob['c4.y2'])
        assert_rel_error(self, prob['c4.y1'], 46.0, 1e-6)
        assert_rel_error(self, prob['c4.y2'], -93.0, 1e-6)

    def test_converge_diverge(self):

        prob = Problem()
        prob.root = ConvergeDiverge()
        prob.setup(check=False)
        prob.run()

        print(prob['c7.y1'])
        assert_rel_error(self, prob['c7.y1'], -102.7, 1e-6)

if __name__ == "__main__":
    unittest.main()
