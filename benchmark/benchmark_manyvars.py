
import unittest

from openmdao.api import Problem, Group
from openmdao.test_suite.build4test import DynComp, create_dyncomps


class BM(unittest.TestCase):
    """Some tests for setup of a component with a large
    number of variables.
    """

    def _build_comp(self, np, no, ns=0):
        prob = Problem(root=Group())
        prob.root.add("C1", DynComp(np, no, ns))
        return prob

    def benchmark_1Kparams(self):
        prob = self._build_comp(1000, 1)
        prob.setup(check=False)
        prob.final_setup()

    def benchmark_2Kparams(self):
        prob = self._build_comp(2000, 1)
        prob.setup(check=False)
        prob.final_setup()

    def benchmark_1Kouts(self):
        prob = self._build_comp(1, 1000)
        prob.setup(check=False)
        prob.final_setup()

    def benchmark_2Kouts(self):
        prob = self._build_comp(1, 2000)
        prob.setup(check=False)
        prob.final_setup()

    def benchmark_1Kvars(self):
        prob = self._build_comp(500, 500)
        prob.setup(check=False)
        prob.final_setup()

    def benchmark_2Kvars(self):
        prob = self._build_comp(1000, 1000)
        prob.setup(check=False)
        prob.final_setup()
