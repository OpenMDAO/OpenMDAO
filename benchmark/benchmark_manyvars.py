
import unittest

import openmdao.api as om
from openmdao.test_suite.build4test import DynComp, create_dyncomps


def _build_comp(np, no, ns=0):
    prob = om.Problem()
    prob.model.add_subsystem("C1", DynComp(np, no, ns))
    return prob


class BM(unittest.TestCase):
    """Some tests for setup of a component with a large
    number of variables.
    """

    def benchmark_1Kparams(self):
        prob = _build_comp(1000, 1)
        prob.setup(check=False)
        prob.final_setup()

    def benchmark_2Kparams(self):
        prob = _build_comp(2000, 1)
        prob.setup(check=False)
        prob.final_setup()

    def benchmark_1Kouts(self):
        prob = _build_comp(1, 1000)
        prob.setup(check=False)
        prob.final_setup()

    def benchmark_2Kouts(self):
        prob = _build_comp(1, 2000)
        prob.setup(check=False)
        prob.final_setup()

    def benchmark_1Kvars(self):
        prob = _build_comp(500, 500)
        prob.setup(check=False)
        prob.final_setup()

    def benchmark_2Kvars(self):
        prob = _build_comp(1000, 1000)
        prob.setup(check=False)
        prob.final_setup()


if __name__ == '__main__':
    prob = _build_comp(1, 2000)
    prob.setup(check=False)
    prob.final_setup()
