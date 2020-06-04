"""
This benchmark documents a performance problem with data transfers that needs to be examined.
"""
import unittest

import openmdao.api as om
from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_group import MultipointBeamGroup


class BenchBeamNP1(unittest.TestCase):

    N_PROCS = 1

    def benchmark_beam_np1(self):

        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01

        num_elements = 50 * 32
        num_cp = 4
        num_load_cases = 32

        prob = om.Problem(model=MultipointBeamGroup(E=E, L=L, b=b, volume=volume,
                                                    num_elements=num_elements, num_cp=num_cp,
                                                    num_load_cases=num_load_cases))

        prob.setup()

        prob.run_model()


class BenchBeamNP2(unittest.TestCase):

    N_PROCS = 2

    def benchmark_beam_np2(self):

        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01

        num_elements = 50 * 32
        num_cp = 4
        num_load_cases = 32

        prob = om.Problem(model=MultipointBeamGroup(E=E, L=L, b=b, volume=volume,
                                                    num_elements=num_elements, num_cp=num_cp,
                                                    num_load_cases=num_load_cases))

        prob.setup()

        prob.run_model()


@unittest.skip("for debugging, not for routine benchmarking")
class BenchBeamNP4(unittest.TestCase):

    N_PROCS = 4

    def benchmark_beam_np4(self):

        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01

        num_elements = 50 * 32
        num_cp = 4
        num_load_cases = 32

        prob = om.Problem(model=MultipointBeamGroup(E=E, L=L, b=b, volume=volume,
                                                    num_elements=num_elements, num_cp=num_cp,
                                                    num_load_cases=num_load_cases))

        prob.setup()

        prob.run_model()
