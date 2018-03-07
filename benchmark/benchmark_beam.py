"""
This benchmark documents a peformance problem with data transfers that needs to be examined.
"""
import unittest

from openmdao.api import Problem
from openmdao.parallel_api import PETScVector
from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_group import MultipointBeamGroup
from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_group_slow import MultipointBeamGroup as MBGSlow


class BenchBeamSlowNP1(unittest.TestCase):

    N_PROCS = 1

    def benchmark_beam_slow_np1(self):

        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01

        num_elements = 50 * 32
        num_cp = 4
        num_load_cases = 32

        prob = Problem(model=MBGSlow(E=E, L=L, b=b, volume=volume,
                                     num_elements=num_elements, num_cp=num_cp,
                                     num_load_cases=num_load_cases))

        prob.setup(vector_class=PETScVector)

        prob.run_model()


class BenchBeamSlowNP2(unittest.TestCase):

    N_PROCS = 2

    def benchmark_beam_slow_np2(self):

        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01

        num_elements = 50 * 32
        num_cp = 4
        num_load_cases = 32

        prob = Problem(model=MBGSlow(E=E, L=L, b=b, volume=volume,
                                     num_elements=num_elements, num_cp=num_cp,
                                     num_load_cases=num_load_cases))

        prob.setup(vector_class=PETScVector)

        prob.run_model()


class BenchBeamSlowNP4(unittest.TestCase):

    N_PROCS = 4

    def benchmark_beam_slow_np4(self):

        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01

        num_elements = 50 * 32
        num_cp = 4
        num_load_cases = 32

        prob = Problem(model=MBGSlow(E=E, L=L, b=b, volume=volume,
                                     num_elements=num_elements, num_cp=num_cp,
                                     num_load_cases=num_load_cases))

        prob.setup(vector_class=PETScVector)

        prob.run_model()


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

        prob = Problem(model=MultipointBeamGroup(E=E, L=L, b=b, volume=volume,
                                                 num_elements=num_elements, num_cp=num_cp,
                                                 num_load_cases=num_load_cases))

        prob.setup(vector_class=PETScVector)

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

        prob = Problem(model=MultipointBeamGroup(E=E, L=L, b=b, volume=volume,
                                                 num_elements=num_elements, num_cp=num_cp,
                                                 num_load_cases=num_load_cases))

        prob.setup(vector_class=PETScVector)

        prob.run_model()


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

        prob = Problem(model=MultipointBeamGroup(E=E, L=L, b=b, volume=volume,
                                                 num_elements=num_elements, num_cp=num_cp,
                                                 num_load_cases=num_load_cases))

        prob.setup(vector_class=PETScVector)

        prob.run_model()