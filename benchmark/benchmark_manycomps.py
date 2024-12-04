
import unittest

import openmdao.api as om
from openmdao.test_suite.build4test import create_dyncomps, build_test_model


class BM(unittest.TestCase):
    """Setup of models with lots of components"""

    def benchmark_100(self):
        p = om.Problem()
        create_dyncomps(p.model, 100, 10, 10, 5)
        p.setup()
        p.final_setup()

    def benchmark_500(self):
        p = om.Problem()
        create_dyncomps(p.model, 500, 10, 10, 5)
        p.setup()
        p.final_setup()

    def benchmark_1K(self):
        p = om.Problem()
        create_dyncomps(p.model, 1000, 10, 10, 5)
        p.setup()
        p.final_setup()

    def benchmark_3K_compute_totals(self):
        p = om.Problem(build_test_model(3000, 4, 4, 3, 13, 1))
        p.driver = om.ScipyOptimizeDriver()
        p.setup()
        p.run_model()

        for i in range(10):
            p.compute_totals()
