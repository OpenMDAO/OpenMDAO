
import unittest

import openmdao.api as om
from openmdao.test_suite.build4test import DynComp, create_dyncomps, make_subtree


class BM(unittest.TestCase):
    """Setup of models with various tree structures"""

    def benchmark_L4_sub2_c10(self):
        p = om.Problem()
        make_subtree(p.model, nsubgroups=2, levels=4, ncomps=10,
                     ninputs=10, noutputs=10, nconns=5)
        p.setup(check=False)
        p.final_setup()

    def benchmark_L6_sub2_c10(self):
        p = om.Problem()
        make_subtree(p.model, nsubgroups=2, levels=6, ncomps=10,
                     ninputs=10, noutputs=10, nconns=5)
        p.setup(check=False)
        p.final_setup()

    def benchmark_L7_sub2_c10(self):
        p = om.Problem()
        make_subtree(p.model, nsubgroups=2, levels=7, ncomps=10,
                     ninputs=10, noutputs=10, nconns=5)
        p.setup(check=False)
        p.final_setup()
