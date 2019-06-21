import unittest

from openmdao.api import Problem
from openmdao.test_suite.components.sellar import SellarNoDerivatives
from openmdao.api import view_connections

class TestSellarFeature(unittest.TestCase):

    def test_sellar(self):
        # Just tests Newton on Sellar with FD derivs.

        prob = Problem()
        prob.model = SellarNoDerivatives()

        prob.setup()
        prob.final_setup()

        # no output checking, just make sure no exceptions raised
        view_connections(prob, show_browser=False)

if __name__ == "__main__":
    unittest.main()
