import unittest


class TestSellarFeature(unittest.TestCase):

    # no output checking, just make sure no exceptions raised
    # Just tests Newton on Sellar with FD derivs.
    def test_feature_sellar(self):
        from openmdao.api import Problem
        from openmdao.test_suite.components.sellar import SellarNoDerivatives
        from openmdao.api import view_connections

        prob = Problem()
        prob.model = SellarNoDerivatives()

        prob.setup()
        prob.final_setup()

        view_connections(prob, outfile= "sellar_connections.html", show_browser=False)

if __name__ == "__main__":
    unittest.main()
