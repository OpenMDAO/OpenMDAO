import unittest
from openmdao.utils.testing_utils import use_tempdirs

@use_tempdirs
class TestSellarFeature(unittest.TestCase):

    # no output checking, just make sure no exceptions raised
    # Just tests Newton on Sellar with FD derivs.
    def test_feature_sellar(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarNoDerivatives

        prob = om.Problem()
        prob.model = SellarNoDerivatives()

        prob.setup()
        prob.final_setup()

        om.view_connections(prob, outfile= "sellar_connections.html", show_browser=False)

if __name__ == "__main__":
    unittest.main()
