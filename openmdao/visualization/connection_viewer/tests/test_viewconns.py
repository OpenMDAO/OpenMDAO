import unittest

import openmdao.api as om
from openmdao.test_suite.components.sellar import SellarNoDerivatives
from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
class TestSellarFeatureHTML(unittest.TestCase):

    # no output checking, just make sure no exceptions raised
    # Just tests Newton on Sellar with FD derivs.
    def test_feature_sellar(self):

        prob = om.Problem()
        prob.model = SellarNoDerivatives()

        prob.setup()
        prob.final_setup()

        om.view_connections(prob, outfile= "sellar_connections.html", show_browser=False)


@use_tempdirs
class TestSellarFeatureCSV(unittest.TestCase):

    # no output checking, just make sure no exceptions raised
    # Just tests Newton on Sellar with FD derivs.
    def test_feature_sellar(self):

        prob = om.Problem()
        prob.model = SellarNoDerivatives()

        prob.setup()
        prob.final_setup()

        om.view_connections(prob, outfile= "sellar_connections.csv")


class TestComp(om.ExplicitComponent):

    def setup(self):
        self.add_discrete_input('foo', val='4')
        self.add_output('bar', val=0.)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        outputs['bar'] = float(discrete_inputs['foo'])


@use_tempdirs
class TestDiscreteViewConnsHTML(unittest.TestCase):
    def test_discrete(self):
        p = om.Problem()

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_discrete_output('foo', val='3')

        p.model.add_subsystem('test_comp', TestComp(), promotes=['*'])

        p.setup()
        p.final_setup()

        om.view_connections(p, show_browser=False)

    def test_pre_final_setup_error(self):

        prob = om.Problem()
        prob.model = SellarNoDerivatives()

        prob.setup()

        with self.assertRaises(RuntimeError) as cm:
            om.view_connections(prob, outfile= "sellar_connections.html",
                                show_values=True, show_browser=False)
        self.assertEqual(str(cm.exception), "view_connections may only be called after final_setup.")



@use_tempdirs
class TestDiscreteViewConnsCSV(unittest.TestCase):
    def test_discrete(self):
        p = om.Problem()

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_discrete_output('foo', val='3')

        p.model.add_subsystem('test_comp', TestComp(), promotes=['*'])

        p.setup()
        p.final_setup()

        om.view_connections(p, outfile="connections.csv")

    def test_pre_final_setup_error(self):

        prob = om.Problem()
        prob.model = SellarNoDerivatives()

        prob.setup()

        with self.assertRaises(RuntimeError) as cm:
            om.view_connections(prob, outfile= "sellar_connections.csv", show_values=True)

        self.assertEqual(str(cm.exception), "view_connections may only be called after final_setup.")

if __name__ == "__main__":
    unittest.main()
