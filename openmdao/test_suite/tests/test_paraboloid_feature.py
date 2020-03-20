"""Test the Paraboloid systems used in the basic User Guide. """

import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.test_suite.components.paraboloid_feature import Paraboloid


class TestSellarFeature(unittest.TestCase):

    def test_paraboloid_feature(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        model.add_subsystem('p2', om.IndepVarComp('y', -4.0))
        model.add_subsystem('comp', Paraboloid())

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['comp.f_xy'], -15.0)

if __name__ == "__main__":
    unittest.main()
