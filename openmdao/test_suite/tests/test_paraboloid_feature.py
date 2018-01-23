"""Test the Paraboloid systems used in the basic User Guide. """

import unittest

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.test_suite.components.paraboloid_feature import Paraboloid


class TestSellarFeature(unittest.TestCase):

    def test_paraboloid_feature(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 3.0))
        model.add_subsystem('p2', IndepVarComp('y', -4.0))
        model.add_subsystem('comp', Paraboloid())

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['comp.f_xy'], -15.0)

if __name__ == "__main__":
    unittest.main()