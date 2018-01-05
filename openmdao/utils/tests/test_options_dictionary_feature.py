from openmdao.api import OptionsDictionary, ExplicitComponent
import unittest
from six import PY3, assertRegex
import numpy as np
from openmdao.utils.assert_utils import assert_rel_error


class TestOptionsDictionaryFeature(unittest.TestCase):

    def test_simple(self):
        from openmdao.api import Problem, IndepVarComp
        from openmdao.test_suite.components.metadata_feature_vector import VectorDoublingComp

        prob = Problem()
        prob.model.add_subsystem('input_comp', IndepVarComp('x', shape=3))
        prob.model.add_subsystem('main_comp', VectorDoublingComp(size=3))
        prob.model.connect('input_comp.x', 'main_comp.x')
        prob.setup()

        prob['input_comp.x'] = [1., 2., 3.]
        prob.run_model()
        assert_rel_error(self, prob['main_comp.y'], [2., 4., 6.])

    def test_with_default(self):
        from openmdao.api import Problem, IndepVarComp
        from openmdao.test_suite.components.metadata_feature_lincomb import LinearCombinationComp

        prob = Problem()
        prob.model.add_subsystem('input_comp', IndepVarComp('x'))
        prob.model.add_subsystem('main_comp', LinearCombinationComp(a=2.))
        prob.model.connect('input_comp.x', 'main_comp.x')
        prob.setup()

        prob['input_comp.x'] = 3
        prob.run_model()
        self.assertEqual(prob['main_comp.y'], 7.)


if __name__ == "__main__":
    unittest.main()
