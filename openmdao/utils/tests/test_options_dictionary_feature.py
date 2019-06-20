import unittest
from six import PY3, assertRegex

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error


class TestOptionsDictionaryFeature(unittest.TestCase):

    def test_simple(self):
        import openmdao.api as om
        from openmdao.test_suite.components.options_feature_vector import VectorDoublingComp

        prob = om.Problem()
        prob.model.add_subsystem('inputs', om.IndepVarComp('x', shape=3))
        prob.model.add_subsystem('double', VectorDoublingComp(size=3))  # 'size' is an option
        prob.model.connect('inputs.x', 'double.x')

        prob.setup()

        prob['inputs.x'] = [1., 2., 3.]

        prob.run_model()
        assert_rel_error(self, prob['double.y'], [2., 4., 6.])

    def test_simple_fail(self):
        import openmdao.api as om
        from openmdao.test_suite.components.options_feature_vector import VectorDoublingComp

        prob = om.Problem()
        prob.model.add_subsystem('inputs', om.IndepVarComp('x', shape=3))
        prob.model.add_subsystem('double', VectorDoublingComp())  # 'size' not specified
        prob.model.connect('inputs.x', 'double.x')

        try:
            prob.setup()
        except RuntimeError as err:
            self.assertEqual(str(err), "Option 'size' is required but has not been set.")

    def test_with_default(self):
        import openmdao.api as om
        from openmdao.test_suite.components.options_feature_lincomb import LinearCombinationComp

        prob = om.Problem()
        prob.model.add_subsystem('inputs', om.IndepVarComp('x'))
        prob.model.add_subsystem('linear', LinearCombinationComp(a=2.))  # 'b' not specified
        prob.model.connect('inputs.x', 'linear.x')

        prob.setup()

        prob['inputs.x'] = 3

        prob.run_model()
        self.assertEqual(prob['linear.y'], 7.)

    def test_simple_array(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.options_feature_array import ArrayMultiplyComp

        prob = om.Problem()
        prob.model.add_subsystem('inputs', om.IndepVarComp('x', 1.))
        prob.model.add_subsystem('a_comp', ArrayMultiplyComp(array=np.array([1, 2, 3])))
        prob.model.connect('inputs.x', 'a_comp.x')

        prob.setup()

        prob['inputs.x'] = 5.

        prob.run_model()

        assert_rel_error(self, prob['a_comp.y'], [5., 10., 15.])

    def test_simple_function(self):
        import openmdao.api as om
        from openmdao.test_suite.components.options_feature_function import UnitaryFunctionComp

        def my_func(x):
            return x*2

        prob = om.Problem()
        prob.model.add_subsystem('inputs', om.IndepVarComp('x', 1.))
        prob.model.add_subsystem('f_comp', UnitaryFunctionComp(func=my_func))
        prob.model.connect('inputs.x', 'f_comp.x')

        prob.setup()

        prob['inputs.x'] = 5.

        prob.run_model()

        assert_rel_error(self, prob['f_comp.y'], 10.)


if __name__ == "__main__":
    unittest.main()
