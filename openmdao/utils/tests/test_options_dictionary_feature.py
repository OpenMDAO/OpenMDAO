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
        prob.model.add_subsystem('inputs', IndepVarComp('x', shape=3))
        prob.model.add_subsystem('double', VectorDoublingComp(size=3))  # 'size' is metadata
        prob.model.connect('inputs.x', 'double.x')

        prob.setup()

        prob['inputs.x'] = [1., 2., 3.]

        prob.run_model()
        assert_rel_error(self, prob['double.y'], [2., 4., 6.])

    def test_simple_fail(self):
        from openmdao.api import Problem, IndepVarComp
        from openmdao.test_suite.components.metadata_feature_vector import VectorDoublingComp

        prob = Problem()
        prob.model.add_subsystem('inputs', IndepVarComp('x', shape=3))
        prob.model.add_subsystem('double', VectorDoublingComp())  # 'size' not specified
        prob.model.connect('inputs.x', 'double.x')

        try:
            prob.setup()
        except RuntimeError as err:
            self.assertEqual(str(err), "Entry 'size' is required but has not been set.")

    def test_with_default(self):
        from openmdao.api import Problem, IndepVarComp
        from openmdao.test_suite.components.metadata_feature_lincomb import LinearCombinationComp

        prob = Problem()
        prob.model.add_subsystem('inputs', IndepVarComp('x'))
        prob.model.add_subsystem('linear', LinearCombinationComp(a=2.))  # 'b' not specified
        prob.model.connect('inputs.x', 'linear.x')

        prob.setup()

        prob['inputs.x'] = 3

        prob.run_model()
        self.assertEqual(prob['linear.y'], 7.)

    def test_simple_array(self):
        import numpy as np

        from openmdao.api import Problem, IndepVarComp
        from openmdao.test_suite.components.metadata_feature_array import ArrayMultiplyComp

        prob = Problem()
        prob.model.add_subsystem('inputs', IndepVarComp('x', 1.))
        prob.model.add_subsystem('a_comp', ArrayMultiplyComp(array=np.array([1, 2, 3])))
        prob.model.connect('inputs.x', 'a_comp.x')

        prob.setup()

        prob['inputs.x'] = 5.

        prob.run_model()

        assert_rel_error(self, prob['a_comp.y'], [5., 10., 15.])

    def test_simple_function(self):
        from openmdao.api import Problem, IndepVarComp
        from openmdao.test_suite.components.metadata_feature_function import UnitaryFunctionComp

        def my_func(x):
            return x*2

        prob = Problem()
        prob.model.add_subsystem('inputs', IndepVarComp('x', 1.))
        prob.model.add_subsystem('f_comp', UnitaryFunctionComp(func=my_func))
        prob.model.connect('inputs.x', 'f_comp.x')

        prob.setup()

        prob['inputs.x'] = 5.

        prob.run_model()

        assert_rel_error(self, prob['f_comp.y'], 10.)


if __name__ == "__main__":
    unittest.main()
