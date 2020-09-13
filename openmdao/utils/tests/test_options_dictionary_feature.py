import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


class TestOptionsDictionaryFeature(unittest.TestCase):

    def test_simple(self):
        import openmdao.api as om
        from openmdao.test_suite.components.options_feature_vector import VectorDoublingComp

        prob = om.Problem()
        prob.model.add_subsystem('double', VectorDoublingComp(size=3))  # 'size' is an option

        prob.setup()

        prob.set_val('double.x', [1., 2., 3.])

        prob.run_model()
        assert_near_equal(prob.get_val('double.y'), [2., 4., 6.])

    def test_simple_fail(self):
        import openmdao.api as om
        from openmdao.test_suite.components.options_feature_vector import VectorDoublingComp

        prob = om.Problem()
        prob.model.add_subsystem('double', VectorDoublingComp())  # 'size' not specified

        try:
            prob.setup()
        except RuntimeError as err:
            self.assertEqual(str(err), "VectorDoublingComp (double): Option 'size' is required but has not been set.")

    def test_with_default(self):
        import openmdao.api as om
        from openmdao.test_suite.components.options_feature_lincomb import LinearCombinationComp

        prob = om.Problem()
        prob.model.add_subsystem('linear', LinearCombinationComp(a=2.))  # 'b' not specified

        prob.setup()

        prob.set_val('linear.x', 3)

        prob.run_model()
        self.assertEqual(prob.get_val('linear.y'), 7.)

    def test_simple_array(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.options_feature_array import ArrayMultiplyComp

        prob = om.Problem()
        prob.model.add_subsystem('a_comp', ArrayMultiplyComp(array=np.array([1, 2, 3])))

        prob.setup()

        prob.set_val('a_comp.x', 5.)

        prob.run_model()
        assert_near_equal(prob.get_val('a_comp.y'), [5., 10., 15.])

    def test_simple_function(self):
        import openmdao.api as om
        from openmdao.test_suite.components.options_feature_function import UnitaryFunctionComp

        def my_func(x):
            return x*2

        prob = om.Problem()
        prob.model.add_subsystem('f_comp', UnitaryFunctionComp(func=my_func))

        prob.setup()

        prob.set_val('f_comp.x', 5.)

        prob.run_model()
        assert_near_equal(prob.get_val('f_comp.y'), 10.)

    def test_simple_values(self):
        import numpy as np
        import openmdao.api as om

        class VectorDoublingComp(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('size', values=[2, 4, 6, 8])

            def setup(self):
                size = self.options['size']

                self.add_input('x', shape=size)
                self.add_output('y', shape=size)
                self.declare_partials('y', 'x', val=2.,
                                      rows=np.arange(size),
                                      cols=np.arange(size))

            def compute(self, inputs, outputs):
                outputs['y'] = 2 * inputs['x']

        prob = om.Problem()
        prob.model.add_subsystem('double', VectorDoublingComp(size=4))

        prob.setup()

        prob.set_val('double.x', [1., 2., 3., 4.])

        prob.run_model()
        assert_near_equal(prob.get_val('double.y'), [2., 4., 6., 8.])

    def test_simple_bounds_valid(self):
        import numpy as np
        import openmdao.api as om

        def check_even(name, value):
            if value % 2 != 0:
                raise ValueError(f"Option '{name}' with value {value} must be an even number.")

        class VectorDoublingComp(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('size', types=int, lower=2, upper=8, check_valid=check_even)

            def setup(self):
                size = self.options['size']

                self.add_input('x', shape=size)
                self.add_output('y', shape=size)
                self.declare_partials('y', 'x', val=2.,
                                      rows=np.arange(size),
                                      cols=np.arange(size))

            def compute(self, inputs, outputs):
                outputs['y'] = 2 * inputs['x']

        try:
            comp = VectorDoublingComp(size=5)
        except Exception as err:
            self.assertEqual(str(err), "Option 'size' with value 5 must be an even number.")


if __name__ == "__main__":
    unittest.main()
