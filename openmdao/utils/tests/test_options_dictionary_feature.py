from openmdao.api import OptionsDictionary, ExplicitComponent
import unittest
from six import PY3, assertRegex
import numpy as np
from openmdao.devtools.testutil import assert_rel_error


class VectorDoublingComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('size', type_=int)

    def setup(self):
        size = self.metadata['size']

        self.add_input('x', shape=size)
        self.add_output('y', shape=size)
        self.declare_partials('y', 'x', val=2., rows=np.arange(size), cols=np.arange(size))

    def compute(self, inputs, outputs):
        outputs['y'] = 2 * inputs['x']


class LinearCombinationComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('a', default=1., type_=(int, float))
        self.metadata.declare('b', default=1., type_=(int, float))

    def setup(self):
        self.add_input('x')
        self.add_output('y')
        self.declare_partials('y', 'x', val=self.metadata['a'])

    def compute(self, inputs, outputs):
        outputs['y'] = self.metadata['a'] * inputs['x'] + self.metadata['b']


class UnitaryFunctionComp(ExplicitComponent):

    def initialize(self):
        from types import FunctionType

        self.metadata.declare('func', values=('exp', 'cos', 'sin'), type_=FunctionType)

    def setup(self):
        self.add_input('x')
        self.add_output('y')
        self.declare_partials('y', 'x', method='fd')

    def compute(self, inputs, outputs):
        func = self.metadata['func']

        if func == 'exp':
            outputs['y'] = np.exp(inputs['x'])
        elif func == 'cos':
            outputs['y'] = np.cos(inputs['x'])
        elif func == 'sin':
            outputs['y'] = np.sin(inputs['x'])
        else:
            outputs['y'] = func(inputs['x'])


class TestOptionsDictionaryFeature(unittest.TestCase):

    def test_simple(self):
        from openmdao.api import Problem

        prob = Problem()
        prob.model = VectorDoublingComp(size=3)
        prob.setup()

        prob['x'] = [1., 2., 3.]
        prob.run_model()
        assert_rel_error(self, prob['y'], [2., 4., 6.])

    def test_with_default(self):
        from openmdao.api import Problem

        prob = Problem()
        prob.model = LinearCombinationComp(a=2.)
        prob.setup()

        prob['x'] = 3
        prob.run_model()
        self.assertEqual(prob['y'], 7.)

    def test_values_and_types(self):
        from openmdao.api import Problem

        prob = Problem()
        prob.model = UnitaryFunctionComp(func='cos')
        prob.setup()

        prob['x'] = 0.
        prob.run_model()
        self.assertEqual(prob['y'], 1.)

        def myfunc(x):
            return x ** 2 + 2

        prob = Problem()
        prob.model = UnitaryFunctionComp(func=myfunc)
        prob.setup()

        prob['x'] = 2.
        prob.run_model()
        self.assertEqual(prob['y'], 6.)


if __name__ == "__main__":
    unittest.main()
