"""
A component that computes y = a*x + b, where a and b
are given as metadata of type 'numpy.ScalarType'.
"""
import numpy as np

from openmdao.api import ExplicitComponent

class LinearCombinationComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('a', default=1., types=np.ScalarType)
        self.metadata.declare('b', default=1., types=np.ScalarType)

    def setup(self):
        self.add_input('x')
        self.add_output('y')
        self.declare_partials('y', 'x', val=self.metadata['a'])

    def compute(self, inputs, outputs):
        outputs['y'] = self.metadata['a'] * inputs['x'] + self.metadata['b']
