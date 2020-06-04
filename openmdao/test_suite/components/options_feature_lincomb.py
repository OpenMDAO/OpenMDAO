"""
A component that computes y = a*x + b, where a and b
are given as an option of type 'numpy.ScalarType'.
"""
import numpy as np

import openmdao.api as om


class LinearCombinationComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('a', default=1., types=np.ScalarType)
        self.options.declare('b', default=1., types=np.ScalarType)

    def setup(self):
        self.add_input('x')
        self.add_output('y')
        self.declare_partials('y', 'x', val=self.options['a'])

    def compute(self, inputs, outputs):
        outputs['y'] = self.options['a'] * inputs['x'] + self.options['b']
