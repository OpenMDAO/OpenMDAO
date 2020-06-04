"""
A component that multiplies an array by an input value, where
the array is given as an option of type 'numpy.ndarray'.
"""
import numpy as np

import openmdao.api as om


class ArrayMultiplyComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('array', types=np.ndarray)

    def setup(self):
        array = self.options['array']

        self.add_input('x', 1.)
        self.add_output('y', shape=array.shape)

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        outputs['y'] = self.options['array'] * inputs['x']
