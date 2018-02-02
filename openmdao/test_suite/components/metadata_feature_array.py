"""
A component that multiplies an array by an input value, where
the array is given as metadata of type 'numpy.ndarray'.
"""
import numpy as np

from openmdao.api import ExplicitComponent

class ArrayMultiplyComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('array', types=np.ndarray)

    def setup(self):
        array = self.metadata['array']

        self.add_input('x', 1.)
        self.add_output('y', shape=array.shape)
        
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        outputs['y'] = self.metadata['array'] * inputs['x']
