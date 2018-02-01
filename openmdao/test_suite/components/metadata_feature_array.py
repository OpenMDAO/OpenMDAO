"""
Component for a metadata feature test.
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
        # self.declare_partials('y', 'x', val=2., rows=np.arange(size), cols=np.arange(size))

    def compute(self, inputs, outputs):
        outputs['y'] = self.metadata['array'] * inputs['x']
