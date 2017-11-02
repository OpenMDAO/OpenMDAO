from __future__ import division
import numpy as np
from openmdao.api import ExplicitComponent


class VolumeComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_elements', types=int)
        self.metadata.declare('b', default=1.)
        self.metadata.declare('L')

    def setup(self):
        num_elements = self.metadata['num_elements']
        b = self.metadata['b']
        L = self.metadata['L']
        L0 = L / num_elements

        self.add_input('h', shape=num_elements)
        self.add_output('volume')

        self.declare_partials('volume', 'h', val=b * L0)

    def compute(self, inputs, outputs):
        num_elements = self.metadata['num_elements']
        b = self.metadata['b']
        L = self.metadata['L']
        L0 = L / num_elements

        outputs['volume'] = np.sum(inputs['h'] * b * L0)
