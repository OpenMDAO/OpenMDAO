from __future__ import division
import numpy as np
from openmdao.api import ExplicitComponent


class MomentOfInertiaComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('b')

    def setup(self):
        num_elements = self.options['num_elements']

        self.add_input('h', shape=num_elements)
        self.add_output('I', shape=num_elements)

        rows = np.arange(num_elements)
        cols = np.arange(num_elements)
        self.declare_partials('I', 'h', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        b = self.options['b']

        outputs['I'] = 1./12. * b * inputs['h'] ** 3

    def compute_partials(self, inputs, partials):
        b = self.options['b']

        partials['I', 'h'] = 1./4. * b * inputs['h'] ** 2
