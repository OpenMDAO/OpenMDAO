import numpy as np

import openmdao.api as om


class MomentOfInertiaComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('b')

    def setup(self):
        num_elements = self.options['num_elements']

        self.add_input('h', shape=num_elements)
        self.add_output('I', shape=num_elements)

    def setup_partials(self):
        rows = cols = np.arange(self.options['num_elements'])
        self.declare_partials('I', 'h', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        outputs['I'] = 1./12. * self.options['b'] * inputs['h'] ** 3

    def compute_partials(self, inputs, partials):
        partials['I', 'h'] = 1./4. * self.options['b'] * inputs['h'] ** 2
