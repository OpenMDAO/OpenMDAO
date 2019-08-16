from __future__ import division
from six.moves import range

import numpy as np

import openmdao.api as om


class DisplacementsComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)

    def setup(self):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes + 2

        self.add_input('d', shape=size)
        self.add_output('displacements', shape=2 * num_nodes)

        arange = np.arange(2 * num_nodes)
        self.declare_partials('displacements', 'd', val=1., rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1

        outputs['displacements'] = inputs['d'][:2 * num_nodes]
