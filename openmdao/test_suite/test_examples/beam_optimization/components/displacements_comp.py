from __future__ import division
from six.moves import range

import numpy as np

from openmdao.api import ExplicitComponent


class DisplacementsComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_elements', types=int)

    def setup(self):
        num_elements = self.metadata['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes + 2

        self.add_input('d', shape=size)
        self.add_output('displacements', shape=2 * num_nodes)

        arange = np.arange(2 * num_nodes)
        self.declare_partials('displacements', 'd', val=1., rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        num_elements = self.metadata['num_elements']
        num_nodes = num_elements + 1

        outputs['displacements'] = inputs['d'][:2 * num_nodes]


class MultiDisplacementsComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_elements', types=int)
        self.metadata.declare('num_rhs', types=int)

    def setup(self):
        num_elements = self.metadata['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes + 2
        num_rhs = self.metadata['num_rhs']

        arange = np.arange(2 * num_nodes)

        for j in range(num_rhs):
            self.add_input('d_%d' % j, shape=size)
            self.add_output('displacements_%d' % j, shape=2 * num_nodes)

            self.declare_partials('displacements_%d' % j, 'd_%d' % j, val=1., rows=arange,
                                  cols=arange)

    def compute(self, inputs, outputs):
        num_elements = self.metadata['num_elements']
        num_nodes = num_elements + 1
        num_rhs = self.metadata['num_rhs']

        for j in range(num_rhs):
            outputs['displacements_%d' % j] = inputs['d_%d' % j][:2 * num_nodes]