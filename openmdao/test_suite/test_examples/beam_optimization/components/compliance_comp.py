from __future__ import division
import numpy as np
from openmdao.api import ExplicitComponent


class ComplianceComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_elements', types=int)
        self.metadata.declare('force_vector', types=np.ndarray)

    def setup(self):
        num_elements = self.metadata['num_elements']
        num_nodes = num_elements + 1
        force_vector = self.metadata['force_vector']

        self.add_input('displacements', shape=2 * num_nodes)
        self.add_output('compliance')

        self.declare_partials('compliance', 'displacements',
            val=force_vector.reshape((1, 2 * num_nodes)))

    def compute(self, inputs, outputs):
        force_vector = self.metadata['force_vector']

        outputs['compliance'] = np.dot(force_vector, inputs['displacements'])
