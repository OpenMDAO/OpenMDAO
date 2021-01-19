import numpy as np

import openmdao.api as om


class ComplianceComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('force_vector', types=np.ndarray)

    def setup(self):
        num_nodes = self.options['num_elements'] + 1

        self.add_input('displacements', shape=2 * num_nodes)
        self.add_output('compliance')

    def setup_partials(self):
        num_nodes = self.options['num_elements'] + 1
        force_vector = self.options['force_vector']
        self.declare_partials('compliance', 'displacements',
                              val=force_vector.reshape((1, 2 * num_nodes)))

    def compute(self, inputs, outputs):
        outputs['compliance'] = np.dot(self.options['force_vector'], inputs['displacements'])
