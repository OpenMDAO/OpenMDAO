import numpy as np

import openmdao.api as om


class MultiComplianceComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('force_vector', types=np.ndarray)
        self.options.declare('num_rhs', types=int)

    def setup(self):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        num_rhs = self.options['num_rhs']

        for j in range(num_rhs):
            self.add_input('displacements_%d' % j, shape=2 * num_nodes)
            self.add_output('compliance_%d' % j)
            force_vector = self.options['force_vector'][:, j]

            self.declare_partials('compliance_%d' % j, 'displacements_%d' % j,
                                  val=force_vector.reshape((1, 2 * num_nodes)))

    def compute(self, inputs, outputs):
        num_rhs = self.options['num_rhs']

        for j in range(num_rhs):
            force_vector = self.options['force_vector'][:, j]
            outputs['compliance_%d' % j] = np.dot(force_vector, inputs['displacements_%d' % j])
