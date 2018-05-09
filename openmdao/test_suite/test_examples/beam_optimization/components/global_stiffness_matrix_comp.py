from __future__ import division
import numpy as np
from openmdao.api import ExplicitComponent


class GlobalStiffnessMatrixComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)

    def setup(self):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1

        self.add_input('K_local', shape=(num_elements, 4, 4))
        self.add_output('K', shape=(2 * num_nodes + 2, 2 * num_nodes + 2))

        rows = np.zeros(16 * num_elements, int)
        indices = np.arange(
            ((2 * num_nodes + 2) * (2 * num_nodes + 2))
        ).reshape((2 * num_nodes + 2, 2 * num_nodes + 2))
        ind1, ind2 = 0, 0
        for ind in range(num_elements):
            ind2 += 16
            ind1_ = 2 * ind
            ind2_ = 2 * ind + 4
            rows[ind1:ind2] = indices[ind1_:ind2_, ind1_:ind2_].flatten()
            ind1 += 16
        cols = np.arange(16 * num_elements)
        self.declare_partials('K', 'K_local', val=1., rows=rows, cols=cols)

        self.set_check_partial_options('K_local', step=1e0)

    def compute(self, inputs, outputs):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1

        outputs['K'][:, :] = 0.
        for ind in range(num_elements):
            ind1_ = 2 * ind
            ind2_ = 2 * ind + 4

            outputs['K'][ind1_:ind2_, ind1_:ind2_] += inputs['K_local'][ind, :, :]

        outputs['K'][2 * num_nodes + 0, 0] = 1.0
        outputs['K'][2 * num_nodes + 1, 1] = 1.0
        outputs['K'][0, 2 * num_nodes + 0] = 1.0
        outputs['K'][1, 2 * num_nodes + 1] = 1.0
