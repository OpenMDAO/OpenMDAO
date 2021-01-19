import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

import openmdao.api as om


class StatesComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('num_elements', types=int)
        self.options.declare('force_vector', types=np.ndarray)

    def setup(self):
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        size = 2 * num_nodes + 2

        self.add_input('K_local', shape=(num_elements, 4, 4))
        self.add_output('d', shape=size)

        cols = np.arange(16*num_elements)
        rows = np.repeat(np.arange(4), 4)
        rows = np.tile(rows, num_elements) + np.repeat(np.arange(num_elements), 16) * 2

        self.declare_partials('d', 'K_local', rows=rows, cols=cols)
        self.declare_partials('d', 'd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        force_vector = np.concatenate([self.options['force_vector'], np.zeros(2)])

        self.K = self.assemble_CSC_K(inputs)
        residuals['d'] = self.K.dot(outputs['d'])  - force_vector

    def solve_nonlinear(self, inputs, outputs):
        force_vector = np.concatenate([self.options['force_vector'], np.zeros(2)])

        self.K = self.assemble_CSC_K(inputs)
        self.lu = splu(self.K)

        outputs['d'] = self.lu.solve(force_vector)

    def linearize(self, inputs, outputs, jacobian):
        num_elements = self.options['num_elements']

        self.K = self.assemble_CSC_K(inputs)
        self.lu = splu(self.K)

        i_elem = np.tile(np.arange(4), 4)
        i_d = np.tile(i_elem, num_elements) + np.repeat(np.arange(num_elements), 16) * 2

        jacobian['d', 'K_local'] = outputs['d'][i_d]

        jacobian['d', 'd'] = self.K.toarray()

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            d_outputs['d'] = self.lu.solve(d_residuals['d'])
        else:
            d_residuals['d'] = self.lu.solve(d_outputs['d'])

    def assemble_CSC_K(self, inputs):
        """
        Assemble the stiffness matrix in sparse CSC format.

        Returns
        -------
        ndarray
            Stiffness matrix as dense ndarray.
        """
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1
        num_entry = num_elements * 12 + 4
        ndim = num_entry + 4

        data = np.zeros((ndim, ), dtype=inputs._get_data().dtype)
        cols = np.empty((ndim, ))
        rows = np.empty((ndim, ))

        # First element.
        data[:16] = inputs['K_local'][0, :, :].flat
        cols[:16] = np.tile(np.arange(4), 4)
        rows[:16] = np.repeat(np.arange(4), 4)

        j = 16
        for ind in range(1, num_elements):
            ind1 = 2 * ind
            K = inputs['K_local'][ind, :, :]

            # NW quadrant gets summed with previous connected element.
            data[j-6:j-4] += K[0, :2]
            data[j-2:j] += K[1, :2]

            # NE quadrant
            data[j:j+4] = K[:2, 2:].flat
            rows[j:j+4] = np.array([ind1, ind1, ind1 + 1, ind1 + 1])
            cols[j:j+4] = np.array([ind1 + 2, ind1 + 3, ind1 + 2, ind1 + 3])

            # SE and SW quadrants together
            data[j+4:j+12] = K[2:, :].flat
            rows[j+4:j+12] = np.repeat(np.arange(ind1 + 2, ind1 + 4), 4)
            cols[j+4:j+12] = np.tile(np.arange(ind1, ind1 + 4), 2)

            j += 12

        data[-4:] = 1.0
        rows[-4] = 2 * num_nodes
        rows[-3] = 2 * num_nodes + 1
        rows[-2] = 0.0
        rows[-1] = 1.0
        cols[-4] = 0.0
        cols[-3] = 1.0
        cols[-2] = 2 * num_nodes
        cols[-1] = 2 * num_nodes + 1

        n_K = 2 * num_nodes + 2
        return coo_matrix((data, (rows, cols)), shape=(n_K, n_K)).tocsc()
