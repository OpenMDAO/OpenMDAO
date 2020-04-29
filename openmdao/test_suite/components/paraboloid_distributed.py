import numpy as np

import openmdao.api as om
from openmdao.utils.array_utils import evenly_distrib_idxs


class DistParab(om.ExplicitComponent):

    def initialize(self):
        self.options['distributed'] = True

        self.options.declare('arr_size', types=int, default=10,
                             desc="Size of input and output vectors.")

    def setup(self):
        arr_size = self.options['arr_size']
        comm = self.comm
        rank = comm.rank

        sizes, offsets = evenly_distrib_idxs(comm.size, arr_size)
        start = offsets[rank]
        self.io_size = sizes[rank]
        self.offset = offsets[rank]
        end = start + self.io_size

        self.add_input('x', val=np.ones(self.io_size),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('y', val=np.ones(self.io_size),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('a', val=-3.0 * np.ones(self.io_size),
                       src_indices=np.arange(start, end, dtype=int))

        self.add_output('f_xy', val=np.ones(self.io_size))

        self.declare_partials('f_xy', ['x', 'y'], method='fd')

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        a = inputs['a']

        outputs['f_xy'] = (x + a)**2 + x * y + (y + 4.0)**2 - 3.0

    def compute_partials(self, inputs, partials):
        pass
        x = inputs['x']
        y = inputs['y']
        a = inputs['a']

        partials['f_xy', 'x'] = np.diag(2.0*x + 2.0 * a + y)
        partials['f_xy', 'y'] = np.diag(2.0*y + 8.0 + x)