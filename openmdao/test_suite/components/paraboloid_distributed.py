""" A distributed version of the paraboloid model with an extra input that can be used to shift
each index.

This version is used for testing, so it will have different options.
"""
import numpy as np

import openmdao.api as om
from openmdao.utils.array_utils import evenly_distrib_idxs


class DistParab(om.ExplicitComponent):

    def initialize(self):
        self.options['distributed'] = True

        self.options.declare('arr_size', types=int, default=10,
                             desc="Size of input and output vectors.")

        self.options.declare('deriv_type', default='dense',
                             values=['dense', 'fd', 'cs', 'sparse'],
                             desc="Method for computing derivatives.")

    def setup(self):
        arr_size = self.options['arr_size']
        deriv_type = self.options['deriv_type']
        comm = self.comm
        rank = comm.rank

        sizes, offsets = evenly_distrib_idxs(comm.size, arr_size)
        start = offsets[rank]
        io_size = sizes[rank]
        self.offset = offsets[rank]
        end = start + io_size

        self.add_input('x', val=np.ones(io_size),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('y', val=np.ones(io_size),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('a', val=-3.0 * np.ones(io_size),
                       src_indices=np.arange(start, end, dtype=int))

        self.add_output('f_xy', val=np.ones(io_size))

        if deriv_type == 'dense':
            self.declare_partials('f_xy', ['x', 'y', 'a'])

        elif deriv_type == 'sparse':
            row_col = np.arange(io_size)
            self.declare_partials('f_xy', ['x', 'y', 'a'], rows=row_col, cols=row_col)

        else:
            self.declare_partials('f_xy', ['x', 'y', 'a'], method=deriv_type)

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        a = inputs['a']

        outputs['f_xy'] = (x + a)**2 + x * y + (y + 4.0)**2 - 3.0

    def compute_partials(self, inputs, partials):
        deriv_type = self.options['deriv_type']
        x = inputs['x']
        y = inputs['y']
        a = inputs['a']

        if deriv_type == 'dense':
            partials['f_xy', 'x'] = np.diag(2.0 * x + 2.0 * a + y)
            partials['f_xy', 'y'] = np.diag(2.0 * y + 8.0 + x)
            partials['f_xy', 'a'] = np.diag(2.0 * a + 2.0 * x)

        elif deriv_type == 'sparse':
            partials['f_xy', 'x'] = 2.0 * x + 2.0 * a + y
            partials['f_xy', 'y'] = 2.0 * y + 8.0 + x
            partials['f_xy', 'a'] = 2.0 * a + 2.0 * x


# Simplified version for feature docs without the extra testing args.

class DistParabFeature(om.ExplicitComponent):

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
        io_size = sizes[rank]
        self.offset = offsets[rank]
        end = start + io_size

        self.add_input('x', val=np.ones(io_size),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('y', val=np.ones(io_size),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('offset', val=-3.0 * np.ones(io_size),
                       src_indices=np.arange(start, end, dtype=int))

        self.add_output('f_xy', val=np.ones(io_size))

        row_col = np.arange(io_size)
        self.declare_partials('f_xy', ['x', 'y', 'offset'], rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        a = inputs['offset']

        outputs['f_xy'] = (x + a)**2 + x * y + (y + 4.0)**2 - 3.0

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        y = inputs['y']
        a = inputs['offset']

        partials['f_xy', 'x'] = 2.0 * x + 2.0 * a + y
        partials['f_xy', 'y'] = 2.0 * y + 8.0 + x
        partials['f_xy', 'offset'] = 2.0 * a + 2.0 * x