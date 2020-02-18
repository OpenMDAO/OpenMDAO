"""
Distributed components.

Components that are used in multiple places for testing distributed components.
"""
import numpy as np
import openmdao.api as om

from openmdao.utils.mpi import MPI
from openmdao.utils.array_utils import evenly_distrib_idxs

class DistribComp(om.ExplicitComponent):
    def initialize(self):
        self.options['distributed'] = True

        self.options.declare('size', types=int, default=1,
                             desc="Size of input and output vectors.")

    def setup(self):
        comm = self.comm
        rank = comm.rank

        size = self.options['size']

        # results in 8 entries for proc 0 and 7 entries for proc 1 when using 2 processes.
        sizes, offsets = evenly_distrib_idxs(comm.size, size)
        start = offsets[rank]
        end = start + sizes[rank]

        self.add_input('invec', np.ones(sizes[rank], float),
                       src_indices=np.arange(start, end, dtype=int))

        self.add_output('outvec', np.ones(sizes[rank], float))

    def compute(self, inputs, outputs):
        if self.comm.rank == 0:
            outputs['outvec'] = inputs['invec'] * 2.0
        else:
            outputs['outvec'] = inputs['invec'] * -3.0

class Summer(om.ExplicitComponent):
    """Sums a distributed input."""

    def initialize(self):
        self.options.declare('size', types=int, default=1,
                             desc="Size of input and output vectors.")

    def setup(self):
        comm = self.comm
        rank = comm.rank

        size = self.options['size']

        # this results in 8 entries for proc 0 and 7 entries for proc 1
        # when using 2 processes.
        sizes, offsets = evenly_distrib_idxs(comm.size, size)
        start = offsets[rank]
        end = start + sizes[rank]

        # NOTE: you must specify src_indices here for the input. Otherwise,
        #       you'll connect the input to [0:local_input_size] of the
        #       full distributed output!
        self.add_input('invec', np.ones(sizes[rank], float),
                       src_indices=np.arange(start, end, dtype=int))

        self.add_output('out', 0.0)

    def compute(self, inputs, outputs):
        data = np.zeros(1)
        data[0] = np.sum(inputs['invec'])

        total = np.zeros(1)
        self.comm.Allreduce(data, total, op=MPI.SUM)

        outputs['out'] = total[0]
