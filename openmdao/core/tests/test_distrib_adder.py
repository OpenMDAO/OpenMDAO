import os

import unittest
import numpy as np

from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp

from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.utils.mpi import MPI

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from openmdao.utils.assert_utils import assert_near_equal


class DistributedAdder(ExplicitComponent):
    """
    Distributes the work of adding 10 to every item in the param vector
    """

    def initialize(self):
        self.options['distributed'] = True

        self.options.declare('size', types=int, default=1,
                             desc="Size of input and output vectors.")

    def setup(self):
        """
        specify the local sizes of the variables and which specific indices this specific
        distributed component will handle. Indices do NOT need to be sequential or
        contiguous!
        """
        comm = self.comm
        rank = comm.rank

        # NOTE: evenly_distrib_idxs is a helper function to split the array
        #       up as evenly as possible
        sizes, offsets = evenly_distrib_idxs(comm.size, self.options['size'])
        local_size, local_offset = sizes[rank], offsets[rank]

        start = local_offset
        end = local_offset + local_size

        self.add_input('x', val=np.zeros(local_size, float),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_output('y', val=np.zeros(local_size, float))

    def compute(self, inputs, outputs):

        # NOTE: Each process will get just its local part of the vector
        # print('process {0:d}: {1}'.format(self.comm.rank, params['x'].shape))

        outputs['y'] = inputs['x'] + 10.


class Summer(ExplicitComponent):
    """
    Aggregation component that collects all the values from the distributed
    vector addition and computes a total
    """

    def initialize(self):
        self.options.declare('size', types=int, default=1,
                             desc="Size of input and output vectors.")

    def setup(self):
        # NOTE: this component depends on the full y array, so OpenMDAO
        #       will automatically gather all the values for it
        self.add_input('y', val=np.zeros(self.options['size']))
        self.add_output('sum', 0.0, shape=1)

    def compute(self, inputs, outputs):
        outputs['sum'] = np.sum(inputs['y'])


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
@unittest.skipIf(os.environ.get("TRAVIS"), "Unreliable on Travis CI.")
class DistributedAdderTest(unittest.TestCase):

    N_PROCS = 3

    def test_distributed_adder(self):
        size = 100  # how many items in the array

        prob = Problem()

        prob.model.add_subsystem('des_vars', IndepVarComp('x', np.ones(size)), promotes=['x'])
        prob.model.add_subsystem('plus', DistributedAdder(size=size), promotes=['x', 'y'])
        summer = prob.model.add_subsystem('summer', Summer(size=size), promotes=['y', 'sum'])

        prob.setup()

        prob['x'] = np.ones(size)

        prob.run_driver()

        inp = summer._inputs['y']
        for i in range(size):
            diff = 11.0 - inp[i]
            if diff > 1.e-6 or diff < -1.e-6:
                raise RuntimeError("Summer input y[%d] is %f but should be 11.0" %
                                   (i, inp[i]))

        assert_near_equal(prob['sum'], 11.0 * size, 1.e-6)


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
