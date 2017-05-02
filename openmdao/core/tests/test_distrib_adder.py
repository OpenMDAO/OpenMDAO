
import unittest
import numpy as np

from openmdao.api import ExplicitComponent, Problem, Group, IndepVarComp

from openmdao.utils.mpi import MPI
from openmdao.utils.array_utils import evenly_distrib_idxs

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from openmdao.devtools.testutil import assert_rel_error


class DistributedAdder(ExplicitComponent):
    """
    Distributes the work of adding 10 to every item in the param vector
    """

    def __init__(self, size=100):
        super(DistributedAdder, self).__init__()

        self.local_size = self.size = int(size)

    def get_req_procs(self):
        """
        min/max number of procs that this component can use
        """
        return (1, self.size)

    def initialize_variables(self):
        """
        specify the local sizes of the variables and which specific indices this specific
        distributed component will handle. Indices do NOT need to be sequential or
        contiguous!
        """
        comm = self.comm
        rank = comm.rank

        # NOTE: evenly_distrib_idxs is a helper function to split the array
        #       up as evenly as possible
        sizes, offsets = evenly_distrib_idxs(comm.size, self.size)
        local_size, local_offset = sizes[rank], offsets[rank]
        self.local_size = int(local_size)

        start = local_offset
        end = local_offset + local_size

        self.add_input('x', val=np.zeros(local_size, float),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_output('y', val=np.zeros(local_size, float))
            #src_indices=np.arange(start, end, dtype=int))

    def compute(self, inputs, outputs):

        #NOTE: Each process will get just its local part of the vector
        #print('process {0:d}: {1}'.format(self.comm.rank, params['x'].shape))

        outputs['y'] = inputs['x'] + 10


class Summer(ExplicitComponent):
    """
    Agreggation component that collects all the values from the distributed
    vector addition and computes a total
    """

    def __init__(self, size=100):
        super(Summer, self).__init__()

        #NOTE: this component depends on the full y array, so OpenMDAO
        #      will automatically gather all the values for it
        self.add_input('y', val=np.zeros(size))
        self.add_output('sum', shape=1)

    def compute(self, inputs, outputs):

        outputs['sum'] = np.sum(inputs['y'])




@unittest.skipUnless(PETScVector, "PETSc is required.")
class DistributedAdderTest(unittest.TestCase):

    N_PROCS = 3

    def test_distributed_adder(self):
        size = 100000 #how many items in the array

        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('des_vars', IndepVarComp('x', np.ones(size)), promotes=['x'])
        prob.model.add_subsystem('plus', DistributedAdder(size), promotes=['x', 'y'])
        prob.model.add_subsystem('summer', Summer(size), promotes=['y', 'sum'])

        prob.setup(vector_class=PETScVector, check=False)

        prob['x'] = np.ones(size)

        prob.run_driver()

        #expected answer is 11
        assert_rel_error(self, prob['sum']/size, 11.0, 1.e-6)

        #from openmdao.devtools.debug import max_mem_usage
        #print "Max mem:", max_mem_usage()


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
