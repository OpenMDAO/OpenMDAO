import numpy as np

from openmdao.api import Component, Problem, Group, IndepVarComp

from openmdao.core.mpi_wrap import MPI
from openmdao.util.array_util import evenly_distrib_idxs
from openmdao.test.mpi_util import MPITestCase

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.api import BasicImpl as impl

from openmdao.test.util import assert_rel_error


class DistributedAdder(Component):
    """
    Distributes the work of adding 10 to every item in the param vector
    """

    def __init__(self, size=100):
        super(DistributedAdder, self).__init__()

        self.local_size = self.size = int(size)

        #NOTE: we declare the variables at full size so that the component will work in serial too
        self.add_param('x', shape=size)
        self.add_output('y', shape=size)

    def get_req_procs(self):
        """
        min/max number of procs that this component can use
        """
        return (1,self.size)

    def setup_distrib(self):
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

        self.set_var_indices('x', val=np.zeros(local_size, float),
            src_indices=np.arange(start, end, dtype=int))
        self.set_var_indices('y', val=np.zeros(local_size, float),
            src_indices=np.arange(start, end, dtype=int))

        #quit()

    def solve_nonlinear(self, params, unknowns, resids):

        #NOTE: Each process will get just its local part of the vector
        #print('process {0:d}: {1}'.format(self.comm.rank, params['x'].shape))

        unknowns['y'] = params['x'] + 10


class Summer(Component):
    """
    Agreggation component that collects all the values from the distributed
    vector addition and computes a total
    """

    def __init__(self, size=100):
        super(Summer, self).__init__()

        #NOTE: this component depends on the full y array, so OpenMDAO
        #      will automatically gather all the values for it
        self.add_param('y', val=np.zeros(size))
        self.add_output('sum', shape=1)

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['sum'] = np.sum(params['y'])




class DistributedAdderTest(MPITestCase):

    N_PROCS = 10

    def test_distributed_adder(self):
        size = 1000000 #how many items in the array

        prob = Problem(impl=impl)
        prob.root = Group()

        prob.root.add('des_vars', IndepVarComp('x', np.ones(size)), promotes=['x'])
        prob.root.add('plus', DistributedAdder(size), promotes=['x', 'y'])
        prob.root.add('summer', Summer(size), promotes=['y', 'sum'])

        prob.setup(check=False)

        prob['x'] = np.ones(size)

        prob.run()

        #expected answer is 11
        assert_rel_error(self, prob['sum']/size, 11.0, 1.e-6)



if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    mpirun_tests()
