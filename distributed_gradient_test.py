from pprint import pprint
import numpy as np
import openmdao.api as om
from mpi4py import MPI
import wingdbstub

from openmdao.utils.assert_utils import assert_near_equal

class DistribParaboloid(om.ExplicitComponent):

    def setup(self):
        
        comm = self.comm
        rank = comm.rank
        
        if rank == 0:
            ndvs = 3
            two_d = (3,3)
            start = 0
            end = 9
        else:
            ndvs = 2
            two_d = (2,2) 
            start = 9
            end = 13
            
        self.options['distributed'] = True

        self.add_input('w', val=1., src_indices=np.array([1])) # this will connect to a non-distributed IVC
        self.add_input('x', shape=two_d, src_indices=np.arange(start, end, dtype=int).reshape(two_d), flat_src_indices=True) # this will connect to a distributed IVC

        self.add_output('y', shape=two_d) # all-gathered output, duplicated on all procs
        self.add_output('z', shape=two_d) # distributed output
        self.declare_partials('y', 'x')
        self.declare_partials('y', 'w')
        self.declare_partials('z', 'x')

    def compute(self, inputs, outputs):
        x = inputs['x']
        local_y = np.sum((x-5)**2)
        y_g = np.zeros(self.comm.size)
        self.comm.Allgather(local_y, y_g)
        outputs['y'] = np.sum(y_g) + (inputs['w']-10)**2
        outputs['z'] = x**2

    def compute_partials(self, inputs, J):
        x = inputs['x']
        J['y', 'x'] = 2*(x-5)
        J['y', 'w'] = 2*(inputs['w']-10)
        J['z', 'x'] = np.diag(2*x)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    p = om.Problem()
    d_ivc = p.model.add_subsystem('distrib_ivc',
                                   om.IndepVarComp(distributed=True),
                                   promotes=['*'])
    if comm.rank == 0:
        ndvs = 3
        two_d = (3,3)
    else:
        ndvs = 2
        two_d = (2,2)

    d_ivc.add_output('x', 2*np.ones(two_d))

    ivc = p.model.add_subsystem('ivc',
                                om.IndepVarComp(distributed=False),
                                promotes=['*'])
    ivc.add_output('w', 2.0)
    p.model.add_subsystem('dp', DistribParaboloid(), promotes=['*'])

    p.model.add_design_var('x', lower=-100, upper=100)
    p.model.add_objective('y')

    #import wingdbstub

    p.setup()
    p.run_model()

    dv_vals = p.driver.get_design_var_values(get_remote=False)

    # Compute totals and check the length of the gradient array on each proc
    objcongrad = p.compute_totals(get_remote=False)
    print("Rank {0}: Length of dy/dx = {1}".format(comm.rank, len(objcongrad[('dp.y', 'distrib_ivc.x')][0])))
    print("Rank {0}: Length of dy/dx should be = {1}".format(comm.rank, ndvs))

    # Check the values of the gradient array
    #print
    assert_near_equal(objcongrad[('dp.y', 'distrib_ivc.x')][0], -6.0*np.ones(ndvs))