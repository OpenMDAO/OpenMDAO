from pprint import pprint
import numpy as np
import openmdao.api as om
from mpi4py import MPI

from openmdao.utils.assert_utils import assert_near_equal

class DistribParaboloid(om.ExplicitComponent):

    def setup(self):
        self.options['distributed'] = True

        if self.comm.rank == 0:
            ndvs = 3
        else:
            ndvs = 2

        self.add_input('w', val=1.) # this will connect to a non-distributed IVC
        self.add_input('x', shape=ndvs) # this will connect to a distributed IVC

        self.add_output('y', shape=1) # all-gathered output, duplicated on all procs
        self.add_output('z', shape=ndvs) # distributed output
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
    else:
        ndvs = 2
    d_ivc.add_output('x', 2*np.ones(ndvs))

    ivc = p.model.add_subsystem('ivc',
                                om.IndepVarComp(distributed=False),
                                promotes=['*'])
    ivc.add_output('w', 2.0)
    p.model.add_subsystem('dp', DistribParaboloid(), promotes=['*'])

    p.model.add_design_var('x', lower=-100, upper=100)
    p.model.add_objective('y')


    p.setup()
    p.run_model()

    dv_vals = p.driver.get_design_var_values(get_remote=False)

    # Compute totals and check the length of the gradient array on each proc
    objcongrad = p.compute_totals(get_remote=True)
    print("Rank {0}: Length of dy/dx = {1}".format(comm.rank, len(objcongrad[('dp.y', 'distrib_ivc.x')][0])))
    print("Rank {0}: Length of dy/dx should be = {1}".format(comm.rank, ndvs))

    # Check the values of the gradient array
    assert_near_equal(objcongrad[('dp.y', 'distrib_ivc.x')][0], -6.0*np.ones(ndvs))