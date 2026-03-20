"""
A simple component used for derivative testing.
"""
import time

import numpy as np

import openmdao.api as om


class MatMultComp(om.ExplicitComponent):
    def __init__(self, mat, approx_method='exact', sleep_time=0.1, add_z=False, **kwargs):
        super().__init__(**kwargs)
        self.mat = mat
        self.approx_method = approx_method
        self.sleep_time = sleep_time
        self.add_z = add_z

    def setup(self):
        self.add_input('x', val=np.ones(self.mat.shape[1]))
        if self.add_z:
            self.add_input('z', val=np.ones(self.mat.shape[1]))
        self.add_output('y', val=np.zeros(self.mat.shape[0]))
        self.num_computes = 0

    def setup_partials(self):
        # rel_element step_calc is only valid for finite difference, not complex step
        kwargs = {'method': self.approx_method}
        if self.approx_method == 'fd':
            kwargs['step'] = 1.E-3
            kwargs['step_calc'] = 'rel_element'
        self.declare_partials('*', '*', **kwargs)

    def compute(self, inputs, outputs):
        if self.add_z:
            outputs['y'] = self.mat.dot(inputs['x'] * inputs['z'])
        else:
            outputs['y'] = self.mat.dot(inputs['x'])
        self.num_computes += 1
        time.sleep(self.sleep_time)


if __name__ == '__main__':
    import sys

    import openmdao.api as om
    from openmdao.utils.mpi import MPI

    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    else:
        size = 20

    if MPI:
        ncom = MPI.COMM_WORLD.size
        if MPI.COMM_WORLD.rank == 0:
            mat = np.random.random(5 * size).reshape((5, size)) - 0.5
        else:
            mat = None
        mat = MPI.COMM_WORLD.bcast(mat, root=0)
        profname = "prof_%d.out" % MPI.COMM_WORLD.rank
    else:
        mat = np.random.random(5 * size).reshape((5, size)) - 0.5
        profname = "prof.out"

    print("mat shape:", mat.shape)

    p = om.Problem()
    model = p.model
    model.add_subsystem('indep', om.IndepVarComp('x', val=np.ones(mat.shape[1])))
    comp = model.add_subsystem('comp', MatMultComp(mat, approx_method='fd', num_par_fd=5))

    model.connect('indep.x', 'comp.x')

    p.setup(mode='fwd', force_alloc_complex=True)
    p.run_model()

    start = time.perf_counter()
    J = p.compute_totals(of=['comp.y'], wrt=['indep.x'])

    print("norm J - mat:", np.linalg.norm(J['comp.y','indep.x'] - comp.mat))
    print("Elapsed time:", time.perf_counter() - start)

