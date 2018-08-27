
from __future__ import division, print_function
import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent


class MatMultComp(ExplicitComponent):
    """
    A simple component used for derivative testing.
    """
    def __init__(self, mat, **kwargs):
        """
        Store the mat for later use.

        Parameters
        ----------
        mat : ndarray
            Matrix used to multiply input x to get output y.
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        super(MatMultComp, self).__init__(**kwargs)
        self.mat = mat

    def setup(self):
        self.add_input('x', val=np.ones(self.mat.shape[1]))
        self.add_output('y', val=np.zeros(self.mat.shape[0]))

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """
        Multiply input by mat to get output.
        """
        outputs['y'] = self.mat.dot(inputs['x'])

    def compute_partials(self, inputs, partials):
        """
        Jacobian for our paraboloid.
        """
        partials['y', 'x'] = self.mat


if __name__ == '__main__':
    import sys
    from openmdao.api import Problem, IndepVarComp
    from openmdao.utils.mpi import MPI
    
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    else:
        size = 100
    
    if MPI:
        ncom = MPI.COMM_WORLD.size
        if MPI.COMM_WORLD.rank == 0:
            mat = np.random.random(ncom * size).reshape((ncom, size)) - 0.5
        else:
            mat = None
        mat = MPI.COMM_WORLD.bcast(mat, root=0)

    p = Problem()
    model = p.model
    model.add_subsystem('indep', IndepVarComp('x', val=np.ones(mat.shape[1])))
    comp = model.add_subsystem('comp', MatMultComp(mat))
    comp.options['num_par_fd'] = 5
    
    model.connect('indep.x', 'comp.x')

    #import wingdbstub

    p.setup(mode='fwd', force_alloc_complex=True)
    p.final_setup()


    #p.check_partials(method='cs')
    J = p.compute_totals(of=['comp.y'], wrt=['indep.x'])
    print(J)

