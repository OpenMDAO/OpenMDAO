import unittest
import numpy as np
from openmdao.utils.mpi import MPI
import traceback

from openmdao.api import Problem
from openmdao.api import ExplicitComponent, IndepVarComp
from openmdao.api import NonlinearRunOnce, LinearRunOnce

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class Comp(ExplicitComponent):
    def initialize(self):
        self.options['distributed'] = True
        self.options.declare('flat', False)

    def setup(self):
        irank = self.comm.Get_rank()
        if irank ==1:
            node_size = 0
        else:
            node_size = 3

        n_list = self.comm.allgather(node_size)
        n1 = np.sum(n_list[:irank])
        n2 = np.sum(n_list[:irank+1])

        self.add_input( 'x',shape=node_size, src_indices=np.arange(n1, n2, dtype=int),
                        flat_src_indices=self.options['flat'])
        self.add_output('y',shape=node_size)

    def compute(self,inputs,outputs):
        outputs['y'] = inputs['x'] + 1.0


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestSrcIndices(unittest.TestCase):
    N_PROCS = 4

    def test_zero_src_indices(self):
        prob = Problem()
        model = prob.model
        model.nonlinear_solver = NonlinearRunOnce()
        model.linear_solver = LinearRunOnce()

        model.add_subsystem('comp1',Comp())
        model.add_subsystem('comp2',Comp())
        model.connect('comp1.y','comp2.x')
        model.connect('comp2.y','comp1.x')

        prob.setup()
        prob.run_model()

        if model.comm.rank == 1:
            np.testing.assert_almost_equal(model.comp1._outputs['y'], np.array([]))
            np.testing.assert_almost_equal(model.comp2._outputs['y'], np.array([]))
        else:
            np.testing.assert_almost_equal(model.comp1._outputs['y'], np.ones(3) * 2.)
            np.testing.assert_almost_equal(model.comp2._outputs['y'], np.ones(3) * 3.)

    def test_zero_src_indices_flat(self):
        prob = Problem()
        model = prob.model
        model.nonlinear_solver = NonlinearRunOnce()
        model.linear_solver = LinearRunOnce()

        model.add_subsystem('comp1',Comp(flat=True))
        model.add_subsystem('comp2',Comp(flat=True))
        model.connect('comp1.y','comp2.x')
        model.connect('comp2.y','comp1.x')

        prob.setup()
        prob.run_model()

        if model.comm.rank == 1:
            np.testing.assert_almost_equal(model.comp1._outputs['y'], np.array([]))
            np.testing.assert_almost_equal(model.comp2._outputs['y'], np.array([]))
        else:
            np.testing.assert_almost_equal(model.comp1._outputs['y'], np.ones(3) * 2.)
            np.testing.assert_almost_equal(model.comp2._outputs['y'], np.ones(3) * 3.)

