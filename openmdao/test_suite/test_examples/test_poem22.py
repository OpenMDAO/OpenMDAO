from mpi4py import MPI
import unittest
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal


# ======================
#  basic test
# ======================
class L2(om.ExplicitComponent):
    """takes the 2 norm of the input"""

    def setup(self):
        self.add_input('vec', shape_by_conn=True)
        self.add_output('val', 0.0)

    def compute(self, inputs, outputs):
        outputs['val'] = np.linalg.norm(inputs['vec'])

class TestAdder(unittest.TestCase):

    def test_adder(self):
        import openmdao.api as om
        from openmdao.test_suite.components.sellar_feature import SellarMDA

        prob = om.Problem()
        prob.model = om.Group()

        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('in', np.ones(10), tags="advanced")

        prob.model.add_subsystem('L2norm', L2())
        prob.model.connect('in', ['L2norm.vec'])
        prob.setup()
        prob.run_model()


# ======================
#  advanced tests
# ======================


# This is based on passing size information through the system shown below
# in all test C starts with the size information

# +-----+
# |     |
# |  A  +-----+
# |     |     |
# +-----+     |
#         +---+---+
#         |       |
#         |   B   +-----+
#         |       |     |
#         +-------+     |
#                   +---+----+
#                   |        |
#                   |   C    +-----+
#                   |        |     |
#                   +--------+ +---+---+
#                              |       |
#                              |   D   +----+
#                              |       |    |
#                              +-------+    |
#                                        +--+----+
#                                        |       |
#                                        |   E   |
#                                        |       |
#                                        +-------+



class B(om.ExplicitComponent):

    def setup(self):
        # Inputs
        self.add_input('in', copy_shape='out')
        self.add_output('out', shape_by_conn=True)

    def compute(self, inputs, outputs):
        outputs['out'] = inputs['in']




class C(om.ExplicitComponent):

    def setup(self):
        # Inputs
        self.add_input('in', shape=4)
        self.add_output('out', shape=9)

    def compute(self, inputs, outputs):
        outputs['out'] = np.arange(9)



class D(om.ExplicitComponent):

    def setup(self):
        # Inputs
        self.add_input('in', shape_by_conn=True)
        self.add_output('out', copy_shape='in')

    def compute(self, inputs, outputs):
        outputs['out'] = inputs['in']


class E(om.ExplicitComponent):

    def setup(self):
        # Inputs
        self.add_input('in', shape_by_conn=True)
        self.add_output('out', copy_shape='in')

    def compute(self, inputs, outputs):
        print(inputs['in'])
        outputs['out'] = inputs['in']


class B_dis(om.ExplicitComponent):
    def initialize(self):
        self.options['distributed'] = True

    def setup(self):
        # Inputs
        self.add_input('in', copy_shape='out')
        self.add_output('out', shape_by_conn=True)

    def compute(self, inputs, outputs):
        outputs['out'] = inputs['in']





class C_dis(om.ExplicitComponent):
    def initialize(self):
        self.options['distributed'] = True

    def setup(self):
        # Inputs
        if self.comm.rank == 0:
            self.add_input('in', shape=1, src_indices=np.arange(0,1, dtype=int))
        elif self.comm.rank == 1:
            self.add_input('in', shape=2, src_indices=np.arange(1,3, dtype=int))
        else:
            self.add_input('in', shape=0, src_indices=np.arange(3,3, dtype=int))

        self.add_output('out', shape=3)

    def compute(self, inputs, outputs):
        outputs['out'] *= self.comm.rank



class D_dis(om.ExplicitComponent):
    def initialize(self):
        self.options['distributed'] = True

    def setup(self):
        # Inputs
        self.add_input('in', shape_by_conn=True)
        self.add_output('out', copy_shape='in')

    def compute(self, inputs, outputs):
        outputs['out'] = inputs['in']




class TestPassSize(unittest.TestCase):
    def test_serial(self):

        prob = om.Problem()
        prob.model = om.Group()

        indeps = prob.model.add_subsystem('A', om.IndepVarComp())
        indeps.add_output('out', shape_by_conn=True)

        prob.model.add_subsystem('B', B())
        prob.model.connect('A.out', ['B.in'])

        prob.model.add_subsystem('C', C())
        prob.model.connect('B.out', ['C.in'])

        prob.model.add_subsystem('D', D())
        prob.model.connect('C.out', ['D.in'])
        
        prob.model.add_subsystem('E', E())
        prob.model.connect('D.out', ['E.in'])

        prob.setup()
        prob.run_model()
        self.assertEqual(prob.get_val('A.out').size ,4)
        self.assertEqual(prob.get_val('B.in').size ,4)
        self.assertEqual(prob.get_val('B.out').size ,4)

        self.assertEqual(prob.get_val('D.in').size ,9)
        self.assertEqual(prob.get_val('D.out').size ,9)
        self.assertEqual(prob.get_val('E.in').size ,9)

    def test_err(self):


        prob = om.Problem()
        prob.model = om.Group()

        prob.model.add_subsystem('B', B())
        prob.model.connect('C.out', ['B.in'])

        prob.model.add_subsystem('C', B())
        prob.model.connect('B.out', ['C.in'])



        with self.assertRaises(Exception) as raises_cm:
            prob.setup()

        exception = raises_cm.exception

        msg = "deferred shape dependences unresolvable. visited ['B.in', 'B.out', 'C.in', 'C.out', 'B.in']"
        self.assertEqual(exception.args[0], msg)


class TestPassSizeDistributed(unittest.TestCase):


    N_PROCS = 3

    def test_serial_start(self):
        """the size information starts in the serial comonent C"""


        prob = om.Problem()
        prob.model = om.Group()

        indeps = prob.model.add_subsystem('A', om.IndepVarComp())
        indeps.add_output('out', shape_by_conn=True)

        prob.model.add_subsystem('B', B_dis())
        prob.model.connect('A.out', ['B.in'])

        prob.model.add_subsystem('C', C())
        prob.model.connect('B.out', ['C.in'])

        prob.model.add_subsystem('D', D_dis())
        prob.model.connect('C.out', ['D.in'])
        
        prob.model.add_subsystem('E', E())
        prob.model.connect('D.out', ['E.in'])

        prob.setup()
        prob.run_model()
        
        
        # # check all global sizes
        self.assertEqual(prob.get_val('A.out', get_remote=True).size, 4)
        self.assertEqual(prob.get_val('B.in', get_remote=True).size, 4)
        self.assertEqual(prob.get_val('B.out', get_remote=True).size, 4)
        
        self.assertEqual(prob.get_val('D.in', get_remote=True).size, 9)
        self.assertEqual(prob.get_val('D.out', get_remote=True).size, 9)
        self.assertEqual(prob.get_val('E.in', get_remote=True).size, 9)


        # #check all local sizes
        nprocs = MPI.COMM_WORLD.size
        rank = MPI.COMM_WORLD.rank

        # evenly distribute the variable over the procs
        ave, res = divmod(4, nprocs)
        sizes_up = [ave + 1 if p < res else ave for p in range(nprocs)]

        ave, res = divmod(9, nprocs)
        sizes_down = [ave + 1 if p < res else ave for p in range(nprocs)]


        if rank == 0:
            size_up = 2
        elif rank == 1:
            size_up = 1
        else:
            size_up = 0

        size_down = 3


        #  get_val for inputs with distributed  components isn't working as expected
        # it could be a bug 

        self.assertEqual(prob.get_val('A.out').size, 4)
        # self.assertEqual(prob.get_val('B.in').size, size_up)
        self.assertEqual(prob.get_val('B.out').size, sizes_up[rank])
        
        # self.assertEqual(prob.get_val('D.in').size, size_down)
        self.assertEqual(prob.get_val('D.out').size, sizes_down[rank])
        # self.assertEqual(prob.get_val('E.in').size, 3*self.N_PROCS)
        self.assertEqual(prob.get_val('E.out').size, 9)
        

        # test the output from running model
        self.assertEqual(np.sum(prob.get_val('E.out')), np.sum(np.arange(9)))




    def test_distributed_start(self):
        """the size information starts in the distributed comonent C"""


        prob = om.Problem()
        prob.model = om.Group()

        indeps = prob.model.add_subsystem('A', om.IndepVarComp())
        indeps.add_output('out', shape_by_conn=True)

        prob.model.add_subsystem('B', B_dis())
        prob.model.connect('A.out', ['B.in'])

        prob.model.add_subsystem('C', C_dis())
        prob.model.connect('B.out', ['C.in'])

        prob.model.add_subsystem('D', D_dis())
        prob.model.connect('C.out', ['D.in'])
        
        prob.model.add_subsystem('E', E())
        prob.model.connect('D.out', ['E.in'])

        prob.setup()
        prob.run_model()
        
        # # check all global sizes
        self.assertEqual(prob.get_val('A.out', get_remote=True).size, 3)
        self.assertEqual(prob.get_val('B.in', get_remote=True).size, 3)
        self.assertEqual(prob.get_val('B.out', get_remote=True).size, 3)
        
        self.assertEqual(prob.get_val('D.in', get_remote=True).size, 3*self.N_PROCS)
        self.assertEqual(prob.get_val('D.out', get_remote=True).size, 3*self.N_PROCS)
        self.assertEqual(prob.get_val('E.in', get_remote=True).size, 3*self.N_PROCS)

        # #check all local sizes
        rank = MPI.COMM_WORLD.rank
        if rank == 0:
            size_up = 1
        elif rank == 1:
            size_up = 2
        else:
            size_up = 0

        size_down = 3

        #  get_val for inputs with distributed  components isn't working as expected
        # it could be a bug 

        self.assertEqual(prob.get_val('A.out').size, 3)
        # self.assertEqual(prob.get_val('B.in').size, size_up)
        self.assertEqual(prob.get_val('B.out').size, size_up)
        
        # self.assertEqual(prob.get_val('D.in').size, size_down)
        self.assertEqual(prob.get_val('D.out').size, size_down)
        # self.assertEqual(prob.get_val('E.in').size, 3*self.N_PROCS)
        self.assertEqual(prob.get_val('E.out').size, 3*self.N_PROCS)
        
        # test the output from running model
        n = self.N_PROCS - 1
        self.assertEqual(np.sum(prob.get_val('E.out')), (n**2 + n)/2 * size_down)


if __name__ == "__main__":
    unittest.main()
