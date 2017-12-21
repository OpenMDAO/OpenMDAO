import os

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

    def __init__(self, size):
        super(DistributedAdder, self).__init__()
        self.distributed = True

        self.local_size = self.size = size

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
        sizes, offsets = evenly_distrib_idxs(comm.size, self.size)
        local_size, local_offset = sizes[rank], offsets[rank]
        self.local_size = local_size

        start = local_offset
        end = local_offset + local_size

        self.add_input('x', val=np.zeros(local_size, float),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_output('y', val=np.zeros(local_size, float))

    def compute(self, inputs, outputs):

        #NOTE: Each process will get just its local part of the vector
        #print('process {0:d}: {1}'.format(self.comm.rank, params['x'].shape))

        outputs['y'] = inputs['x'] + 10.


class Summer(ExplicitComponent):
    """
    Aggregation component that collects all the values from the distributed
    vector addition and computes a total
    """

    def __init__(self, size):
        super(Summer, self).__init__()
        self.size = size

    def setup(self):
        #NOTE: this component depends on the full y array, so OpenMDAO
        #      will automatically gather all the values for it
        self.add_input('y', val=np.zeros(self.size))
        self.add_output('sum', 0.0, shape=1)

    def compute(self, inputs, outputs):
        outputs['sum'] = np.sum(inputs['y'])


@unittest.skipIf(PETScVector is None or os.environ.get("TRAVIS"),
                 "PETSc is required." if PETScVector is None
                 else "Unreliable on Travis CI.")
class DistributedAdderTest(unittest.TestCase):

    N_PROCS = 2

    def test_distributed_list_vars(self):

        import pydevd
        from openmdao.utils.mpi import MPI
        if MPI.COMM_WORLD.rank:
            pydevd.settrace('localhost', port=9876, stdoutToServer=True, stderrToServer=True)
        else:
            pydevd.settrace('localhost', port=9877, stdoutToServer=True, stderrToServer=True)

        size = 100 #how many items in the array

        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('des_vars', IndepVarComp('x', np.ones(size)), promotes=['x'])
        prob.model.add_subsystem('plus', DistributedAdder(size), promotes=['x', 'y'])
        summer = prob.model.add_subsystem('summer', Summer(size), promotes=['y', 'sum'])

        prob.setup(vector_class=PETScVector, check=False)

        prob['x'] = np.arange(size)

        prob.run_driver()

        from six.moves import cStringIO

        stream = cStringIO()

        inputs = prob.model.list_inputs(values=True, print_arrays=True, out_stream=stream)
        self.assertEqual(inputs[0][0], 'plus.x')
        self.assertEqual(inputs[1][0], 'summer.y')
        self.assertEqual(inputs[0][1]['value'].size, 50) # should only return the half that is local
        self.assertEqual(inputs[1][1]['value'].size, 100)

        text = stream.getvalue()
        if prob.comm.rank: # Only rank 0 prints
            self.assertEqual(len(text), 0)
        else:
            print(text)


        stream = cStringIO()
        outputs = prob.model.list_outputs(values=True,
                                          units=True,
                                          shape=True,
                                          bounds=True,
                                          residuals=True,
                                          scaling=True,
                                          hierarchical=True,
                                          print_arrays=True,
                                          out_stream=stream)
        self.assertEqual(outputs[0][0], 'des_vars.x')
        self.assertEqual(outputs[1][0], 'plus.y')
        self.assertEqual(outputs[2][0], 'summer.sum')
        self.assertEqual(outputs[0][1]['value'].size, 100)
        self.assertEqual(outputs[1][1]['value'].size, 50)
        self.assertEqual(outputs[2][1]['value'].size, 1)

        text = stream.getvalue()
        if prob.comm.rank: # Only rank 0 prints
            self.assertEqual(len(text), 0)
        else:
            print(text)

        # inp = summer._inputs['y']


        # for i in range(size):
        #     diff = 11.0 - inp[i]
        #     if diff > 1.e-6 or diff < -1.e-6:
        #         raise RuntimeError("Summer input y[%d] is %f but should be 11.0" %
        #                             (i, inp[i]))
        #
        # assert_rel_error(self, prob['sum'], 11.0 * size, 1.e-6)

    def test_list_vars_remote_voi(self):
        # import pydevd
        # from openmdao.utils.mpi import MPI
        # if MPI.COMM_WORLD.rank:
        #     pydevd.settrace('localhost', port=9876, stdoutToServer=True, stderrToServer=True)
        # else:
        #     pydevd.settrace('localhost', port=9877, stdoutToServer=True, stderrToServer=True)
        #
        from openmdao.utils.general_utils import set_pyoptsparse_opt
        from openmdao.utils.mpi import MPI

        if MPI:
            from openmdao.api import PETScVector
            vector_class = PETScVector
        else:
            PETScVector = None

        # check that pyoptsparse is installed. if it is, try to use SLSQP.
        OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')

        if OPTIMIZER:
            from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver

        from openmdao.core.parallel_group import ParallelGroup
        from openmdao.components.exec_comp import ExecComp

        class Mygroup(Group):

            def setup(self):
                self.add_subsystem('indep_var_comp', IndepVarComp('x'), promotes=['*'])
                self.add_subsystem('Cy', ExecComp('y=2*x'), promotes=['*'])
                self.add_subsystem('Cc', ExecComp('c=x+2'), promotes=['*'])

                self.add_design_var('x')
                self.add_constraint('c', lower=-3.)

        prob = Problem()

        prob.model.add_subsystem('par', ParallelGroup())

        prob.model.par.add_subsystem('G1', Mygroup())
        prob.model.par.add_subsystem('G2', Mygroup())

        prob.model.add_subsystem('Obj', ExecComp('obj=y1+y2'))

        prob.model.connect('par.G1.y', 'Obj.y1')
        prob.model.connect('par.G2.y', 'Obj.y2')

        prob.model.add_objective('Obj.obj')

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        # prob.driver.recording_options['record_desvars'] = True
        # prob.driver.recording_options['record_responses'] = True
        # prob.driver.recording_options['record_objectives'] = True
        # prob.driver.recording_options['record_constraints'] = True
        # prob.driver.recording_options['includes'] = ['par.G1.Cy.y','par.G2.Cy.y']
        #
        # prob.driver.add_recorder(self.recorder)

        prob.setup(vector_class=PETScVector)
        prob.run_driver()
        prob.cleanup()

        from six.moves import cStringIO

        stream = cStringIO()

        inputs = prob.model.list_inputs(values=True, print_arrays=True, out_stream=stream)
        text = stream.getvalue()
        if prob.comm.rank: # Only rank 0 prints
            self.assertEqual(len(text), 0)
        # else:
        #     print(text)
        #   Need some kind of check on the text here qqq TODO
        stream = cStringIO()
        outputs = prob.model.list_outputs(values=True,
                                          units=True,
                                          shape=True,
                                          bounds=True,
                                          residuals=True,
                                          scaling=True,
                                          hierarchical=True,
                                          print_arrays=True,
                                          out_stream=stream)
        # self.assertEqual(outputs[0][0], 'des_vars.x')
        # self.assertEqual(outputs[1][0], 'plus.y')
        # self.assertEqual(outputs[2][0], 'summer.sum')
        # self.assertEqual(outputs[0][1]['value'].size, 100)
        # self.assertEqual(outputs[1][1]['value'].size, 50)
        # self.assertEqual(outputs[2][1]['value'].size, 1)

        text = stream.getvalue()
        if prob.comm.rank: # Only rank 0 prints
            self.assertEqual(len(text), 0)
        else:
            print(text)

if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
