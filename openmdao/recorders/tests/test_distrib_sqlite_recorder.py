import errno
import os
import unittest
from time import time
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np

from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.mpi import MPI

from openmdao.api import ExecComp, ExplicitComponent, Problem, \
    Group, ParallelGroup, IndepVarComp, SqliteRecorder, ScipyOptimizeDriver
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.recorders.tests.sqlite_recorder_test_utils import \
    assertDriverIterDataRecorded, assertProblemDataRecorded
from openmdao.recorders.tests.recorder_test_utils import run_driver

if MPI:
    from openmdao.api import PETScVector
else:
    PETScVector = None


class DistributedAdder(ExplicitComponent):
    """
    Distributes the work of adding 10 to every item in the param vector
    """

    def __init__(self, size):
        super().__init__()

        self.options['distributed'] = True

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

        # NOTE: Each process will get just its local part of the vector
        # print('process {0:d}: {1}'.format(self.comm.rank, params['x'].shape))

        outputs['y'] = inputs['x'] + 10.


class Summer(ExplicitComponent):
    """
    Aggregation component that collects all the values from the distributed
    vector addition and computes a total
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def setup(self):
        # NOTE: this component depends on the full y array, so OpenMDAO
        #       will automatically gather all the values for it
        self.add_input('y', val=np.zeros(self.size))
        self.add_output('sum', 0.0, shape=1)

    def compute(self, inputs, outputs):
        outputs['sum'] = np.sum(inputs['y'])


class Mygroup(Group):

    def setup(self):
        self.add_subsystem('indep_var_comp', IndepVarComp('x'), promotes=['*'])
        self.add_subsystem('Cy', ExecComp('y=2*x'), promotes=['*'])
        self.add_subsystem('Cc', ExecComp('c=x+2'), promotes=['*'])

        self.add_design_var('x')
        self.add_constraint('c', lower=-3.)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class DistributedRecorderTest(unittest.TestCase):

    N_PROCS = 2

    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "sqlite_test")
        self.recorder = SqliteRecorder(self.filename)
        self.eps = 1e-5

    def tearDown(self):
        try:
            rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def test_distrib_record_system(self):
        prob = Problem()

        try:
            prob.model.add_recorder(self.recorder)
        except RuntimeError as err:
            msg = "<class Group>: Recording of Systems when running parallel code is not supported yet"
            self.assertEqual(str(err), msg)
        else:
            self.fail('RuntimeError expected.')

    def test_distrib_record_solver(self):
        prob = Problem()
        try:
            prob.model.nonlinear_solver.add_recorder(self.recorder)
        except RuntimeError as err:
            msg = "Recording of Solvers when running parallel code is not supported yet"
            self.assertEqual(str(err), msg)
        else:
            self.fail('RuntimeError expected.')

    def test_distrib_record_driver(self):
        size = 100  # how many items in the array
        prob = Problem()

        prob.model.add_subsystem('des_vars', IndepVarComp('x', np.ones(size)), promotes=['x'])
        prob.model.add_subsystem('plus', DistributedAdder(size), promotes=['x', 'y'])
        prob.model.add_subsystem('summer', Summer(size), promotes=['y', 'sum'])
        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['includes'] = ['y']
        prob.driver.add_recorder(self.recorder)

        prob.model.add_design_var('x')
        prob.model.add_objective('sum')

        prob.setup()

        prob['x'] = np.ones(size)

        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0,)]

        expected_desvars = {
            "des_vars.x": prob['des_vars.x'],
        }

        expected_objectives = {
            "summer.sum": prob['summer.sum'],
        }

        expected_outputs = expected_desvars.copy()
        expected_outputs['plus.y'] = prob.get_val('plus.y', get_remote=True)

        if prob.comm.rank == 0:
            expected_outputs.update(expected_objectives)

            expected_data = ((coordinate, (t0, t1), expected_outputs, None, None),)
            assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_recording_remote_voi(self):
        # Create a parallel model
        model = Group()

        model.add_subsystem('par', ParallelGroup())
        model.par.add_subsystem('G1', Mygroup())
        model.par.add_subsystem('G2', Mygroup())
        model.connect('par.G1.y', 'Obj.y1')
        model.connect('par.G2.y', 'Obj.y2')

        model.add_subsystem('Obj', ExecComp('obj=y1+y2'))
        model.add_objective('Obj.obj')

        # Configure driver to record VOIs on both procs
        driver = ScipyOptimizeDriver(disp=False)

        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['includes'] = ['par.G1.y', 'par.G2.y']

        driver.add_recorder(self.recorder)

        # Create problem and run driver
        prob = Problem(model, driver)
        prob.add_recorder(self.recorder)
        prob.setup(mode='fwd')

        t0, t1 = run_driver(prob)
        prob.record('final')
        t2 = time()

        prob.cleanup()

        # Since the test will compare the last case recorded, just check the
        # current values in the problem. This next section is about getting those values

        # These involve collective gathers so all ranks need to run this
        expected_outputs = driver.get_design_var_values(get_remote=True)
        expected_outputs.update(driver.get_objective_values())
        expected_outputs.update(driver.get_constraint_values())

        # includes for outputs are specified as promoted names but we need absolute names
        prom2abs = model._var_allprocs_prom2abs_list['output']
        abs_includes = [prom2abs[n][0] for n in prob.driver.recording_options['includes']]

        # Absolute path names of includes on this rank
        rrank = model.comm.rank
        rowned = model._owning_rank
        local_includes = [n for n in abs_includes if rrank == rowned[n]]

        # Get values for all vars on this rank
        inputs, outputs, residuals = model.get_nonlinear_vectors()

        # Get values for includes on this rank
        local_vars = {n: outputs[n] for n in local_includes}

        # Gather values for includes on all ranks
        all_vars = model.comm.gather(local_vars, root=0)

        if prob.comm.rank == 0:
            # Only on rank 0 do we have all the values. The all_vars variable is a list of
            # dicts from all ranks 0,1,... In this case, just ranks 0 and 1
            dct = {}
            for d in all_vars:
                dct.update(d)

            expected_includes = {
                'par.G1.Cy.y': dct['par.G1.Cy.y'],
                'par.G2.Cy.y': dct['par.G2.Cy.y'],
            }

            expected_outputs.update(expected_includes)

            coordinate = [0, 'ScipyOptimize_SLSQP', (driver.iter_count-1,)]

            expected_data = ((coordinate, (t0, t1), expected_outputs, None, None),)
            assertDriverIterDataRecorded(self, expected_data, self.eps)

            expected_data = (('final', (t1, t2), expected_outputs),)
            assertProblemDataRecorded(self, expected_data, self.eps)


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
