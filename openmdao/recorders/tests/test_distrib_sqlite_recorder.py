import errno
import os
import sqlite3
import unittest
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np

from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.mpi import MPI

if MPI:
    from openmdao.api import PETScVector
else:
    PETScVector = None

# check that pyoptsparse is installed. if it is, try to use SLSQP.
OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver

from openmdao.api import ExecComp, ExplicitComponent, Problem, \
    Group, ParallelGroup, IndepVarComp, SqliteRecorder
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.recorders.tests.sqlite_recorder_test_utils import assertDriverIterDataRecorded
from openmdao.recorders.tests.recorder_test_utils import run_driver


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


class Mygroup(Group):

    def setup(self):
        self.add_subsystem('indep_var_comp', IndepVarComp('x'), promotes=['*'])
        self.add_subsystem('Cy', ExecComp('y=2*x'), promotes=['*'])
        self.add_subsystem('Cc', ExecComp('c=x+2'), promotes=['*'])

        self.add_design_var('x')
        self.add_constraint('c', lower=-3.)


@unittest.skipIf(PETScVector is None, "PETSc is required.")
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
        prob.model = Group()

        try:
            prob.model.add_recorder(self.recorder)
        except RuntimeError as err:
            self.assertEqual(str(err), "Recording of Systems when running parallel code is not supported yet")
        else:
            self.fail('RuntimeError expected.')

    def test_distrib_record_solver(self):
        prob = Problem()
        prob.model = Group()
        try:
            prob.model.nonlinear_solver.add_recorder(self.recorder)
        except RuntimeError as err:
            self.assertEqual(str(err), "Recording of Solvers when running parallel code is not supported yet")
        else:
            self.fail('RuntimeError expected.')

    def test_distrib_record_driver(self):
        size = 100  # how many items in the array
        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('des_vars', IndepVarComp('x', np.ones(size)), promotes=['x'])
        prob.model.add_subsystem('plus', DistributedAdder(size), promotes=['x', 'y'])
        prob.model.add_subsystem('summer', Summer(size), promotes=['y', 'sum'])
        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['record_responses'] = True
        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['includes'] = []
        prob.driver.add_recorder(self.recorder)

        prob.model.add_design_var('x')
        prob.model.add_objective('sum')

        prob.setup(check=False)

        prob['x'] = np.ones(size)

        t0, t1 = run_driver(prob)
        prob.cleanup()

        if prob.comm.rank == 0:
            coordinate = [0, 'Driver', (0,)]

            expected_desvars = {
                "des_vars.x": prob['des_vars.x'],
            }

            expected_objectives = {
                "summer.sum": prob['summer.sum'],
            }

            expected_outputs = expected_desvars
            expected_outputs.update(expected_objectives)

            expected_data = ((coordinate, (t0, t1), expected_outputs, None),)
            assertDriverIterDataRecorded(self, expected_data, self.eps)

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP")
    def test_recording_remote_voi(self):
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

        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['record_responses'] = True
        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['includes'] = ['par.G1.Cy.y','par.G2.Cy.y']

        prob.driver.add_recorder(self.recorder)

        prob.setup()
        t0, t1 = run_driver(prob)
        prob.cleanup()

        # Since the test will compare the last case recorded, just check the
        #   current values in the problem. This next section is about getting those values

        # These involve collective gathers so all ranks need to run this
        expected_outputs = prob.driver.get_design_var_values()
        expected_outputs.update(prob.driver.get_objective_values())
        expected_outputs.update(prob.driver.get_constraint_values())

        # Determine the expected values for the sysincludes
        # this gets all of the outputs but just locally
        rrank = prob.comm.rank  # root ( aka model ) rank.
        rowned = prob.model._owning_rank
        # names of sysincl vars on this rank
        local_inclnames = [n for n in prob.driver.recording_options['includes'] if rrank == rowned[n]]
        # Get values for vars on this rank
        inputs, outputs, residuals = prob.model.get_nonlinear_vectors()
        #   Potential local sysvars are in this
        sysvars = outputs._views
        # Just get the values for the sysincl vars on this rank
        local_vars = {c: sysvars[c] for c in local_inclnames}
        # Gather up the values for all the sysincl vars on all ranks
        all_vars = prob.model.comm.gather(local_vars, root=0)

        if prob.comm.rank == 0:
            # Only on rank 0 do we have all the values and only on rank 0
            #   are we doing the testing.
            # The all_vars variable is list of dicts from rank 0,1,... In this case just ranks 0 and 1
            dct = all_vars[-1]
            for d in all_vars[:-1]:
                dct.update(d)

            expected_includes = {
                'par.G1.Cy.y': dct['par.G1.Cy.y'],
                'par.G2.Cy.y': dct['par.G2.Cy.y'],
            }

            expected_outputs.update(expected_includes)

            coordinate = [0, 'SLSQP', (48,)]

            expected_data = ((coordinate, (t0, t1), expected_outputs, None),)
            assertDriverIterDataRecorded(self, expected_data, self.eps)


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
