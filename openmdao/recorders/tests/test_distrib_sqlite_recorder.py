import errno
import os
import unittest
from time import time
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np

from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.mpi import MPI

import openmdao.api as om
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.recorders.tests.sqlite_recorder_test_utils import \
    assertDriverIterDataRecorded, assertProblemDataRecorded
from openmdao.recorders.tests.recorder_test_utils import run_driver

if MPI:
    from openmdao.api import PETScVector
else:
    PETScVector = None


class DistributedAdder(om.ExplicitComponent):
    """
    Distributes the work of adding 10 to every item in the input vectors
    """

    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes

    def setup(self):
        """
        specify the local sizes of the variables and which specific indices this specific
        distributed component will handle. Indices do NOT need to be sequential or
        contiguous!
        """
        comm = self.comm
        rank = comm.rank

        for n, size in enumerate(self.sizes):
            # NOTE: evenly_distrib_idxs is a helper function to split the array
            #       up as evenly as possible
            local_sizes, _ = evenly_distrib_idxs(comm.size, size)
            local_size = local_sizes[rank]

            self.add_input(f'in{n}', val=np.zeros(local_size, float), distributed=True)
            self.add_output(f'out{n}', val=np.zeros(local_size, float), distributed=True)

    def compute(self, inputs, outputs):

        # NOTE: Each process will get just its local part of the vector
        for n, size in enumerate(self.sizes):
            outputs[f'out{n}'] = inputs[f'in{n}'] + 10.


class Summer(om.ExplicitComponent):
    """
    Aggregation component that collects all the values from the distributed
    vectors and computes a total
    """

    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes

    def setup(self):
        for n, size in enumerate(self.sizes):
            self.add_input(f'summand{n}', val=np.zeros(size))

        self.add_output('sum', 0.0, shape=1)

    def compute(self, inputs, outputs):
        val = 0.

        for name in inputs:
             val += np.sum(inputs[name])

        outputs['sum'] = val


class Mygroup(om.Group):

    def setup(self):
        self.add_subsystem('indep_var_comp', om.IndepVarComp('x'), promotes=['*'])
        self.add_subsystem('Cy', om.ExecComp('y=2*x'), promotes=['*'])
        self.add_subsystem('Cc', om.ExecComp('c=x+2'), promotes=['*'])

        self.add_design_var('x')
        self.add_constraint('c', lower=-3.)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class DistributedRecorderTest(unittest.TestCase):

    N_PROCS = 2

    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "sqlite_test")
        self.recorder = om.SqliteRecorder(self.filename)
        self.eps = 1e-5

    def tearDown(self):
        try:
            rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def test_distrib_record_system(self):
        prob = om.Problem()

        try:
            prob.model.add_recorder(self.recorder)
        except RuntimeError as err:
            msg = "<class Group>: Recording of Systems when running parallel code is not supported yet"
            self.assertEqual(str(err), msg)
        else:
            self.fail('RuntimeError expected.')

    def test_distrib_record_solver(self):
        prob = om.Problem()
        try:
            prob.model.nonlinear_solver.add_recorder(self.recorder)
        except RuntimeError as err:
            msg = "Recording of Solvers when running parallel code is not supported yet"
            self.assertEqual(str(err), msg)
        else:
            self.fail('RuntimeError expected.')

    def test_distrib_record_driver(self):
        # create distributed variables of different sizes to catch mismatched collective calls
        sizes = [7, 10, 12, 25, 33, 42]

        prob = om.Problem()

        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
        for n, size in enumerate(sizes):
            ivc.add_output(f'in{n}', np.ones(size), distributed=True)
            prob.model.add_design_var(f'in{n}')

        prob.model.add_subsystem('adder', DistributedAdder(sizes), promotes=['*'])

        prob.model.add_subsystem('summer', Summer(sizes), promotes_outputs=['sum'])
        for n, size in enumerate(sizes):
            prob.model.promotes('summer', inputs=[f'summand{n}'], src_indices=om.slicer[:], src_shape=size)
        prob.model.add_objective('sum')

        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['includes'] = [f'out{n}' for n in range(len(sizes))]
        prob.driver.add_recorder(self.recorder)

        prob.setup()
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0,)]

        expected_desvars = {}
        for n in range(len(sizes)):
            expected_desvars[f'ivc.in{n}'] = prob.get_val(f'ivc.in{n}', get_remote=True)

        expected_objectives = { "summer.sum": prob['summer.sum'] }

        expected_outputs = expected_desvars.copy()
        for n in range(len(sizes)):
            expected_outputs[f'adder.out{n}'] = prob.get_val(f'adder.out{n}', get_remote=True)

        if prob.comm.rank == 0:
            expected_outputs.update(expected_objectives)

            expected_data = ((coordinate, (t0, t1), expected_outputs, None, None),)
            assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_recording_remote_voi(self):
        # Create a parallel model
        model = om.Group()

        model.add_subsystem('par', om.ParallelGroup())
        model.par.add_subsystem('G1', Mygroup())
        model.par.add_subsystem('G2', Mygroup())
        model.connect('par.G1.y', 'Obj.y1')
        model.connect('par.G2.y', 'Obj.y2')

        model.add_subsystem('Obj', om.ExecComp('obj=y1+y2'))
        model.add_objective('Obj.obj')

        # Configure driver to record VOIs on both procs
        driver = om.ScipyOptimizeDriver(disp=False)

        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['includes'] = ['par.G1.y', 'par.G2.y']

        driver.add_recorder(self.recorder)

        # Create problem and run driver
        prob = om.Problem(model, driver)
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

    def test_input_desvar(self):
        # this failed with a KeyError before the fix
        class TopComp(om.ExplicitComponent):

            def setup(self):

                size = 10

                self.add_input('c_ae_C', np.zeros(size))
                self.add_input('theta_c2_C', np.zeros(size))
                self.add_output('c_ae', np.zeros(size))

            def compute(self, inputs, outputs):
                pass

            def compute_partials(self, inputs, partials):
                pass

        prob = om.Problem()
        model = prob.model

        geom = model.add_subsystem('tcomp', TopComp())

        model.add_design_var('tcomp.theta_c2_C', lower=-20., upper=20., indices=range(2, 9))
        model.add_constraint('tcomp.c_ae', lower=0.e0,)

        # Attach recorder to the problem
        prob.add_recorder(self.recorder)
        prob.recording_options['record_inputs'] = True
        prob.recording_options['record_outputs'] = True
        prob.recording_options['includes'] = ['*']

        prob.setup()

        prob.run_model()
        prob.record('final')


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
