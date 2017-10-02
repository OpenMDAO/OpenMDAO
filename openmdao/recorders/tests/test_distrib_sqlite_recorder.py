import errno
import os
import sqlite3
import unittest
from shutil import rmtree
from tempfile import mkdtemp

import numpy as np

# import pydevd
# from openmdao.utils.mpi import MPI
# if MPI.COMM_WORLD.rank:
#     pydevd.settrace('localhost', port=9876, stdoutToServer=True, stderrToServer=True)
# else:
#     pydevd.settrace('localhost', port=9877, stdoutToServer=True, stderrToServer=True)


from openmdao.utils.mpi import MPI
from openmdao.devtools.testutil import assert_rel_error

if MPI:
    from openmdao.api import PETScVector
    vector_class = PETScVector
    try:
        from openmdao.api import pyOptSparseDriver
    except ImportError:
        pyOptSparseDriver = None
else:
    PETScVector = None
    pyOptSparseDriver = None


from openmdao.api import ExecComp, ExplicitComponent, Problem, \
    Group, ParallelGroup, IndepVarComp, SqliteRecorder
from openmdao.utils.array_utils import evenly_distrib_idxs
from sqlite_recorder_test_utils import _assertDriverIterationDataRecorded
from recorder_test_utils import run_driver

# try:
#     from openmdao.vectors.petsc_vector import PETScVector
# except ImportError:
#     PETScVector = None

class DistributedAdder(ExplicitComponent):
    """
    Distributes the work of adding 10 to every item in the param vector
    """

    def __init__(self, size):
        super(DistributedAdder, self).__init__()
        self.distributed = True

        self.local_size = self.size = size

    def get_req_procs(self):
        """
        min/max number of procs that this component can use
        """
        return (1, self.size)

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








@unittest.skipIf(PETScVector is None or os.environ.get("TRAVIS"),
                 "PETSc is required." if PETScVector is None
                 else "Unreliable on Travis CI.")
class DistributedRecorderTest(unittest.TestCase):

    N_PROCS = 2

    def setUp(self):
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "sqlite_test")
        print('self.filename',self.filename)
        self.recorder = SqliteRecorder(self.filename)
        self.eps = 1e-5

    def tearDown(self):
        try:
            pass
            # rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def assertDriverIterationDataRecorded(self, expected, tolerance):
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        _assertDriverIterationDataRecorded(self, cur, expected, tolerance)
        con.close()

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
        self.recorder.options['record_desvars'] = True
        self.recorder.options['record_responses'] = True
        self.recorder.options['record_objectives'] = True
        self.recorder.options['record_constraints'] = True
        prob.driver.add_recorder(self.recorder)

        prob.model.add_design_var('x')
        prob.model.add_objective('sum')

        prob.setup(vector_class=PETScVector, check=False)

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

            self.assertDriverIterationDataRecorded(((coordinate, (t0, t1), expected_desvars, None,
                                                     expected_objectives, None, None),), self.eps)

    def test_recording_remote_voi(self):
        import pydevd

        # from openmdao.utils.mpi import MPI
        # if MPI.COMM_WORLD.rank:
        #     pydevd.settrace('localhost', port=9877, stdoutToServer=True, stderrToServer=True)
        # else:
        #     pydevd.settrace('localhost', port=9876, stdoutToServer=True, stderrToServer=True)

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

        self.recorder.options['record_desvars'] = True
        self.recorder.options['record_responses'] = True
        self.recorder.options['record_objectives'] = True
        self.recorder.options['record_constraints'] = True
        self.recorder.options['system_includes'] = ['par.G1.Cy.y','par.G2.Cy.y']


        prob.driver.add_recorder(self.recorder)

        prob.setup(vector_class=PETScVector)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        # Need to put these outside of the if statement below
        #   because they are collective calls for doing gather
        expected_desvars = prob.driver.get_design_var_values()
        expected_objectives = prob.driver.get_objective_values()
        expected_constraints = prob.driver.get_constraint_values()


        # this gets all of the outputs but just locally
        rrank = prob.comm.rank  # root ( aka model ) rank.

        rowned = prob.model._owning_rank['output']
        local_sysinclnames = [n for n in self.recorder.options['system_includes'] if rrank == rowned[n]]

        inputs, outputs, residuals = prob.model.get_nonlinear_vectors()
        #   Potential local sysvars are in
        sysvars = outputs._names
        local_sysvars = {c: sysvars[c] for c in local_sysinclnames}
        all_vars = prob.model.comm.gather(local_sysvars, root=0)

        if prob.comm.rank == 0:
            dct = all_vars[-1]
            for d in all_vars[:-1]:
                dct.update(d)

            expected_sysincludes = {
                'par.G1.Cy.y': dct['par.G1.Cy.y'],
                'par.G2.Cy.y': dct['par.G2.Cy.y'],
            }

        #
        # def _gather_vars(self, root, local_vars):
        #     all_vars = root.comm.gather(local_vars, root=0)
        # returns list of dicts. Each item in list is from one of the ranks
        #
        #     if root.comm.rank == 0:
        #         dct = all_vars[-1]
        #         for d in all_vars[:-1]:
        #             dct.update(d)
        #         return dct

        # qqq = prob['par.G1.y'] # par.G1.Cy

        inputs, outputs, residuals = prob.model.get_nonlinear_vectors()

        model = prob.model
        comm = model.comm
        vec = model._outputs._views_flat

        varname = 'par.G2.Cy.y'
        varowner = prob.model._owning_rank['output'][varname]
        varindex = prob.model._var_allprocs_abs_names['output'].index(varname)
        varsize = model._var_sizes['nonlinear']['output'][varowner, varindex]
        if varowner == comm.rank:
            val = vec[varname].copy()
        else:
            val = np.empty(varsize)
        comm.Bcast(val, root=varowner)

        pass

        # vec = model._outputs._views_flat
        # {'par.G2.indep_var_comp.x': array([1.75696747e+15]), 'par.G2.Cy.y': array([3.51393495e+15]),
        #  'par.G2.Cc.c': array([1.75696747e+15]), 'Obj.obj': array([-7.19306049e+16])}

        # prob.driver._remote_dvs
        # {'par.G1.indep_var_comp.x': (0, 1), 'par.G2.indep_var_comp.x': (1, 1)}
        # where the tuple is owner, size

        # prob.model._owning_rank['output']['par.G1.Cy.y']

        #             sizes = model._var_sizes['nonlinear']['output']
        # [[1 1 1 0 0 0 1]
        #  [0 0 0 1 1 1 1]]

        # prob.model._var_allprocs_abs_names['output']
        # < class 'list'>: ['par.G1.indep_var_comp.x', 'par.G1.Cy.y', 'par.G1.Cc.c', 'par.G2.indep_var_comp.x',
        #                   'par.G2.Cy.y', 'par.G2.Cc.c', 'Obj.obj']

        # owner, size = remote_vois[name]
        # if owner == comm.rank:
        #     if indices is None:
        #         val = vec[name].copy()
        #     else:
        #         val = vec[name][indices]
        # else:
        #     if indices is not None:
        #         size = len(indices)
        #     val = np.empty(size)
        # comm.Bcast(val, root=owner)

        if prob.comm.rank == 0:
            coordinate = [0, 'SLSQP', (49,)]


            # 'par.G1.indep_var_comp.x'
            # G2 is first
            #                ([1.75696747e+15], [-3.77222699e+16])

            # desvars expected
            # {'par.G1.x': array([ -3.77222699e+16])}

            # qqq = desvars['par.G2.x']
            # expected_desvars = {
            #     "par.G1.indep_var_comp.x": prob['par.G1.x'],
            #     "par.G2.indep_var_comp.x": prob['par.G2.x'],
            # }
            #
            # # actual : <class 'list'>: [([ -7.19306049e+16],)]
            # # expected: {'Obj.obj': array([ -7.19306049e+16])}
            # expected_objectives = {
            #     "Obj.obj": prob['Obj.obj'],
            # }
            #
            # # Constraints actual
            # # [('par.G2.Cc.c', '<f8', (1,)), ('par.G1.Cc.c', '<f8', (1,))]
            # # <class 'list'>: [([  1.75696747e+15], [ -3.77222699e+16])]
            # expected_constraints = {
            #     "par.G1.Cc.c": prob['par.G1.Cc.c'],
            #     "par.G2.Cc.c": prob['par.G2.Cc.c'],
            # }

            # No sysincludes actual
            self.assertDriverIterationDataRecorded(((coordinate, (t0, t1), expected_desvars, None,
                                                     expected_objectives, expected_constraints,
                                                     expected_sysincludes),), self.eps)

#
# @unittest.skipIf(PETScVector is None or os.environ.get("TRAVIS"),
#                  "PETSc is required." if PETScVector is None
#                  else "Unreliable on Travis CI.")
# class RemoteVOITestCase(unittest.TestCase):
#
#     N_PROCS = 2
#
#     def test_recording_remote_voi(self):
#         prob = Problem()
#
#         prob.model.add_subsystem('par', ParallelGroup())
#
#         prob.model.par.add_subsystem('G1', Mygroup())
#         prob.model.par.add_subsystem('G2', Mygroup())
#
#         prob.model.add_subsystem('Obj', ExecComp('obj=y1+y2'))
#
#         prob.model.connect('par.G1.y', 'Obj.y1')
#         prob.model.connect('par.G2.y', 'Obj.y2')
#
#         prob.model.add_objective('Obj.obj')
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SLSQP'
#         prob.setup(vector_class=PETScVector)
#
#         prob.run_driver()
#
#         J = prob.compute_total_derivs(of=['Obj.obj', 'par.G1.c', 'par.G2.c'],
#                                       wrt=['par.G1.x', 'par.G2.x'])
#
#         assert_rel_error(self, J['Obj.obj', 'par.G1.x'], np.array([[2.0]]), 1e-6)
#         assert_rel_error(self, J['Obj.obj', 'par.G2.x'], np.array([[2.0]]), 1e-6)
#         assert_rel_error(self, J['par.G1.c', 'par.G1.x'], np.array([[1.0]]), 1e-6)
#         assert_rel_error(self, J['par.G1.c', 'par.G2.x'], np.array([[0.0]]), 1e-6)
#         assert_rel_error(self, J['par.G2.c', 'par.G1.x'], np.array([[0.0]]), 1e-6)
#         assert_rel_error(self, J['par.G2.c', 'par.G2.x'], np.array([[1.0]]), 1e-6)


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
