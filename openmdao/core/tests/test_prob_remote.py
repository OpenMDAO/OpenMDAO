import unittest

import numpy as np
import openmdao.api as om

from openmdao.api import Problem, ExecComp, Group, ParallelGroup, IndepVarComp
from openmdao.utils.mpi import MPI
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.proc_allocators.default_allocator import DefaultAllocator
from openmdao.test_suite.components.expl_comp_array import TestExplCompArrayDense
from openmdao.utils.assert_utils import assert_near_equal


if MPI:
    try:
        from openmdao.vectors.petsc_vector import PETScVector
    except ImportError:
        PETScVector = None


# check that pyoptsparse is installed. if it is, try to use SLSQP.
OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')
if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ProbRemoteTestCase(unittest.TestCase):

    N_PROCS = 2

    def test_remote_var_access(self):
        # build the model
        prob = Problem()

        group = prob.model.add_subsystem('group', ParallelGroup())

        comp = ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3', y=2.0)
        group.add_subsystem('comp1', comp)

        comp = ExecComp('g = x*y', y=2.0)
        group.add_subsystem('comp2', comp)

        prob.setup()

        prob['group.comp1.x'] = 4.
        prob['group.comp2.x'] = 5.

        prob.run_model()

        np.testing.assert_almost_equal(prob.get_val('group.comp1.f', get_remote=True), 42., decimal=5)
        np.testing.assert_almost_equal(prob.get_val('group.comp2.g', get_remote=True), 10., decimal=5)

    def test_remote_var_access_prom(self):
        prob = Problem()

        group = prob.model.add_subsystem('group', ParallelGroup(), promotes=['f', 'g'])

        group.add_subsystem('indep1', IndepVarComp('f'), promotes=['*'])
        group.add_subsystem('indep2', IndepVarComp('g'), promotes=['*'])

        prob.model.add_subsystem('summ', ExecComp('z = f + g'), promotes=['f', 'g'])
        prob.model.add_subsystem('prod', ExecComp('z = f * g'), promotes=['f', 'g'])

        prob.setup()

        prob['f'] = 4.
        prob['g'] = 5.

        prob.run_model()

        np.testing.assert_almost_equal(prob['summ.z'], 9., decimal=5)
        np.testing.assert_almost_equal(prob['prod.z'], 20., decimal=5)

    def test_is_local(self):
        p = Problem()
        p.model.add_subsystem('indep', IndepVarComp('x', 1.0))
        par = p.model.add_subsystem('par', ParallelGroup())
        par.add_subsystem('C1', ExecComp('y=2*x'))
        par.add_subsystem('C2', ExecComp('y=3*x'))
        p.model.connect('indep.x', ['par.C1.x', 'par.C2.x'])

        with self.assertRaises(RuntimeError) as cm:
            loc = p.is_local('indep.x')
        self.assertEqual(str(cm.exception), "Problem: is_local('indep.x') was called before setup() completed.")

        with self.assertRaises(RuntimeError) as cm:
            loc = p.is_local('par.C1')
        self.assertEqual(str(cm.exception), "Problem: is_local('par.C1') was called before setup() completed.")

        with self.assertRaises(RuntimeError) as cm:
            loc = p.is_local('par.C1.y')
        self.assertEqual(str(cm.exception), "Problem: is_local('par.C1.y') was called before setup() completed.")

        with self.assertRaises(RuntimeError) as cm:
            loc = p.is_local('par.C1.x')
        self.assertEqual(str(cm.exception), "Problem: is_local('par.C1.x') was called before setup() completed.")

        p.setup()
        p.final_setup()

        self.assertTrue(p.is_local('indep'), 'indep should be local')
        self.assertTrue(p.is_local('indep.x'), 'indep.x should be local')

        if p.comm.rank == 0:
            self.assertTrue(p.is_local('par.C1'), 'par.C1 should be local')
            self.assertTrue(p.is_local('par.C1.x'), 'par.C1.x should be local')
            self.assertTrue(p.is_local('par.C1.y'), 'par.C1.y should be local')

            self.assertFalse(p.is_local('par.C2'), 'par.C1 should be remote')
            self.assertFalse(p.is_local('par.C2.x'), 'par.C1.x should be remote')
            self.assertFalse(p.is_local('par.C2.y'), 'par.C1.y should be remote')
        else:
            self.assertFalse(p.is_local('par.C1'), 'par.C1 should be remote')
            self.assertFalse(p.is_local('par.C1.x'), 'par.C1.x should be remote')
            self.assertFalse(p.is_local('par.C1.y'), 'par.C1.y should be remote')

            self.assertTrue(p.is_local('par.C2'), 'par.C2 should be local')
            self.assertTrue(p.is_local('par.C2.x'), 'par.C2.x should be local')
            self.assertTrue(p.is_local('par.C2.y'), 'par.C2.y should be local')

    def test_get_remote(self):

        N = 3

        class DistribComp(om.ExplicitComponent):

            def initialize(self):
                self.options['distributed'] = True

            def setup(self):
                rank = self.comm.rank
                sizes, offsets = evenly_distrib_idxs(self.comm.size, N)

                self.add_input('x', shape=1)
                self.add_output('y', shape=sizes[rank])
                
            def compute(self, inputs, outputs):
                rank = self.comm.rank
                sizes, offsets = evenly_distrib_idxs(self.comm.size, N)

                outputs['y'] = inputs['x']*np.ones((sizes[rank],))
                if rank == 0:
                    outputs['y'][0] = 2.

        class MyModel(om.Group):

            def setup(self):
                self.add_subsystem('ivc', om.IndepVarComp('x', 0.), promotes_outputs=['*'])
                self.add_subsystem('dst', DistribComp(), promotes_inputs=['*'])
                self.add_subsystem('sum', om.ExecComp('z = sum(y)', y=np.zeros((N,)), z=0.0))
                self.connect('dst.y', 'sum.y')

                self.add_subsystem('par', om.ParallelGroup(), promotes_inputs=['*'])
                self.par.add_subsystem('c1', om.ExecComp(['y=2.0*x']), promotes_inputs=['*'])
                self.par.add_subsystem('c2', om.ExecComp(['y=5.0*x']), promotes_inputs=['*'])

        prob = om.Problem(model=MyModel())
        prob.setup(mode='fwd')

        prob['x'] = 7.0
        prob.run_model()

        # get_remote=True
        assert_near_equal(prob.get_val('dst.y', get_remote=True), [2., 7., 7.])
        assert_near_equal(prob.get_val('par.c1.y', get_remote=True), [14.])
        assert_near_equal(prob.get_val('par.c2.y', get_remote=True), [35.])

        if prob.comm.rank == 0:
            # get_remote=False
            assert_near_equal(prob.get_val('dst.y', get_remote=False), [2., 7.])
            assert_near_equal(prob.get_val('par.c1.y', get_remote=False), [14.])
            with self.assertRaises(RuntimeError) as cm:
                prob.get_val('par.c2.y', get_remote=False)
            self.assertEqual(str(cm.exception), 
                             ("Problem: Variable 'par.c2.y' is not local to rank 0. "
                              "You can retrieve values from  other processes using "
                              "`problem.get_val(<name>, get_remote=True)`."))

            # get_remote=None
            with self.assertRaises(RuntimeError) as cm:
                prob['dst.y']
            self.assertEqual(str(cm.exception), 
                             ("MyModel (<model>): dst.y is a distributed variable, "
                             "You can retrieve values from all processes using "
                             "`get_val(<name>, get_remote=True)' or from the local "
                             "process using `get_val(<name>, get_remote=False)'."))
            assert_near_equal(prob['par.c1.y'], [14.])
            with self.assertRaises(RuntimeError) as cm:
                prob['par.c2.y']
            self.assertEqual(str(cm.exception), 
                             ("Problem: Variable 'par.c2.y' is not local to rank 0. "
                              "You can retrieve values from  other processes using "
                              "`problem.get_val(<name>, get_remote=True)`."))
        else:
            # get_remote=False
            assert_near_equal(prob.get_val('dst.y', get_remote=False), [7.])
            with self.assertRaises(RuntimeError) as cm:
                prob.get_val('par.c1.y', get_remote=False)
            self.assertEqual(str(cm.exception), 
                             ("Problem: Variable 'par.c1.y' is not local to rank 1. "
                              "You can retrieve values from  other processes using "
                              "`problem.get_val(<name>, get_remote=True)`."))
            assert_near_equal(prob.get_val('par.c2.y', get_remote=False), [35.])

            # get_remote=None
            with self.assertRaises(RuntimeError) as cm:
                prob['dst.y']
            self.assertEqual(str(cm.exception), 
                             ("MyModel (<model>): dst.y is a distributed variable, "
                             "You can retrieve values from all processes using "
                             "`get_val(<name>, get_remote=True)' or from the local "
                             "process using `get_val(<name>, get_remote=False)'."))
            with self.assertRaises(RuntimeError) as cm:
                prob['par.c1.y']
            self.assertEqual(str(cm.exception), 
                             ("Problem: Variable 'par.c1.y' is not local to rank 1. "
                              "You can retrieve values from  other processes using "
                              "`problem.get_val(<name>, get_remote=True)`."))
            assert_near_equal(prob['par.c2.y'], [35.])


@unittest.skip("FIXME: test is unreliable on CI... (timeout)")
#@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ProbRemote4TestCase(unittest.TestCase):

    N_PROCS = 4

    def test_prob_split_comm(self):
        colors = [0, 0, 1, 1]
        comm = MPI.COMM_WORLD.Split(colors[MPI.COMM_WORLD.rank])

        # split the size 4 comm into 2 size 2 comms
        self.assertEqual(comm.size, 2)

        prob = Problem(comm=comm)
        model = prob.model

        p1 = model.add_subsystem('p1', IndepVarComp('x', 99.0))
        p1.add_design_var('x', lower=-50.0, upper=50.0)

        par = model.add_subsystem('par', ParallelGroup())
        c1 = par.add_subsystem('C1', ExecComp('y = x*x'))
        c2 = par.add_subsystem('C2', ExecComp('y = x*x'))

        model.add_subsystem('obj', ExecComp('o = a + b + 2.'))

        model.connect('p1.x', ['par.C1.x', 'par.C2.x'])
        model.connect('par.C1.y', 'obj.a')
        model.connect('par.C2.y', 'obj.b')

        model.add_objective('obj.o')

        prob.set_solver_print(level=0)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False

        prob.setup()
        prob.run_model()

        failed = prob.run_driver()

        all_failed = comm.allgather(failed)
        if any(all_failed):
            all_msgs = comm.allgather(str(prob.driver.pyopt_solution.optInform))
            for i, tup in enumerate(zip(all_failed, all_msgs)):
                failed, msg = tup
                if failed:
                    self.fail("Optimization failed on rank %d: %s" % (i, msg))

        objs = comm.allgather(prob['obj.o'])
        for i, obj in enumerate(objs):
            assert_near_equal(obj, 2.0, 1e-6)
