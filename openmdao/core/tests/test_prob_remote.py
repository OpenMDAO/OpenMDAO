import unittest

import os
import numpy as np
import openmdao.api as om

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
        prob = om.Problem()

        group = prob.model.add_subsystem('group', om.ParallelGroup())

        comp = om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3', y=2.0)
        group.add_subsystem('comp1', comp)

        comp = om.ExecComp('g = x*y', y=2.0)
        group.add_subsystem('comp2', comp)

        prob.setup()

        prob['group.comp1.x'] = 4.
        prob['group.comp2.x'] = 5.

        prob.run_model()

        np.testing.assert_almost_equal(prob.get_val('group.comp1.f', get_remote=True), 42., decimal=5)
        np.testing.assert_almost_equal(prob.get_val('group.comp2.g', get_remote=True), 10., decimal=5)

    def test_remote_var_access_prom(self):
        prob = om.Problem()

        group = prob.model.add_subsystem('group', om.ParallelGroup(), promotes=['f', 'g'])

        group.add_subsystem('indep1', om.IndepVarComp('f'), promotes=['*'])
        group.add_subsystem('indep2', om.IndepVarComp('g'), promotes=['*'])

        prob.model.add_subsystem('summ', om.ExecComp('z = f + g'), promotes=['f', 'g'])
        prob.model.add_subsystem('prod', om.ExecComp('z = f * g'), promotes=['f', 'g'])

        prob.setup()

        prob['f'] = 4.
        prob['g'] = 5.

        prob.run_model()

        np.testing.assert_almost_equal(prob['summ.z'], 9., decimal=5)
        np.testing.assert_almost_equal(prob['prod.z'], 20., decimal=5)

    def test_is_local(self):
        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
        par = p.model.add_subsystem('par', om.ParallelGroup())
        par.add_subsystem('C1', om.ExecComp('y=2*x'))
        par.add_subsystem('C2', om.ExecComp('y=3*x'))
        p.model.connect('indep.x', ['par.C1.x', 'par.C2.x'])

        with self.assertRaisesRegex(RuntimeError,
            "Problem .*: is_local\('indep\.x'\) was called before setup\(\) completed\."):
            loc = p.is_local('indep.x')

        with self.assertRaisesRegex(RuntimeError,
            "Problem .*: is_local\('par\.C1'\) was called before setup\(\) completed\."):
            loc = p.is_local('par.C1')

        with self.assertRaisesRegex(RuntimeError,
            "Problem .*: is_local\('par\.C1\.y'\) was called before setup\(\) completed\."):
            loc = p.is_local('par.C1.y')

        with self.assertRaisesRegex(RuntimeError,
            "Problem .*: is_local\('par\.C1\.x'\) was called before setup\(\) completed\."):
            loc = p.is_local('par.C1.x')

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

            def setup(self):
                rank = self.comm.rank
                sizes, offsets = evenly_distrib_idxs(self.comm.size, N)

                self.add_input('x', shape=1, src_indices=[0], distributed=True)
                self.add_output('y', shape=sizes[rank], distributed=True)

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
                self.connect('dst.y', 'sum.y', src_indices=om.slicer[:])

                self.add_subsystem('par', om.ParallelGroup(), promotes_inputs=['*'])
                self.par.add_subsystem('c1', om.ExecComp(['y=2.0*x']), promotes_inputs=['*'])
                self.par.add_subsystem('c2', om.ExecComp(['y=5.0*x']), promotes_inputs=['*'])

        prob = om.Problem(model=MyModel())
        prob.setup(mode='fwd')

        prob['x'] = 7.0
        prob.run_model()

        # get_remote=True
        assert_near_equal(prob.get_val('x', get_remote=True), [7.])
        assert_near_equal(prob.get_val('dst.x', get_remote=True), [7., 7.])  # ???????
        assert_near_equal(prob.get_val('dst.y', get_remote=True), [2., 7., 7.])
        assert_near_equal(prob.get_val('par.c1.x', get_remote=True), [7.])
        assert_near_equal(prob.get_val('par.c1.y', get_remote=True), [14.])
        assert_near_equal(prob.get_val('par.c2.x', get_remote=True), [7.])
        assert_near_equal(prob.get_val('par.c2.y', get_remote=True), [35.])

        self.maxDiff = None

        remote_msg = ("<model> <class MyModel>: Variable '{name}' is not local to rank {rank}. "
                      "You can retrieve values from other processes using "
                      "`get_val(<name>, get_remote=True)`.")

        distrib_msg = ("<model> <class MyModel>: Variable '{name}' is a distributed variable. "
                       "You can retrieve values from all processes using "
                       "`get_val(<name>, get_remote=True)` or from the local "
                       "process using `get_val(<name>, get_remote=False)`.")

        if prob.comm.rank == 0:
            #
            # get_remote=False
            #

            # get the local part of the distributed inputs/outputs
            assert_near_equal(prob.get_val('dst.x', get_remote=False), [7.])  # from src ('ivc.x')
            assert_near_equal(prob.model.get_val('dst.x', get_remote=False, from_src=False), [7.])
            assert_near_equal(prob.get_val('dst.y', get_remote=False), [2., 7.])

            # par.c1 is local
            assert_near_equal(prob.get_val('par.c1.x', get_remote=False), [7.])  # from src ('ivc.x')
            assert_near_equal(prob.model.get_val('par.c1.x', get_remote=False, from_src=False), [7.])
            assert_near_equal(prob.get_val('par.c1.y', get_remote=False), [14.])

            # par.c2 is remote
            assert_near_equal(prob.get_val('par.c2.x', get_remote=False), [7.])  # from src ('ivc.x')

            with self.assertRaises(RuntimeError) as cm:
                prob.model.get_val('par.c2.x', get_remote=False, from_src=False)
            self.assertEqual(str(cm.exception), remote_msg.format(name='par.c2.x', rank=0))

            with self.assertRaises(RuntimeError) as cm:
                prob.get_val('par.c2.y', get_remote=False)
            self.assertEqual(str(cm.exception), remote_msg.format(name='par.c2.y', rank=0))

            #
            # get_remote=None
            #

            with self.assertRaises(RuntimeError) as cm:
                prob['dst.x']
            self.assertEqual(str(cm.exception), distrib_msg.format(name='dst.x'))

            with self.assertRaises(RuntimeError) as cm:
                prob.model.get_val('dst.x', get_remote=None, from_src=False)
            self.assertEqual(str(cm.exception), distrib_msg.format(name='dst.x'))

            with self.assertRaises(RuntimeError) as cm:
                prob['dst.y']
            self.assertEqual(str(cm.exception), distrib_msg.format(name='dst.y'))

            # par.c1 is local
            assert_near_equal(prob['par.c1.x'], [7.])  # from src ('ivc.x')
            assert_near_equal(prob.model.get_val('par.c1.x', get_remote=None, from_src=False), [7.])
            assert_near_equal(prob['par.c1.y'], [14.])

            # par.c2 is remote
            assert_near_equal(prob['par.c2.x'], [7.])  # from src ('ivc.x')

            with self.assertRaises(RuntimeError) as cm:
                prob.model.get_val('par.c2.x', get_remote=None, from_src=False)
            self.assertEqual(str(cm.exception), remote_msg.format(name='par.c2.x', rank=0))

            with self.assertRaises(RuntimeError) as cm:
                prob['par.c2.y']
            self.assertEqual(str(cm.exception), remote_msg.format(name='par.c2.y', rank=0))

        else:
            #
            # get_remote=False
            #

            # get the local part of the distributed inputs/outputs
            assert_near_equal(prob.get_val('dst.x', get_remote=False), [7.])  # from src ('ivc.x')
            assert_near_equal(prob.model.get_val('dst.x', get_remote=False, from_src=False), [7.])
            assert_near_equal(prob.get_val('dst.y', get_remote=False), [7.])

            # par.c1 is remote
            prob.get_val('par.c1.x', get_remote=False)  # from src ('ivc.x')

            with self.assertRaises(RuntimeError) as cm:
                prob.model.get_val('par.c1.x', get_remote=False, from_src=False)
            self.assertEqual(str(cm.exception), remote_msg.format(name='par.c1.x', rank=1))

            with self.assertRaises(RuntimeError) as cm:
                prob.get_val('par.c1.y', get_remote=False)
            self.assertEqual(str(cm.exception), remote_msg.format(name='par.c1.y', rank=1))

            # par.c2 is local
            assert_near_equal(prob.get_val('par.c2.x', get_remote=False), [7.])  # from src ('ivc.x')
            assert_near_equal(prob.model.get_val('par.c2.x', get_remote=False, from_src=False), [7.])
            assert_near_equal(prob.get_val('par.c2.y', get_remote=False), [35.])

            #
            # get_remote=None
            #

            with self.assertRaises(RuntimeError) as cm:
                prob['dst.x']
            self.assertEqual(str(cm.exception), distrib_msg.format(name='dst.x'))

            with self.assertRaises(RuntimeError) as cm:
                prob.model.get_val('dst.x', get_remote=None, from_src=False)
            self.assertEqual(str(cm.exception), distrib_msg.format(name='dst.x'))

            with self.assertRaises(RuntimeError) as cm:
                prob['dst.y']
            self.assertEqual(str(cm.exception), distrib_msg.format(name='dst.y'))

            # par.c1 is remote
            assert_near_equal(prob['par.c1.x'], [7.])  # from src ('ivc.x')

            with self.assertRaises(RuntimeError) as cm:
                prob.model.get_val('par.c1.x', get_remote=None, from_src=False)
            self.assertEqual(str(cm.exception), remote_msg.format(name='par.c1.x', rank=1))

            with self.assertRaises(RuntimeError) as cm:
                prob['par.c1.y']
            self.assertEqual(str(cm.exception), remote_msg.format(name='par.c1.y', rank=1))

            # par.c2 is local
            assert_near_equal(prob['par.c2.x'], [7.])  # from src ('ivc.x')
            assert_near_equal(prob.model.get_val('par.c2.x', get_remote=None, from_src=False), [7.])
            assert_near_equal(prob['par.c2.y'], [35.])


@unittest.skipIf(os.environ.get("TRAVIS"), "Unreliable on Travis CI.")
@unittest.skipIf(os.environ.get("GITHUB_ACTION"), "Unreliable on GitHub Actions workflows.")
@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
@unittest.skipIf(OPT is None or OPTIMIZER is None, "pyOptSparse is required.")
class ProbRemote4TestCase(unittest.TestCase):

    N_PROCS = 4

    def test_prob_split_comm(self):
        colors = [0, 0, 1, 1]
        comm = MPI.COMM_WORLD.Split(colors[MPI.COMM_WORLD.rank])

        # split the size 4 comm into 2 size 2 comms
        self.assertEqual(comm.size, 2)

        prob = om.Problem(comm=comm)
        model = prob.model

        p1 = model.add_subsystem('p1', om.IndepVarComp('x', 99.0))
        p1.add_design_var('x', lower=-50.0, upper=50.0)

        par = model.add_subsystem('par', om.ParallelGroup())
        c1 = par.add_subsystem('C1', om.ExecComp('y = x*x'))
        c2 = par.add_subsystem('C2', om.ExecComp('y = x*x'))

        model.add_subsystem('obj', om.ExecComp('o = a + b + 2.'))

        model.connect('p1.x', ['par.C1.x', 'par.C2.x'])
        model.connect('par.C1.y', 'obj.a')
        model.connect('par.C2.y', 'obj.b')

        model.add_objective('obj.o')

        prob.set_solver_print(level=0)

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        prob.driver.options['print_results'] = False

        prob.setup()
        prob.run_model()

        failed = prob.run_driver()

        all_failed = comm.allgather(failed)
        if any(all_failed):
            msg = 'No solution found' if prob.driver.pyopt_solution is None \
                else str(prob.driver.pyopt_solution.optInform)
            all_msgs = comm.allgather(msg)
            for i, tup in enumerate(zip(all_failed, all_msgs)):
                failed, msg = tup
                if failed:
                    self.fail(f"Optimization failed on rank {i}: {msg}")

        try:
            obj = prob['obj.o']
            failed = False
        except Exception as err:
            obj = str(err)
            failed = True

        all_status = comm.allgather((obj, failed))
        for i, tup in enumerate(all_status):
            obj, failed = tup
            if failed:
                self.fail(f"Failed to retrieve objective on rank {i}: {obj}")
            assert_near_equal(obj, 2.0, 1e-6)
