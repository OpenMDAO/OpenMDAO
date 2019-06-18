"""Test the parallel groups."""

from __future__ import division, print_function

import unittest
import numpy as np

from openmdao.api import Problem, Group, ParallelGroup, ExecComp, IndepVarComp, \
                         ExplicitComponent, ImplicitComponent, DefaultVector

from openmdao.utils.mpi import MPI

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from openmdao.test_suite.groups.parallel_groups import \
    FanOutGrouped, FanInGrouped2, Diamond, ConvergeDiverge

from openmdao.utils.assert_utils import assert_rel_error
from openmdao.utils.logger_utils import TestLogger



@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestParallelGroups(unittest.TestCase):

    N_PROCS = 2

    def test_fan_out_grouped(self):
        prob = Problem(FanOutGrouped())

        of=['c2.y', "c3.y"]
        wrt=['iv.x']

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        J = prob.compute_totals(of=['c2.y', "c3.y"], wrt=['iv.x'])

        assert_rel_error(self, J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

        assert_rel_error(self, prob['c2.y'], -6.0, 1e-6)
        assert_rel_error(self, prob['c3.y'], 15.0, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        J = prob.compute_totals(of=['c2.y', "c3.y"], wrt=['iv.x'])

        assert_rel_error(self, J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

        assert_rel_error(self, prob['c2.y'], -6.0, 1e-6)
        assert_rel_error(self, prob['c3.y'], 15.0, 1e-6)

    def test_fan_in_grouped(self):

        prob = Problem()
        prob.model = FanInGrouped2()
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()


        indep_list = ['p1.x', 'p2.x']
        unknown_list = ['c3.y']

        assert_rel_error(self, prob['c3.y'], 29.0, 1e-6)

        J = prob.compute_totals(of=unknown_list, wrt=indep_list)
        assert_rel_error(self, J['c3.y', 'p1.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'p2.x'][0][0], 35.0, 1e-6)

        assert_rel_error(self, prob['c3.y'], 29.0, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        assert_rel_error(self, prob['c3.y'], 29.0, 1e-6)

        J = prob.compute_totals(of=unknown_list, wrt=indep_list)
        assert_rel_error(self, J['c3.y', 'p1.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'p2.x'][0][0], 35.0, 1e-6)

        assert_rel_error(self, prob['c3.y'], 29.0, 1e-6)

    def test_fan_in_grouped_feature(self):

        from openmdao.api import Problem, IndepVarComp, ParallelGroup, ExecComp, PETScVector

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 1.0))
        model.add_subsystem('p2', IndepVarComp('x', 1.0))

        parallel = model.add_subsystem('parallel', ParallelGroup())
        parallel.add_subsystem('c1', ExecComp(['y=-2.0*x']))
        parallel.add_subsystem('c2', ExecComp(['y=5.0*x']))

        model.add_subsystem('c3', ExecComp(['y=3.0*x1+7.0*x2']))

        model.connect("parallel.c1.y", "c3.x1")
        model.connect("parallel.c2.y", "c3.x2")

        model.connect("p1.x", "parallel.c1.x")
        model.connect("p2.x", "parallel.c2.x")

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        assert_rel_error(self, prob['c3.y'], 29.0, 1e-6)

    def test_diamond(self):

        prob = Problem()
        prob.model = Diamond()
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['c4.y1'], 46.0, 1e-6)
        assert_rel_error(self, prob['c4.y2'], -93.0, 1e-6)

        indep_list = ['iv.x']
        unknown_list = ['c4.y1', 'c4.y2']

        J = prob.compute_totals(of=unknown_list, wrt=indep_list)
        assert_rel_error(self, J['c4.y1', 'iv.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['c4.y2', 'iv.x'][0][0], -40.5, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        assert_rel_error(self, prob['c4.y1'], 46.0, 1e-6)
        assert_rel_error(self, prob['c4.y2'], -93.0, 1e-6)

        J = prob.compute_totals(of=unknown_list, wrt=indep_list)
        assert_rel_error(self, J['c4.y1', 'iv.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['c4.y2', 'iv.x'][0][0], -40.5, 1e-6)

    def test_converge_diverge(self):

        prob = Problem()
        prob.model = ConvergeDiverge()
        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_rel_error(self, prob['c7.y1'], -102.7, 1e-6)

        indep_list = ['iv.x']
        unknown_list = ['c7.y1']

        J = prob.compute_totals(of=unknown_list, wrt=indep_list)
        assert_rel_error(self, J['c7.y1', 'iv.x'][0][0], -40.75, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        assert_rel_error(self, prob['c7.y1'], -102.7, 1e-6)

        J = prob.compute_totals(of=unknown_list, wrt=indep_list)
        assert_rel_error(self, J['c7.y1', 'iv.x'][0][0], -40.75, 1e-6)

        assert_rel_error(self, prob['c7.y1'], -102.7, 1e-6)

    def test_zero_shape(self):
        raise unittest.SkipTest("zero shapes not fully supported yet")
        class MultComp(ExplicitComponent):
            def __init__(self, mult):
                self.mult = mult
                super(MultComp, self).__init__()

            def setup(self):
                if self.comm.rank == 0:
                    self.add_input('x', shape=1)
                    self.add_output('y', shape=1)
                else:
                    self.add_input('x', shape=0)
                    self.add_output('y', shape=0)

            def compute(self, inputs, outputs):
                outputs['y'] = inputs['x'] * self.mult

            def compute_partials(self, inputs, partials):
                partials['y', 'x'] = np.array([self.mult])

        prob = Problem()

        model = prob.model
        model.add_subsystem('iv', IndepVarComp('x', 1.0))
        model.add_subsystem('c1', MultComp(3.0))

        model.sub = model.add_subsystem('sub', ParallelGroup())
        model.sub.add_subsystem('c2', MultComp(-2.0))
        model.sub.add_subsystem('c3', MultComp(5.0))

        model.add_subsystem('c2', MultComp(1.0))
        model.add_subsystem('c3', MultComp(1.0))

        model.connect('iv.x', 'c1.x')

        model.connect('c1.y', 'sub.c2.x')
        model.connect('c1.y', 'sub.c3.x')

        model.connect('sub.c2.y', 'c2.x')
        model.connect('sub.c3.y', 'c3.x')

        of=['c2.y', "c3.y"]
        wrt=['iv.x']

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        J = prob.compute_totals(of=['c2.y', "c3.y"], wrt=['iv.x'])

        assert_rel_error(self, J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

        assert_rel_error(self, prob['c2.y'], -6.0, 1e-6)
        assert_rel_error(self, prob['c3.y'], 15.0, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        J = prob.compute_totals(of=['c2.y', "c3.y"], wrt=['iv.x'])

        assert_rel_error(self, J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

        assert_rel_error(self, prob['c2.y'], -6.0, 1e-6)
        assert_rel_error(self, prob['c3.y'], 15.0, 1e-6)

    def test_setup_messages(self):

        class Noisy(ConvergeDiverge):
            def check_config(self, logger):
                logger.error(msg)
                logger.warning(msg)
                logger.info(msg)

        prob = Problem(Noisy())

        # check that error is thrown if not using PETScVector
        if MPI:
            msg = ("The `distributed_vector_class` argument must be `PETScVector` when "
                   "running in parallel under MPI but 'DefaultVector' was specified.")
            with self.assertRaises(ValueError) as cm:
                prob.setup(check=False, mode='fwd', distributed_vector_class=DefaultVector)

            self.assertEqual(str(cm.exception), msg)
        else:
            prob.setup(check=False, mode='fwd')

        # check that we get setup messages only on proc 0
        msg = 'Only want to see this on rank 0'
        testlogger = TestLogger()
        prob.setup(check=True, mode='fwd',
                   logger=testlogger)
        prob.final_setup()

        if prob.comm.rank > 0:
            self.assertEqual(len(testlogger.get('error')), 0)
            self.assertEqual(len(testlogger.get('warning')), 0)
            self.assertEqual(len(testlogger.get('info')), 0)
        else:
            self.assertEqual(len(testlogger.get('error')), 1)
            self.assertTrue(testlogger.contains('warning',
                                                "Only want to see this on rank 0"))
            self.assertEqual(len(testlogger.get('info')), 1)
            self.assertTrue(msg in testlogger.get('error')[0])
            self.assertTrue(msg in testlogger.get('info')[0])


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestParallelListStates(unittest.TestCase):

    N_PROCS = 4

    def test_list_states_allprocs(self):
        class StateComp(ImplicitComponent):

            def initialize(self):
                self.mtx = np.array([
                    [0.99, 0.01],
                    [0.01, 0.99],
                ])

            def setup(self):
                self.add_input('rhs', val=np.ones(2))
                self.add_output('x', val=np.zeros(2))

                self.declare_partials(of='*', wrt='*')

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['x'] = self.mtx.dot(outputs['x']) - inputs['rhs']

            def solve_nonlinear(self, inputs, outputs):
                outputs['x'] = np.linalg.solve(self.mtx, inputs['rhs'])

        p = Problem(model=ParallelGroup())
        p.model.add_subsystem('C1', StateComp())
        p.model.add_subsystem('C2', StateComp())
        p.model.add_subsystem('C3', ExecComp('y=2.0*x'))
        p.model.add_subsystem('C4', StateComp())
        p.setup()
        p.final_setup()
        self.assertEqual(p.model._list_states_allprocs(), ['C1.x', 'C2.x', 'C4.x'])


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class MatMatParDevTestCase(unittest.TestCase):
    N_PROCS = 2

    def test_size_1_matmat(self):
        p = Problem()
        indeps = p.model.add_subsystem('indeps', IndepVarComp('x', np.ones(2)))
        indeps.add_output('y', 1.0)
        par = p.model.add_subsystem('par', ParallelGroup())
        par.add_subsystem('C1', ExecComp('y=2*x', x=np.zeros(2), y=np.zeros(2)))
        par.add_subsystem('C2', ExecComp('y=3*x'))
        p.model.connect("indeps.x", "par.C1.x")
        p.model.connect("indeps.y", "par.C2.x")
        p.model.add_design_var('indeps.x', vectorize_derivs=True, parallel_deriv_color='foo')
        p.model.add_design_var('indeps.y', vectorize_derivs=True, parallel_deriv_color='foo')
        par.add_objective('C2.y')
        par.add_constraint('C1.y', lower=0.0)
        p.setup(mode='fwd')
        p.run_model()

        # prior to bug fix, this would raise an exception
        J = p.compute_totals()
        np.testing.assert_array_equal(J['par.C1.y', 'indeps.x'], np.eye(2)*2.)
        np.testing.assert_array_equal(J['par.C2.y', 'indeps.x'], np.zeros((1,2)))
        np.testing.assert_array_equal(J['par.C1.y', 'indeps.y'], np.zeros((2,1)))
        np.testing.assert_array_equal(J['par.C2.y', 'indeps.y'], np.array([[3.]]))


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
