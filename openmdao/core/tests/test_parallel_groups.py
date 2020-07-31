"""Test the parallel groups."""

import unittest
import itertools

# note: this is a Python 3.3 change, clean this up for OpenMDAO 3.x
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy as np

import openmdao.api as om
from openmdao.utils.mpi import MPI

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from openmdao.test_suite.groups.parallel_groups import \
    FanOutGrouped, FanInGrouped2, Diamond, ConvergeDiverge

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.logger_utils import TestLogger
from openmdao.error_checking.check_config import _default_checks


class Noisy(ConvergeDiverge):
    def check_config(self, logger):
        msg = 'Only want to see this on rank 0'
        logger.error(msg)
        logger.warning(msg)
        logger.info(msg)


def _test_func_name(func, num, param):
    args = []
    for p in param.args:
        if not isinstance(p, Iterable):
            p = {p}
        for item in p:
            try:
                arg = item.__name__
            except:
                arg = str(item)
            args.append(arg)
    return func.__name__ + '_' + '_'.join(args)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestParallelGroups(unittest.TestCase):

    N_PROCS = 2

    @parameterized.expand(itertools.product([(om.LinearRunOnce, None)],
                                            [om.NonlinearBlockGS, om.NonlinearRunOnce]),
                          name_func=_test_func_name)
    def test_fan_out_grouped(self, solv_tup, nlsolver):
        prob = om.Problem(FanOutGrouped())

        of=['c2.y', "c3.y"]
        wrt=['iv.x']

        solver, jactype = solv_tup

        prob.model.linear_solver = solver()
        if jactype is not None:
            prob.model.options['assembled_jac_type'] = jactype
        prob.model.nonlinear_solver = nlsolver()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        J = prob.compute_totals(of=['c2.y', "c3.y"], wrt=['iv.x'])

        assert_near_equal(J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_near_equal(J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

        assert_near_equal(prob['c2.y'], -6.0, 1e-6)
        assert_near_equal(prob['c3.y'], 15.0, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        J = prob.compute_totals(of=['c2.y', "c3.y"], wrt=['iv.x'])

        assert_near_equal(J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_near_equal(J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

        assert_near_equal(prob['c2.y'], -6.0, 1e-6)
        assert_near_equal(prob['c3.y'], 15.0, 1e-6)

    @parameterized.expand(itertools.product([om.LinearRunOnce],
                                            [om.NonlinearBlockGS, om.NonlinearRunOnce]),
                          name_func=_test_func_name)
    def test_fan_in_grouped(self, solver, nlsolver):

        prob = om.Problem()
        prob.model = FanInGrouped2()

        prob.model.linear_solver = solver()
        prob.model.nonlinear_solver = nlsolver()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        indep_list = ['p1.x', 'p2.x']
        unknown_list = ['c3.y']

        assert_near_equal(prob['c3.y'], 29.0, 1e-6)

        J = prob.compute_totals(of=unknown_list, wrt=indep_list)
        assert_near_equal(J['c3.y', 'p1.x'][0][0], -6.0, 1e-6)
        assert_near_equal(J['c3.y', 'p2.x'][0][0], 35.0, 1e-6)

        # do this a second time to test caching of dist rows/cols
        J = prob.compute_totals(of=unknown_list, wrt=indep_list)
        assert_near_equal(J['c3.y', 'p1.x'][0][0], -6.0, 1e-6)
        assert_near_equal(J['c3.y', 'p2.x'][0][0], 35.0, 1e-6)

        assert_near_equal(prob['c3.y'], 29.0, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        assert_near_equal(prob['c3.y'], 29.0, 1e-6)

        J = prob.compute_totals(of=unknown_list, wrt=indep_list)
        assert_near_equal(J['c3.y', 'p1.x'][0][0], -6.0, 1e-6)
        assert_near_equal(J['c3.y', 'p2.x'][0][0], 35.0, 1e-6)

        # do this a second time to test caching of dist rows/cols
        J = prob.compute_totals(of=unknown_list, wrt=indep_list)
        assert_near_equal(J['c3.y', 'p1.x'][0][0], -6.0, 1e-6)
        assert_near_equal(J['c3.y', 'p2.x'][0][0], 35.0, 1e-6)

        assert_near_equal(prob['c3.y'], 29.0, 1e-6)

    def test_fan_in_grouped_feature(self):

        import openmdao.api as om

        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', 1.)

        parallel = model.add_subsystem('parallel', om.ParallelGroup(), promotes_inputs=[('c1.x', 'x'), ('c2.x', 'x')])
        parallel.add_subsystem('c1', om.ExecComp(['y=-2.0*x']))
        parallel.add_subsystem('c2', om.ExecComp(['y=5.0*x']))

        model.add_subsystem('c3', om.ExecComp(['y=3.0*x1+7.0*x2']))

        model.connect("parallel.c1.y", "c3.x1")
        model.connect("parallel.c2.y", "c3.x2")

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        assert_near_equal(prob['c3.y'], 29.0, 1e-6)

    @parameterized.expand(itertools.product([om.LinearRunOnce],
                                            [om.NonlinearBlockGS, om.NonlinearRunOnce]),
                          name_func=_test_func_name)
    def test_diamond(self, solver, nlsolver):

        prob = om.Problem()
        prob.model = Diamond()

        prob.model.linear_solver = solver()
        prob.model.nonlinear_solver = nlsolver()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob['c4.y1'], 46.0, 1e-6)
        assert_near_equal(prob['c4.y2'], -93.0, 1e-6)

        indep_list = ['iv.x']
        unknown_list = ['c4.y1', 'c4.y2']

        J = prob.compute_totals(of=unknown_list, wrt=indep_list)
        assert_near_equal(J['c4.y1', 'iv.x'][0][0], 25, 1e-6)
        assert_near_equal(J['c4.y2', 'iv.x'][0][0], -40.5, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        assert_near_equal(prob['c4.y1'], 46.0, 1e-6)
        assert_near_equal(prob['c4.y2'], -93.0, 1e-6)

        J = prob.compute_totals(of=unknown_list, wrt=indep_list)
        assert_near_equal(J['c4.y1', 'iv.x'][0][0], 25, 1e-6)
        assert_near_equal(J['c4.y2', 'iv.x'][0][0], -40.5, 1e-6)

    @parameterized.expand(itertools.product([om.LinearRunOnce],
                                            [om.NonlinearBlockGS, om.NonlinearRunOnce]),
                          name_func=_test_func_name)
    def test_converge_diverge(self, solver, nlsolver):

        prob = om.Problem()
        prob.model = ConvergeDiverge()

        prob.model.linear_solver = solver()
        prob.model.nonlinear_solver = nlsolver()

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob['c7.y1'], -102.7, 1e-6)

        indep_list = ['iv.x']
        unknown_list = ['c7.y1']

        J = prob.compute_totals(of=unknown_list, wrt=indep_list)
        assert_near_equal(J['c7.y1', 'iv.x'][0][0], -40.75, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        assert_near_equal(prob['c7.y1'], -102.7, 1e-6)

        J = prob.compute_totals(of=unknown_list, wrt=indep_list)
        assert_near_equal(J['c7.y1', 'iv.x'][0][0], -40.75, 1e-6)

        assert_near_equal(prob['c7.y1'], -102.7, 1e-6)

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

        prob = om.Problem()

        model = prob.model
        model.add_subsystem('iv', om.IndepVarComp('x', 1.0))
        model.add_subsystem('c1', MultComp(3.0))

        model.sub = model.add_subsystem('sub', om.ParallelGroup())
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

        assert_near_equal(J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_near_equal(J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

        assert_near_equal(prob['c2.y'], -6.0, 1e-6)
        assert_near_equal(prob['c3.y'], 15.0, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        J = prob.compute_totals(of=['c2.y', "c3.y"], wrt=['iv.x'])

        assert_near_equal(J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_near_equal(J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

        assert_near_equal(prob['c2.y'], -6.0, 1e-6)
        assert_near_equal(prob['c3.y'], 15.0, 1e-6)

    def test_setup_messages_bad_vec_type(self):

        prob = om.Problem(Noisy())

        # check that error is thrown if not using PETScVector
        if MPI:
            msg = ("Problem: The `distributed_vector_class` argument must be `PETScVector` when "
                   "running in parallel under MPI but 'DefaultVector' was specified.")
            with self.assertRaises(ValueError) as cm:
                prob.setup(check=False, mode='fwd', distributed_vector_class=om.DefaultVector)

            self.assertEqual(str(cm.exception), msg)
        else:
            prob.setup(check=False, mode='fwd')

    def test_setup_messages_only_on_proc0(self):
        prob = om.Problem(Noisy())

        # check that we get setup messages only on proc 0
        msg = 'Only want to see this on rank 0'
        testlogger = TestLogger()
        prob.setup(check=True, mode='fwd', logger=testlogger)
        prob.final_setup()

        if prob.comm.rank > 0:
            self.assertEqual(len(testlogger.get('error')), 0)
            self.assertEqual(len(testlogger.get('warning')), 0)
            self.assertEqual(len(testlogger.get('info')), 0)
        else:
            self.assertEqual(len(testlogger.get('error')), 1)
            self.assertTrue(testlogger.contains('warning',
                                                "Only want to see this on rank 0"))
            self.assertEqual(len(testlogger.get('info')), len(_default_checks) + 1)
            self.assertTrue(msg in testlogger.get('error')[0])
            for info in testlogger.get('info'):
                if msg in info:
                    break
            else:
                self.fail("Didn't find '%s' in info messages." % msg)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestParallelListStates(unittest.TestCase):

    N_PROCS = 4

    def test_list_states_allprocs(self):
        class StateComp(om.ImplicitComponent):

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

        p = om.Problem()
        par = p.model.add_subsystem('par', om.ParallelGroup())
        par.add_subsystem('C1', StateComp())
        par.add_subsystem('C2', StateComp())
        par.add_subsystem('C3', om.ExecComp('y=2.0*x'))
        par.add_subsystem('C4', StateComp())

        p.setup()
        p.final_setup()
        self.assertEqual(sorted(p.model._list_states_allprocs()), ['par.C1.x', 'par.C2.x', 'par.C4.x'])


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class MatMatParDevTestCase(unittest.TestCase):
    N_PROCS = 2

    def test_size_1_matmat(self):
        p = om.Problem()
        indeps = p.model.add_subsystem('indeps', om.IndepVarComp('x', np.ones(2)))
        indeps.add_output('y', 1.0)
        par = p.model.add_subsystem('par', om.ParallelGroup())
        par.add_subsystem('C1', om.ExecComp('y=2*x', x=np.zeros(2), y=np.zeros(2)))
        par.add_subsystem('C2', om.ExecComp('y=3*x'))
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


class ExComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        # Inputs
        self.add_input('accel', val=np.zeros(nn))
        self.add_output('deltav_dot', val=np.zeros(nn))
        # Setup partials
        ar = np.arange(self.options['num_nodes'])
        self.declare_partials(of='deltav_dot', wrt='accel', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        outputs['deltav_dot'] = inputs['accel']


class SubGroup(om.Group):
    def __init__(self, size, **kwargs):
        super(SubGroup, self).__init__(**kwargs)
        self.size = size

    def setup(self):
        ivc = om.IndepVarComp()
        ivc.add_output('accel', val=np.ones(self.size))
        self.add_subsystem('rhs', ivc)
        self.add_subsystem('ode', ExComp(num_nodes=self.size))
        self.connect('rhs.accel', 'ode.accel')
        self.add_design_var('rhs.accel', 3.0)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestParallelJacBug(unittest.TestCase):

    N_PROCS = 2

    def test_par_jac_bug(self):

        p = om.Problem()
        model = p.model
        par = model.add_subsystem('par', om.ParallelGroup())
        par.add_subsystem('p1', SubGroup(1))
        par.add_subsystem('p2', SubGroup(1))
        p.setup(mode='rev')
        p.run_model()
        J1 = p.driver._compute_totals(of=['par.p1.ode.deltav_dot'], wrt=['par.p1.ode.deltav_dot'],
                                      return_format='array')
        Jsave = J1.copy()
        J2 = p.driver._compute_totals(of=['par.p1.ode.deltav_dot'], wrt=['par.p1.ode.deltav_dot'],
                                      return_format='array')

        self.assertLess(np.max(np.abs(J2 - Jsave)), 1e-20)


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
