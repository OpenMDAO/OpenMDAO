
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

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.logger_utils import TestLogger
from openmdao.error_checking.check_config import _default_checks
from openmdao.core.tests.test_distrib_derivs import DistribExecComp

def _test_func_name(func, num, param):
    args = []
    for p in param.args:
        if isinstance(p, str):
            p = {p}
        elif not isinstance(p, Iterable):
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

    @parameterized.expand(itertools.product(['fwd', 'rev'], [True, False]),
                          name_func=_test_func_name)
    def test_dup_dup(self, mode, auto):
        # duplicated vars on both ends
        prob = om.Problem()
        model = prob.model

        if not auto:
            model.add_subsystem('indep', om.IndepVarComp('x', 1.0))

        model.add_subsystem('C1', om.ExecComp('y = 2.5 * x'))

        if auto:
            wrt = ['C1.x']
        else:
            model.connect('indep.x', 'C1.x')
            wrt=['indep.x']

        of=['C1.y']

        prob.model.linear_solver = om.LinearRunOnce()

        prob.setup(check=False, mode=mode)
        prob.set_solver_print(level=0)
        prob.run_model()

        J = prob.compute_totals(of=of, wrt=wrt)

        assert_near_equal(J['C1.y', wrt[0]][0][0], 2.5, 1e-6)
        assert_near_equal(prob['C1.y'], 2.5, 1e-6)

    @parameterized.expand(itertools.product(['fwd', 'rev'], [True, False]),
                          name_func=_test_func_name)
    def test_dup_par(self, mode, auto):
        # duplicated output, parallel input
        prob = om.Problem()
        model = prob.model

        if not auto:
            model.add_subsystem('indep', om.IndepVarComp('x', 1.0), promotes=['x'])

        par = model.add_subsystem('par', om.ParallelGroup(), promotes=['x'])
        par.add_subsystem('C1', om.ExecComp('y = 2.5 * x'), promotes=['x'])
        par.add_subsystem('C2', om.ExecComp('y = 7 * x'), promotes=['x'])

        wrt = ['x']

        of=['par.C1.y', 'par.C2.y']

        prob.model.linear_solver = om.LinearRunOnce()

        prob.setup(check=False, mode=mode)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('par.C1.y', get_remote=True), 2.5, 1e-6)
        assert_near_equal(prob.get_val('par.C2.y', get_remote=True), 7., 1e-6)

        J = prob.compute_totals(of=of, wrt=wrt)

        assert_near_equal(J['par.C1.y', 'x'][0][0], 2.5, 1e-6)
        assert_near_equal(J['par.C2.y', 'x'][0][0], 7., 1e-6)

        assert_near_equal(prob.get_val('par.C1.y', get_remote=True), 2.5, 1e-6)
        assert_near_equal(prob.get_val('par.C2.y', get_remote=True), 7., 1e-6)

    @parameterized.expand(itertools.product(['fwd', 'rev'], [True, False]),
                          name_func=_test_func_name)
    def test_dup_dup_and_par(self, mode, auto):
        # duplicated and parallel outputs, dup input
        prob = om.Problem()
        model = prob.model

        if not auto:
            model.add_subsystem('indep', om.IndepVarComp('x', 1.0), promotes=['x'])

        model.add_subsystem('dup', om.ExecComp('y = 1.5 * x'), promotes=['x'])
        par = model.add_subsystem('par', om.ParallelGroup(), promotes=['x'])
        par.add_subsystem('C1', om.ExecComp('y = 2.5 * x'), promotes=['x'])
        par.add_subsystem('C2', om.ExecComp('y = 7 * x'), promotes=['x'])

        of=['par.C1.y', 'par.C2.y', 'dup.y']
        wrt=['x']

        prob.model.linear_solver = om.LinearRunOnce()

        prob.setup(check=False, mode=mode)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('par.C1.y', get_remote=True), 2.5, 1e-6)
        assert_near_equal(prob.get_val('par.C2.y', get_remote=True), 7., 1e-6)
        assert_near_equal(prob.get_val('dup.y', get_remote=True), 1.5, 1e-6)

        J = prob.compute_totals(of=of, wrt=wrt)

        assert_near_equal(J['par.C1.y', 'x'][0][0], 2.5, 1e-6)
        assert_near_equal(prob.get_val('par.C1.y', get_remote=True), 2.5, 1e-6)
        assert_near_equal(J['par.C2.y', 'x'][0][0], 7., 1e-6)
        assert_near_equal(prob.get_val('par.C2.y', get_remote=True), 7., 1e-6)
        assert_near_equal(J['dup.y', 'x'][0][0], 1.5, 1e-6)
        assert_near_equal(prob.get_val('dup.y', get_remote=True), 1.5, 1e-6)

    def test_dup_par_par_derivs(self):
        # duplicated output, parallel input
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('indep', om.IndepVarComp('x', 1.0), promotes=['x'])

        par = model.add_subsystem('par', om.ParallelGroup(), promotes=['x'])
        par.add_subsystem('C1', om.ExecComp('y = 2.5 * x'), promotes=['x'])
        par.add_subsystem('C2', om.ExecComp('y = 7 * x'), promotes=['x'])

        model.add_design_var('x')

        model.add_constraint('par.C1.y', upper=0.0, parallel_deriv_color='parc')
        model.add_constraint('par.C2.y', upper=0.0, parallel_deriv_color='parc')

        prob.model.linear_solver = om.LinearBlockGS()

        prob.setup(check=False, mode='rev')
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('par.C1.y', get_remote=True), 2.5, 1e-6)
        assert_near_equal(prob.get_val('par.C2.y', get_remote=True), 7., 1e-6)

        J = prob.driver._compute_totals()

        assert_near_equal(J['par.C1.y', 'indep.x'][0][0], 2.5, 1e-6)
        assert_near_equal(prob.get_val('par.C1.y', get_remote=True), 2.5, 1e-6)
        assert_near_equal(J['par.C2.y', 'indep.x'][0][0], 7., 1e-6)
        assert_near_equal(prob.get_val('par.C2.y', get_remote=True), 7., 1e-6)

    @parameterized.expand(itertools.product(['fwd', 'rev'], [False]),
                          name_func=_test_func_name)
    def test_dup_dist(self, mode, auto):
        # Note: Auto-ivc not supported for distributed inputs.

        # duplicated output, parallel input
        prob = om.Problem()
        model = prob.model
        size = 3

        sizes = [2, 1]
        rank = prob.comm.rank

        if not auto:
            model.add_subsystem('indep', om.IndepVarComp('x', np.ones(size)), promotes=['x'])

        model.add_subsystem('C1', DistribExecComp(['y=2.5*x', 'y=3.5*x'], arr_size=size), promotes=['x'])

        of=['C1.y']
        wrt=['x']

        prob.model.linear_solver = om.LinearRunOnce()

        prob.setup(check=False, mode=mode)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('C1.y', get_remote=True),
                         np.array([2.5,2.5,3.5], dtype=float), 1e-6)

        J = prob.compute_totals(of=of, wrt=wrt)

        expected = np.array([[2.5, 0, 0], [0, 2.5, 0], [0,0,3.5]], dtype=float)

        assert_near_equal(J['C1.y', 'x'], expected, 1e-6)
        assert_near_equal(prob.get_val('C1.y', get_remote=True),
                         np.array([2.5,2.5,3.5], dtype=float), 1e-6)

    @parameterized.expand(itertools.product(['fwd', 'rev']),
                          name_func=_test_func_name)
    def test_par_dup(self, mode):
        # duplicated output, parallel input
        prob = om.Problem()
        model = prob.model

        par = model.add_subsystem('par', om.ParallelGroup())
        par.add_subsystem('indep1', om.IndepVarComp('x', 1.0))
        par.add_subsystem('indep2', om.IndepVarComp('x', 1.0))
        model.add_subsystem('C1', om.ExecComp('y = 2.5 * x1 + 3.5 * x2'))
        model.connect('par.indep1.x', 'C1.x1')
        model.connect('par.indep2.x', 'C1.x2')

        of=['C1.y']
        wrt=['par.indep1.x', 'par.indep2.x']

        prob.model.linear_solver = om.LinearRunOnce()

        prob.setup(check=False, mode=mode)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob['C1.y'], 6., 1e-6)

        J = prob.compute_totals(of=of, wrt=wrt)

        assert_near_equal(J['C1.y', 'par.indep1.x'][0][0], 2.5, 1e-6)
        assert_near_equal(J['C1.y', 'par.indep2.x'][0][0], 3.5, 1e-6)
        assert_near_equal(prob['C1.y'], 6., 1e-6)

    @parameterized.expand(itertools.product(['fwd', 'rev'], [False]),
                          name_func=_test_func_name)
    def test_dist_dup(self, mode, auto):
        # duplicated output, parallel input
        # Note: Auto-ivc not supported for distributed inputs.
        prob = om.Problem()
        model = prob.model
        size = 3

        rank = prob.comm.rank
        if not auto:
            model.add_subsystem('indep', om.IndepVarComp('x', np.ones(size)), promotes=['x'])
        model.add_subsystem('C1', DistribExecComp(['y=2.5*x', 'y=3.5*x'], arr_size=size), promotes=['x'])
        model.add_subsystem('sink', om.ExecComp('y=-1.5 * x', x=np.zeros(size), y=np.zeros(size)))

        model.connect('C1.y', 'sink.x', src_indices=om.slicer[:])

        of=['sink.y']
        wrt=['x']

        prob.model.linear_solver = om.LinearRunOnce()

        prob.setup(check=False, mode=mode)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('sink.y', get_remote=True),
                         np.array([-3.75,-3.75,-5.25], dtype=float), 1e-6)

        J = prob.compute_totals(of=of, wrt=wrt)

        expected = np.array([[-3.75, 0, 0], [0, -3.75, 0], [0,0,-5.25]], dtype=float)

        assert_near_equal(J['sink.y', 'x'], expected, 1e-6)
        assert_near_equal(prob.get_val('sink.y', get_remote=True),
                         np.array([-3.75,-3.75,-5.25], dtype=float), 1e-6)

    @parameterized.expand(itertools.product(['fwd', 'rev'], [True, False]),
                          name_func=_test_func_name)
    def test_par_dist(self, mode, auto):
        # duplicated output, parallel input
        prob = om.Problem()
        model = prob.model
        size = 3

        sizes = [2, 1]
        rank = prob.comm.rank
        model.add_subsystem('indep', om.IndepVarComp('x', np.ones(size)))
        par = model.add_subsystem('par', om.ParallelGroup())
        par.add_subsystem('C1', om.ExecComp('y = 3 * x', x=np.zeros(size), y=np.zeros(size)))
        par.add_subsystem('C2', om.ExecComp('y = 5 * x', x=np.zeros(size), y=np.zeros(size)))
        model.add_subsystem('C3', DistribExecComp(['y=1.5*x1+2.5*x2', 'y=2.5*x1-.5*x2'], arr_size=size))

        model.connect('indep.x', 'par.C1.x')
        model.connect('indep.x', 'par.C2.x')
        model.connect('par.C1.y', 'C3.x1')
        model.connect('par.C2.y', 'C3.x2')

        of=['C3.y']
        wrt=['indep.x']

        prob.model.linear_solver = om.LinearRunOnce()

        prob.setup(check=False, mode=mode)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob.get_val('C3.y', get_remote=True),
                         np.array([17,17,5], dtype=float), 1e-6)

        J = prob.compute_totals(of=of, wrt=wrt)

        expected = np.array([[17, 0, 0], [0, 17, 0], [0,0,5]], dtype=float)

        assert_near_equal(J['C3.y', 'indep.x'], expected, 1e-6)
        assert_near_equal(prob.get_val('C3.y', get_remote=True),
                         np.array([17,17,5], dtype=float), 1e-6)

    @parameterized.expand(itertools.product(['fwd', 'rev'], [True, False]),
                          name_func=_test_func_name)
    def test_crossover(self, mode, auto):
        # multiple crossovers in fwd and rev
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('ivc', om.IndepVarComp('x'))

        par1 = model.add_subsystem('par1', om.ParallelGroup())
        par1.add_subsystem('C1', om.ExecComp('y = 1.5 * x'))
        par1.add_subsystem('C2', om.ExecComp('y = 2.5 * x'))

        model.add_subsystem('C3', om.ExecComp('y = 3.5 * x1 - .5 * x2'))

        par2 = model.add_subsystem('par2', om.ParallelGroup())
        par2.add_subsystem('C4', om.ExecComp('y = 4.5 * x'))
        par2.add_subsystem('C5', om.ExecComp('y = 5.5 * x'))

        model.add_subsystem('C6', om.ExecComp('y = 6.5 * x1 + 1.1 * x2'))

        model.connect('ivc.x', 'par1.C1.x')
        model.connect('ivc.x', 'par1.C2.x')
        model.connect('par1.C1.y', 'C3.x1')
        model.connect('par1.C2.y', 'C3.x2')
        model.connect('C3.y', 'par2.C4.x')
        model.connect('C3.y', 'par2.C5.x')
        model.connect('par2.C4.y', 'C6.x1')
        model.connect('par2.C5.y', 'C6.x2')

        of = ['C6.y']
        wrt = ['ivc.x']

        prob.setup(check=False, mode=mode)
        prob.set_solver_print(level=0)
        prob.run_model()

        np.testing.assert_allclose(prob.get_val('C6.y', get_remote=True),
                                   141.2)

        J = prob.compute_totals(of=of, wrt=wrt)
        print(J)

        np.testing.assert_allclose(J['C6.y', 'ivc.x'][0][0], 141.2)
        np.testing.assert_allclose(prob.get_val('C6.y', get_remote=True),
                                   141.2)

