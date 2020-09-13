""" Test out some crucial linear GS tests in parallel with distributed comps."""

import unittest
import itertools

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.paraboloid_distributed import DistParab
from openmdao.utils.mpi import MPI
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.utils.assert_utils import assert_near_equal

try:
    from pyoptsparse import Optimization as pyoptsparse_opt
except ImportError:
    pyoptsparse_opt = None

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

if MPI:
    rank = MPI.COMM_WORLD.rank
else:
    rank = 0


class DistribExecComp(om.ExecComp):
    """
    An ExecComp that uses N procs and takes input var slices.  Unlike a normal
    ExecComp, it only supports a single expression per proc.  If you give it
    multiple expressions, it will use a different one in each proc, repeating
    the last one in any remaining procs.
    """

    def __init__(self, exprs, arr_size=11, **kwargs):
        super(DistribExecComp, self).__init__(exprs, **kwargs)
        self.arr_size = arr_size
        self.options['distributed'] = True

    def setup(self):
        outs = set()
        allvars = set()
        exprs = self._exprs
        kwargs = self._kwargs

        comm = self.comm
        rank = comm.rank

        if len(self._exprs) > comm.size:
            raise RuntimeError("DistribExecComp only supports up to 1 expression per MPI process.")

        if len(self._exprs) < comm.size:
            # repeat the last expression for any leftover procs
            self._exprs.extend([self._exprs[-1]] * (comm.size - len(self._exprs)))

        self._exprs = [self._exprs[rank]]

        # find all of the variables and which ones are outputs
        for expr in exprs:
            lhs, _ = expr.split('=', 1)
            outs.update(self._parse_for_out_vars(lhs))
            allvars.update(self._parse_for_vars(expr))

        sizes, offsets = evenly_distrib_idxs(comm.size, self.arr_size)
        start = offsets[rank]
        end = start + sizes[rank]

        for name in outs:
            if name not in kwargs or not isinstance(kwargs[name], dict):
                kwargs[name] = {}
            kwargs[name]['value'] = np.ones(sizes[rank], float)

        for name in allvars:
            if name not in outs:
                if name not in kwargs or not isinstance(kwargs[name], dict):
                    kwargs[name] = {}
                meta = kwargs[name]
                meta['value'] = np.ones(sizes[rank], float)
                meta['src_indices'] = np.arange(start, end, dtype=int)

        super(DistribExecComp, self).setup()


class DistribCoordComp(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super(DistribCoordComp, self).__init__(**kwargs)

        self.options['distributed'] = True

    def setup(self):
        comm = self.comm
        rank = comm.rank

        if rank == 0:
            self.add_input('invec', np.zeros((5, 3)),
                           src_indices=[[(0, 0), (0, 1), (0, 2)],
                                        [(1, 0), (1, 1), (1, 2)],
                                        [(2, 0), (2, 1), (2, 2)],
                                        [(3, 0), (3, 1), (3, 2)],
                                        [(4, 0), (4, 1), (4, 2)]])
            self.add_output('outvec', np.zeros((5, 3)))
        else:
            self.add_input('invec', np.zeros((4, 3)),
                           src_indices=[[(5, 0), (5, 1), (5, 2)],
                                        [(6, 0), (6, 1), (6, 2)],
                                        [(7, 0), (7, 1), (7, 2)],
                                        # use some negative indices here to
                                        # make sure they work
                                        [(-1, 0), (8, 1), (-1, 2)]])
            self.add_output('outvec', np.zeros((4, 3)))

    def compute(self, inputs, outputs):
        if self.comm.rank == 0:
            outputs['outvec'] = inputs['invec'] * 2.0
        else:
            outputs['outvec'] = inputs['invec'] * 3.0


def _test_func_name(func, num, param):
    args = []
    for p in param.args:
        try:
            arg = p.__name__
        except:
            arg = str(p)
        args.append(arg)
    return func.__name__ + '_' + '_'.join(args)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class MPITests2(unittest.TestCase):

    N_PROCS = 2

    def test_distrib_shape(self):
        points = np.array([
            [0., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 1., 1.],
            [1., 0., 0.],

            [1., 0., 1.],
            [1., 1., 0.],
            [1., 1., 1.],
            [0., 0., 2.],
        ])

        prob = om.Problem()

        prob.model.add_subsystem('indep', om.IndepVarComp('x', points))
        prob.model.add_subsystem('comp', DistribCoordComp())
        prob.model.add_subsystem('total', om.ExecComp('y=x',
                                                   x=np.zeros((9, 3)),
                                                   y=np.zeros((9, 3))))
        prob.model.connect('indep.x', 'comp.invec')
        prob.model.connect('comp.outvec', 'total.x')

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        final = points.copy()
        final[0:5] *= 2.0
        final[5:9] *= 3.0

        assert_near_equal(prob['total.y'], final)

    def test_two_simple(self):
        size = 3
        group = om.Group()

        group.add_subsystem('P', om.IndepVarComp('x', np.arange(size)),
                            promotes_outputs=['x'])
        group.add_subsystem('C1', DistribExecComp(['y=2.0*x', 'y=3.0*x'], arr_size=size,
                                                  x=np.zeros(size),
                                                  y=np.zeros(size)),
                            promotes_inputs=['x'])
        group.add_subsystem('C2', om.ExecComp(['z=3.0*y'],
                                           y=np.zeros(size),
                                           z=np.zeros(size)))

        prob = om.Problem()
        prob.model = group
        prob.model.linear_solver = om.LinearBlockGS()
        prob.model.connect('C1.y', 'C2.y')


        prob.setup(check=False, mode='fwd')
        prob.run_model()

        J = prob.compute_totals(['C2.z'], ['x'])
        assert_near_equal(J['C2.z', 'x'], np.diag([6.0, 6.0, 9.0]), 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        J = prob.compute_totals(['C2.z'], ['x'])
        assert_near_equal(J['C2.z', 'x'], np.diag([6.0, 6.0, 9.0]), 1e-6)

    @parameterized.expand(itertools.product([om.NonlinearRunOnce, om.NonlinearBlockGS]),
                          name_func=_test_func_name)
    def test_fan_out_grouped(self, nlsolver):
        size = 3
        prob = om.Problem()
        prob.model = root = om.Group()
        root.add_subsystem('P', om.IndepVarComp('x', np.ones(size, dtype=float)))
        root.add_subsystem('C1', DistribExecComp(['y=3.0*x', 'y=2.0*x'], arr_size=size,
                                                 x=np.zeros(size, dtype=float),
                                                 y=np.zeros(size, dtype=float)))
        sub = root.add_subsystem('sub', om.ParallelGroup())
        sub.add_subsystem('C2', om.ExecComp('y=1.5*x',
                                         x=np.zeros(size),
                                         y=np.zeros(size)))
        sub.add_subsystem('C3', om.ExecComp(['y=5.0*x'],
                                         x=np.zeros(size, dtype=float),
                                         y=np.zeros(size, dtype=float)))

        root.add_subsystem('C2', om.ExecComp(['y=x'],
                                          x=np.zeros(size, dtype=float),
                                          y=np.zeros(size, dtype=float)))
        root.add_subsystem('C3', om.ExecComp(['y=x'],
                                          x=np.zeros(size, dtype=float),
                                          y=np.zeros(size, dtype=float)))
        root.connect('sub.C2.y', 'C2.x')
        root.connect('sub.C3.y', 'C3.x')

        root.connect("C1.y", "sub.C2.x")
        root.connect("C1.y", "sub.C3.x")
        root.connect("P.x", "C1.x")

        root.nonlinear_solver = nlsolver()

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        diag1 = [4.5, 4.5, 3.0]
        diag2 = [15.0, 15.0, 10.0]

        assert_near_equal(prob['C2.y'], diag1)
        assert_near_equal(prob['C3.y'], diag2)

        diag1 = np.diag(diag1)
        diag2 = np.diag(diag2)

        J = prob.compute_totals(of=['C2.y', "C3.y"], wrt=['P.x'])
        assert_near_equal(J['C2.y', 'P.x'], diag1, 1e-6)
        assert_near_equal(J['C3.y', 'P.x'], diag2, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        J = prob.compute_totals(of=['C2.y', "C3.y"], wrt=['P.x'])
        assert_near_equal(J['C2.y', 'P.x'], diag1, 1e-6)
        assert_near_equal(J['C3.y', 'P.x'], diag2, 1e-6)

    @parameterized.expand(itertools.product([om.NonlinearRunOnce, om.NonlinearBlockGS]),
                          name_func=_test_func_name)
    def test_fan_in_grouped(self, nlsolver):
        size = 3

        prob = om.Problem()
        prob.model = root = om.Group()

        root.add_subsystem('P1', om.IndepVarComp('x', np.ones(size, dtype=float)))
        root.add_subsystem('P2', om.IndepVarComp('x', np.ones(size, dtype=float)))
        sub = root.add_subsystem('sub', om.ParallelGroup())

        sub.add_subsystem('C1', om.ExecComp(['y=-2.0*x'],
                                         x=np.zeros(size, dtype=float),
                                         y=np.zeros(size, dtype=float)))
        sub.add_subsystem('C2', om.ExecComp(['y=5.0*x'],
                                         x=np.zeros(size, dtype=float),
                                         y=np.zeros(size, dtype=float)))
        root.add_subsystem('C3', DistribExecComp(['y=3.0*x1+7.0*x2', 'y=1.5*x1+3.5*x2'],
                                                 arr_size=size,
                                                 x1=np.zeros(size, dtype=float),
                                                 x2=np.zeros(size, dtype=float),
                                                 y=np.zeros(size, dtype=float)))
        root.add_subsystem('C4', om.ExecComp(['y=x'],
                                          x=np.zeros(size, dtype=float),
                                          y=np.zeros(size, dtype=float)))

        root.connect("sub.C1.y", "C3.x1")
        root.connect("sub.C2.y", "C3.x2")
        root.connect("P1.x", "sub.C1.x")
        root.connect("P2.x", "sub.C2.x")
        root.connect("C3.y", "C4.x")

        root.nonlinear_solver = nlsolver()

        prob.set_solver_print(0)
        prob.setup(mode='fwd')
        prob.run_driver()

        diag1 = np.diag([-6.0, -6.0, -3.0])
        diag2 = np.diag([35.0, 35.0, 17.5])

        J = prob.compute_totals(of=['C4.y'], wrt=['P1.x', 'P2.x'])
        assert_near_equal(J['C4.y', 'P1.x'], diag1, 1e-6)
        assert_near_equal(J['C4.y', 'P2.x'], diag2, 1e-6)

        prob.setup(check=False, mode='rev')

        prob.run_driver()

        J = prob.compute_totals(of=['C4.y'], wrt=['P1.x', 'P2.x'])
        assert_near_equal(J['C4.y', 'P1.x'], diag1, 1e-6)
        assert_near_equal(J['C4.y', 'P2.x'], diag2, 1e-6)

    def test_distrib_voi_dense(self):
        size = 7

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', np.ones((size, )))
        ivc.add_output('y', np.ones((size, )))
        ivc.add_output('a', -3.0 + 0.6 * np.arange(size))

        model.add_subsystem('p', ivc, promotes=['*'])
        model.add_subsystem("parab", DistParab(arr_size=size, deriv_type='dense'), promotes=['*'])
        model.add_subsystem('sum', om.ExecComp('f_sum = sum(f_xy)',
                                               f_sum=np.ones((size, )),
                                               f_xy=np.ones((size, ))),
                            promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_constraint('f_xy', lower=0.0)
        model.add_objective('f_sum', index=-1)

        prob.setup(mode='fwd', force_alloc_complex=True)

        prob.run_model()

        desvar = prob.driver.get_design_var_values()
        con = prob.driver.get_constraint_values()

        assert_near_equal(desvar['p.x'], np.ones(size), 1e-6)
        assert_near_equal(con['parab.f_xy'],
                          np.array([27.0, 24.96, 23.64, 23.04, 23.16, 24.0, 25.56]),
                          1e-6)

        J = prob.check_totals(method='fd')
        assert_near_equal(J['parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-5)

        J = prob.check_totals(method='cs')
        assert_near_equal(J['parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-14)
        assert_near_equal(J['parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-14)
        assert_near_equal(J['sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-14)
        assert_near_equal(J['sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-14)

        # rev mode

        prob.setup(mode='rev', force_alloc_complex=True)

        prob.run_model()

        desvar = prob.driver.get_design_var_values()
        con = prob.driver.get_constraint_values()

        assert_near_equal(desvar['p.x'], np.ones(size), 1e-6)
        assert_near_equal(con['parab.f_xy'],
                          np.array([27.0, 24.96, 23.64, 23.04, 23.16, 24.0, 25.56]),
                          1e-6)

        J = prob.check_totals(method='fd')
        assert_near_equal(J['parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-5)

        J = prob.check_totals(method='cs')
        assert_near_equal(J['parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-14)
        assert_near_equal(J['parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-14)
        assert_near_equal(J['sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-14)
        assert_near_equal(J['sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-14)

    def test_distrib_voi_sparse(self):
        size = 7

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', np.ones((size, )))
        ivc.add_output('y', np.ones((size, )))
        ivc.add_output('a', -3.0 + 0.6 * np.arange(size))

        model.add_subsystem('p', ivc, promotes=['*'])
        model.add_subsystem("parab", DistParab(arr_size=size, deriv_type='sparse'), promotes=['*'])
        model.add_subsystem('sum', om.ExecComp('f_sum = sum(f_xy)',
                                               f_sum=np.ones((size, )),
                                               f_xy=np.ones((size, ))),
                            promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_constraint('f_xy', lower=0.0)
        model.add_objective('f_sum', index=-1)

        prob.setup(mode='fwd', force_alloc_complex=True)

        prob.run_model()

        desvar = prob.driver.get_design_var_values()
        con = prob.driver.get_constraint_values()

        assert_near_equal(desvar['p.x'], np.ones(size), 1e-6)
        assert_near_equal(con['parab.f_xy'],
                          np.array([27.0, 24.96, 23.64, 23.04, 23.16, 24.0, 25.56]),
                          1e-6)

        J = prob.check_totals(method='fd')
        assert_near_equal(J['parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-5)

        J = prob.check_totals(method='cs')
        assert_near_equal(J['parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-14)
        assert_near_equal(J['parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-14)
        assert_near_equal(J['sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-14)
        assert_near_equal(J['sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-14)

        # rev mode

        prob.setup(mode='rev', force_alloc_complex=True)

        prob.run_model()

        desvar = prob.driver.get_design_var_values()
        con = prob.driver.get_constraint_values()

        assert_near_equal(desvar['p.x'], np.ones(size), 1e-6)
        assert_near_equal(con['parab.f_xy'],
                          np.array([27.0, 24.96, 23.64, 23.04, 23.16, 24.0, 25.56]),
                          1e-6)

        J = prob.check_totals(method='fd')
        assert_near_equal(J['parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-5)

        J = prob.check_totals(method='cs')
        assert_near_equal(J['parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-14)
        assert_near_equal(J['parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-14)
        assert_near_equal(J['sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-14)
        assert_near_equal(J['sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-14)

    def test_distrib_voi_fd(self):
        size = 7

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', np.ones((size, )))
        ivc.add_output('y', np.ones((size, )))
        ivc.add_output('a', -3.0 + 0.6 * np.arange(size))

        model.add_subsystem('p', ivc, promotes=['*'])
        model.add_subsystem("parab", DistParab(arr_size=size, deriv_type='fd'), promotes=['*'])
        model.add_subsystem('sum', om.ExecComp('f_sum = sum(f_xy)',
                                               f_sum=np.ones((size, )),
                                               f_xy=np.ones((size, ))),
                            promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_constraint('f_xy', lower=0.0)
        model.add_objective('f_sum', index=-1)

        prob.setup(mode='fwd', force_alloc_complex=True)

        prob.run_model()

        desvar = prob.driver.get_design_var_values()
        con = prob.driver.get_constraint_values()

        assert_near_equal(desvar['p.x'], np.ones(size), 1e-6)
        assert_near_equal(con['parab.f_xy'],
                          np.array([27.0, 24.96, 23.64, 23.04, 23.16, 24.0, 25.56]),
                          1e-6)

        J = prob.check_totals(method='cs')
        assert_near_equal(J['parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-5)

        # rev mode

        prob.setup(mode='rev', force_alloc_complex=True)

        prob.run_model()

        desvar = prob.driver.get_design_var_values()
        con = prob.driver.get_constraint_values()

        assert_near_equal(desvar['p.x'], np.ones(size), 1e-6)
        assert_near_equal(con['parab.f_xy'],
                          np.array([27.0, 24.96, 23.64, 23.04, 23.16, 24.0, 25.56]),
                          1e-6)

        J = prob.check_totals(method='cs')
        assert_near_equal(J['parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-5)

    def test_distrib_voi_group_fd(self):
        # Only supports groups where the inputs to the distributed component whose inputs are
        # distributed to procs via src_indices don't cross the boundary.
        size = 7

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', np.ones((size, )))
        ivc.add_output('y', np.ones((size, )))

        model.add_subsystem('p', ivc, promotes=['*'])
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

        ivc2 = om.IndepVarComp()
        ivc2.add_output('a', -3.0 + 0.6 * np.arange(size))

        sub.add_subsystem('p2', ivc2, promotes=['*'])
        sub.add_subsystem('dummy', om.ExecComp(['xd = x', "yd = y"],
                                               x=np.ones(size), xd=np.ones(size),
                                               y=np.ones(size), yd=np.ones(size)),
                          promotes_inputs=['*'])

        sub.add_subsystem("parab", DistParab(arr_size=size), promotes_outputs=['*'], promotes_inputs=['a'])
        sub.add_subsystem('sum', om.ExecComp('f_sum = sum(f_xy)',
                                             f_sum=np.ones((size, )),
                                             f_xy=np.ones((size, ))),
                          promotes=['*'])

        sub.connect('dummy.xd', 'parab.x')
        sub.connect('dummy.yd', 'parab.y')

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_constraint('f_xy', lower=0.0)
        model.add_objective('f_sum', index=-1)

        sub.approx_totals(method='fd')

        prob.setup(mode='fwd', force_alloc_complex=True)

        prob.run_model()

        desvar = prob.driver.get_design_var_values()
        con = prob.driver.get_constraint_values()

        assert_near_equal(desvar['p.x'], np.ones(size), 1e-6)
        assert_near_equal(con['sub.parab.f_xy'],
                          np.array([27.0, 24.96, 23.64, 23.04, 23.16, 24.0, 25.56]),
                          1e-6)

        J = prob.check_totals(method='fd')
        assert_near_equal(J['sub.parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sub.parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sub.sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sub.sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-5)

        # rev mode

        prob.setup(mode='rev', force_alloc_complex=True)

        prob.run_model()

        desvar = prob.driver.get_design_var_values()
        con = prob.driver.get_constraint_values()

        assert_near_equal(desvar['p.x'], np.ones(size), 1e-6)
        assert_near_equal(con['sub.parab.f_xy'],
                          np.array([27.0, 24.96, 23.64, 23.04, 23.16, 24.0, 25.56]),
                          1e-6)

        J = prob.check_totals(method='fd')
        assert_near_equal(J['sub.parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sub.parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sub.sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sub.sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-5)

    def test_distrib_group_fd_unsupported_config(self):
        size = 7

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', np.ones(size))
        ivc.add_output('y', np.ones(size))
        ivc.add_output('a', -3.0 + 0.6 * np.arange(size))

        model.add_subsystem('p', ivc, promotes=['*'])
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

        sub.add_subsystem("parab", DistParab(arr_size=size), promotes=['*'])
        sub.add_subsystem('sum', om.ExecComp('f_sum = sum(f_xy)',
                                             f_sum=np.ones((size, )),
                                             f_xy=np.ones((size, ))),
                          promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_constraint('f_xy', lower=0.0)
        model.add_objective('f_sum', index=-1)

        sub.approx_totals(method='fd')

        prob.setup(force_alloc_complex=True)

        with self.assertRaises(RuntimeError) as context:
            prob.run_model()

        msg = "Group (sub) : Approx_totals is not supported on a group with a distributed "
        msg += "component whose input 'sub.parab.x' is distributed using src_indices. "
        self.assertEqual(str(context.exception), msg)

    def test_distrib_voi_multiple_con(self):
        # This test contains 2 distributed constraints and 2 global ones.
        class NonDistComp(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('arr_size', types=int, default=10,
                                     desc="Size of input and output vectors.")

            def setup(self):
                arr_size = self.options['arr_size']

                self.add_input('f_xy', val=np.ones(arr_size))
                self.add_output('g', val=np.ones(arr_size))

                self.mat = np.array([3.0, -1, 5, 7, 13, 11, -3])[:arr_size]

                row_col = np.arange(arr_size)
                self.declare_partials('g', ['f_xy'], rows=row_col, cols=row_col, val=self.mat)

            def compute(self, inputs, outputs):
                x = inputs['f_xy']
                outputs['g'] = x * self.mat

        size = 7
        size2 = 5

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', np.ones((size, )))
        ivc.add_output('y', np.ones((size, )))
        ivc.add_output('a', -3.0 + 0.6 * np.arange(size))
        ivc.add_output('x2', np.ones(size2))
        ivc.add_output('y2', np.ones(size2))
        ivc.add_output('a2', -4.0 + 0.4 * np.arange(size2))

        model.add_subsystem('p', ivc, promotes=['*'])
        model.add_subsystem("parab", DistParab(arr_size=size), promotes=['*'])
        model.add_subsystem("ndp", NonDistComp(arr_size=size), promotes=['*'])
        model.add_subsystem("parab2", DistParab(arr_size=size2))
        model.add_subsystem("ndp2", NonDistComp(arr_size=size2))

        model.add_subsystem('sum', om.ExecComp('f_sum = sum(f_xy)',
                                               f_sum=np.ones((size, )),
                                               f_xy=np.ones((size, ))),
                            promotes=['*'])

        model.connect('x2', 'parab2.x')
        model.connect('y2', 'parab2.y')
        model.connect('a2', 'parab2.a')
        model.connect('parab2.f_xy', 'ndp2.f_xy')

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_design_var('x2', lower=-50.0, upper=50.0)
        model.add_design_var('y2', lower=-50.0, upper=50.0)
        model.add_constraint('f_xy', lower=0.0)
        model.add_constraint('g', lower=0.0)
        model.add_constraint('parab2.f_xy', lower=0.0)
        model.add_constraint('ndp2.g', lower=0.0)
        model.add_objective('f_sum', index=-1)

        for mode in ['fwd', 'rev']:
            prob.setup(mode=mode, force_alloc_complex=True)

            prob.run_model()

            J = prob.check_totals(method='fd')
            assert_near_equal(J['parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-5)
            assert_near_equal(J['parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-5)
            assert_near_equal(J['ndp.g', 'p.x']['abs error'][0], 0.0, 2e-5)
            assert_near_equal(J['ndp.g', 'p.y']['abs error'][0], 0.0, 2e-5)
            assert_near_equal(J['parab2.f_xy', 'p.x2']['abs error'][0], 0.0, 1e-5)
            assert_near_equal(J['parab2.f_xy', 'p.y2']['abs error'][0], 0.0, 1e-5)
            assert_near_equal(J['ndp2.g', 'p.x2']['abs error'][0], 0.0, 2e-5)
            assert_near_equal(J['ndp2.g', 'p.y2']['abs error'][0], 0.0, 2e-5)
            assert_near_equal(J['sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-5)
            assert_near_equal(J['sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-5)

            J = prob.check_totals(method='cs')
            assert_near_equal(J['parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-14)
            assert_near_equal(J['parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-14)
            assert_near_equal(J['ndp.g', 'p.x']['abs error'][0], 0.0, 1e-13)
            assert_near_equal(J['ndp.g', 'p.y']['abs error'][0], 0.0, 1e-13)
            assert_near_equal(J['parab2.f_xy', 'p.x2']['abs error'][0], 0.0, 1e-14)
            assert_near_equal(J['parab2.f_xy', 'p.y2']['abs error'][0], 0.0, 1e-14)
            assert_near_equal(J['ndp2.g', 'p.x2']['abs error'][0], 0.0, 1e-13)
            assert_near_equal(J['ndp2.g', 'p.y2']['abs error'][0], 0.0, 1e-13)
            assert_near_equal(J['sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-14)
            assert_near_equal(J['sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-14)


class DistribStateImplicit(om.ImplicitComponent):
    """
    This component is unusual in that it has a distributed variable 'states' that
    is not connected to any other variables in the model.  The input 'a' sets the local
    values of 'states' and the output 'out_var' is the sum of all of the distributed values
    of 'states'.
    """

    def setup(self):
        self.options['distributed'] = True

        self.add_input('a', val=10., units='m', src_indices=[0])

        rank = self.comm.rank

        GLOBAL_SIZE = 5
        sizes, offsets = evenly_distrib_idxs(self.comm.size, GLOBAL_SIZE)

        self.add_output('states', shape=int(sizes[rank]))

        self.add_output('out_var', shape=1)

        self.local_size = sizes[rank]

        self.linear_solver = om.PETScKrylov()

    def solve_nonlinear(self, i, o):
        o['states'] = i['a']

        local_sum = np.zeros(1)
        local_sum[0] = np.sum(o['states'])
        tmp = np.zeros(1)
        self.comm.Allreduce(local_sum, tmp, op=MPI.SUM)

        o['out_var'] = tmp[0]

    def apply_nonlinear(self, i, o, r):
        r['states'] = o['states'] - i['a']

        local_sum = np.zeros(1)
        local_sum[0] = np.sum(o['states'])
        global_sum = np.zeros(1)
        self.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)

        r['out_var'] = o['out_var'] - global_sum[0]

    def apply_linear(self, i, o, d_i, d_o, d_r, mode):
        if mode == 'fwd':
            if 'states' in d_o:
                d_r['states'] += d_o['states']

                local_sum = np.array([np.sum(d_o['states'])])
                global_sum = np.zeros(1)
                self.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
                d_r['out_var'] -= global_sum

            if 'out_var' in d_o:
                    d_r['out_var'] += d_o['out_var']

            if 'a' in d_i:
                    d_r['states'] -= d_i['a']

        elif mode == 'rev':
            if 'states' in d_o:
                d_o['states'] += d_r['states']

                tmp = np.zeros(1)
                if self.comm.rank == 0:
                    tmp[0] = d_r['out_var'].copy()
                self.comm.Bcast(tmp, root=0)

                d_o['states'] -= tmp

            if 'out_var' in d_o:
                d_o['out_var'] += d_r['out_var']

            if 'a' in d_i:
                    d_i['a'] -= np.sum(d_r['states'])


class DistParab2(om.ExplicitComponent):

    def initialize(self):
        self.options['distributed'] = True

        self.options.declare('arr_size', types=int, default=10,
                             desc="Size of input and output vectors.")

    def setup(self):
        arr_size = self.options['arr_size']
        comm = self.comm
        rank = comm.rank

        sizes, offsets = evenly_distrib_idxs(comm.size, arr_size)
        start = offsets[rank]
        self.io_size = sizes[rank]
        self.offset = offsets[rank]
        end = start + self.io_size

        self.add_input('x', val=np.ones(self.io_size),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('y', val=np.ones(self.io_size),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('a', val=-3.0 * np.ones(self.io_size),
                       src_indices=np.arange(start, end, dtype=int))

        self.add_output('f_xy', val=np.ones(self.io_size))

        self.declare_partials('f_xy', ['x', 'y'])

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        a = inputs['a']

        outputs['f_xy'] = (x + a)**2 + x * y + (y + a + 4.0)**2 - 3.0

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        y = inputs['y']
        a = inputs['a']

        partials['f_xy', 'x'] = np.diag(2.0*x + 2.0 * a + y)
        partials['f_xy', 'y'] = np.diag(2.0*y + 2.0 * a + 8.0 + x)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class MPITests3(unittest.TestCase):

    N_PROCS = 3

    def test_distrib_apply(self):
        p = om.Problem()

        p.model.add_subsystem('des_vars', om.IndepVarComp('a', val=10., units='m'), promotes=['*'])
        p.model.add_subsystem('icomp', DistribStateImplicit(), promotes=['*'])

        expected = np.array([5.])

        p.setup(mode='fwd')
        p.run_model()
        jac = p.compute_totals(of=['out_var'], wrt=['a'], return_format='dict')
        assert_near_equal(jac['out_var']['a'][0], expected, 1e-6)

        p.setup(mode='rev')
        p.run_model()
        jac = p.compute_totals(of=['out_var'], wrt=['a'], return_format='dict')
        assert_near_equal(jac['out_var']['a'][0], expected, 1e-6)

    def test_distrib_con_indices(self):
        size = 7
        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', np.ones(size))
        ivc.add_output('y', np.ones(size))
        ivc.add_output('a', -3.0 + 0.6 * np.arange(size))

        model.add_subsystem('p', ivc, promotes=['*'])

        model.add_subsystem("parab", DistParab2(arr_size=size), promotes=['*'])

        model.add_subsystem('sum', om.ExecComp('f_sum = sum(f_xy)',
                                               f_sum=np.ones(1),
                                               f_xy=np.ones(size)),
                            promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_constraint('f_xy', lower=0.0, indices=[3])

        prob.setup(force_alloc_complex=True, mode='fwd')

        prob.run_model()

        con = prob.driver.get_constraint_values()
        assert_near_equal(con['parab.f_xy'],
                          np.array([12.48]),
                          1e-6)

        totals = prob.check_totals(method='cs')
        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-6)

        of = ['parab.f_xy']
        J = prob.driver._compute_totals(of=of, wrt=['p.x', 'p.y'], return_format='dict')
        assert_near_equal(J['parab.f_xy']['p.x'], np.array([[-0. , -0. , -0., 0.6 , -0. , -0. , -0. ]]),
                          1e-11)
        assert_near_equal(J['parab.f_xy']['p.y'], np.array([[-0. , -0. , -0., 8.6, -0. , -0. , -0. ]]),
                          1e-11)

        prob.setup(force_alloc_complex=True, mode='rev')

        prob.run_model()

        totals = prob.check_totals(method='cs')
        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-6)

        of = ['parab.f_xy']
        J = prob.driver._compute_totals(of=of, wrt=['p.x', 'p.y'], return_format='dict')
        assert_near_equal(J['parab.f_xy']['p.x'], np.array([[-0. , -0. , -0., 0.6 , -0. , -0. , -0. ]]),
                          1e-11)
        assert_near_equal(J['parab.f_xy']['p.y'], np.array([[-0. , -0. , -0., 8.6, -0. , -0. , -0. ]]),
                          1e-11)

    def test_distrib_obj_indices(self):
        size = 7
        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', np.ones(size))
        ivc.add_output('y', np.ones(size))
        ivc.add_output('a', -3.0 + 0.6 * np.arange(size))

        model.add_subsystem('p', ivc, promotes=['*'])

        model.add_subsystem("parab", DistParab2(arr_size=size), promotes=['*'])

        model.add_subsystem('sum', om.ExecComp('f_sum = sum(f_xy)',
                                               f_sum=np.ones(1),
                                               f_xy=np.ones(size)),
                            promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy', index=-1)

        prob.setup(force_alloc_complex=True, mode='fwd')

        prob.run_model()

        totals = prob.check_totals(method='cs')
        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-6)

        prob.setup(force_alloc_complex=True, mode='rev')
        prob.run_model()

        totals = prob.check_totals(method='cs')
        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-6)

    def test_distrib_con_indices_negative(self):
        size = 7
        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', np.ones(size))
        ivc.add_output('y', np.ones(size))
        ivc.add_output('a', -3.0 + 0.6 * np.arange(size))

        model.add_subsystem('p', ivc, promotes=['*'])

        model.add_subsystem("parab", DistParab2(arr_size=size), promotes=['*'])

        model.add_subsystem('sum', om.ExecComp('f_sum = sum(f_xy)',
                                               f_sum=np.ones(1),
                                               f_xy=np.ones(size)),
                            promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_constraint('f_xy', lower=0.0, indices=[-5, -1])
        model.add_objective('f_sum', index=-1)

        prob.setup(force_alloc_complex=True, mode='fwd')

        prob.run_model()

        con = prob.driver.get_constraint_values()
        assert_near_equal(con['parab.f_xy'],
                          np.array([ 8.88, 31.92]),
                          1e-6)

        totals = prob.check_totals(method='cs')
        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-6)

        of = ['parab.f_xy']
        J = prob.driver._compute_totals(of=of, wrt=['p.x', 'p.y'], return_format='dict')
        assert_near_equal(J['parab.f_xy']['p.x'], np.array([[-0. , -0. , -0.6, -0. , -0. , -0. , -0. ],
                                                            [-0. , -0. , -0. , -0. , -0. , -0. ,  4.2]]),
                          1e-11)
        assert_near_equal(J['parab.f_xy']['p.y'], np.array([[-0. , -0. ,  7.4, -0. , -0. , -0. , -0. ],
                                                            [-0. , -0. , -0. , -0. , -0. , -0. , 12.2]]),
                          1e-11)

        prob.setup(force_alloc_complex=True, mode='rev')

        prob.run_model()

        totals = prob.check_totals(method='cs')
        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-6)

        of = ['parab.f_xy']
        J = prob.driver._compute_totals(of=of, wrt=['p.x', 'p.y'], return_format='dict')
        assert_near_equal(J['parab.f_xy']['p.x'], np.array([[-0. , -0. , -0.6, -0. , -0. , -0. , -0. ],
                                                            [-0. , -0. , -0. , -0. , -0. , -0. ,  4.2]]),
                          1e-11)
        assert_near_equal(J['parab.f_xy']['p.y'], np.array([[-0. , -0. ,  7.4, -0. , -0. , -0. , -0. ],
                                                            [-0. , -0. , -0. , -0. , -0. , -0. , 12.2]]),
                          1e-11)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class MPITestsBug(unittest.TestCase):

    N_PROCS = 2

    def test_index_voi_bg(self):
        # Before this bug was fixed, one proc would raise an exception, and this test would
        # lock up.

        class Phase(om.Group):

            def initialize(self):
                self.state_options = {}
                self.options.declare('ode_class', default=None)

            def add_state(self, name, targets=None):
                if name not in self.state_options:
                    self.state_options[name] = {}
                    self.state_options[name]['name'] = name
                    self.state_options[name]['shape'] = (1, )
                    self.state_options[name]['targets'] = (targets, )

            def setup(self):
                indep = om.IndepVarComp()

                for name, options in self.state_options.items():
                    indep.add_output(name='states:{0}'.format(name),
                                     shape=(3, np.prod(options['shape'])))

                self.add_subsystem('indep_states', indep, promotes_outputs=['*'])

                for name, options in self.state_options.items():
                    self.add_design_var(name='states:{0}'.format(name))

                ode_class = self.options['ode_class']
                rhs_disc = ode_class(num_nodes=4)

                self.add_subsystem('rhs_disc', rhs_disc)

                for name, options in self.state_options.items():
                    self.connect('states:{0}'.format(name),
                                  ['rhs_disc.{0}'.format(tgt) for tgt in options['targets']])

        class vanderpol_ode_group(om.Group):

            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']

                self.add_subsystem(name='vanderpol_ode_delay',
                                   subsys=vanderpol_ode_delay(num_nodes=nn),
                                   promotes_inputs=['x1'])

                self.add_subsystem(name='vanderpol_ode_rate_collect',
                                   subsys=vanderpol_ode_rate_collect(num_nodes=nn),
                                   promotes_outputs=['x0dot'])

                self.connect('vanderpol_ode_delay.x0dot', 'vanderpol_ode_rate_collect.partx0dot')

        class vanderpol_ode_delay(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('num_nodes', types=int)
                self.options['distributed'] = True

            def setup(self):
                nn = self.options['num_nodes']
                comm = self.comm
                rank = comm.rank

                sizes, offsets = evenly_distrib_idxs(comm.size, nn)
                start = offsets[rank]
                end = start + sizes[rank]

                self.add_input('x1', val=np.ones(sizes[rank]),
                               src_indices=np.arange(start, end, dtype=int))

                self.add_output('x0dot', val=np.ones(sizes[rank]))

                r = c = np.arange(sizes[rank])
                self.declare_partials(of='x0dot', wrt='x1',  rows=r, cols=c)

            def compute(self, inputs, outputs):
                x1 = inputs['x1']
                outputs['x0dot'] = 5.0 * x1**2

            def compute_partials(self, inputs, jacobian):
                x1 = inputs['x1']
                jacobian['x0dot', 'x1'] = 10.0 * x1

        class vanderpol_ode_rate_collect(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']
                comm = self.comm
                self.rank = comm.rank

                self.add_input('partx0dot', val=np.ones(nn))

                self.add_output('x0dot', val=np.ones(nn))

                # partials
                cols = np.arange(nn)
                self.declare_partials(of='x0dot', wrt='partx0dot', rows=cols, cols=cols, val=1.0)

            def compute(self, inputs, outputs):
                outputs['x0dot'] = inputs['partx0dot']


        p = om.Problem()

        phase = Phase(ode_class=vanderpol_ode_group)
        phase.add_state('x1', targets='x1')
        p.model = phase

        phase.add_objective('rhs_disc.x0dot', index=-1)

        p.setup(mode='rev')
        p.final_setup()
        p.run_model()

        of ='rhs_disc.x0dot'
        wrt = 'states:x1'
        totals = p.check_totals(of=of, wrt=wrt, compact_print=False)

        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-6)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class MPIFeatureTests(unittest.TestCase):

    N_PROCS = 2

    def test_distribcomp_derivs_feature(self):
        import numpy as np
        import openmdao.api as om
        from openmdao.test_suite.components.distributed_components import DistribCompDerivs, SummerDerivs
        from openmdao.utils.assert_utils import assert_check_partials

        size = 15

        model = om.Group()

        # Distributed component "C2" requires an IndepVarComp to supply inputs.
        model.add_subsystem("indep", om.IndepVarComp('x', np.zeros(size)))
        model.add_subsystem("C2", DistribCompDerivs(size=size))
        model.add_subsystem("C3", SummerDerivs(size=size))

        model.connect('indep.x', 'C2.invec')
        model.connect('C2.outvec', 'C3.invec')

        prob = om.Problem(model)
        prob.setup()

        prob.set_val('indep.x', np.ones(size))
        prob.run_model()

        assert_near_equal(prob.get_val('C2.invec'),
                          np.ones(8) if model.comm.rank == 0 else np.ones(7))
        assert_near_equal(prob.get_val('C2.outvec'),
                          2*np.ones(8) if model.comm.rank == 0 else -3*np.ones(7))
        assert_near_equal(prob.get_val('C3.sum'), -5.)

        assert_check_partials(prob.check_partials())

        J = prob.compute_totals(of=['C2.outvec'], wrt=['indep.x'])
        assert_near_equal(J[('C2.outvec', 'indep.x')],
                          np.eye(15)*np.append(2*np.ones(8), -3*np.ones(7)))

    @unittest.skipUnless(pyoptsparse_opt, "pyOptsparse is required.")
    def test_distributed_constraint(self):
        import numpy as np
        import openmdao.api as om

        from openmdao.test_suite.components.paraboloid_distributed import DistParabFeature

        size = 7

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', np.ones((size, )))
        ivc.add_output('y', -1.42 * np.ones((size, )))
        ivc.add_output('offset', -3.0 + 0.6 * np.arange(size))

        model.add_subsystem('p', ivc, promotes=['*'])
        model.add_subsystem("parab", DistParabFeature(arr_size=size),
                            promotes=['*'])
        model.add_subsystem('sum', om.ExecComp('f_sum = sum(f_xy)',
                                               f_sum=np.ones(1),
                                               f_xy=np.ones(size)),
                            promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_constraint('f_xy', lower=0.0)
        model.add_objective('f_sum', index=-1)

        prob.driver = om.pyOptSparseDriver(optimizer='SLSQP')
        prob.setup(force_alloc_complex=True)

        prob.run_driver()

        desvar = prob.get_val('p.x', get_remote=True)
        obj = prob.get_val('f_sum', get_remote=True)

        assert_near_equal(desvar, np.array([2.65752672, 2.60433212, 2.51005989, 1.91021257,
                                            1.3100763,  0.70992863, 0.10978096]), 1e-6)
        assert_near_equal(obj, 11.5015, 1e-6)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ZeroLengthInputsOutputs(unittest.TestCase):

    N_PROCS = 4
    # this test case targets situations when zero-length inputs
    # or outputs are located on some processors
    # issue 1350

    def test_distribcomp_zerolengthinputsoutputs(self):
        from openmdao.test_suite.components.distributed_components import DistribCompDerivs, SummerDerivs
        from openmdao.utils.assert_utils import assert_check_partials

        size = 3  # set to one less than number of procs, leave zero inputs/outputs on proc 3

        model = om.Group()
        model.add_subsystem("indep", om.IndepVarComp('x', np.zeros(size)))
        model.add_subsystem("C2", DistribCompDerivs(size=size))
        model.add_subsystem("C3", SummerDerivs(size=size))

        model.connect('indep.x', 'C2.invec')
        model.connect('C2.outvec', 'C3.invec')

        prob = om.Problem(model)
        prob.setup()

        prob['indep.x'] = np.ones(size)
        prob.run_model()

        if model.comm.rank < 3:
            assert_near_equal(prob['C2.invec'],
                            np.ones(1) if model.comm.rank == 0 else np.ones(1))
            assert_near_equal(prob['C2.outvec'],
                            2*np.ones(1) if model.comm.rank == 0 else -3*np.ones(1))
        assert_near_equal(prob['C3.sum'], -4.)

        assert_check_partials(prob.check_partials())

        J = prob.compute_totals(of=['C2.outvec'], wrt=['indep.x'])
        assert_near_equal(J[('C2.outvec', 'indep.x')],
                          np.eye(3)*np.append(2*np.ones(1), -3*np.ones(2)))


class DistribCompDenseJac(om.ExplicitComponent):

    def initialize(self):
        self.options['distributed'] = True
        self.options.declare('size', default=7)

    def setup(self):
        N = self.options['size']
        rank = self.comm.rank
        self.add_input('x', shape=1, src_indices=rank)
        sizes, offsets = evenly_distrib_idxs(self.comm.size, N)
        self.add_output('y', shape=sizes[rank])
        # automatically infer dimensions without specifying rows, cols
        self.declare_partials('y', 'x')

    def compute(self, inputs, outputs):
        N = self.options['size']
        rank = self.comm.rank
        sizes, offsets = evenly_distrib_idxs(self.comm.size, N)
        outputs['y'] = -2.33 * inputs['x'] * np.ones((sizes[rank],))

    def compute_partials(self, inputs, J):
        N = self.options['size']
        rank = self.comm.rank
        sizes, offsets = evenly_distrib_idxs(self.comm.size, N)
        # Define jacobian element by element with variable size array
        J['y','x'] = -2.33 * np.ones((sizes[rank],))


class DeclarePartialsWithoutRowCol(unittest.TestCase):
    N_PROCS = 3

    def test_distrib_dense_jacobian(self):
        # this case checks for specifying dense jacobians without
        # specifying row/col indices in the declaration
        # issue 1336
        size = 7
        model = om.Group()
        dvs = om.IndepVarComp()
        dvs.add_output('x', val=6.0)
        model.add_subsystem('dvs', dvs, promotes_outputs=['*'])
        model.add_subsystem('distcomp',DistribCompDenseJac(size=size), promotes_inputs=['*'])
        model.add_subsystem('execcomp',om.ExecComp('z = 2.2 * y', y=np.zeros((size,)), z=np.zeros((size,))))
        model.connect('distcomp.y', 'execcomp.y')
        model.add_design_var('x', lower=0.0, upper=10.0, scaler=1.0)
        model.add_constraint('execcomp.z', lower=4.2, scaler=1.0)
        model.add_objective('x')

        prob = om.Problem(model)
        prob.setup(mode='fwd')

        prob['dvs.x'] = 7.5
        prob.run_model()
        assert_near_equal(prob['execcomp.z'], np.ones((size,))*-38.4450, 1e-9)

        data = prob.check_totals(out_stream=None)
        assert_near_equal(data[('execcomp.z', 'dvs.x')]['abs error'][0], 0.0, 1e-6)


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
