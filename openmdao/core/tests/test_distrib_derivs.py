""" Test out some crucial linear GS tests in parallel with distributed comps."""

from openmdao.jacobians.jacobian import Jacobian
import unittest
import itertools

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.distributed_components import DistribCompDerivs, SummerDerivs
from openmdao.test_suite.components.paraboloid_distributed import DistParab, DistParabFeature, \
    DistParabDeprecated
from openmdao.utils.mpi import MPI
from openmdao.utils.om_warnings import OMDeprecationWarning
from openmdao.utils.name_maps import rel_name2abs_name
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials, \
    assert_check_totals, assert_warning

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
        super().__init__(exprs, **kwargs)
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
            v, _ = self._parse_for_names(expr)
            allvars.update(v)

        sizes, offsets = evenly_distrib_idxs(comm.size, self.arr_size)
        start = offsets[rank]
        end = start + sizes[rank]

        for name in outs:
            if name not in kwargs or not isinstance(kwargs[name], dict):
                kwargs[name] = {}
            kwargs[name]['val'] = np.ones(sizes[rank], float)

        for name in allvars:
            if name not in outs:
                if name not in kwargs or not isinstance(kwargs[name], dict):
                    kwargs[name] = {}
                meta = kwargs[name]
                meta['val'] = np.ones(sizes[rank], float)
                meta['src_indices'] = np.arange(start, end, dtype=int)

        super().setup()


class DistribCoordComp(om.ExplicitComponent):

    def setup(self):
        comm = self.comm
        rank = comm.rank

        if rank == 0:
            self.add_input('invec', np.zeros((5, 3)), distributed=True,
                           src_indices=[[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4],[0,1,2,0,1,2,0,1,2,0,1,2,0,1,2]])
            self.add_output('outvec', np.zeros((5, 3)), distributed=True)
        else:
            self.add_input('invec', np.zeros((4, 3)), distributed=True,
                           # use some negative indices here to
                           # make sure they work
                           src_indices=[[5,5,5,6,6,6,7,7,7,-1,8,-1],[0,1,2,0,1,2,0,1,2,0,1,2]])
            self.add_output('outvec', np.zeros((4, 3)), distributed=True)

    def compute(self, inputs, outputs):
        if self.comm.rank == 0:
            outputs['outvec'] = inputs['invec'] * 2.0
        else:
            outputs['outvec'] = inputs['invec'] * 3.0


class MixedDistrib2(om.ExplicitComponent):  # for double diamond case

    def setup(self):
        self.add_input('in_dist', shape_by_conn=True, distributed=True)
        self.add_input('in_nd', shape_by_conn=True)
        self.add_output('out_dist', copy_shape='in_dist', distributed=True)
        self.add_output('out_nd', copy_shape='in_nd')

    def compute(self, inputs, outputs):
        Id = inputs['in_dist']
        Is = inputs['in_nd']

        f_Id = Id**2 - 2.0*Id + 4.0
        f_Is = Is ** 0.5
        g_Is = Is**2 + 3.0*Is - 5.0
        g_Id = Id ** 0.5

        # Distributed output
        outputs['out_dist'] = f_Id + np.sum(f_Is)

        # We need to gather the summed values to compute the total sum over all procs.
        local_sum = np.array(np.sum(g_Id))
        total_sum = local_sum.copy()
        self.comm.Allreduce(local_sum, total_sum, op=MPI.SUM)
        outputs['out_nd'] = g_Is + total_sum

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        Id = inputs['in_dist']
        Is = inputs['in_nd']

        df_dId = 2.0 * Id - 2.0
        df_dIs = 0.5 / Is ** 0.5
        dg_dId = 0.5 / Id ** 0.5
        dg_dIs = 2.0 * Is + 3.0

        nId = len(Id)
        nIs = len(Is)

        if mode == 'fwd':
            if 'out_dist' in d_outputs:
                if 'in_dist' in d_inputs:
                    d_outputs['out_dist'] += df_dId * d_inputs['in_dist']
                if 'in_nd' in d_inputs:
                    d_outputs['out_dist'] += np.tile(df_dIs, nId).reshape((nId, nIs)).dot(d_inputs['in_nd'])
            if 'out_nd' in d_outputs:
                if 'in_dist' in d_inputs:
                    d_outputs['out_nd'] += self.comm.allreduce(np.tile(dg_dId, nIs).reshape((nIs, nId)).dot(d_inputs['in_dist']))
                if 'in_nd' in d_inputs:
                    d_outputs['out_nd'] += dg_dIs * d_inputs['in_nd']

        else:
            if 'out_dist' in d_outputs:
                if 'in_dist' in d_inputs:
                    d_inputs['in_dist'] += df_dId * d_outputs['out_dist']
                if 'in_nd' in d_inputs:
                    d_inputs['in_nd'] += self.comm.allreduce(np.tile(df_dIs, nId).reshape((nId, nIs)).T.dot(d_outputs['out_dist']))

            if 'out_nd' in d_outputs:
                if 'in_dist' in d_inputs:
                    d_inputs['in_dist'] += np.tile(dg_dId, nIs).reshape((nIs, nId)).T.dot(d_outputs['out_nd'])
                if 'in_nd' in d_inputs:
                    d_inputs['in_nd'] += dg_dIs * d_outputs['out_nd']


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
        prob.model.connect('comp.outvec', 'total.x', src_indices=om.slicer[:], flat_src_indices=True)

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
        prob.model.connect('C1.y', 'C2.y', src_indices=om.slicer[:])


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

        root.connect("C1.y", "sub.C2.x", src_indices=om.slicer[:])
        root.connect("C1.y", "sub.C3.x", src_indices=om.slicer[:])
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
        root.connect("C3.y", "C4.x", src_indices=om.slicer[:])

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
                            promotes_outputs=['*'])

        model.promotes('sum', inputs=['f_xy'], src_indices=om.slicer[:])

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

        J = prob.check_totals(method='fd', out_stream=None)
        assert_near_equal(J['parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-5)

        J = prob.check_totals(method='cs', out_stream=None)
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

        J = prob.check_totals(method='fd', show_only_incorrect=True)
        assert_near_equal(J['parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-5)

        J = prob.check_totals(method='cs', show_only_incorrect=True)
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
                            promotes_outputs=['*'])

        model.promotes('sum', inputs=['f_xy'], src_indices=om.slicer[:])

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

        J = prob.check_totals(method='fd', show_only_incorrect=True)
        assert_near_equal(J['parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-5)

        J = prob.check_totals(method='cs', show_only_incorrect=True)
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

        J = prob.check_totals(method='fd', show_only_incorrect=True)
        assert_near_equal(J['parab.f_xy', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['parab.f_xy', 'p.y']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.x']['abs error'][0], 0.0, 1e-5)
        assert_near_equal(J['sum.f_sum', 'p.y']['abs error'][0], 0.0, 1e-5)

        J = prob.check_totals(method='cs', show_only_incorrect=True)
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
                            promotes_outputs=['*'])

        model.promotes('sum', inputs=['f_xy'], src_indices=om.slicer[:])

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

        J = prob.check_totals(out_stream=None, method='cs')
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

        J = prob.check_totals(method='cs', show_only_incorrect=True)
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
                          promotes_outputs=['*'])

        sub.promotes('sum', inputs=['f_xy'], src_indices=om.slicer[:])

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

        J = prob.check_totals(method='fd', show_only_incorrect=True)
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

        J = prob.check_totals(method='fd', show_only_incorrect=True)
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
                          promotes_outputs=['*'])

        sub.promotes('sum', inputs=['f_xy'], src_indices=om.slicer[:])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_constraint('f_xy', lower=0.0)
        model.add_objective('f_sum', index=-1)

        sub.approx_totals(method='fd')

        prob.setup(force_alloc_complex=True)

        with self.assertRaises(RuntimeError) as context:
            prob.run_model()

        msg = "'sub' <class Group>: Approx_totals is not supported on a group with a distributed "
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
        model.add_subsystem("ndp", NonDistComp(arr_size=size), promotes_outputs=['*'])
        model.promotes('ndp', inputs=['f_xy'], src_indices=om.slicer[:])
        model.add_subsystem("parab2", DistParab(arr_size=size2))
        model.add_subsystem("ndp2", NonDistComp(arr_size=size2))

        model.add_subsystem('sum', om.ExecComp('f_sum = sum(f_xy)',
                                               f_sum=np.ones((size, )),
                                               f_xy=np.ones((size, ))),
                            promotes_outputs=['*'])

        model.promotes('sum', inputs=['f_xy'], src_indices=om.slicer[:])

        model.connect('x2', 'parab2.x')
        model.connect('y2', 'parab2.y')
        model.connect('a2', 'parab2.a')
        model.connect('parab2.f_xy', 'ndp2.f_xy', src_indices=om.slicer[:])

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

            J = prob.check_totals(method='fd', show_only_incorrect=True)
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

            J = prob.check_totals(method='cs', show_only_incorrect=True)
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

    def run_mixed_distrib2_prob(self, mode):
        size = 5
        comm = MPI.COMM_WORLD
        rank = comm.rank
        sizes, offsets = evenly_distrib_idxs(comm.size, size)

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x_dist', np.zeros(sizes[rank]), distributed=True)
        ivc.add_output('x_nd', np.zeros(size))

        model.add_subsystem("indep", ivc)
        model.add_subsystem("D1", MixedDistrib2())

        model.connect('indep.x_dist', 'D1.in_dist')
        model.connect('indep.x_nd', 'D1.in_nd')

        model.add_design_var('indep.x_nd')
        model.add_design_var('indep.x_dist')
        model.add_constraint('D1.out_dist', lower=0.0)
        model.add_constraint('D1.out_nd', lower=0.0)

        prob.setup(force_alloc_complex=True, mode=mode)

        # Set initial values of distributed variable.
        x_dist_init = 3.0 + np.arange(size)[offsets[rank]:offsets[rank] + sizes[rank]]
        x_dist_init /= np.max(x_dist_init)
        prob.set_val('indep.x_dist', x_dist_init)

        # Set initial values of non-distributed variable.
        x_nd_init = 1.0 + 2.0*np.arange(size)
        x_nd_init /= np.max(x_nd_init)
        prob.set_val('indep.x_nd', x_nd_init)

        prob.run_model()

        return prob

    def test_distrib_mixeddistrib2_totals_rev(self):
        prob = self.run_mixed_distrib2_prob('rev')

        totals = prob.check_totals(show_only_incorrect=True, method='cs')
        assert_check_totals(totals)

    def test_distrib_mixeddistrib2_partials_rev(self):
        prob = self.run_mixed_distrib2_prob('rev')

        partials = prob.check_partials(show_only_incorrect=True, method='cs')
        assert_check_partials(partials)

    def test_distrib_mixeddistrib2_totals_fwd(self):
        prob = self.run_mixed_distrib2_prob('fwd')

        totals = prob.check_totals(show_only_incorrect=True, method='cs')
        assert_check_totals(totals)

    def test_distrib_mixeddistrib2_partials_fwd(self):
        prob = self.run_mixed_distrib2_prob('fwd')

        partials = prob.check_partials(show_only_incorrect=True, method='cs')
        assert_check_partials(partials)

    def test_distrib_cascade_rev(self):
        # Tests the derivatives on a complicated model that is the distributed equivalent
        # of a double diamond.
        size = 5
        comm = MPI.COMM_WORLD
        rank = comm.rank
        sizes, offsets = evenly_distrib_idxs(comm.size, size)

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x_dist', np.zeros(sizes[rank]), distributed=True)
        ivc.add_output('x_nd', np.zeros(size))

        model.add_subsystem("indep", ivc)
        model.add_subsystem("D1", MixedDistrib2())
        model.add_subsystem("D2", MixedDistrib2())
        model.add_subsystem("D3", MixedDistrib2())
        model.add_subsystem("D4", MixedDistrib2())

        model.connect('indep.x_dist', 'D1.in_dist')
        model.connect('indep.x_nd', 'D1.in_nd')
        model.connect('D1.out_dist', 'D2.in_dist')
        model.connect('D1.out_nd', 'D2.in_nd')
        model.connect('D2.out_dist', 'D3.in_dist')
        model.connect('D2.out_nd', 'D3.in_nd')
        model.connect('D3.out_dist', 'D4.in_dist')
        model.connect('D3.out_nd', 'D4.in_nd')

        model.add_design_var('indep.x_nd')
        model.add_design_var('indep.x_dist')
        model.add_constraint('D4.out_dist', lower=0.0)
        model.add_constraint('D4.out_nd', lower=0.0)

        msg = "'D4' <class MixedDistrib2>: It appears this component mixes " \
              "distributed/non-distributed inputs and outputs, so it may break starting with " \
              "OpenMDAO 3.25, where the convention used when passing data between " \
              "distributed and non-distributed inputs and outputs within a matrix free component " \
              "will change. See https://github.com/OpenMDAO/POEMs/blob/master/POEM_075.md for " \
              "details."
        with assert_warning(OMDeprecationWarning, msg):
            prob.setup(force_alloc_complex=True, mode='rev')

        # Set initial values of distributed variable.
        x_dist_init = 3.0 + np.arange(size)[offsets[rank]:offsets[rank] + sizes[rank]]
        x_dist_init /= np.max(x_dist_init)
        prob.set_val('indep.x_dist', x_dist_init)

        # Set initial values of non-distributed variable.
        x_nd_init = 1.0 + 2.0*np.arange(size)
        x_nd_init /= np.max(x_nd_init)
        prob.set_val('indep.x_nd', x_nd_init)

        prob.run_model()

        totals = prob.check_totals(show_only_incorrect=True, method='cs')
        assert_check_totals(totals, rtol=1e-12)


class DistribStateImplicit(om.ImplicitComponent):
    """
    This component is unusual in that it has a distributed variable 'states' that
    is not connected to any other variables in the model.  The input 'a' sets the local
    values of 'states' and the output 'out_var' is the sum of all of the distributed values
    of 'states'.
    """

    def setup(self):
        self.add_input('a', val=10., units='m', src_indices=[0], flat_src_indices=True, distributed=True)

        rank = self.comm.rank

        sizes, _ = evenly_distrib_idxs(self.comm.size, 5)

        self.add_output('states', shape=int(sizes[rank]), distributed=True)
        self.add_output('out_var', shape=1, distributed=True)

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

        self.add_input('x', val=np.ones(self.io_size), distributed=True,
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('y', val=np.ones(self.io_size), distributed=True,
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('a', val=-3.0 * np.ones(self.io_size), distributed=True,
                       src_indices=np.arange(start, end, dtype=int))

        self.add_output('f_xy', val=np.ones(self.io_size), distributed=True)

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
                            promotes_outputs=['*'])

        model.promotes('sum', inputs=['f_xy'], src_indices=om.slicer[:])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_constraint('f_xy', lower=0.0, indices=[3], flat_indices=True)

        prob.setup(force_alloc_complex=True, mode='fwd')

        prob.run_model()

        con = prob.driver.get_constraint_values()
        assert_near_equal(con['parab.f_xy'],
                          np.array([12.48]),
                          1e-6)

        totals = prob.check_totals(method='cs', show_only_incorrect=True)
        assert_check_totals(totals, rtol=1e-6)

        of = ['parab.f_xy']
        J = prob.driver._compute_totals(of=of, wrt=['p.x', 'p.y'], return_format='dict')
        assert_near_equal(J['parab.f_xy']['p.x'], np.array([[-0. , -0. , -0., 0.6 , -0. , -0. , -0. ]]),
                          1e-11)
        assert_near_equal(J['parab.f_xy']['p.y'], np.array([[-0. , -0. , -0., 8.6, -0. , -0. , -0. ]]),
                          1e-11)

        prob.setup(force_alloc_complex=True, mode='rev')

        prob.run_model()

        totals = prob.check_totals(method='cs', show_only_incorrect=True)
        assert_check_totals(totals, rtol=1e-6)

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
                            promotes_outputs=['*'])

        model.promotes('sum', inputs=['f_xy'], src_indices=om.slicer[:])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy', index=-1)

        prob.setup(force_alloc_complex=True, mode='fwd')

        prob.run_model()

        totals = prob.check_totals(method='cs', show_only_incorrect=True)
        assert_check_totals(totals, rtol=1e-6)

        prob.setup(force_alloc_complex=True, mode='rev')
        prob.run_model()

        totals = prob.check_totals(method='cs', show_only_incorrect=True)
        assert_check_totals(totals, rtol=1e-6)

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
                            promotes_outputs=['*'])

        model.promotes('sum', inputs=['f_xy'], src_indices=om.slicer[:])

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

        totals = prob.check_totals(method='cs', show_only_incorrect=True)
        assert_check_totals(totals, rtol=1e-6)

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

        totals = prob.check_totals(method='cs', show_only_incorrect=True)
        assert_check_totals(totals, rtol=1e-6)

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

                self.connect('vanderpol_ode_delay.x0dot', 'vanderpol_ode_rate_collect.partx0dot',
                             src_indices=om.slicer[:])

        class vanderpol_ode_delay(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                nn = self.options['num_nodes']
                comm = self.comm
                rank = comm.rank

                sizes, offsets = evenly_distrib_idxs(comm.size, nn)
                start = offsets[rank]
                end = start + sizes[rank]

                self.add_input('x1', val=np.ones(sizes[rank]), distributed=True,
                               src_indices=np.arange(start, end, dtype=int),
                               flat_src_indices=True)

                self.add_output('x0dot', val=np.ones(sizes[rank]), distributed=True)

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
        totals = p.check_totals(of=of, wrt=wrt, compact_print=False, show_only_incorrect=True)
        assert_check_totals(totals, atol=1e-5, rtol=1e-6)

    @unittest.skipUnless(pyoptsparse_opt, "pyOptsparse is required.")
    def test_zero_entry_distrib(self):
        # this test errored out before the fix
        dist_shape = 1 if MPI.COMM_WORLD.rank > 0 else 2

        current_om_convention = True

        class SerialComp(om.ExplicitComponent):
            def setup(self):
                self.add_input("dv")
                self.add_output("aoa_serial")

            def compute(self, inputs, outputs):
                outputs["aoa_serial"] = 2.0 * inputs["dv"]

            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                if mode == "fwd":
                    if "aoa_serial" in d_outputs:
                        if "dv" in d_inputs:
                            d_outputs["aoa_serial"] += 2.0 * d_inputs["dv"]
                if mode == "rev":
                    if "aoa_serial" in d_outputs:
                        if "dv" in d_inputs:
                            d_inputs["dv"] += 2.0 * d_outputs["aoa_serial"]


        class MixedSerialInComp(om.ExplicitComponent):
            def setup(self):
                self.add_input("aoa_serial")
                self.add_output("flow_state_dist", shape=dist_shape, distributed=True)

            def compute(self, inputs, outputs):
                outputs["flow_state_dist"][:] = 0.5 * inputs["aoa_serial"]

            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                if mode == "fwd":
                    if "flow_state_dist" in d_outputs:
                        if "aoa_serial" in d_inputs:
                            d_outputs["flow_state_dist"] += 0.5 * d_inputs["aoa_serial"]
                if mode == "rev":
                    if "flow_state_dist" in d_outputs:
                        if "aoa_serial" in d_inputs:
                            if current_om_convention:
                                d_inputs["aoa_serial"] += 0.5 * np.sum(d_outputs["flow_state_dist"])
                            else:
                                d_inputs["aoa_serial"] += 0.5 * self.comm.allreduce(np.sum(d_outputs["flow_state_dist"]))


        class MixedSerialOutComp(om.ExplicitComponent):
            def setup(self):
                self.add_input("aoa_serial")
                self.add_input("force_dist", shape=dist_shape, distributed=True)
                self.add_output("lift_serial")
                self.add_output("cons_dist", shape=dist_shape, distributed=True)

            def compute(self, inputs, outputs):
                outputs["lift_serial"] = 2.0 * inputs["aoa_serial"] + self.comm.allreduce(3.0 * np.sum(inputs["force_dist"]))
                if self.comm.rank == 0:
                    outputs["cons_dist"] = inputs["force_dist"]
                else:
                    outputs["cons_dist"] = 0.0

            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                if mode == "fwd":
                    if "lift_serial" in d_outputs:
                        if "aoa_serial" in d_inputs:
                            d_outputs["lift_serial"] += 2.0 * d_inputs["aoa_serial"]
                        if "force_dist" in d_inputs:
                            d_outputs["lift_serial"] += 3.0 * self.comm.allreduce(np.sum(d_inputs["force_dist"]))
                else:  # rev
                    if "lift_serial" in d_outputs:
                        if "aoa_serial" in d_inputs:
                            d_inputs["aoa_serial"] += 2.0 * d_outputs["lift_serial"]
                        if "force_dist" in d_inputs:
                            if current_om_convention:
                                d_inputs["force_dist"] += 3.0 * self.comm.allreduce(d_outputs["lift_serial"])
                            else:
                                d_inputs["force_dist"] += 3.0 * d_outputs["lift_serial"]

                    if "cons_dist" in d_outputs:
                        if "aoa_serial" in d_inputs:
                            d_inputs["aoa_serial"] += 0.0
                        if "force_dist" in d_inputs:
                            if self.comm.rank == 0:
                                # when any entry in a derivative is 0, we hit errors
                                d_inputs["force_dist"] = d_outputs["cons_dist"]
                                # set one entry to 0 - comment out and error goes away
                                d_inputs["force_dist"][-1] = 0
                            else:
                                d_inputs["force_dist"] = 0.0


        class DistComp(om.ExplicitComponent):
            def setup(self):
                self.add_input("flow_state_dist", shape=dist_shape, distributed=True)
                self.add_output("force_dist", shape=dist_shape, distributed=True)

            def compute(self, inputs, outputs):
                outputs["force_dist"] = 3.0 * inputs["flow_state_dist"]

            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                if mode == "fwd":
                    if "force_dist" in d_outputs and "flow_state_dist" in d_inputs:
                        d_outputs["force_dist"] += 3.0 * d_inputs["flow_state_dist"]
                if mode == "rev":
                    if "force_dist" in d_outputs and "flow_state_dist" in d_inputs:
                        d_inputs["flow_state_dist"] += 3.0 * d_outputs["force_dist"]

        prob = om.Problem()
        model = prob.model
        ivc = model.add_subsystem("ivc", om.IndepVarComp())
        ivc.add_output("dv", val=1.0)

        model.add_subsystem("serial_comp", SerialComp())
        model.add_subsystem("mixed_in_comp", MixedSerialInComp())
        model.add_subsystem("dist_comp", DistComp())
        model.add_subsystem("mixed_out_comp", MixedSerialOutComp())
        model.add_design_var("ivc.dv")
        model.connect("ivc.dv", "serial_comp.dv")
        model.connect("serial_comp.aoa_serial", "mixed_in_comp.aoa_serial")
        model.connect("mixed_in_comp.flow_state_dist", "dist_comp.flow_state_dist")
        model.connect("dist_comp.force_dist", "mixed_out_comp.force_dist")
        model.connect("serial_comp.aoa_serial", "mixed_out_comp.aoa_serial")
        model.add_objective("mixed_out_comp.lift_serial")
        model.add_constraint("mixed_out_comp.cons_dist")

        prob.setup(mode="rev")
        prob.driver = om.pyOptSparseDriver()
        prob.driver.options["optimizer"] = "SLSQP"
        prob.run_driver()


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class MPIFeatureTests(unittest.TestCase):

    N_PROCS = 2

    def test_distribcomp_derivs_feature(self):

        size = 15

        model = om.Group()

        # Distributed component "C2" requires an IndepVarComp to supply inputs.
        model.add_subsystem("indep", om.IndepVarComp('x', np.zeros(size)))
        model.add_subsystem("C2", DistribCompDerivs(size=size))
        model.add_subsystem("C3", SummerDerivs(size=size))

        model.connect('indep.x', 'C2.invec')
        model.connect('C2.outvec', 'C3.invec', src_indices=om.slicer[:])

        prob = om.Problem(model)

        prob.setup()

        prob.set_val('indep.x', np.ones(size))
        prob.run_model()

        assert_near_equal(prob.get_val('C2.invec'),
                          np.ones(8) if model.comm.rank == 0 else np.ones(7))
        assert_near_equal(prob.get_val('C2.outvec'),
                          2*np.ones(8) if model.comm.rank == 0 else -3*np.ones(7))
        assert_near_equal(prob.get_val('C3.sum'), -5.)

        assert_check_partials(prob.check_partials(show_only_incorrect=True))

        J = prob.compute_totals(of=['C2.outvec'], wrt=['indep.x'])
        assert_near_equal(J[('C2.outvec', 'indep.x')],
                          np.eye(15)*np.append(2*np.ones(8), -3*np.ones(7)))

    @unittest.skipUnless(pyoptsparse_opt, "pyOptsparse is required.")
    def test_distributed_constraint(self):

        size = 7

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', np.ones(size))
        ivc.add_output('y', -1.42 * np.ones(size))

        model.add_subsystem('p', ivc, promotes=['*'])
        model.add_subsystem("parab", DistParabFeature(arr_size=size), promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_constraint('f_xy', lower=0.0)
        model.add_objective('f_sum', index=-1)

        prob.driver = om.pyOptSparseDriver(optimizer='SLSQP')
        prob.setup()

        prob.run_driver()

        desvar = prob.get_val('p.x', get_remote=True)
        obj = prob.get_val('f_sum', get_remote=True)

        assert_near_equal(desvar, np.array([2.65752672, 2.60433212, 2.51005989, 1.91021257,
                                            1.3100763,  0.70992863, 0.10978096]), 1e-6)
        assert_near_equal(obj, 11.5015, 1e-6)

    @unittest.skipUnless(pyoptsparse_opt, "pyOptsparse is required.")
    def test_distributed_constraint_deprecated(self):
        """ Test distributed constraint with deprecated usage of src_indices. """

        size = 7

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', np.ones((size, )))
        ivc.add_output('y', -1.42 * np.ones((size, )))
        ivc.add_output('offset', -3.0 + 0.6 * np.arange(size))

        model.add_subsystem('p', ivc, promotes=['*'])
        model.add_subsystem("parab", DistParabDeprecated(arr_size=size),
                            promotes=['*'])
        model.add_subsystem('sum', om.ExecComp('f_sum = sum(f_xy)',
                                               f_sum=np.ones(1),
                                               f_xy=np.ones(size)),
                            promotes_outputs=['*'])
        model.promotes('sum', inputs=['f_xy'], src_indices=om.slicer[:])

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

        size = 3  # set to one less than number of procs, leave zero inputs/outputs on proc 3

        model = om.Group()
        model.add_subsystem("indep", om.IndepVarComp('x', np.zeros(size)))
        model.add_subsystem("C2", DistribCompDerivs(size=size))
        model.add_subsystem("C3", SummerDerivs(size=size))

        model.connect('indep.x', 'C2.invec')
        model.connect('C2.outvec', 'C3.invec', src_indices=om.slicer[:])

        prob = om.Problem(model)

        prob.setup()

        prob['indep.x'] = np.ones(size)
        prob.run_model()

        if model.comm.rank < 3:
            assert_near_equal(prob.get_val('C2.invec', get_remote=False),
                            np.ones(1) if model.comm.rank == 0 else np.ones(1))
            assert_near_equal(prob.get_val('C2.outvec', get_remote=False),
                            2*np.ones(1) if model.comm.rank == 0 else -3*np.ones(1))
        assert_near_equal(prob['C3.sum'], -4.)

        assert_check_partials(prob.check_partials())

        J = prob.compute_totals(of=['C2.outvec'], wrt=['indep.x'])
        assert_near_equal(J[('C2.outvec', 'indep.x')],
                          np.eye(3)*np.append(2*np.ones(1), -3*np.ones(2)))

        # Make sure that the code for handling element stepsize also works on a distributed model.
        assert(prob.check_partials(step_calc='rel_element', show_only_incorrect=True))


class DistribCompDenseJac(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('size', default=7)

    def setup(self):
        N = self.options['size']
        rank = self.comm.rank
        self.add_input('x', shape=1, src_indices=rank, distributed=True)
        sizes, offsets = evenly_distrib_idxs(self.comm.size, N)
        self.add_output('y', shape=sizes[rank], distributed=True)
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
        model.connect('distcomp.y', 'execcomp.y', src_indices=om.slicer[:])
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


class TestBugs(unittest.TestCase):

    def test_distributed_ivc_as_desvar(self):
        # Covers a case where a distributed IVC output is used as a desvar with indices.

        class DVS(om.IndepVarComp):
            def setup(self):
                self.add_output('state', np.ones(4), distributed=True)

        class SolverComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('state',shape_by_conn=True, distributed=True)
                self.add_output('func', distributed=True)
                self.declare_partials('func','state',method='fd')

            def compute(self, inputs, outputs):
                outputs['func'] += np.sum(inputs['state'])

        prob = om.Problem()
        dvs = prob.model.add_subsystem('dvs',DVS())
        prob.model.add_subsystem('solver', SolverComp())
        prob.model.connect('dvs.state','solver.state')
        prob.model.add_design_var('dvs.state', indices=[0,2])
        prob.model.add_objective('solver.func')

        prob.setup()
        prob.run_model()
        totals = prob.check_totals(wrt='dvs.state', show_only_incorrect=True)
        assert_near_equal(totals['solver.func', 'dvs.state']['abs error'][0], 0.0, tolerance=1e-7)


def f_out_dist(Id, Is):
    return Id**2 - 2.0*Id + 4.0 + np.sum(1.5 * Is ** 2)

def f_out_nd(Id, Is):
    return Is**2 + 3.0*Is - 5.0 + np.sum(1.5 * Id ** 2)


class Distrib_Derivs(om.ExplicitComponent):

    def setup(self):

        self.add_input('in_dist', shape_by_conn=True, distributed=True)
        self.add_input('in_nd', shape_by_conn=True)

        self.add_output('out_dist', copy_shape='in_dist', distributed=True)
        self.add_output('out_nd', copy_shape='in_nd')

    def compute(self, inputs, outputs):
        Id = inputs['in_dist']
        Is = inputs['in_nd']

        # Our local distributed output is a function of the local distributed input and
        # the non-distributed input.
        outputs['out_dist'] = f_out_dist(Id, Is)

        if self.comm.size > 1:

            # We need to gather the summed values to compute the total sum over all procs.
            local_sum = np.array([np.sum(1.5 * Id ** 2)])
            total_sum = local_sum.copy()
            self.comm.Allreduce(local_sum, total_sum, op=MPI.SUM)

            # so the non-distributed output is a function of the non-distributed input and the full distributed
            # input.
            outputs['out_nd'] = Is**2 + 3.0*Is - 5.0 + total_sum[0]
        else:
            outputs['out_nd'] = f_out_nd(Id, Is)


class Distrib_Derivs_Matfree(Distrib_Derivs):
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        Id = inputs['in_dist']
        Is = inputs['in_nd']

        size = len(Is)
        local_size = len(Id)

        df_dIs = 3. * Is
        dg_dId = 3. * Id

        if mode == 'fwd':
            if 'out_dist' in d_outputs:
                if 'in_dist' in d_inputs:
                    d_outputs['out_dist'] += (2.0 * Id - 2.0) * d_inputs['in_dist']
                if 'in_nd' in d_inputs:
                    d_outputs['out_dist'] += np.tile(df_dIs, local_size).reshape((local_size, size)).dot(d_inputs['in_nd'])
            if 'out_nd' in d_outputs:
                if 'in_dist' in d_inputs:
                    deriv = np.tile(dg_dId, size).reshape((size, local_size)).dot(d_inputs['in_dist'])
                    deriv_sum = np.zeros(deriv.size)
                    self.comm.Allreduce(deriv, deriv_sum, op=MPI.SUM)
                    d_outputs['out_nd'] += deriv_sum
                if 'in_nd' in d_inputs:
                    d_outputs['out_nd'] += (2.0 * Is + 3.0) * d_inputs['in_nd']
        else:  # rev
            if 'out_dist' in d_outputs:
                if 'in_dist' in d_inputs:
                    d_inputs['in_dist'] += (2.0 * Id - 2.0) * d_outputs['out_dist']
                if 'in_nd' in d_inputs:
                    deriv = np.tile(df_dIs, local_size).reshape((local_size, size)).T.dot(d_outputs['out_dist'])
                    deriv_sum = np.zeros(deriv.size)
                    self.comm.Allreduce(deriv, deriv_sum, op=MPI.SUM)
                    d_inputs['in_nd'] += deriv_sum

            if 'out_nd' in d_outputs:
                if 'in_dist' in d_inputs:
                    d_inputs['in_dist'] += np.tile(dg_dId, size).reshape((size, local_size)).T.dot(d_outputs['out_nd'])
                if 'in_nd' in d_inputs:
                    d_inputs['in_nd'] += (2.0 * Is + 3.0) * d_outputs['out_nd']


class Distrib_Derivs_Matfree_Old(Distrib_Derivs):
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        Id = inputs['in_dist']
        Is = inputs['in_nd']

        size = len(Is)
        local_size = len(Id)

        df_dIs = 3. * Is
        dg_dId = 3. * Id

        if mode == 'fwd':
            if 'out_dist' in d_outputs:
                if 'in_dist' in d_inputs:
                    d_outputs['out_dist'] += (2.0 * Id - 2.0) * d_inputs['in_dist']
                if 'in_nd' in d_inputs:
                    d_outputs['out_dist'] += np.tile(df_dIs, local_size).reshape((local_size, size)).dot(d_inputs['in_nd'])
            if 'out_nd' in d_outputs:
                if 'in_dist' in d_inputs:
                    deriv = np.tile(dg_dId, size).reshape((size, local_size)).dot(d_inputs['in_dist'])
                    deriv_sum = np.zeros(deriv.size)
                    self.comm.Allreduce(deriv, deriv_sum, op=MPI.SUM)
                    d_outputs['out_nd'] += deriv_sum
                if 'in_nd' in d_inputs:
                    d_outputs['out_nd'] += (2.0 * Is + 3.0) * d_inputs['in_nd']
        else:  # rev
            if 'out_dist' in d_outputs:
                if 'in_dist' in d_inputs:
                    d_inputs['in_dist'] += (2.0 * Id - 2.0) * d_outputs['out_dist']
                if 'in_nd' in d_inputs:
                    d_inputs['in_nd'] += 2. * np.tile(df_dIs, local_size).reshape((local_size, size)).T.dot(d_outputs['out_dist'])
            if 'out_nd' in d_outputs:
                if 'in_dist' in d_inputs:
                    full = np.zeros(d_outputs['out_nd'].size)
                    # add up contributions from the non-distributed variable that is duplicated over
                    # all of the procs.
                    self.comm.Allreduce(d_outputs['out_nd'], full, op=MPI.SUM)
                    d_inputs['in_dist'] += .5 * np.tile(dg_dId, size).reshape((size, local_size)).T.dot(full)
                if 'in_nd' in d_inputs:
                    d_inputs['in_nd'] += (2.0 * Is + 3.0) * d_outputs['out_nd']


class Distrib_DerivsFD(Distrib_Derivs):

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')


class Distrib_DerivsErr(Distrib_Derivs):

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute_partials(self, inputs, partials):
        pass  # do nothing here.  Error will occur before calling this.

# this is similar to Distrib_Derivs except we add in the product of inputs rather than the
# summation of inputs.
class Distrib_Derivs_Prod(om.ExplicitComponent):

    def setup(self):

        self.add_input('in_dist', shape_by_conn=True, distributed=True)
        self.add_input('in_nd', shape_by_conn=True)

        self.add_output('out_dist', copy_shape='in_dist', distributed=True)
        self.add_output('out_nd', copy_shape='in_nd')

    def compute(self, inputs, outputs):
        Id = inputs['in_dist']
        Is = inputs['in_nd']

        # Our local distributed output is a function of local distributed input and the
        # non-distributed input.
        outputs['out_dist'] = Id**2 - 2.0*Id + 4.0 + np.prod(1.5 * Is ** 2)

        if self.comm.size > 1:
            # We need to gather the multplied values to compute the total product over all procs.
            local_prod = np.array([np.prod(1.5 * Id ** 2)])
            total_prod = local_prod.copy()
            self.comm.Allreduce(local_prod, total_prod, op=MPI.PROD)

        outputs['out_nd'] = Is**2 + 3.0*Is - 5.0 + total_prod


class Distrib_Derivs_Prod_Matfree(Distrib_Derivs_Prod):
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        Id = inputs['in_dist']
        Is = inputs['in_nd']

        size = len(Is)
        local_size = len(Id)

        idx = self._var_allprocs_abs2idx[rel_name2abs_name(self, 'in_dist')]
        sizes = self._var_sizes['input'][:, idx]
        start = np.sum(sizes[:self.comm.rank])
        end = start + sizes[self.comm.rank]

        d_dIs = np.zeros((local_size, size))
        for i in range(size):
            d_dIs[:, i] = 3.*Is[i]*np.prod([1.5*Is[j]**2 for j in range(size) if i != j])

        # unfortunately we need the full distributed input here in order to compute the d_dId
        # matrix.
        Idfull = np.hstack(self.comm.allgather(Id))
        d_dId = np.zeros((size, local_size))
        for i in range(start, end):
            d_dId[:, i-start] = 3.*Idfull[i]*np.prod([1.5*Idfull[j]**2 for j in range(size) if i != j])

        if mode == 'fwd':
            if 'out_dist' in d_outputs:
                if 'in_dist' in d_inputs:
                    d_outputs['out_dist'] += (2.0 * Id - 2.0) * d_inputs['in_dist']
                if 'in_nd' in d_inputs:
                    d_outputs['out_dist'] += d_dIs.dot(d_inputs['in_nd'])
            if 'out_nd' in d_outputs:
                if 'in_dist' in d_inputs:
                    deriv = d_dId.dot(d_inputs['in_dist'])
                    deriv_sum = np.zeros(deriv.size)
                    self.comm.Allreduce(deriv, deriv_sum, op=MPI.SUM)
                    d_outputs['out_nd'] += deriv_sum
                if 'in_nd' in d_inputs:
                    d_outputs['out_nd'] += (2.0 * Is + 3.0) * d_inputs['in_nd']
        else:  # rev
            if 'out_dist' in d_outputs:
                if 'in_dist' in d_inputs:
                    d_inputs['in_dist'] += (2.0 * Id - 2.0) * d_outputs['out_dist']
                if 'in_nd' in d_inputs:
                    deriv = d_dIs.T.dot(d_outputs['out_dist'])
                    deriv_sum = np.zeros(deriv.size)
                    self.comm.Allreduce(deriv, deriv_sum, op=MPI.SUM)
                    d_inputs['in_nd'] += deriv_sum
            if 'out_nd' in d_outputs:
                if 'in_dist' in d_inputs:
                    d_inputs['in_dist'] += d_dId.T.dot(d_outputs['out_nd'])
                if 'in_nd' in d_inputs:
                    d_inputs['in_nd'] += (2.0 * Is + 3.0) * d_outputs['out_nd']


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestDistribBugs(unittest.TestCase):

    N_PROCS = 2

    def get_problem(self, comp_class, mode='auto', stacked=False):
        size = 5

        comm = MPI.COMM_WORLD
        rank = comm.rank
        sizes, offsets = evenly_distrib_idxs(comm.size, size)

        model = om.Group()

        ivc = om.IndepVarComp()
        ivc.add_output('x_dist', np.zeros(sizes[rank]), distributed=True)
        ivc.add_output('x_serial', np.zeros(size))

        model.add_subsystem("indep", ivc)
        model.add_subsystem("D1", comp_class())
        if stacked:
            model.add_subsystem("D2", comp_class())

        model.connect('indep.x_dist', 'D1.in_dist')
        model.connect('indep.x_serial', 'D1.in_nd')
        if stacked:
            model.connect('D1.out_dist', 'D2.in_dist')
            model.connect('D1.out_nd', 'D2.in_nd')

        prob = om.Problem(model)
        prob.setup(mode=mode, force_alloc_complex=True)

        self.x_dist_init = x_dist_init = (3.0 + np.arange(size)[offsets[rank]:offsets[rank] + sizes[rank]]) * .1
        self.x_serial_init = x_serial_init = (1.0 + 2.0*np.arange(size)) * .1

        # This set operates on the entire vector.
        prob.set_val('indep.x_dist', x_dist_init)
        prob.set_val('indep.x_serial', x_serial_init)

        prob.run_model()

        return prob

    def _compare_totals(self, totals):
        fails = []
        for key, val in totals.items():
            try:
                analytic = val['J_fwd']
                fd = val['J_fd']
            except Exception as err:
                self.fail(f"For key {key}: {err}")
            try:
                assert_near_equal(val['rel error'][0], 0.0, 1e-6)
            except ValueError as err:
                fails.append((key, val, err))
        if fails:
            msg = '\n\n'.join([f"Totals differ for {key}:\nAnalytic:\n{val['J_fwd']}\nFD:\n{val['J_fd']}\n{err}" for key, val, err in fails])
            self.fail(msg)

    def test_get_val(self):
        prob = self.get_problem(Distrib_Derivs_Matfree, stacked=False)
        indep = prob.model.indep
        full_dist_init = np.hstack(indep.comm.allgather(indep._outputs['x_dist'].flat[:]))
        D1_out_dist_full = f_out_dist(full_dist_init, self.x_serial_init)
        if prob.model.comm.rank == 0:
            D1_out_dist = f_out_dist(self.x_dist_init, self.x_serial_init)
        else:
            D1_out_dist = f_out_dist(self.x_dist_init, self.x_serial_init)

        D1_out_nd = f_out_nd(full_dist_init, self.x_serial_init)

        vnames = ['indep.x_dist', 'indep.x_serial', 'D1.out_dist', 'D1.out_nd']
        expected = [self.x_dist_init, self.x_serial_init, D1_out_dist, D1_out_nd]
        expected_remote = [full_dist_init, self.x_serial_init, D1_out_dist_full, D1_out_nd]
        for var, ex, ex_remote in zip(vnames, expected, expected_remote):
            val = prob.get_val(var)
            full_val = prob.get_val(var, get_remote=True)
            assert_near_equal(val, ex, tolerance=1e-8)
            assert_near_equal(full_val, ex_remote, tolerance=1e-8)

    def test_check_totals_fwd(self):
        prob = self.get_problem(Distrib_Derivs_Matfree, mode='fwd')
        totals = prob.check_totals(method='cs', out_stream=None, of=['D1.out_nd', 'D1.out_dist'],
                                        wrt=['indep.x_serial', 'indep.x_dist'])
        self._compare_totals(totals)

    def test_check_totals_prod_fwd(self):
        prob = self.get_problem(Distrib_Derivs_Prod_Matfree, mode='fwd')
        totals = prob.check_totals(method='cs', out_stream=None, of=['D1.out_nd', 'D1.out_dist'],
                                        wrt=['indep.x_serial', 'indep.x_dist'])
        self._compare_totals(totals)

    def test_check_totals_rev(self):
        prob = self.get_problem(Distrib_Derivs_Matfree, mode='rev')
        totals = prob.check_totals(method='cs', out_stream=None, of=['D1.out_nd', 'D1.out_dist'],
                                                   wrt=['indep.x_serial', 'indep.x_dist'])
        self._compare_totals(totals)

    def test_check_totals_rev_old(self):
        prob = self.get_problem(Distrib_Derivs_Matfree_Old, mode='rev')
        data = prob.check_totals(method='cs',
                                 of=['D1.out_nd', 'D1.out_dist'], wrt=['indep.x_serial', 'indep.x_dist'],
                                 show_only_incorrect=True)
        with self.assertRaises(ValueError) as cm:
            assert_check_totals(data)

        msg = "During total derivative computation, the following partial derivatives resulted in serial inputs that were inconsistent across processes: ['D1.out_dist wrt D1.in_nd']."
        self.assertEquals(str(cm.exception), msg)

    def test_check_partials_cs_old(self):
        prob = self.get_problem(Distrib_Derivs_Matfree_Old)
        data = prob.check_partials(method='cs', show_only_incorrect=True)
        with self.assertRaises(ValueError) as cm:
            assert_check_partials(data)

        msg = "Inconsistent derivs across processes for keys: [('out_dist', 'in_nd')].\nCheck that distributed outputs are properly reduced when computing\nderivatives of serial inputs."
        self.assertTrue(str(cm.exception).endswith(msg))

    def test_check_totals_prod_rev(self):
        prob = self.get_problem(Distrib_Derivs_Prod_Matfree, mode='rev')
        totals = prob.check_totals(method='cs', out_stream=None, of=['D1.out_nd', 'D1.out_dist'],
                                                   wrt=['indep.x_serial', 'indep.x_dist'])
        self._compare_totals(totals)

    def test_check_totals_fwd_stacked(self):
        prob = self.get_problem(Distrib_Derivs_Matfree, mode='fwd', stacked=True)
        totals = prob.check_totals(method='cs', out_stream=None, of=['D2.out_nd', 'D2.out_dist'],
                                        wrt=['indep.x_serial', 'indep.x_dist'])
        self._compare_totals(totals)

    def test_check_totals_prod_fwd_stacked(self):
        prob = self.get_problem(Distrib_Derivs_Prod_Matfree, mode='fwd', stacked=True)
        totals = prob.check_totals(method='cs', out_stream=None, of=['D2.out_nd', 'D2.out_dist'],
                                        wrt=['indep.x_serial', 'indep.x_dist'])
        self._compare_totals(totals)

    def test_check_totals_rev_stacked(self):
        prob = self.get_problem(Distrib_Derivs_Matfree, mode='rev', stacked=True)
        totals = prob.check_totals(method='cs', out_stream=None, of=['D2.out_nd', 'D2.out_dist'],
                                                   wrt=['indep.x_serial', 'indep.x_dist'])
        self._compare_totals(totals)

    def test_check_totals_prod_rev_stacked(self):
        prob = self.get_problem(Distrib_Derivs_Prod_Matfree, mode='rev', stacked=True)
        totals = prob.check_totals(method='cs', out_stream=None, of=['D2.out_nd', 'D2.out_dist'],
                                                   wrt=['indep.x_serial', 'indep.x_dist'])
        self._compare_totals(totals)

    def test_check_partials_cs(self):
        prob = self.get_problem(Distrib_Derivs_Matfree)
        data = prob.check_partials(method='cs', show_only_incorrect=True)
        assert_check_partials(data)

    def test_check_partials_prod_cs(self):
        prob = self.get_problem(Distrib_Derivs_Prod_Matfree)
        data = prob.check_partials(method='cs', show_only_incorrect=True)
        assert_check_partials(data)

    def test_check_partials_fd(self):
        prob = self.get_problem(Distrib_Derivs_Matfree)
        data = prob.check_partials(method='fd', show_only_incorrect=True)
        assert_check_partials(data, atol=6e-6, rtol=1.5e-6)

    def test_check_partials_prod_fd(self):
        prob = self.get_problem(Distrib_Derivs_Prod_Matfree)
        data = prob.check_partials(method='fd', show_only_incorrect=True)
        assert_check_partials(data, rtol=5e-6, atol=3e-6)

    def test_check_err(self):
        with self.assertRaises(RuntimeError) as cm:
            prob = self.get_problem(Distrib_DerivsErr)

        msg = "'D1' <class Distrib_DerivsErr>: component has defined partial ('out_nd', 'in_dist') which is a non-distributed output wrt a distributed input. This is only supported using the matrix free API."
        self.assertEqual(str(cm.exception), msg)

    def test_fd_check_err(self):
        with self.assertRaises(RuntimeError) as cm:
            prob = self.get_problem(Distrib_DerivsFD, mode='fwd')

        msg = "'D1' <class Distrib_DerivsFD>: component has defined partial ('out_nd', 'in_dist') which is a non-distributed output wrt a distributed input. This is only supported using the matrix free API."
        self.assertEqual(str(cm.exception), msg)

    def test_constraint_aliases(self):
        size = 7

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', np.ones((size, )))
        ivc.add_output('y', np.ones((size, )))
        ivc.add_output('a', -3.0 + 0.6 * np.arange(size))

        model.add_subsystem('p', ivc, promotes=['*'])
        model.add_subsystem("parab", DistParab(arr_size=size, deriv_type='dense'), promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)

        model.add_constraint('f_xy', indices=[5], flat_indices=True, lower=10.0)
        model.add_constraint('f_xy', indices=[1], flat_indices=True, alias='a2', lower=0.5)

        prob.setup(mode='fwd', force_alloc_complex=True)

        prob.run_driver()

        desvar = prob.driver.get_design_var_values()
        con = prob.driver.get_constraint_values()

        assert_near_equal(con['parab.f_xy'], 24.0)
        assert_near_equal(con['a2'], 24.96)

        totals = prob.check_totals(method='cs', out_stream=None)
        self._compare_totals(totals)


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
