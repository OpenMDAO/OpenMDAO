""" Testing for Problem.check_partials and check_totals."""

from io import StringIO


import unittest

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.expl_comp_array import TestExplCompArrayDense
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDerivatives, SellarDis1withDerivatives, \
     SellarDis2withDerivatives
from openmdao.test_suite.components.simple_comps import DoubleArrayComp
from openmdao.test_suite.groups.parallel_groups import FanInSubbedIDVC, Diamond
from openmdao.utils.assert_utils import assert_near_equal, assert_check_totals
from openmdao.core.tests.test_check_partials import ParaboloidTricky, MyCompGoodPartials, \
    MyCompBadPartials, DirectionalVectorizedMatFreeComp
from openmdao.test_suite.scripts.circle_opt import CircleOpt
from openmdao.core.constants import _UNDEFINED
import openmdao.core.total_jac as tot_jac_mod

from openmdao.utils.mpi import MPI

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

try:
    from pyoptsparse import Optimization as pyoptsparse_opt
except ImportError:
    pyoptsparse_opt = None


class DistribParaboloid(om.ExplicitComponent):

    def setup(self):
        if self.comm.rank == 0:
            ndvs = 3
        else:
            ndvs = 2

        self.add_input('w', val=1., distributed=True) # this will connect to a non-distributed IVC
        self.add_input('x', shape=ndvs, distributed=True) # this will connect to a distributed IVC

        self.add_output('y', shape=2, distributed=True) # all-gathered output, duplicated on all procs
        self.add_output('z', shape=ndvs, distributed=True) # distributed output
        self.declare_partials('y', 'x')
        self.declare_partials('y', 'w')
        self.declare_partials('z', 'x')

    def compute(self, inputs, outputs):
        x = inputs['x']
        local_y = np.sum((x-5)**2)
        y_g = np.zeros(self.comm.size)
        self.comm.Allgather(local_y, y_g)
        val = np.sum(y_g) + (inputs['w']-10)**2
        outputs['y'] = np.array([val, val*3.])
        outputs['z'] = x**2

    def compute_partials(self, inputs, J):
        x = inputs['x']
        J['y', 'x'] = np.array([2*(x-5), 6*(x-5)])
        J['y', 'w'] = np.array([2*(inputs['w']-10), 6*(inputs['w']-10)])
        J['z', 'x'] = np.diag(2*x)


class DistribParaboloid2D(om.ExplicitComponent):

    def setup(self):

        comm = self.comm
        rank = comm.rank

        if rank == 0:
            vshape = (3,2)
        else:
            vshape = (2,2)

        self.add_input('w', val=1., distributed=True) # this will connect to a non-distributed IVC
        self.add_input('x', shape=vshape, distributed=True) # this will connect to a distributed IVC

        self.add_output('y', distributed=True) # all-gathered output, duplicated on all procs
        self.add_output('z', shape=vshape, distributed=True) # distributed output
        self.declare_partials('y', 'x')
        self.declare_partials('y', 'w')
        self.declare_partials('z', 'x')

    def compute(self, inputs, outputs):
        x = inputs['x']
        local_y = np.sum((x-5)**2)
        y_g = np.zeros(self.comm.size)
        self.comm.Allgather(local_y, y_g)
        outputs['y'] = np.sum(y_g) + (inputs['w']-10)**2
        outputs['z'] = x**2

    def compute_partials(self, inputs, J):
        x = inputs['x'].flatten()
        J['y', 'x'] = 2*(x-5)
        J['y', 'w'] = 2*(inputs['w']-10)
        J['z', 'x'] = np.diag(2*x)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestProblemComputeTotalsGetRemoteFalse(unittest.TestCase):

    N_PROCS = 2

    def _do_compute_totals(self, mode):
        comm = MPI.COMM_WORLD

        p = om.Problem()
        d_ivc = p.model.add_subsystem('distrib_ivc',
                                    om.IndepVarComp(distributed=True),
                                    promotes=['*'])
        if comm.rank == 0:
            ndvs = 3
        else:
            ndvs = 2
        d_ivc.add_output('x', 2*np.ones(ndvs))

        ivc = p.model.add_subsystem('ivc',
                                    om.IndepVarComp(distributed=False),
                                    promotes=['*'])
        ivc.add_output('w', 2.0)
        p.model.add_subsystem('dp', DistribParaboloid(), promotes=['*'])

        p.model.add_design_var('x', lower=-100, upper=100)
        p.model.add_objective('y')

        p.setup(mode=mode)
        p.run_model()

        dv_vals = p.driver.get_design_var_values(get_remote=False)

        # Compute totals and check the length of the gradient array on each proc
        J = p.compute_totals(get_remote=False)

        # Check the values of the gradient array
        assert_near_equal(J[('y', 'x')][0], -6.0*np.ones(ndvs))
        assert_near_equal(J[('y', 'x')][1], -18.0*np.ones(ndvs))

    def test_distrib_compute_totals_fwd(self):
        self._do_compute_totals('fwd')

    def test_distrib_compute_totals_rev(self):
        self._do_compute_totals('rev')

    def _do_compute_totals_2D(self, mode):
        # this test has some non-flat variables
        comm = MPI.COMM_WORLD

        p = om.Problem()
        d_ivc = p.model.add_subsystem('distrib_ivc',
                                      om.IndepVarComp(distributed=True),
                                      promotes=['*'])
        if comm.rank == 0:
            ndvs = 6
            two_d = (3,2)
        else:
            ndvs = 4
            two_d = (2,2)

        d_ivc.add_output('x', 2*np.ones(two_d))

        ivc = p.model.add_subsystem('ivc',
                                    om.IndepVarComp(distributed=False),
                                    promotes=['*'])
        ivc.add_output('w', 2.0)
        p.model.add_subsystem('dp', DistribParaboloid2D(), promotes_outputs=['*'])
        p.model.connect('w', 'dp.w', src_indices=np.array([0]), flat_src_indices=True)
        p.model.connect('x', 'dp.x')

        p.model.add_design_var('x', lower=-100, upper=100)
        p.model.add_objective('y')

        p.setup(mode=mode)
        p.run_model()

        dv_vals = p.driver.get_design_var_values(get_remote=False)

        # Compute totals and check the length of the gradient array on each proc
        J = p.compute_totals(get_remote=False)

        # Check the values of the gradient array
        assert_near_equal(J[('y', 'x')][0], -6.0*np.ones(ndvs))

    def test_distrib_compute_totals_2D_fwd(self):
        self._do_compute_totals_2D('fwd')

    def test_distrib_compute_totals_2D_rev(self):
        self._do_compute_totals_2D('rev')

    def _remotevar_compute_totals(self, mode):
        indep_list = ['iv.x']
        unknown_list = [
            'c1.y1',
            'c1.y2',
            'sub.c2.y1',
            'sub.c3.y1',
            'c4.y1',
            'c4.y2',
        ]

        full_expected = {
            ('c1.y1', 'iv.x'): [[8.]],
            ('c1.y2', 'iv.x'): [[3.]],
            ('sub.c2.y1', 'iv.x'): [[4.]],
            ('sub.c3.y1', 'iv.x'): [[10.5]],
            ('c4.y1', 'iv.x'): [[25.]],
            ('c4.y2', 'iv.x'): [[-40.5]],
        }

        prob = om.Problem()
        prob.model = Diamond()

        prob.setup(mode=mode)
        prob.set_solver_print(level=0)
        prob.run_model()

        assert_near_equal(prob['c4.y1'], 46.0, 1e-6)
        assert_near_equal(prob['c4.y2'], -93.0, 1e-6)

        J = prob.compute_totals(of=unknown_list, wrt=indep_list)
        for key, val in full_expected.items():
            assert_near_equal(J[key], val, 1e-6)

        reduced_expected = {key: v for key, v in full_expected.items() if key[0] in prob.model._var_abs2meta['output']}

        J = prob.compute_totals(of=unknown_list, wrt=indep_list, get_remote=False)
        for key, val in reduced_expected.items():
            assert_near_equal(J[key], val, 1e-6)
        self.assertEqual(len(J), len(reduced_expected))

    def test_remotevar_compute_totals_fwd(self):
        self._remotevar_compute_totals('fwd')

    def test_remotevar_compute_totals_rev(self):
        self._remotevar_compute_totals('rev')


class I2O2JacVec(om.ExplicitComponent):

    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def setup(self):
        self.add_input('in1', val=np.ones(self.size))
        self.add_input('in2', val=np.ones(self.size))
        self.add_output('out1', val=np.ones(self.size))
        self.add_output('out2', val=np.ones(self.size))

    def compute(self, inputs, outputs):
        outputs['out1'] = inputs['in1'] * inputs['in2']
        outputs['out2'] = 3. * inputs['in1'] + 5. * inputs['in2']

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'out1' in d_outputs:
                if 'in1' in d_inputs:
                    d_outputs['out1'] += inputs['in2'] * d_inputs['in1']
                if 'in2' in d_inputs:
                    d_outputs['out1'] += inputs['in1'] * d_inputs['in2']
            if 'out2' in d_outputs:
                if 'in1' in d_inputs:
                    d_outputs['out2'] += 3. * d_inputs['in1']
                if 'in2' in d_inputs:
                    d_outputs['out2'] += 5. * d_inputs['in2']
        else:  # rev
            if 'out1' in d_outputs:
                if 'in1' in d_inputs:
                    d_inputs['in1'] += inputs['in2'] * d_outputs['out1']
                if 'in2' in d_inputs:
                    d_inputs['in2'] += inputs['in1'] * d_outputs['out1']
            if 'out2' in d_outputs:
                if 'in1' in d_inputs:
                    d_inputs['in1'] += 3. * d_outputs['out2']
                if 'in2' in d_inputs:
                    d_inputs['in2'] += 5. * d_outputs['out2']


class Simple(om.ExplicitComponent):
    """
    Simple component that counts compute/compute_partials/_solve_linear.
    """

    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.ncomputes = 0
        self.ncompute_partials = 0
        self.nsolve_linear = 0

    def setup(self):
        self.add_input('x', val=np.ones(self.size))
        self.add_output('y', val=np.ones(self.size))

        rows = np.arange(self.size)
        self.declare_partials(of=['y'], wrt=['x'], rows=rows, cols=rows)

    def compute(self, inputs, outputs):
        outputs['y'] = inputs['x'] * 2.0
        self.ncomputes += 1

    def compute_partials(self, inputs, partials):
        partials['y', 'x'] = np.ones(self.size) * 2.0
        self.ncompute_partials += 1

    def _solve_linear(self, mode, scope_out=_UNDEFINED, scope_in=_UNDEFINED):
        super()._solve_linear(mode, scope_out, scope_in)
        self.nsolve_linear += 1


class SparseJacVec(om.ExplicitComponent):
    """
    Simple matrix free component where some of/wrt partials are zero.
    """

    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def setup(self):
        self.add_input('in1', val=np.ones(self.size))
        self.add_input('in2', val=np.ones(self.size))
        self.add_input('in3', val=np.ones(self.size))
        self.add_input('in4', val=np.ones(self.size))
        self.add_output('out1', val=np.ones(self.size))
        self.add_output('out2', val=np.ones(self.size))
        self.add_output('out3', val=np.ones(self.size))
        self.add_output('out4', val=np.ones(self.size))

        # declare partials to improve var sparsity in the full model
        self.declare_partials(of=['out1', 'out2'], wrt=['in1', 'in2'])
        self.declare_partials(of=['out3', 'out4'], wrt=['in3', 'in4'])

    def compute(self, inputs, outputs):
        outputs['out1'] = inputs['in1'] * inputs['in2']
        outputs['out2'] = 3. * inputs['in1'] + 5. * inputs['in2']
        outputs['out3'] = inputs['in3'] * inputs['in4']
        outputs['out4'] = 7. * inputs['in3'] + 9. * inputs['in4']

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == 'fwd':
            if 'out1' in d_outputs:
                if 'in1' in d_inputs:
                    d_outputs['out1'] += inputs['in2'] * d_inputs['in1']
                if 'in2' in d_inputs:
                    d_outputs['out1'] += inputs['in1'] * d_inputs['in2']
            if 'out2' in d_outputs:
                if 'in1' in d_inputs:
                    d_outputs['out2'] += 3. * d_inputs['in1']
                if 'in2' in d_inputs:
                    d_outputs['out2'] += 5. * d_inputs['in2']
            if 'out3' in d_outputs:
                if 'in3' in d_inputs:
                    d_outputs['out3'] += inputs['in4'] * d_inputs['in3']
                if 'in4' in d_inputs:
                    d_outputs['out3'] += inputs['in3'] * d_inputs['in4']
            if 'out4' in d_outputs:
                if 'in3' in d_inputs:
                    d_outputs['out4'] += 7. * d_inputs['in3']
                if 'in4' in d_inputs:
                    d_outputs['out4'] += 9. * d_inputs['in4']
        else:  # rev
            if 'out1' in d_outputs:
                if 'in1' in d_inputs:
                    d_inputs['in1'] += inputs['in2'] * d_outputs['out1']
                if 'in2' in d_inputs:
                    d_inputs['in2'] += inputs['in1'] * d_outputs['out1']
            if 'out2' in d_outputs:
                if 'in1' in d_inputs:
                    d_inputs['in1'] += 3. * d_outputs['out2']
                if 'in2' in d_inputs:
                    d_inputs['in2'] += 5. * d_outputs['out2']
            if 'out3' in d_outputs:
                if 'in3' in d_inputs:
                    d_inputs['in3'] += inputs['in4'] * d_outputs['out3']
                if 'in4' in d_inputs:
                    d_inputs['in4'] += inputs['in3'] * d_outputs['out3']
            if 'out4' in d_outputs:
                if 'in3' in d_inputs:
                    d_inputs['in3'] += 7. * d_outputs['out4']
                if 'in4' in d_inputs:
                    d_inputs['in4'] += 9. * d_outputs['out4']


class TestProblemCheckTotals(unittest.TestCase):

    def test_cs(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()
        prob.model.linear_solver = om.ScipyKrylov()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup(force_alloc_complex=True)

        prob.model.nonlinear_solver.options['atol'] = 1e-15
        prob.model.nonlinear_solver.options['rtol'] = 1e-15

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        # check derivatives with complex step and a larger step size.
        stream = StringIO()
        totals = prob.check_totals(method='cs', out_stream=stream)

        lines = stream.getvalue().splitlines()

        # Make sure auto-ivc sources are translated to promoted input names.
        self.assertTrue('x' in lines[4])

        self.assertTrue('9.80614' in lines[5], "'9.80614' not found in '%s'" % lines[5])
        self.assertTrue('9.80614' in lines[6], "'9.80614' not found in '%s'" % lines[6])
        self.assertTrue('cs:None' in lines[6], "'cs:None not found in '%s'" % lines[6])

        assert_near_equal(totals['con2', 'x']['J_fwd'], [[0.09692762]], 1e-5)
        assert_near_equal(totals['con2', 'x']['J_fd'], [[0.09692762]], 1e-5)

        # Test compact_print output
        compact_stream = StringIO()
        compact_totals = prob.check_totals(method='fd', out_stream=compact_stream,
            compact_print=True)

        compact_lines = compact_stream.getvalue().splitlines()

        self.assertTrue("of '<variable>'" in compact_lines[5],
            "of '<variable>' not found in '%s'" % compact_lines[5])
        self.assertTrue('9.7743e+00' in compact_lines[-2],
            "'9.7743e+00' not found in '%s'" % compact_lines[-2])

    def test_check_totals_show_progress(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup(force_alloc_complex=True)

        prob.model.nonlinear_solver.options['atol'] = 1e-15
        prob.model.nonlinear_solver.options['rtol'] = 1e-15

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        # check derivatives with complex step and a larger step size.
        stream = StringIO()
        totals = prob.check_totals(method='fd', show_progress=True, out_stream=stream)

        lines = stream.getvalue().splitlines()
        self.assertTrue("1/3: Checking derivatives with respect to: 'd1.x [2]' ..." in lines[0])
        self.assertTrue("2/3: Checking derivatives with respect to: 'd1.z [0]' ..." in lines[1])
        self.assertTrue("3/3: Checking derivatives with respect to: 'd1.z [1]' ..." in lines[2])

        prob.run_model()

        # Check to make sure nothing is going to output
        stream = StringIO()
        totals = prob.check_totals(method='fd', show_progress=False, out_stream=stream)

        lines = stream.getvalue()
        self.assertFalse("Checking derivatives with respect to" in lines)

        prob.check_totals(method='fd', show_progress=True)

    def test_desvar_as_obj(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_objective('x')

        prob.set_solver_print(level=0)

        prob.setup(force_alloc_complex=True)

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        # check derivatives with complex step and a larger step size.
        stream = StringIO()
        totals = prob.check_totals(method='cs', out_stream=stream)

        lines = stream.getvalue().splitlines()

        self.assertTrue('1.000' in lines[5])
        self.assertTrue('1.000' in lines[6])
        self.assertTrue('0.000' in lines[8])
        self.assertTrue('0.000' in lines[10])

        assert_near_equal(totals['x', 'x']['J_fwd'], [[1.0]], 1e-5)
        assert_near_equal(totals['x', 'x']['J_fd'], [[1.0]], 1e-5)

    def test_desvar_and_response_with_indices(self):

        class ArrayComp2D(om.ExplicitComponent):
            """
            A fairly simple array component.
            """

            def setup(self):

                self.JJ = np.array([[1.0, 3.0, -2.0, 7.0],
                                    [6.0, 2.5, 2.0, 4.0],
                                    [-1.0, 0.0, 8.0, 1.0],
                                    [1.0, 4.0, -5.0, 6.0]])

                # Params
                self.add_input('x1', np.zeros([4]))

                # Unknowns
                self.add_output('y1', np.zeros([4]))

                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                """
                Execution.
                """
                outputs['y1'] = self.JJ.dot(inputs['x1'])

            def compute_partials(self, inputs, partials):
                """
                Analytical derivatives.
                """
                partials[('y1', 'x1')] = self.JJ

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('x_param1', om.IndepVarComp('x1', np.ones((4))),
                            promotes=['x1'])
        mycomp = model.add_subsystem('mycomp', ArrayComp2D(), promotes=['x1', 'y1'])

        model.add_design_var('x1', indices=[1, 3])
        model.add_constraint('y1', indices=[0, 2])

        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        Jbase = mycomp.JJ
        of = ['y1']
        wrt = ['x1']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_near_equal(J['y1', 'x1'][0][0], Jbase[0, 1], 1e-8)
        assert_near_equal(J['y1', 'x1'][0][1], Jbase[0, 3], 1e-8)
        assert_near_equal(J['y1', 'x1'][1][0], Jbase[2, 1], 1e-8)
        assert_near_equal(J['y1', 'x1'][1][1], Jbase[2, 3], 1e-8)

        totals = prob.check_totals()
        jac = totals[('y1', 'x1')]['J_fd']
        assert_near_equal(jac[0][0], Jbase[0, 1], 1e-8)
        assert_near_equal(jac[0][1], Jbase[0, 3], 1e-8)
        assert_near_equal(jac[1][0], Jbase[2, 1], 1e-8)
        assert_near_equal(jac[1][1], Jbase[2, 3], 1e-8)

        # just verify that this doesn't raise an exception
        prob.driver.scaling_report(show_browser=False)


        # Objective instead

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('x_param1', om.IndepVarComp('x1', np.ones((4))),
                            promotes=['x1'])
        mycomp = model.add_subsystem('mycomp', ArrayComp2D(), promotes=['x1', 'y1'])

        model.add_design_var('x1', indices=[1, 3])
        model.add_objective('y1', index=1)

        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        Jbase = mycomp.JJ
        of = ['y1']
        wrt = ['x1']

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_near_equal(J['y1', 'x1'][0][0], Jbase[1, 1], 1e-8)
        assert_near_equal(J['y1', 'x1'][0][1], Jbase[1, 3], 1e-8)

        totals = prob.check_totals()
        jac = totals[('y1', 'x1')]['J_fd']
        assert_near_equal(jac[0][0], Jbase[1, 1], 1e-8)
        assert_near_equal(jac[0][1], Jbase[1, 3], 1e-8)

        # just verify that this doesn't raise an exception
        prob.driver.scaling_report(show_browser=False)

    def test_cs_suppress(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup(force_alloc_complex=True)

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        # check derivatives with complex step and a larger step size.
        totals = prob.check_totals(method='cs', out_stream=None)

        data = totals['con2', 'x']
        self.assertTrue('J_fwd' in data)
        self.assertTrue('rel error' in data)
        self.assertTrue('abs error' in data)
        self.assertTrue('magnitude' in data)

    def test_two_desvar_as_con(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_constraint('x', upper=0.0)
        prob.model.add_constraint('z', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup()

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        totals = prob.check_totals(method='fd', step=1.0e-1, out_stream=None)

        assert_near_equal(totals['x', 'x']['J_fwd'], [[1.0]], 1e-5)
        assert_near_equal(totals['x', 'x']['J_fd'], [[1.0]], 1e-5)
        assert_near_equal(totals['z', 'z']['J_fwd'], np.eye(2), 1e-5)
        assert_near_equal(totals['z', 'z']['J_fd'], np.eye(2), 1e-5)
        assert_near_equal(totals['x', 'z']['J_fwd'], [[0.0, 0.0]], 1e-5)
        assert_near_equal(totals['x', 'z']['J_fd'], [[0.0, 0.0]], 1e-5)
        assert_near_equal(totals['z', 'x']['J_fwd'], [[0.0], [0.0]], 1e-5)
        assert_near_equal(totals['z', 'x']['J_fd'], [[0.0], [0.0]], 1e-5)

    def test_full_con_with_index_desvar(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('z', lower=-100, upper=100, indices=[1])
        prob.model.add_constraint('z', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup()

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        totals = prob.check_totals(method='fd', step=1.0e-1, out_stream=None)

        assert_near_equal(totals['z', 'z']['J_fwd'], [[0.0], [1.0]], 1e-5)
        assert_near_equal(totals['z', 'z']['J_fd'], [[0.0], [1.0]], 1e-5)

    def test_full_desvar_with_index_con(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_constraint('z', upper=0.0, indices=[1])

        prob.set_solver_print(level=0)

        prob.setup()

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        totals = prob.check_totals(method='fd', step=1.0e-1, out_stream=None)

        assert_near_equal(totals['z', 'z']['J_rev'], [[0.0, 1.0]], 1e-5)
        assert_near_equal(totals['z', 'z']['J_fd'], [[0.0, 1.0]], 1e-5)

    def test_full_desvar_with_index_obj(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('z', index=1)

        prob.set_solver_print(level=0)

        prob.setup()

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        totals = prob.check_totals(method='fd', step=1.0e-1, out_stream=None)

        assert_near_equal(totals['z', 'z']['J_rev'], [[0.0, 1.0]], 1e-5)
        assert_near_equal(totals['z', 'z']['J_fd'], [[0.0, 1.0]], 1e-5)

    def test_bug_fd_with_sparse(self):
        # This bug was found via the x57 model in pointer.

        class TimeComp(om.ExplicitComponent):

            def setup(self):
                self.node_ptau = node_ptau = np.array([-1., 0., 1.])

                self.add_input('t_duration', val=1.)
                self.add_output('time', shape=len(node_ptau))

                # Setup partials
                nn = 3
                rs = np.arange(nn)
                cs = np.zeros(nn)

                self.declare_partials(of='time', wrt='t_duration', rows=rs, cols=cs, val=1.0)

            def compute(self, inputs, outputs):
                node_ptau = self.node_ptau
                t_duration = inputs['t_duration']

                outputs['time'][:] = 0.5 * (node_ptau + 33) * t_duration

            def compute_partials(self, inputs, jacobian):
                node_ptau = self.node_ptau

                jacobian['time', 't_duration'] = 0.5 * (node_ptau + 33)

        class CellComp(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('num_nodes', types=int)

            def setup(self):
                n = self.options['num_nodes']

                self.add_input('I_Li', val=3.25*np.ones(n))
                self.add_output('zSOC', val=np.ones(n))

                # Partials
                ar = np.arange(n)
                self.declare_partials(of='zSOC', wrt='I_Li', rows=ar, cols=ar)

            def compute(self, inputs, outputs):
                I_Li = inputs['I_Li']
                outputs['zSOC'] = -I_Li / (3600.0)

            def compute_partials(self, inputs, partials):
                partials['zSOC', 'I_Li'] = -1./(3600.0)

        class GaussLobattoPhase(om.Group):

            def setup(self):
                self.connect('t_duration', 'time.t_duration')

                indep = om.IndepVarComp()
                indep.add_output('t_duration', val=1.0)
                self.add_subsystem('time_extents', indep, promotes_outputs=['*'])
                self.add_design_var('t_duration', 5.0, 25.0)

                time_comp = TimeComp()
                self.add_subsystem('time', time_comp, promotes_outputs=['time'])

                self.add_subsystem(name='cell', subsys=CellComp(num_nodes=3))

                self.linear_solver = om.ScipyKrylov()
                self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
                self.nonlinear_solver.options['maxiter'] = 1

            def initialize(self):
                self.options.declare('ode_class', desc='System defining the ODE.')

        p = om.Problem(model=GaussLobattoPhase())

        p.model.add_objective('time', index=-1)

        p.model.linear_solver = om.ScipyKrylov(assemble_jac=True)

        p.setup(mode='fwd')
        p.set_solver_print(level=0)
        p.run_model()

        # Make sure we don't bomb out with an error.
        J = p.check_totals(out_stream=None)

        assert_near_equal(J[('time', 't_duration')]['J_fwd'][0], 17.0, 1e-5)
        assert_near_equal(J[('time', 't_duration')]['J_fd'][0], 17.0, 1e-5)

        # Try again with a direct solver and sparse assembled hierarchy.

        p = om.Problem()
        p.model.add_subsystem('sub', GaussLobattoPhase())

        p.model.sub.add_objective('time', index=-1)

        p.model.linear_solver = om.DirectSolver(assemble_jac=True)

        p.setup(mode='fwd')
        p.set_solver_print(level=0)
        p.run_model()

        # Make sure we don't bomb out with an error.
        J = p.check_totals(out_stream=None)

        assert_near_equal(J[('sub.time', 'sub.t_duration')]['J_fwd'][0], 17.0, 1e-5)
        assert_near_equal(J[('sub.time', 'sub.t_duration')]['J_fd'][0], 17.0, 1e-5)

        # Make sure check_totals cleans up after itself by running it a second time
        J = p.check_totals(out_stream=None)

        assert_near_equal(J[('sub.time', 'sub.t_duration')]['J_fwd'][0], 17.0, 1e-5)
        assert_near_equal(J[('sub.time', 'sub.t_duration')]['J_fd'][0], 17.0, 1e-5)

    def test_vector_scaled_derivs(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp(name="x", val=np.ones((2, ))))
        comp = model.add_subsystem('comp', DoubleArrayComp())
        model.connect('px.x', 'comp.x1')

        model.add_design_var('px.x', ref=np.array([2.0, 3.0]), ref0=np.array([0.5, 1.5]))
        model.add_objective('comp.y1', ref=np.array([[7.0, 11.0]]), ref0=np.array([5.2, 6.3]))
        model.add_constraint('comp.y2', lower=0.0, upper=1.0,
                             ref=np.array([[2.0, 4.0]]), ref0=np.array([1.2, 2.3]))

        prob.setup()
        prob.run_driver()

        # First, test that we get scaled results in compute and check totals.

        derivs = prob.compute_totals(of=['comp.y1'], wrt=['px.x'], return_format='dict',
                                     driver_scaling=True)

        oscale = np.array([1.0/(7.0-5.2), 1.0/(11.0-6.3)])
        iscale = np.array([2.0-0.5, 3.0-1.5])
        J = np.zeros((2, 2))
        J[:] = comp.JJ[0:2, 0:2]

        # doing this manually so that I don't inadvertantly make an error in
        # the vector math in both the code and test.
        J[0, 0] *= oscale[0]*iscale[0]
        J[0, 1] *= oscale[0]*iscale[1]
        J[1, 0] *= oscale[1]*iscale[0]
        J[1, 1] *= oscale[1]*iscale[1]
        assert_near_equal(J, derivs['comp.y1']['px.x'], 1.0e-3)

        cderiv = prob.check_totals(driver_scaling=True, out_stream=None)
        assert_near_equal(cderiv['comp.y1', 'px.x']['J_fwd'], J, 1.0e-3)

        # cleanup after FD
        prob.run_model()

        # Now, test that default is unscaled.

        derivs = prob.compute_totals(of=['comp.y1'], wrt=['px.x'], return_format='dict')

        J = comp.JJ[0:2, 0:2]
        assert_near_equal(J, derivs['comp.y1']['px.x'], 1.0e-3)

        cderiv = prob.check_totals(out_stream=None)
        assert_near_equal(cderiv['comp.y1', 'px.x']['J_fwd'], J, 1.0e-3)

    def test_cs_around_newton(self):
        # Basic sellar test.

        prob = om.Problem()
        model = prob.model
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        sub.linear_solver = om.DirectSolver(assemble_jac=False)

        # Need this.
        model.linear_solver = om.LinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(level=0)

        prob.run_model()

        totals = prob.check_totals(method='cs', out_stream=None)

        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-10)

    def test_cs_around_newton_new_method(self):
        # The old method of nudging the Newton and forcing it to reconverge could not achieve the
        # same accuracy on this model. (1e8 vs 1e12)

        class SellarDerivatives(om.Group):

            def setup(self):
                self.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
                self.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

                self.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
                self.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])
                sub = self.add_subsystem('sub', om.Group(), promotes=['*'])

                sub.linear_solver = om.DirectSolver(assemble_jac=True)
                sub.options['assembled_jac_type'] = 'csc'

                obj = sub.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)', obj=0.0,
                                                         x=0.0, z=np.array([0.0, 0.0]), y1=0.0, y2=0.0),
                                  promotes=['obj', 'x', 'z', 'y1', 'y2'])
                obj.declare_partials(of='*', wrt='*', method='cs')

                con1 = sub.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1', con1=0.0, y1=0.0),
                                  promotes=['con1', 'y1'])
                con2 = sub.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0', con2=0.0, y2=0.0),
                                  promotes=['con2', 'y2'])
                con1.declare_partials(of='*', wrt='*', method='cs')
                con2.declare_partials(of='*', wrt='*', method='cs')

                self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
                self.linear_solver = om.DirectSolver(assemble_jac=False)


        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.set_solver_print(level=0)
        prob.setup(force_alloc_complex=True)

        prob.run_model()

        wrt = ['z', 'x']
        of = ['obj', 'con1', 'con2']

        totals = prob.check_totals(of=of, wrt=wrt, method='cs', compact_print=False)
        assert_check_totals(totals, atol=1e-12, rtol=1e-12)

    def test_cs_around_newton_in_comp(self):
        # CS around Newton in an ImplicitComponent.
        class MyComp(om.ImplicitComponent):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
                self.linear_solver = om.DirectSolver()

            def setup(self):
                self.add_input('mm', np.ones(1))
                self.add_output('Re', np.ones((1, 1)))
                self.add_output('temp', np.ones((1, 1)))

            def setup_partials(self):
                self.declare_partials('Re', 'Re', val=1.0)
                self.declare_partials('temp', ['temp', 'mm'])
                self.declare_partials('Re', ['mm'])

            def apply_nonlinear(self, inputs, outputs, residuals):
                mm = inputs['mm']
                T = np.array([389.97])
                pp = np.array([.0260239151])
                cf = 0.01
                temp = outputs['temp']
                kelvin = T / 1.8
                RE = 1.479301E9 * pp * (kelvin + 110.4) / kelvin ** 2
                su = T + 198.72
                comb = 4.593153E-6 * 0.8 * su / (RE * mm * T ** 1.5)
                reyn = RE * mm
                residuals['Re'] = outputs['Re'] - reyn
                temp_ratio = 1.0 + 0.45 * (temp / T - 1.0)
                fact = 0.035 * mm * mm
                temp_ratio += fact
                CFL = cf / (1.0 + 3.59 * np.sqrt(cf) * temp_ratio)
                prod = comb * temp ** 3
                residuals['temp'] = (1.0 / (1.0 + prod / CFL) + temp) * 0.5 - temp

            def linearize(self, inputs, outputs, partials):
                mm = inputs['mm']
                temp = outputs['temp']
                T = np.array([389.97])
                pp = np.array([.0260239151])
                cf = 0.01
                kelvin = T / 1.8
                RE = 1.479301E9 * pp * (kelvin + 110.4) / kelvin ** 2
                partials['Re', 'mm'] = -RE
                su = T + 198.72
                comb = 4.593153E-6 * 0.8 * su / (RE * mm * T ** 1.5)
                dcomb_dmm = -4.593153E-6 * 0.8 * su / (RE * mm * mm * T ** 1.5)
                temp_ratio = 1.0 + 0.45 * temp / T - 1.0
                fact = 0.035 * mm * mm
                temp_ratio += fact
                dwtr_dmm = 0.07 * mm
                dwtr_dwt = 0.45 / T
                den = 1.0 + 3.59 * np.sqrt(cf) * temp_ratio
                CFL = cf / den
                dCFL_dwtr = - cf * 3.59 * np.sqrt(cf) / den ** 2
                dCFL_dmm = dCFL_dwtr * dwtr_dmm
                dCFL_dwt = dCFL_dwtr * dwtr_dwt
                term = comb * temp ** 3
                den = 1.0 + term / CFL
                dreswt_dcomb = -0.5 * temp ** 3 / (CFL * den ** 2)
                dreswt_dCFL = 0.5 * term / (CFL * den) ** 2
                dreswt_dwt = -0.5 - 1.5 * comb * temp ** 2 / (CFL * den ** 2)
                partials['temp', 'mm'] = dreswt_dcomb * dcomb_dmm +  dreswt_dCFL * dCFL_dmm
                partials['temp', 'temp'] = (dreswt_dCFL * dCFL_dwt + dreswt_dwt)

        class DDG(om.Group):
            def setup(self):
                self.add_subsystem('MyComp', MyComp(), promotes=['*'])

        prob = om.Problem(model=DDG())
        model = prob.model

        model.add_objective('Re')
        model.add_design_var('mm')

        prob.setup(force_alloc_complex=True)
        prob.set_solver_print(level=0)

        prob.set_val("mm", val=0.2)

        prob.run_model()
        totals = prob.check_totals(method='cs', out_stream=None)

        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-12)

    def test_cs_around_broyden(self):
        # Basic sellar test.

        prob = om.Problem()
        model = prob.model
        sub = model.add_subsystem('sub', om.Group(), promotes=['*'])

        model.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        sub.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        sub.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                   z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        sub.nonlinear_solver = om.BroydenSolver()
        sub.linear_solver = om.DirectSolver()

        # Need this.
        model.linear_solver = om.LinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(level=0)

        prob.run_model()

        totals = prob.check_totals(method='cs', out_stream=None)

        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-6)

    def test_cs_around_newton_top_sparse(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()

        prob.setup(force_alloc_complex=True)

        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)
        prob.run_model()

        chk = prob.check_totals(of=['obj', 'con1'], wrt=['x', 'z'], method='cs', out_stream=None)
        assert_check_totals(chk, atol=3e-8, rtol=3e-8)

    def test_cs_around_broyden_top_sparse(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()

        prob.setup(force_alloc_complex=True)

        prob.model.nonlinear_solver = om.BroydenSolver()
        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)
        prob.run_model()

        chk = prob.check_totals(of=['obj', 'con1'], wrt=['x', 'z'], method='cs', out_stream=None)
        assert_check_totals(chk, atol=7e-8, rtol=7e-8)

    def test_check_totals_on_approx_model(self):
        prob = om.Problem()
        prob.model = SellarDerivatives()

        prob.model.approx_totals(method='cs')

        prob.setup(force_alloc_complex=True)

        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        prob.model.linear_solver = om.DirectSolver()
        prob.run_model()

        chk = prob.check_totals(of=['obj', 'con1'], wrt=['x', 'z'], method='cs',
                                   step = 1e-39, # needs to be different than arrox_totals or error
                                   out_stream=None)
        assert_check_totals(chk, atol=3e-8, rtol=3e-8)

    def test_cs_error_allocate(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p', om.IndepVarComp('x', 3.0), promotes=['*'])
        model.add_subsystem('comp', ParaboloidTricky(), promotes=['*'])
        prob.setup()
        prob.run_model()

        msg = "\nProblem .*: To enable complex step, specify 'force_alloc_complex=True' when calling " + \
                "setup on the problem, e\.g\. 'problem\.setup\(force_alloc_complex=True\)'"
        with self.assertRaisesRegex(RuntimeError, msg):
            prob.check_totals(method='cs')

    def test_fd_zero_check(self):

        class BadComp(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', 3.0)
                self.add_output('y', 3.0)

                self.declare_partials('y', 'x')

            def compute(self, inputs, outputs):
                pass

            def compute_partials(self, inputs, partials):
                partials['y', 'x'] = 3.0 * inputs['x'] + 5

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p', om.IndepVarComp('x', 3.0))
        model.add_subsystem('comp', BadComp())
        model.connect('p.x', 'comp.x')

        model.add_design_var('p.x')
        model.add_objective('comp.y')

        prob.setup()
        prob.run_model()

        # This test verifies fix of a TypeError (division by None)
        J = prob.check_totals(out_stream=None)
        assert_near_equal(J['comp.y', 'p.x']['J_fwd'], [[14.0]], 1e-6)
        assert_near_equal(J['comp.y', 'p.x']['J_fd'], [[0.0]], 1e-6)

    def test_response_index(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p', om.IndepVarComp('x', np.ones(2)), promotes=['*'])
        model.add_subsystem('comp', om.ExecComp('y=2*x', x=np.ones(2), y=np.ones(2)),
                            promotes=['*'])

        model.add_design_var('x')
        model.add_constraint('y', indices=[1], lower=0.0)

        prob.setup()
        prob.run_model()

        stream = StringIO()
        prob.check_totals(out_stream=stream)
        lines = stream.getvalue().splitlines()
        self.assertTrue('index size: 1' in lines[4])

    def test_linear_cons(self):
        # Linear constraints were mistakenly forgotten.
        p = om.Problem()
        p.model.add_subsystem('stuff', om.ExecComp(['y = x', 'cy = x', 'lcy = 3*x'],
                                                   x={'units': 'inch'},
                                                   y={'units': 'kg'},
                                                   lcy={'units': 'kg'}),
                              promotes=['*'])

        p.model.add_design_var('x', units='ft')
        p.model.add_objective('y', units='lbm')
        p.model.add_constraint('lcy', units='lbm', lower=0, linear=True)

        p.setup()
        p['x'] = 1.0
        p.run_model()

        stream = StringIO()
        J_driver = p.check_totals(out_stream=stream)
        lines = stream.getvalue().splitlines()

        self.assertTrue("Full Model: 'lcy' wrt 'x' (Linear constraint)" in lines[4])
        self.assertTrue("Absolute Error (Jfor - Jfd)" in lines[8])
        self.assertTrue("Relative Error (Jfor - Jfd) / Jfd" in lines[10])

        assert_near_equal(J_driver['y', 'x']['J_fwd'][0, 0], 1.0)
        assert_near_equal(J_driver['lcy', 'x']['J_fwd'][0, 0], 3.0)

    def test_alias_constraints(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('obj', om.ExecComp('o = areas[0, 0] + areas[1, 1]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        model.add_design_var('widths', lower=-50.0, upper=50.0)
        model.add_objective('o')

        model.add_constraint('areas', equals=24.0, indices=[0], flat_indices=True)
        model.add_constraint('areas', equals=21.0, indices=[1], flat_indices=True, alias='a2')
        model.add_constraint('areas', equals=3.5, indices=[2], flat_indices=True, alias='a3')
        model.add_constraint('areas', equals=17.5, indices=[3], flat_indices=True, alias='a4')

        prob.setup(mode='fwd')

        prob.run_driver()

        totals = prob.check_totals(out_stream=None)

        assert_near_equal(totals['areas', 'widths']['abs error'][0], 0.0, 1e-6)
        assert_near_equal(totals['a2', 'widths']['abs error'][0], 0.0, 1e-6)
        assert_near_equal(totals['a3', 'widths']['abs error'][0], 0.0, 1e-6)
        assert_near_equal(totals['a4', 'widths']['abs error'][0], 0.0, 1e-6)

        l = prob.list_driver_vars(show_promoted_name=True, print_arrays=False,
                                  cons_opts=['indices', 'alias'])

        # Rev mode

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('obj', om.ExecComp('o = areas[0, 0] + areas[1, 1]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        model.add_design_var('widths', lower=-50.0, upper=50.0)
        model.add_objective('o')

        model.add_constraint('areas', equals=24.0, indices=[0], flat_indices=True)
        model.add_constraint('areas', equals=21.0, indices=[1], flat_indices=True, alias='a2')
        model.add_constraint('areas', equals=3.5, indices=[2], flat_indices=True, alias='a3')
        model.add_constraint('areas', equals=17.5, indices=[3], flat_indices=True, alias='a4')

        prob.setup(mode='rev')

        result = prob.run_driver()

        totals = prob.check_totals(out_stream=None)

        assert_near_equal(totals['areas', 'widths']['abs error'][1], 0.0, 1e-6)
        assert_near_equal(totals['a2', 'widths']['abs error'][1], 0.0, 1e-6)
        assert_near_equal(totals['a3', 'widths']['abs error'][1], 0.0, 1e-6)
        assert_near_equal(totals['a4', 'widths']['abs error'][1], 0.0, 1e-6)

    def test_alias_constraints_nested(self):
        # Tests a bug where we need to lookup the constraint alias on a response that is from
        # a child system.
        prob = om.Problem()
        model = prob.model

        sub = model.add_subsystem('sub', om.Group())

        sub.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        sub.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        sub.add_subsystem('comp', Paraboloid(), promotes=['*'])
        sub.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        sub.add_design_var('x', lower=-50.0, upper=50.0)
        sub.add_design_var('y', lower=-50.0, upper=50.0)
        sub.add_objective('f_xy')
        sub.add_constraint('c', upper=-15.0, alias="Stuff")

        prob.setup()

        prob.run_model()

        totals = prob.check_totals(out_stream=None)
        assert_check_totals(totals)

    def test_exceed_tol_show_only_incorrect(self):

        prob = om.Problem()
        top = prob.model
        top.add_subsystem('goodcomp', MyCompGoodPartials())
        top.add_subsystem('badcomp', MyCompBadPartials())
        top.add_subsystem('C1', om.ExecComp('y=2.*x'))
        top.add_subsystem('C2', om.ExecComp('y=3.*x'))

        top.connect('goodcomp.y', 'C1.x')
        top.connect('badcomp.z', 'C2.x')

        top.add_objective('C1.y')
        top.add_constraint('C2.y', lower=0)
        top.add_design_var('goodcomp.x1')
        top.add_design_var('goodcomp.x2')
        top.add_design_var('badcomp.y1')
        top.add_design_var('badcomp.y2')

        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()
        prob.compute_totals()

        stream = StringIO()
        prob.check_totals(out_stream=stream, show_only_incorrect=True)

        self.assertEqual(stream.getvalue().count("'C2.y' wrt 'badcomp.y1'"), 1)
        self.assertEqual(stream.getvalue().count("'C2.y' wrt 'badcomp.y2'"), 1)
        self.assertEqual(stream.getvalue().count("'C1.y' wrt 'goodcomp.y1'"), 0)
        self.assertEqual(stream.getvalue().count("'C1.y' wrt 'goodcomp.y2'"), 0)
        self.assertEqual(stream.getvalue().count("'C2.y' wrt 'goodcomp.y1'"), 0)
        self.assertEqual(stream.getvalue().count("'C2.y' wrt 'goodcomp.y2'"), 0)

    def test_compact_print_exceed_tol_show_only_incorrect(self):

        prob = om.Problem()
        top = prob.model
        top.add_subsystem('goodcomp', MyCompGoodPartials())
        top.add_subsystem('badcomp', MyCompBadPartials())
        top.add_subsystem('C1', om.ExecComp('y=2.*x'))
        top.add_subsystem('C2', om.ExecComp('y=3.*x'))

        top.connect('goodcomp.y', 'C1.x')
        top.connect('badcomp.z', 'C2.x')

        top.add_objective('C1.y')
        top.add_constraint('C2.y', lower=0)
        top.add_design_var('goodcomp.x1')
        top.add_design_var('goodcomp.x2')
        top.add_design_var('badcomp.y1')
        top.add_design_var('badcomp.y2')

        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()
        prob.compute_totals()

        stream = StringIO()
        prob.check_totals(out_stream=stream, show_only_incorrect=True, compact_print=True)

        self.assertEqual(stream.getvalue().count('>ABS_TOL'), 2)
        self.assertEqual(stream.getvalue().count('>REL_TOL'), 2)

    def test_directional_vectorized_matrix_free_fwd(self):

        prob = om.Problem()
        prob.model.add_subsystem('comp', DirectionalVectorizedMatFreeComp(n=5))
        prob.model.add_design_var('comp.in')
        prob.model.add_objective('comp.out', index=0)

        prob.setup(force_alloc_complex=True, mode='fwd')
        prob.run_model()

        stream = StringIO()
        data = prob.check_totals(method='cs', out_stream=stream, directional=True)
        content = stream.getvalue()

        self.assertEqual(content.count('Reverse Magnitude:'), 0)
        self.assertEqual(content.count('Forward Magnitude:'), 1)
        self.assertEqual(content.count('Fd Magnitude:'), 1)
        self.assertEqual(content.count('Directional Derivative (Jfor)'), 1)
        self.assertEqual(content.count('Directional CS Derivative (Jfd)'), 1)
        self.assertTrue('Relative Error (Jfor - Jfd) / Jfd : ' in content)
        self.assertTrue('Absolute Error (Jfor - Jfd) : ' in content)
        assert_near_equal(data[(('comp.out',), 'comp.in')]['directional_fd_fwd'], 0., tolerance=2e-15)

    def test_directional_vectorized_matrix_free_rev_index_0(self):

        prob = om.Problem()
        prob.model.add_subsystem('comp', DirectionalVectorizedMatFreeComp(n=5))
        prob.model.add_design_var('comp.in')
        prob.model.add_objective('comp.out', index=0)

        prob.setup(force_alloc_complex=True, mode='rev')
        prob.run_model()

        stream = StringIO()
        data = prob.check_totals(method='cs', out_stream=stream, directional=True)
        content = stream.getvalue()

        self.assertEqual(content.count('comp.out (index size: 1)'), 1)
        self.assertEqual(content.count('Reverse Magnitude:'), 1)
        self.assertEqual(content.count('Forward Magnitude:'), 0)
        self.assertEqual(content.count('Fd Magnitude:'), 1)
        self.assertEqual(content.count('Directional Derivative (Jrev)'), 1)
        self.assertEqual(content.count('Directional CS Derivative (Jfd)'), 1)
        self.assertTrue('Relative Error ([rev, fd] Dot Product Test) / Jfd : ' in content)
        self.assertTrue('Absolute Error ([rev, fd] Dot Product Test) : ' in content)
        assert_near_equal(data[('comp.out', ('comp.in',))]['directional_fd_rev'], 0., tolerance=2e-15)

    def test_directional_vectorized_matrix_free_rev(self):

        prob = om.Problem()
        prob.model.add_subsystem('comp', DirectionalVectorizedMatFreeComp(n=5))
        prob.model.add_design_var('comp.in')
        prob.model.add_constraint('comp.out', lower=0.)

        prob.setup(force_alloc_complex=True, mode='rev')
        prob.run_model()

        stream = StringIO()
        data = prob.check_totals(method='cs', out_stream=stream, directional=True)
        content = stream.getvalue()

        self.assertEqual(content.count("'comp.out' wrt (d)('comp.in',)"), 1)
        self.assertEqual(content.count('Reverse Magnitude:'), 1)
        self.assertEqual(content.count('Forward Magnitude:'), 0)
        self.assertEqual(content.count('Fd Magnitude:'), 1)
        self.assertEqual(content.count('Directional Derivative (Jrev)'), 1)
        self.assertEqual(content.count('Directional CS Derivative (Jfd)'), 1)
        self.assertTrue('Relative Error ([rev, fd] Dot Product Test) / Jfd : ' in content)
        self.assertTrue('Absolute Error ([rev, fd] Dot Product Test) : ' in content)
        assert_near_equal(data[('comp.out', ('comp.in',))]['directional_fd_rev'], 0., tolerance=2e-15)

    def test_directional_vectorized_matrix_free_rev_2in2out(self):

        prob = om.Problem()
        prob.model.add_subsystem('comp', I2O2JacVec(size=5))
        prob.model.add_design_var('comp.in1')
        prob.model.add_design_var('comp.in2')
        prob.model.add_constraint('comp.out1', lower=0.)
        prob.model.add_constraint('comp.out2', lower=0.)

        prob.setup(force_alloc_complex=True, mode='rev')
        prob.run_model()

        stream = StringIO()
        data = prob.check_totals(method='cs', out_stream=stream, directional=True)
        content = stream.getvalue()

        self.assertEqual(content.count("'comp.out1' wrt (d)('comp.in1', 'comp.in2')"), 1)
        self.assertEqual(content.count("'comp.out2' wrt (d)('comp.in1', 'comp.in2')"), 1)
        self.assertEqual(content.count('Reverse Magnitude:'), 2)
        self.assertEqual(content.count('Forward Magnitude:'), 0)
        self.assertEqual(content.count('Fd Magnitude:'), 2)
        self.assertEqual(content.count('Directional Derivative (Jrev)'), 2)
        self.assertEqual(content.count('Directional CS Derivative (Jfd)'), 2)
        self.assertTrue(content.count('Relative Error ([rev, fd] Dot Product Test) / Jfd :'), 2)
        self.assertTrue(content.count('Absolute Error ([rev, fd] Dot Product Test) :'), 2)
        assert_near_equal(data[('comp.out1', ('comp.in1', 'comp.in2'))]['directional_fd_rev'], 0., tolerance=2e-15)
        assert_near_equal(data[('comp.out2', ('comp.in1', 'comp.in2'))]['directional_fd_rev'], 0., tolerance=2e-15)

    def test_directional_vectorized_matrix_free_rev_2in2out_compact(self):

        prob = om.Problem()
        prob.model.add_subsystem('comp', I2O2JacVec(size=5))
        prob.model.add_design_var('comp.in1')
        prob.model.add_design_var('comp.in2')
        prob.model.add_constraint('comp.out1', lower=0.)
        prob.model.add_constraint('comp.out2', lower=0.)

        prob.setup(force_alloc_complex=True, mode='rev')
        prob.run_model()

        stream = StringIO()
        data = prob.check_totals(method='cs', out_stream=stream, compact_print=True, directional=True)
        content = stream.getvalue().strip()
        self.assertEqual(content.count("('comp.in1', 'comp.in2')"), 2)
        assert_near_equal(data[('comp.out1', ('comp.in1', 'comp.in2'))]['directional_fd_rev'], 0., tolerance=2e-15)
        assert_near_equal(data[('comp.out2', ('comp.in1', 'comp.in2'))]['directional_fd_rev'], 0., tolerance=2e-15)

    def test_directional_vectorized_matrix_free_fwd_2in2out(self):

        prob = om.Problem()
        prob.model.add_subsystem('comp', I2O2JacVec(size=5))
        prob.model.add_design_var('comp.in1')
        prob.model.add_design_var('comp.in2')
        prob.model.add_constraint('comp.out1', lower=0.)
        prob.model.add_constraint('comp.out2', lower=0.)

        prob.setup(force_alloc_complex=True, mode='fwd')
        prob.run_model()

        stream = StringIO()
        data = prob.check_totals(method='cs', out_stream=stream, directional=True)
        content = stream.getvalue()

        self.assertEqual(content.count("('comp.out1', 'comp.out2') wrt (d)'comp.in1'"), 1)
        self.assertEqual(content.count("('comp.out1', 'comp.out2') wrt (d)'comp.in2'"), 1)
        self.assertEqual(content.count('Reverse Magnitude:'), 0)
        self.assertEqual(content.count('Forward Magnitude:'), 2)
        self.assertEqual(content.count('Fd Magnitude:'), 2)
        self.assertEqual(content.count('Directional Derivative (Jrev)'), 0)
        self.assertEqual(content.count('Directional Derivative (Jfor)'), 2)
        self.assertEqual(content.count('Directional CS Derivative (Jfd)'), 2)
        self.assertEqual(content.count('Relative Error ([rev, fd] Dot Product Test) / Jfd :'), 0)
        self.assertEqual(content.count('Absolute Error ([rev, fd] Dot Product Test) :'), 0)
        assert_near_equal(np.linalg.norm(data[(('comp.out1', 'comp.out2'), 'comp.in1')]['directional_fd_fwd']), 0., tolerance=2e-15)
        assert_near_equal(np.linalg.norm(data[(('comp.out1', 'comp.out2'), 'comp.in2')]['directional_fd_fwd']), 0., tolerance=2e-15)

    def test_directional_vectorized_matrix_free_fwd_2in2out_compact(self):

        prob = om.Problem()
        prob.model.add_subsystem('comp', I2O2JacVec(size=5))
        prob.model.add_design_var('comp.in1')
        prob.model.add_design_var('comp.in2')
        prob.model.add_constraint('comp.out1', lower=0.)
        prob.model.add_constraint('comp.out2', lower=0.)

        prob.setup(force_alloc_complex=True, mode='fwd')
        prob.run_model()

        stream = StringIO()
        data = prob.check_totals(method='cs', out_stream=stream, compact_print=True, directional=True)
        content = stream.getvalue().strip()
        self.assertEqual(content.count("('comp.out1', 'comp.out2')"), 2)
        assert_near_equal(np.linalg.norm(data[(('comp.out1', 'comp.out2'), 'comp.in1')]['directional_fd_fwd']), 0., tolerance=2e-15)
        assert_near_equal(np.linalg.norm(data[(('comp.out1', 'comp.out2'), 'comp.in2')]['directional_fd_fwd']), 0., tolerance=2e-15)

    def test_directional_dymosish(self):
        class CollocationComp(om.ExplicitComponent):

            def initialize(self):
                self.options.declare(
                    'state_options', types=dict,
                    desc='Dictionary of state names/options for the phase')

            def configure_io(self):
                num_col_nodes = 4
                state_options = self.options['state_options']

                self.var_names = var_names = {}
                for state_name in state_options:
                    var_names[state_name] = {
                        'f_approx': f'f_approx:{state_name}',
                        'defect': f'defects:{state_name}',
                    }

                for state_name, options in state_options.items():
                    shape = options['shape']
                    var_names = self.var_names[state_name]

                    self.add_input(
                        name=var_names['f_approx'],
                        shape=(num_col_nodes,) + shape,
                        desc=f'Estimated derivative of state {state_name} at the collocation nodes')

                    self.add_output(
                        name=var_names['defect'],
                        shape=(num_col_nodes,) + shape,
                        desc=f'Interior defects of state {state_name}')

                    self.add_constraint(name=var_names['defect'],
                                        equals=0.0)

                # Setup partials
                num_col_nodes = 4
                state_options = self.options['state_options']

                for state_name, options in state_options.items():
                    shape = options['shape']
                    size = np.prod(shape)

                    r = np.arange(num_col_nodes * size)

                    var_names = self.var_names[state_name]

                    self.declare_partials(of=var_names['defect'],
                                        wrt=var_names['f_approx'],
                                        rows=r, cols=r)

            def compute(self, inputs, outputs):
                state_options = self.options['state_options']

                for state_name in state_options:
                    var_names = self.var_names[state_name]

                    f_approx = inputs[var_names['f_approx']]

                    outputs[var_names['defect']] = f_approx

            def compute_partials(self, inputs, partials):
                for state_name, options in self.options['state_options'].items():
                    size = np.prod(options['shape'])
                    var_names = self.var_names[state_name]

                    k = np.repeat(1.0, size)

                    partials[var_names['defect'], var_names['f_approx']] = k

        class StateInterpComp(om.ExplicitComponent):

            def initialize(self):
                self.options.declare(
                    'state_options', types=dict,
                    desc='Dictionary of state names/options for the phase')

            def configure_io(self):
                num_disc_nodes = 8
                num_col_nodes = 4

                state_options = self.options['state_options']

                self.xd_str = {}
                self.xdotc_str = {}

                for state_name, options in state_options.items():
                    shape = options['shape']

                    self.add_input(
                        name=f'state_disc:{state_name}',
                        shape=(num_disc_nodes,) + shape,
                        desc=f'Values of state {state_name} at discretization nodes')

                    self.add_output(
                        name=f'staterate_col:{state_name}',
                        shape=(num_col_nodes,) + shape,
                        desc=f'Interpolated rate of state {state_name} at collocation nodes')

                    self.xd_str[state_name] = f'state_disc:{state_name}'
                    self.xdotc_str[state_name] = f'staterate_col:{state_name}'

                Ad = self.Ad = np.zeros((num_col_nodes, num_disc_nodes))
                Ad[0, 0] = -0.75
                Ad[0, 1] = -0.75
                Ad[1, 2] = -0.75
                Ad[1, 3] = -0.75
                Ad[2, 4] = -0.75
                Ad[2, 5] = -0.75
                Ad[3, 6] = -0.75
                Ad[3, 7] = -0.75

                for name, options in state_options.items():
                    self.declare_partials(of=self.xdotc_str[name], wrt=self.xd_str[name],
                                        val=Ad)

            def compute(self, inputs, outputs):
                state_options = self.options['state_options']
                Ad = self.Ad

                for name in state_options:
                    xdotc_str = self.xdotc_str[name]
                    xd_str = self.xd_str[name]

                    outputs[xdotc_str] = np.dot(Ad, inputs[xd_str])

        class TimeComp(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('num_nodes', types=int,
                                    desc='The total number of points at which times are required in the'
                                        'phase.')

            def configure_io(self):
                num_nodes = self.options['num_nodes']
                self.add_output('t_phase', val=np.ones(num_nodes))

        class PseudospectralBase:

            def setup_time(self, phase):

                time_comp = TimeComp(num_nodes=2)

                phase.add_subsystem('time', time_comp, promotes_inputs=['*'], promotes_outputs=['*'])

            def configure_time(self, phase):
                phase.time.configure_io()

            def setup_states(self, phase):
                indep = om.IndepVarComp()

                phase.add_subsystem('indep_states', indep,
                                    promotes_outputs=['*'])

            def configure_states(self, phase):
                num_state_input_nodes = 8
                indep = phase.indep_states

                for name, options in phase.state_options.items():
                    shape = options['shape']
                    default_val = np.zeros((num_state_input_nodes, 1))
                    indep.add_output(name=f'states:{name}',
                                    shape=(num_state_input_nodes,) + shape,
                                    val=default_val)

                    phase.add_design_var(name=f'states:{name}',
                                        indices=np.arange(num_state_input_nodes),
                                        flat_indices=True)

            def setup_ode(self, phase):
                phase.add_subsystem('state_interp',
                                    subsys=StateInterpComp(state_options=phase.state_options))

            def configure_ode(self, phase):
                map_input_indices_to_disc = np.arange(8)

                phase.state_interp.configure_io()

                for name in phase.state_options:
                    phase.connect(f'states:{name}',
                                f'state_interp.state_disc:{name}',
                                src_indices=om.slicer[map_input_indices_to_disc, ...])

            def setup_defects(self, phase):
                phase.add_subsystem('collocation_constraint',
                                    CollocationComp(state_options=phase.state_options))

            def configure_defects(self, phase):
                phase.collocation_constraint.configure_io()

                for name in phase.state_options:
                    phase.connect(f'state_interp.staterate_col:{name}',
                                f'collocation_constraint.f_approx:{name}')

            def configure_objective(self, phase):
                for name, options in phase._objectives.items():
                    index = options['index']

                    shape = (1, )
                    obj_path = 't_phase'

                    size = int(np.prod(shape))

                    idx = 0 if index is None else index
                    obj_index = -size + idx

                    super(Phase, phase).add_objective(obj_path, index=obj_index, flat_indices=True)

        class Phase(om.Group):

            def __init__(self, **kwargs):
                _kwargs = kwargs.copy()

                self.state_options = state_options = {}
                self._objectives = {}

                state_options['y'] = {}
                state_options['y']['targets'] = []
                state_options['y']['shape'] = (1, )
                state_options['y']['rate_source'] = 'ydot'
                state_options['y']['val'] = 0.0

                state_options['v'] = {}
                state_options['v']['targets'] = ['v']
                state_options['v']['shape'] = (1, )
                state_options['v']['rate_source'] = 'vdot'
                state_options['v']['val'] = 0.0

                self.time_options = {}
                self.time_options['initial_val'] = 0.0
                self.time_options['duration_val'] = 1.0
                self.time_options['input_initial'] = False
                self.time_options['input_duration'] = False

                super(Phase, self).__init__(**_kwargs)

            def initialize(self):
                self.options.declare('transcription')

            def add_objective(self, name, loc='final', index=None, shape=(1, )):

                obj_dict = {'name': name,
                            'loc': loc,
                            'index': index,
                            'shape': shape}
                self._objectives[name] = obj_dict

            def setup(self):
                transcription = self.options['transcription']
                transcription.setup_time(self)

                transcription.setup_states(self)
                transcription.setup_ode(self)

                transcription.setup_defects(self)

            def configure(self):
                transcription = self.options['transcription']

                transcription.configure_time(self)
                transcription.configure_states(self)

                transcription.configure_ode(self)

                transcription.configure_defects(self)
                transcription.configure_objective(self)

        p = om.Problem()

        phase = p.model.add_subsystem('Z', Phase(transcription=PseudospectralBase()))
        phase.add_objective('time_phase', loc='final')

        p.setup(force_alloc_complex=True, mode='rev')

        p.run_model()

        data = p.check_totals(method='cs', directional=True)
        assert_check_totals(data, atol=1e-6, rtol=1e-6)

    def _build_sparse_model(self, driver, coloring=False, size=5):
        prob = om.Problem()
        prob.driver = driver

        prob.model.add_subsystem('comp1', Simple(size=size))
        prob.model.add_subsystem('comp2', Simple(size=size))
        prob.model.add_subsystem('comp3', Simple(size=size))
        prob.model.add_subsystem('comp4', Simple(size=size))

        prob.model.add_subsystem('comp', SparseJacVec(size=size))

        prob.model.add_subsystem('comp5', Simple(size=size))
        prob.model.add_subsystem('comp6', Simple(size=size))
        prob.model.add_subsystem('comp7', Simple(size=size))
        prob.model.add_subsystem('comp8', Simple(size=size))

        prob.model.connect('comp1.y', 'comp.in1')
        prob.model.connect('comp2.y', 'comp.in2')
        prob.model.connect('comp3.y', 'comp.in3')
        prob.model.connect('comp4.y', 'comp.in4')

        prob.model.connect('comp.out1', 'comp5.x')
        prob.model.connect('comp.out2', 'comp6.x')
        prob.model.connect('comp.out3', 'comp7.x')
        prob.model.connect('comp.out4', 'comp8.x')

        prob.model.add_design_var('comp1.x', lower=-.5)
        prob.model.add_design_var('comp2.x', lower=0)
        prob.model.add_design_var('comp3.x', lower=-1.)
        prob.model.add_design_var('comp4.x', lower=-5.)

        prob.model.add_constraint('comp5.y', lower=-999., upper=999.)
        prob.model.add_constraint('comp6.y', lower=-999., upper=999.)
        prob.model.add_constraint('comp7.y', lower=-999., upper=999.)
        prob.model.add_objective('comp8.y', index=0)

        if coloring:
            prob.driver.declare_coloring()

        return prob

    def test_sparse_matfree_fwd(self):
        prob = self._build_sparse_model(driver=om.ScipyOptimizeDriver())
        m = prob.model
        prob.setup(force_alloc_complex=True, mode='fwd')
        prob.run_model()

        assert_check_totals(prob.check_totals(method='cs', out_stream=None))

        nsolves = [c.nsolve_linear for c in [m.comp5, m.comp6, m.comp7, m.comp8]]
        # each DV is size 5. each output depends on 2 inputs, so 10 linear solves each.
        # A 'dense' matfree comp would have resulted in 20 (5 x 4) linear solves each.
        expected = [10, 10, 10, 10]

        for slv, ex in zip(nsolves, expected):
            self.assertEqual(slv, ex)

    def test_sparse_matfree_fwd_coloring_scipy(self):
        prob = self._build_sparse_model(driver=om.ScipyOptimizeDriver(), coloring=True)
        m = prob.model
        prob.setup(force_alloc_complex=True, mode='fwd')
        prob.run_model()

        assert_check_totals(prob.check_totals(method='cs', out_stream=None))

        prob.run_driver()

        # reset lin solve counts
        for c in  [m.comp5, m.comp6, m.comp7, m.comp8]:
            c.nsolve_linear = 0

        J = prob.compute_totals()

        nsolves = [c.nsolve_linear for c in [m.comp5, m.comp6, m.comp7, m.comp8]]
        # Coloring requires 2 linear solves, mixing all dependencies, so each comp gets
        # 2 linear solves.
        expected = [2, 2, 2, 2]

        for slv, ex in zip(nsolves, expected):
            self.assertEqual(slv, ex)

    @unittest.skipIf(pyoptsparse_opt is None, "pyOptSparseDriver is required.")
    def test_sparse_matfree_fwd_coloring_pyoptsparse(self):
        prob = self._build_sparse_model(coloring=True, driver=om.pyOptSparseDriver())
        m = prob.model
        prob.setup(force_alloc_complex=True, mode='fwd')
        prob.run_model()

        assert_check_totals(prob.check_totals(method='cs', out_stream=None))

        prob.run_driver()

        # reset lin solve counts
        for c in  [m.comp5, m.comp6, m.comp7, m.comp8]:
            c.nsolve_linear = 0

        J = prob.compute_totals()

        nsolves = [c.nsolve_linear for c in [m.comp5, m.comp6, m.comp7, m.comp8]]
        # Coloring requires 2 linear solves, mixing all dependencies, so each comp gets
        # 2 linear solves.
        expected = [2, 2, 2, 2]

        for slv, ex in zip(nsolves, expected):
            self.assertEqual(slv, ex)

    def test_sparse_matfree_rev(self):
        prob = self._build_sparse_model(driver=om.ScipyOptimizeDriver())
        m = prob.model
        prob.setup(force_alloc_complex=True, mode='rev')
        prob.run_model()

        assert_check_totals(prob.check_totals(method='cs', out_stream=None))

        nsolves = [c.nsolve_linear for c in [m.comp1, m.comp2, m.comp3, m.comp4]]
        # 3 constraints of size 5 plus size 1 objective.  First 2 DVs depend on first 2
        # constraints, so 10 linear solves.  Last 2 DVs depend on 3rd constraint (size 5)
        # plus objective, so 6 linear solves.
        # A 'dense' matfree comp would have resulted in 16 ((5 x 3) + 1) linear solves each.
        expected = [10, 10, 6, 6]

        for slv, ex in zip(nsolves, expected):
            self.assertEqual(slv, ex)

    def test_sparse_matfree_rev_coloring_scipy(self):
        prob = self._build_sparse_model(driver=om.ScipyOptimizeDriver(), coloring=True)
        m = prob.model
        prob.setup(force_alloc_complex=True, mode='rev')
        prob.run_model()

        prob.run_driver()  # activates coloring
        assert_check_totals(prob.check_totals(method='cs', out_stream=None))

        # reset lin solve counts
        for c in  [m.comp1, m.comp2, m.comp3, m.comp4]:
            c.nsolve_linear = 0

        J = prob.compute_totals()

        nsolves = [c.nsolve_linear for c in [m.comp1, m.comp2, m.comp3, m.comp4]]
        # coloring requires 2 rev solves, which combine all dependencies, so each
        # comp gets 2 linear solves
        expected = [2, 2, 2, 2]

        for slv, ex in zip(nsolves, expected):
            self.assertEqual(slv, ex)

    @unittest.skipIf(pyoptsparse_opt is None, "pyOptSparseDriver is required.")
    def test_sparse_matfree_rev_coloring_pyoptsparse(self):
        prob = self._build_sparse_model(driver=om.pyOptSparseDriver(), coloring=True)
        m = prob.model
        prob.setup(force_alloc_complex=True, mode='rev')
        prob.run_model()

        prob.run_driver()  # activates coloring
        assert_check_totals(prob.check_totals(method='cs', out_stream=None))

        # reset lin solve counts
        for c in  [m.comp1, m.comp2, m.comp3, m.comp4]:
            c.nsolve_linear = 0

        J = prob.compute_totals()

        nsolves = [c.nsolve_linear for c in [m.comp1, m.comp2, m.comp3, m.comp4]]
        # coloring requires 2 rev solves, which combine all dependencies, so each
        # comp gets 2 linear solves
        expected = [2, 2, 2, 2]

        for slv, ex in zip(nsolves, expected):
            self.assertEqual(slv, ex)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestProblemCheckTotalsMPI(unittest.TestCase):

    N_PROCS = 2

    def test_indepvarcomp_under_par_sys(self):

        prob = om.Problem()
        prob.model = FanInSubbedIDVC()

        prob.setup(check=False, mode='rev')
        prob.set_solver_print(level=0)
        prob.run_model()

        J = prob.check_totals(out_stream=None)
        assert_near_equal(J['sum.y', 'sub.sub1.p1.x']['J_rev'], [[2.0]], 1.0e-6)
        assert_near_equal(J['sum.y', 'sub.sub2.p2.x']['J_rev'], [[4.0]], 1.0e-6)
        assert_near_equal(J['sum.y', 'sub.sub1.p1.x']['J_fd'], [[2.0]], 1.0e-6)
        assert_near_equal(J['sum.y', 'sub.sub2.p2.x']['J_fd'], [[4.0]], 1.0e-6)


class TestCheckTotalsMultipleSteps(unittest.TestCase):
    def test_single_fd_step_fwd(self):
        p = om.Problem(model=CircleOpt(), driver=om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False))
        p.setup(mode='fwd')
        p.run_model()
        stream = StringIO()
        J = p.check_totals(step=[1e-6], out_stream=stream)
        contents = stream.getvalue()
        nsubjacs = 18
        self.assertEqual(contents.count("Full Model:"), nsubjacs)
        self.assertEqual(contents.count("Fd Magnitude:"), nsubjacs)
        self.assertEqual(contents.count("Absolute Error (Jfor - Jfd), step="), 0)
        self.assertEqual(contents.count("Absolute Error (Jfor - Jfd)"), nsubjacs)
        self.assertEqual(contents.count("Relative Error (Jfor - Jfd) / Jf"), nsubjacs)
        self.assertEqual(contents.count("Raw FD Derivative (Jfd), step="), 0)
        self.assertEqual(contents.count("Raw FD Derivative (Jfd)"), nsubjacs)

    def test_single_fd_step_rev(self):
        p = om.Problem(model=CircleOpt(), driver=om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False))
        p.setup(mode='rev')
        p.run_model()
        stream = StringIO()
        J = p.check_totals(step=[1e-6], out_stream=stream)
        contents = stream.getvalue()
        nsubjacs = 18
        self.assertEqual(contents.count("Full Model:"), nsubjacs)
        self.assertEqual(contents.count("Fd Magnitude:"), nsubjacs)
        self.assertEqual(contents.count("Absolute Error (Jrev - Jfd), step="), 0)
        self.assertEqual(contents.count("Absolute Error (Jrev - Jfd)"), nsubjacs)
        self.assertEqual(contents.count("Relative Error (Jrev - Jfd) / J"), nsubjacs)
        self.assertEqual(contents.count("Raw FD Derivative (Jfd), step="), 0)
        self.assertEqual(contents.count("Raw FD Derivative (Jfd)"), nsubjacs)

    def test_single_fd_step_compact(self):
        for mode in ('fwd', 'rev'):
            with self.subTest(f"{mode} derivatives"):
                p = om.Problem(model=CircleOpt(), driver=om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False))
                p.setup(mode=mode)
                p.run_model()
                stream = StringIO()
                J = p.check_totals(step=[1e-6], compact_print=True, out_stream=stream)
                contents = stream.getvalue()
                nsubjacs = 18
                self.assertEqual(contents.count("step"), 0)
                # check number of rows/cols
                self.assertEqual(contents.count("+-------------------------------+------------------+-------------+-------------+-------------+-------------+--------------------+"), nsubjacs + 1)

    def test_single_cs_step_compact(self):
        for mode in ('fwd', 'rev'):
            with self.subTest(f"{mode} derivatives"):
                p = om.Problem(model=CircleOpt(), driver=om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False))
                p.setup(mode=mode, force_alloc_complex=True)
                p.run_model()
                stream = StringIO()
                J = p.check_totals(method='cs', step=1e-30, compact_print=True, out_stream=stream)
                contents = stream.getvalue()
                nsubjacs = 18
                self.assertEqual(contents.count("step"), 0)
                # check number of rows/cols
                self.assertEqual(contents.count("+-------------------------------+------------------+-------------+-------------+-------------+-------------+------------+"), nsubjacs + 1)

    def test_multi_fd_steps_fwd(self):
        p = om.Problem(model=CircleOpt(), driver=om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False))
        p.setup(mode='fwd')
        p.run_model()
        stream = StringIO()
        J = p.check_totals(step=[1e-6, 1e-7], out_stream=stream)
        contents = stream.getvalue()
        nsubjacs = 18
        self.assertEqual(contents.count("Full Model:"), nsubjacs)
        self.assertEqual(contents.count("Fd Magnitude:"), nsubjacs * 2)
        self.assertEqual(contents.count("Absolute Error (Jfor - Jfd), step="), nsubjacs * 2)
        self.assertEqual(contents.count("Relative Error (Jfor - Jfd) / Jf"), nsubjacs * 2)
        self.assertEqual(contents.count("Raw FD Derivative (Jfd), step="), nsubjacs * 2)

    def test_multi_fd_steps_fwd_directional(self):
        p = om.Problem(model=CircleOpt(), driver=om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False))
        p.setup(mode='fwd')
        p.run_model()
        stream = StringIO()
        J = p.check_totals(step=[1e-6, 1e-7], directional=True, out_stream=stream)
        contents = stream.getvalue()
        self.assertEqual(contents.count("Full Model:"), 3)
        self.assertEqual(contents.count("Fd Magnitude:"), 6)
        self.assertEqual(contents.count("Absolute Error (Jfor - Jfd), step="), 6)
        self.assertEqual(contents.count("Relative Error (Jfor - Jfd) / Jf"), 6)
        self.assertEqual(contents.count("Directional FD Derivative (Jfd), step="), 6)

    def test_multi_fd_steps_rev(self):
        p = om.Problem(model=CircleOpt(), driver=om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False))
        p.setup(mode='rev')
        p.run_model()
        stream = StringIO()
        J = p.check_totals(step=[1e-6, 1e-7], out_stream=stream)
        contents = stream.getvalue()
        nsubjacs = 18
        self.assertEqual(contents.count("Full Model:"), nsubjacs)
        self.assertEqual(contents.count("Fd Magnitude:"), nsubjacs * 2)
        self.assertEqual(contents.count("Absolute Error (Jrev - Jfd), step="), nsubjacs * 2)
        self.assertEqual(contents.count("Relative Error (Jrev - Jfd) / J"), nsubjacs * 2)
        self.assertEqual(contents.count("Raw FD Derivative (Jfd), step="), nsubjacs * 2)

    def test_multi_fd_steps_rev_directional(self):
        p = om.Problem(model=CircleOpt(), driver=om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False))
        p.setup(mode='rev')
        p.run_model()
        stream = StringIO()
        J = p.check_totals(step=[1e-6, 1e-7], directional=True, out_stream=stream)
        contents = stream.getvalue()
        self.assertEqual(contents.count("Full Model:"), 6)
        self.assertEqual(contents.count("Fd Magnitude:"), 12)
        self.assertEqual(contents.count("Absolute Error ([rev, fd] Dot Product Test), step="), 12)
        self.assertEqual(contents.count("Relative Error ([rev, fd] Dot Product Test) / Jfd, step="), 12)
        self.assertEqual(contents.count("Directional FD Derivative (Jfd) Dot Product, step="), 12)

    def test_multi_fd_steps_compact(self):
        for mode in ('fwd', 'rev'):
            with self.subTest(f"{mode} derivatives"):
                p = om.Problem(model=CircleOpt(), driver=om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False))
                p.setup(mode=mode)
                p.run_model()
                stream = StringIO()
                J = p.check_totals(step=[1e-6, 1e-7], compact_print=True, out_stream=stream)
                contents = stream.getvalue()
                nsubjacs = 18
                self.assertEqual(contents.count("step"), 1)
                # check number of rows/cols
                self.assertEqual(contents.count("+-------------------------------+------------------+-------------+-------------+-------------+-------------+-------------+--------------------+"), (nsubjacs*2) + 1)

    def test_multi_cs_steps_compact(self):
        for mode in ('fwd', 'rev'):
            with self.subTest(f"{mode} derivatives"):
                p = om.Problem(model=CircleOpt(), driver=om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False))
                p.setup(mode=mode, force_alloc_complex=True)
                p.run_model()
                stream = StringIO()
                J = p.check_totals(method='cs', step=[1e-20, 1e-30], compact_print=True, out_stream=stream)
                contents = stream.getvalue()
                nsubjacs = 18
                self.assertEqual(contents.count("step"), 1)
                # check number of rows/cols
                self.assertEqual(contents.count("+-------------------------------+------------------+-------------+-------------+-------------+-------------+-------------+------------+"), (nsubjacs*2) + 1)

    def test_multi_fd_steps_compact_directional(self):
        expected_divs = {
            'fwd': 7,
            'rev': 13,
        }
        try:
            rand_save = tot_jac_mod._directional_rng
            for mode in ('fwd', 'rev'):
                with self.subTest(f"{mode} derivatives"):
                    tot_jac_mod._directional_rng = np.random.default_rng(99)  # keep random seeds the same for directional check
                    p = om.Problem(model=CircleOpt(), driver=om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False))
                    p.setup(mode=mode)
                    p.run_model()
                    stream = StringIO()
                    J = p.check_totals(step=[1e-6, 1e-7], compact_print=True, directional=True, out_stream=stream)
                    contents = stream.getvalue()
                    self.assertEqual(contents.count("step"), 1)
                    # check number of rows/cols
                    nrows = expected_divs[mode]
                    self.assertEqual(contents.count('\n+-'), nrows)
        finally:
            tot_jac_mod._directional_rng = rand_save



if __name__ == "__main__":
    unittest.main()
