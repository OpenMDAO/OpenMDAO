""" Test out some specialized parallel derivatives features"""


from io import StringIO
import sys
import unittest
import time
from packaging.version import Version

import numpy as np

import openmdao.api as om
from openmdao.test_suite.groups.parallel_groups import FanOutGrouped, FanInGrouped, FanInGrouped2
from openmdao.utils.assert_utils import assert_near_equal, assert_check_totals
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.mpi import MPI


if MPI:
    try:
        from openmdao.vectors.petsc_vector import PETScVector
    except ImportError:
        PETScVector = None


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ParDerivTestCase(unittest.TestCase):

    N_PROCS = 2

    def test_fan_in_serial_sets_rev(self):

        prob = om.Problem()
        prob.model = FanInGrouped()
        prob.model.linear_solver = om.LinearBlockGS()
        prob.model.sub.linear_solver = om.LinearBlockGS()

        prob.model.add_design_var('x1')
        prob.model.add_design_var('x2')
        prob.model.add_objective('c3.y')

        prob.setup(check=False, mode='rev')
        prob.run_model()

        data = prob.check_totals(of=['c3.y'], wrt=['x1', 'x2'])
        assert_check_totals(data)

    def test_fan_in_serial_sets_rev_ivc(self):

        prob = om.Problem()
        prob.model = FanInGrouped2()

        prob.model.add_design_var('p1.x')
        prob.model.add_design_var('p2.x')
        prob.model.add_objective('c3.y')

        prob.setup(check=False, mode='rev')
        prob.run_model()

        data = prob.check_totals(of=['c3.y'], wrt=['p1.x', 'p2.x'])
        assert_check_totals(data)

    def test_fan_in_serial_sets_fwd(self):

        prob = om.Problem()
        prob.model = FanInGrouped()
        prob.model.linear_solver = om.LinearBlockGS()
        prob.model.sub.linear_solver = om.LinearBlockGS()

        prob.model.add_design_var('x1')
        prob.model.add_design_var('x2')
        prob.model.add_objective('c3.y')

        prob.setup(check=False, mode='fwd')
        prob.run_driver()

        indep_list = ['x1', 'x2']
        unknown_list = ['c3.y']

        J = prob.compute_totals(unknown_list, indep_list, return_format='flat_dict')
        assert_near_equal(J['c3.y', 'x1'][0][0], -6.0, 1e-6)
        assert_near_equal(J['c3.y', 'x2'][0][0], 35.0, 1e-6)

    def test_fan_out_serial_sets_fwd(self):

        prob = om.Problem()
        prob.model = FanOutGrouped()
        prob.model.linear_solver = om.LinearBlockGS()
        prob.model.sub.linear_solver = om.LinearBlockGS()

        prob.model.add_design_var('iv.x')
        prob.model.add_constraint('c2.y', upper=0.0)
        prob.model.add_constraint('c3.y', upper=0.0)

        prob.setup(check=False, mode='fwd')
        prob.run_driver()

        unknown_list = ['c2.y', 'c3.y']
        indep_list = ['iv.x']

        J = prob.compute_totals(unknown_list, indep_list, return_format='flat_dict')
        assert_near_equal(J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_near_equal(J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

    def test_fan_out_serial_sets_rev(self):

        prob = om.Problem()
        prob.model = FanOutGrouped()
        prob.model.linear_solver = om.LinearBlockGS()
        prob.model.sub.linear_solver = om.LinearBlockGS()

        prob.model.add_design_var('iv.x')
        prob.model.add_constraint('c2.y', upper=0.0)
        prob.model.add_constraint('c3.y', upper=0.0)

        prob.setup(check=False, mode='rev')
        prob.run_driver()

        unknown_list = ['c3.y','c2.y'] #['c2.y', 'c3.y']
        indep_list = ['iv.x']

        J = prob.compute_totals(unknown_list, indep_list, return_format='flat_dict')
        assert_near_equal(J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_near_equal(J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

    def test_debug_print_option_totals_color(self):

        prob = om.Problem()
        prob.model = FanInGrouped()

        # An extra unconnected desvar was in the original test.
        prob.model.add_subsystem('p', om.IndepVarComp('x3', 0.0), promotes=['x3'])

        prob.model.linear_solver = om.LinearBlockGS()
        prob.model.sub.linear_solver = om.LinearBlockGS()

        prob.model.add_design_var('x1', parallel_deriv_color='par_dv')
        prob.model.add_design_var('x2', parallel_deriv_color='par_dv')
        prob.model.add_design_var('x3')
        prob.model.add_objective('c3.y')

        prob.driver.options['debug_print'] = ['totals']

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_driver()

        indep_list = ['x1', 'x2', 'x3']
        unknown_list = ['c3.y']

        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            _ = prob.compute_totals(unknown_list, indep_list, return_format='flat_dict',
                                    debug_print=not prob.comm.rank)
        finally:
            sys.stdout = stdout

        output = strout.getvalue()

        if not prob.comm.rank:
            self.assertTrue('Solving color: par_dv (x1, x2)' in output)
            self.assertTrue('In mode: fwd.' in output)
            self.assertTrue("('x3', [2])" in output)

    def test_fan_out_parallel_sets_rev(self):

        prob = om.Problem()
        prob.model = FanOutGrouped()
        prob.model.linear_solver = om.LinearBlockGS()
        prob.model.sub.linear_solver = om.LinearBlockGS()

        prob.model.add_design_var('iv.x')
        prob.model.add_constraint('c2.y', upper=0.0, parallel_deriv_color='par_resp')
        prob.model.add_constraint('c3.y', upper=0.0, parallel_deriv_color='par_resp')

        prob.setup(check=False, mode='rev')
        prob.run_driver()

        unknown_list = ['c2.y', 'c3.y']
        indep_list = ['iv.x']

        J = prob.compute_totals(unknown_list, indep_list, return_format='flat_dict')
        assert_near_equal(J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_near_equal(J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

        # Piggyback to make sure the distributed norm calculation is correct.
        vec = prob.model._vectors['residual']['linear']
        norm_val = vec.get_norm()
        # NOTE: BAN updated the norm value for the PR that removed vec_names vectors.
        # the seeds for the constraints are now back to -1 instead of -.5
        assert_near_equal(norm_val, 6.557438524302, 1e-6)

    def test_ln_nl_complex_alloc_bug(self):
        # This verifies a fix for an MPI hang when allocating the vectors. If one proc
        # needs a complex vector, then they all do.

        class ImpComp(om.ImplicitComponent):

            def setup(self):
                self.add_input('x', 3.0)
                self.add_output('y', 4.0)

                self. declare_partials('y', 'x', method='cs')
                self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
                self.linear_solver = om.DirectSolver()

            def apply_nonlinear(self, inputs, outputs, residuals, discrete_inputs=None,
                                discrete_outputs=None):
                pass

        class Sub1(om.Group):

            def setup(self):
                self.add_subsystem('imp', ImpComp())

        class Sub2(om.Group):

            def setup(self):
                self.add_subsystem('exp', om.ExecComp('y = x'))

        class Par(om.ParallelGroup):

            def setup(self):
                self.add_subsystem('sub1', Sub1())
                self.add_subsystem('sub2', Sub2())

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('par', Par())

        prob.setup()

        # Hangs on this step before bug fix.
        prob.final_setup()


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class DecoupledTestCase(unittest.TestCase):
    N_PROCS = 2
    asize = 3

    def setup_model(self):
        asize = self.asize
        prob = om.Problem()
        root = prob.model
        root.linear_solver = om.LinearBlockGS()

        Indep1 = root.add_subsystem('Indep1', om.IndepVarComp('x', np.arange(asize, dtype=float)+1.0))
        Indep2 = root.add_subsystem('Indep2', om.IndepVarComp('x', np.arange(asize+2, dtype=float)+1.0))
        G1 = root.add_subsystem('G1', om.ParallelGroup())
        G1.linear_solver = om.LinearBlockGS()

        c1 = G1.add_subsystem('c1', om.ExecComp('y = ones(3).T*x.dot(arange(3.,6.))',
                                                x=np.zeros(asize), y=np.zeros(asize)))
        c2 = G1.add_subsystem('c2', om.ExecComp('y = x[:%d] * 2.0' % asize,
                                                x=np.zeros(asize+2), y=np.zeros(asize)))

        Con1 = root.add_subsystem('Con1', om.ExecComp('y = x * 5.0',
                                                      x=np.zeros(asize), y=np.zeros(asize)))
        Con2 = root.add_subsystem('Con2', om.ExecComp('y = x * 4.0',
                                                      x=np.zeros(asize), y=np.zeros(asize)))
        root.connect('Indep1.x', 'G1.c1.x')
        root.connect('Indep2.x', 'G1.c2.x')
        root.connect('G1.c1.y', 'Con1.x')
        root.connect('G1.c2.y', 'Con2.x')

        return prob

    def test_serial_fwd(self):
        asize = self.asize
        prob = self.setup_model()

        prob.model.add_design_var('Indep1.x')
        prob.model.add_design_var('Indep2.x')
        prob.model.add_constraint('Con1.y', upper=0.0)
        prob.model.add_constraint('Con2.y', upper=0.0)

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        J = prob.compute_totals(['Con1.y', 'Con2.y'], ['Indep1.x', 'Indep2.x'],
                                return_format='flat_dict')

        assert_near_equal(J['Con1.y', 'Indep1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.zeros((asize, asize+2))
        expected[:,:asize] = np.eye(asize)*8.0
        assert_near_equal(J['Con2.y', 'Indep2.x'], expected, 1e-6)

    def test_parallel_fwd(self):
        asize = self.asize
        prob = self.setup_model()

        prob.model.add_design_var('Indep1.x', parallel_deriv_color='pardv')
        prob.model.add_design_var('Indep2.x', parallel_deriv_color='pardv')
        prob.model.add_constraint('Con1.y', upper=0.0)
        prob.model.add_constraint('Con2.y', upper=0.0)

        prob.setup(check=False, mode='fwd')
        prob.run_driver()

        J = prob.compute_totals(['Con1.y', 'Con2.y'], ['Indep1.x', 'Indep2.x'],
                                return_format='flat_dict')

        assert_near_equal(J['Con1.y', 'Indep1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.zeros((asize, asize+2))
        expected[:,:asize] = np.eye(asize)*8.0
        assert_near_equal(J['Con2.y', 'Indep2.x'], expected, 1e-6)

    def test_serial_rev(self):
        asize = self.asize
        prob = self.setup_model()

        prob.model.add_design_var('Indep1.x')
        prob.model.add_design_var('Indep2.x')
        prob.model.add_constraint('Con1.y', upper=0.0)
        prob.model.add_constraint('Con2.y', upper=0.0)

        prob.setup(check=False, mode='rev')
        prob.run_driver()

        J = prob.compute_totals(['Con1.y', 'Con2.y'], ['Indep1.x', 'Indep2.x'],
                                return_format='flat_dict')

        assert_near_equal(J['Con1.y', 'Indep1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.zeros((asize, asize+2))
        expected[:,:asize] = np.eye(asize)*8.0
        assert_near_equal(J['Con2.y', 'Indep2.x'], expected, 1e-6)

    def test_parallel_rev(self):
        asize = self.asize

        prob = self.setup_model()

        prob.model.add_design_var('Indep1.x')
        prob.model.add_design_var('Indep2.x')
        prob.model.add_constraint('Con1.y', upper=0.0, parallel_deriv_color='parc')
        prob.model.add_constraint('Con2.y', upper=0.0, parallel_deriv_color='parc')

        prob.setup(check=False, mode='rev')
        prob.run_driver()

        J = prob.compute_totals(['Con1.y', 'Con2.y'], ['Indep1.x', 'Indep2.x'],
                                return_format='flat_dict')

        assert_near_equal(J['Con1.y', 'Indep1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.zeros((asize, asize+2))
        expected[:,:asize] = np.eye(asize)*8.0
        assert_near_equal(J['Con2.y', 'Indep2.x'], expected, 1e-6)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class IndicesTestCase(unittest.TestCase):

    N_PROCS = 2

    def setup_model(self, mode):
        asize = 3
        prob = om.Problem()
        root = prob.model
        root.linear_solver = om.LinearBlockGS()

        p = root.add_subsystem('p', om.IndepVarComp('x', np.arange(asize, dtype=float)+1.0))
        G1 = root.add_subsystem('G1', om.ParallelGroup())
        G1.linear_solver = om.LinearBlockGS()

        c2 = G1.add_subsystem('c2', om.ExecComp('y = x * 2.0',
                                                x=np.zeros(asize), y=np.zeros(asize)))
        c3 = G1.add_subsystem('c3', om.ExecComp('y = ones(3).T*x.dot(arange(3.,6.))',
                                                x=np.zeros(asize), y=np.zeros(asize)))
        c4 = root.add_subsystem('c4', om.ExecComp('y = x * 4.0',
                                                  x=np.zeros(asize), y=np.zeros(asize)))
        c5 = root.add_subsystem('c5', om.ExecComp('y = x * 5.0',
                                                  x=np.zeros(asize), y=np.zeros(asize)))

        prob.model.add_design_var('p.x', indices=[1, 2])
        prob.model.add_constraint('c4.y', upper=0.0, indices=[1], parallel_deriv_color='par_resp')
        prob.model.add_constraint('c5.y', upper=0.0, indices=[2], parallel_deriv_color='par_resp')

        root.connect('p.x', 'G1.c2.x')
        root.connect('p.x', 'G1.c3.x')
        root.connect('G1.c2.y', 'c4.x')
        root.connect('G1.c3.y', 'c5.x')

        prob.setup(check=False, mode=mode)
        prob.run_driver()

        return prob

    def test_indices_fwd(self):
        prob = self.setup_model('fwd')

        J = prob.compute_totals(['c4.y', 'c5.y'], ['p.x'],
                                return_format='flat_dict')

        assert_near_equal(J['c5.y', 'p.x'][0], np.array([20., 25.]), 1e-6)
        assert_near_equal(J['c4.y', 'p.x'][0], np.array([8., 0.]), 1e-6)

    def test_indices_rev(self):
        prob = self.setup_model('rev')
        J = prob.compute_totals(['c4.y', 'c5.y'], ['p.x'],
                                return_format='flat_dict')

        assert_near_equal(J['c5.y', 'p.x'][0], np.array([20., 25.]), 1e-6)
        assert_near_equal(J['c4.y', 'p.x'][0], np.array([8., 0.]), 1e-6)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class IndicesTestCase2(unittest.TestCase):

    N_PROCS = 2

    def setup_model(self, mode):
        asize = 3
        prob = om.Problem()
        root = prob.model

        root.linear_solver = om.LinearBlockGS()

        G1 = root.add_subsystem('G1', om.ParallelGroup())
        G1.linear_solver = om.LinearBlockGS()

        par1 = G1.add_subsystem('par1', om.Group())
        par1.linear_solver = om.LinearBlockGS()
        par2 = G1.add_subsystem('par2', om.Group())
        par2.linear_solver = om.LinearBlockGS()

        p1 = par1.add_subsystem('p', om.IndepVarComp('x', np.arange(asize, dtype=float)+1.0))
        p2 = par2.add_subsystem('p', om.IndepVarComp('x', np.arange(asize, dtype=float)+10.0))

        c2 = par1.add_subsystem('c2', om.ExecComp('y = x * 2.0',
                                                  x=np.zeros(asize), y=np.zeros(asize)))
        c3 = par2.add_subsystem('c3', om.ExecComp('y = ones(3).T*x.dot(arange(3.,6.))',
                                                  x=np.zeros(asize), y=np.zeros(asize)))
        c4 = par1.add_subsystem('c4', om.ExecComp('y = x * 4.0',
                                                  x=np.zeros(asize), y=np.zeros(asize)))
        c5 = par2.add_subsystem('c5', om.ExecComp('y = x * 5.0',
                                                  x=np.zeros(asize), y=np.zeros(asize)))

        prob.model.add_design_var('G1.par1.p.x', indices=[1, 2])
        prob.model.add_design_var('G1.par2.p.x', indices=[1, 2])
        prob.model.add_constraint('G1.par1.c4.y', upper=0.0, indices=[1], parallel_deriv_color='par_resp')
        prob.model.add_constraint('G1.par2.c5.y', upper=0.0, indices=[2], parallel_deriv_color='par_resp')

        root.connect('G1.par1.p.x', 'G1.par1.c2.x')
        root.connect('G1.par2.p.x', 'G1.par2.c3.x')
        root.connect('G1.par1.c2.y', 'G1.par1.c4.x')
        root.connect('G1.par2.c3.y', 'G1.par2.c5.x')

        prob.setup(check=False, mode=mode)
        prob.run_driver()

        return prob

    def test_indices_fwd(self):
        prob = self.setup_model('fwd')

        dvs = prob.model.get_design_vars()
        self.assertEqual(set(dvs), set(['G1.par1.p.x', 'G1.par2.p.x']))

        responses = prob.model.get_responses()
        self.assertEqual(set(responses), set(['G1.par1.c4.y', 'G1.par2.c5.y']))

        J = prob.compute_totals(of=['G1.par1.c4.y', 'G1.par2.c5.y'],
                                wrt=['G1.par1.p.x', 'G1.par2.p.x'],
                                return_format='flat_dict')

        assert_near_equal(J['G1.par2.c5.y', 'G1.par2.p.x'][0], np.array([20., 25.]), 1e-6)
        assert_near_equal(J['G1.par1.c4.y', 'G1.par1.p.x'][0], np.array([8., 0.]), 1e-6)

    def test_indices_rev(self):
        prob = self.setup_model('rev')
        J = prob.compute_totals(['G1.par1.c4.y', 'G1.par2.c5.y'],
                                ['G1.par1.p.x', 'G1.par2.p.x'],
                                return_format='flat_dict')

        assert_near_equal(J['G1.par2.c5.y', 'G1.par2.p.x'][0], np.array([20., 25.]), 1e-6)
        assert_near_equal(J['G1.par1.c4.y', 'G1.par1.p.x'][0], np.array([8., 0.]), 1e-6)

    def test_src_indices_rev(self):
        class DummyComp(om.ExplicitComponent):
            def initialize(self):
                self.options.declare('a',default=0.)
                self.options.declare('b',default=0.)

            def setup(self):
                self.add_input('x')
                self.add_output('y', 0.)

            def compute(self, inputs, outputs):
                outputs['y'] = self.options['a']*inputs['x'] + self.options['b']

            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                if mode=='rev':
                    if 'y' in d_outputs:
                        if 'x' in d_inputs:
                            d_inputs['x'] += self.options['a'] * d_outputs['y']
                            # print(self.pathname, 'compute_jvp: dinputs[x]', d_inputs['x'])
                else:
                    raise RuntimeError("fwd mode not supported")

        class DummyGroup(om.ParallelGroup):
            def setup(self):
                self.add_subsystem('C1',DummyComp(a=1,b=2.))
                self.add_subsystem('C2',DummyComp(a=3.,b=4.))

        class Top(om.Group):
            def setup(self):
                self.add_subsystem('dvs',om.IndepVarComp(), promotes=['*'])

                # this only currently works if we make dvs.x a distributed output.
                if self.comm.rank == 0:
                    self.dvs.add_output('x', [1.], distributed=True)
                else:
                    self.dvs.add_output('x', [2.], distributed=True)

                # making dvs.x a non-distributed variable as below results in
                # one deriv being zero and the other being the sum of the two
                # parallel derivs.
                # self.dvs.add_output('x',[1.,2.])

                self.add_subsystem('par',DummyGroup())
                self.connect('x','par.C1.x',src_indices=[0])
                self.connect('x','par.C2.x',src_indices=[1])

        prob = om.Problem(model=Top())
        prob.model.add_design_var('x',lower=0.,upper=1.)

        # None or string
        deriv_color = 'deriv_color'

        # compute derivatives for made-up y constraints in parallel
        prob.model.add_constraint('par.C1.y',
                                lower=1.0,
                                parallel_deriv_color=deriv_color)
        prob.model.add_constraint('par.C2.y',
                                lower=1.0,
                                parallel_deriv_color=deriv_color)

        prob.setup(mode='rev')
        prob.run_model()
        assert_check_totals(prob.check_totals())


class SumComp(om.ExplicitComponent):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def setup(self):
        self.add_input('x', val=np.zeros(self.size))
        self.add_output('y', val=0.0)

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        outputs['y'] = np.sum(inputs['x'])

    def compute_partials(self, inputs, partials):
        partials['y', 'x'] = np.ones(inputs['x'].size)


class SlowComp(om.ExplicitComponent):
    """
    Component with a delay that multiplies the input by a multiplier.
    """

    def __init__(self, delay=1.0, size=3, mult=2.0):
        super().__init__()
        self.delay = delay
        self.size = size
        self.mult = mult

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_output('y', val=np.zeros(self.size))

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        outputs['y'] = inputs['x'] * self.mult

    def compute_partials(self, inputs, partials):
        partials['y', 'x'] = self.mult

    def _apply_linear(self, jac, mode, scope_out=None, scope_in=None):
        time.sleep(self.delay)
        super()._apply_linear(jac, mode, scope_out, scope_in)


class PartialDependGroup(om.Group):
    def setup(self):
        size = 4

        Comp1 = self.add_subsystem('Comp1', SumComp(size))
        pargroup = self.add_subsystem('ParallelGroup1', om.ParallelGroup())

        self.set_input_defaults('Comp1.x', val=np.arange(size, dtype=float)+1.0)

        self.linear_solver = om.LinearBlockGS()
        self.linear_solver.options['iprint'] = -1
        pargroup.linear_solver = om.LinearBlockGS()
        pargroup.linear_solver.options['iprint'] = -1

        delay = .1
        Con1 = pargroup.add_subsystem('Con1', SlowComp(delay=delay, size=2, mult=2.0))
        Con2 = pargroup.add_subsystem('Con2', SlowComp(delay=delay, size=2, mult=-3.0))

        self.connect('Comp1.y', 'ParallelGroup1.Con1.x')
        self.connect('Comp1.y', 'ParallelGroup1.Con2.x')

        color = 'parcon'
        self.add_design_var('Comp1.x')
        self.add_constraint('ParallelGroup1.Con1.y', lower=0.0, parallel_deriv_color=color)
        self.add_constraint('ParallelGroup1.Con2.y', upper=0.0, parallel_deriv_color=color)


# This one hangs on Travis for numpy 1.12 and we can't reproduce the error anywhere where we can
# debug it, so we're skipping it for numpy 1.12.
@unittest.skipUnless(MPI and PETScVector and Version(np.__version__) >= Version("1.13"),
                     "MPI, PETSc, and numpy >= 1.13 are required.")
class ParDerivColorFeatureTestCase(unittest.TestCase):
    N_PROCS = 2

    def test_feature_rev(self):

        size = 4

        of = ['ParallelGroup1.Con1.y', 'ParallelGroup1.Con2.y']
        wrt = ['Comp1.x']

        p = om.Problem(model=PartialDependGroup())

        p.setup(mode='rev')
        p.run_model()

        J = p.compute_totals(of, wrt, return_format='dict')

        assert_near_equal(J['ParallelGroup1.Con1.y']['Comp1.x'][0], np.ones(size)*2., 1e-6)
        assert_near_equal(J['ParallelGroup1.Con2.y']['Comp1.x'][0], np.ones(size)*-3., 1e-6)

    def test_feature_fwd(self):

        size = 4

        of = ['ParallelGroup1.Con1.y', 'ParallelGroup1.Con2.y']
        wrt = ['Comp1.x']

        p = om.Problem(model=PartialDependGroup())
        p.setup(mode='fwd')
        p.run_model()

        J = p.compute_totals(of, wrt, return_format='dict')

        assert_near_equal(J['ParallelGroup1.Con1.y']['Comp1.x'][0], np.ones(size)*2., 1e-6)
        assert_near_equal(J['ParallelGroup1.Con2.y']['Comp1.x'][0], np.ones(size)*-3., 1e-6)

    def test_fwd_vs_rev(self):

        size = 4

        of = ['ParallelGroup1.Con1.y', 'ParallelGroup1.Con2.y']
        wrt = ['Comp1.x']

        # run in rev mode
        p = om.Problem(model=PartialDependGroup())
        p.setup(mode='rev')

        p.run_model()

        elapsed_rev = time.perf_counter()
        Jrev = p.compute_totals(of, wrt, return_format='dict')
        elapsed_rev = time.perf_counter() - elapsed_rev

        # run in fwd mode and compare times for deriv calculation
        p.setup(mode='fwd')
        p.run_model()

        elapsed_fwd = time.perf_counter()
        Jfwd = p.compute_totals(of, wrt, return_format='dict')
        elapsed_fwd = time.perf_counter() - elapsed_fwd

        assert_near_equal(Jfwd['ParallelGroup1.Con1.y']['Comp1.x'][0], np.ones(size)*2., 1e-6)
        assert_near_equal(Jfwd['ParallelGroup1.Con2.y']['Comp1.x'][0], np.ones(size)*-3., 1e-6)

        assert_near_equal(Jrev['ParallelGroup1.Con1.y']['Comp1.x'][0], np.ones(size)*2., 1e-6)
        assert_near_equal(Jrev['ParallelGroup1.Con2.y']['Comp1.x'][0], np.ones(size)*-3., 1e-6)

        # make sure that rev mode is faster than fwd mode
        self.assertGreater(elapsed_fwd / elapsed_rev, 1.0)


class CleanupTestCase(unittest.TestCase):
    # This is to test for a bug john found that caused his ozone problem to fail
    # to converge.  The problem was due to garbage in the doutputs vector that
    # was coming from transfers to irrelevant variables during Group._apply_linear.
    def setUp(self):
        p = self.p = om.Problem()
        root = p.model
        root.linear_solver = om.LinearBlockGS()
        root.linear_solver.options['err_on_non_converge'] = True

        inputs = root.add_subsystem("inputs", om.IndepVarComp("x", 1.0))
        G1 = root.add_subsystem("G1", om.Group())
        dparam = G1.add_subsystem("dparam", om.ExecComp("y = .5*x"))
        G1_inputs = G1.add_subsystem("inputs", om.IndepVarComp("x", 1.5))
        start = G1.add_subsystem("start", om.ExecComp("y = .7*x"))
        timecomp = G1.add_subsystem("time", om.ExecComp("y = -.2*x"))

        G2 = G1.add_subsystem("G2", om.Group())
        stage_step = G2.add_subsystem("stage_step",
                                      om.ExecComp("y = -0.1*x + .5*x2 - .4*x3 + .9*x4"))
        ode = G2.add_subsystem("ode", om.ExecComp("y = .8*x - .6*x2"))
        dummy = G2.add_subsystem("dummy", om.IndepVarComp("x", 1.3))

        step = G1.add_subsystem("step", om.ExecComp("y = -.2*x + .4*x2 - .4*x3"))
        output = G1.add_subsystem("output", om.ExecComp("y = .6*x"))

        con = root.add_subsystem("con", om.ExecComp("y = .2 * x"))
        obj = root.add_subsystem("obj", om.ExecComp("y = .3 * x"))

        root.connect("inputs.x", "G1.dparam.x")

        G1.connect("inputs.x", ["start.x", "time.x"])
        G1.connect("dparam.y", "G2.ode.x")
        G1.connect("start.y", ["step.x", "G2.stage_step.x4"])
        G1.connect("time.y", ["step.x2", "G2.stage_step.x3"])
        G1.connect("step.y", "output.x")
        G1.connect("G2.ode.y", ["step.x3", "G2.stage_step.x"])

        G2.connect("stage_step.y", "ode.x2")
        G2.connect("dummy.x", "stage_step.x2")

        root.connect("G1.output.y", ["con.x", "obj.x"])

        root.add_design_var('inputs.x')
        root.add_constraint('con.y')
        root.add_constraint('obj.y')

    def test_rev(self):
        p = self.p
        p.setup(check=False, mode='rev')
        p.run_model()

        # test will fail if this fails to converge
        J = p.compute_totals(['con.y', 'obj.y'],
                             ['inputs.x'], return_format='dict')


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class CheckParallelDerivColoringEfficiency(unittest.TestCase):
    # these tests check that redudant calls to compute_jacvec_product
    # are not performed when running parallel derivatives
    # ref issue 1405

    N_PROCS = 3

    def setup_model(self, size):
        class DelayComp(om.ExplicitComponent):

            def initialize(self):
                self.counter = 0
                self.options.declare('time', default=3.0)
                self.options.declare('size', default=1)

            def setup(self):
                size = self.options['size']
                self.add_input('x', shape=size)
                self.add_output('y', shape=size)
                self.add_output('y2', shape=size)

            def compute(self, inputs, outputs):
                waittime = self.options['time']
                size = self.options['size']
                outputs['y'] = np.linspace(3, 10, size) * inputs['x']
                outputs['y2'] = np.linspace(2, 4, size) * inputs['x']

            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                waittime = self.options['time']
                size = self.options['size']
                if mode == 'fwd':
                    time.sleep(waittime)
                    if 'x' in d_inputs:
                        self.counter += 1
                        if 'y' in d_outputs:
                            d_outputs['y'] += np.linspace(3, 10, size)*d_inputs['x']
                        if 'y2' in d_outputs:
                            d_outputs['y2'] += np.linspace(2, 4, size)*d_inputs['x']
                elif mode == 'rev':
                    if 'x' in d_inputs:
                        self.counter += 1
                        time.sleep(waittime)
                        if 'y' in d_outputs:
                            d_inputs['x'] += np.linspace(3, 10, size)*d_outputs['y']
                        if 'y2' in d_outputs:
                            d_inputs['x'] += np.linspace(2, 4, size)*d_outputs['y2']
        model = om.Group()
        iv = om.IndepVarComp()
        mysize = size
        iv.add_output('x', val=3.0 * np.ones((mysize, )))
        model.add_subsystem('iv', iv)
        pg = model.add_subsystem('pg', om.ParallelGroup(), promotes=['*'])
        pg.add_subsystem('dc1', DelayComp(size=mysize, time=0.0))
        pg.add_subsystem('dc2', DelayComp(size=mysize, time=0.0))
        pg.add_subsystem('dc3', DelayComp(size=mysize, time=0.0))
        model.connect('iv.x', ['dc1.x', 'dc2.x', 'dc3.x'])
        model.linear_solver = om.LinearRunOnce()
        model.add_design_var('iv.x', lower=-1.0, upper=1.0)

        return model

    def test_parallel_deriv_coloring_for_redundant_calls(self):
        model = self.setup_model(size=6)
        pdc = 'a'
        model.add_constraint('dc1.y', indices=[0], lower=-1.0, upper=1.0, parallel_deriv_color=pdc)
        # setting dc2.y2 to a parallel deriv color is no longer valid, i.e. setting multiple variables
        # on the same component to the same parallel color makes no sense because they can't be solved
        # in parallel.  In the past, we maintained separate vectors and rhs for each par deriv var
        # so they *could* be solved for simultaneously, but we now just use a single linear vector and
        # rhs so this isn't possible.
        model.add_constraint('dc2.y2', indices=[1], lower=-1.0, upper=1.0)
        model.add_constraint('dc2.y', indices=[3], lower=-1.0, upper=1.0, parallel_deriv_color=pdc)
        model.add_objective('dc3.y', index=2, parallel_deriv_color=pdc)

        prob = om.Problem(model=model)

        prob.setup(mode='rev', force_alloc_complex=True)
        prob.run_model()
        data = prob.check_totals(method='cs', out_stream=None)
        assert_near_equal(data[('dc1.y', 'iv.x')]['abs error'].reverse, 0.0, 1e-6)
        assert_near_equal(data[('dc2.y2', 'iv.x')]['abs error'].reverse, 0.0, 1e-6)
        assert_near_equal(data[('dc2.y', 'iv.x')]['abs error'].reverse, 0.0, 1e-6)
        assert_near_equal(data[('dc3.y', 'iv.x')]['abs error'].reverse, 0.0, 1e-6)

        comm = MPI.COMM_WORLD
        # should only need one jacvec product per linear solve
        dc1count = dc2count = dc3count = 0.0
        dc1count = comm.allreduce(prob.model.pg.dc1.counter, op=MPI.SUM)
        dc2count = comm.allreduce(prob.model.pg.dc2.counter, op=MPI.SUM)
        dc3count = comm.allreduce(prob.model.pg.dc3.counter, op=MPI.SUM)
        # one linear solve on proc 0
        self.assertEqual(dc1count, 1)
        # two solves on proc 1
        self.assertEqual(dc2count, 2)
        # one solve on proc 2
        self.assertEqual(dc3count, 1)

    def test_parallel_deriv_coloring_for_redundant_calls_vector(self):
        model = self.setup_model(size=5)
        pdc = 'a'
        model.add_constraint('dc1.y', lower=-1.0, upper=1.0, parallel_deriv_color=pdc)
        model.add_constraint('dc2.y2', lower=-1.0, upper=1.0)
        model.add_constraint('dc2.y', lower=-1.0, upper=1.0, parallel_deriv_color=pdc)
        model.add_objective('dc3.y', index=2)

        prob = om.Problem(model=model)
        prob.setup(mode='rev', force_alloc_complex=True)
        prob.run_model()
        data = prob.check_totals(method='cs', out_stream=None)
        assert_near_equal(data[('dc1.y', 'iv.x')]['abs error'].reverse, 0.0, 1e-6)
        assert_near_equal(data[('dc2.y2', 'iv.x')]['abs error'].reverse, 0.0, 1e-6)
        assert_near_equal(data[('dc2.y', 'iv.x')]['abs error'].reverse, 0.0, 1e-6)
        assert_near_equal(data[('dc3.y', 'iv.x')]['abs error'].reverse, 0.0, 1e-6)

        # should only need one jacvec product per linear solve
        comm = MPI.COMM_WORLD
        dc1count = dc2count = dc3count = 0.0
        dc1count = comm.allreduce(prob.model.pg.dc1.counter, op=MPI.SUM)
        dc2count = comm.allreduce(prob.model.pg.dc2.counter, op=MPI.SUM)
        dc3count = comm.allreduce(prob.model.pg.dc3.counter, op=MPI.SUM)
        # five linear solves on proc 0
        self.assertEqual(dc1count, 5)
        # ten solves on proc 1
        self.assertEqual(dc2count, 10)
        # one solve on proc 2
        self.assertEqual(dc3count, 1)

    def test_parallel_deriv_coloring_overlap_err(self):
        model = self.setup_model(size=6)
        pdc = 'a'
        model.add_constraint('dc1.y', indices=[0], lower=-1.0, upper=1.0, parallel_deriv_color=pdc)
        model.add_constraint('dc2.y2', indices=[1], lower=-1.0, upper=1.0, parallel_deriv_color=pdc)
        model.add_constraint('dc2.y', indices=[3], lower=-1.0, upper=1.0, parallel_deriv_color=pdc)
        model.add_objective('dc3.y', index=2, parallel_deriv_color=pdc)

        prob = om.Problem(model=model, name='parallel_deriv_coloring_overlap_err')
        prob.setup(mode='rev')
        with self.assertRaises(Exception) as ctx:
            prob.final_setup()
        self.assertEqual(str(ctx.exception),
           "Parallel derivative color 'a' has responses ['pg.dc2.y', 'pg.dc2.y2'] with overlapping dependencies on the same rank.")


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestAutoIVCParDerivBug(unittest.TestCase):
    N_PROCS = 4

    def test_auto_ivc_par_deriv_bug(self):
        class Simple(om.ExplicitComponent):
            def __init__(self, mult1, mult2, mult3, mult4, **kwargs):
                super().__init__(**kwargs)
                self.mult1 = mult1
                self.mult2 = mult2
                self.mult3 = mult3
                self.mult4 = mult4

            def setup(self):
                self.add_input('x1', 0.1)
                self.add_input('x2', 0.01)

                self.add_output('y1', 0.0)
                self.add_output('y2', 0.0)

                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                outputs['y1'] = self.mult1 * inputs['x1'] + self.mult3 * inputs['x2']
                outputs['y2'] = self.mult2 * inputs['x1'] ** 2 + self.mult4 * inputs['x2'] ** 2

            def compute_partials(self, inputs, partials):
                partials['y1', 'x1'] = self.mult1
                partials['y1', 'x2'] = self.mult3
                partials['y2', 'x1'] = 2 * self.mult2 * inputs['x1']
                partials['y2', 'x2'] = 2 * self.mult4 * inputs['x2']

        prob = om.Problem()
        par = prob.model.add_subsystem('par', om.ParallelGroup())
        par.add_subsystem("C1", Simple(mult1=0.5, mult2=0.15, mult3=0.25, mult4=0.35))
        par.add_subsystem("C2", Simple(mult1=0.75, mult2=0.65, mult3=0.45, mult4=0.15))

        prob.model.add_design_var('par.C1.x1', lower=-50, upper=50)
        prob.model.add_design_var('par.C1.x2', lower=-50, upper=50)
        prob.model.add_design_var('par.C2.x1', lower=-50, upper=50)
        prob.model.add_design_var('par.C2.x2', lower=-50, upper=50)

        # Use the parallel derivative option to solve constraint simultaneously
        prob.model.add_constraint('par.C1.y1', equals=1.0, parallel_deriv_color="pd1")
        prob.model.add_constraint('par.C2.y1', equals=2.5, parallel_deriv_color="pd1")
        prob.model.add_objective('par.C1.y2')

        prob.setup(mode='rev', force_alloc_complex=True)

        prob.run_model()

        assert_check_totals(prob.check_totals(method='cs', out_stream=None))


class LinearComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("a", desc="slope")
        self.options.declare("b", desc="y-intercept")

    def setup(self):
        self.a = self.options["a"]
        self.b = self.options["b"]
        self.add_input("x", val=0.0)
        self.add_output("y", val=1.0)
        self.add_output("z", val=0.0)
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        outputs["y"] = self.a * inputs["x"] + self.b
        outputs["z"] = self.a * inputs["x"] ** 2 + self.b * inputs["x"] + 1.0

class LinearGroup(om.Group):
    def initialize(self):
        self.options.declare("a", desc="slope")
        self.options.declare("b", desc="y-intercept")

    def setup(self):
        ivc = self.add_subsystem("ivc", om.IndepVarComp("x", val=0.0), promotes=["*"])
        self.add_subsystem("eval", LinearComp(a=self.options["a"], b=self.options["b"]), promotes=["*"])
        # Make x a dv for the linear equation
        self.add_design_var("x", lower=-100.0, upper=100.0)
        # Add constraint to find x intercept
        self.add_constraint("y", equals=0.0, parallel_deriv_color="lift_con")


@use_tempdirs
@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestParDerivRelevance(unittest.TestCase):
    N_PROCS = 3

    def test_par_deriv_relevance(self):
        from openmdao.utils.general_utils import set_pyoptsparse_opt

        # check that pyoptsparse is installed. if it is, try to use SLSQP.
        OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')

        if OPTIMIZER:
            from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
        else:
            raise unittest.SkipTest("pyOptSparseDriver is required.")

        prob = om.Problem()
        model = prob.model

        # Solve linear equation in parallel
        parallel = model.add_subsystem('parallel', om.ParallelGroup())
        parallel.add_subsystem('line1', LinearGroup(a=1.0, b=1.0))
        parallel.add_subsystem('line2', LinearGroup(a=-1.0, b=1.0))
        parallel.add_subsystem('line3', LinearGroup(a=5.0, b=3.14159))

        # Add a dummy constraint because openmdao requires one
        model.add_objective("parallel.line1.z")

        # Setup to solve constrained problem
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = "SLSQP"

        prob.setup()
        prob.run_driver()

        # Solution should be (-1.0, 1.0, -0.6283)
        assert_near_equal(prob.get_val("parallel.line1.x", get_remote=True), -1.0)
        assert_near_equal(prob.get_val("parallel.line2.x", get_remote=True), 1.0)
        assert_near_equal(prob.get_val("parallel.line3.x", get_remote=True), -0.628318)


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
