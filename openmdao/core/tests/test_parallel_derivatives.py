""" Test out some specialized parallel derivatives features"""


from io import StringIO
import sys
import unittest
import time
import random
from distutils.version import LooseVersion

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.sellar import SellarDerivatives, \
    SellarDis1withDerivatives, SellarDis2withDerivatives
from openmdao.test_suite.groups.parallel_groups import FanOutGrouped, FanInGrouped
from openmdao.utils.assert_utils import assert_near_equal
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
        prob.run_driver()

        indep_list = ['x1', 'x2']
        unknown_list = ['c3.y']

        J = prob.compute_totals(unknown_list, indep_list, return_format='dict')
        assert_near_equal(J['c3.y']['x1'][0][0], -6.0, 1e-6)
        assert_near_equal(J['c3.y']['x2'][0][0], 35.0, 1e-6)

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

    def test_fan_in_parallel_sets_fwd(self):

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

        prob.setup(check=False, mode='fwd')
        prob.run_driver()

        indep_list = ['x1', 'x2']
        unknown_list = ['c3.y']

        J = prob.compute_totals(unknown_list, indep_list, return_format='flat_dict')
        assert_near_equal(J['c3.y', 'x1'][0][0], -6.0, 1e-6)
        assert_near_equal(J['c3.y', 'x2'][0][0], 35.0, 1e-6)

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
            self.assertTrue('In mode: fwd, Solving variable(s) using simul coloring:' in output)
            self.assertTrue("('p.x3', [2])" in output)

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
        vec = prob.model._vectors['residual']['c2.y']
        norm_val = vec.get_norm()
        # NOTE: BAN updated the norm value for the PR that added seed splitting, i.e.
        # the seed, c2.y in this case, is half what it was before (-.5 vs. -1).
        assert_near_equal(norm_val, 6.422616289332565, 1e-6)


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

    def test_parallel_fwd_multi(self):
        asize = self.asize
        prob = self.setup_model()

        prob.model.add_design_var('Indep1.x', parallel_deriv_color='pardv', vectorize_derivs=True)
        prob.model.add_design_var('Indep2.x', parallel_deriv_color='pardv', vectorize_derivs=True)
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


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class MatMatTestCase(unittest.TestCase):
    N_PROCS = 2
    asize = 3

    def setup_model(self):
        asize = self.asize
        prob = om.Problem()
        root = prob.model
        root.linear_solver = om.LinearBlockGS()

        p1 = root.add_subsystem('p1', om.IndepVarComp('x', np.arange(asize, dtype=float)+1.0))
        p2 = root.add_subsystem('p2', om.IndepVarComp('x', np.arange(asize, dtype=float)+1.0))
        G1 = root.add_subsystem('G1', om.ParallelGroup())
        G1.linear_solver = om.LinearBlockGS()

        c1 = G1.add_subsystem('c1', om.ExecComp('y = ones(3).T*x.dot(arange(3.,6.))',
                                                x=np.zeros(asize), y=np.zeros(asize)))
        c2 = G1.add_subsystem('c2', om.ExecComp('y = x * 2.0',
                                                x=np.zeros(asize), y=np.zeros(asize)))

        c3 = root.add_subsystem('c3', om.ExecComp('y = x * 5.0',
                                                  x=np.zeros(asize), y=np.zeros(asize)))
        c4 = root.add_subsystem('c4', om.ExecComp('y = x * 4.0',
                                                  x=np.zeros(asize), y=np.zeros(asize)))
        root.connect('p1.x', 'G1.c1.x')
        root.connect('p2.x', 'G1.c2.x')
        root.connect('G1.c1.y', 'c3.x')
        root.connect('G1.c2.y', 'c4.x')

        return prob

    def test_parallel_fwd(self):
        asize = self.asize
        prob = self.setup_model()

        prob.model.add_design_var('p1.x', parallel_deriv_color='par')
        prob.model.add_design_var('p2.x', parallel_deriv_color='par')
        prob.model.add_constraint('c3.y', upper=0.0)
        prob.model.add_constraint('c4.y', upper=0.0)

        prob.setup(check=False, mode='fwd')
        prob.run_driver()

        J = prob.compute_totals(['c3.y', 'c4.y'], ['p1.x', 'p2.x'],
                                return_format='flat_dict')

        assert_near_equal(J['c3.y', 'p1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.eye(asize)*8.0
        assert_near_equal(J['c4.y', 'p2.x'], expected, 1e-6)

    def test_parallel_multi_fwd(self):
        asize = self.asize
        prob = self.setup_model()

        prob.model.add_design_var('p1.x', parallel_deriv_color='par', vectorize_derivs=True)
        prob.model.add_design_var('p2.x', parallel_deriv_color='par', vectorize_derivs=True)
        prob.model.add_constraint('c3.y', upper=0.0)
        prob.model.add_constraint('c4.y', upper=0.0)

        prob.setup(check=False, mode='fwd')
        prob.run_driver()

        J = prob.compute_totals(['c3.y', 'c4.y'], ['p1.x', 'p2.x'],
                                return_format='flat_dict')

        assert_near_equal(J['c3.y', 'p1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.eye(asize)*8.0
        assert_near_equal(J['c4.y', 'p2.x'], expected, 1e-6)

    def test_parallel_rev(self):
        asize = self.asize
        prob = self.setup_model()

        prob.model.add_design_var('p1.x')
        prob.model.add_design_var('p2.x')
        prob.model.add_constraint('c3.y', upper=0.0, parallel_deriv_color='par')
        prob.model.add_constraint('c4.y', upper=0.0, parallel_deriv_color='par')

        prob.setup(check=False, mode='rev')
        prob.run_driver()

        J = prob.compute_totals(['c3.y', 'c4.y'], ['p1.x', 'p2.x'],
                                return_format='flat_dict')

        assert_near_equal(J['c3.y', 'p1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.eye(asize)*8.0
        assert_near_equal(J['c4.y', 'p2.x'], expected, 1e-6)

    def test_parallel_multi_rev(self):
        asize = self.asize
        prob = self.setup_model()

        prob.model.add_design_var('p1.x')
        prob.model.add_design_var('p2.x')
        prob.model.add_constraint('c3.y', upper=0.0, parallel_deriv_color='par', vectorize_derivs=True)
        prob.model.add_constraint('c4.y', upper=0.0, parallel_deriv_color='par', vectorize_derivs=True)

        prob.setup(check=False, mode='rev')
        prob.run_driver()

        J = prob.compute_totals(['c3.y', 'c4.y'], ['p1.x', 'p2.x'],
                                return_format='flat_dict')

        assert_near_equal(J['c3.y', 'p1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.eye(asize)*8.0
        assert_near_equal(J['c4.y', 'p2.x'], expected, 1e-6)


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

    def _apply_linear(self, jac, vec_names, rel_systems, mode, scope_out=None, scope_in=None):
        time.sleep(self.delay)
        super()._apply_linear(jac, vec_names, rel_systems, mode, scope_out, scope_in)


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
@unittest.skipUnless(MPI and PETScVector and LooseVersion(np.__version__) >= LooseVersion("1.13"),
                     "MPI, PETSc, and numpy >= 1.13 are required.")
class ParDerivColorFeatureTestCase(unittest.TestCase):
    N_PROCS = 2

    def test_feature_rev(self):
        import time

        import numpy as np

        import openmdao.api as om
        from openmdao.core.tests.test_parallel_derivatives import PartialDependGroup

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
        import time

        import numpy as np

        import openmdao.api as om
        from openmdao.core.tests.test_parallel_derivatives import PartialDependGroup

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
        import time

        import numpy as np

        import openmdao.api as om
        from openmdao.core.tests.test_parallel_derivatives import PartialDependGroup

        size = 4

        of = ['ParallelGroup1.Con1.y', 'ParallelGroup1.Con2.y']
        wrt = ['Comp1.x']

        # run in rev mode
        p = om.Problem(model=PartialDependGroup())
        p.setup(mode='rev')

        p.run_model()

        elapsed_rev = time.time()
        Jrev = p.compute_totals(of, wrt, return_format='dict')
        elapsed_rev = time.time() - elapsed_rev

        # run in fwd mode and compare times for deriv calculation
        p = om.Problem(model=PartialDependGroup())
        p.setup(mode='fwd')
        p.run_model()

        elapsed_fwd = time.time()
        Jfwd = p.compute_totals(of, wrt, return_format='dict')
        elapsed_fwd = time.time() - elapsed_fwd

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
                self.declare_partials('y', 'x')
                self.declare_partials('y2', 'x')

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
        model.add_constraint('dc2.y2', indices=[1], lower=-1.0, upper=1.0, parallel_deriv_color=pdc)
        model.add_constraint('dc2.y', indices=[3], lower=-1.0, upper=1.0, parallel_deriv_color=pdc)
        model.add_objective('dc3.y', index=2, parallel_deriv_color=pdc)

        prob = om.Problem(model=model)
        prob.setup(mode='rev', force_alloc_complex=True)
        prob.run_model()
        data = prob.check_totals(method='cs', out_stream=None)
        assert_near_equal(data[('pg.dc1.y', 'iv.x')]['abs error'][0], 0.0, 1e-6)
        assert_near_equal(data[('pg.dc2.y2', 'iv.x')]['abs error'][0], 0.0, 1e-6)
        assert_near_equal(data[('pg.dc2.y', 'iv.x')]['abs error'][0], 0.0, 1e-6)
        assert_near_equal(data[('pg.dc3.y', 'iv.x')]['abs error'][0], 0.0, 1e-6)

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
        model.add_constraint('dc2.y2', lower=-1.0, upper=1.0, parallel_deriv_color=pdc)
        model.add_constraint('dc2.y', lower=-1.0, upper=1.0, parallel_deriv_color=pdc)
        # objective is a scalar - gets its own color to avoid being called 10x
        model.add_objective('dc3.y', index=2, parallel_deriv_color='b')

        prob = om.Problem(model=model)
        prob.setup(mode='rev', force_alloc_complex=True)
        prob.run_model()
        data = prob.check_totals(method='cs', out_stream=None)
        assert_near_equal(data[('pg.dc1.y', 'iv.x')]['abs error'][0], 0.0, 1e-6)
        assert_near_equal(data[('pg.dc2.y2', 'iv.x')]['abs error'][0], 0.0, 1e-6)
        assert_near_equal(data[('pg.dc2.y', 'iv.x')]['abs error'][0], 0.0, 1e-6)
        assert_near_equal(data[('pg.dc3.y', 'iv.x')]['abs error'][0], 0.0, 1e-6)

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

if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
