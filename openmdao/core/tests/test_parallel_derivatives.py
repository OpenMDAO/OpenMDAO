""" Test out some specialized parallel derivatives features"""

from __future__ import print_function

from six.moves import cStringIO as StringIO
import sys
import unittest
import time
import random
from distutils.version import LooseVersion

import numpy as np

from openmdao.api import Group, ParallelGroup, Problem, IndepVarComp, LinearBlockGS, DefaultVector, \
    ExecComp, ExplicitComponent, PETScVector, ScipyKrylov, NonlinearBlockGS
from openmdao.utils.mpi import MPI
from openmdao.test_suite.components.sellar import SellarDerivatives, SellarDis1withDerivatives, SellarDis2withDerivatives
from openmdao.test_suite.groups.parallel_groups import FanOutGrouped, FanInGrouped
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.recorders.recording_iteration_stack import recording_iteration


if MPI:
    from openmdao.api import PETScVector
    vector_class = PETScVector
else:
    vector_class = DefaultVector


@unittest.skipIf(MPI and not PETScVector, "only run under MPI if we have PETSc.")
class ParDerivTestCase(unittest.TestCase):

    N_PROCS = 2

    def test_fan_in_serial_sets_rev(self):

        prob = Problem()
        prob.model = FanInGrouped()
        prob.model.linear_solver = LinearBlockGS()
        prob.model.sub.linear_solver = LinearBlockGS()

        prob.model.add_design_var('iv.x1')
        prob.model.add_design_var('iv.x2')
        prob.model.add_objective('c3.y')

        prob.setup(vector_class=vector_class, check=False, mode='rev')
        prob.run_driver()

        indep_list = ['iv.x1', 'iv.x2']
        unknown_list = ['c3.y']

        J = prob.compute_totals(unknown_list, indep_list, return_format='dict')
        assert_rel_error(self, J['c3.y']['iv.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y']['iv.x2'][0][0], 35.0, 1e-6)

    def test_fan_in_serial_sets_fwd(self):

        prob = Problem()
        prob.model = FanInGrouped()
        prob.model.linear_solver = LinearBlockGS()
        prob.model.sub.linear_solver = LinearBlockGS()

        prob.model.add_design_var('iv.x1')
        prob.model.add_design_var('iv.x2')
        prob.model.add_objective('c3.y')

        prob.setup(vector_class=vector_class, check=False, mode='fwd')
        prob.run_driver()

        indep_list = ['iv.x1', 'iv.x2']
        unknown_list = ['c3.y']

        J = prob.compute_totals(unknown_list, indep_list, return_format='flat_dict')
        assert_rel_error(self, J['c3.y', 'iv.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x2'][0][0], 35.0, 1e-6)

    def test_fan_out_serial_sets_fwd(self):

        prob = Problem()
        prob.model = FanOutGrouped()
        prob.model.linear_solver = LinearBlockGS()
        prob.model.sub.linear_solver = LinearBlockGS()

        prob.model.add_design_var('iv.x')
        prob.model.add_constraint('c2.y', upper=0.0)
        prob.model.add_constraint('c3.y', upper=0.0)

        prob.setup(vector_class=vector_class, check=False, mode='fwd')
        prob.run_driver()

        unknown_list = ['c2.y', 'c3.y']
        indep_list = ['iv.x']

        J = prob.compute_totals(unknown_list, indep_list, return_format='flat_dict')
        assert_rel_error(self, J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

    def test_fan_out_serial_sets_rev(self):

        prob = Problem()
        prob.model = FanOutGrouped()
        prob.model.linear_solver = LinearBlockGS()
        prob.model.sub.linear_solver = LinearBlockGS()

        prob.model.add_design_var('iv.x')
        prob.model.add_constraint('c2.y', upper=0.0)
        prob.model.add_constraint('c3.y', upper=0.0)

        prob.setup(vector_class=vector_class, check=False, mode='rev')
        prob.run_driver()

        unknown_list = ['c3.y','c2.y'] #['c2.y', 'c3.y']
        indep_list = ['iv.x']

        J = prob.compute_totals(unknown_list, indep_list, return_format='flat_dict')
        assert_rel_error(self, J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

    def test_fan_in_parallel_sets_fwd(self):

        prob = Problem()
        prob.model = FanInGrouped()
        prob.model.linear_solver = LinearBlockGS()
        prob.model.sub.linear_solver = LinearBlockGS()

        prob.model.add_design_var('iv.x1', parallel_deriv_color='par_dv')
        prob.model.add_design_var('iv.x2', parallel_deriv_color='par_dv')
        prob.model.add_design_var('iv.x3')
        prob.model.add_objective('c3.y')

        prob.setup(vector_class=vector_class, check=False, mode='fwd')
        prob.run_driver()

        indep_list = ['iv.x1', 'iv.x2']
        unknown_list = ['c3.y']

        J = prob.compute_totals(unknown_list, indep_list, return_format='flat_dict')
        assert_rel_error(self, J['c3.y', 'iv.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x2'][0][0], 35.0, 1e-6)

    def test_debug_print_option_totals_color(self):

        prob = Problem()
        prob.model = FanInGrouped()
        prob.model.linear_solver = LinearBlockGS()
        prob.model.sub.linear_solver = LinearBlockGS()

        prob.model.add_design_var('iv.x1', parallel_deriv_color='par_dv')
        prob.model.add_design_var('iv.x2', parallel_deriv_color='par_dv')
        prob.model.add_design_var('iv.x3')
        prob.model.add_objective('c3.y')

        prob.driver.options['debug_print'] = ['totals']

        prob.setup(vector_class=vector_class, check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_driver()

        indep_list = ['iv.x1', 'iv.x2', 'iv.x3']
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
            self.assertTrue('Solving color: par_dv (iv.x1, iv.x2)' in output)
            self.assertTrue('Solving variable: iv.x3' in output)

    def test_fan_out_parallel_sets_rev(self):

        prob = Problem()
        prob.model = FanOutGrouped()
        prob.model.linear_solver = LinearBlockGS()
        prob.model.sub.linear_solver = LinearBlockGS()

        prob.model.add_design_var('iv.x')
        prob.model.add_constraint('c2.y', upper=0.0, parallel_deriv_color='par_resp')
        prob.model.add_constraint('c3.y', upper=0.0, parallel_deriv_color='par_resp')

        prob.setup(vector_class=vector_class, check=False, mode='rev')
        prob.run_driver()

        unknown_list = ['c2.y', 'c3.y']
        indep_list = ['iv.x']

        J = prob.compute_totals(unknown_list, indep_list, return_format='flat_dict')
        assert_rel_error(self, J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)


@unittest.skipIf(MPI and not PETScVector, "only run under MPI if we have PETSc.")
class DecoupledTestCase(unittest.TestCase):
    N_PROCS = 2
    asize = 3

    def setup_model(self):
        asize = self.asize
        prob = Problem()
        #import wingdbstub
        root = prob.model
        root.linear_solver = LinearBlockGS()

        Indep1 = root.add_subsystem('Indep1', IndepVarComp('x', np.arange(asize, dtype=float)+1.0))
        Indep2 = root.add_subsystem('Indep2', IndepVarComp('x', np.arange(asize+2, dtype=float)+1.0))
        G1 = root.add_subsystem('G1', ParallelGroup())
        G1.linear_solver = LinearBlockGS()

        c1 = G1.add_subsystem('c1', ExecComp('y = ones(3).T*x.dot(arange(3.,6.))',
                                                  x=np.zeros(asize), y=np.zeros(asize)))
        c2 = G1.add_subsystem('c2', ExecComp('y = x[:%d] * 2.0' % asize,
                                        x=np.zeros(asize+2), y=np.zeros(asize)))

        Con1 = root.add_subsystem('Con1', ExecComp('y = x * 5.0',
                                          x=np.zeros(asize), y=np.zeros(asize)))
        Con2 = root.add_subsystem('Con2', ExecComp('y = x * 4.0',
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

        prob.setup(vector_class=vector_class, check=False, mode='fwd')
        prob.run_model()

        J = prob.compute_totals(['Con1.y', 'Con2.y'], ['Indep1.x', 'Indep2.x'],
                                return_format='flat_dict')

        assert_rel_error(self, J['Con1.y', 'Indep1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.zeros((asize, asize+2))
        expected[:,:asize] = np.eye(asize)*8.0
        assert_rel_error(self, J['Con2.y', 'Indep2.x'], expected, 1e-6)

    def test_parallel_fwd(self):
        asize = self.asize
        prob = self.setup_model()

        #import wingdbstub

        prob.model.add_design_var('Indep1.x', parallel_deriv_color='pardv')
        prob.model.add_design_var('Indep2.x', parallel_deriv_color='pardv')
        prob.model.add_constraint('Con1.y', upper=0.0)
        prob.model.add_constraint('Con2.y', upper=0.0)

        prob.setup(vector_class=vector_class, check=False, mode='fwd')
        prob.run_driver()

        J = prob.compute_totals(['Con1.y', 'Con2.y'], ['Indep1.x', 'Indep2.x'],
                                return_format='flat_dict')

        assert_rel_error(self, J['Con1.y', 'Indep1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.zeros((asize, asize+2))
        expected[:,:asize] = np.eye(asize)*8.0
        assert_rel_error(self, J['Con2.y', 'Indep2.x'], expected, 1e-6)

    def test_parallel_fwd_multi(self):
        asize = self.asize
        prob = self.setup_model()

        #import wingdbstub

        prob.model.add_design_var('Indep1.x', parallel_deriv_color='pardv', vectorize_derivs=True)
        prob.model.add_design_var('Indep2.x', parallel_deriv_color='pardv', vectorize_derivs=True)
        prob.model.add_constraint('Con1.y', upper=0.0)
        prob.model.add_constraint('Con2.y', upper=0.0)

        prob.setup(vector_class=vector_class, check=False, mode='fwd')
        prob.run_driver()

        J = prob.compute_totals(['Con1.y', 'Con2.y'], ['Indep1.x', 'Indep2.x'],
                                return_format='flat_dict')

        assert_rel_error(self, J['Con1.y', 'Indep1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.zeros((asize, asize+2))
        expected[:,:asize] = np.eye(asize)*8.0
        assert_rel_error(self, J['Con2.y', 'Indep2.x'], expected, 1e-6)

    def test_serial_rev(self):
        asize = self.asize
        prob = self.setup_model()

        prob.model.add_design_var('Indep1.x')
        prob.model.add_design_var('Indep2.x')
        prob.model.add_constraint('Con1.y', upper=0.0)
        prob.model.add_constraint('Con2.y', upper=0.0)

        prob.setup(vector_class=vector_class, check=False, mode='rev')
        prob.run_driver()

        J = prob.compute_totals(['Con1.y', 'Con2.y'], ['Indep1.x', 'Indep2.x'],
                                return_format='flat_dict')

        assert_rel_error(self, J['Con1.y', 'Indep1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.zeros((asize, asize+2))
        expected[:,:asize] = np.eye(asize)*8.0
        assert_rel_error(self, J['Con2.y', 'Indep2.x'], expected, 1e-6)

    def test_parallel_rev(self):
        asize = self.asize
        
        prob = self.setup_model()

        prob.model.add_design_var('Indep1.x')
        prob.model.add_design_var('Indep2.x')
        prob.model.add_constraint('Con1.y', upper=0.0, parallel_deriv_color='parc')
        prob.model.add_constraint('Con2.y', upper=0.0, parallel_deriv_color='parc')

        prob.setup(vector_class=vector_class, check=False, mode='rev')
        prob.run_driver()

        J = prob.compute_totals(['Con1.y', 'Con2.y'], ['Indep1.x', 'Indep2.x'],
                                return_format='flat_dict')

        assert_rel_error(self, J['Con1.y', 'Indep1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.zeros((asize, asize+2))
        expected[:,:asize] = np.eye(asize)*8.0
        assert_rel_error(self, J['Con2.y', 'Indep2.x'], expected, 1e-6)


@unittest.skipIf(MPI and not PETScVector, "only run under MPI if we have PETSc.")
class IndicesTestCase(unittest.TestCase):

    N_PROCS = 2

    def setup_model(self, mode):
        asize = 3
        prob = Problem()
        root = prob.model
        root.linear_solver = LinearBlockGS()

        p = root.add_subsystem('p', IndepVarComp('x', np.arange(asize, dtype=float)+1.0))
        G1 = root.add_subsystem('G1', ParallelGroup())
        G1.linear_solver = LinearBlockGS()

        c2 = G1.add_subsystem('c2', ExecComp('y = x * 2.0',
                                        x=np.zeros(asize), y=np.zeros(asize)))
        c3 = G1.add_subsystem('c3', ExecComp('y = ones(3).T*x.dot(arange(3.,6.))',
                                        x=np.zeros(asize), y=np.zeros(asize)))
        c4 = root.add_subsystem('c4', ExecComp('y = x * 4.0',
                                          x=np.zeros(asize), y=np.zeros(asize)))
        c5 = root.add_subsystem('c5', ExecComp('y = x * 5.0',
                                          x=np.zeros(asize), y=np.zeros(asize)))

        prob.model.add_design_var('p.x', indices=[1, 2])
        prob.model.add_constraint('c4.y', upper=0.0, indices=[1], parallel_deriv_color='par_resp')
        prob.model.add_constraint('c5.y', upper=0.0, indices=[2], parallel_deriv_color='par_resp')

        root.connect('p.x', 'G1.c2.x')
        root.connect('p.x', 'G1.c3.x')
        root.connect('G1.c2.y', 'c4.x')
        root.connect('G1.c3.y', 'c5.x')

        prob.setup(vector_class=vector_class, check=False, mode=mode)
        prob.run_driver()

        return prob

    def test_indices_fwd(self):
        prob = self.setup_model('fwd')

        J = prob.compute_totals(['c4.y', 'c5.y'], ['p.x'],
                                return_format='flat_dict')

        assert_rel_error(self, J['c5.y', 'p.x'][0], np.array([20., 25.]), 1e-6)
        assert_rel_error(self, J['c4.y', 'p.x'][0], np.array([8., 0.]), 1e-6)

    def test_indices_rev(self):
        prob = self.setup_model('rev')
        J = prob.compute_totals(['c4.y', 'c5.y'], ['p.x'],
                                return_format='flat_dict')

        assert_rel_error(self, J['c5.y', 'p.x'][0], np.array([20., 25.]), 1e-6)
        assert_rel_error(self, J['c4.y', 'p.x'][0], np.array([8., 0.]), 1e-6)


@unittest.skipIf(MPI and not PETScVector, "only run under MPI if we have PETSc.")
class IndicesTestCase2(unittest.TestCase):

    N_PROCS = 2

    def setup_model(self, mode):
        asize = 3
        prob = Problem()
        root = prob.model

        root.linear_solver = LinearBlockGS()

        G1 = root.add_subsystem('G1', Group())# ParallelGroup())
        G1.linear_solver = LinearBlockGS()

        par1 = G1.add_subsystem('par1', Group())
        par1.linear_solver = LinearBlockGS()
        par2 = G1.add_subsystem('par2', Group())
        par2.linear_solver = LinearBlockGS()

        p1 = par1.add_subsystem('p', IndepVarComp('x', np.arange(asize, dtype=float)+1.0))
        p2 = par2.add_subsystem('p', IndepVarComp('x', np.arange(asize, dtype=float)+10.0))

        c2 = par1.add_subsystem('c2', ExecComp('y = x * 2.0',
                                        x=np.zeros(asize), y=np.zeros(asize)))
        c3 = par2.add_subsystem('c3', ExecComp('y = ones(3).T*x.dot(arange(3.,6.))',
                                        x=np.zeros(asize), y=np.zeros(asize)))
        c4 = par1.add_subsystem('c4', ExecComp('y = x * 4.0',
                                          x=np.zeros(asize), y=np.zeros(asize)))
        c5 = par2.add_subsystem('c5', ExecComp('y = x * 5.0',
                                          x=np.zeros(asize), y=np.zeros(asize)))

        prob.model.add_design_var('G1.par1.p.x', indices=[1, 2])
        prob.model.add_design_var('G1.par2.p.x', indices=[1, 2])
        prob.model.add_constraint('G1.par1.c4.y', upper=0.0, indices=[1], parallel_deriv_color='par_resp')
        prob.model.add_constraint('G1.par2.c5.y', upper=0.0, indices=[2], parallel_deriv_color='par_resp')

        root.connect('G1.par1.p.x', 'G1.par1.c2.x')
        root.connect('G1.par2.p.x', 'G1.par2.c3.x')
        root.connect('G1.par1.c2.y', 'G1.par1.c4.x')
        root.connect('G1.par2.c3.y', 'G1.par2.c5.x')

        prob.setup(vector_class=vector_class, check=False)
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

        assert_rel_error(self, J['G1.par2.c5.y', 'G1.par2.p.x'][0], np.array([20., 25.]), 1e-6)
        assert_rel_error(self, J['G1.par1.c4.y', 'G1.par1.p.x'][0], np.array([8., 0.]), 1e-6)

    def test_indices_rev(self):
        prob = self.setup_model('rev')
        J = prob.compute_totals(['G1.par1.c4.y', 'G1.par2.c5.y'],
                                ['G1.par1.p.x', 'G1.par2.p.x'],
                                return_format='flat_dict')

        assert_rel_error(self, J['G1.par2.c5.y', 'G1.par2.p.x'][0], np.array([20., 25.]), 1e-6)
        assert_rel_error(self, J['G1.par1.c4.y', 'G1.par1.p.x'][0], np.array([8., 0.]), 1e-6)


@unittest.skipIf(MPI and not PETScVector, "only run under MPI if we have PETSc.")
class MatMatTestCase(unittest.TestCase):
    N_PROCS = 2
    asize = 3

    def setup_model(self):
        asize = self.asize
        prob = Problem()
        root = prob.model
        root.linear_solver = LinearBlockGS()

        p1 = root.add_subsystem('p1', IndepVarComp('x', np.arange(asize, dtype=float)+1.0))
        p2 = root.add_subsystem('p2', IndepVarComp('x', np.arange(asize, dtype=float)+1.0))
        G1 = root.add_subsystem('G1', ParallelGroup())
        G1.linear_solver = LinearBlockGS()

        c1 = G1.add_subsystem('c1', ExecComp('y = ones(3).T*x.dot(arange(3.,6.))',
                                                  x=np.zeros(asize), y=np.zeros(asize)))
        c2 = G1.add_subsystem('c2', ExecComp('y = x * 2.0',
                                        x=np.zeros(asize), y=np.zeros(asize)))

        c3 = root.add_subsystem('c3', ExecComp('y = x * 5.0',
                                          x=np.zeros(asize), y=np.zeros(asize)))
        c4 = root.add_subsystem('c4', ExecComp('y = x * 4.0',
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

        prob.setup(vector_class=vector_class, check=False, mode='fwd')
        prob.run_driver()

        J = prob.compute_totals(['c3.y', 'c4.y'], ['p1.x', 'p2.x'],
                                return_format='flat_dict')

        assert_rel_error(self, J['c3.y', 'p1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.eye(asize)*8.0
        assert_rel_error(self, J['c4.y', 'p2.x'], expected, 1e-6)

    def test_parallel_multi_fwd(self):
        asize = self.asize
        prob = self.setup_model()

        prob.model.add_design_var('p1.x', parallel_deriv_color='par', vectorize_derivs=True)
        prob.model.add_design_var('p2.x', parallel_deriv_color='par', vectorize_derivs=True)
        prob.model.add_constraint('c3.y', upper=0.0)
        prob.model.add_constraint('c4.y', upper=0.0)

        prob.setup(vector_class=vector_class, check=False, mode='fwd')
        prob.run_driver()

        J = prob.compute_totals(['c3.y', 'c4.y'], ['p1.x', 'p2.x'],
                                return_format='flat_dict')

        assert_rel_error(self, J['c3.y', 'p1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.eye(asize)*8.0
        assert_rel_error(self, J['c4.y', 'p2.x'], expected, 1e-6)

    def test_parallel_rev(self):
        asize = self.asize
        prob = self.setup_model()

        prob.model.add_design_var('p1.x')
        prob.model.add_design_var('p2.x')
        prob.model.add_constraint('c3.y', upper=0.0, parallel_deriv_color='par')
        prob.model.add_constraint('c4.y', upper=0.0, parallel_deriv_color='par')

        prob.setup(vector_class=vector_class, check=False, mode='rev')
        prob.run_driver()

        J = prob.compute_totals(['c3.y', 'c4.y'], ['p1.x', 'p2.x'],
                                return_format='flat_dict')

        assert_rel_error(self, J['c3.y', 'p1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.eye(asize)*8.0
        assert_rel_error(self, J['c4.y', 'p2.x'], expected, 1e-6)

    def test_parallel_multi_rev(self):
        asize = self.asize
        prob = self.setup_model()

        prob.model.add_design_var('p1.x')
        prob.model.add_design_var('p2.x')
        prob.model.add_constraint('c3.y', upper=0.0, parallel_deriv_color='par', vectorize_derivs=True)
        prob.model.add_constraint('c4.y', upper=0.0, parallel_deriv_color='par', vectorize_derivs=True)

        prob.setup(vector_class=vector_class,  check=False, mode='rev')
        prob.run_driver()

        J = prob.compute_totals(['c3.y', 'c4.y'], ['p1.x', 'p2.x'],
                                return_format='flat_dict')

        assert_rel_error(self, J['c3.y', 'p1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.eye(asize)*8.0
        assert_rel_error(self, J['c4.y', 'p2.x'], expected, 1e-6)


class SumComp(ExplicitComponent):
    def __init__(self, size):
        super(SumComp, self).__init__()
        self.size = size

    def setup(self):
        self.add_input('x', val=np.zeros(self.size))
        self.add_output('y', val=0.0)

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        outputs['y'] = np.sum(inputs['x'])

    def compute_partials(self, inputs, partials):
        partials['y', 'x'] = np.ones(inputs['x'].size)


class SlowComp(ExplicitComponent):
    """
    Component with a delay that multiplies the input by a multiplier.
    """

    def __init__(self, delay=1.0, size=3, mult=2.0):
        super(SlowComp, self).__init__()
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

    def _apply_linear(self, vec_names, rel_systems, mode, scope_out=None, scope_in=None):
        time.sleep(self.delay)
        super(SlowComp, self)._apply_linear(vec_names, rel_systems, mode, scope_out, scope_in)


class PartialDependGroup(Group):
    def setup(self):
        size = 4

        Indep1 = self.add_subsystem('Indep1', IndepVarComp('x', np.arange(size, dtype=float)+1.0))
        Comp1 = self.add_subsystem('Comp1', SumComp(size))
        pargroup = self.add_subsystem('ParallelGroup1', ParallelGroup())

        self.linear_solver = LinearBlockGS()
        self.linear_solver.options['iprint'] = -1
        pargroup.linear_solver = LinearBlockGS()
        pargroup.linear_solver.options['iprint'] = -1

        delay = .1
        Con1 = pargroup.add_subsystem('Con1', SlowComp(delay=delay, size=2, mult=2.0))
        Con2 = pargroup.add_subsystem('Con2', SlowComp(delay=delay, size=2, mult=-3.0))

        self.connect('Indep1.x', 'Comp1.x')
        self.connect('Comp1.y', 'ParallelGroup1.Con1.x')
        self.connect('Comp1.y', 'ParallelGroup1.Con2.x')

        color = 'parcon'
        self.add_design_var('Indep1.x')
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

        from openmdao.api import Problem, PETScVector
        from openmdao.core.tests.test_parallel_derivatives import PartialDependGroup

        size = 4

        of = ['ParallelGroup1.Con1.y', 'ParallelGroup1.Con2.y']
        wrt = ['Indep1.x']

        # run first in fwd mode
        p = Problem(model=PartialDependGroup())
        p.setup(vector_class=PETScVector, mode='rev')
        p.run_model()

        J = p.compute_totals(of, wrt, return_format='dict')

        assert_rel_error(self, J['ParallelGroup1.Con1.y']['Indep1.x'][0], np.ones(size)*2., 1e-6)
        assert_rel_error(self, J['ParallelGroup1.Con2.y']['Indep1.x'][0], np.ones(size)*-3., 1e-6)

    def test_fwd_vs_rev(self):
        import time

        import numpy as np

        from openmdao.api import Problem, PETScVector
        from openmdao.core.tests.test_parallel_derivatives import PartialDependGroup

        recording_iteration.stack = []
        size = 4

        of = ['ParallelGroup1.Con1.y', 'ParallelGroup1.Con2.y']
        wrt = ['Indep1.x']

        # run first in fwd mode
        p = Problem(model=PartialDependGroup())
        p.setup(vector_class=PETScVector, mode='fwd')
        p.run_model()

        elapsed_fwd = time.time()
        J = p.compute_totals(of, wrt, return_format='dict')
        elapsed_fwd = time.time() - elapsed_fwd

        assert_rel_error(self, J['ParallelGroup1.Con1.y']['Indep1.x'][0], np.ones(size)*2., 1e-6)
        assert_rel_error(self, J['ParallelGroup1.Con2.y']['Indep1.x'][0], np.ones(size)*-3., 1e-6)

        recording_iteration.stack = []

        # now run in rev mode and compare times for deriv calculation
        p = Problem(model=PartialDependGroup())
        p.setup(vector_class=PETScVector, check=False, mode='rev')

        p.run_model()

        elapsed_rev = time.time()
        J = p.compute_totals(of, wrt, return_format='dict')
        elapsed_rev = time.time() - elapsed_rev

        assert_rel_error(self, J['ParallelGroup1.Con1.y']['Indep1.x'][0], np.ones(size)*2., 1e-6)
        assert_rel_error(self, J['ParallelGroup1.Con2.y']['Indep1.x'][0], np.ones(size)*-3., 1e-6)

        # make sure that rev mode is faster than fwd mode
        self.assertGreater(elapsed_fwd / elapsed_rev, 1.0)


class CleanupTestCase(unittest.TestCase):
    # This is to test for a bug john found that caused his ozone problem to fail
    # to converge.  The problem was due to garbage in the doutputs vector that
    # was coming from transfers to irrelevant variables during Group._apply_linear.
    def setUp(self):
        p = self.p = Problem()
        root = p.model
        root.linear_solver = LinearBlockGS()
        root.linear_solver.options['err_on_maxiter'] = True

        inputs = root.add_subsystem("inputs", IndepVarComp("x", 1.0))
        G1 = root.add_subsystem("G1", Group())
        dparam = G1.add_subsystem("dparam", ExecComp("y = .5*x"))
        G1_inputs = G1.add_subsystem("inputs", IndepVarComp("x", 1.5))
        start = G1.add_subsystem("start", ExecComp("y = .7*x"))
        timecomp = G1.add_subsystem("time", ExecComp("y = -.2*x"))

        G2 = G1.add_subsystem("G2", Group())
        stage_step = G2.add_subsystem("stage_step",
                                      ExecComp("y = -0.1*x + .5*x2 - .4*x3 + .9*x4"))
        ode = G2.add_subsystem("ode", ExecComp("y = .8*x - .6*x2"))
        dummy = G2.add_subsystem("dummy", IndepVarComp("x", 1.3))

        step = G1.add_subsystem("step", ExecComp("y = -.2*x + .4*x2 - .4*x3"))
        output = G1.add_subsystem("output", ExecComp("y = .6*x"))

        con = root.add_subsystem("con", ExecComp("y = .2 * x"))
        obj = root.add_subsystem("obj", ExecComp("y = .3 * x"))

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


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
