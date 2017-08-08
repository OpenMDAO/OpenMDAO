""" Test out some specialized parallel derivatives features"""

from __future__ import print_function

import unittest
import numpy as np

from openmdao.api import Group, ParallelGroup, Problem, IndepVarComp, LinearBlockGS, DefaultVector, \
    ExecComp
from openmdao.utils.mpi import MPI
from openmdao.test_suite.groups.parallel_groups import FanOutGrouped, FanInGrouped
from openmdao.devtools.testutil import assert_rel_error

if MPI:
    from openmdao.api import PETScVector
    vector_class = PETScVector
else:
    vector_class = DefaultVector


class ParDerivTestCase(unittest.TestCase):

    N_PROCS = 2

    def test_fan_in_serial_sets_rev(self):

        prob = Problem()
        prob.model = FanInGrouped()
        prob.model.linear_solver = LinearBlockGS()
        prob.model.get_subsystem('sub').linear_solver = LinearBlockGS()

        prob.model.add_design_var('iv.x1')
        prob.model.add_design_var('iv.x2')
        prob.model.add_objective('c3.y')

        prob.setup(vector_class=vector_class, check=False, mode='rev')
        prob.run_driver()

        indep_list = ['iv.x1', 'iv.x2']
        unknown_list = ['c3.y']

        J = prob.compute_total_derivs(unknown_list, indep_list, return_format='dict')
        assert_rel_error(self, J['c3.y']['iv.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y']['iv.x2'][0][0], 35.0, 1e-6)

    def test_fan_in_serial_sets_fwd(self):

        prob = Problem()
        prob.model = FanInGrouped()
        prob.model.linear_solver = LinearBlockGS()
        prob.model.get_subsystem('sub').linear_solver = LinearBlockGS()

        prob.model.add_design_var('iv.x1')
        prob.model.add_design_var('iv.x2')
        prob.model.add_objective('c3.y')

        prob.setup(vector_class=vector_class, check=False, mode='fwd')
        prob.run_driver()

        indep_list = ['iv.x1', 'iv.x2']
        unknown_list = ['c3.y']

        J = prob.compute_total_derivs(unknown_list, indep_list, return_format='flat_dict')
        assert_rel_error(self, J['c3.y', 'iv.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x2'][0][0], 35.0, 1e-6)

    def test_fan_out_serial_sets_fwd(self):

        prob = Problem()
        prob.model = FanOutGrouped()
        prob.model.linear_solver = LinearBlockGS()
        prob.model.get_subsystem('sub').linear_solver = LinearBlockGS()

        prob.model.add_design_var('iv.x')
        prob.model.add_constraint('c2.y', upper=0.0)
        prob.model.add_constraint('c3.y', upper=0.0)

        prob.setup(vector_class=vector_class, check=False, mode='fwd')
        prob.run_driver()

        unknown_list = ['c2.y', 'c3.y']
        indep_list = ['iv.x']

        J = prob.compute_total_derivs(unknown_list, indep_list, return_format='flat_dict')
        assert_rel_error(self, J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

    def test_fan_out_serial_sets_rev(self):

        prob = Problem()
        prob.model = FanOutGrouped()
        prob.model.linear_solver = LinearBlockGS()
        prob.model.get_subsystem('sub').linear_solver = LinearBlockGS()

        prob.model.add_design_var('iv.x')
        prob.model.add_constraint('c2.y', upper=0.0)
        prob.model.add_constraint('c3.y', upper=0.0)

        prob.setup(vector_class=vector_class, check=False, mode='rev')
        prob.run_driver()

        unknown_list = ['c2.y', 'c3.y']
        indep_list = ['iv.x']

        J = prob.compute_total_derivs(unknown_list, indep_list, return_format='flat_dict')
        assert_rel_error(self, J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

    def test_fan_in_parallel_sets_fwd(self):

        prob = Problem()
        prob.model = FanInGrouped()
        prob.model.linear_solver = LinearBlockGS()
        prob.model.get_subsystem('sub').linear_solver = LinearBlockGS()

        prob.model.add_design_var('iv.x1', parallel_deriv_color='par_dv')
        prob.model.add_design_var('iv.x2', parallel_deriv_color='par_dv')
        prob.model.add_design_var('iv.x3')
        prob.model.add_objective('c3.y')

        prob.setup(vector_class=vector_class, check=False, mode='fwd')
        prob.run_driver()

        indep_list = ['iv.x1', 'iv.x2']
        unknown_list = ['c3.y']

        J = prob.compute_total_derivs(unknown_list, indep_list, return_format='flat_dict')
        assert_rel_error(self, J['c3.y', 'iv.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x2'][0][0], 35.0, 1e-6)

    def test_fan_out_parallel_sets_rev(self):

        prob = Problem()
        prob.model = FanOutGrouped()
        prob.model.linear_solver = LinearBlockGS()
        prob.model.get_subsystem('sub').linear_solver = LinearBlockGS()

        prob.model.add_design_var('iv.x')
        prob.model.add_constraint('c2.y', upper=0.0, parallel_deriv_color='par_resp')
        prob.model.add_constraint('c3.y', upper=0.0, parallel_deriv_color='par_resp')

        prob.setup(vector_class=vector_class, check=False, mode='rev')
        prob.run_driver()

        unknown_list = ['sub.c2.y', 'sub.c3.y']
        indep_list = ['iv.x']

        J = prob.compute_total_derivs(unknown_list, indep_list, return_format='flat_dict')
        assert_rel_error(self, J['sub.c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['sub.c3.y', 'iv.x'][0][0], 15.0, 1e-6)


class DecoupledTestCase(unittest.TestCase):
    N_PROCS = 2
    asize = 3

    def setup_model(self):
        asize = self.asize
        prob = Problem()
        #import wingdbstub
        root = prob.model
        root.linear_solver = LinearBlockGS()

        p1 = root.add_subsystem('p1', IndepVarComp('x', np.arange(asize, dtype=float)+1.0))
        p2 = root.add_subsystem('p2', IndepVarComp('x', np.arange(asize+2, dtype=float)+1.0))
        G1 = root.add_subsystem('G1', ParallelGroup())
        G1.linear_solver = LinearBlockGS()

        c1 = G1.add_subsystem('c1', ExecComp('y = numpy.ones(3).T*x.dot(numpy.arange(3.,6.))',
                                                  x=np.zeros(asize), y=np.zeros(asize)))
        c2 = G1.add_subsystem('c2', ExecComp('y = x[:%d] * 2.0' % asize,
                                        x=np.zeros(asize+2), y=np.zeros(asize)))

        c3 = root.add_subsystem('c3', ExecComp('y = x * 5.0',
                                          x=np.zeros(asize), y=np.zeros(asize)))
        c4 = root.add_subsystem('c4', ExecComp('y = x * 4.0',
                                          x=np.zeros(asize), y=np.zeros(asize)))
        root.connect('p1.x', 'G1.c1.x')
        root.connect('p2.x', 'G1.c2.x')
        root.connect('G1.c1.y', 'c3.x')
        root.connect('G1.c2.y', 'c4.x')

        return prob

    def test_serial_fwd(self):
        asize = self.asize
        prob = self.setup_model()

        prob.model.add_design_var('p1.x')
        prob.model.add_design_var('p2.x')
        prob.model.add_constraint('c3.y', upper=0.0)
        prob.model.add_constraint('c4.y', upper=0.0)

        prob.setup(vector_class=vector_class, check=False, mode='fwd')
        prob.run_driver()

        J = prob.compute_total_derivs(['c3.y', 'c4.y'], ['p1.x', 'p2.x'],
                                      return_format='flat_dict')

        assert_rel_error(self, J['c3.y', 'p1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.zeros((asize, asize+2))
        expected[:,:asize] = np.eye(asize)*8.0
        assert_rel_error(self, J['c4.y', 'p2.x'], expected, 1e-6)

    def test_parallel_fwd(self):
        asize = self.asize
        prob = self.setup_model()

        #import wingdbstub

        prob.model.add_design_var('p1.x', parallel_deriv_color='pardv', vectorize_derivs=True)
        prob.model.add_design_var('p2.x', parallel_deriv_color='pardv', vectorize_derivs=True)
        prob.model.add_constraint('c3.y', upper=0.0)
        prob.model.add_constraint('c4.y', upper=0.0)

        prob.setup(vector_class=vector_class, check=False, mode='fwd')
        prob.run_driver()

        J = prob.compute_total_derivs(['c3.y', 'c4.y'], ['p1.x', 'p2.x'],
                                      return_format='flat_dict')

        assert_rel_error(self, J['c3.y', 'p1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.zeros((asize, asize+2))
        expected[:,:asize] = np.eye(asize)*8.0
        assert_rel_error(self, J['c4.y', 'p2.x'], expected, 1e-6)

    def test_serial_rev(self):
        asize = self.asize
        prob = self.setup_model()

        prob.model.add_design_var('p1.x')
        prob.model.add_design_var('p2.x')
        prob.model.add_constraint('c3.y', upper=0.0)
        prob.model.add_constraint('c4.y', upper=0.0)

        prob.setup(vector_class=vector_class, check=False, mode='rev')
        prob.run_driver()

        J = prob.compute_total_derivs(['c3.y', 'c4.y'], ['p1.x', 'p2.x'],
                                      return_format='flat_dict')

        assert_rel_error(self, J['c3.y', 'p1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.zeros((asize, asize+2))
        expected[:,:asize] = np.eye(asize)*8.0
        assert_rel_error(self, J['c4.y', 'p2.x'], expected, 1e-6)

    def test_parallel_rev(self):
        asize = self.asize
        prob = self.setup_model()

        prob.model.add_design_var('p1.x')
        prob.model.add_design_var('p2.x')
        prob.model.add_constraint('c3.y', upper=0.0, parallel_deriv_color='parc')
        prob.model.add_constraint('c4.y', upper=0.0, parallel_deriv_color='parc')

        prob.setup(vector_class=vector_class, check=False, mode='rev')
        prob.run_driver()

        J = prob.compute_total_derivs(['c3.y', 'c4.y'], ['p1.x', 'p2.x'],
                                      return_format='flat_dict')

        assert_rel_error(self, J['c3.y', 'p1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.zeros((asize, asize+2))
        expected[:,:asize] = np.eye(asize)*8.0
        assert_rel_error(self, J['c4.y', 'p2.x'], expected, 1e-6)

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
        c3 = G1.add_subsystem('c3', ExecComp('y = numpy.ones(3).T*x.dot(numpy.arange(3.,6.))',
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

        J = prob.compute_total_derivs(['c4.y', 'c5.y'], ['p.x'],
                                      return_format='flat_dict')

        assert_rel_error(self, J['c5.y', 'p.x'][0], np.array([20., 25.]), 1e-6)
        assert_rel_error(self, J['c4.y', 'p.x'][0], np.array([8., 0.]), 1e-6)

    def test_indices_rev(self):
        prob = self.setup_model('rev')
        J = prob.compute_total_derivs(['c4.y', 'c5.y'], ['p.x'],
                                      return_format='flat_dict')

        assert_rel_error(self, J['c5.y', 'p.x'][0], np.array([20., 25.]), 1e-6)
        assert_rel_error(self, J['c4.y', 'p.x'][0], np.array([8., 0.]), 1e-6)


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
        c3 = par2.add_subsystem('c3', ExecComp('y = numpy.ones(3).T*x.dot(numpy.arange(3.,6.))',
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

        J = prob.compute_total_derivs(of=['G1.par1.c4.y', 'G1.par2.c5.y'],
                                      wrt=['G1.par1.p.x', 'G1.par2.p.x'],
                                      return_format='flat_dict')

        assert_rel_error(self, J['G1.par2.c5.y', 'G1.par2.p.x'][0], np.array([20., 25.]), 1e-6)
        assert_rel_error(self, J['G1.par1.c4.y', 'G1.par1.p.x'][0], np.array([8., 0.]), 1e-6)

    def test_indices_rev(self):
        prob = self.setup_model('rev')
        J = prob.compute_total_derivs(['G1.par1.c4.y', 'G1.par2.c5.y'],
                                      ['G1.par1.p.x', 'G1.par2.p.x'],
                                      return_format='flat_dict')

        assert_rel_error(self, J['G1.par2.c5.y', 'G1.par2.p.x'][0], np.array([20., 25.]), 1e-6)
        assert_rel_error(self, J['G1.par1.c4.y', 'G1.par1.p.x'][0], np.array([8., 0.]), 1e-6)


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

        c1 = G1.add_subsystem('c1', ExecComp('y = numpy.ones(3).T*x.dot(numpy.arange(3.,6.))',
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

        J = prob.compute_total_derivs(['c3.y', 'c4.y'], ['p1.x', 'p2.x'],
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

        J = prob.compute_total_derivs(['c3.y', 'c4.y'], ['p1.x', 'p2.x'],
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

        J = prob.compute_total_derivs(['c3.y', 'c4.y'], ['p1.x', 'p2.x'],
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

        J = prob.compute_total_derivs(['c3.y', 'c4.y'], ['p1.x', 'p2.x'],
                                      return_format='flat_dict')

        assert_rel_error(self, J['c3.y', 'p1.x'], np.array([[15., 20., 25.],[15., 20., 25.], [15., 20., 25.]]), 1e-6)
        expected = np.eye(asize)*8.0
        assert_rel_error(self, J['c4.y', 'p2.x'], expected, 1e-6)


if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
