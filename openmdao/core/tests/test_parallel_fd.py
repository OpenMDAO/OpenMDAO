
from __future__ import print_function

import time
import numpy as np
import unittest
TestCase = unittest.TestCase
from six import iterkeys

from openmdao.api import Group, ParallelGroup, Problem, IndepVarComp, ExplicitComponent, ExecComp, DirectSolver
from openmdao.utils.mpi import MPI
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.test_suite.parametric_suite import parametric_suite
from openmdao.test_suite.components.matmultcomp import MatMultComp

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class ScalableComp(ExplicitComponent):

    def __init__(self, size, mult=2.0, add=1.0):
        super(ScalableComp, self).__init__()

        self._size = size
        self._mult = mult
        self._add = add


    def setup(self):
        self._ncalls = 0

        self.add_input('x', np.zeros(self._size))
        self.add_output('y', np.zeros(self._size))

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """ Doesn't do much. """
        self._ncalls += 1
        outputs['y'] = inputs['x']*self._mult + self._add

    def compute_partials(self, inputs, partials):
        partials['y', 'x'] = np.eye(self._size) * self._mult


def setup_1comp_model(par_fds, size, mult, add, method):
    prob = Problem()
    prob.model.add_subsystem('P1', IndepVarComp('x', np.ones(size)))
    prob.model.add_subsystem('C1', ScalableComp(size, mult, add))

    prob.model.options['num_par_fd'] = par_fds

    prob.model.connect('P1.x', 'C1.x')

    prob.model.add_design_var('P1.x')
    prob.model.add_objective('C1.y')

    prob.model.approx_totals(method=method)

    prob.setup(mode='fwd')
    prob.run_model()

    return prob


def setup_diamond_model(par_fds, size, method, par_fd_at):
    assert par_fd_at in ('model', 'par')

    prob = Problem()
    if par_fd_at == 'model':
        prob.model.options['num_par_fd'] = par_fds
        prob.model.approx_totals(method=method)
    root = prob.model

    root.add_subsystem('P1', IndepVarComp('x', np.ones(size)))

    par = root.add_subsystem("par", Group())
    if par_fd_at == 'par':
        par.options['num_par_fd'] = par_fds
        par.approx_totals(method=method)

    par.add_subsystem('C1', ExecComp('y=2.0*x+1.0', x=np.zeros(size), y=np.zeros(size)))
    par.add_subsystem('C2', ExecComp('y=3.0*x+5.0', x=np.zeros(size), y=np.zeros(size)))
    root.add_subsystem('C3', ExecComp('y=-3.0*x1+4.0*x2+1.0', x1=np.zeros(size), x2=np.zeros(size), y=np.zeros(size)))

    root.connect("P1.x", "par.C1.x")
    root.connect("P1.x", "par.C2.x")

    root.connect("par.C1.y", "C3.x1")
    root.connect("par.C2.y", "C3.x2")

    root.add_design_var('P1.x')
    root.add_objective('C3.y')

    prob.setup(mode='fwd')
    prob.run_model()

    return prob


class SerialSimpleFDTestCase(TestCase):

    def test_serial_fd(self):
        size = 15
        mult = 2.0
        add = 1.0
        prob = setup_1comp_model(1, size, mult, add, 'fd')

        J = prob.compute_totals(['C1.y'], ['P1.x'], return_format='dict')
        assert_rel_error(self, J['C1.y']['P1.x'], np.eye(size)*mult, 1e-6)

    def test_serial_cs(self):
        size = 15
        mult = 2.0
        add = 1.0
        prob = setup_1comp_model(1, size, mult, add, 'cs')

        J = prob.compute_totals(['C1.y'], ['P1.x'], return_format='dict')
        assert_rel_error(self, J['C1.y']['P1.x'], np.eye(size)*mult, 1e-6)


class ParallelSimpleFDTestCase2(TestCase):

    N_PROCS = 2

    def test_parallel_fd2(self):
        size = 15
        mult = 2.0
        add = 1.0

        prob = setup_1comp_model(2, size, mult, add, 'fd')

        J = prob.compute_totals(['C1.y'], ['P1.x'], return_format='dict')
        assert_rel_error(self, J['C1.y']['P1.x'], np.eye(size)*mult, 1e-6)

    def test_parallel_cs2(self):
        size = 15
        mult = 2.0
        add = 1.0

        prob = setup_1comp_model(2, size, mult, add, 'cs')

        J = prob.compute_totals(['C1.y'], ['P1.x'], return_format='dict')
        assert_rel_error(self, J['C1.y']['P1.x'], np.eye(size)*mult, 1e-6)


class ParallelFDTestCase5(TestCase):

    N_PROCS = 5

    def test_parallel_fd5(self):
        size = 15
        mult = 2.0
        add = 1.0
        prob = setup_1comp_model(5, size, mult, add, 'fd')

        J = prob.compute_totals(['C1.y'], ['P1.x'], return_format='dict')
        assert_rel_error(self, J['C1.y']['P1.x'], np.eye(size)*mult, 1e-6)

    def test_parallel_cs5(self):
        size = 15
        mult = 2.0
        add = 1.0
        prob = setup_1comp_model(5, size, mult, add, 'cs')

        J = prob.compute_totals(['C1.y'], ['P1.x'], return_format='dict')
        assert_rel_error(self, J['C1.y']['P1.x'], np.eye(size)*mult, 1e-6)


class SerialDiamondFDTestCase(TestCase):

    def test_diamond_fd_totals(self):
        size = 15
        prob = setup_diamond_model(1, size, 'fd', 'model')
        assert_rel_error(self, prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.compute_totals(['C3.y'], ['P1.x'], return_format='dict')
        assert_rel_error(self, J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)

    def test_diamond_cs_totals(self):
        size = 15
        prob = setup_diamond_model(1, size, 'cs', 'model')
        assert_rel_error(self, prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.compute_totals(['C3.y'], ['P1.x'], return_format='dict')
        assert_rel_error(self, J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)

    def test_bad_num_par_fds(self):
        try:
            setup_diamond_model(0, 10, 'fd', 'model')
        except Exception as err:
            self.assertEquals(str(err), "'': num_par_fds must be >= 1 but value is 0.")


class ParallelDiamondFDTestCase(TestCase):

    N_PROCS = 4

    def test_diamond_fd_totals(self):
        size = 15
        prob = setup_diamond_model(2, size, 'fd', 'model')
        assert_rel_error(self, prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.compute_totals(['C3.y'], ['P1.x'], return_format='dict')
        assert_rel_error(self, J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)

    def test_diamond_fd_nested_par_fd_totals(self):
        size = 15
        prob = setup_diamond_model(4, size, 'fd', 'par')
        assert_rel_error(self, prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.compute_totals(['C3.y'], ['P1.x'], return_format='dict')
        assert_rel_error(self, J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)

    def test_diamond_fd_totals_num_fd_bigger_than_psize(self):
        size = 1
        prob = setup_diamond_model(2, size, 'fd', 'model')
        assert_rel_error(self, prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.compute_totals(['C3.y'], ['P1.x'], return_format='dict')
        assert_rel_error(self, J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)

    def test_diamond_cs_totals(self):
        size = 15
        prob = setup_diamond_model(2, size, 'cs', 'model')
        assert_rel_error(self, prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.compute_totals(['C3.y'], ['P1.x'], return_format='dict')
        assert_rel_error(self, J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)

    def test_diamond_cs_totals_nested_par_cs(self):
        size = 15
        prob = setup_diamond_model(4, size, 'cs', 'par')
        assert_rel_error(self, prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.compute_totals(['C3.y'], ['P1.x'], return_format='dict')
        assert_rel_error(self, J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)

    def test_diamond_cs_totals_num_fd_bigger_than_psize(self):
        size = 1
        prob = setup_diamond_model(2, size, 'cs', 'model')
        assert_rel_error(self, prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.compute_totals(['C3.y'], ['P1.x'], return_format='dict')
        assert_rel_error(self, J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)


class ParallelFDParametricTestCase(unittest.TestCase):
    N_PROCS = 2

    @parametric_suite(
        assembled_jac=[False],
        jacobian_type=['dense'],
        partial_type=['array'],
        partial_method=['fd', 'cs'],
        num_var=[3],
        var_shape=[(2, 3), (2,)],
        connection_type=['explicit'],
        run_by_default=True,
    )
    def test_subset(self, param_instance):
        param_instance.linear_solver_class = DirectSolver
        param_instance.linear_solver_options = {}  # defaults not valid for DirectSolver

        param_instance.setup()
        problem = param_instance.problem
        model = problem.model

        expected_values = model.expected_values
        if expected_values:
            actual = {key: problem[key] for key in iterkeys(expected_values)}
            assert_rel_error(self, actual, expected_values, 1e-4)

        expected_totals = model.expected_totals
        if expected_totals:
            # Forward Derivatives Check
            totals = param_instance.compute_totals('fwd')
            assert_rel_error(self, totals, expected_totals, 1e-4)

            # Reverse Derivatives Check
            totals = param_instance.compute_totals('rev')
            assert_rel_error(self, totals, expected_totals, 1e-4)


@unittest.skipUnless(PETScVector, "PETSc is required.")
class MatMultTestCase(unittest.TestCase):
    N_PROCS = 4

    def run_model(self, size, num_par_fd, method):
        if MPI:
            if MPI.COMM_WORLD.rank == 0:
                mat = np.random.random(5 * size).reshape((5, size)) - 0.5
            else:
                mat = None
            mat = MPI.COMM_WORLD.bcast(mat, root=0)
        else:
            mat = np.random.random(5 * size).reshape((5, size)) - 0.5

        p = Problem()

        model = p.model
        model.add_subsystem('indep', IndepVarComp('x', val=np.ones(mat.shape[1])))
        comp = model.add_subsystem('comp', MatMultComp(mat, approx_method=method))
        comp.options['num_par_fd'] = num_par_fd

        model.connect('indep.x', 'comp.x')

        p.setup(mode='fwd', force_alloc_complex=(method == 'cs'))
        p.run_model()

        comp.num_computes = 0

        J = p.compute_totals(of=['comp.y'], wrt=['indep.x'])

        ncomputes = comp.num_computes if comp.comm.rank == 0 else 0

        norm = np.linalg.norm(J['comp.y','indep.x'] - comp.mat)
        self.assertLess(norm, 1.e-7)
        if MPI:
            self.assertEqual(MPI.COMM_WORLD.allreduce(ncomputes), size)

    def test_20_by_4_fd(self):
        self.run_model(20, 4, 'fd')

    def test_21_by_4_fd(self):
        self.run_model(21, 4, 'fd')

    def test_21_by_2_fd(self):
        self.run_model(21, 2, 'fd')

    def test_21_by_3_fd(self):
        self.run_model(21, 3, 'fd')

    def test_22_by_3_fd(self):
        self.run_model(22, 3, 'fd')

    def test_20_by_4_cs(self):
        self.run_model(20, 4, 'cs')

    def test_21_by_4_cs(self):
        self.run_model(21, 4, 'cs')

    def test_21_by_2_cs(self):
        self.run_model(21, 2, 'cs')

    def test_21_by_3_cs(self):
        self.run_model(21, 3, 'cs')

    def test_22_by_3_cs(self):
        self.run_model(22, 3, 'cs')


@unittest.skipUnless(PETScVector, "PETSc is required.")
class MatMultParallelTestCase(unittest.TestCase):
    N_PROCS = 8

    def run_model(self, size, num_par_fd1, num_par_fd2, method, total=False):
        if MPI:
            if MPI.COMM_WORLD.rank == 0:
                mat1 = np.random.random(5 * size).reshape((5, size)) - 0.5
            else:
                mat1 = None
            mat1 = MPI.COMM_WORLD.bcast(mat1, root=0)
        else:
            mat1 = np.random.random(5 * size).reshape((5, size)) - 0.5

        mat2 = mat1 * 5.0

        p = Problem()

        #import wingdbstub

        model = p.model
        model.add_subsystem('indep', IndepVarComp('x', val=np.ones(mat1.shape[1])))
        par = model.add_subsystem('par', ParallelGroup())

        if total:
            meth = 'exact'
        else:
            meth = method
        C1 = par.add_subsystem('C1', MatMultComp(mat1, approx_method=meth))
        C2 = par.add_subsystem('C2', MatMultComp(mat2, approx_method=meth))

        if total:
            model.options['num_par_fd'] = num_par_fd1
            model.approx_totals(method=method)
        else:
            C1.options['num_par_fd'] = num_par_fd1
            C2.options['num_par_fd'] = num_par_fd2

        model.connect('indep.x', 'par.C1.x')
        model.connect('indep.x', 'par.C2.x')

        p.setup(mode='fwd', force_alloc_complex=(method == 'cs'))
        p.run_model()

        J = p.compute_totals(of=['par.C1.y', 'par.C2.y'], wrt=['indep.x'])

        norm = np.linalg.norm(J['par.C1.y','indep.x'] - C1.mat)
        self.assertLess(norm, 1.e-7)

        norm = np.linalg.norm(J['par.C2.y','indep.x'] - C2.mat)
        self.assertLess(norm, 1.e-7)

    def test_20_by_4_fd(self):
        self.run_model(20, 4, 4, 'fd')

    def test_20_by_4_3_fd(self):
        self.run_model(20, 4, 3, 'fd')

    def test_21_by_4_fd(self):
        self.run_model(21, 4, 4, 'fd')

    def test_21_by_2_fd(self):
        self.run_model(21, 2, 2, 'fd')

    def test_21_by_3_fd(self):
        self.run_model(21, 3, 3, 'fd')

    def test_22_by_3_fd(self):
        self.run_model(22, 3, 3, 'fd')

    def test_22_by_4_fd_total(self):
        self.run_model(22, 4, 4, 'fd', total=True)

    def test_22_fd_total_no_par_fd(self):
        # this tests regular FD when not all vars are local
        self.run_model(22, 1, 1, 'fd', total=True)

    def test_20_by_4_cs(self):
        self.run_model(20, 4, 4, 'cs')

    def test_20_by_4_3_cs(self):
        self.run_model(20, 4, 3, 'cs')

    def test_21_by_4_cs(self):
        self.run_model(21, 4, 4, 'cs')

    def test_21_by_2_cs(self):
        self.run_model(21, 2, 2, 'cs')

    def test_21_by_3_cs(self):
        self.run_model(21, 3, 3, 'cs')

    def test_22_by_3_cs(self):
        self.run_model(22, 3, 3, 'cs')

    def test_22_by_4_cs_total(self):
        self.run_model(22, 4, 4, 'cs', total=True)

    def test_22_cs_total_no_par_fd(self):
        # this tests regular CS when not all vars are local
        self.run_model(22, 1, 1, 'cs', total=True)


if __name__ == '__main__':
    unittest.main()

