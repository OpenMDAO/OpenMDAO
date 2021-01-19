

import time
import itertools
import numpy as np
import unittest
TestCase = unittest.TestCase

import openmdao.api as om
from openmdao.utils.mpi import MPI
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.test_suite.parametric_suite import parametric_suite
from openmdao.test_suite.components.matmultcomp import MatMultComp

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class ScalableComp(om.ExplicitComponent):

    def __init__(self, size, mult=2.0, add=1.0):
        super().__init__()

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
    prob = om.Problem(model=om.Group(num_par_fd=par_fds))
    prob.model.add_subsystem('P1', om.IndepVarComp('x', np.ones(size)))
    prob.model.add_subsystem('C1', ScalableComp(size, mult, add))

    prob.model.connect('P1.x', 'C1.x')

    prob.model.add_design_var('P1.x')
    prob.model.add_objective('C1.y')

    prob.model.approx_totals(method=method)

    prob.setup(mode='fwd')
    prob.run_model()

    return prob


def setup_diamond_model(par_fds, size, method, par_fd_at):
    assert par_fd_at in ('model', 'par')

    if par_fd_at == 'model':
        prob = om.Problem(model=om.Group(num_par_fd=par_fds))
        prob.model.approx_totals(method=method)
    else:
        prob = om.Problem()
    root = prob.model

    root.add_subsystem('P1', om.IndepVarComp('x', np.ones(size)))

    if par_fd_at == 'par':
        par = root.add_subsystem("par", om.Group(num_par_fd=par_fds))
        par.approx_totals(method=method)
    else:
        par = root.add_subsystem("par", om.Group())

    par.add_subsystem('C1', om.ExecComp('y=2.0*x+1.0', x=np.zeros(size), y=np.zeros(size)))
    par.add_subsystem('C2', om.ExecComp('y=3.0*x+5.0', x=np.zeros(size), y=np.zeros(size)))
    root.add_subsystem('C3', om.ExecComp('y=-3.0*x1+4.0*x2+1.0', x1=np.zeros(size), x2=np.zeros(size), y=np.zeros(size)))

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
        assert_near_equal(J['C1.y']['P1.x'], np.eye(size)*mult, 1e-6)

    def test_serial_cs(self):
        size = 15
        mult = 2.0
        add = 1.0
        prob = setup_1comp_model(1, size, mult, add, 'cs')

        J = prob.compute_totals(['C1.y'], ['P1.x'], return_format='dict')
        assert_near_equal(J['C1.y']['P1.x'], np.eye(size)*mult, 1e-6)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ParallelSimpleFDTestCase2(TestCase):

    N_PROCS = 2

    def test_parallel_fd2(self):
        size = 15
        mult = 2.0
        add = 1.0

        prob = setup_1comp_model(2, size, mult, add, 'fd')

        J = prob.compute_totals(['C1.y'], ['P1.x'], return_format='dict')
        assert_near_equal(J['C1.y']['P1.x'], np.eye(size)*mult, 1e-6)

    def test_parallel_cs2(self):
        size = 15
        mult = 2.0
        add = 1.0

        prob = setup_1comp_model(2, size, mult, add, 'cs')

        J = prob.compute_totals(['C1.y'], ['P1.x'], return_format='dict')
        assert_near_equal(J['C1.y']['P1.x'], np.eye(size)*mult, 1e-6)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ParallelFDTestCase5(TestCase):

    N_PROCS = 5

    def test_parallel_fd5(self):
        size = 15
        mult = 2.0
        add = 1.0
        prob = setup_1comp_model(5, size, mult, add, 'fd')

        J = prob.compute_totals(['C1.y'], ['P1.x'], return_format='dict')
        assert_near_equal(J['C1.y']['P1.x'], np.eye(size)*mult, 1e-6)

    def test_parallel_cs5(self):
        size = 15
        mult = 2.0
        add = 1.0
        prob = setup_1comp_model(5, size, mult, add, 'cs')

        J = prob.compute_totals(['C1.y'], ['P1.x'], return_format='dict')
        assert_near_equal(J['C1.y']['P1.x'], np.eye(size)*mult, 1e-6)


class SerialDiamondFDTestCase(TestCase):

    def test_diamond_fd_totals(self):
        size = 15
        prob = setup_diamond_model(1, size, 'fd', 'model')
        assert_near_equal(prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.compute_totals(['C3.y'], ['P1.x'], return_format='dict')
        assert_near_equal(J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)

    def test_diamond_cs_totals(self):
        size = 15
        prob = setup_diamond_model(1, size, 'cs', 'model')
        assert_near_equal(prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.compute_totals(['C3.y'], ['P1.x'], return_format='dict')
        assert_near_equal(J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)

    def test_bad_num_par_fds(self):
        try:
            setup_diamond_model(0, 10, 'fd', 'model')
        except Exception as err:
            self.assertEquals(str(err), "Value (0) of option 'num_par_fd' is less than minimum allowed value of 1.")


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ParallelDiamondFDTestCase(TestCase):

    N_PROCS = 4

    def test_diamond_fd_totals(self):
        size = 15
        prob = setup_diamond_model(2, size, 'fd', 'model')
        assert_near_equal(prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.compute_totals(['C3.y'], ['P1.x'], return_format='dict')
        assert_near_equal(J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)

    def test_diamond_fd_nested_par_fd_totals(self):
        size = 15
        prob = setup_diamond_model(4, size, 'fd', 'par')
        assert_near_equal(prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.compute_totals(['C3.y'], ['P1.x'], return_format='dict')
        assert_near_equal(J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)

    def test_diamond_fd_totals_num_fd_bigger_than_psize(self):
        size = 1
        prob = setup_diamond_model(2, size, 'fd', 'model')
        assert_near_equal(prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.compute_totals(['C3.y'], ['P1.x'], return_format='dict')
        assert_near_equal(J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)

    def test_diamond_cs_totals(self):
        size = 15
        prob = setup_diamond_model(2, size, 'cs', 'model')
        assert_near_equal(prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.compute_totals(['C3.y'], ['P1.x'], return_format='dict')
        assert_near_equal(J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)

    def test_diamond_cs_totals_nested_par_cs(self):
        size = 15
        prob = setup_diamond_model(4, size, 'cs', 'par')
        assert_near_equal(prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.compute_totals(['C3.y'], ['P1.x'], return_format='dict')
        assert_near_equal(J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)

    def test_diamond_cs_totals_num_fd_bigger_than_psize(self):
        size = 1
        prob = setup_diamond_model(2, size, 'cs', 'model')
        assert_near_equal(prob['C3.y'], np.ones(size)*24.0, 1e-6)

        J = prob.compute_totals(['C3.y'], ['P1.x'], return_format='dict')
        assert_near_equal(J['C3.y']['P1.x'], np.eye(size)*6.0, 1e-6)


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
class MatMultTestCase(unittest.TestCase):
    N_PROCS = 4

    @parameterized.expand(itertools.product([20, 21, 22], [2, 3, 4], ['fd', 'cs']),
                          name_func=_test_func_name)
    def test_par_fd(self, size, num_par_fd, method):
        if MPI:
            if MPI.COMM_WORLD.rank == 0:
                mat = np.random.random(5 * size).reshape((5, size)) - 0.5
            else:
                mat = None
            mat = MPI.COMM_WORLD.bcast(mat, root=0)
        else:
            mat = np.random.random(5 * size).reshape((5, size)) - 0.5

        p = om.Problem()

        model = p.model

        model.add_subsystem('indep', om.IndepVarComp('x', val=np.ones(mat.shape[1])))
        comp = model.add_subsystem('comp', MatMultComp(mat, approx_method=method, num_par_fd=num_par_fd))

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

        norm = np.linalg.norm(comp._jacobian['y', 'x'] - comp.mat)
        self.assertLess(norm, 1.e-7)

        # make sure check_partials works
        data = p.check_partials(out_stream=None)
        norm = np.linalg.norm(data['comp']['y', 'x']['J_fd'] - comp.mat)
        self.assertLess(norm, 1.e-7)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
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

        if total:
            grp = om.Group(num_par_fd=num_par_fd1)
        else:
            grp = om.Group()
        p = om.Problem(model=grp)

        model = p.model
        model.add_subsystem('indep', om.IndepVarComp('x', val=np.ones(mat1.shape[1])))
        par = model.add_subsystem('par', om.ParallelGroup())

        if total:
            C1 = par.add_subsystem('C1', MatMultComp(mat1, approx_method='exact'))
            C2 = par.add_subsystem('C2', MatMultComp(mat2, approx_method='exact'))
            model.approx_totals(method=method)
        else:
            C1 = par.add_subsystem('C1', MatMultComp(mat1, approx_method=method, num_par_fd=num_par_fd1))
            C2 = par.add_subsystem('C2', MatMultComp(mat2, approx_method=method, num_par_fd=num_par_fd2))

        model.connect('indep.x', 'par.C1.x')
        model.connect('indep.x', 'par.C2.x')

        p.setup(mode='fwd', force_alloc_complex=(method == 'cs'))
        p.run_model()

        J = p.compute_totals(of=['par.C1.y', 'par.C2.y'], wrt=['indep.x'])

        norm = np.linalg.norm(J['par.C1.y','indep.x'] - C1.mat)
        self.assertLess(norm, 1.e-7)

        norm = np.linalg.norm(J['par.C2.y','indep.x'] - C2.mat)
        self.assertLess(norm, 1.e-7)

        if not total:
            # if total is True, the partials won't be computed during the compute_totals call
            if C1 in par._subsystems_myproc:
                norm = np.linalg.norm(C1._jacobian['y', 'x'] - C1.mat)
                self.assertLess(norm, 1.e-7)

            if C2 in par._subsystems_myproc:
                norm = np.linalg.norm(C2._jacobian['y', 'x'] - C2.mat)
                self.assertLess(norm, 1.e-7)

        # make sure check_partials works
        data = p.check_partials(out_stream=None)
        if 'par.C1' in data:
            norm = np.linalg.norm(data['par.C1']['y', 'x']['J_fd'] - C1.mat)
            self.assertLess(norm, 1.e-7)

        if 'par.C2' in data:
            norm = np.linalg.norm(data['par.C2']['y', 'x']['J_fd'] - C2.mat)
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


def _setup_problem(mat, total_method='exact', partial_method='exact', total_num_par_fd=1,
                   partial_num_par_fd=1, approx_totals=False):
    p = om.Problem(model=om.Group(num_par_fd=total_num_par_fd))
    model = p.model
    model.add_subsystem('indep', om.IndepVarComp('x', val=np.ones(mat.shape[1])))
    model.add_subsystem('comp', MatMultComp(mat, approx_method=partial_method,
                        num_par_fd=partial_num_par_fd))

    model.connect('indep.x', 'comp.x')

    if approx_totals:
        p.model.approx_totals()

    p.setup(mode='fwd', force_alloc_complex='cs' in (total_method, partial_method))
    return p


class ParFDWarningsTestCase(unittest.TestCase):
    def setUp(self):
        size = 20
        self.mat = np.random.random(5 * size).reshape((5, size)) - 0.5

    def test_total_no_mpi(self):
        msg = "<model> <class Group>: MPI is not active but num_par_fd = 3. No parallel finite difference will be performed."

        with assert_warning(UserWarning, msg):
            _setup_problem(self.mat, total_method='fd', total_num_par_fd = 3, approx_totals=True)

    def test_partial_no_mpi(self):
        msg = "'comp' <class MatMultComp>: MPI is not active but num_par_fd = 3. No parallel finite difference will be performed."

        with assert_warning(UserWarning, msg):
            _setup_problem(self.mat, partial_method='fd', partial_num_par_fd = 3)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ParFDErrorsMPITestCase(unittest.TestCase):
    N_PROCS = 3

    def setUp(self):
        size = 20
        self.mat = np.random.random(5 * size).reshape((5, size)) - 0.5

    def test_no_approx_totals(self):
        with self.assertRaises(RuntimeError) as ctx:
            _setup_problem(self.mat, total_method='fd', total_num_par_fd = 3, approx_totals=False)

        self.assertEqual(str(ctx.exception), "<model> <class Group>: num_par_fd = 3 but FD is not active.")

    def test_no_partial_approx(self):
        p = _setup_problem(self.mat, partial_num_par_fd = 3, approx_totals=False)
        with self.assertRaises(RuntimeError) as ctx:
            p.final_setup()

        self.assertEqual(str(ctx.exception), "'comp' <class MatMultComp>: num_par_fd is > 1 but no FD is active.")


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class ParFDFeatureTestCase(unittest.TestCase):
    N_PROCS = 3

    def test_fd_totals(self):
        mat = np.arange(30, dtype=float).reshape(5, 6)

        p = om.Problem(model=om.Group(num_par_fd=3))
        model = p.model
        model.approx_totals(method='fd')
        comp = model.add_subsystem('comp', MatMultComp(mat))

        model.set_input_defaults('comp.x', val=np.ones(mat.shape[1]))
        p.setup(mode='fwd')
        p.run_model()

        pre_count = comp.num_computes

        J = p.compute_totals(of=['comp.y'], wrt=['comp.x'], return_format='array')

        post_count =  comp.num_computes

        # how many computes were used in this proc to compute the total jacobian?
        # Each proc should be doing 2 computes.
        jac_count = post_count - pre_count

        self.assertEqual(jac_count, 2)

        # J and mat should be the same
        self.assertLess(np.linalg.norm(J - mat), 1.e-7)

    def test_fd_partials(self):
        mat = np.arange(30, dtype=float).reshape(5, 6)

        p = om.Problem()
        model = p.model
        comp = model.add_subsystem('comp', MatMultComp(mat, approx_method='fd', num_par_fd=3))

        model.set_input_defaults('comp.x', val=np.ones(mat.shape[1]))
        p.setup(mode='fwd')
        p.run_model()

        pre_count = comp.num_computes

        # calling compute_totals will result in the computation of partials for comp
        p.compute_totals(of=['comp.y'], wrt=['comp.x'])

        # get the partial jacobian matrix
        J = comp._jacobian['y', 'x']

        post_count =  comp.num_computes

        # how many computes were used in this proc to compute the partial jacobian?
        # Each proc should be doing 2 computes.
        jac_count = post_count - pre_count

        self.assertEqual(jac_count, 2)

        # J and mat should be the same
        self.assertLess(np.linalg.norm(J - mat), 1.e-7)

if __name__ == '__main__':
    unittest.main()

