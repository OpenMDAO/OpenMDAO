""" Test the Jacobian objects."""

import itertools
import unittest
from parameterized import parameterized

from six import assertRaisesRegex
from six.moves import range

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from openmdao.api import IndepVarComp, Group, Problem, \
                         ExplicitComponent, ImplicitComponent, ExecComp, \
                         NewtonSolver, ScipyKrylov, \
                         LinearBlockGS, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, \
     SellarDis2withDerivatives


class MyExplicitComp(ExplicitComponent):
    def __init__(self, jac_type):
        super(MyExplicitComp, self).__init__()
        self._jac_type = jac_type

    def setup(self):
        self.add_input('x', val=np.zeros(2))
        self.add_input('y', val=np.zeros(2))
        self.add_output('f', val=np.zeros(2))

        val = self._jac_type(np.array([[1., 1.], [1., 1.]]))
        if isinstance(val, list):
            self.declare_partials('f', ['x','y'], rows=val[1], cols=val[2], val=val[0])
        else:
            self.declare_partials('f', ['x','y'], val=val)

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        outputs['f'][0] = (x[0]-3.0)**2 + x[0]*x[1] + (x[1]+4.0)**2 - 3.0 + \
                           y[0]*17. - y[0]*y[1] + 2.*y[1]
        outputs['f'][1] = outputs['f'][0]*3.0

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        y = inputs['y']
        jac1 = self._jac_type(np.array([
            [2.0*x[0] - 6.0 + x[1], 2.0*x[1] + 8.0 + x[0]],
            [(2.0*x[0] - 6.0 + x[1])*3., (2.0*x[1] + 8.0 + x[0])*3.]
        ]))

        if isinstance(jac1, list):
            jac1 = jac1[0]


        partials['f', 'x'] = jac1

        jac2 = self._jac_type(np.array([
            [17.-y[1], 2.-y[0]],
            [(17.-y[1])*3., (2.-y[0])*3.]
        ]))

        if isinstance(jac2, list):
            jac2 = jac2[0]

        partials['f', 'y'] = jac2


class MyExplicitComp2(ExplicitComponent):
    def __init__(self, jac_type):
        super(MyExplicitComp2, self).__init__()
        self._jac_type = jac_type

    def setup(self):
        self.add_input('w', val=np.zeros(3))
        self.add_input('z', val=0.0)
        self.add_output('f', val=0.0)

        val = self._jac_type(np.array([[7.]]))
        if isinstance(val, list):
            self.declare_partials('f', 'z', rows=val[1], cols=val[2], val=val[0])
        else:
            self.declare_partials('f', 'z', val=val)

        val = self._jac_type(np.array([[1., 1., 1.]]))
        if isinstance(val, list):
            self.declare_partials('f', 'w', rows=val[1], cols=val[2], val=val[0])
        else:
            self.declare_partials('f', 'w', val=val)

    def compute(self, inputs, outputs):
        w = inputs['w']
        z = inputs['z']
        outputs['f'] = (w[0]-5.0)**2 + (w[1]+1.0)**2 + w[2]*6. + z*7.

    def compute_partials(self, inputs, partials):
        w = inputs['w']
        z = inputs['z']
        jac = self._jac_type(np.array([[
            2.0*w[0] - 10.0,
            2.0*w[1] + 2.0,
            6.
        ]]))

        if isinstance(jac, list):
            jac = jac[0]

        partials['f', 'w'] = jac


class ExplicitSetItemComp(ExplicitComponent):
    def __init__(self, dtype, value, shape, constructor):
        self._dtype = dtype
        self._shape = shape
        self._value = value
        self._constructor = constructor
        super(ExplicitSetItemComp, self).__init__()

    def setup(self):
        if self._shape == 'scalar':
            in_val = 1
            out_val = 1
        elif self._shape == '1D_array':
            in_val = np.array([1])
            out_val = np.array([1, 2, 3, 4, 5])
        elif self._shape == '2D_array':
            in_val = np.array([1, 2, 3])
            out_val = np.array([1, 2, 3])

        if self._dtype == 'int':
            scale = 1
        elif self._dtype == 'float':
            scale = 1.
        elif self._dtype == 'complex':
            scale = 1j

        self.add_input('in', val=in_val*scale)
        self.add_output('out', val=out_val*scale)

        self.declare_partials(of='*', wrt='*')

    def compute_partials(self, inputs, partials):
        partials['out', 'in'] = self._constructor(self._value)


def arr2list(arr):
    """Convert a numpy array to a 'sparse' list."""
    data = []
    rows = []
    cols = []

    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            rows.append(row)
            cols.append(col)
            data.append(arr[row, col])

    return [np.array(data), np.array(rows), np.array(cols)]

def arr2revlist(arr):
    """Convert a numpy array to a 'sparse' list in reverse order."""
    lst = arr2list(arr)
    return [lst[0][::-1], lst[1][::-1], lst[2][::-1]]

def inverted_coo(arr):
    """Convert an ordered coo matrix into one with columns in reverse order
    so we can test unsorted coo matrices.
    """
    shape = arr.shape
    arr = coo_matrix(arr)
    return coo_matrix((arr.data[::-1], (arr.row[::-1], arr.col[::-1])), shape=shape)

def inverted_csr(arr):
    """Convert an ordered coo matrix into a csr with columns in reverse order
    so we can test unsorted csr matrices.
    """
    return inverted_coo(arr).tocsr()


def _test_func_name(func, num, param):
    args = []
    for p in param.args:
        try:
            arg = p.__name__
        except:
            arg = str(p)
        args.append(arg)
    return 'test_jacobian_src_indices_' + '_'.join(args)


class TestJacobian(unittest.TestCase):

    @parameterized.expand(itertools.product(
        ['dense', 'csc'],
        [np.array, coo_matrix, csr_matrix, inverted_coo, inverted_csr, arr2list, arr2revlist],
        [False, True],  # not nested, nested
        [0, 1],  # extra calls to linearize
        ), testcase_func_name=_test_func_name
    )
    def test_src_indices(self, assembled_jac, comp_jac_class, nested, lincalls):

        self._setup_model(assembled_jac, comp_jac_class, nested, lincalls)

        # if we multiply our jacobian (at x,y = ones) by our work vec of 1's,
        # we get fwd_check
        fwd_check = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -24., -74., -8.])

        # if we multiply our jacobian's transpose by our work vec of 1's,
        # we get rev_check
        rev_check = np.array([-35., -5., 9., -63., -3., 1., -6., 1.])

        self._check_fwd(self.prob, fwd_check)
        # to catch issues with constant subjacobians, repeatedly call linearize
        for i in range(lincalls):
            self.prob.model.run_linearize()
        self._check_fwd(self.prob, fwd_check)
        self._check_rev(self.prob, rev_check)

    def _setup_model(self, assembled_jac, comp_jac_class, nested, lincalls):
        self.prob = prob = Problem(model=Group())
        if nested:
            top = prob.model.add_subsystem('G1', Group())
        else:
            top = prob.model

        indep = top.add_subsystem('indep', IndepVarComp())
        indep.add_output('a', val=np.ones(3))
        indep.add_output('b', val=np.ones(2))

        top.add_subsystem('C1', MyExplicitComp(comp_jac_class))
        top.add_subsystem('C2', MyExplicitComp2(comp_jac_class))
        top.connect('indep.a', 'C1.x', src_indices=[2,0])
        top.connect('indep.b', 'C1.y')
        top.connect('indep.a', 'C2.w', src_indices=[0,2,1])
        top.connect('C1.f', 'C2.z', src_indices=[1])

        top.nonlinear_solver = NewtonSolver()
        top.nonlinear_solver.linear_solver = ScipyKrylov(maxiter=100)
        top.linear_solver = ScipyKrylov(
            maxiter=200, atol=1e-10, rtol=1e-10, assemble_jac=True)
        top.options['assembled_jac_type'] = assembled_jac

        prob.set_solver_print(level=0)

        prob.setup(check=False)

        prob.run_model()

    def _check_fwd(self, prob, check_vec):
        d_inputs, d_outputs, d_residuals = prob.model.get_linear_vectors()

        work = d_outputs._clone()
        work.set_const(1.0)

        # fwd apply_linear test
        d_outputs.set_const(1.0)
        prob.model.run_apply_linear(['linear'], 'fwd')
        d_residuals.set_data(d_residuals.get_data() - check_vec)
        self.assertAlmostEqual(d_residuals.get_norm(), 0)

        # fwd solve_linear test
        d_outputs.set_const(0.0)
        d_residuals.set_data(check_vec)

        prob.model.run_solve_linear(['linear'], 'fwd')

        d_outputs -= work
        self.assertAlmostEqual(d_outputs.get_norm(), 0, delta=1e-6)

    def _check_rev(self, prob, check_vec):
        d_inputs, d_outputs, d_residuals = prob.model.get_linear_vectors()

        work = d_outputs._clone()
        work.set_const(1.0)

        # rev apply_linear test
        d_residuals.set_const(1.0)
        prob.model.run_apply_linear(['linear'], 'rev')
        d_outputs.set_data(d_outputs.get_data() - check_vec)
        self.assertAlmostEqual(d_outputs.get_norm(), 0)

        # rev solve_linear test
        d_residuals.set_const(0.0)
        d_outputs.set_data(check_vec)
        prob.model.run_solve_linear(['linear'], 'rev')
        d_residuals -= work
        self.assertAlmostEqual(d_residuals.get_norm(), 0, delta=1e-6)

    dtypes = [
        ('int', 1),
        ('float', 2.1),
        # ('complex', 3.2 + 1.1j), # TODO: enable when Vectors support complex entries.
    ]

    shapes = [
        ('scalar', lambda x: x, (1, 1)),
        ('1D_array', lambda x: np.array([x + i for i in range(5)]), (5, 1)),
        ('2D_array', lambda x: np.array([[x + i + 2 * j for i in range(3)] for j in range(3)]),
         (3, 3))
    ]

    @parameterized.expand(itertools.product(dtypes, shapes), testcase_func_name=
    lambda f, n, p: '_'.join(['test_jacobian_set_item', p.args[0][0], p.args[1][0]]))
    def test_jacobian_set_item(self, dtypes, shapes):

        shape, constructor, expected_shape = shapes
        dtype, value = dtypes

        prob = Problem(model=Group())
        comp = ExplicitSetItemComp(dtype, value, shape, constructor)
        prob.model.add_subsystem('C1', comp)
        prob.setup(check=False)

        prob.set_solver_print(level=0)
        prob.run_model()
        prob.model.run_apply_nonlinear()
        prob.model.run_linearize()

        expected = constructor(value)
        J = prob.model._subsystems_allprocs[0]._jacobian
        jac_out = J['out', 'in'] * -1

        self.assertEqual(len(jac_out.shape), 2)
        expected_dtype = np.promote_types(dtype, float)
        self.assertEqual(jac_out.dtype, expected_dtype)
        assert_rel_error(self, jac_out, np.atleast_2d(expected).reshape(expected_shape), 1e-15)

    def test_group_assembled_jac_with_ext_mat(self):

        class TwoSellarDis1(ExplicitComponent):
            """
            Component containing Discipline 1 -- no derivatives version.
            """
            def setup(self):
                self.add_input('z', val=np.zeros(2))
                self.add_input('x', val=np.zeros(2))
                self.add_input('y2', val=np.ones(2))
                self.add_output('y1', val=np.ones(2))

                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                z1 = inputs['z'][0]
                z2 = inputs['z'][1]
                x1 = inputs['x']
                y2 = inputs['y2']

                outputs['y1'][0] = z1**2 + z2 + x1[0] - 0.2*y2[0]
                outputs['y1'][1] = z1**2 + z2 + x1[0] - 0.2*y2[0]

            def compute_partials(self, inputs, partials):
                """
                Jacobian for Sellar discipline 1.
                """
                partials['y1', 'y2'] =np.array([[-0.2, 0.], [0., -0.2]])
                partials['y1', 'z'] = np.array([[2.0 * inputs['z'][0], 1.0], [2.0 * inputs['z'][0], 1.0]])
                partials['y1', 'x'] = np.eye(2)


        class TwoSellarDis2(ExplicitComponent):
            def setup(self):
                self.add_input('z', val=np.zeros(2))
                self.add_input('y1', val=np.ones(2))
                self.add_output('y2', val=np.ones(2))

                self.declare_partials('*', '*', method='fd')

            def compute(self, inputs, outputs):

                z1 = inputs['z'][0]
                z2 = inputs['z'][1]
                y1 = inputs['y1']

                # Note: this may cause some issues. However, y1 is constrained to be
                # above 3.16, so lets just let it converge, and the optimizer will
                # throw it out
                if y1[0].real < 0.0:
                    y1[0] *= -1
                if y1[1].real < 0.0:
                    y1[1] *= -1

                outputs['y2'][0] = y1[0]**.5 + z1 + z2
                outputs['y2'][1] = y1[1]**.5 + z1 + z2

            def compute_partials(self, inputs, J):
                y1 = inputs['y1']
                if y1[0].real < 0.0:
                    y1[0] *= -1
                if y1[1].real < 0.0:
                    y1[1] *= -1

                J['y2', 'y1'] = np.array([[.5*y1[0]**-.5, 0.], [0., .5*y1[1]**-.5]])
                J['y2', 'z'] = np.array([[1.0, 1.0], [1.0, 1.0]])


        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('px', IndepVarComp('x', np.array([1.0, 1.0])), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])
        sup = model.add_subsystem('sup', Group(), promotes=['*'])

        sub1 = sup.add_subsystem('sub1', Group(), promotes=['*'])
        sub2 = sup.add_subsystem('sub2', Group(), promotes=['*'])

        d1 = sub1.add_subsystem('d1', TwoSellarDis1(), promotes=['x', 'z', 'y1', 'y2'])
        sub2.add_subsystem('d2', TwoSellarDis2(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1[0] - y1[1]', y1=np.array([0.0, 0.0])),
                            promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2[0] + y2[1] - 24.0', y2=np.array([0.0, 0.0])),
                            promotes=['con2', 'y2'])

        model.linear_solver = LinearBlockGS()
        sup.linear_solver = LinearBlockGS()

        sub1.linear_solver = DirectSolver(assemble_jac=True)
        sub2.linear_solver = DirectSolver(assemble_jac=True)
        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        of = ['con1', 'con2']
        wrt = ['x', 'z']

        # Make sure we don't get a size mismatch.
        derivs = prob.compute_totals(of=of, wrt=wrt)

    def test_assembled_jac_bad_key(self):
        # this test fails if AssembledJacobian._update sets in_start with 'output' instead of 'input'
        prob = Problem()
        prob.model = Group(assembled_jac_type='dense')
        prob.model.add_subsystem('indep', IndepVarComp('x', 1.0))
        prob.model.add_subsystem('C1', ExecComp('c=a*2.0+b'))
        c2 = prob.model.add_subsystem('C2', ExecComp('d=a*2.0+b+c'))
        c3 = prob.model.add_subsystem('C3', ExecComp('ee=a*2.0'))

        prob.model.nonlinear_solver = NewtonSolver()
        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.connect('indep.x', 'C1.a')
        prob.model.connect('indep.x', 'C2.a')
        prob.model.connect('C1.c', 'C2.b')
        prob.model.connect('C2.d', 'C3.a')
        prob.set_solver_print(level=0)
        prob.setup(check=False)
        prob.run_model()
        assert_rel_error(self, prob['C3.ee'], 8.0, 0000.1)

    def test_assembled_jacobian_submat_indexing_dense(self):
        prob = Problem(model=Group(assembled_jac_type='dense'))
        indeps = prob.model.add_subsystem('indeps', IndepVarComp())
        indeps.add_output('x', 1.0)
        indeps.add_output('y', 5.0)
        indeps.add_output('z', 9.0)

        G1 = prob.model.add_subsystem('G1', Group())
        G1.add_subsystem('C1', ExecComp('y=2.0*x*x'))
        G1.add_subsystem('C2', ExecComp('y=3.0*x*x'))

        prob.model.nonlinear_solver = NewtonSolver()
        G1.linear_solver = DirectSolver(assemble_jac=True)

        # before the fix, we got bad offsets into the _ext_mtx matrix.
        # to get entries in _ext_mtx, there must be at least one connection
        # to an input in the system that owns the AssembledJacobian, from
        # a source that is outside of that system. In this case, the 'indeps'
        # system is outside of the 'G1' group which owns the AssembledJacobian.
        prob.model.connect('indeps.y', 'G1.C1.x')
        prob.model.connect('indeps.z', 'G1.C2.x')

        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['G1.C1.y'], 50.0)
        assert_rel_error(self, prob['G1.C2.y'], 243.0)

    def test_assembled_jacobian_submat_indexing_csc(self):
        prob = Problem(model=Group(assembled_jac_type='dense'))
        indeps = prob.model.add_subsystem('indeps', IndepVarComp())
        indeps.add_output('x', 1.0)
        indeps.add_output('y', 5.0)
        indeps.add_output('z', 9.0)

        G1 = prob.model.add_subsystem('G1', Group())
        G1.add_subsystem('C1', ExecComp('y=2.0*x*x'))
        G1.add_subsystem('C2', ExecComp('y=3.0*x*x'))

        #prob.model.nonlinear_solver = NewtonSolver()
        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        G1.linear_solver = DirectSolver(assemble_jac=True)
        G1.nonlinear_solver = NewtonSolver()

        # before the fix, we got bad offsets into the _ext_mtx matrix.
        # to get entries in _ext_mtx, there must be at least one connection
        # to an input in the system that owns the AssembledJacobian, from
        # a source that is outside of that system. In this case, the 'indeps'
        # system is outside of the 'G1' group which owns the AssembledJacobian.
        prob.model.connect('indeps.y', 'G1.C1.x')
        prob.model.connect('indeps.z', 'G1.C2.x')

        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['G1.C1.y'], 50.0)
        assert_rel_error(self, prob['G1.C2.y'], 243.0)

    def test_declare_partial_reference(self):
        # Test for a bug where declare partial is given an array reference
        # that compute also uses and could get corrupted

        class Comp(ExplicitComponent):
            def setup(self):
                self.add_input('x', val=1.0, shape=2)
                self.add_output('y', val=1.0, shape=2)

                self.val = 2 * np.ones(2)
                self.rows = np.arange(2)
                self.cols = np.arange(2)
                self.declare_partials(
                    'y', 'x', val=self.val, rows=self.rows, cols=self.cols)
            def compute(self, inputs, outputs):
                outputs['y'][:] = 0.
                np.add.at(
                    outputs['y'], self.rows,
                    self.val * inputs['x'][self.cols])

        prob = Problem(model=Comp())
        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['y'], 2 * np.ones(2))

    def test_assembled_jacobian_unsupported_cases(self):

        class ParaboloidApply(ImplicitComponent):

            def setup(self):
                self.add_input('x', val=0.0)
                self.add_input('y', val=0.0)

                self.add_output('f_xy', val=0.0)

            def linearize(self, inputs, outputs, jacobian):
                return

            def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals,
                             mode):
                d_residuals['x'] += (np.exp(outputs['x']) - 2*inputs['a']**2 * outputs['x'])*d_outputs['x']
                d_residuals['x'] += (-2 * inputs['a'] * outputs['x']**2)*d_inputs['a']

        # One level deep

        prob = Problem()
        model = prob.model = Group(assembled_jac_type='dense')
        model.linear_solver = DirectSolver(assemble_jac=True)

        model.add_subsystem('p1', IndepVarComp('x', val=1.0))
        model.add_subsystem('p2', IndepVarComp('y', val=1.0))
        model.add_subsystem('comp', ParaboloidApply())

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        prob.setup()

        msg = "AssembledJacobian not supported for matrix-free subcomponent."
        with assertRaisesRegex(self, Exception, msg):
            prob.run_model()

        # Nested

        prob = Problem()
        model = prob.model = Group(assembled_jac_type='dense')
        model.linear_solver = DirectSolver(assemble_jac=True)

        sub = model.add_subsystem('sub', Group())

        model.add_subsystem('p1', IndepVarComp('x', val=1.0))
        model.add_subsystem('p2', IndepVarComp('y', val=1.0))
        sub.add_subsystem('comp', ParaboloidApply())

        model.connect('p1.x', 'sub.comp.x')
        model.connect('p2.y', 'sub.comp.y')

        prob.setup()

        msg = "AssembledJacobian not supported for matrix-free subcomponent."
        with assertRaisesRegex(self, Exception, msg):
            prob.run_model()

        # Try a component that is derived from a matrix-free one

        class FurtherDerived(ParaboloidApply):
            def do_nothing(self):
                pass

        prob = Problem()
        model = prob.model = Group(assembled_jac_type='dense')
        model.linear_solver = DirectSolver(assemble_jac=True)

        model.add_subsystem('p1', IndepVarComp('x', val=1.0))
        model.add_subsystem('p2', IndepVarComp('y', val=1.0))
        model.add_subsystem('comp', FurtherDerived())

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        prob.setup()

        msg = "AssembledJacobian not supported for matrix-free subcomponent."
        with assertRaisesRegex(self, Exception, msg):
            prob.run_model()

        # Make sure regular comps don't give an error.

        prob = Problem()
        model = prob.model = Group(assembled_jac_type='dense')
        model.linear_solver = DirectSolver(assemble_jac=True)

        model.add_subsystem('p1', IndepVarComp('x', val=1.0))
        model.add_subsystem('p2', IndepVarComp('y', val=1.0))
        model.add_subsystem('comp', Paraboloid())

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        prob.setup()
        prob.final_setup()

        class ParaboloidJacVec(Paraboloid):

            def linearize(self, inputs, outputs, jacobian):
                return

            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, d_residuals, mode):
                d_residuals['x'] += (np.exp(outputs['x']) - 2*inputs['a']**2 * outputs['x'])*d_outputs['x']
                d_residuals['x'] += (-2 * inputs['a'] * outputs['x']**2)*d_inputs['a']

        # One level deep

        prob = Problem()
        model = prob.model = Group(assembled_jac_type='dense')
        model.linear_solver = DirectSolver(assemble_jac=True)

        model.add_subsystem('p1', IndepVarComp('x', val=1.0))
        model.add_subsystem('p2', IndepVarComp('y', val=1.0))
        model.add_subsystem('comp', ParaboloidJacVec())

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        prob.setup()

        msg = "AssembledJacobian not supported for matrix-free subcomponent."
        with assertRaisesRegex(self, Exception, msg):
            prob.run_model()

    def test_access_undeclared_subjac(self):

        class Undeclared(ExplicitComponent):

            def setup(self):
                self.add_input('x', val=0.0)
                self.add_output('y', val=0.0)

            def compute(self, inputs, outputs):
                pass

            def compute_partials(self, inputs, partials):
                partials['y', 'x'] = 1.0


        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', val=1.0))
        model.add_subsystem('comp', Undeclared())

        model.connect('p1.x', 'comp.x')

        prob.setup()
        prob.run_model()

        msg = 'Variable name pair \("{}", "{}"\) must first be declared.'
        with assertRaisesRegex(self, KeyError, msg.format('y', 'x')):
            J = prob.compute_totals(of=['comp.y'], wrt=['p1.x'])

    def test_one_src_2_tgts_with_src_indices_densejac(self):
        size = 4
        prob = Problem(model=Group(assembled_jac_type='dense'))
        indeps = prob.model.add_subsystem('indeps', IndepVarComp('x', np.ones(size)))

        G1 = prob.model.add_subsystem('G1', Group())
        G1.add_subsystem('C1', ExecComp('z=2.0*y+3.0*x', x=np.zeros(size//2), y=np.zeros(size//2),
                                        z=np.zeros(size//2)))

        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.add_objective('G1.C1.z')
        prob.model.add_design_var('indeps.x')

        prob.model.connect('indeps.x', 'G1.C1.x', src_indices=[0,1])
        prob.model.connect('indeps.x', 'G1.C1.y', src_indices=[2,3])

        prob.setup(check=False)
        prob.run_model()

        J = prob.compute_totals(of=['G1.C1.z'], wrt=['indeps.x'])
        assert_rel_error(self, J['G1.C1.z', 'indeps.x'], np.array([[ 3.,  0.,  2.,  0.],
                                                                   [-0.,  3.,  0.,  2.]]), .0001)

    def test_one_src_2_tgts_with_src_indices_cscjac_error(self):
        size = 4
        prob = Problem()
        indeps = prob.model.add_subsystem('indeps', IndepVarComp('x', np.ones(size)))

        G1 = prob.model.add_subsystem('G1', Group())
        G1.add_subsystem('C1', ExecComp('z=2.0*y+3.0*x', x=np.zeros(size//2), y=np.zeros(size//2),
                                        z=np.zeros(size//2)))

        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.add_objective('G1.C1.z')
        prob.model.add_design_var('indeps.x')

        prob.model.connect('indeps.x', 'G1.C1.x', src_indices=[0,1])
        prob.model.connect('indeps.x', 'G1.C1.y', src_indices=[2,3])

        prob.setup()

        with self.assertRaises(Exception) as context:
            prob.final_setup()
        self.assertEqual(str(context.exception),
                         "Keys [('G1.C1.z', 'G1.C1.x'), ('G1.C1.z', 'G1.C1.y')] map to the same "
                         "sub-jacobian of a CSC or CSR partial jacobian and at least one of them "
                         "is either not dense or uses src_indices.  This can occur when multiple "
                         "inputs on the same component are connected to the same output.")

    def test_one_src_2_tgts_csc_error(self):
        size = 10
        prob = Problem()
        indeps = prob.model.add_subsystem('indeps', IndepVarComp('x', np.ones(size)))

        G1 = prob.model.add_subsystem('G1', Group())
        G1.add_subsystem('C1', ExecComp('z=2.0*y+3.0*x', x=np.zeros(size), y=np.zeros(size),
                                        z=np.zeros(size)))

        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.add_objective('G1.C1.z')
        prob.model.add_design_var('indeps.x')

        prob.model.connect('indeps.x', 'G1.C1.x')
        prob.model.connect('indeps.x', 'G1.C1.y')

        prob.setup(mode='fwd')
        prob.run_model()

        J = prob.compute_totals(of=['G1.C1.z'], wrt=['indeps.x'])
        assert_rel_error(self, J['G1.C1.z', 'indeps.x'], np.eye(10)*5.0, .0001)


if __name__ == '__main__':
    unittest.main()
