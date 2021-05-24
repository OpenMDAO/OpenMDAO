""" Test the Jacobian objects."""

import itertools
import sys
import unittest

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from openmdao.api import IndepVarComp, Group, Problem, \
                         ExplicitComponent, ImplicitComponent, ExecComp, \
                         NewtonSolver, ScipyKrylov, \
                         LinearBlockGS, DirectSolver
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.array_utils import rand_sparsity
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.api import ScipyOptimizeDriver

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized


class MyExplicitComp(ExplicitComponent):
    def __init__(self, jac_type):
        super().__init__()
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
        super().__init__()
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
        super().__init__()

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


class SimpleCompWithPrintPartials(ExplicitComponent):

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0, upper=1.0)

        self.declare_partials(of='*', wrt='*')

        self.count = 0
        self.partials_name_pairs = []
        self.partials_values = []

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        outputs['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        y = inputs['y']

        partials['f_xy', 'x'] = 2.0*x - 6.0 + y
        partials['f_xy', 'y'] = 2.0*y + 8.0 + x

        if self.count < 1:  # Only want to save these this once for the test
            for k in partials:
                self.partials_name_pairs.append(k)

            for k, v in partials.items():
                self.partials_values.append((k,v))

        self.count += 1

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
        ), name_func=_test_func_name
    )
    def test_src_indices(self, assembled_jac, comp_jac_class, nested, lincalls):

        self._setup_model(assembled_jac, comp_jac_class, nested, lincalls)

        # if we multiply our jacobian (at x,y = ones) by our work vec of 1's,
        # we get fwd_check
        fwd_check = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 24., 74., 8.])

        # if we multiply our jacobian's transpose by our work vec of 1's,
        # we get rev_check
        rev_check = np.array([35., 5., -9., 63., 3., -1., 6., -1.])

        self._check_fwd(self.prob, fwd_check)
        # to catch issues with constant subjacobians, repeatedly call linearize
        for i in range(lincalls):
            self.prob.model.run_linearize()
        self._check_fwd(self.prob, fwd_check)
        self._check_rev(self.prob, rev_check)

    def _setup_model(self, assembled_jac, comp_jac_class, nested, lincalls):
        self.prob = prob = Problem()
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

        top.nonlinear_solver = NewtonSolver(solve_subsystems=False)
        top.nonlinear_solver.linear_solver = ScipyKrylov(maxiter=100)
        top.linear_solver = ScipyKrylov(
            maxiter=200, atol=1e-10, rtol=1e-10, assemble_jac=True)
        top.options['assembled_jac_type'] = assembled_jac

        prob.set_solver_print(level=0)

        prob.setup()

        prob.run_model()

    def _check_fwd(self, prob, check_vec):
        d_inputs, d_outputs, d_residuals = prob.model.get_linear_vectors()

        work = np.ones(d_outputs._data.size)

        # fwd apply_linear test
        d_outputs.set_val(1.0)
        prob.model.run_apply_linear(['linear'], 'fwd')
        d_residuals.set_val(d_residuals.asarray() - check_vec)
        self.assertAlmostEqual(d_residuals.get_norm(), 0)

        # fwd solve_linear test
        d_outputs.set_val(0.0)
        d_residuals.set_val(check_vec)

        prob.model.run_solve_linear(['linear'], 'fwd')

        d_outputs -= work
        self.assertAlmostEqual(d_outputs.get_norm(), 0, delta=1e-6)

    def _check_rev(self, prob, check_vec):
        d_inputs, d_outputs, d_residuals = prob.model.get_linear_vectors()

        work = np.ones(d_outputs._data.size)

        # rev apply_linear test
        d_residuals.set_val(1.0)
        prob.model.run_apply_linear(['linear'], 'rev')
        d_outputs.set_val(d_outputs.asarray() - check_vec)
        self.assertAlmostEqual(d_outputs.get_norm(), 0)

        # rev solve_linear test
        d_residuals.set_val(0.0)
        d_outputs.set_val(check_vec)
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

    @parameterized.expand(itertools.product(dtypes, shapes), name_func=
    lambda f, n, p: '_'.join(['test_jacobian_set_item', p.args[0][0], p.args[1][0]]))
    def test_jacobian_set_item(self, dtypes, shapes):

        shape, constructor, expected_shape = shapes
        dtype, value = dtypes

        prob = Problem()
        comp = ExplicitSetItemComp(dtype, value, shape, constructor)
        comp = prob.model.add_subsystem('C1', comp)
        prob.setup()

        prob.set_solver_print(level=0)
        prob.run_model()
        prob.model.run_apply_nonlinear()
        prob.model.run_linearize()

        expected = constructor(value)
        J = comp._jacobian
        jac_out = J['out', 'in']

        self.assertEqual(len(jac_out.shape), 2)
        expected_dtype = np.promote_types(dtype, float)
        self.assertEqual(jac_out.dtype, expected_dtype)
        assert_near_equal(jac_out, np.atleast_2d(expected).reshape(expected_shape), 1e-15)

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
        model = prob.model

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
        prob.model.add_subsystem('C1', ExecComp('c=a*2.0+b', a=0., b=0., c=0.))
        c2 = prob.model.add_subsystem('C2', ExecComp('d=a*2.0+b+c', a=0., b=0., c=0., d=0.))
        c3 = prob.model.add_subsystem('C3', ExecComp('ee=a*2.0', a=0., ee=0.))

        prob.model.nonlinear_solver = NewtonSolver(solve_subsystems=False)
        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        prob.model.connect('indep.x', 'C1.a')
        prob.model.connect('indep.x', 'C2.a')
        prob.model.connect('C1.c', 'C2.b')
        prob.model.connect('C2.d', 'C3.a')
        prob.set_solver_print(level=0)
        prob.setup()
        prob.run_model()
        assert_near_equal(prob['C3.ee'], 8.0, 0000.1)

    def test_assembled_jacobian_submat_indexing_dense(self):
        prob = Problem(model=Group(assembled_jac_type='dense'))
        indeps = prob.model.add_subsystem('indeps', IndepVarComp())
        indeps.add_output('x', 1.0)
        indeps.add_output('y', 5.0)
        indeps.add_output('z', 9.0)

        G1 = prob.model.add_subsystem('G1', Group())
        G1.add_subsystem('C1', ExecComp('y=2.0*x*x'))
        G1.add_subsystem('C2', ExecComp('y=3.0*x*x'))

        prob.model.nonlinear_solver = NewtonSolver(solve_subsystems=False)
        G1.linear_solver = DirectSolver(assemble_jac=True)

        # before the fix, we got bad offsets into the _ext_mtx matrix.
        # to get entries in _ext_mtx, there must be at least one connection
        # to an input in the system that owns the AssembledJacobian, from
        # a source that is outside of that system. In this case, the 'indeps'
        # system is outside of the 'G1' group which owns the AssembledJacobian.
        prob.model.connect('indeps.y', 'G1.C1.x')
        prob.model.connect('indeps.z', 'G1.C2.x')

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['G1.C1.y'], 50.0)
        assert_near_equal(prob['G1.C2.y'], 243.0)

    def test_assembled_jacobian_submat_indexing_csc(self):
        prob = Problem(model=Group(assembled_jac_type='dense'))
        indeps = prob.model.add_subsystem('indeps', IndepVarComp())
        indeps.add_output('x', 1.0)
        indeps.add_output('y', 5.0)
        indeps.add_output('z', 9.0)

        G1 = prob.model.add_subsystem('G1', Group())
        G1.add_subsystem('C1', ExecComp('y=2.0*x*x'))
        G1.add_subsystem('C2', ExecComp('y=3.0*x*x'))

        # prob.model.nonlinear_solver = NewtonSolver(solve_subsystems=False)
        prob.model.linear_solver = DirectSolver(assemble_jac=True)

        G1.linear_solver = DirectSolver(assemble_jac=True)
        G1.nonlinear_solver = NewtonSolver(solve_subsystems=False)

        # before the fix, we got bad offsets into the _ext_mtx matrix.
        # to get entries in _ext_mtx, there must be at least one connection
        # to an input in the system that owns the AssembledJacobian, from
        # a source that is outside of that system. In this case, the 'indeps'
        # system is outside of the 'G1' group which owns the AssembledJacobian.
        prob.model.connect('indeps.y', 'G1.C1.x')
        prob.model.connect('indeps.z', 'G1.C2.x')

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['G1.C1.y'], 50.0)
        assert_near_equal(prob['G1.C2.y'], 243.0)

    def test_declare_partial_reference(self):
        # Test for a bug where declare_partials is given an array reference
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

        assert_near_equal(prob['y'], 2 * np.ones(2))

    def test_declare_partials_row_col_size_mismatch(self):
        # Make sure we have clear error messages.

        class Comp1(ExplicitComponent):
            def setup(self):
                self.add_input('x', val=np.array((2, 2)))
                self.add_output('y', val=np.array((2, 2)))

                self.declare_partials('y', 'x', rows=np.array([0, 1]), cols=np.array([0]))

            def compute(self, inputs, outputs):
                pass

        class Comp2(ExplicitComponent):
            def setup(self):
                self.add_input('x', val=np.array((2, 2)))
                self.add_output('y', val=np.array((2, 2)))

                self.declare_partials('y', 'x', rows=np.array([0]), cols=np.array([0, 1]))

            def compute(self, inputs, outputs):
                pass

        prob = Problem()
        model = prob.model
        model.add_subsystem('comp', Comp1())

        msg = "'comp' <class Comp1>: d\(y\)/d\(x\): declare_partials has been called with rows and cols, which" + \
              " should be arrays of equal length, but rows is length 2 while " + \
              "cols is length 1."
        with self.assertRaisesRegex(RuntimeError, msg):
            prob.setup()

        prob = Problem()
        model = prob.model
        model.add_subsystem('comp', Comp2())

        msg = "'comp' <class Comp2>: d\(y\)/d\(x\): declare_partials has been called with rows and cols, which" + \
            " should be arrays of equal length, but rows is length 1 while " + \
            "cols is length 2."
        with self.assertRaisesRegex(RuntimeError, msg):
            prob.setup()

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
        with self.assertRaisesRegex(Exception, msg):
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
        with self.assertRaisesRegex(Exception, msg):
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
        with self.assertRaisesRegex(Exception, msg):
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

        class JacVecComp(Paraboloid):

            def setup_partials(self):
                pass

            def linearize(self, inputs, outputs, jacobian):
                return

            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                pass

        # One level deep

        prob = Problem()
        model = prob.model = Group(assembled_jac_type='dense')
        model.linear_solver = DirectSolver(assemble_jac=True)

        model.add_subsystem('p1', IndepVarComp('x', val=1.0))
        model.add_subsystem('p2', IndepVarComp('y', val=1.0))
        model.add_subsystem('comp', JacVecComp())

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        prob.setup()

        msg = "AssembledJacobian not supported for matrix-free subcomponent."
        with self.assertRaisesRegex(Exception, msg):
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
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', val=1.0))
        model.add_subsystem('comp', Undeclared())

        model.connect('p1.x', 'comp.x')

        prob.setup()
        prob.run_model()

        msg = 'Variable name pair \("{}", "{}"\) must first be declared.'
        with self.assertRaisesRegex(KeyError, msg.format('y', 'x')):
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

        prob.setup()
        prob.run_model()

        J = prob.compute_totals(of=['G1.C1.z'], wrt=['indeps.x'])
        assert_near_equal(J['G1.C1.z', 'indeps.x'], np.array([[ 3.,  0.,  2.,  0.],
                                                                   [-0.,  3.,  0.,  2.]]), .0001)

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
        assert_near_equal(J['G1.C1.z', 'indeps.x'], np.eye(10)*5.0, .0001)

    def test_dict_properties(self):
        # Make sure you can use the partials variable passed to compute_partials as a dict
        prob = Problem()

        indeps = prob.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        indeps.add_output('x', .5)
        indeps.add_output('y', 10.0)
        comp = SimpleCompWithPrintPartials()
        prob.model.add_subsystem('paraboloid', comp, promotes_inputs=['x', 'y'])

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('paraboloid.f_xy')

        prob.setup()
        prob.run_driver()

        expected = [
            (('paraboloid.f_xy', 'paraboloid.f_xy'),[-1.]),
            (('paraboloid.f_xy', 'paraboloid.x'),[[0.]]),
            (('paraboloid.f_xy', 'paraboloid.y'),[[0.]]),
        ]
        self.assertEqual(sorted(comp.partials_name_pairs), sorted(e[0] for e in sorted(expected)))

        self.assertEqual(sorted(comp.partials_name_pairs),
                         sorted(e[0] for e in sorted(expected)))
        for act, exp in zip(
                [e[1] for e in sorted(comp.partials_values)],
                [e[1] for e in sorted(expected)],
                ):
            assert_near_equal(act,exp, 1e-5)

    def test_compute_totals_relevancy(self):
        # When a model has desvars and responses defined, components that don't lie in the relevancy
        # graph between them do not take part in the linear solve. This led to some derivatives
        # being returned as zero in certain instances when it was called with wrt or of not in the
        # set.
        class DParaboloid(ExplicitComponent):

            def setup(self):
                ndvs = 3
                self.add_input('w', val=1.)
                self.add_input('x', shape=ndvs)

                self.add_output('y', shape=1)
                self.add_output('z', shape=ndvs)
                self.declare_partials('y', 'x')
                self.declare_partials('y', 'w')
                self.declare_partials('z', 'x')

            def compute(self, inputs, outputs):
                x = inputs['x']
                y_g = np.sum((x-5)**2)
                outputs['y'] = np.sum(y_g) + (inputs['w']-10)**2
                outputs['z'] = x**2

            def compute_partials(self, inputs, J):
                x = inputs['x']
                J['y', 'x'] = 2*(x-5)
                J['y', 'w'] = 2*(inputs['w']-10)
                J['z', 'x'] = np.diag(2*x)

        p = Problem()
        d_ivc = p.model.add_subsystem('distrib_ivc',
                                       IndepVarComp(),
                                       promotes=['*'])
        ndvs = 3
        d_ivc.add_output('x', 2*np.ones(ndvs))

        ivc = p.model.add_subsystem('ivc',
                                    IndepVarComp(),
                                    promotes=['*'])
        ivc.add_output('w', 2.0)
        p.model.add_subsystem('dp', DParaboloid(), promotes=['*'])


        p.model.add_design_var('x', lower=-100, upper=100)
        p.model.add_objective('y')

        p.setup(mode='rev')
        p.run_model()
        J = p.compute_totals(of=['y', 'z'], wrt=['w', 'x'])

        assert(J['y','w'][0,0] == -16)

    def test_wildcard_partials_bug(self):
        # Test for a bug where using wildcards when declaring partials resulted in extra
        # derivatives of an output wrt other outputs.

        class ODE(ExplicitComponent):

            def setup(self):

                self.add_input('a', 1.0)
                self.add_output('x', 1.0)
                self.add_output('y', 1.0)

                self.declare_partials(of='*', wrt='*', method='cs')

            def compute(self, inputs, outputs):
                a = inputs['a']
                outputs['x'] = 3.0 * a
                outputs['y'] = 7.0 * a

        p = Problem()

        p.model.add_subsystem('ode', ODE())

        p.model.linear_solver = DirectSolver()
        p.model.add_design_var('ode.a')
        p.model.add_constraint('ode.x', lower=0.0)
        p.model.add_constraint('ode.y', lower=0.0)

        p.setup()
        p.run_model()

        p.compute_totals()
        keys = p.model.ode._jacobian._subjacs_info
        self.assertTrue(('ode.x', 'ode.y') not in keys)
        self.assertTrue(('ode.y', 'ode.x') not in keys)

    def test_set_col(self):
        class MyComp(ExplicitComponent):
            def setup(self):
                self.ofsizes = [3, 5, 2]
                self.wrtsizes = [4, 1, 3]
                for i, sz in enumerate(self.wrtsizes):
                    self.add_input(f"x{i}", val=np.ones(sz))
                for i, sz in enumerate(self.ofsizes):
                    self.add_output(f"y{i}", val=np.ones(sz))

                boolarr = rand_sparsity((sum(self.ofsizes), sum(self.wrtsizes)), .3, dtype=bool)
                self.sparsity = np.asarray(boolarr.toarray(), dtype=float)
                ofstart = ofend = 0
                for i, ofsz in enumerate(self.ofsizes):
                    wrtstart = wrtend = 0
                    ofend += ofsz
                    for j, wrtsz in enumerate(self.wrtsizes):
                        wrtend += wrtsz
                        sub = self.sparsity[ofstart:ofend, wrtstart:wrtend]
                        rows, cols = np.nonzero(sub)
                        self.declare_partials([f"y{i}"], [f"x{j}"], rows=rows, cols=cols)
                        wrtstart = wrtend
                    ofstart = ofend

            def compute(self, inputs, outputs):
                outputs.set_val(self.sparsity.dot(inputs.asarray()) * 2.)

            def compute_partials(self, inputs, partials):
                # these partials are actually constant, but...
                ofstart = ofend = 0
                for i, ofsz in enumerate(self.ofsizes):
                    wrtstart = wrtend = 0
                    ofend += ofsz
                    for j, wrtsz in enumerate(self.wrtsizes):
                        wrtend += wrtsz
                        sub = self.sparsity[ofstart:ofend, wrtstart:wrtend]
                        subinfo = self._subjacs_info[(f'comp.y{i}', f'comp.x{j}')]
                        partials[f'y{i}', f'x{j}'] = sub[subinfo['rows'], subinfo['cols']] * 2.
                        wrtstart = wrtend
                    ofstart = ofend

        p = Problem()
        comp = p.model.add_subsystem('comp', MyComp())
        p.setup()
        for i, sz in enumerate(comp.wrtsizes):
            p[f'comp.x{i}'] = np.random.random(sz)
        p.run_model()
        ofs = [f'comp.y{i}' for i in range(len(comp.ofsizes))]
        wrts = [f'comp.x{i}' for i in range(len(comp.wrtsizes))]
        p.check_partials(out_stream=None, show_only_incorrect=True)
        p.model.comp._jacobian.set_col(p.model.comp, 5, comp.sparsity[:, 5] * 99)
        
        # check dy0/dx2 (3x3)
        subinfo = comp._subjacs_info['comp.y0', 'comp.x2']
        arr = np.zeros(subinfo['shape'])
        arr[subinfo['rows'], subinfo['cols']] = subinfo['value']
        assert_near_equal(arr[:, 0], comp.sparsity[0:3, 5] * 99)
        
        # check dy1/dx2 (5x3)
        subinfo = comp._subjacs_info['comp.y1', 'comp.x2']
        arr = np.zeros(subinfo['shape'])
        arr[subinfo['rows'], subinfo['cols']] = subinfo['value']
        assert_near_equal(arr[:, 0], comp.sparsity[3:8, 5] * 99)
        
        # check dy2/dx2 (2x3)
        subinfo = comp._subjacs_info['comp.y2', 'comp.x2']
        arr = np.zeros(subinfo['shape'])
        arr[subinfo['rows'], subinfo['cols']] = subinfo['value']
        assert_near_equal(arr[:, 0], comp.sparsity[8:, 5] * 99)


class MySparseComp(ExplicitComponent):
    def setup(self):
        self.add_input('y', np.zeros(2))
        self.add_input('x', np.zeros(2))
        self.add_output('z', np.zeros(2))

        # partials use sparse list format
        self.declare_partials('z', 'x', rows=[0, 1], cols=[0, 1])
        self.declare_partials('z', 'y', rows=[0, 1], cols=[1, 0])

    def compute(self, inputs, outputs):
        outputs['z'] = np.array([3.0*inputs['x'][0]**3 + 4.0*inputs['y'][1]**2,
                                 5.0*inputs['x'][1]**2 * inputs['y'][0]])

    def compute_partials(self, inputs, partials):
        partials['z', 'x'] = np.array([9.0*inputs['x'][0]**2, 10.0*inputs['x'][1]*inputs['y'][0]])
        partials['z', 'y'] = np.array([8.0*inputs['y'][1], 5.0*inputs['x'][1]**2])


class MyDenseComp(ExplicitComponent):
    def setup(self):
        self.add_input('y', np.zeros(2))
        self.add_input('x', np.zeros(2))
        self.add_output('z', np.zeros(2))

        # partials are dense
        self.declare_partials('z', 'x')
        self.declare_partials('z', 'y')

    def compute(self, inputs, outputs):
        outputs['z'] = np.array([3.0*inputs['x'][0]**3 + 4.0*inputs['y'][1]**2,
                                 5.0*inputs['x'][1]**2 * inputs['y'][0]])

    def compute_partials(self, inputs, partials):
        partials['z', 'x'] = np.array([[9.0*inputs['x'][0]**2, 0.0], [0.0, 10.0*inputs['x'][1]*inputs['y'][0]]])
        partials['z', 'y'] = np.array([[0.0, 8.0*inputs['y'][1]], [5.0*inputs['x'][1]**2, 0.0]])


class OverlappingPartialsTestCase(unittest.TestCase):
    def test_repeated_src_indices_csc(self):
        size = 2
        p = Problem()
        indeps = p.model.add_subsystem('indeps', IndepVarComp('x', np.ones(size)))

        p.model.add_subsystem('C1', ExecComp('z=3.0*x[0]**3 + 2.0*x[1]**2', x=np.zeros(size)))

        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.model.connect('indeps.x', 'C1.x', src_indices=[1,1])
        p.setup()
        p.run_model()

        J = p.compute_totals(of=['C1.z'], wrt=['indeps.x'], return_format='array')
        np.testing.assert_almost_equal(p.model._assembled_jac._int_mtx._matrix.toarray(),
                                       np.array([[-1.,  0.,  0.],
                                                 [ 0., -1.,  0.],
                                                 [ 0., 13., -1.]]))

    def test_repeated_src_indices_dense(self):
        size = 2
        p = Problem()
        indeps = p.model.add_subsystem('indeps', IndepVarComp('x', np.ones(size)))

        p.model.add_subsystem('C1', ExecComp('z=3.0*x[0]**3 + 2.0*x[1]**2', x=np.zeros(size)))

        p.model.options['assembled_jac_type'] = 'dense'
        p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.model.connect('indeps.x', 'C1.x', src_indices=[1,1])
        p.setup()
        p.run_model()

        J = p.compute_totals(of=['C1.z'], wrt=['indeps.x'], return_format='array')
        np.testing.assert_almost_equal(p.model._assembled_jac._int_mtx._matrix,
                                       np.array([[-1.,  0.,  0.],
                                                 [ 0., -1.,  0.],
                                                 [ 0., 13., -1.]]))

    def test_multi_inputs_same_src_dense_comp(self):
        p = Problem()
        indeps = p.model.add_subsystem('indeps', IndepVarComp('x', np.ones(2)))

        p.model.add_subsystem('C1', MyDenseComp())
        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.model.connect('indeps.x', ('C1.x', 'C1.y'))
        p.setup()
        p.run_model()

        J = p.compute_totals(of=['C1.z'], wrt=['indeps.x'], return_format='array')
        np.testing.assert_almost_equal(p.model._assembled_jac._int_mtx._matrix.toarray(),
                                       np.array([[-1.,  0.,  0.,  0.],
                                                 [ 0., -1.,  0.,  0.],
                                                 [ 9.,  8., -1.,  0.],
                                                 [ 5.,  10.,  0., -1.]]))

    def test_multi_inputs_same_src_sparse_comp(self):
        p = Problem()
        indeps = p.model.add_subsystem('indeps', IndepVarComp('x', np.ones(2)))

        p.model.add_subsystem('C1', MySparseComp())
        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.model.connect('indeps.x', ('C1.x', 'C1.y'))
        p.setup()
        p.run_model()

        J = p.compute_totals(of=['C1.z'], wrt=['indeps.x'], return_format='array')
        np.testing.assert_almost_equal(p.model._assembled_jac._int_mtx._matrix.toarray(),
                                       np.array([[-1.,  0.,  0.,  0.],
                                                 [ 0., -1.,  0.,  0.],
                                                 [ 9.,  8., -1.,  0.],
                                                 [ 5.,  10.,  0., -1.]]))


class MaskingTestCase(unittest.TestCase):
    def test_csc_masking(self):
        class CCBladeResidualComp(ImplicitComponent):

            def initialize(self):
                self.options.declare('num_nodes', types=int)
                self.options.declare('num_radial', types=int)

            def setup(self):
                num_nodes = self.options['num_nodes']
                num_radial = self.options['num_radial']

                self.add_input('chord', shape=(1, num_radial))
                self.add_input('theta', shape=(1, num_radial))

                self.add_output('phi', lower=-0.5*np.pi, upper=0.0,
                                shape=(num_nodes, num_radial))
                self.add_output('Tp', shape=(num_nodes, num_radial))

                of_names = ('phi', 'Tp')
                row_col = np.arange(num_radial)

                for name in of_names:
                    self.declare_partials(name, 'chord', rows=row_col, cols=row_col)
                    self.declare_partials(name, 'theta', rows=row_col, cols=row_col, val=0.0)
                    self.declare_partials(name, 'phi', rows=row_col, cols=row_col)

                self.declare_partials('Tp', 'Tp', rows=row_col, cols=row_col, val=1.)

            def linearize(self, inputs, outputs, partials):

                partials['phi', 'chord'] = np.array([1., 2, 3, 4])
                partials['phi', 'phi'] = np.array([5., 6, 7, 8])

                partials['Tp', 'chord'] = np.array([9., 10, 11, 12])
                partials['Tp', 'phi'] = np.array([13., 14, 15, 16])


        prob = Problem()
        model = prob.model

        comp = IndepVarComp()
        comp.add_output('chord', val=np.ones((4, )))
        model.add_subsystem('indep_var_comp', comp, promotes=['*'])

        comp = CCBladeResidualComp(num_nodes=1, num_radial=4, assembled_jac_type='csc')

        comp.linear_solver = DirectSolver(assemble_jac=True)
        model.add_subsystem('ccblade_comp', comp, promotes_inputs=['chord'], promotes_outputs=['Tp'])


        prob.setup(mode='fwd')
        prob.run_model()
        totals = prob.compute_totals(of=['Tp'], wrt=['chord'], return_format='array')

        expected = np.array([
        [-6.4,0.,0.,0.],
        [ 0.,-5.33333333,0.,0.],
        [ 0.,0.,-4.57142857,0.],
        [ 0.,0.,0.,-4.]]
        )

        np.testing.assert_allclose(totals, expected)


if __name__ == '__main__':
    unittest.main()
