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
                         NewtonSolver, ScipyIterativeSolver, \
                         DenseJacobian, CSRJacobian, CSCJacobian, COOJacobian
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDerivatives


class MyExplicitComp(ExplicitComponent):
    def __init__(self, jac_type):
        super(MyExplicitComp, self).__init__()
        self._jac_type = jac_type

    def initialize_variables(self):
        self.add_input('x', val=np.zeros(2))
        self.add_input('y', val=np.zeros(2))
        self.add_output('f', val=np.zeros(2))

    def initialize_partials(self):
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

    def compute_partial_derivs(self, inputs, outputs, partials):
        x = inputs['x']
        y = inputs['y']
        partials['f', 'x'] = self._jac_type(np.array([
            [2.0*x[0] - 6.0 + x[1], 2.0*x[1] + 8.0 + x[0]],
            [(2.0*x[0] - 6.0 + x[1])*3., (2.0*x[1] + 8.0 + x[0])*3.]
        ]))

        partials['f', 'y'] = self._jac_type(np.array([
            [17.-y[1], 2.-y[0]],
            [(17.-y[1])*3., (2.-y[0])*3.]
        ]))

class MyExplicitComp2(ExplicitComponent):
    def __init__(self, jac_type):
        super(MyExplicitComp2, self).__init__()
        self._jac_type = jac_type

    def initialize_variables(self):
        self.add_input('w', val=np.zeros(3))
        self.add_input('z', val=0.0)
        self.add_output('f', val=0.0)

    def initialize_partials(self):
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

    def compute_partial_derivs(self, inputs, outputs, partials):
        w = inputs['w']
        z = inputs['z']
        partials['f', 'w'] = self._jac_type(np.array([[
            2.0*w[0] - 10.0,
            2.0*w[1] + 2.0,
            6.
        ]]))

class ExplicitSetItemComp(ExplicitComponent):
    def __init__(self, dtype, value, shape, constructor):
        self._dtype = dtype
        self._shape = shape
        self._value = value
        self._constructor = constructor
        super(ExplicitSetItemComp, self).__init__()

    def initialize_variables(self):
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

    def compute_partial_derivs(self, inputs, outputs, partials):
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
        [DenseJacobian, CSRJacobian, CSCJacobian, COOJacobian],
        [np.array, coo_matrix, csr_matrix, inverted_coo, inverted_csr, arr2list, arr2revlist],
        [False, True],  # not nested, nested
        [0, 1],  # extra calls to linearize
        ), testcase_func_name=_test_func_name
    )
    def test_src_indices(self, jacobian_class, comp_jac_class, nested, lincalls):

        self._setup_model(jacobian_class, comp_jac_class, nested, lincalls)

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

    def _setup_model(self, jac_class, comp_jac_class, nested, lincalls):
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

        top.jacobian = jac_class()
        top.nl_solver = NewtonSolver()
        top.nl_solver.ln_solver = ScipyIterativeSolver(maxiter=100)
        top.ln_solver = ScipyIterativeSolver(
            maxiter=200, atol=1e-10, rtol=1e-10)
        prob.model.suppress_solver_output = True

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
        ('scalar', lambda x: x),
        ('1D_array', lambda x: np.array([x + i for i in range(5)])),
        ('2D_array', lambda x: np.array([[x + i + 2 * j for i in range(3)] for j in range(3)]))
    ]

    @parameterized.expand(itertools.product(dtypes, shapes), testcase_func_name=
        lambda f, n, p: '_'.join(['test_jacobian_set_item', p.args[0][0], p.args[1][0]]))
    def test_jacobian_set_item(self, dtypes, shapes):

        shape, constructor = shapes
        dtype, value = dtypes

        prob = Problem(model=Group())
        comp = ExplicitSetItemComp(dtype, value, shape, constructor)
        prob.model.add_subsystem('C1', comp)
        prob.setup(check=False)

        prob.model.suppress_solver_output = True
        prob.run_model()
        prob.model.run_apply_nonlinear()
        prob.model.run_linearize()

        expected = constructor(value)
        with prob.model._subsystems_allprocs[0].jacobian_context() as J:
            jac_out = J['out', 'in'] * -1

        self.assertEqual(len(jac_out.shape), 2)
        expected_dtype = np.promote_types(dtype, float)
        self.assertEqual(jac_out.dtype, expected_dtype)
        assert_rel_error(self, jac_out.squeeze(), expected, 1e-15)

    def test_component_assembled_jac(self):
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nl_solver = NewtonSolver()

        d1 = prob.model.get_subsystem('d1')

        d1.jacobian = DenseJacobian()
        prob.model.suppress_solver_output = True

        prob.setup(check=False)
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_assembled_jac_bad_key(self):
        # this test fails if AssembledJacobian._update sets in_start with 'output' instead of 'input'
        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('indep', IndepVarComp('x', 1.0))
        prob.model.add_subsystem('C1', ExecComp('c=a*2.0+b'))
        c2 = prob.model.add_subsystem('C2', ExecComp('d=a*2.0+b+c'))
        c3 = prob.model.add_subsystem('C3', ExecComp('ee=a*2.0'))

        prob.model.nl_solver = NewtonSolver()
        c3.jacobian = DenseJacobian()

        prob.model.connect('indep.x', 'C1.a')
        prob.model.connect('indep.x', 'C2.a')
        prob.model.connect('C1.c', 'C2.b')
        prob.model.connect('C2.d', 'C3.a')
        prob.model.suppress_solver_output = True
        prob.setup(check=False)
        prob.run_model()
        assert_rel_error(self, prob['C3.ee'], 8.0, 0000.1)

    def test_jacobian_changed_group(self):
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nl_solver = NewtonSolver()

        prob.model.jacobian = DenseJacobian()

        prob.setup(check=False)

        prob.model.jacobian = DenseJacobian()

        msg = ": jacobian has changed and setup was not called."
        with assertRaisesRegex(self, Exception, msg):
            prob.run_model()

    def test_jacobian_changed_component(self):
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nl_solver = NewtonSolver()

        prob.setup(check=False)

        d1 = prob.model.get_subsystem('d1')
        d1.jacobian = DenseJacobian()

        msg = "d1: jacobian has changed and setup was not called."
        with assertRaisesRegex(self, Exception, msg):
            prob.run_model()

    def test_assembled_jacobian_submat_indexing(self):
        prob = Problem()
        indeps = prob.model.add_subsystem('indeps', IndepVarComp())
        indeps.add_output('x', 1.0)
        indeps.add_output('y', 5.0)
        indeps.add_output('z', 9.0)

        G1 = prob.model.add_subsystem('G1', Group())
        G1.add_subsystem('C1', ExecComp('y=2.0*x*x'))
        G1.add_subsystem('C2', ExecComp('y=3.0*x*x'))

        prob.model.nl_solver = NewtonSolver()
        G1.jacobian = DenseJacobian()

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

    def test_sparse_jac_with_subsolver_error(self):
        prob = Problem()
        indeps = prob.model.add_subsystem('indeps', IndepVarComp('x', 1.0))

        G1 = prob.model.add_subsystem('G1', Group())
        G1.add_subsystem('C1', ExecComp('y=2.0*x'))
        G1.add_subsystem('C2', ExecComp('y=3.0*x'))

        G1.nl_solver = NewtonSolver()
        prob.model.jacobian = CSRJacobian()

        prob.model.connect('indeps.x', 'G1.C1.x')
        prob.model.connect('indeps.x', 'G1.C2.x')

        with self.assertRaises(Exception) as context:
            prob.setup(check=False)
        self.assertEqual(str(context.exception),
                         "System 'G1' has a solver of type 'NewtonSolver'but a sparse "
                         "AssembledJacobian has been set in a higher level system.")

    def test_assembled_jacobian_unsupported_cases(self):

        class ParaboloidApply(ImplicitComponent):

            def initialize_variables(self):
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
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', val=1.0))
        model.add_subsystem('p2', IndepVarComp('y', val=1.0))
        model.add_subsystem('comp', ParaboloidApply())

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        model.jacobian = DenseJacobian()

        msg = "AssembledJacobian not supported if any subcomponent is matrix-free."
        with assertRaisesRegex(self, Exception, msg):
            prob.setup()

        # Nested

        prob = Problem()
        model = prob.model = Group()

        sub = model.add_subsystem('sub', Group())

        model.add_subsystem('p1', IndepVarComp('x', val=1.0))
        model.add_subsystem('p2', IndepVarComp('y', val=1.0))
        sub.add_subsystem('comp', ParaboloidApply())

        model.connect('p1.x', 'sub.comp.x')
        model.connect('p2.y', 'sub.comp.y')

        model.jacobian = DenseJacobian()

        msg = "AssembledJacobian not supported if any subcomponent is matrix-free."
        with assertRaisesRegex(self, Exception, msg):
            prob.setup()

        # Try a component that is derived from a matrix-free one

        class FurtherDerived(ParaboloidApply):
            def do_nothing(self):
                pass

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', val=1.0))
        model.add_subsystem('p2', IndepVarComp('y', val=1.0))
        model.add_subsystem('comp', FurtherDerived())

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        model.jacobian = DenseJacobian()

        msg = "AssembledJacobian not supported if any subcomponent is matrix-free."
        with assertRaisesRegex(self, Exception, msg):
            prob.setup()

        # Make sure regular comps don't give an error.

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', val=1.0))
        model.add_subsystem('p2', IndepVarComp('y', val=1.0))
        model.add_subsystem('comp', Paraboloid())

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        model.jacobian = DenseJacobian()

        prob.setup()

        class ParaboloidJacVec(Paraboloid):

            def linearize(self, inputs, outputs, jacobian):
                return

            def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, d_residuals,
                                       mode):
                d_residuals['x'] += (np.exp(outputs['x']) - 2*inputs['a']**2 * outputs['x'])*d_outputs['x']
                d_residuals['x'] += (-2 * inputs['a'] * outputs['x']**2)*d_inputs['a']

        # One level deep

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', val=1.0))
        model.add_subsystem('p2', IndepVarComp('y', val=1.0))
        model.add_subsystem('comp', ParaboloidJacVec())

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        model.jacobian = DenseJacobian()

        msg = "AssembledJacobian not supported if any subcomponent is matrix-free."
        with assertRaisesRegex(self, Exception, msg):
            prob.setup()


if __name__ == '__main__':
    unittest.main()
