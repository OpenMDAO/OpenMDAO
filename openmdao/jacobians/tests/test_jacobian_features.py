import unittest
import numpy as np
import scipy as sp
import itertools
from scipy.sparse import coo_matrix, csr_matrix, issparse
from six import iteritems
from nose_parameterized import parameterized

from openmdao.api import IndepVarComp, Group, Problem, ExplicitComponent, DenseMatrix, \
     GlobalJacobian, CsrMatrix, CooMatrix, ScipyIterativeSolver
from openmdao.jacobians.default_jacobian import DefaultJacobian
from openmdao.devtools.testutil import assert_rel_error


class SimpleComp(ExplicitComponent):
    def initialize_variables(self):
        self.add_input('x', shape=1)
        self.add_input('y1', shape=2)
        self.add_input('y2', shape=2)
        self.add_input('z', shape=(2, 2))

        self.add_output('f', shape=1)
        self.add_output('g', shape=(2, 2))

        self.initialize_partials()

    def initialize_partials(self):
        pass

    def compute(self, inputs, outputs):
        outputs['f'] = np.sum(inputs['z']) + inputs['x']
        outputs['g'] = np.outer(inputs['y1'], inputs['y2']) + inputs['x']*np.eye(2)

    def compute_jacobian(self, inputs, outputs, jacobian):
        jacobian['f', 'x'] = 1.
        jacobian['f', 'z'] = np.ones((2, 2)).flat[:]

        jacobian['g', 'y1'] = np.hstack((np.array([[1, 1], [0, 0]]).flat,
                                         np.array([[0, 0], [1, 1]]).flat))

        jacobian['g', 'y2'] = np.hstack((np.array([[1, 0], [1, 0]]).flat,
                                         np.array([[0, 1], [0, 1]]).flat))

        jacobian['g', 'x'] = np.eye(2)


class SimpleCompConst(SimpleComp):
    def initialize_partials(self):
        self.declare_partials('f', ['y1', 'y2'], dependent=False)
        self.declare_partials('g', 'z', dependent=False)

        self.declare_partials('f', 'x', val=1.)
        self.declare_partials('f', 'z', val=np.ones((1, 4)))
        self.declare_partials('g', 'y1', val=[[1, 0], [1, 0], [0, 1], [0, 1]])
        self.declare_partials('g', 'y2', val=1., cols=[0, 0, 1, 1], rows=[0, 3, 0, 3])
        self.declare_partials('g', 'x', val=sp.sparse.coo_matrix(((1., 1.), ((0, 3), (0, 0)))))

    def compute_jacobian(self, inputs, outputs, jacobian):
        pass


class SimpleCompKwarg(SimpleComp):
    def __init__(self, partial_kwargs):
        self.partial_kwargs = partial_kwargs
        super(SimpleCompKwarg, self).__init__()

    def initialize_partials(self):
        self.declare_partials(**self.partial_kwargs)

    def compute_jacobian(self, inputs, outputs, jacobian):
        pass


class TestJacobianFeatures(unittest.TestCase):

    def setUp(self):
        self.model = model = Group()
        model.add_subsystem('input_comp', IndepVarComp(
            (('x', 1.),
             ('y1', np.ones(2)),
             ('y2', np.ones(2)),
             ('z', np.ones((2, 2))))
            ), promotes=['x', 'y1', 'y2', 'z'])

        self.problem = Problem(model=model)
        model.suppress_solver_output = True
        model.ln_solver = ScipyIterativeSolver()
        model.jacobian = GlobalJacobian(matrix_class=CooMatrix)

    def test_dependence(self):
        problem = self.problem
        model = problem.model
        model.add_subsystem('simple', SimpleCompConst(), promotes=['x', 'y1', 'y2', 'z', 'f', 'g'])
        problem.setup(check=False)
        problem.run_model()

        # Note: since this test is looking for something not user-facing, it is inherently fragile
        # w.r.t. internal implementations.
        model._linearize()
        jac = model._jacobian._int_mtx._matrix

        # Testing dependence by examining the number of entries in the Jacobian. If non-zeros are
        # removed during array creation (e.g. `eliminate_zeros` function on scipy.sparse matrices),
        # then this test will fail since there are zero entries in the sub-Jacobians.

        # 14 for outputs w.r.t. themselves
        # 1 for df/dx
        # 4 for df/dz
        # 8 for dg/dy1
        # 4 for dg/dy2
        # 2 for dg/dx
        expected_nnz = 14 + 1 + 4 + 8 + 4 + 2

        self.assertEqual(jac.nnz, expected_nnz)

    @parameterized.expand([
        ({'of': 'f', 'wrt': 'z', 'val': np.ones((1, 5))},
         'simple: d(f)/d(z): Expected 1x4 but val is 1x5'),
        ({'of': 'f', 'wrt': 'z', 'rows': [0, -1, 4], 'cols': [0, 0, 0]},
         'simple: d(f)/d(z): row indices must be non-negative'),
        ({'of': 'f', 'wrt': 'z', 'rows': [0, 0, 0], 'cols': [0, -1, 4]},
         'simple: d(f)/d(z): col indices must be non-negative'),
        ({'of': 'f', 'wrt': 'z', 'rows': [0, 0], 'cols': [0, 4]},
         'simple: d(f)/d(z): Expected 1x4 but declared at least 1x5'),
        ({'of': 'f', 'wrt': 'z', 'rows': [0, 10]},
         'If one of rows/cols is specified, then both must be specified'),
        ({'of': 'f', 'wrt': 'z', 'cols': [0, 10]},
         'If one of rows/cols is specified, then both must be specified'),
        ({'of': 'f', 'wrt': 'z', 'rows': [0], 'cols': [0, 3]},
         'rows and cols must have the same shape, rows: (1,), cols: (2,)'),
        ({'of': 'f', 'wrt': 'z', 'rows': [0, 0, 0], 'cols': [0, 1, 3], 'val':[0, 1]},
         'If rows and cols are specified, val must be a scalar or have the same shape, '
         'val: (2,), rows/cols: (3,)'),
    ])
    def test_bad_sizes(self, partials_kwargs, error_msg):
        comp = SimpleCompKwarg(partials_kwargs)
        problem = self.problem
        model = problem.model
        model.add_subsystem('simple', comp, promotes=['x', 'y1', 'y2', 'z', 'f', 'g'])
        with self.assertRaises(ValueError) as ex:
            problem.setup(check=False)

        self.assertEqual(error_msg, str(ex.exception))


if __name__ == '__main__':
    unittest.main()
