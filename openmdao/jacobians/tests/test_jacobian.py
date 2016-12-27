
import unittest

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse

from openmdao.api import IndepVarComp, Group, Problem, ExplicitComponent, DenseMatrix, \
     GlobalJacobian, NewtonSolver, ScipyIterativeSolver, CsrMatrix, CooMatrix
from openmdao.devtools.testutil import assert_rel_error
from nose_parameterized import parameterized
import itertools


class MyExplicitComp(ExplicitComponent):
    def __init__(self, jac_type):
        super(MyExplicitComp, self).__init__()
        self._jac_type = jac_type

    def initialize_variables(self):
        self.add_input('x', val=np.zeros(2))
        self.add_input('y', val=np.zeros(2))
        self.add_output('f', val=np.zeros(2))

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        outputs['f'][0] = (x[0]-3.0)**2 + x[0]*x[1] + (x[1]+4.0)**2 - 3.0 + \
                           y[0]*17. - y[0]*y[1] + 2.*y[1]
        outputs['f'][1] = outputs['f'][0]*3.0

    def compute_jacobian(self, inputs, outputs, jacobian):
        x = inputs['x']
        y = inputs['y']
        jacobian['f', 'x'] = self._jac_type(np.array([
            [2.0*x[0] - 6.0 + x[1], 2.0*x[1] + 8.0 + x[0]],
            [(2.0*x[0] - 6.0 + x[1])*3., (2.0*x[1] + 8.0 + x[0])*3.]
        ]))

        jacobian['f', 'y'] = self._jac_type(np.array([
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

    def compute(self, inputs, outputs):
        w = inputs['w']
        z = inputs['z']
        outputs['f'] = (w[0]-5.0)**2 + (w[1]+1.0)**2 + w[2]*6. + z*7.

    def compute_jacobian(self, inputs, outputs, jacobian):
        w = inputs['w']
        z = inputs['z']
        jacobian['f', 'w'] = self._jac_type(np.array([[
            2.0*w[0] - 10.0,
            2.0*w[1] + 2.0,
            6.
        ]]))

        jacobian['f', 'z'] = self._jac_type(np.array([[
            7.
        ]]))


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


class TestJacobian(unittest.TestCase):

    @parameterized.expand(itertools.product(
        [DenseMatrix, CsrMatrix, CooMatrix],
        [np.array, coo_matrix, csr_matrix, inverted_coo, inverted_csr, arr2list, arr2revlist]
        ), testcase_func_name=
            lambda func, num, param: 'test_jacobian_src_indices_' + '_'.join(p.__name__ for p in param.args)
    )
    def test_src_indices(self, matrix_class, comp_jac_class):

        self._setup_model(matrix_class, comp_jac_class)

        # if we multiply our jacobian (at x,y = ones) by our work vec of 1's,
        # we get fwd_check
        fwd_check = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -24., -74., -8.])

        # if we multiply our jacobian's transpose by our work vec of 1's,
        # we get rev_check
        rev_check = np.array([-35., -5., 9., -63., -3., 1., -6., 1.])

        self._check_fwd(self.prob, fwd_check)
        self._check_rev(self.prob, rev_check)

    def _setup_model(self, mat_class, comp_jac_class):
        self.prob = prob = Problem(root=Group())
        prob.root.add_subsystem('indep',
                                IndepVarComp((
                                    ('a', np.ones(3)),
                                    ('b', np.ones(2)),
                                )))
        C1 = prob.root.add_subsystem('C1', MyExplicitComp(comp_jac_class))
        C2 = prob.root.add_subsystem('C2', MyExplicitComp2(comp_jac_class))
        prob.root.connect('indep.a', 'C1.x', src_indices=[2,0])
        prob.root.connect('indep.b', 'C1.y')
        prob.root.connect('indep.a', 'C2.w', src_indices=[0,2,1])
        prob.root.connect('C1.f', 'C2.z', src_indices=[1])

        prob.setup(check=False)
        prob.root.jacobian = GlobalJacobian(matrix_class=mat_class)
        prob.root.nl_solver = NewtonSolver(
            subsolvers={'linear': ScipyIterativeSolver(
                maxiter=100,
            )}
        )
        prob.root.ln_solver = ScipyIterativeSolver(
            maxiter=200, atol=1e-10, rtol=1e-10)
        prob.root.suppress_solver_output = True

        prob.run()

    def _check_fwd(self, prob, check_vec):
        work = prob.root._vectors['output']['']._clone()
        work.set_const(1.0)

        # fwd apply_linear test
        prob.root._vectors['output'][''].set_const(1.0)
        prob.root._apply_linear([''], 'fwd')
        res = prob.root._vectors['residual']['']
        res.set_data(res.get_data() - check_vec)
        self.assertAlmostEqual(
            prob.root._vectors['residual'][''].get_norm(), 0)

        # fwd solve_linear test
        prob.root._vectors['output'][''].set_const(0.0)
        prob.root._vectors['residual'][''].set_data(check_vec)
        prob.root._solve_linear([''], 'fwd')
        prob.root._vectors['output'][''] -= work
        self.assertAlmostEqual(
            prob.root._vectors['output'][''].get_norm(), 0, delta=1e-6)

    def _check_rev(self, prob, check_vec):
        work = prob.root._vectors['output']['']._clone()
        work.set_const(1.0)

        # rev apply_linear test
        prob.root._vectors['residual'][''].set_const(1.0)
        prob.root._apply_linear([''], 'rev')
        outs = prob.root._vectors['output']['']
        outs.set_data(outs.get_data() - check_vec)
        self.assertAlmostEqual(
            prob.root._vectors['output'][''].get_norm(), 0)

        # rev solve_linear test
        prob.root._vectors['residual'][''].set_const(0.0)
        prob.root._vectors['output'][''].set_data(check_vec)
        prob.root._solve_linear([''], 'rev')
        prob.root._vectors['residual'][''] -= work
        self.assertAlmostEqual(
            prob.root._vectors['residual'][''].get_norm(), 0, delta=1e-6)
