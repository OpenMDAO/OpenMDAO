
import unittest

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse

from openmdao.api import IndepVarComp, Group, Problem, ExplicitComponent, DenseMatrix, \
     GlobalJacobian, NewtonSolver, ScipyIterativeSolver, CsrMatrix, CooMatrix
from openmdao.devtools.testutil import assert_rel_error


class MyExplicitComp(ExplicitComponent):
    def __init__(self, jac_type):
        super(MyExplicitComp, self).__init__()
        self._jac_type = jac_type

    def initialize_variables(self):
        self.add_input('x', val=np.zeros(2))
        self.add_output('f', val=0.0)

    def compute(self, inputs, outputs):
        x = inputs['x']
        outputs['f'] = (x[0]-3.0)**2 + x[0]*x[1] + (x[1]+4.0)**2 - 3.0

    def compute_jacobian(self, inputs, outputs, jacobian):
        x = inputs['x']
        jacobian['f', 'x'] = self._jac_type(np.array([[
            2.0*x[0] - 6.0 + x[1],
            2.0*x[1] + 8.0 + x[0]
        ]]))


class TestJacobianSrcIndicesDenseDense(unittest.TestCase):

    def setUp(self):
        self.prob = self._setup_model(DenseMatrix, np.array)

    def test_src_indices(self):

        # if we multiply our jacobian by our work vec of 1's, we get [1.0, 1.0, 1.0, -7.0]
        fwd_check = np.array([1.0, 1.0, 1.0, -7.])

        # if we multiply our jacobian's transpose by our work vec of 1's,
        # we get [-10., 1., 4., 1.]
        rev_check = np.array([-10., 1., 4., 1.])

        self._check_fwd(self.prob, fwd_check)
        self._check_rev(self.prob, rev_check)

    def _setup_model(self, mat_class, comp_jac_class):
        prob = Problem(root=Group())
        prob.root.add_subsystem('indep',
                                IndepVarComp((
                                    ('a', np.ones(3)),
                                )))
        C1 = prob.root.add_subsystem('C1', MyExplicitComp(comp_jac_class))
        prob.root.connect('indep.a', 'C1.x', src_indices=[2,0])
        prob.setup(check=False)
        prob.root.jacobian = GlobalJacobian(Matrix=mat_class)
        prob.root.nl_solver = NewtonSolver(
            subsolvers={'linear': ScipyIterativeSolver(
                maxiter=100,
            )}
        )
        prob.root.ln_solver = ScipyIterativeSolver(
            maxiter=200, atol=1e-10, rtol=1e-10)
        prob.root.suppress_solver_output = True

        prob.run()

        return prob

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


class TestJacobianSrcIndicesDenseCoo(TestJacobianSrcIndicesDenseDense):
    def setUp(self):
        self.prob = self._setup_model(DenseMatrix, coo_matrix)

class TestJacobianSrcIndicesDenseCsr(TestJacobianSrcIndicesDenseDense):
    def setUp(self):
        self.prob = self._setup_model(DenseMatrix, csr_matrix)

class TestJacobianSrcIndicesCsrCsr(TestJacobianSrcIndicesDenseDense):
    def setUp(self):
        self.prob = self._setup_model(CsrMatrix, csr_matrix)

class TestJacobianSrcIndicesCsrDense(TestJacobianSrcIndicesDenseDense):
    def setUp(self):
        self.prob = self._setup_model(CsrMatrix, np.array)

class TestJacobianSrcIndicesCsrList(TestJacobianSrcIndicesDenseDense):
    def setUp(self):
        self.prob = self._setup_model(CsrMatrix, arr2list)
