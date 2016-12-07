
import unittest

import numpy as np
from scipy.sparse import coo_matrix, issparse

from openmdao.api import IndepVarComp, Group, Problem, ExplicitComponent, DenseMatrix, \
     GlobalJacobian, NewtonSolver, ScipyIterativeSolver
from openmdao.devtools.testutil import assert_rel_error


class MyExplicitComp(ExplicitComponent):

    def initialize_variables(self):
        self.add_input('x', val=np.zeros(2))
        self.add_output('f', val=0.0)

    def compute(self, inputs, outputs):
        x = inputs['x']
        outputs['f'] = (x[0]-3.0)**2 + x[0]*x[1] + (x[1]+4.0)**2 - 3.0

    def compute_jacobian(self, inputs, outputs, jacobian):
        x = inputs['x']
        jacobian['f', 'x'] = np.array([[
            2.0*x[0] - 6.0 + x[1],
            2.0*x[1] + 8.0 + x[0]
        ]])


class TestSparseJacobian(unittest.TestCase):

    def test_simple(self):
        prob = Problem(root=Group())
        prob.root.add_subsystem('indep',
                                IndepVarComp((
                                    ('a', np.ones(2)),
                                )))
        C1 = prob.root.add_subsystem('C1', MyExplicitComp())
        prob.root.connect('indep.a', 'C1.x')
        prob.setup(check=False)
        prob.root.jacobian = GlobalJacobian(Matrix=DenseMatrix)
        prob.root.nl_solver = NewtonSolver(
            subsolvers={'linear': ScipyIterativeSolver(
                maxiter=100,
            )}
        )
        prob.root.ln_solver = ScipyIterativeSolver(
            maxiter=200, atol=1e-10, rtol=1e-10)
        prob.root.suppress_solver_output = True

        prob.run()

        work = prob.root._vectors['output']['']._clone()
        work.set_const(1.0)

        # 1. fwd apply_linear test
        prob.root._vectors['output'][''].set_const(1.0)
        prob.root._apply_linear([''], 'fwd')
        res = prob.root._vectors['residual']['']
        # if we multiply our jacobian by our work vec of 1's, we get [1.0, 1.0, -7.0]
        res.set_data(res.get_data()-np.array([1.0, 1.0, -7.]))
        self.assertAlmostEqual(
            prob.root._vectors['residual'][''].get_norm(), 0)

        # 2. rev apply_linear test
        prob.root._vectors['residual'][''].set_const(1.0)
        prob.root._apply_linear([''], 'rev')
        outs = prob.root._vectors['output']['']
        # if we multiply our jacobian's transpose by our work vec of 1's, 
        # we get [4., -10., 1.]
        outs.set_data(outs.get_data()-np.array([4., -10., 1.]))
        self.assertAlmostEqual(
            prob.root._vectors['output'][''].get_norm(), 0)

        # 3. fwd solve_linear test
        prob.root._vectors['output'][''].set_const(0.0)
        prob.root._vectors['residual'][''].set_data(np.array([1.0, 1.0, -7.]))
        prob.root._solve_linear([''], 'fwd')
        prob.root._vectors['output'][''] -= work
        self.assertAlmostEqual(
            prob.root._vectors['output'][''].get_norm(), 0, delta=1e-6)

        # 4. rev solve_linear test
        prob.root._vectors['residual'][''].set_const(0.0)
        prob.root._vectors['output'][''].set_data(np.array([4., -10., 1.]))
        prob.root._solve_linear([''], 'rev')
        prob.root._vectors['residual'][''] -= work
        self.assertAlmostEqual(
            prob.root._vectors['residual'][''].get_norm(), 0, delta=1e-6)
