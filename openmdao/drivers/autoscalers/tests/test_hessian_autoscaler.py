"""Tests for HessianAutoscaler."""

import unittest

import numpy as np

import openmdao
from openmdao.drivers.autoscalers.hessian_autoscaler import HessianAutoscaler
from openmdao.drivers.autoscalers.arnoldi_algorithm import arnoldi, extract_tridiag
import openmdao.api as om


class TestHessianAutoscaler(unittest.TestCase):
    """Tests of Hessian-based variable transformations."""
    
    def setUp(self):
        """Create the quadratic used by the tests."""
        self.A = np.array([
            [7, 9, 7, 7],
            [6, 5, 9, 8],
            [5, 3, 8, 6],
            [8, 1, 8, 1],
        ])
        self.Q, _ = np.linalg.qr(self.A)
        self.eigs = np.array([1.0, 0.1, 0.01, 0.001])
        self.start_vec = np.array([ 1.24531358, -1.41731239,  0.23793453, -1.16510183])

    def test_known_diagonal_scaling(self):
        """Tests if the transformations x = M @ y and y = M^{-1} @ x work correctly."""
        autoscaler = HessianAutoscaler(num_modes=2)
        autoscaler.eigenvals_m = np.array([16.0, 4.0])
        autoscaler.eigenvecs_m = np.eye(2)

        # For this Hessian, M = diag(1/sqrt(16), 1/sqrt(4)).
        # Therefore M @ [8, 6] = [2, 3].
        y = np.array([8.0, 6.0])
        expected_x = np.array([2.0, 3.0])

        x = autoscaler._compute_My(y)

        np.testing.assert_allclose(
            x, expected_x, rtol=1e-12, atol=1e-12
        )

        # The inverse transformation must recover the original y.
        recovered_y = autoscaler._compute_M_inverse_x(expected_x)

        np.testing.assert_allclose(
            recovered_y, y, rtol=1e-12, atol=1e-12
        )
    
    def _obj_func(self, x):
        """A quadratic objective objective function for testing."""
        return 0.5 * x.T @ self.Q @ np.diag(self.eigs) @ self.Q.T @ x
    
    def _grad_obj_func(self, x):
        """Gradient of the quadratic objective function."""
        return self.Q @ np.diag(self.eigs) @ self.Q.T @ x
    
    def test_arnoldi_algorithm(self):
        # Test arnoldi algorithm with a known matrix and initial vector.
        autoscaler = HessianAutoscaler(num_modes=3)
        eigenvals_m, eigenvecs_m = autoscaler._get_eigen_decomp(
            self._grad_obj_func, 3, np.ones(4), False, self.start_vec)
        
        # eigenvectors orthonormality check and eigenvalues accuracy check
        np.testing.assert_array_almost_equal(eigenvecs_m.T @ eigenvecs_m, np.eye(3), decimal=10)
        np.testing.assert_array_almost_equal(eigenvals_m, self.eigs[:3], decimal=8)
    
    def test_unconstrained_quadratic_optimization(self):
        """Verify that the autoscaler preserves the quadratic's known optimum."""
        H = self.Q @ np.diag(self.eigs) @ self.Q.T

        class Quadratic(om.ExplicitComponent):
            """Quadratic objective with an exact gradient."""

            def setup(self):
                self.add_input('x', val=np.ones(4))
                self.add_output('f', val=0.0)
                self.declare_partials('f', 'x')

            def compute(self, inputs, outputs):
                x = inputs['x']
                outputs['f'] = 0.5 * x @ H @ x

            def compute_partials(self, inputs, partials):
                partials['f', 'x'] = (H @ inputs['x']).reshape(1, 4)

        prob = om.Problem(reports=False)
        self.addCleanup(prob.cleanup)

        prob.model.add_subsystem('quad', Quadratic(), promotes=['*'])
        prob.model.add_design_var('x')
        prob.model.add_objective('f')

        prob.driver = om.ScipyOptimizeDriver(
            optimizer='SLSQP',
            tol=1e-12,
            disp=False,
        )
        prob.driver.options['maxiter'] = 100
        prob.driver.autoscaler = HessianAutoscaler(
            num_modes=3,
            num_krylov=4,
            start_vec=self.start_vec,
        )

        prob.setup()
        prob.set_val('x', np.ones(4))

        result = prob.run_driver()

        self.assertTrue(result.success, str(result))
        np.testing.assert_allclose(
            prob.get_val('x'), np.zeros(4), rtol=0.0, atol=1e-10,
        )
        np.testing.assert_allclose(
            prob.get_val('f'), 0.0, rtol=0.0, atol=1e-10,
        )
        
    def test_multiple_dvs_with_nonlinear_constraints(self):
        """Verify that the autoscaler works with and preserves the optimum of a 
        problem with multiple DVs and nonlinear constraints."""
        
        class MultiDVNonlinearProblem(om.ExplicitComponent):
            """
            Design variables:
                x : shape (2,)
                z : shape (3,)

            Objective:
                smooth quadratic with cross-coupling

            Constraints:
                g : shape (3,), nonlinear
            """

            def setup(self):
                self.add_input('x', shape=2)
                self.add_input('z', shape=3)

                self.add_output('f', val=0.0)
                self.add_output('g', shape=3)

                self.declare_partials('f', 'x')
                self.declare_partials('f', 'z')
                self.declare_partials('g', 'x')
                self.declare_partials('g', 'z')

            def compute(self, inputs, outputs):
                x = inputs['x']
                z = inputs['z']

                x0, x1 = x
                z0, z1, z2 = z

                outputs['f'] = (
                    (x0 - 1.0)**2
                    + 2.0 * (x1 + 1.0)**2
                    + (z0 - 0.5)**2
                    + 3.0 * (z1 - 1.5)**2
                    + 0.5 * (z2 + 0.5)**2
                    + 0.2 * x0 * z0
                    + 0.1 * x1 * z2
                )

                outputs['g'][0] = x0**2 + z0**2
                outputs['g'][1] = np.sin(x1) + z1 * z2
                outputs['g'][2] = (x0 + z2)**2 + np.exp(0.1 * z1)

            def compute_partials(self, inputs, partials):
                x = inputs['x']
                z = inputs['z']

                x0, x1 = x
                z0, z1, z2 = z

                partials['f', 'x'] = np.array([
                    2.0 * (x0 - 1.0) + 0.2 * z0,
                    4.0 * (x1 + 1.0) + 0.1 * z2,
                ])

                partials['f', 'z'] = np.array([
                    2.0 * (z0 - 0.5) + 0.2 * x0,
                    6.0 * (z1 - 1.5),
                    (z2 + 0.5) + 0.1 * x1,
                ])

                partials['g', 'x'] = np.array([
                    [2.0 * x0, 0.0],
                    [0.0, np.cos(x1)],
                    [2.0 * (x0 + z2), 0.0],
                ])

                partials['g', 'z'] = np.array([
                    [2.0 * z0, 0.0, 0.0],
                    [0.0, z2, z1],
                    [0.0, 0.1 * np.exp(0.1 * z1), 2.0 * (x0 + z2)],
                ])
                
        prob = om.Problem(reports=False)
        self.addCleanup(prob.cleanup)

        ivc = om.IndepVarComp()
        ivc.add_output('x', val=np.array([2.0, 0.0]))
        ivc.add_output('z', val=np.array([1.5, 0.5, 1.0]))

        prob.model.add_subsystem('ivc', ivc, promotes=['x', 'z'])
        prob.model.add_subsystem(
            'comp',
            MultiDVNonlinearProblem(),
            promotes=['x', 'z', 'f', 'g'],
        )

        prob.driver = om.ScipyOptimizeDriver(
            optimizer='SLSQP',
            tol=1e-12,
            disp=False,
        )
        prob.driver.options['maxiter'] = 200
        prob.driver.autoscaler = HessianAutoscaler(
            num_modes=3,
            num_krylov=4,
            start_vec=np.ones(5),
        )

        prob.model.add_design_var('x')
        prob.model.add_design_var('z')
    
        prob.model.add_objective('f')
        
        prob.model.add_constraint(
            'g',
            lower=np.array([-1.0e30, -2.0, -1.0e30]),
            upper=np.array([1.1, 1.0e30, 6.0]),
        )
        prob.setup()
        prob.run_driver()

        result = prob.run_driver()

        self.assertTrue(result.success, str(result))
        np.testing.assert_allclose(
            prob.get_val('x'), np.array([0.959595980676, -0.989974909468]), 
            rtol=0.0, atol=1e-10,
        )
        np.testing.assert_allclose(
            prob.get_val('z'), np.array([0.404040441479, 1.499999972608, -0.401002490665]), 
            rtol=0.0, atol=1e-10,
        )
        self.assertAlmostEqual(prob.get_val('f')[0], 0.133183337131, places=10)
        np.testing.assert_allclose(
            prob.get_val('g'), np.array([1.08407312448, -1.43751593643, 1.473860926628]), 
            rtol=0.0, atol=1e-10,
        )
        
    def test_design_var_bounds(self):
        """ Test that design variables with bounds are handled correctly by the autoscaler.
            Variables with bounds should be transformed to general constraints"""
        prob = om.Problem(reports=False)

        prob.model.add_subsystem(
            'quad',
            om.ExecComp('f = (x - 3)**2 + (z + 2)**2'),
            promotes=['*'],
        )

        prob.model.add_design_var('x', lower=0.0, upper=2.0)
        prob.model.add_design_var('z', lower=-1.0, upper=1.0)
        prob.model.add_objective('f')

        prob.driver = om.ScipyOptimizeDriver(
            optimizer='SLSQP', tol=1e-12, disp=False,
        )
        prob.driver.autoscaler = HessianAutoscaler(num_modes=2)
        prob.setup()
        prob.driver.autoscaler.transform_desvar_bounds(prob)
        prob.set_val('x', 1.0)
        prob.set_val('z', 0.0)

        prob.run_driver()

        # print('Success:', result.success)

        self.assertAlmostEqual(prob.get_val('x')[0], 2.0, places=10)
        self.assertAlmostEqual(prob.get_val('z')[0], -1.0, places=10)
        self.assertAlmostEqual(prob.get_val('f')[0], 2.0, places=10)
        
        expected_bounds = {'x': (0.0, 2.0), 'z': (-1.0, 1.0)}

        design_vars = prob.model.get_design_vars()
        constraints = prob.model.get_constraints()
        for name, (lower, upper) in expected_bounds.items():
            dv = design_vars[name]
            con = constraints[name]

            # check if bounds have been removed from the design variables
            self.assertTrue(np.all(np.isneginf(dv['lower'])))
            self.assertTrue(np.all(np.isposinf(dv['upper'])))

            # check if the bounds have been added as constraints
            np.testing.assert_allclose(con['lower'], lower)
            np.testing.assert_allclose(con['upper'], upper)
            self.assertIsNone(con['equals'])
            self.assertTrue(con['linear'])

        prob.cleanup()
        

if __name__ == '__main__':
    unittest.main()