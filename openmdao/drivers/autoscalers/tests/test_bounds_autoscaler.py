"""Tests for BoundsAutoscaler."""
import unittest

import numpy as np

import openmdao.api as om
from openmdao.drivers.autoscalers.bounds_autoscaler import BoundsAutoscaler
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.utils.assert_utils import assert_near_equal


class TestBoundsAutoscalerMock(unittest.TestCase):
    """Unit tests for BoundsAutoscaler using mock drivers."""

    def _make_mock(self, dv_meta, cons_meta=None, objs_meta=None):
        class MockDriver:
            def __init__(self):
                self._designvars = dv_meta
                self._cons = cons_meta or {}
                self._objs = objs_meta or {}

        return MockDriver()

    def test_scalar_bounds_scaler_and_adder(self):
        driver = self._make_mock({
            'x': {'total_scaler': 5.0, 'total_adder': 2.0,
                  'lower': -3.0, 'upper': 7.0, 'size': 1,
                  'distributed': False}
        })

        autoscaler = BoundsAutoscaler()
        autoscaler.setup(driver)

        meta = autoscaler._var_meta['design_var']['x']
        # 1 / (7 - (-3)) = 0.1, adder = 3.0
        self.assertAlmostEqual(meta['total_scaler'], 0.1, places=12)
        self.assertAlmostEqual(meta['total_adder'], 3.0, places=12)

        # Original driver metadata must not be mutated.
        self.assertEqual(driver._designvars['x']['total_scaler'], 5.0)
        self.assertEqual(driver._designvars['x']['total_adder'], 2.0)

        # Scaled bounds should be [0, 1].
        lower_vec, upper_vec, _ = autoscaler.get_bounds_scaling('design_var')
        self.assertAlmostEqual(lower_vec['x'][0], 0.0, places=12)
        self.assertAlmostEqual(upper_vec['x'][0], 1.0, places=12)

    def test_vector_bounds(self):
        driver = self._make_mock({
            'x': {'total_scaler': None, 'total_adder': None,
                  'lower': np.array([0.0, -10.0, 5.0]),
                  'upper': np.array([1.0, 10.0, 25.0]),
                  'size': 3, 'distributed': False}
        })

        autoscaler = BoundsAutoscaler()
        autoscaler.setup(driver)

        meta = autoscaler._var_meta['design_var']['x']
        np.testing.assert_allclose(meta['total_scaler'],
                                   np.array([1.0, 1.0 / 20.0, 1.0 / 20.0]))
        np.testing.assert_allclose(meta['total_adder'],
                                   np.array([0.0, 10.0, -5.0]))

        lower_vec, upper_vec, _ = autoscaler.get_bounds_scaling('design_var')
        np.testing.assert_allclose(lower_vec['x'], np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(upper_vec['x'], np.ones(3), atol=1e-12)

    def test_scalar_bound_broadcasts_over_size(self):
        driver = self._make_mock({
            'x': {'total_scaler': None, 'total_adder': None,
                  'lower': -2.0, 'upper': 2.0,
                  'size': 4, 'distributed': False}
        })

        autoscaler = BoundsAutoscaler()
        autoscaler.setup(driver)

        meta = autoscaler._var_meta['design_var']['x']
        # For array-sized DVs, scaler/adder become arrays.
        np.testing.assert_allclose(meta['total_scaler'], np.full(4, 0.25))
        np.testing.assert_allclose(meta['total_adder'], np.full(4, 2.0))

    def test_infinite_lower_bound_raises(self):
        from openmdao.core.constants import INF_BOUND
        driver = self._make_mock({
            'x': {'total_scaler': None, 'total_adder': None,
                  'lower': -INF_BOUND, 'upper': 1.0,
                  'size': 1, 'distributed': False}
        })

        autoscaler = BoundsAutoscaler()
        with self.assertRaises(RuntimeError) as ctx:
            autoscaler.setup(driver)
        self.assertIn("finite lower and upper bounds", str(ctx.exception))

    def test_infinite_upper_bound_raises(self):
        from openmdao.core.constants import INF_BOUND
        driver = self._make_mock({
            'x': {'total_scaler': None, 'total_adder': None,
                  'lower': 0.0, 'upper': INF_BOUND,
                  'size': 1, 'distributed': False}
        })

        autoscaler = BoundsAutoscaler()
        with self.assertRaises(RuntimeError) as ctx:
            autoscaler.setup(driver)
        self.assertIn("finite lower and upper bounds", str(ctx.exception))

    def test_zero_range_raises(self):
        driver = self._make_mock({
            'x': {'total_scaler': None, 'total_adder': None,
                  'lower': 1.0, 'upper': 1.0,
                  'size': 1, 'distributed': False}
        })

        autoscaler = BoundsAutoscaler()
        with self.assertRaises(RuntimeError) as ctx:
            autoscaler.setup(driver)
        self.assertIn("upper > lower", str(ctx.exception))

    def test_discrete_dv_skipped(self):
        driver = self._make_mock({
            'x': {'total_scaler': 3.0, 'total_adder': -1.0,
                  'discrete': True, 'size': 1, 'distributed': False}
        })

        autoscaler = BoundsAutoscaler()
        autoscaler.setup(driver)

        # Discrete DVs are passed through unchanged.
        meta = autoscaler._var_meta['design_var']['x']
        self.assertEqual(meta['total_scaler'], 3.0)
        self.assertEqual(meta['total_adder'], -1.0)


class TestBoundsAutoscalerIntegration(unittest.TestCase):
    """Integration tests for BoundsAutoscaler with ScipyOptimizeDriver."""

    def test_COBYLA_multiscale(self):
        # Verifies that BoundsAutoscaler lets COBYLA's scalar rhobeg option produce a
        # meaningful step for every design variable when the DVs span very different
        # orders of magnitude.

        class ScaledParaboloid(om.ExplicitComponent):
            def setup(self):
                self.add_input('x_big', val=100.0)
                self.add_input('y_ratio', val=0.8)
                self.add_output('f', val=0.0)

            def compute(self, inputs, outputs):
                # Minimum at x_big = 1500, y_ratio = 0.3.
                x = inputs['x_big']
                y = inputs['y_ratio']
                outputs['f'] = (x - 1500.0) ** 2 / 1.0e6 + (y - 0.3) ** 2

        prob = om.Problem()
        prob.model.add_subsystem('comp', ScaledParaboloid(), promotes=['*'])
        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='COBYLA', tol=1e-8, disp=False)
        prob.driver.autoscaler = BoundsAutoscaler()
        # rhobeg = 0.1 -> 10% of (upper - lower) for each DV.
        prob.driver.opt_settings['rhobeg'] = 0.1

        prob.model.add_design_var('x_big', lower=0.0, upper=3000.0)
        prob.model.add_design_var('y_ratio', lower=0.0, upper=1.0)
        prob.model.add_objective('f')

        prob.setup()
        prob.set_val('x_big', 100.0)
        prob.set_val('y_ratio', 0.8)

        failed = not prob.run_driver().success
        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver._scipy_optimize_result))

        assert_near_equal(prob.get_val('x_big'), 1500.0, 1e-3)
        assert_near_equal(prob.get_val('y_ratio'), 0.3, 1e-3)

        # Sanity check: without the BoundsAutoscaler and the same small rhobeg, COBYLA
        # doesn't move the large-magnitude DV enough to reach the optimum.
        prob2 = om.Problem()
        prob2.model.add_subsystem('comp', ScaledParaboloid(), promotes=['*'])
        prob2.set_solver_print(level=0)

        prob2.driver = om.ScipyOptimizeDriver(optimizer='COBYLA', tol=1e-8, disp=False)
        prob2.driver.opt_settings['rhobeg'] = 0.1
        prob2.driver.options['maxiter'] = 200

        prob2.model.add_design_var('x_big', lower=0.0, upper=3000.0)
        prob2.model.add_design_var('y_ratio', lower=0.0, upper=1.0)
        prob2.model.add_objective('f')

        prob2.setup()
        prob2.set_val('x_big', 100.0)
        prob2.set_val('y_ratio', 0.8)
        prob2.run_driver()

        self.assertGreater(abs(prob2.get_val('x_big').item() - 1500.0), 100.0,
                           "Unnormalized COBYLA with tiny rhobeg unexpectedly reached "
                           "the optimum; the multiscale test is not exercising the "
                           "intended code path.")

    def test_SLSQP_matches_unnormalized(self):
        # Using BoundsAutoscaler with a gradient-based optimizer should not change the
        # final solution; the base Autoscaler machinery applies the correct chain rule
        # to totals via apply_jac_scaling.

        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', val=50.)
        model.set_input_defaults('y', val=50.)
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)
        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)
        prob.driver.autoscaler = BoundsAutoscaler()

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup()
        failed = not prob.run_driver().success
        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver._scipy_optimize_result))

        # Same expected minimum as the unnormalized test at (7.166667, -7.833334).
        assert_near_equal(prob['x'], 7.16667, 1e-5)
        assert_near_equal(prob['y'], -7.833334, 1e-5)

    def test_requires_finite_bounds(self):
        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', val=50.)
        model.set_input_defaults('y', val=50.)
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)
        prob.driver = om.ScipyOptimizeDriver(optimizer='COBYLA', disp=False)
        prob.driver.autoscaler = BoundsAutoscaler()

        # x has no bounds -- BoundsAutoscaler should reject this configuration.
        model.add_design_var('x')
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup()

        with self.assertRaises(RuntimeError) as ctx:
            prob.run_driver()

        self.assertIn("finite lower and upper bounds", str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
