"""Tests for DefaultAutoscaler."""
import unittest
import numpy as np
from openmdao.drivers.autoscalers.autoscaler import Autoscaler
from openmdao.vectors.optimizer_vector import OptimizerVector


class TestAutoscaler(unittest.TestCase):
    """Test basic DefaultAutoscaler operations with mock drivers."""

    def test_apply_scaling_design_vars_with_scaler_and_adder(self):
        """Test scaling design variables with both scaler and adder."""
        # Create a mock driver with known scaling values
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x': {'total_scaler': 2.0, 'total_adder': 1.0}
                }
                self._cons = {}
                self._objs = {}

        driver = MockDriver()

        # Create autoscaler and setup
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        # Create design variable vector in model space
        meta = {'x': {'start_idx': 0, 'end_idx': 1, 'size': 1}}
        model_value = np.array([3.0])  # Model space value
        dv_vector = OptimizerVector('design_var', model_value.copy(), meta)

        # Scale: optimizer = (model + adder) * scaler = (3.0 + 1.0) * 2.0 = 8.0
        autoscaler._apply_vec_scaling(dv_vector)
        scaled_value = dv_vector.asarray()[0]
        self.assertAlmostEqual(scaled_value, 8.0, places=10)

    def test_apply_unscaling_design_vars_with_scaler_and_adder(self):
        """Test unscaling design variables with both scaler and adder."""
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x': {'total_scaler': 2.0, 'total_adder': 1.0}
                }
                self._cons = {}
                self._objs = {}

        driver = MockDriver()

        # Create autoscaler and setup
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        # Create design variable vector in optimizer space
        meta = {'x': {'start_idx': 0, 'end_idx': 1, 'size': 1}}
        optimizer_value = np.array([8.0])  # Optimizer space value
        dv_vector = OptimizerVector('design_var', optimizer_value.copy(), meta, driver_scaling=True)

        # Unscale: model = optimizer / scaler - adder = 8.0 / 2.0 - 1.0 = 3.0
        unscaled = autoscaler._apply_vec_unscaling(dv_vector)
        self.assertAlmostEqual(unscaled.asarray()[0], 3.0, places=10)

    def test_apply_scaling_with_none_scaler_and_adder(self):
        """Test scaling when scaler or adder is None."""
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x': {'total_scaler': None, 'total_adder': None}
                }
                self._cons = {}
                self._objs = {}

        driver = MockDriver()
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        meta = {'x': {'start_idx': 0, 'end_idx': 1, 'size': 1}}
        original_value = np.array([3.0])
        dv_vector = OptimizerVector('design_var', original_value.copy(), meta)

        # With None values, scaling should have no effect
        autoscaler._apply_vec_scaling(dv_vector)
        scaled_value = dv_vector.asarray()[0]
        self.assertAlmostEqual(scaled_value, 3.0, places=10)

    def test_apply_scaling_with_only_scaler(self):
        """Test scaling when only scaler is provided (adder is None)."""
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x': {'total_scaler': 2.0, 'total_adder': None}
                }
                self._cons = {}
                self._objs = {}

        driver = MockDriver()
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        meta = {'x': {'start_idx': 0, 'end_idx': 1, 'size': 1}}
        model_value = np.array([3.0])
        dv_vector = OptimizerVector('design_var', model_value.copy(), meta)

        # Scale: optimizer = (model + None) * scaler = 3.0 * 2.0 = 6.0
        autoscaler._apply_vec_scaling(dv_vector)
        scaled_value = dv_vector.asarray()[0]
        self.assertAlmostEqual(scaled_value, 6.0, places=10)

    def test_apply_scaling_with_only_adder(self):
        """Test scaling when only adder is provided (scaler is None)."""
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x': {'total_scaler': None, 'total_adder': 1.0}
                }
                self._cons = {}
                self._objs = {}

        driver = MockDriver()
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        meta = {'x': {'start_idx': 0, 'end_idx': 1, 'size': 1}}
        model_value = np.array([3.0])
        dv_vector = OptimizerVector('design_var', model_value.copy(), meta)

        # Scale: optimizer = (model + adder) * None = (3.0 + 1.0) = 4.0
        autoscaler._apply_vec_scaling(dv_vector)
        scaled_value = dv_vector.asarray()[0]
        self.assertAlmostEqual(scaled_value, 4.0, places=10)

    def test_in_place_modification_apply_scaling(self):
        """Test that apply_scaling modifies vector._data in-place."""
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x': {'total_scaler': 2.0, 'total_adder': 1.0}
                }
                self._cons = {}
                self._objs = {}

        driver = MockDriver()
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        meta = {'x': {'start_idx': 0, 'end_idx': 1, 'size': 1}}
        data = np.array([3.0])
        dv_vector = OptimizerVector('design_var', data, meta)

        # Get reference to data
        data_ref = dv_vector.asarray()
        initial_id = id(data_ref)

        # Scale
        autoscaler._apply_vec_scaling(dv_vector)

        # Verify data is modified in-place
        final_id = id(dv_vector.asarray())
        self.assertEqual(initial_id, final_id)
        self.assertAlmostEqual(dv_vector.asarray()[0], 8.0, places=10)

    def test_multiple_variables_apply_scaling(self):
        """Test scaling with multiple variables."""
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x1': {'total_scaler': 2.0, 'total_adder': 0.0},
                    'x2': {'total_scaler': 1.0, 'total_adder': 5.0}
                }
                self._cons = {}
                self._objs = {}

        driver = MockDriver()
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        meta = {
            'x1': {'start_idx': 0, 'end_idx': 1, 'size': 1},
            'x2': {'start_idx': 1, 'end_idx': 2, 'size': 1}
        }
        data = np.array([3.0, 10.0])
        dv_vector = OptimizerVector('design_var', data, meta)

        # Scale
        autoscaler._apply_vec_scaling(dv_vector)

        # x1: (3.0 + 0.0) * 2.0 = 6.0
        # x2: (10.0 + 5.0) * 1.0 = 15.0
        np.testing.assert_array_almost_equal(dv_vector.asarray(), [6.0, 15.0], decimal=10)

    def test_scaling_unscaling_roundtrip_design_vars(self):
        """Test that scaling followed by unscaling returns original values."""
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x': {'total_scaler': 2.0, 'total_adder': 1.0}
                }
                self._cons = {}
                self._objs = {}

        driver = MockDriver()
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        meta = {'x': {'start_idx': 0, 'end_idx': 1, 'size': 1}}
        original_value = np.array([3.0])
        dv_vector = OptimizerVector('design_var', original_value.copy(), meta)
        original_data = dv_vector.asarray().copy()

        # Scale: (3.0 + 1.0) * 2.0 = 8.0
        autoscaler._apply_vec_scaling(dv_vector)
        self.assertAlmostEqual(dv_vector.asarray()[0], 8.0, places=10)

        # Unscale: 8.0 / 2.0 - 1.0 = 3.0
        unscaled = autoscaler._apply_vec_unscaling(dv_vector)
        np.testing.assert_array_almost_equal(unscaled.asarray(), original_data, decimal=10)

    def test_apply_scaling_constraints(self):
        """Test scaling constraint vectors."""
        class MockDriver:
            def __init__(self):
                self._designvars = {}
                self._cons = {
                    'c1': {'total_scaler': 0.5, 'total_adder': 2.0},
                    'c2': {'total_scaler': 1.0, 'total_adder': None}
                }
                self._objs = {}

        driver = MockDriver()
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        meta = {
            'c1': {'start_idx': 0, 'end_idx': 1, 'size': 1},
            'c2': {'start_idx': 1, 'end_idx': 2, 'size': 1}
        }
        data = np.array([5.0, 10.0])
        cons_vector = OptimizerVector('constraint', data, meta)

        # Scale
        autoscaler._apply_vec_scaling(cons_vector)

        # c1: (5.0 + 2.0) * 0.5 = 3.5
        # c2: (10.0 + None) * 1.0 = 10.0
        np.testing.assert_array_almost_equal(cons_vector.asarray(), [3.5, 10.0], decimal=10)

    def test_apply_scaling_objectives(self):
        """Test scaling objective vectors."""
        class MockDriver:
            def __init__(self):
                self._designvars = {}
                self._cons = {}
                self._objs = {
                    'obj': {'total_scaler': 10.0, 'total_adder': -5.0}
                }

        driver = MockDriver()
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        meta = {
            'obj': {'start_idx': 0, 'end_idx': 1, 'size': 1}
        }
        data = np.array([100.0])
        objs_vector = OptimizerVector('objective', data, meta)

        # Scale: (100.0 + (-5.0)) * 10.0 = 950.0
        autoscaler._apply_vec_scaling(objs_vector)
        self.assertAlmostEqual(objs_vector.asarray()[0], 950.0, places=10)


class TestLagrangeMultiplierUnscaling(unittest.TestCase):
    """Test Lagrange multiplier unscaling."""

    def test_unscale_lagrange_multipliers_design_vars_only(self):
        """Test unscaling Lagrange multipliers for design variables only."""
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x': {'total_scaler': 2.0, 'total_adder': 1.0}
                }
                self._cons = {}
                self._objs = {
                    'obj': {'total_scaler': 4.0, 'total_adder': None}
                }

        driver = MockDriver()
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        # Design variable multiplier in optimizer space
        desvar_mults = {'x': np.array([8.0])}
        con_mults = {}

        # Unscale: λ = (scaler_x / scaler_f) * λ_scaled = (2.0 / 4.0) * 8.0 = 4.0
        desvar_unscaled, con_unscaled = autoscaler.apply_mult_unscaling(desvar_mults, con_mults)

        self.assertAlmostEqual(desvar_mults['x'][0], 4.0, places=10)
        self.assertEqual(len(con_mults), 0)

    def test_unscale_lagrange_multipliers_constraints_only(self):
        """Test unscaling Lagrange multipliers for constraints only."""
        class MockDriver:
            def __init__(self):
                self._designvars = {}
                self._cons = {
                    'c': {'total_scaler': 0.5, 'total_adder': None}
                }
                self._objs = {
                    'obj': {'total_scaler': 2.0, 'total_adder': None}
                }

        driver = MockDriver()
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        # Constraint multiplier in optimizer space
        desvar_mults = {}
        con_mults = {'c': np.array([10.0])}

        # Unscale: λ = (scaler_c / scaler_f) * λ_scaled = (0.5 / 2.0) * 10.0 = 2.5
        desvar_unscaled, con_unscaled = autoscaler.apply_mult_unscaling(desvar_mults, con_mults)

        self.assertAlmostEqual(con_mults['c'][0], 2.5, places=10)
        self.assertEqual(len(desvar_mults), 0)

    def test_unscale_lagrange_multipliers_both(self):
        """Test unscaling Lagrange multipliers for both design variables and constraints."""
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x': {'total_scaler': 2.0, 'total_adder': None}
                }
                self._cons = {
                    'c': {'total_scaler': 0.5, 'total_adder': None}
                }
                self._objs = {
                    'obj': {'total_scaler': 4.0, 'total_adder': None}
                }

        driver = MockDriver()
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        # Multipliers in optimizer space
        desvar_mults = {'x': np.array([8.0])}
        con_mults = {'c': np.array([10.0])}

        # Unscale
        desvar_unscaled, con_unscaled = autoscaler.apply_mult_unscaling(desvar_mults, con_mults)

        # Design var: λ = (2.0 / 4.0) * 8.0 = 4.0
        self.assertAlmostEqual(desvar_mults['x'][0], 4.0, places=10)
        # Constraint: λ = (0.5 / 4.0) * 10.0 = 1.25
        self.assertAlmostEqual(con_mults['c'][0], 1.25, places=10)

    def test_unscale_lagrange_multipliers_with_none_scaler(self):
        """Test unscaling when scaler is None (defaults to 1.0)."""
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x': {'total_scaler': None, 'total_adder': None}
                }
                self._cons = {}
                self._objs = {
                    'obj': {'total_scaler': None, 'total_adder': None}
                }

        driver = MockDriver()
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        desvar_mults = {'x': np.array([8.0])}
        con_mults = {}

        # Unscale: λ = (1.0 / 1.0) * 8.0 = 8.0
        desvar_unscaled, con_unscaled = autoscaler.apply_mult_unscaling(desvar_mults, con_mults)

        self.assertAlmostEqual(desvar_mults['x'][0], 8.0, places=10)

    def test_unscale_lagrange_multipliers_in_place_modification(self):
        """Test that unscale_lagrange_multipliers modifies dicts in-place."""
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x': {'total_scaler': 2.0, 'total_adder': None}
                }
                self._cons = {}
                self._objs = {
                    'obj': {'total_scaler': 4.0, 'total_adder': None}
                }

        driver = MockDriver()
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        desvar_mults = {'x': np.array([8.0])}
        con_mults = {}

        # Get references before
        desvar_ref = desvar_mults['x']
        desvar_id = id(desvar_ref)

        # Unscale
        desvar_unscaled, con_unscaled = autoscaler.apply_mult_unscaling(desvar_mults, con_mults)

        # Verify in-place modification
        self.assertIs(desvar_unscaled, desvar_mults)
        self.assertEqual(id(desvar_mults['x']), desvar_id)
        self.assertAlmostEqual(desvar_mults['x'][0], 4.0, places=10)

    def test_unscale_lagrange_multipliers_multiple_variables(self):
        """Test unscaling with multiple design variables and constraints."""
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x1': {'total_scaler': 2.0, 'total_adder': None},
                    'x2': {'total_scaler': 4.0, 'total_adder': None}
                }
                self._cons = {
                    'c1': {'total_scaler': 0.5, 'total_adder': None},
                    'c2': {'total_scaler': 1.0, 'total_adder': None}
                }
                self._objs = {
                    'obj': {'total_scaler': 8.0, 'total_adder': None}
                }

        driver = MockDriver()
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        desvar_mults = {
            'x1': np.array([16.0]),
            'x2': np.array([32.0])
        }
        con_mults = {
            'c1': np.array([10.0]),
            'c2': np.array([20.0])
        }

        # Unscale
        desvar_unscaled, con_unscaled = autoscaler.apply_mult_unscaling(desvar_mults, con_mults)

        # Design vars: λ = (scaler_x / scaler_f) * λ_scaled
        # x1: (2.0 / 8.0) * 16.0 = 4.0
        # x2: (4.0 / 8.0) * 32.0 = 16.0
        self.assertAlmostEqual(desvar_mults['x1'][0], 4.0, places=10)
        self.assertAlmostEqual(desvar_mults['x2'][0], 16.0, places=10)

        # Constraints: λ = (scaler_c / scaler_f) * λ_scaled
        # c1: (0.5 / 8.0) * 10.0 = 0.625
        # c2: (1.0 / 8.0) * 20.0 = 2.5
        self.assertAlmostEqual(con_mults['c1'][0], 0.625, places=10)
        self.assertAlmostEqual(con_mults['c2'][0], 2.5, places=10)

    def test_unscale_lagrange_multipliers_empty_dicts(self):
        """Test unscaling with empty multiplier dictionaries."""
        class MockDriver:
            def __init__(self):
                self._designvars = {}
                self._cons = {}
                self._objs = {
                    'obj': {'total_scaler': 2.0, 'total_adder': None}
                }

        driver = MockDriver()
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        desvar_mults = {}
        con_mults = {}

        # Should not raise an error
        desvar_unscaled, con_unscaled = autoscaler.apply_mult_unscaling(desvar_mults, con_mults)

        self.assertEqual(len(desvar_unscaled), 0)
        self.assertEqual(len(con_unscaled), 0)


class TestScalingWithArrayValues(unittest.TestCase):
    """Test scaling with array-valued design variables and constraints."""

    def test_apply_scaling_array_valued_desvar(self):
        """Test scaling when design variable has array-valued scaler."""
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x': {'total_scaler': np.array([2.0, 3.0]), 'total_adder': np.array([1.0, 0.5])}
                }
                self._cons = {}
                self._objs = {}

        driver = MockDriver()
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        meta = {'x': {'start_idx': 0, 'end_idx': 2, 'size': 2}}
        model_value = np.array([10.0, 20.0])
        dv_vector = OptimizerVector('design_var', model_value.copy(), meta)

        # Scale: (model + adder) * scaler
        # (10.0 + 1.0) * 2.0 = 22.0
        # (20.0 + 0.5) * 3.0 = 61.5
        autoscaler._apply_vec_scaling(dv_vector)
        expected = np.array([22.0, 61.5])
        np.testing.assert_array_almost_equal(dv_vector.asarray(), expected, decimal=10)

    def test_apply_unscaling_array_valued_desvar(self):
        """Test unscaling when design variable has array-valued scaler."""
        class MockDriver:
            def __init__(self):
                self._designvars = {
                    'x': {'total_scaler': np.array([2.0, 3.0]), 'total_adder': np.array([1.0, 0.5])}
                }
                self._cons = {}
                self._objs = {}

        driver = MockDriver()
        autoscaler = Autoscaler()
        autoscaler.setup(driver)

        meta = {'x': {'start_idx': 0, 'end_idx': 2, 'size': 2}}
        optimizer_value = np.array([22.0, 61.5])
        dv_vector = OptimizerVector('design_var', optimizer_value, meta, driver_scaling=True)

        # Unscale: optimizer / scaler - adder
        # 22.0 / 2.0 - 1.0 = 10.0
        # 61.5 / 3.0 - 0.5 = 20.0
        unscaled = autoscaler._apply_vec_unscaling(dv_vector)
        expected = np.array([10.0, 20.0])
        np.testing.assert_array_almost_equal(unscaled.asarray(), expected, decimal=10)


if __name__ == '__main__':
    unittest.main()
