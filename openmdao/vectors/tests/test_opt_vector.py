"""Tests for OptimizerVector."""

import unittest

import numpy as np

from openmdao.vectors.optimizer_vector import OptimizerVector
from openmdao.utils.assert_utils import assert_near_equal


class TestOptimizerVector(unittest.TestCase):
    """Test OptimizerVector basic functionality."""

    def setUp(self):
        """Set up test data."""
        self.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
        self.metadata = {
            'x': {'start_idx': 0, 'end_idx': 2, 'size': 2},
            'y': {'start_idx': 2, 'end_idx': 4, 'size': 2},
            'z': {'start_idx': 4, 'end_idx': 6, 'size': 2},
        }

    def test_getitem_scalar(self):
        """Test getting a scalar variable."""
        data = np.array([1.5, 2.5, 3.5])
        metadata = {'x': {'start_idx': 0, 'end_idx': 1, 'size': 1}}
        vec = OptimizerVector('design_var', data, metadata)

        result = vec['x']
        assert_near_equal(result, np.array([1.5]))

    def test_getitem_vector(self):
        """Test getting a vector variable."""
        vec = OptimizerVector('design_var', self.data, self.metadata)

        result = vec['x']
        assert_near_equal(result, np.array([1.0, 2.0]))

        result = vec['y']
        assert_near_equal(result, np.array([3.0, 4.0]))

    def test_getitem_nonexistent(self):
        """Test KeyError for nonexistent variable."""
        vec = OptimizerVector('design_var', self.data, self.metadata)

        with self.assertRaises(KeyError) as cm:
            vec['nonexistent']

        self.assertIn('not found', str(cm.exception))

    def test_setitem_scalar(self):
        """Test setting a scalar variable."""
        data = np.array([1.5, 2.5, 3.5])
        metadata = {'x': {'start_idx': 0, 'end_idx': 1, 'size': 1}}
        vec = OptimizerVector('design_var', data, metadata)

        vec['x'] = 10.0
        assert_near_equal(data[0], 10.0)

    def test_setitem_vector(self):
        """Test setting a vector variable."""
        vec = OptimizerVector('design_var', self.data, self.metadata)

        vec['x'] = [10.0, 20.0]
        assert_near_equal(self.data[:2], np.array([10.0, 20.0]))

        vec['y'] = np.array([30.0, 40.0])
        assert_near_equal(self.data[2:4], np.array([30.0, 40.0]))

    def test_setitem_scalar_from_array(self):
        """Test setting a scalar variable from an array."""
        data = np.array([1.5, 2.5, 3.5])
        metadata = {'x': {'start_idx': 0, 'end_idx': 1, 'size': 1}}
        vec = OptimizerVector('design_var', data, metadata)

        vec['x'] = np.array([99.0])
        assert_near_equal(data[0], 99.0)

    def test_setitem_nonexistent(self):
        """Test KeyError when setting nonexistent variable."""
        vec = OptimizerVector('design_var', self.data, self.metadata)

        with self.assertRaises(KeyError) as cm:
            vec['nonexistent'] = 5.0

        self.assertIn('not found', str(cm.exception))

    def test_contains(self):
        """Test __contains__ method."""
        vec = OptimizerVector('design_var', self.data, self.metadata)

        self.assertIn('x', vec)
        self.assertIn('y', vec)
        self.assertIn('z', vec)
        self.assertNotIn('nonexistent', vec)

    def test_len(self):
        """Test __len__ method."""
        vec = OptimizerVector('design_var', self.data, self.metadata)
        self.assertEqual(len(vec), 3)

    def test_iter(self):
        """Test __iter__ method."""
        vec = OptimizerVector('design_var', self.data, self.metadata)

        names = list(vec)
        self.assertEqual(names, ['x', 'y', 'z'])

    def test_keys(self):
        """Test keys method."""
        vec = OptimizerVector('design_var', self.data, self.metadata)

        keys = list(vec.keys())
        self.assertEqual(keys, ['x', 'y', 'z'])

    def test_values(self):
        """Test values method."""
        vec = OptimizerVector('design_var', self.data, self.metadata)

        values = list(vec.values())
        self.assertEqual(len(values), 3)
        assert_near_equal(values[0], np.array([1.0, 2.0]))
        assert_near_equal(values[1], np.array([3.0, 4.0]))
        assert_near_equal(values[2], np.array([5.0, 6.0]))

    def test_items(self):
        """Test items method."""
        vec = OptimizerVector('design_var', self.data, self.metadata)

        items = list(vec.items())
        self.assertEqual(len(items), 3)

        self.assertEqual(items[0][0], 'x')
        assert_near_equal(items[0][1], np.array([1.0, 2.0]))

        self.assertEqual(items[1][0], 'y')
        assert_near_equal(items[1][1], np.array([3.0, 4.0]))

        self.assertEqual(items[2][0], 'z')
        assert_near_equal(items[2][1], np.array([5.0, 6.0]))

    def test_voi_type(self):
        """Test voi_type attribute."""
        vec1 = OptimizerVector('design_var', self.data, self.metadata)
        self.assertEqual(vec1.voi_type, 'design_var')

        vec2 = OptimizerVector('constraint', self.data, self.metadata)
        self.assertEqual(vec2.voi_type, 'constraint')

        vec3 = OptimizerVector('objective', self.data, self.metadata)
        self.assertEqual(vec3.voi_type, 'objective')


class TestOptimizerVectorAsarray(unittest.TestCase):
    """Test OptimizerVector.asarray method and filtering."""

    def setUp(self):
        """Set up test data with metadata for filtering."""
        self.data = np.arange(10, dtype=float)
        self.metadata = {
            'linear_eq1': {'start_idx': 0, 'end_idx': 2, 'size': 2, 'linear': True, 'equals': 5.0},
            'linear_ineq1': {'start_idx': 2, 'end_idx': 4, 'size': 2, 'linear': True, 'equals': None},
            'nonlinear_eq1': {'start_idx': 4, 'end_idx': 6, 'size': 2, 'linear': False, 'equals': 3.0},
            'nonlinear_ineq1': {'start_idx': 6, 'end_idx': 8, 'size': 2, 'linear': False, 'equals': None},
            'nonlinear_ineq2': {'start_idx': 8, 'end_idx': 10, 'size': 2, 'linear': False, 'equals': None},
        }

    def test_asarray_no_filter(self):
        """Test asarray without filters returns full array view."""
        vec = OptimizerVector('constraint', self.data, self.metadata)

        result = vec.asarray()
        assert_near_equal(result, self.data)

        # Verify it's a view (modifying result should modify data)
        result[0] = 999.0
        self.assertEqual(self.data[0], 999.0)

    def test_asarray_single_filter(self):
        """Test asarray with single filter criterion."""
        # Reset data
        self.data = np.arange(10, dtype=float)
        vec = OptimizerVector('constraint', self.data, self.metadata)

        # Filter for linear constraints only
        result = vec.asarray(linear=True)
        expected_indices = [0, 1, 2, 3]
        assert_near_equal(result, self.data[expected_indices])

    def test_asarray_single_filter_nonlinear(self):
        """Test asarray filtering for nonlinear constraints."""
        # Reset data
        self.data = np.arange(10, dtype=float)
        vec = OptimizerVector('constraint', self.data, self.metadata)

        # Filter for nonlinear constraints only
        result = vec.asarray(linear=False)
        expected_indices = [4, 5, 6, 7, 8, 9]
        assert_near_equal(result, self.data[expected_indices])

    def test_asarray_multiple_filters(self):
        """Test asarray with multiple filter criteria (AND logic)."""
        # Reset data
        self.data = np.arange(10, dtype=float)
        vec = OptimizerVector('constraint', self.data, self.metadata)

        # Filter for nonlinear equality constraints
        result = vec.asarray(linear=False, equals=3.0)
        expected_indices = [4, 5]
        assert_near_equal(result, self.data[expected_indices])

    def test_asarray_filter_inequalities(self):
        """Test asarray filtering for inequality constraints."""
        # Reset data
        self.data = np.arange(10, dtype=float)
        vec = OptimizerVector('constraint', self.data, self.metadata)

        # Filter for inequalities (equals=None)
        result = vec.asarray(equals=None)
        expected_indices = [2, 3, 6, 7, 8, 9]
        assert_near_equal(result, self.data[expected_indices])

    def test_asarray_no_match(self):
        """Test asarray when no variables match filters."""
        vec = OptimizerVector('constraint', self.data, self.metadata)

        # Filter for something that doesn't exist
        result = vec.asarray(linear=True, equals=999.0)
        self.assertEqual(len(result), 0)

    def test_asarray_filter_caching(self):
        """Test that filter results are cached."""
        vec = OptimizerVector('constraint', self.data, self.metadata)

        # First call computes filter
        result1 = vec.asarray(linear=True)

        # Check that cache key exists
        cache_key = tuple(sorted({'linear': True}.items()))
        self.assertIn(cache_key, vec._filters)

        # Second call should use cache
        result2 = vec.asarray(linear=True)
        assert_near_equal(result1, result2)

    def test_asarray_filter_result_is_copy(self):
        """Test that filtered result is a copy, not a view."""
        vec = OptimizerVector('constraint', self.data, self.metadata)

        result = vec.asarray(linear=True)
        original_val = result[0]

        # Modify the result
        result[0] = 999.0

        # Original data should NOT be modified (it's a copy)
        self.assertEqual(self.data[0], original_val)


class TestOptimizerVectorMetadata(unittest.TestCase):
    """Test OptimizerVector metadata property."""

    def test_metadata_property(self):
        """Test metadata property returns the metadata dict."""
        data = np.array([1.0, 2.0])
        metadata = {'x': {'start_idx': 0, 'end_idx': 2, 'size': 2}}
        vec = OptimizerVector('design_var', data, metadata)

        returned_meta = vec.metadata
        self.assertIs(returned_meta, metadata)
        self.assertEqual(returned_meta['x']['start_idx'], 0)


class TestOptimizerVectorEdgeCases(unittest.TestCase):
    """Test OptimizerVector edge cases."""

    def test_empty_vector(self):
        """Test empty OptimizerVector."""
        data = np.array([], dtype=float)
        metadata = {}
        vec = OptimizerVector('design_var', data, metadata)

        self.assertEqual(len(vec), 0)
        self.assertEqual(list(vec), [])
        self.assertEqual(list(vec.values()), [])

    def test_single_element(self):
        """Test OptimizerVector with single element."""
        data = np.array([42.0])
        metadata = {'x': {'start_idx': 0, 'end_idx': 1, 'size': 1}}
        vec = OptimizerVector('design_var', data, metadata)

        self.assertEqual(len(vec), 1)
        assert_near_equal(vec['x'], np.array([42.0]))

    def test_large_multidimensional_array(self):
        """Test OptimizerVector with large multi-dimensional data."""
        shape = (5, 4, 3)
        size = np.prod(shape)
        data = np.arange(size, dtype=float).reshape(shape)
        data_flat = data.flatten()

        metadata = {
            'tensor': {'start_idx': 0, 'end_idx': size, 'size': size},
        }

        vec = OptimizerVector('design_var', data_flat, metadata)

        result = vec['tensor']
        assert_near_equal(result, data_flat)

    def test_set_with_different_dtypes(self):
        """Test setting values with different dtypes."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        metadata = {
            'x': {'start_idx': 0, 'end_idx': 2, 'size': 2},
            'y': {'start_idx': 2, 'end_idx': 4, 'size': 2},
        }
        vec = OptimizerVector('design_var', data, metadata)

        # Set with integer
        vec['x'] = 5
        assert_near_equal(vec['x'], np.array([5.0, 5.0]))

        # Set with float list
        vec['y'] = [6.5, 7.5]
        assert_near_equal(vec['y'], np.array([6.5, 7.5]))

    def test_multiple_variables_same_size(self):
        """Test multiple variables with same size."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        metadata = {
            'a': {'start_idx': 0, 'end_idx': 2, 'size': 2},
            'b': {'start_idx': 2, 'end_idx': 4, 'size': 2},
            'c': {'start_idx': 4, 'end_idx': 6, 'size': 2},
        }
        vec = OptimizerVector('design_var', data, metadata)

        # All should have same size
        for name in ['a', 'b', 'c']:
            self.assertEqual(len(vec[name]), 2)


if __name__ == '__main__':
    unittest.main()
