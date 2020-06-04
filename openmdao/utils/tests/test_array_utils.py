import unittest

import numpy as np

from openmdao.utils.array_utils import array_connection_compatible, abs_complex, dv_abs_complex
from openmdao.utils.assert_utils import assert_near_equal


class TestArrayConnectionCompatible(unittest.TestCase):

    def test_ones_at_both_ends(self):
        shape1 = (1, 1, 15, 3, 1, 7, 1, 1, 1, 1)
        shape2 = (1, 15, 3, 1, 7)
        self.assertTrue(array_connection_compatible(shape1, shape2))

    def test_ones_at_start(self):
        shape1 = (1, 1, 15, 3, 1, 7)
        shape2 = (1, 15, 3, 1, 7)
        self.assertTrue(array_connection_compatible(shape1, shape2))

    def test_ones_at_end(self):
        shape1 = (15, 3, 1, 7, 1, 1, 1, 1)
        shape2 = (1, 15, 3, 1, 7)
        self.assertTrue(array_connection_compatible(shape1, shape2))

    def test_ones_to_ones(self):
        shape1 = (1, 1, 1, 1, 1, 1, 1, 1)
        shape2 = (1, 1, 1, 1, 1)
        self.assertTrue(array_connection_compatible(shape1, shape2))

    def test_ones_to_one(self):
        shape1 = (1, 1, 1, 1, 1, 1, 1, 1)
        shape2 = (1,)
        self.assertTrue(array_connection_compatible(shape1, shape2))

    def test_matrix_to_vectorized_matrix(self):
        shape1 = (3, 3)
        shape2 = (1, 3, 3)
        self.assertTrue(array_connection_compatible(shape1, shape2))

    def test_known_incompatable(self):
        shape1 = (3, 3)
        shape2 = (3, 1, 3)
        self.assertFalse(array_connection_compatible(shape1, shape2))


class TestArrayUtils(unittest.TestCase):

    def test_abs_complex(self):

        x = np.array([3.0 + 0.5j, -4.0 - 1.5j, -5.0 + 2.5j, -6.0 - 3.5j])
        y = abs_complex(x)

        self.assertEqual(y[0], 3.0 + 0.5j)
        self.assertEqual(y[1], 4.0 + 1.5j)
        self.assertEqual(y[2], 5.0 - 2.5j)
        self.assertEqual(y[3], 6.0 + 3.5j)

        x = np.array([3.0 + 0.5j, -4.0 - 1.5j, -5.0 + 2.5j, -6.0 - 3.5j])
        dx = 1.0 + 2j * np.ones((4, 3), dtype=np.complex)

        yy, dy = dv_abs_complex(x, dx)

        row = np.array([1.0 + 2j, 1.0 + 2j, 1.0 + 2j])
        dy_check = np.vstack((row, -row, -row, -row))
        assert_near_equal(dy, dy_check, 1e-10)


if __name__ == "__main__":
    unittest.main()
