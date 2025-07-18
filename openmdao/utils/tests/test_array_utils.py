import unittest

import numpy as np

from openmdao.utils.array_utils import array_connection_compatible, abs_complex, dv_abs_complex, \
    convert_neg, submat_sparsity_iter
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
        dx = 1.0 + 2j * np.ones((4, 3), dtype=complex)

        yy, dy = dv_abs_complex(x, dx)

        row = np.array([1.0 + 2j, 1.0 + 2j, 1.0 + 2j])
        dy_check = np.vstack((row, -row, -row, -row))
        assert_near_equal(dy, dy_check, 1e-10)

    def test_convert_neg(self):
        a = np.arange(16).reshape((4,4))
        a[2, 3] = -5
        a[0, 1] = -7

        b = convert_neg(a.copy(), a.size)
        self.assertEqual(b[2, 3], 11)
        self.assertEqual(b[0, 1], 9)

        inds = np.where(a != b)
        self.assertTrue(np.all(inds[0] == np.array([0,2])))
        self.assertTrue(np.all(inds[1] == np.array([1,3])))


class TestSubmatSparsityIter(unittest.TestCase):

    def _check_results(self, expected, result):
        expected = list(expected)
        result = list(result)
        self.assertEqual(len(expected), len(result))

        for extup, tup in zip(expected, result):
            exof, exwrt, exrows, excols, exshape = extup
            of, wrt, rows, cols, shape = tup
            self.assertEqual(of, exof)
            self.assertEqual(wrt, exwrt)
            if exrows is None:
                self.assertEqual(exrows, rows)
                self.assertEqual(excols, cols)
            else:
                self.assertTrue(np.all(rows == exrows))
                self.assertTrue(np.all(cols == excols))
            self.assertEqual(shape, exshape)

    def test_empty_matrix(self):
        row_var_size_iter = iter([])
        col_var_size_iter = iter([])
        nzrows = np.array([])
        nzcols = np.array([])
        shape = (0, 0)
        result = list(submat_sparsity_iter(row_var_size_iter, col_var_size_iter, nzrows, nzcols, shape))
        self.assertEqual(result, [])

    def test_single_element_matrix(self):
        row_var_size_iter = iter([('a', 1)])
        col_var_size_iter = iter([('b', 1)])
        nzrows = np.array([0])
        nzcols = np.array([0])
        shape = (1, 1)
        result = list(submat_sparsity_iter(row_var_size_iter, col_var_size_iter, nzrows, nzcols, shape))
        expected =  [('a', 'b', np.array([0]), np.array([0]), (1, 1))]
        self._check_results(expected, result)

    def test_multiple_elements_matrix(self):
        row_var_size_iter = iter([('a', 2), ('b', 2)])
        col_var_size_iter = iter([('c', 2), ('d', 2)])
        nzrows = np.array([0, 1, 2, 3])
        nzcols = np.array([0, 1, 2, 3])
        shape = (4, 4)
        result = list(submat_sparsity_iter(row_var_size_iter, col_var_size_iter, nzrows, nzcols, shape))
        expected = [
            ('a', 'c', np.array([0, 1]), np.array([0, 1]), (2, 2)),
            ('a', 'd', np.array([]), np.array([]), (2, 2)),
            ('b', 'c', np.array([]), np.array([]), (2, 2)),
            ('b', 'd', np.array([0, 1]), np.array([0, 1]), (2, 2)),
        ]
        self._check_results(expected, result)

    def test_sparse_matrix(self):
        row_var_size_iter = iter([('a', 2), ('b', 2)])
        col_var_size_iter = iter([('c', 2), ('d', 2)])
        nzrows = np.array([0, 3])
        nzcols = np.array([0, 3])
        shape = (4, 4)
        result = list(submat_sparsity_iter(row_var_size_iter, col_var_size_iter, nzrows, nzcols, shape))
        expected = [
            ('a', 'c', np.array([0]), np.array([0]), (2, 2)),
            ('a', 'd', np.array([]), np.array([]), (2, 2)),
            ('b', 'c', np.array([]), np.array([]), (2, 2)),
            ('b', 'd', np.array([1]), np.array([1]), (2, 2))
        ]
        self._check_results(expected, result)


if __name__ == "__main__":
    unittest.main()
