import unittest

import numpy as np
from numpy.testing import assert_equal

from openmdao.utils.indexer import indexer


class IndexerTestCase(unittest.TestCase):
    def test_int(self):
        ind = indexer(4)
        src = np.arange(10)
        assert_equal(src[ind()], 4)
        assert_equal(ind.shape(), (1,))

    def test_neg_int(self):
        ind = indexer(-4)
        src = np.arange(10)
        assert_equal(src[ind()], 6)
        assert_equal(ind.shape(), (1,))

    def test_full_slice(self):
        ind = indexer[:]
        src = np.arange(10)
        assert_equal(src[ind()], src)
        assert_equal(ind.shape(), (1,))

    def test_neg_start_slice(self):
        ind = indexer[-3:-6:-1]
        src = np.arange(10)
        assert_equal(src[ind()], [7,6,5])
        with self.assertRaises(RuntimeError) as cm:
            ind(final=True)
        self.assertEqual(cm.exception.args[0], "indexer(slice(-3, -6, -1)) source does not have a known src_shape.")

    def test_none_slice(self):
        pass

    def test_slice_ellipsis(self):
        pass

    def test_slice_step_neg(self):
        pass

    def test_simple_arr(self):
        ind = indexer([5, 3, 7, 1])
        src = np.arange(10)
        assert_equal(src[ind()], [5,3,7,1])
        assert_equal(ind.shape(), (4,))

    def test_arr_to_slice(self):
        ind = indexer([1,3,5,7,9])
        src = np.arange(10)
        assert_equal(src[ind()], [1,3,5,7,9])
        assert_equal(ind.shape(), (5,))
        self.assertTrue(isinstance(ind(), slice))
        self.assertEqual(ind(), slice(1, 10, 2))

    def test_neg_arr(self):
        pass


class IndexerMultiDimTestCase(unittest.TestCase):
    def test_multi_slice(self):
        pass

    def test_slice_neg(self):
        pass

    def test_slice_none(self):
        pass

    def test_slice_ellipsis(self):
        pass

    def test_slice_step_neg(self):
        pass

    def test_simple_arr(self):
        pass

    def test_neg_arr(self):
        pass


if __name__ == '__main__':
    unittest.main()
