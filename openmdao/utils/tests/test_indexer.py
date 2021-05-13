import unittest

import numpy as np
from numpy.testing import assert_equal

from openmdao.utils.indexer import indexer


class IndexerTestCase(unittest.TestCase):
    def test_int(self):
        ind = indexer(4)
        src = np.arange(10)
        assert_equal(ind(), 4)
        assert_equal(ind.as_array(), np.array([4]))
        assert_equal(ind.shape(), 1)
        assert_equal(ind.shaped(), 4)
        assert_equal(ind.as_slice(), slice(4, 5))

    def test_neg_int(self):
        ind = indexer(-4)
        src = np.arange(10)
        assert_equal(ind(), np.array([-4]))
        assert_equal(ind.as_array(), np.array([-4]))
        assert_equal(ind.shape(), 1)

        with self.assertRaises(ValueError) as cm:
            ind.shaped()
        self.assertEqual(cm.exception.args[0], "IntIndex value has no source shape and index is -4.")

        ind.set_src_shape(src.shape)
        assert_equal(ind.shaped(), 6)

    def test_simple_arr(self):
        ind = indexer([5, 3, 7, 1])
        src = np.arange(10)
        assert_equal(src[ind.as_array()], [5,3,7,1])
        assert_equal(src[ind.shaped()], [5,3,7,1])
        assert_equal(ind.shape(), (4,))
        try:
            ind.as_slice()
        except Exception as err:
            self.assertEqual(str(err), "array index cannot be converted to a slice.")
        else:
            self.fail("Exception expected")

    def test_contiguous_arr(self):
        ind = indexer([3, 4, 5])
        src = np.arange(10)
        assert_equal(src[ind.as_array()], [3, 4, 5])
        assert_equal(src[ind.shaped()], [3, 4, 5])
        assert_equal(ind.shape(), (3,))
        assert_equal(ind.as_slice(), slice(3, 6, 1))
        assert_equal(ind.shaped(), slice(3, 6, 1))

    def test_arr_to_slice_step2(self):
        ind = indexer([1,3,5,7,9])
        src = np.arange(10)
        assert_equal(src[ind.as_array()], [1,3,5,7,9])
        assert_equal(src[ind()], [1,3,5,7,9])
        assert_equal(src[ind.as_slice()], [1,3,5,7,9])
        assert_equal(ind.shape(), (5,))
        self.assertEqual(ind.as_slice(), slice(1, 10, 2))

    def test_neg_arr(self):
        ind = indexer([5, 3, 7, -1])
        src = np.arange(10)
        assert_equal(src[ind.as_array()], [5,3,7,9])
        assert_equal(ind.shape(), (4,))
        try:
            ind.as_slice()
        except Exception as err:
            self.assertEqual(str(err), "array index cannot be converted to a slice.")
        else:
            self.fail("Exception expected")

        try:
            ind.shaped()
        except Exception as err:
            self.assertEqual(str(err), "Can't determine extent of array because source shape is not known.")
        else:
            self.fail("Exception expected")

        ind.set_src_shape(10)
        assert_equal(ind.shaped(), [5,3,7,9])

    def test_neg_arr_sliceable(self):
        ind = indexer([-1, -3, -5])
        src = np.arange(10)
        assert_equal(src[ind.as_array()], [9, 7, 5])
        assert_equal(ind.shape(), (3,))
        try:
            ind.as_slice()
        except Exception as err:
            self.assertEqual(str(err), "array index cannot be converted to a slice.")
        else:
            self.fail("Exception expected")

        try:
            ind.shaped()
        except Exception as err:
            self.assertEqual(str(err), "Can't determine extent of array because source shape is not known.")
        else:
            self.fail("Exception expected")

        ind.set_src_shape(10)
        assert_equal(ind.shaped(), slice(9, 4, -2))

        slc = ind.as_slice()
        assert_equal(slc, slice(9, 4, -2))

    def test_full_slice(self):
        ind = indexer[:]
        src = np.arange(10)
        assert_equal(ind(), slice(None))
        assert_equal(ind.as_slice(), slice(None))
        with self.assertRaises(RuntimeError) as cm:
            ind.shape()
        self.assertEqual(cm.exception.args[0], "Can't get shape of slice(None, None, None) because source shape is unknown.")
        with self.assertRaises(RuntimeError) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't convert slice(None, None, None) to array because source shape is unknown.")
        ind.set_src_shape(src.shape)
        assert_equal(ind.shape(), 10)
        assert_equal(ind.shaped(), slice(0, 10, 1))
        assert_equal(ind.as_array(), np.arange(10, dtype=int))

    def test_neg_start_slice(self):
        ind = indexer[-3:-6:-1]
        src = np.arange(10)
        assert_equal(ind.as_slice(), slice(-3, -6, -1))
        with self.assertRaises(RuntimeError) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't convert slice(-3, -6, -1) to array because source shape is unknown.")
        ind.set_src_shape(src.shape)
        assert_equal(ind.shape(), 3)
        assert_equal(ind.shaped(), slice(7, 4, -1))
        assert_equal(ind.as_array(), np.array([7, 6, 5]))

    def test_none_start_slice(self):
        ind = indexer[:5]
        src = np.arange(10)
        assert_equal(ind.as_slice(), slice(None, 5, None))
        assert_equal(ind.as_array(), np.array([0, 1, 2, 3, 4]))
        ind.set_src_shape(src.shape)
        assert_equal(ind.shape(), 5)
        assert_equal(ind.shaped(), slice(0, 5, 1))

    def test_none_stop_slice(self):
        ind = indexer[3:]
        src = np.arange(10)
        assert_equal(ind.as_slice(), slice(3, None, None))
        with self.assertRaises(RuntimeError) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't convert slice(3, None, None) to array because source shape is unknown.")
        ind.set_src_shape(src.shape)
        assert_equal(ind.shape(), 7)
        assert_equal(ind.shaped(), slice(3, 10, 1))
        assert_equal(ind.as_array(), np.array([3, 4, 5, 6, 7, 8, 9]))

    def test_slice_ellipsis(self):
        self.fail("No test yet")

    def test_slice_step_neg(self):
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
