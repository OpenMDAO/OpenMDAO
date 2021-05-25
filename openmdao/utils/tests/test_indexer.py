import unittest

import numpy as np
from numpy.testing import assert_equal

from openmdao.utils.indexer import indexer


class IndexerTestCase(unittest.TestCase):
    def test_int(self):
        ind = indexer[4]
        src = np.arange(10)
        assert_equal(ind(), 4)
        assert_equal(ind.as_array(), np.array([4]))
        assert_equal(ind.shape(), ())
        assert_equal(ind.shaped(), 4)
        assert_equal(ind.as_slice(), slice(4, 5))

    def test_neg_int(self):
        ind = indexer[-4]
        src = np.arange(10)
        assert_equal(ind(), np.array([-4]))
        assert_equal(ind.as_array(), np.array([-4]))
        assert_equal(ind.shape(), ())

        with self.assertRaises(ValueError) as cm:
            ind.shaped()
        self.assertEqual(cm.exception.args[0], "Can't get shaped version of -4 because it has no source shape.")

        ind.set_src_shape(src.shape)
        assert_equal(ind.shaped(), 6)

    def test_simple_arr(self):
        ind = indexer[[5, 3, 7, 1]]
        src = np.arange(10)
        assert_equal(src[ind.as_array()], [5,3,7,1])
        assert_equal(src[ind.shaped()], [5,3,7,1])
        assert_equal(ind.shape(), (4,))
        with self.assertRaises(ValueError) as cm:
            ind.as_slice()
        self.assertEqual(cm.exception.args[0],  "Can't convert [5 3 7 1] to a slice.")

    def test_contiguous_arr(self):
        ind = indexer[[3, 4, 5]]
        src = np.arange(10)
        assert_equal(src[ind.as_array()], [3, 4, 5])
        assert_equal(src[ind.shaped()], [3, 4, 5])
        assert_equal(ind.shape(), 3)
        assert_equal(ind.as_slice(), slice(3, 6, 1))
        assert_equal(ind.shaped(), slice(3, 6, 1))

    def test_arr_to_slice_step2(self):
        ind = indexer[[1,3,5,7,9]]
        src = np.arange(10)
        assert_equal(src[ind.as_array()], [1,3,5,7,9])
        assert_equal(src[ind()], [1,3,5,7,9])
        assert_equal(src[ind.as_slice()], [1,3,5,7,9])
        assert_equal(ind.shape(), 5)
        self.assertEqual(ind.as_slice(), slice(1, 10, 2))

    def test_neg_arr(self):
        ind = indexer[[5, 3, 7, -1]]
        src = np.arange(10)
        assert_equal(src[ind.as_array()], [5,3,7,9])
        assert_equal(ind.shape(), (4,))
        try:
            ind.as_slice()
        except Exception as err:
            self.assertEqual(str(err), "Can't get shaped slice of [ 5  3  7 -1] because it has no source shape.")
        else:
            self.fail("Exception expected")

        try:
            ind.shaped()
        except Exception as err:
            self.assertEqual(str(err), "Can't get shaped version of [ 5  3  7 -1] because it has no source shape.")
        else:
            self.fail("Exception expected")

        ind.set_src_shape(10)
        assert_equal(ind.shaped(), [5,3,7,9])

    def test_neg_arr_sliceable(self):
        # this array can be converted to a slice after the src shape has been set
        ind = indexer[[-1, -3, -5]]
        src = np.arange(10)

        try:
            ind.shaped()
        except Exception as err:
            self.assertEqual(str(err), "Can't get shaped version of [-1 -3 -5] because it has no source shape.")
        else:
            self.fail("Exception expected")

        ind.set_src_shape(10)
        slc = ind.as_slice()
        assert_equal(ind.shape(), (3,))
        assert_equal(slc, slice(9, 4, -2))
        assert_equal(ind.shaped(), slice(9, 4, -2))
        assert_equal(src[ind.as_array()], [9, 7, 5])

    def test_full_slice(self):
        ind = indexer[:]
        src = np.arange(10)
        assert_equal(ind(), slice(None))
        assert_equal(ind.as_slice(), slice(None))
        with self.assertRaises(RuntimeError) as cm:
            ind.shape()
        self.assertEqual(cm.exception.args[0], "Can't get shape of slice(None, None, None) because source shape is unknown.")
        with self.assertRaises(ValueError) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't get shaped array of slice(None, None, None) because it has no source shape.")
        ind.set_src_shape(src.shape)
        assert_equal(ind.shape(), 10)
        assert_equal(ind.shaped(), slice(0, 10, 1))
        assert_equal(ind.as_array(), np.arange(10, dtype=int))

    def test_neg_start_slice(self):
        ind = indexer[-3:-6:-1]
        src = np.arange(10)
        assert_equal(ind.as_slice(), slice(-3, -6, -1))
        with self.assertRaises(ValueError) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't get shaped array of slice(-3, -6, -1) because it has no source shape.")
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
        assert_equal(ind.shaped(), slice(None, 5, None))

    def test_none_stop_slice(self):
        ind = indexer[3:]
        src = np.arange(10)
        assert_equal(ind.as_slice(), slice(3, None, None))
        with self.assertRaises(ValueError) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't get shaped array of slice(3, None, None) because it has no source shape.")
        ind.set_src_shape(src.shape)
        assert_equal(ind.shape(), 7)
        assert_equal(ind.shaped(), slice(3, 10, 1))
        assert_equal(ind.as_array(), np.array([3, 4, 5, 6, 7, 8, 9]))

    def test_slice_neg_step(self):
        ind = indexer[6:2:-1]
        src = np.arange(10)
        assert_equal(ind.as_slice(), slice(6, 2, -1))
        assert_equal(ind.as_array(), np.array([6, 5, 4, 3]))
        assert_equal(ind.shape(), 4)
        assert_equal(ind.shaped(), slice(6, 2, -1))


class IndexerMultiDimTestCase(unittest.TestCase):
    def test_multi_slice(self):
        ind = indexer[:,:,:]
        src = np.arange(27).reshape((3,3,3))
        assert_equal(ind(), (slice(None), slice(None), slice(None)))
        assert_equal(ind.as_slice(), (slice(None), slice(None), slice(None)))
        with self.assertRaises(RuntimeError) as cm:
            ind.shape()
        self.assertEqual(cm.exception.args[0], "Can't get shape of slice(None, None, None) because source shape is unknown.")
        with self.assertRaises(ValueError) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't determine extent of array because source shape is not known.")
        ind.set_src_shape(src.shape)
        assert_equal(ind.shape(), (3,3,3))
        assert_equal(ind.shaped(), (slice(0, 3, 1), slice(0, 3, 1), slice(0, 3, 1)))
        assert_equal(ind.as_array(), np.arange(27, dtype=np.int32))

    def test_slice_neg(self):
        ind = indexer[:-1,:,:2]
        src = np.arange(27).reshape((3,3,3))
        assert_equal(ind(), (slice(None, -1), slice(None), slice(None, 2)))
        assert_equal(ind.as_slice(), (slice(None, -1), slice(None), slice(None, 2)))
        with self.assertRaises(RuntimeError) as cm:
            ind.shape()
        self.assertEqual(cm.exception.args[0], "Can't get shape of slice(None, -1, None) because source shape is unknown.")
        with self.assertRaises(ValueError) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't determine extent of array because source shape is not known.")
        ind.set_src_shape(src.shape)
        assert_equal(ind.shape(), (2,3,2))
        assert_equal(ind.shaped(), (slice(0, 2, 1), slice(0, 3, 1), slice(None, 2, None)))
        assert_equal(ind.as_array(), np.arange(27, dtype=np.int32).reshape((3,3,3))[:-1,:,:2].ravel())

    def test_mult_arr(self):
        src = np.arange(27).reshape((3,3,3))
        ind = indexer[[0,2], :, [1,2]]
        assert_equal(ind(), ([0,2], slice(None, None, None), [1,2]))
        with self.assertRaises(ValueError) as cm:
            ind.as_slice()
        self.assertEqual(cm.exception.args[0], "Can't convert [0 2] to a slice.")
        with self.assertRaises(RuntimeError) as cm:
            ind.shape()
        self.assertEqual(cm.exception.args[0], "Can't get shape of slice(None, None, None) because source shape is unknown.")
        with self.assertRaises(ValueError) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't determine extent of array because source shape is not known.")
        ind.set_src_shape(src.shape)
        assert_equal(ind.shape(), (2,3))
        assert_equal(ind.shaped(), ([0, 2], slice(0, 3, 1), [1, 2]))
        assert_equal(ind.as_array(), np.arange(27, dtype=np.int32).reshape((3,3,3))[[0,2], :, [1,2]].ravel())


if __name__ == '__main__':
    unittest.main()
