import unittest

import numpy as np
from numpy.testing import assert_equal

from openmdao.utils.indexer import indexer, combine_ranges


class IndexerTestCase(unittest.TestCase):
    def test_int(self):
        ind = indexer[4]
        src = np.arange(10)

        assert_equal(ind(), 4)
        assert_equal(ind.as_array(), np.array([4]))
        assert_equal(ind.flat(), np.array([4]))

        assert_equal(src[ind()], 4)
        assert_equal(src[ind.as_array()], np.array([4]))
        assert_equal(src[ind.flat()], np.array([4]))

        ind.set_src_shape(src.shape)
        assert_equal(src[ind.shaped_array()], np.array([4]))
        assert_equal(ind.shaped_array(), np.array([4]))
        assert_equal(ind.indexed_src_shape, (1,))
        assert_equal(ind.min_src_dim, 1)

    def test_neg_int(self):
        ind = indexer[-4]
        src = np.arange(10)

        assert_equal(ind(), -4)
        assert_equal(ind.as_array(), np.array([-4]))
        assert_equal(ind.flat(), np.array([-4]))

        assert_equal(src[ind()], 6)
        assert_equal(src[ind.as_array()], np.array([6]))
        assert_equal(src[ind.flat()], np.array([6]))

        ind.set_src_shape(src.shape)
        assert_equal(ind.indexed_src_shape, (1,))
        assert_equal(ind.min_src_dim, 1)

        assert_equal(ind.shaped_array(), np.array([6]))
        assert_equal(src[ind.shaped_array()], np.array([6]))
        assert_equal(ind.shaped_instance()(), 6)

    def test_int_nonflat(self):
        ind = indexer[1]
        src = np.arange(9).reshape((3,3))

        assert_equal(ind(), 1)
        assert_equal(ind.as_array(), np.array([1]))
        assert_equal(ind.flat(), np.array([1]))

        assert_equal(src[ind()], np.array([3,4,5]))
        assert_equal(src[ind.as_array()], np.array([[3,4,5]]))
        assert_equal(src[ind.flat()], np.array([[3,4,5]]))

        ind.set_src_shape(src.shape)
        assert_equal(src[ind.shaped_array()], np.array([[3,4,5]]))
        assert_equal(ind.shaped_array(), np.array([1]))
        assert_equal(ind.indexed_src_shape, (3,))
        assert_equal(ind.min_src_dim, 1)

    def test_simple_noncontig_arr(self):
        ind = indexer[[5, 3, 7, 1]]
        src = np.arange(10)

        assert_equal(ind(), np.array([5,3,7,1]))
        assert_equal(ind.as_array(), np.array([5,3,7,1]))
        assert_equal(ind.flat(), np.array([5,3,7,1]))

        assert_equal(src[ind()], np.array([5,3,7,1]))
        assert_equal(src[ind.as_array()], np.array([5,3,7,1]))
        assert_equal(src[ind.flat()], np.array([5,3,7,1]))

        assert_equal(ind.min_src_dim, 1)

        ind.set_src_shape(src.shape)
        assert_equal(ind.indexed_src_shape, (4,))
        assert_equal(ind.shaped_instance()(), np.array([5,3,7,1]))
        assert_equal(ind.shaped_array(), np.array([5,3,7,1]))
        assert_equal(src[ind.shaped_array()], np.array([5,3,7,1]))

    def test_contiguous_arr(self):
        ind = indexer[[3, 4, 5]]
        src = np.arange(10)

        assert_equal(ind(), [3, 4, 5])
        assert_equal(ind.as_array(), np.array([3, 4, 5]))
        assert_equal(ind.flat(), [3, 4, 5])

        assert_equal(src[ind()], np.array([3, 4, 5]))
        assert_equal(src[ind.as_array()], np.array([3, 4, 5]))
        assert_equal(src[ind.flat()], np.array([3, 4, 5]))

        assert_equal(ind.min_src_dim, 1)

        ind.set_src_shape(src.shape)
        assert_equal(ind.indexed_src_shape, (3,))
        assert_equal(ind.shaped_instance()(), [3, 4, 5])
        assert_equal(src[ind.shaped_array()], np.array([3, 4, 5]))
        assert_equal(ind.shaped_array(), np.array([3, 4, 5]))

    def test_arr_to_slice_step2(self):
        ind = indexer[[1,3,5,7,9]]
        src = np.arange(10)

        assert_equal(ind(), [1,3,5,7,9])
        assert_equal(ind.flat(), [1,3,5,7,9])
        assert_equal(ind.as_array(), np.array([1,3,5,7,9]))

        assert_equal(src[ind()], np.array([1,3,5,7,9]))
        assert_equal(src[ind.as_array()], np.array([1,3,5,7,9]))
        assert_equal(src[ind.flat()], np.array([1,3,5,7,9]))

        ind.set_src_shape(src.shape)
        assert_equal(src[ind.shaped_array()], np.array([1,3,5,7,9]))
        assert_equal(ind.shaped_array(), np.array([1,3,5,7,9]))
        assert_equal(ind.indexed_src_shape, (5,))
        assert_equal(ind.min_src_dim, 1)

    def test_neg_arr(self):
        ind = indexer[[5, 3, 7, -1]]
        src = np.arange(10)

        assert_equal(ind(), np.array([5, 3, 7, -1]))
        assert_equal(ind.as_array(), np.array([5, 3, 7, -1]))
        assert_equal(ind.flat(), np.array([5, 3, 7, -1]))

        assert_equal(src[ind()], np.array([5,3,7,9]))
        assert_equal(src[ind.as_array()], np.array([5,3,7,9]))
        assert_equal(src[ind.flat()], np.array([5,3,7,9]))

        assert_equal(ind.indexed_src_shape, (4,))
        assert_equal(ind.min_src_dim, 1)

        try:
            ind.shaped_array()
        except Exception as err:
            self.assertEqual(str(err), "Can't get shaped array of [ 5  3  7 -1] because it has no source shape.")
        else:
            self.fail("Exception expected")

        ind.set_src_shape(src.shape)
        assert_equal(ind.shaped_array(), np.array([5,3,7,9]))
        assert_equal(src[ind.shaped_array()], np.array([5,3,7,9]))
        assert_equal(ind.shaped_instance()(), np.array([5,3,7,9]))

    def test_neg_arr(self):
        ind = indexer[[-1, -3, -5]]
        src = np.arange(10)

        assert_equal(ind(), np.array([-1, -3, -5]))
        assert_equal(ind.as_array(), np.array([-1, -3, -5]))
        assert_equal(ind.flat(), np.array([-1, -3, -5]))

        assert_equal(src[ind()], np.array([9, 7, 5]))
        assert_equal(src[ind.as_array()], np.array([9, 7, 5]))
        assert_equal(src[ind.flat()], np.array([9, 7, 5]))

        assert_equal(ind.min_src_dim, 1)

        try:
            ind.shaped_array()
        except Exception as err:
            self.assertEqual(str(err), "Can't get shaped array of [-1 -3 -5] because it has no source shape.")
        else:
            self.fail("Exception expected")

        ind.set_src_shape(src.shape)
        assert_equal(ind.indexed_src_shape, (3,))
        assert_equal(ind.indexed_src_shape, (3,))
        assert_equal(ind.min_src_dim, 1)

        assert_equal(ind.shaped_instance()(), np.array([9, 7, 5]))
        assert_equal(ind.shaped_array(), np.array([9, 7, 5]))

    def test_full_slice(self):
        ind = indexer[:]
        src = np.arange(10)

        assert_equal(ind(), slice(None, None, 1))
        assert_equal(ind.flat(), slice(None, None, 1))

        assert_equal(src[ind()], np.arange(10, dtype=int))
        assert_equal(src[ind.flat()], np.arange(10, dtype=int))

        assert_equal(ind.min_src_dim, 1)

        with self.assertRaises(Exception) as cm:
            ind.indexed_src_shape
        self.assertEqual(cm.exception.args[0], "Can't get indexed_src_shape of slice(None, None, 1) because source shape is unknown.")
        with self.assertRaises(Exception) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't get shaped array of slice(None, None, 1) because it has no source shape.")

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_instance()(), slice(0, 10, 1))
        assert_equal(ind.as_array(), np.arange(10, dtype=int))
        assert_equal(ind.indexed_src_shape, (10,))
        assert_equal(ind.min_src_dim, 1)

    def test_neg_start_slice(self):
        ind = indexer[-3:-6:-1]
        src = np.arange(10)

        assert_equal(ind(), slice(-3, -6, -1))
        assert_equal(ind.flat(), slice(-3, -6, -1))

        assert_equal(src[ind()], np.array([7, 6, 5]))
        assert_equal(src[ind.flat()], np.array([7, 6, 5]))

        assert_equal(ind.min_src_dim, 1)

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_instance()(), slice(7, 4, -1))
        assert_equal(ind.as_array(), np.array([7, 6, 5]))
        assert_equal(ind.indexed_src_shape, (3,))
        assert_equal(ind.min_src_dim, 1)

    def test_none_start_slice(self):
        ind = indexer[:5]
        src = np.arange(10)

        assert_equal(ind(), slice(None, 5, 1))
        assert_equal(ind.flat(), slice(None, 5, 1))

        assert_equal(src[ind()], np.array([0, 1, 2, 3, 4]))
        assert_equal(src[ind.flat()], np.array([0, 1, 2, 3, 4]))

        assert_equal(ind.min_src_dim, 1)

        ind.set_src_shape(src.shape)
        assert_equal(ind.as_array(), np.array([0, 1, 2, 3, 4]))
        assert_equal(src[ind.as_array()], np.array([0, 1, 2, 3, 4]))

        assert_equal(ind.shaped_instance()(), slice(None, 5, 1))
        assert_equal(ind.indexed_src_shape, (5,))
        assert_equal(ind.min_src_dim, 1)

    def test_none_stop_slice(self):
        ind = indexer[3:]
        src = np.arange(10)

        assert_equal(ind(), slice(3, None, 1))
        assert_equal(ind.flat(), slice(3, None, 1))

        assert_equal(src[ind()], np.array([3, 4, 5, 6, 7, 8, 9]))
        assert_equal(src[ind.flat()], np.array([3, 4, 5, 6, 7, 8, 9]))

        assert_equal(ind.min_src_dim, 1)

        with self.assertRaises(Exception) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't get shaped array of slice(3, None, 1) because it has no source shape.")

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_instance()(), slice(3, 10, 1))
        assert_equal(ind.as_array(), np.array([3, 4, 5, 6, 7, 8, 9]))
        assert_equal(ind.indexed_src_shape, (7,))
        assert_equal(ind.min_src_dim, 1)

    def test_slice_neg_step(self):
        ind = indexer[6:2:-1]
        src = np.arange(10)

        assert_equal(ind(), slice(6, 2, -1))
        assert_equal(ind.flat(), slice(6, 2, -1))

        assert_equal(src[ind()], np.array([6, 5, 4, 3]))
        assert_equal(src[ind.flat()], np.array([6, 5, 4, 3]))

        assert_equal(ind.min_src_dim, 1)

        ind.set_src_shape(src.shape)

        assert_equal(ind.as_array(), np.array([6, 5, 4, 3]))
        assert_equal(src[ind.as_array()], np.array([6, 5, 4, 3]))
        assert_equal(src[ind.shaped_array()], np.array([6, 5, 4, 3]))
        assert_equal(ind.shaped_instance()(), slice(6, 2, -1))
        assert_equal(ind.indexed_src_shape, (4,))
        assert_equal(ind.min_src_dim, 1)

    def test_rev_slice(self):
        ind = indexer[::-1]
        src = np.arange(10)

        assert_equal(ind(), slice(None, None, -1))
        assert_equal(ind.flat(), slice(None, None, -1))

        assert_equal(src[ind()], np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))
        assert_equal(src[ind.flat()], np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))

        assert_equal(ind.min_src_dim, 1)

        ind.set_src_shape(src.shape)

        assert_equal(ind.as_array(), np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))
        assert_equal(src[ind.as_array()], np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))
        assert_equal(src[ind.shaped_array()], np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))
        assert_equal(ind.shaped_instance()(), slice(None, None, -1))
        assert_equal(ind.indexed_src_shape, (10,))
        assert_equal(ind.min_src_dim, 1)

    def test_rev_slice2D(self):
        ind = indexer[::-1]
        src = np.arange(4).reshape((2,2))

        assert_equal(ind(), slice(None, None, -1))
        assert_equal(ind.flat(), slice(None, None, -1))

        expected = src[::-1]

        assert_equal(src[ind()], expected)
        assert_equal(src[ind.flat()], expected)

        assert_equal(ind.min_src_dim, 1)

        ind.set_src_shape(src.shape)

        assert_equal(ind.as_array(), expected.flatten())
        assert_equal(src[ind()], expected)
        assert_equal(ind.shaped_instance()(), slice(None, None, -1))
        assert_equal(ind.indexed_src_shape, (2,2))
        assert_equal(ind.min_src_dim, 1)

class IndexerMultiDimTestCase(unittest.TestCase):
    def test_multi_slice(self):
        ind = indexer[:,:,:]
        src = np.arange(27).reshape((3,3,3))

        assert_equal(ind(), (slice(None, None, 1), slice(None, None, 1), slice(None, None, 1)))
        assert_equal(src[ind()], src)

        with self.assertRaises(Exception) as cm:
            ind.flat()
        self.assertEqual(cm.exception.args[0], "Can't get shaped array of (slice(None, None, None), slice(None, None, None), slice(None, None, None)) because it has no source shape.")

        with self.assertRaises(Exception) as cm:
            ind.indexed_src_shape
        self.assertEqual(cm.exception.args[0], "Can't get indexed_src_shape of (slice(None, None, None), slice(None, None, None), slice(None, None, None)) because source shape is unknown.")
        with self.assertRaises(Exception) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't determine extent of array because source shape is not known.")

        ind.set_src_shape(src.shape)

        assert_equal(src.flat[ind.flat()], np.arange(27, dtype=int))

        assert_equal(ind.shaped_instance()(), (slice(0, 3, 1), slice(0, 3, 1), slice(0, 3, 1)))
        assert_equal(ind.as_array(flat=False), src)
        assert_equal(ind.as_array(), src.flat[:])
        assert_equal(ind.indexed_src_shape, (3,3,3))
        assert_equal(ind.min_src_dim, 3)

    def test_flat_slice_into_nd_source(self):
        ind = indexer[1:]
        src = np.arange(27).reshape((3,3,3))
        ind.set_src_shape(src.shape)

        assert_equal(ind(), slice(1, None, 1))

        expected = np.stack([np.arange(9, 18).reshape((3, 3)), np.arange(18, 27).reshape((3, 3))])

        assert_equal(expected, src[1:])
        assert_equal(src[ind()], expected)

        shaped_ind = ind.shaped_instance()

        assert_equal(shaped_ind.as_array(flat=True), expected.ravel())

        expected_nonflat = (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
                            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2]),
                            np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]))

        assert_equal(shaped_ind.as_array(flat=False), expected_nonflat)

    def test_slice_neg(self):
        ind = indexer[:-1,:,:2]
        src = np.arange(27).reshape((3,3,3))

        assert_equal(ind(), (slice(None, -1, 1), slice(None, None, 1), slice(None, 2, 1)))
        assert_equal(src[ind()], src[:-1,:,:2])

        with self.assertRaises(Exception) as cm:
            ind.indexed_src_shape
        self.assertEqual(cm.exception.args[0], "Can't get indexed_src_shape of (slice(None, -1, None), slice(None, None, None), slice(None, 2, None)) because source shape is unknown.")
        with self.assertRaises(Exception) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't determine extent of array because source shape is not known.")

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_instance()(), (slice(0, 2, 1), slice(0, 3, 1), slice(None, 2, 1)))
        assert_equal(ind.as_array(), np.arange(27, dtype=np.int32).reshape((3,3,3))[:-1,:,:2].ravel())
        assert_equal(ind.as_array(flat=False), np.arange(27, dtype=np.int32).reshape((3,3,3))[:-1,:,:2])
        assert_equal(ind.indexed_src_shape, (2,3,2))
        assert_equal(ind.min_src_dim, 3)

    def test_mult_arr(self):
        src = np.arange(27).reshape((3,3,3))
        ind = indexer[[0,2], :, [1,2]]

        assert_equal(ind(), ([0,2], slice(None, None, 1), [1,2]))
        with self.assertRaises(Exception) as cm:
            ind.indexed_src_shape
        self.assertEqual(cm.exception.args[0], "Can't get indexed_src_shape of ([0, 2], slice(None, None, None), [1, 2]) because source shape is unknown.")
        with self.assertRaises(Exception) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't determine extent of array because source shape is not known.")

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_instance()(), ([0, 2], slice(0, 3, 1), [1, 2]))
        assert_equal(ind.as_array(), np.arange(27, dtype=np.int32).reshape((3,3,3))[[0,2], :, [1,2]].ravel())
        assert_equal(ind.as_array(flat=False), np.arange(27, dtype=np.int32).reshape((3,3,3))[[0,2], :, [1,2]])
        assert_equal(ind.indexed_src_shape, (2,3))
        assert_equal(ind.min_src_dim, 3)

    def test_mult_arr_shaped(self):
        src = np.arange(9).reshape((3,3))
        ind = indexer[[[0,0],[2,2]], [[0,2], [0, 2]]]  # shoud get a 2x2 array with corner values of the 3x3

        assert_equal(ind(), ([[0,0],[2,2]], [[0,2], [0, 2]]))
        with self.assertRaises(Exception) as cm:
            ind.indexed_src_shape
        self.assertEqual(cm.exception.args[0], "Can't get indexed_src_shape of ([[0, 0], [2, 2]], [[0, 2], [0, 2]]) because source shape is unknown.")
        with self.assertRaises(Exception) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't determine extent of array because source shape is not known.")

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_instance()(), ([[0,0],[2,2]], [[0,2], [0, 2]]))
        assert_equal(ind.as_array(), np.arange(9, dtype=np.int32).reshape((3,3))[[[0,0],[2,2]], [[0,2], [0, 2]]].ravel())
        assert_equal(ind.as_array(flat=False), np.arange(9, dtype=np.int32).reshape((3,3))[[[0,0],[2,2]], [[0,2], [0, 2]]])
        assert_equal(ind.indexed_src_shape, (2,2))
        assert_equal(ind.min_src_dim, 2)

    def test_flat_shaped_src_inds(self):
        src = np.arange(24).reshape((8,3))  # 2D source
        ind = indexer([1,3,5,4,22, -4, 11, 3], flat_src=True)

        assert_equal(ind(), [1,3,5,4,22, -4, 11, 3])

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_array(), np.array([1,3,5,4,22, 20, 11, 3]))
        assert_equal(ind.as_array(), np.array([1,3,5,4,22, -4, 11, 3]))
        assert_equal(ind.as_array(flat=False), np.array([1,3,5,4,22, -4, 11, 3]))
        assert_equal(ind.indexed_src_shape, (8,))
        assert_equal(ind.min_src_dim, 1)


class TestCombineRanges(unittest.TestCase):

    def test_empty(self):
        ranges = []
        result = combine_ranges(ranges)
        self.assertEqual(result, [])

    def test_single_range(self):
        ranges = [(1, 5)]
        result = combine_ranges(ranges)
        self.assertEqual(result, [(1, 5)])

    def test_contig_ranges(self):
        ranges = [(1, 5), (5, 10), (10, 15)]
        result = combine_ranges(ranges)
        self.assertEqual(result, [(1, 15)])

    def test_non_overlapping_ranges(self):
        ranges = [(1, 5), (6, 10), (11, 15)]
        result = combine_ranges(ranges)
        self.assertEqual(result, [(1, 5), (6, 10), (11, 15)])

    def test_mixed_ranges(self):
        ranges = [(1, 5), (5, 10), (11, 15)]
        result = combine_ranges(ranges)
        self.assertEqual(result, [(1, 10), (11, 15)])


if __name__ == '__main__':
    unittest.main()
