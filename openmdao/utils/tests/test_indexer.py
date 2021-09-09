import unittest

import numpy as np
from numpy.testing import assert_equal

import openmdao.api as om
from openmdao.utils.indexer import indexer
from openmdao.utils.assert_utils import assert_near_equal


class IndexerTestCase(unittest.TestCase):
    def test_int(self):
        ind = indexer[4]
        src = np.arange(10)

        assert_equal(ind(), 4)
        assert_equal(ind.as_array(), np.array([4]))
        assert_equal(ind.shaped_array(), np.array([4]))
        assert_equal(ind.flat(), np.array([4]))

        assert_equal(src[ind()], 4)
        assert_equal(src[ind.as_array()], np.array([4]))
        assert_equal(src[ind.shaped_array()], np.array([4]))
        assert_equal(src[ind.flat()], np.array([4]))

        assert_equal(ind.shape, 1)
        assert_equal(ind.src_ndim, 1)

    def test_neg_int(self):
        ind = indexer[-4]
        src = np.arange(10)

        assert_equal(ind(), -4)
        assert_equal(ind.as_array(), np.array([-4]))
        assert_equal(ind.flat(), np.array([-4]))

        assert_equal(src[ind()], 6)
        assert_equal(src[ind.as_array()], np.array([6]))
        assert_equal(src[ind.flat()], np.array([6]))

        assert_equal(ind.shape, 1)
        assert_equal(ind.src_ndim, 1)

        try:
            ind.shaped_array()
        except Exception as err:
            self.assertEqual(str(err), "Can't get shaped array of -4 because it has no source shape.")
        else:
            self.fail("Exception expected")

        ind.set_src_shape(src.shape)
        assert_equal(ind.shaped_array(), np.array([6]))
        assert_equal(src[ind.shaped_array()], np.array([6]))
        assert_equal(ind.shaped_instance()(), 6)

    def test_simple_noncontig_arr(self):
        ind = indexer[[5, 3, 7, 1]]
        src = np.arange(10)

        assert_equal(ind(), np.array([5,3,7,1]))
        assert_equal(ind.as_array(), np.array([5,3,7,1]))
        assert_equal(ind.shaped_array(), np.array([5,3,7,1]))
        assert_equal(ind.flat(), np.array([5,3,7,1]))

        assert_equal(src[ind()], np.array([5,3,7,1]))
        assert_equal(src[ind.as_array()], np.array([5,3,7,1]))
        assert_equal(src[ind.shaped_array()], np.array([5,3,7,1]))
        assert_equal(src[ind.flat()], np.array([5,3,7,1]))
        assert_equal(ind.shaped_instance()(), np.array([5,3,7,1]))

        assert_equal(ind.shape, (4,))
        assert_equal(ind.src_ndim, 1)

    def test_contiguous_arr(self):
        ind = indexer[[3, 4, 5]]
        src = np.arange(10)

        assert_equal(ind(), [3, 4, 5])
        assert_equal(ind.as_array(), np.array([3, 4, 5]))
        assert_equal(ind.shaped_array(), np.array([3, 4, 5]))
        assert_equal(ind.flat(), [3, 4, 5])

        assert_equal(src[ind()], np.array([3, 4, 5]))
        assert_equal(src[ind.as_array()], np.array([3, 4, 5]))
        assert_equal(src[ind.shaped_array()], np.array([3, 4, 5]))
        assert_equal(src[ind.flat()], np.array([3, 4, 5]))

        assert_equal(ind.shape, (3,))
        assert_equal(ind.src_ndim, 1)
        assert_equal(ind.shaped_instance()(), [3, 4, 5])

    def test_arr_to_slice_step2(self):
        ind = indexer[[1,3,5,7,9]]
        src = np.arange(10)

        assert_equal(ind(), [1,3,5,7,9])
        assert_equal(ind.as_array(), np.array([1,3,5,7,9]))
        assert_equal(ind.shaped_array(), np.array([1,3,5,7,9]))
        assert_equal(ind.flat(), [1,3,5,7,9])

        assert_equal(src[ind()], np.array([1,3,5,7,9]))
        assert_equal(src[ind.as_array()], np.array([1,3,5,7,9]))
        assert_equal(src[ind.shaped_array()], np.array([1,3,5,7,9]))
        assert_equal(src[ind.flat()], np.array([1,3,5,7,9]))

        assert_equal(ind.shape, (5,))
        assert_equal(ind.src_ndim, 1)

    def test_neg_arr(self):
        ind = indexer[[5, 3, 7, -1]]
        src = np.arange(10)

        assert_equal(ind(), np.array([5, 3, 7, -1]))
        assert_equal(ind.as_array(), np.array([5, 3, 7, -1]))
        assert_equal(ind.flat(), np.array([5, 3, 7, -1]))

        assert_equal(src[ind()], np.array([5,3,7,9]))
        assert_equal(src[ind.as_array()], np.array([5,3,7,9]))
        assert_equal(src[ind.flat()], np.array([5,3,7,9]))

        assert_equal(ind.shape, (4,))
        assert_equal(ind.src_ndim, 1)

        try:
            ind.shaped_array()
        except Exception as err:
            self.assertEqual(str(err), "Can't get shaped array of [ 5  3  7 -1] because it has no source shape.")
        else:
            self.fail("Exception expected")

        ind.set_src_shape(10)
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

        assert_equal(ind.shape, (3,))
        assert_equal(ind.src_ndim, 1)

        try:
            ind.shaped_array()
        except Exception as err:
            self.assertEqual(str(err), "Can't get shaped array of [-1 -3 -5] because it has no source shape.")
        else:
            self.fail("Exception expected")

        ind.set_src_shape(10)
        assert_equal(ind.shape, (3,))
        assert_equal(ind.src_ndim, 1)

        assert_equal(ind.shaped_instance()(), np.array([9, 7, 5]))
        assert_equal(ind.shaped_array(), np.array([9, 7, 5]))

    def test_full_slice(self):
        ind = indexer[:]
        src = np.arange(10)

        assert_equal(ind(), slice(None))
        assert_equal(ind.flat(), slice(None))

        assert_equal(src[ind()], np.arange(10, dtype=int))
        assert_equal(src[ind.flat()], np.arange(10, dtype=int))

        assert_equal(ind.src_ndim, 1)

        with self.assertRaises(RuntimeError) as cm:
            ind.shape
        self.assertEqual(cm.exception.args[0], "Can't get shape of slice(None, None, None) because source shape is unknown.")
        with self.assertRaises(ValueError) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't get shaped array of slice(None, None, None) because it has no source shape.")

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_instance()(), slice(0, 10, 1))
        assert_equal(ind.as_array(), np.arange(10, dtype=int))
        assert_equal(ind.shape, (10,))
        assert_equal(ind.src_ndim, 1)

    def test_neg_start_slice(self):
        ind = indexer[-3:-6:-1]
        src = np.arange(10)

        assert_equal(ind(), slice(-3, -6, -1))
        assert_equal(ind.flat(), slice(-3, -6, -1))

        assert_equal(src[ind()], np.array([7, 6, 5]))
        assert_equal(src[ind.flat()], np.array([7, 6, 5]))

        assert_equal(ind.shape, (3,))
        assert_equal(ind.src_ndim, 1)

        with self.assertRaises(ValueError) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't get shaped array of slice(-3, -6, -1) because it has no source shape.")

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_instance()(), slice(7, 4, -1))
        assert_equal(ind.as_array(), np.array([7, 6, 5]))
        assert_equal(ind.shape, (3,))
        assert_equal(ind.src_ndim, 1)

    def test_none_start_slice(self):
        ind = indexer[:5]
        src = np.arange(10)

        assert_equal(ind(), slice(None, 5, None))
        assert_equal(ind.flat(), slice(None, 5, None))
        assert_equal(ind.as_array(), np.array([0, 1, 2, 3, 4]))

        assert_equal(src[ind()], np.array([0, 1, 2, 3, 4]))
        assert_equal(src[ind.flat()], np.array([0, 1, 2, 3, 4]))
        assert_equal(src[ind.as_array()], np.array([0, 1, 2, 3, 4]))

        assert_equal(ind.shape, (5,))
        assert_equal(ind.src_ndim, 1)

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_instance()(), slice(None, 5, None))
        assert_equal(ind.shape, (5,))
        assert_equal(ind.src_ndim, 1)

    def test_none_stop_slice(self):
        ind = indexer[3:]
        src = np.arange(10)

        assert_equal(ind(), slice(3, None))
        assert_equal(ind.flat(), slice(3, None))

        assert_equal(src[ind()], np.array([3, 4, 5, 6, 7, 8, 9]))
        assert_equal(src[ind.flat()], np.array([3, 4, 5, 6, 7, 8, 9]))

        assert_equal(ind.src_ndim, 1)

        with self.assertRaises(ValueError) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't get shaped array of slice(3, None, None) because it has no source shape.")

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_instance()(), slice(3, 10, 1))
        assert_equal(ind.as_array(), np.array([3, 4, 5, 6, 7, 8, 9]))
        assert_equal(ind.shape, (7,))
        assert_equal(ind.src_ndim, 1)

    def test_slice_neg_step(self):
        ind = indexer[6:2:-1]
        src = np.arange(10)

        assert_equal(ind(), slice(6, 2, -1))
        assert_equal(ind.flat(), slice(6, 2, -1))
        assert_equal(ind.as_array(), np.array([6, 5, 4, 3]))

        assert_equal(src[ind()], np.array([6, 5, 4, 3]))
        assert_equal(src[ind.flat()], np.array([6, 5, 4, 3]))
        assert_equal(src[ind.as_array()], np.array([6, 5, 4, 3]))
        assert_equal(src[ind.shaped_array()], np.array([6, 5, 4, 3]))

        assert_equal(ind.shape, (4,))
        assert_equal(ind.src_ndim, 1)

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_instance()(), slice(6, 2, -1))
        assert_equal(ind.shape, (4,))
        assert_equal(ind.src_ndim, 1)


class IndexerMultiDimTestCase(unittest.TestCase):
    def test_multi_slice(self):
        ind = indexer[:,:,:]
        src = np.arange(27).reshape((3,3,3))

        assert_equal(ind(), (slice(None), slice(None), slice(None)))
        assert_equal(src[ind()], src)

        with self.assertRaises(Exception) as cm:
            ind.flat()
        self.assertEqual(cm.exception.args[0], "Can't get shaped array of (slice(None, None, None), slice(None, None, None), slice(None, None, None)) because it has no source shape.")

        with self.assertRaises(RuntimeError) as cm:
            ind.shape
        self.assertEqual(cm.exception.args[0], "Can't get shape of slice(None, None, None) because source shape is unknown.")
        with self.assertRaises(ValueError) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't determine extent of array because source shape is not known.")

        ind.set_src_shape(src.shape)

        assert_equal(src.flat[ind.flat()], np.arange(27, dtype=int))

        assert_equal(ind.shaped_instance()(), (slice(0, 3, 1), slice(0, 3, 1), slice(0, 3, 1)))
        assert_equal(ind.as_array(flat=False), src)
        assert_equal(ind.as_array(), src.flat[:])
        assert_equal(ind.shape, (3,3,3))
        assert_equal(ind.src_ndim, 3)

    def test_slice_neg(self):
        ind = indexer[:-1,:,:2]
        src = np.arange(27).reshape((3,3,3))

        assert_equal(ind(), (slice(None, -1), slice(None), slice(None, 2)))
        assert_equal(src[ind()], src[:-1,:,:2])

        with self.assertRaises(RuntimeError) as cm:
            ind.shape
        self.assertEqual(cm.exception.args[0], "Can't get shape of slice(None, -1, None) because source shape is unknown.")
        with self.assertRaises(ValueError) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't determine extent of array because source shape is not known.")

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_instance()(), (slice(0, 2, 1), slice(0, 3, 1), slice(None, 2, None)))
        assert_equal(ind.as_array(), np.arange(27, dtype=np.int32).reshape((3,3,3))[:-1,:,:2].ravel())
        assert_equal(ind.as_array(flat=False), np.arange(27, dtype=np.int32).reshape((3,3,3))[:-1,:,:2])
        assert_equal(ind.shape, (2,3,2))
        assert_equal(ind.src_ndim, 3)

    def test_mult_arr(self):
        src = np.arange(27).reshape((3,3,3))
        ind = indexer[[0,2], :, [1,2]]

        assert_equal(ind(), ([0,2], slice(None, None, None), [1,2]))
        with self.assertRaises(RuntimeError) as cm:
            ind.shape
        self.assertEqual(cm.exception.args[0], "Can't get shape of slice(None, None, None) because source shape is unknown.")
        with self.assertRaises(ValueError) as cm:
            ind.as_array()
        self.assertEqual(cm.exception.args[0], "Can't determine extent of array because source shape is not known.")

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_instance()(), ([0, 2], slice(0, 3, 1), [1, 2]))
        assert_equal(ind.as_array(), np.arange(27, dtype=np.int32).reshape((3,3,3))[[0,2], :, [1,2]].ravel())
        assert_equal(ind.as_array(flat=False), np.arange(27, dtype=np.int32).reshape((3,3,3))[[0,2], :, [1,2]])
        assert_equal(ind.shape, (2,3))
        assert_equal(ind.src_ndim, 3)

    def test_nonflat_1D_src_inds(self):
        # test using our special format where the src inds are basically an array of a certain shape
        # containing tuples that indicate the indices into each dimension of the source.
        src = np.arange(24).reshape((8,3))  # 2D source
        ind = indexer([(1,2), (3,2), (5,0), (4, 1)], new_style=False)

        assert_equal(ind(), (np.array([1, 3, 5, 4]), np.array([2, 2, 0, 1])))

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_array(), np.array([5,11,15,13]))
        assert_equal(ind.as_array(), np.array([5,11,15,13]))
        assert_equal(ind.as_array(flat=False), np.array([5,11,15,13]))
        assert_equal(ind.shape, (4,))
        assert_equal(ind.src_ndim, 2)

    def test_nonflat_2D_src_inds(self):
        # test using our special format where the src inds are basically an array of a certain shape
        # containing tuples that indicate the indices into each dimension of the source.
        src = np.arange(24).reshape((8,3))  # 2D source
        ind = indexer([((1,2), (3,2)),
                       ((5,0), (4, 1))], new_style=False)

        assert_equal(ind(), (np.array([1, 3, 5, 4]), np.array([2, 2, 0, 1])))

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_array(), np.array([5,11,15,13]))
        assert_equal(ind.as_array(), np.array([5,11,15,13]))
        assert_equal(ind.as_array(flat=False), np.array([[5,11],[15,13]]))
        assert_equal(ind.shape, (2,2))
        assert_equal(ind.src_ndim, 2)

    def test_nonflat_2D_neg_src_inds(self):
        # test using our special format where the src inds are basically an array of a certain shape
        # containing tuples that indicate the indices into each dimension of the source.
        src = np.arange(24).reshape((8,3))  # 2D source
        ind = indexer([((1,2), (3,2)),
                       ((5,0), (4, -2))], new_style=False)

        assert_equal(ind(), (np.array([1, 3, 5, 4]), np.array([ 2,  2,  0, -2])))

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_array(), np.array([5,11,15,13]))
        assert_equal(ind.as_array(), np.array([5,11,15,13]))
        assert_equal(ind.as_array(flat=False), np.array([[5,11],[15,13]]))
        assert_equal(ind.shape, (2,2))
        assert_equal(ind.src_ndim, 2)

    def test_flat_shaped_src_inds(self):
        src = np.arange(24).reshape((8,3))  # 2D source
        ind = indexer([[1,3,5,4], [22, -4, 11, 3]], flat=True)

        assert_equal(ind(), [[1,3,5,4], [22, -4, 11, 3]])

        ind.set_src_shape(src.shape)

        assert_equal(ind.shaped_array(), np.array([1,3,5,4,22, 20, 11, 3]))
        assert_equal(ind.as_array(), np.array([1,3,5,4,22, -4, 11, 3]))
        assert_equal(ind.as_array(flat=False), np.array([[1,3,5,4], [22, -4, 11, 3]]))
        assert_equal(ind.shape, (2,4))
        assert_equal(ind.src_ndim, 1)


class ConversionTestCase(unittest.TestCase):
    def test_old_style(self):
        inds = om.to_numpy_style([(1,2), (3,2), (5,0), (4, 1)])
        self.assertEqual(inds, ([1, 3, 5, 4], [2,2,0,1]))

        inds = om.to_numpy_style([[(0, 0), (-1, 1)],
                                    [(2, 1), (1, 1)]])
        self.assertEqual(inds, ([0, -1, 2, 1], [0, 1, 1, 1]))


if __name__ == '__main__':
    unittest.main()
