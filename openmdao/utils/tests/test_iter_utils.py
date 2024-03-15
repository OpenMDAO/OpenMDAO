import unittest

from openmdao.utils.iter_utils import size2range_iter, meta2item_iter, meta2items_iter


class TestSizeIter2RangeIter(unittest.TestCase):
    def test_size2range_iter(self):
        # Test with an empty iterator
        size_iter = iter([])
        result = list(size2range_iter(size_iter))
        self.assertEqual(result, [])

        # Test with a single element iterator
        size_iter = iter([('var1', 5)])
        result = list(size2range_iter(size_iter))
        self.assertEqual(result, [('var1', (0, 5))])

        # Test with multiple elements
        size_iter = iter([('var1', 5), ('var2', 3), ('var3', 7)])
        result = list(size2range_iter(size_iter))
        self.assertEqual(result, [('var1', (0, 5)), ('var2', (5, 8)), ('var3', (8, 15))])

        # Test with non-zero start
        size_iter = iter([('var1', 5), ('var2', 3), ('var3', 7)])
        result = list(size2range_iter(size_iter))
        self.assertEqual(result, [('var1', (0, 5)), ('var2', (5, 8)), ('var3', (8, 15))])

        # Test with sizes of zero
        size_iter = iter([('var1', 0), ('var2', 0), ('var3', 0)])
        result = list(size2range_iter(size_iter))
        self.assertEqual(result, [('var1', (0, 0)), ('var2', (0, 0)), ('var3', (0, 0))])

        # Test with mixed sizes of zero and nonzero
        size_iter = iter([('var1', 0), ('var2', 5), ('var3', 0)])
        result = list(size2range_iter(size_iter))
        self.assertEqual(result, [('var1', (0, 0)), ('var2', (0, 5)), ('var3', (5, 5))])


class TestMeta2ItemIter(unittest.TestCase):
    def test_meta2item_iter_empty(self):
        # Test with an empty iterator
        meta_iter = iter([])
        result = list(meta2item_iter(meta_iter, 'item1'))
        self.assertEqual(result, [])

    def test_meta2item_iter_single(self):
        # Test with a single element iterator
        meta_iter = iter([('var1', {'item1': 5})])
        result = list(meta2item_iter(meta_iter, 'item1'))
        self.assertEqual(result, [('var1', 5)])

    def test_meta2item_iter_multiple(self):
        # Test with multiple elements
        meta_iter = iter([('var1', {'item1': 5}), ('var2', {'item1': 3}), ('var3', {'item1': 7})])
        result = list(meta2item_iter(meta_iter, 'item1'))
        self.assertEqual(result, [('var1', 5), ('var2', 3), ('var3', 7)])

    def test_meta2item_iter_missing_item(self):
        # Test with missing item in metadata
        meta_iter = iter([('var1', {'item1': 5}), ('var2', {'item2': 3}), ('var3', {'item1': 7})])
        with self.assertRaises(KeyError):
            list(meta2item_iter(meta_iter, 'item1'))

    def test_meta2item_iter_different_items(self):
        # Test with different items in metadata
        meta_iter = iter([('var1', {'item1': 5, 'item2': 10}), ('var2', {'item1': 3, 'item2': 6}), ('var3', {'item1': 7, 'item2': 14})])
        result = list(meta2item_iter(meta_iter, 'item2'))
        self.assertEqual(result, [('var1', 10), ('var2', 6), ('var3', 14)])


class TestMeta2ItemsIter(unittest.TestCase):
    def test_meta2items_iter_empty(self):
        # Test with an empty iterator
        meta_iter = iter([])
        items = ['item1', 'item2']
        result = list(meta2items_iter(meta_iter, items))
        self.assertEqual(result, [])

    def test_meta2items_iter_single(self):
        # Test with a single element iterator
        meta_iter = iter([('var1', {'item1': 5, 'item2': 10})])
        items = ['item1', 'item2']
        result = list(meta2items_iter(meta_iter, items))
        self.assertEqual(result, [['var1', 5, 10]])

    def test_meta2items_iter_multiple(self):
        # Test with multiple elements
        meta_iter = iter([('var1', {'item1': 5, 'item2': 10}), ('var2', {'item1': 3, 'item2': 6}), ('var3', {'item1': 7, 'item2': 14})])
        items = ['item1', 'item2']
        result = list(meta2items_iter(meta_iter, items))
        self.assertEqual(result, [['var1', 5, 10], ['var2', 3, 6], ['var3', 7, 14]])

    def test_meta2items_iter_missing_item(self):
        # Test with missing item in metadata
        meta_iter = iter([('var1', {'item1': 5, 'item2': 10}), ('var2', {'item1': 3}), ('var3', {'item1': 7, 'item2': 14})])
        items = ['item1', 'item2']
        with self.assertRaises(KeyError):
            list(meta2items_iter(meta_iter, items))

    def test_meta2items_iter_different_items(self):
        # Test with different items in metadata
        meta_iter = iter([('var1', {'item1': 5, 'item2': 10, 'item3': 15}), ('var2', {'item1': 3, 'item2': 6, 'item3': 9}), ('var3', {'item1': 7, 'item2': 14, 'item3': 21})])
        items = ['item1', 'item3']
        result = list(meta2items_iter(meta_iter, items))
        self.assertEqual(result, [['var1', 5, 15], ['var2', 3, 9], ['var3', 7, 21]])


if __name__ == '__main__':
    unittest.main()
