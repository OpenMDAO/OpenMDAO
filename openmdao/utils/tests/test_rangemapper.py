import unittest

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.rangemapper import RangeMapper, RangeTree, FlatRangeMapper, MAX_FLAT_RANGE_SIZE


_data = {
    'a': 1,  # 0:1
    'b': 8,  # 1:9
    'x': 6,  # 9:15
    'y': 21, # 15:36
    'z': 6,  # 36:42
}


class TestRangeMapper(unittest.TestCase):
    def test_create(self):
        mapper = RangeMapper.create(_data.items())
        self.assertEqual(type(mapper), FlatRangeMapper)
        mapper = RangeMapper.create(_data.items(), max_flat_range_size=40)
        self.assertEqual(type(mapper), RangeTree)

    def test_get_item(self):
        for mclass in (RangeTree, FlatRangeMapper):
            with self.subTest(msg=f'{mclass.__name__} test'):
                mapper = mclass(_data.items())
                inds = [0, 1, 7, 9, 14, 15, 22, 41, 42, 43]
                expected = ['a', 'b', 'b', 'x', 'x', 'y', 'y', 'z', None, None]
                for i, ex_i in zip(inds, expected):
                    got = mapper[i]
                    self.assertEqual(got, ex_i)

    def test_key2range(self):
        for mclass in (RangeTree, FlatRangeMapper):
            with self.subTest(msg=f'{mclass.__name__} test'):
                mapper = mclass(_data.items())
                keys = ['a', 'b', 'x', 'y', 'z']
                expected = [(0, 1), (1, 9), (9, 15), (15, 36), (36, 42)]
                for key, ex in zip(keys, expected):
                    got = mapper.key2range(key)
                    self.assertEqual(got, ex)

                try:
                    mapper.key2range('bad')
                except KeyError:
                    pass
                else:
                    self.fail("Expected KeyError")

    def test_key2size(self):
        for mclass in (RangeTree, FlatRangeMapper):
            with self.subTest(msg=f'{mclass.__name__} test'):
                mapper = mclass(_data.items())
                keys = ['a', 'b', 'x', 'y', 'z']
                expected = [1, 8, 6, 21, 6]
                for key, ex in zip(keys, expected):
                    got = mapper.key2size(key)
                    self.assertEqual(got, ex)

                try:
                    mapper.key2size('bad')
                except KeyError:
                    pass
                else:
                    self.fail("Expected KeyError")

    def test_index2key_rel(self):
        for mclass in (RangeTree, FlatRangeMapper):
            with self.subTest(msg=f'{mclass.__name__} test'):
                mapper = mclass(_data.items())
                inds = [0, 1, 7, 9, 14, 15, 22, 41, 42, 43]
                expected = [('a',0), ('b',0), ('b',6), ('x',0), ('x',5), ('y',0), ('y',7), ('z',5), (None, None), (None, None)]
                for i, ex in zip(inds, expected):
                    got = mapper.index2key_rel(i)
                    self.assertEqual(got, ex)

    def test_iter(self):
        for mclass in (RangeTree, FlatRangeMapper):
            with self.subTest(msg=f'{mclass.__name__} test'):
                mapper = mclass(_data.items())
                expected = [('a', 0, 1), ('b', 1, 9), ('x', 9, 15), ('y', 15, 36), ('z', 36, 42)]
                for got, ex in zip(mapper, expected):
                    self.assertEqual(got, ex)


class TestFlatRangeMapper(unittest.TestCase):
    def setUp(self):
        self.sizes = [('a', 1), ('b', 2), ('c', 3)]
        self.mapper = FlatRangeMapper(self.sizes)

    def test_init(self):
        self.assertEqual(self.mapper._key2range, {'a': (0, 1), 'b': (1, 3), 'c': (3, 6)})
        self.assertEqual(self.mapper.ranges, [('a', 0, 1), ('b', 1, 3), ('b', 1, 3), ('c', 3, 6), ('c', 3, 6), ('c', 3, 6)])

    def test_get_item(self):
        self.assertEqual(self.mapper[0], 'a')
        self.assertEqual(self.mapper[1], 'b')
        self.assertEqual(self.mapper[2], 'b')
        self.assertEqual(self.mapper[3], 'c')
        self.assertEqual(self.mapper[4], 'c')
        self.assertEqual(self.mapper[5], 'c')
        self.assertIsNone(self.mapper[6])

    def test_iter(self):
        expected = [('a', 0, 1), ('b', 1, 3), ('c', 3, 6)]
        for got, ex in zip(self.mapper, expected):
            self.assertEqual(got, ex)

    def test_index2key_rel(self):
        self.assertEqual(self.mapper.index2key_rel(0), ('a', 0))
        self.assertEqual(self.mapper.index2key_rel(1), ('b', 0))
        self.assertEqual(self.mapper.index2key_rel(2), ('b', 1))
        self.assertEqual(self.mapper.index2key_rel(3), ('c', 0))
        self.assertEqual(self.mapper.index2key_rel(4), ('c', 1))
        self.assertEqual(self.mapper.index2key_rel(5), ('c', 2))
        self.assertEqual(self.mapper.index2key_rel(6), (None, None))


class TestRangeTree(unittest.TestCase):
    def setUp(self):
        self.sizes = [('a', 1), ('b', 2), ('c', 3)]
        self.range_tree = RangeTree(self.sizes)

    def test_init(self):
        self.assertEqual(self.range_tree.root.key, 'b')
        self.assertEqual(self.range_tree.root.start, 1)
        self.assertEqual(self.range_tree.root.stop, 3)
        self.assertEqual(self.range_tree.root.left.key, 'a')
        self.assertEqual(self.range_tree.root.right.key, 'c')

    def test_get_item(self):
        self.assertEqual(self.range_tree[0], 'a')
        self.assertEqual(self.range_tree[1], 'b')
        self.assertEqual(self.range_tree[2], 'b')
        self.assertEqual(self.range_tree[3], 'c')
        self.assertEqual(self.range_tree[4], 'c')
        self.assertEqual(self.range_tree[5], 'c')
        self.assertIsNone(self.range_tree[6])

    def test_iter(self):
        expected = [('a', 0, 1), ('b', 1, 3), ('c', 3, 6)]
        for got, ex in zip(self.range_tree, expected):
            self.assertEqual(got, ex)

    def test_index2key_rel(self):
        self.assertEqual(self.range_tree.index2key_rel(0), ('a', 0))
        self.assertEqual(self.range_tree.index2key_rel(1), ('b', 0))
        self.assertEqual(self.range_tree.index2key_rel(2), ('b', 1))
        self.assertEqual(self.range_tree.index2key_rel(3), ('c', 0))
        self.assertEqual(self.range_tree.index2key_rel(4), ('c', 1))
        self.assertEqual(self.range_tree.index2key_rel(5), ('c', 2))
        self.assertEqual(self.range_tree.index2key_rel(6), (None, None))

    def test_build(self):
        sizes = [('d', 4), ('e', 5), ('f', 6)]
        tree = RangeTree(sizes)
        root = tree.root
        self.assertEqual(root.key, 'e')
        self.assertEqual(root.start, 4)
        self.assertEqual(root.stop, 9)
        self.assertEqual(root.left.key, 'd')
        self.assertEqual(root.right.key, 'f')


class TestRangeMapper(unittest.TestCase):
    def setUp(self):
        self.sizes = [('a', 1), ('b', 2), ('c', 3)]
        self.mapper = RangeMapper.create(self.sizes)

    def test_create(self):
        mapper = RangeMapper.create(self.sizes)
        self.assertIsInstance(mapper, FlatRangeMapper)
        mapper = RangeMapper.create(self.sizes, max_flat_range_size=2)
        self.assertIsInstance(mapper, RangeTree)

    def test_key2range(self):
        self.assertEqual(self.mapper.key2range('a'), (0, 1))
        self.assertEqual(self.mapper.key2range('b'), (1, 3))
        self.assertEqual(self.mapper.key2range('c'), (3, 6))
        with self.assertRaises(KeyError):
            self.mapper.key2range('d')

    def test_key2size(self):
        self.assertEqual(self.mapper.key2size('a'), 1)
        self.assertEqual(self.mapper.key2size('b'), 2)
        self.assertEqual(self.mapper.key2size('c'), 3)
        with self.assertRaises(KeyError):
            self.mapper.key2size('d')

    def test_inds2keys(self):
        self.assertEqual(self.mapper.inds2keys([0, 1, 2, 3, 4, 5]), {'a', 'b', 'c'})
        self.assertEqual(self.mapper.inds2keys([0]), {'a'})
        self.assertEqual(self.mapper.inds2keys([1, 2]), {'b'})
        self.assertEqual(self.mapper.inds2keys([3, 4, 5]), {'c'})

    def test_between_iter(self):
        self.assertEqual(list(self.mapper.between_iter('a', 'c')), [('a', 0, 1), ('b', 0, 2), ('c', 0, 3)])
        self.assertEqual(list(self.mapper.between_iter('a', 'b')), [('a', 0, 1), ('b', 0, 2)])
        self.assertEqual(list(self.mapper.between_iter('b', 'c')), [('b', 0, 2), ('c', 0, 3)])

    def test_overlap_iter(self):
        other_mapper = RangeMapper.create([('x', 2), ('y', 3), ('z', 4)])
        self.assertEqual(list(self.mapper.overlap_iter('a', other_mapper)), [('a', 0, 1, 'x', 0, 1)])
        self.assertEqual(list(self.mapper.overlap_iter('b', other_mapper)), [('b', 0, 1, 'x', 1, 2), ('b', 1, 2, 'y', 0, 1)])
        self.assertEqual(list(self.mapper.overlap_iter('c', other_mapper)), [('c', 0, 2, 'y', 1, 3), ('c', 2, 3, 'z', 0, 1)])

    def test_dump(self):
        with unittest.mock.patch('builtins.print') as mock_print:
            self.mapper.dump()
            mock_print.assert_any_call('a: 0 - 1')
            mock_print.assert_any_call('b: 1 - 3')
            mock_print.assert_any_call('c: 3 - 6')

if __name__ == "__main__":
    unittest.main()
