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
                    self.assertEqual(mapper[i], ex_i)

    def test_key2range(self):
        for mclass in (RangeTree, FlatRangeMapper):
            with self.subTest(msg=f'{mclass.__name__} test'):
                mapper = mclass(_data.items())
                keys = ['a', 'b', 'x', 'y', 'z']
                expected = [(0, 1), (1, 9), (9, 15), (15, 36), (36, 42)]
                for key, ex in zip(keys, expected):
                    self.assertEqual(mapper.key2range(key), ex)

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
                    self.assertEqual(mapper.key2size(key), ex)

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
                expected = [('a',0), ('b',0), ('b',6), ('x',0), ('x',5), ('y',0), ('y',7), ('z',5),
                            (None, None), (None, None)]
                for i, ex in zip(inds, expected):
                    self.assertEqual(mapper.index2key_rel(i), ex)

    def test_iter(self):
        for mclass in (RangeTree, FlatRangeMapper):
            with self.subTest(msg=f'{mclass.__name__} test'):
                mapper = mclass(_data.items())
                expected = [('a', 0, 1), ('b', 1, 9), ('x', 9, 15), ('y', 15, 36), ('z', 36, 42)]
                for got, ex in zip(mapper, expected):
                    self.assertEqual(got, ex)

    def test_between_iter(self):
        for mclass in (RangeTree, FlatRangeMapper):
            with self.subTest(msg=f'{mclass.__name__} test'):
                mapper = mclass(_data.items())
                self.assertEqual(list(mapper.between_iter('a', 'a')), [('a', 0, 1)])
                self.assertEqual(list(mapper.between_iter('b', 'y')), [('b', 0, 8), ('x', 0, 6), ('y', 0, 21)])
                self.assertEqual(list(mapper.between_iter('y', 'z')), [('y', 0, 21), ('z', 0, 6)])
                self.assertEqual(list(mapper.between_iter('z', 'z')), [('z', 0, 6)])

    def test_overlap_iter(self):
        otherdata = {
            'a': 1,   # 0:1
            'b': 10,  # 1:11
            'x': 6,   # 11:17
            'y': 18,  # 17:35
            'z': 7,   # 35:42
        }

        for mclass in (RangeTree, FlatRangeMapper):
            with self.subTest(msg=f'{mclass.__name__} test'):
                mapper = mclass(_data.items())
                other_mapper = mclass(otherdata.items())
                self.assertEqual(list(mapper.overlap_iter('a', other_mapper)), [('a', 0, 1, 'a', 0, 1)])
                self.assertEqual(list(mapper.overlap_iter('y', other_mapper)), [('y', 0, 2, 'x', 4, 6), ('y', 2, 20, 'y', 0, 18), ('y', 20, 21, 'z', 0, 1)])
                self.assertEqual(list(mapper.overlap_iter('z', other_mapper)), [('z', 0, 6, 'z', 1, 7)])

    def test_dump(self):
        with unittest.mock.patch('builtins.print') as mock_print:
            for mclass in (RangeTree, FlatRangeMapper):
                with self.subTest(msg=f'{mclass.__name__} test'):
                    mapper = mclass(_data.items())
                    mapper.dump()
                    mock_print.assert_any_call('a: 0 - 1')
                    mock_print.assert_any_call('b: 1 - 9')
                    mock_print.assert_any_call('x: 9 - 15')
                    mock_print.assert_any_call('y: 15 - 36')
                    mock_print.assert_any_call('z: 36 - 42')


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

    def test_build(self):
        sizes = [('d', 4), ('e', 5), ('f', 6)]
        tree = RangeTree(sizes)
        root = tree.root
        self.assertEqual(root.key, 'e')
        self.assertEqual(root.start, 4)
        self.assertEqual(root.stop, 9)
        self.assertEqual(root.left.key, 'd')
        self.assertEqual(root.right.key, 'f')


if __name__ == "__main__":
    unittest.main()
