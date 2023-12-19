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

if __name__ == "__main__":
    unittest.main()
