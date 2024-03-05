import unittest

import numpy as np
from numpy.testing import assert_equal

import openmdao.api as om
from openmdao.utils.relevance import Relevance, SetChecker, _vars2systems, \
    _get_set_checker
from openmdao.utils.assert_utils import assert_near_equal


_full_set = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'}

class TestSetChecker(unittest.TestCase):
    def test_contains(self):
        checker = SetChecker(set(['a', 'b', 'c']))
        self.assertTrue('a' in checker)
        self.assertFalse('d' in checker)

    def test_iter(self):
        checker = SetChecker(set(['a', 'b', 'c']))
        self.assertEqual(sorted(list(checker)), ['a', 'b', 'c'])

    def test_len(self):
        checker = SetChecker(set(['a', 'b', 'c']))
        self.assertEqual(len(checker), 3)

    def test_to_set(self):
        checker = SetChecker(set(['a', 'b', 'c']))
        self.assertEqual(checker.to_set(), set(['a', 'b', 'c']))

    def test_intersection(self):
        checker = SetChecker(set(['a', 'b', 'c']))
        self.assertEqual(checker.intersection(set(['b', 'c', 'd'])), set(['b', 'c']))

    def test_invert(self):
        checker = SetChecker(set(['a', 'b', 'c']), full_set=set(['a', 'b', 'c', 'd', 'e']), invert=True)
        self.assertTrue('d' in checker)
        self.assertFalse('a' in checker)
        self.assertEqual(len(checker), 2)
        self.assertEqual(checker.to_set(), set(['d', 'e']))
        self.assertEqual(checker.intersection(set(['b', 'c', 'd'])), set(['d']))


class TestRelevance(unittest.TestCase):
    def test_vars2systems(self):
        names = ['abc.def.g', 'xyz.pdq.bbb', 'aaa.xxx', 'foobar.y']
        expected = {'abc', 'abc.def', 'xyz', 'xyz.pdq', 'aaa', 'foobar', ''}
        self.assertEqual(_vars2systems(names), expected)

    def test_set_checker_invert(self):
        checker = _get_set_checker({'a', 'b', 'c', 'f', 'g', 'h', 'i', 'j'}, _full_set)
        self.assertEqual(checker._invert, True)
        self.assertEqual(checker._full_set, _full_set)
        self.assertEqual(checker._set, {'d', 'e'})
        self.assertTrue('c' in checker)
        self.assertFalse('d' in checker)

    def test_set_checker(self):
        checker = _get_set_checker({'a','c'}, _full_set)
        self.assertEqual(checker._invert, False)
        self.assertEqual(checker._full_set, None)
        self.assertEqual(checker._set, {'a', 'c'})
        self.assertTrue('c' in checker)
        self.assertFalse('d' in checker)
