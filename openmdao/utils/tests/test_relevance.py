import unittest

import numpy as np
from numpy.testing import assert_equal

import openmdao.api as om
from openmdao.utils.relevance import Relevance, _vars2systems
from openmdao.utils.assert_utils import assert_near_equal


class TestRelevance(unittest.TestCase):
    def test_vars2systems(self):
        names = ['abc.def.g', 'xyz.pdq.bbb', 'aaa.xxx', 'foobar.y']
        expected = {'abc', 'abc.def', 'xyz', 'xyz.pdq', 'aaa', 'foobar', ''}
        self.assertEqual(_vars2systems(names), expected)
