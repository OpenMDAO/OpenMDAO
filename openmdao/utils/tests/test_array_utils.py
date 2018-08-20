from __future__ import print_function, division, absolute_import


import unittest

from openmdao.utils.array_utils import array_connection_compatible

class TestArrayConnectionCompatible(unittest.TestCase):

    def test_ones_at_both_ends(self):
        shape1 = (1, 1, 15, 3, 1, 7, 1, 1, 1, 1)
        shape2 = (1, 15, 3, 1, 7)
        self.assertTrue(array_connection_compatible(shape1, shape2))

    def test_ones_at_start(self):
        shape1 = (1, 1, 15, 3, 1, 7)
        shape2 = (1, 15, 3, 1, 7)
        self.assertTrue(array_connection_compatible(shape1, shape2))

    def test_ones_at_end(self):
        shape1 = (15, 3, 1, 7, 1, 1, 1, 1)
        shape2 = (1, 15, 3, 1, 7)
        self.assertTrue(array_connection_compatible(shape1, shape2))

    def test_ones_to_ones(self):
        shape1 = (1, 1, 1, 1, 1, 1, 1, 1)
        shape2 = (1, 1, 1, 1, 1)
        self.assertTrue(array_connection_compatible(shape1, shape2))

    def test_ones_to_one(self):
        shape1 = (1, 1, 1, 1, 1, 1, 1, 1)
        shape2 = (1,)
        self.assertTrue(array_connection_compatible(shape1, shape2))

    def test_matrix_to_vectorized_matrix(self):
        shape1 = (3, 3)
        shape2 = (1, 3, 3)
        self.assertTrue(array_connection_compatible(shape1, shape2))

    def test_known_incompatable(self):
        shape1 = (3, 3)
        shape2 = (3, 1, 3)
        self.assertFalse(array_connection_compatible(shape1, shape2))
