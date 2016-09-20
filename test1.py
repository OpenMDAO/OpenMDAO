from __future__ import division
import numpy
import unittest

from system import Component, Group
from problem import Problem

#      (A) -> x
# x -> (B) -> f


class CompA(Component):

    def initialize_variables(self, comm):
        self.add_output('x')


class CompB(Component):

    def initialize_variables(self, comm):
        self.add_input('x')
        self.add_output('f')


class GroupG(Group):

    def add_subsystems(self):
        self.add_subsystem(CompA('A'))
        self.add_subsystem(CompB('B'))
        self.connect('A.x', 'B.x')


class Test(unittest.TestCase):

    def setUp(self):
        self.p = Problem(GroupG('G')).setup()

    def assertEqualArrays(self, a, b):
        self.assertTrue(numpy.linalg.norm(a-b) < 1e-15)

    def test_subsystems(self):
        root = self.p.root

        self.assertEqual(len(root.subsystems_allprocs), 2)
        self.assertEqual(len(root.subsystems_myproc), 2)

    def test_variable_names(self):
        root = self.p.root
        compA = root.get_subsystem('G.A')
        self.assertEqual(compA.variable_names['output'], ['x'])

    def test_variable_myproc_indices(self):
        root_inds = self.p.root.variable_myproc_indices
        compA_inds = self.p.root.get_subsystem('G.A').variable_myproc_indices
        compB_inds = self.p.root.get_subsystem('G.B').variable_myproc_indices

        self.assertEqualArrays(root_inds['input'], numpy.array([0]))
        self.assertEqualArrays(root_inds['output'], numpy.array([0,1]))

        self.assertEqualArrays(compA_inds['input'], numpy.array([]))
        self.assertEqualArrays(compA_inds['output'], numpy.array([0]))

        self.assertEqualArrays(compB_inds['input'], numpy.array([0]))
        self.assertEqualArrays(compB_inds['output'], numpy.array([1]))

    def test_variable_ranges(self):
        root_rng = self.p.root.variable_range
        compA_rng = self.p.root.get_subsystem('G.A').variable_range
        compB_rng = self.p.root.get_subsystem('G.B').variable_range

        self.assertEqualArrays(root_rng['input'], numpy.array([0,1]))
        self.assertEqualArrays(root_rng['output'], numpy.array([0,2]))

        self.assertEqualArrays(compA_rng['input'], numpy.array([0,0]))
        self.assertEqualArrays(compA_rng['output'], numpy.array([0,1]))

        self.assertEqualArrays(compB_rng['input'], numpy.array([0,1]))
        self.assertEqualArrays(compB_rng['output'], numpy.array([1,2]))

    def test_connections(self):
        root = self.p.root

        print root.variable_connections_found
        self.assertEqual(root.variable_connections_found[0][0], 0)
        self.assertEqual(root.variable_connections_found[0][1], 0)


if __name__ == '__main__':
    unittest.main()
