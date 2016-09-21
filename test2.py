from __future__ import division
import numpy
import unittest

from system import Component, Group
from problem import Problem

# Systems: R > C1, C2, C3, C4
# Variables: v1, v2, v3, v4; all depend on each other


class Comp1(Component):

    def initialize_variables(self, comm):
        self.add_input('v2')
        self.add_input('v3')
        self.add_input('v4')
        self.add_output('v1')


class Comp2(Component):

    def initialize_variables(self, comm):
        self.add_input('v1')
        self.add_input('v3')
        self.add_input('v4')
        self.add_output('v2')


class Comp3(Component):

    def initialize_variables(self, comm):
        self.add_input('v1')
        self.add_input('v2')
        self.add_input('v4')
        self.add_output('v3')


class Comp4(Component):

    def initialize_variables(self, comm):
        self.add_input('v1')
        self.add_input('v2')
        self.add_input('v3')
        self.add_output('v4')


class GroupG(Group):

    def add_subsystems(self):
        self.add_subsystem(Comp1('C1', promotes_all=True))
        self.add_subsystem(Comp2('C2', promotes_all=True))
        self.add_subsystem(Comp3('C3', promotes_all=True))
        self.add_subsystem(Comp4('C4', promotes_all=True))


class Test(unittest.TestCase):

    def setUp(self):
        self.p = Problem(GroupG('G')).setup()

    def test_variable_names(self):
        root = self.p.root
        names = root.variable_names['output']
        self.assertEqual(names, ['v1', 'v2', 'v3', 'v4'])


if __name__ == '__main__':
    unittest.main()
