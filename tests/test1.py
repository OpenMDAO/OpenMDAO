from __future__ import division
import numpy
import unittest

from Blue.API import Problem, IndepVarComponent, ExplicitComponent, Group

#      (A) -> x
# x -> (B) -> f


class CompA(ExplicitComponent):

    def initialize_variables(self, comm):
        self.add_output('x')


class CompB(ExplicitComponent):

    def initialize_variables(self, comm):
        self.add_input('x')
        self.add_output('f')

    def compute(self):
        self.outputs['f'] = 2 * self.inputs['x']


class GroupG(Group):

    def add_subsystems(self):
        self.add_subsystem(IndepVarComponent('A', [('x', 0)]))
        self.add_subsystem(CompB('B'))
        self.connect('A.x', 'B.x')


class Test(unittest.TestCase):

    def setUp(self):
        group = GroupG('G')
        group.add_subsystems()
        self.p = Problem(group).setup()

    def assertEqualArrays(self, a, b):
        self.assertTrue(numpy.linalg.norm(a-b) < 1e-15)

    def test_subsystems(self):
        root = self.p.root

        self.assertEqual(len(root.subsystems_allprocs), 2)
        self.assertEqual(len(root.subsystems_myproc), 2)

    def test_variable_allprocs_names(self):
        root = self.p.root
        compA = root.get_subsystem('G.A')
        self.assertEqual(compA.variable_allprocs_names['output'], ['x'])

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

    def test_variable_allprocs_ranges(self):
        root_rng = self.p.root.variable_allprocs_range
        compA_rng = self.p.root.get_subsystem('G.A').variable_allprocs_range
        compB_rng = self.p.root.get_subsystem('G.B').variable_allprocs_range

        self.assertEqualArrays(root_rng['input'], numpy.array([0,1]))
        self.assertEqualArrays(root_rng['output'], numpy.array([0,2]))

        self.assertEqualArrays(compA_rng['input'], numpy.array([0,0]))
        self.assertEqualArrays(compA_rng['output'], numpy.array([0,1]))

        self.assertEqualArrays(compB_rng['input'], numpy.array([0,1]))
        self.assertEqualArrays(compB_rng['output'], numpy.array([1,2]))

    def test_connections(self):
        root = self.p.root

        self.assertEqual(root.variable_connections_indices[0][0], 0)
        self.assertEqual(root.variable_connections_indices[0][1], 0)

    def te0st_transfer(self):
        root = self.p.root

        if root.mpi_comm.size == 1:
            compA = root.get_subsystem('G.A')
            compB = root.get_subsystem('G.B')

            print root.outputs['A.x']
            print compA.outputs['x']
            print compB.inputs['x']

            compA.outputs['x'] = 10

            print
            print root.outputs['A.x']
            print compA.outputs['x']
            print compB.inputs['x']

            root.vector_transfers[0][None](root.inputs, root.outputs)

            print
            print root.outputs['A.x']
            print compA.outputs['x']
            print compB.inputs['x']

    def test_GS(self):
        self.setUp()

        root = self.p.root

        if root.mpi_comm.size == 1:
            compA = root.get_subsystem('G.A')
            compB = root.get_subsystem('G.B')

            print
            print root.outputs['A.x']
            print compA.outputs['x']
            print compB.inputs['x']
            print compB.outputs['f']

            compA.outputs['x'] = 10
            root.solve_nonlinear()

            print
            print root.outputs['A.x']
            print compA.outputs['x']
            print compB.inputs['x']
            print compB.outputs['f']

if __name__ == '__main__':
    unittest.main()
