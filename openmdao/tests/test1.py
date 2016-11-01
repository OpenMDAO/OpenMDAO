from __future__ import division
import numpy
import unittest

from openmdao.api import Problem, IndepVarComp, ExplicitComponent, Group
from openmdao.parallel_api import PETScVector

#      (A) -> x
# x -> (B) -> f


class CompA(ExplicitComponent):

    def initialize_variables(self):
        self.add_output('x')


class CompB(ExplicitComponent):

    def initialize_variables(self):
        self.add_input('x')
        self.add_output('f')

    def compute(self, inputs, outputs):
        outputs['f'] = 2 * inputs['x']


class GroupG(Group):

    def add_subsystems(self):
        self.add_subsystem('A', IndepVarComp('x', 0.))
        self.add_subsystem('B', CompB())
        self.connect('A.x', 'B.x')


class Test(unittest.TestCase):

    def setUp(self):
        group = GroupG()
        group.add_subsystems()
        self.p = Problem(group).setup(PETScVector)
        self.p.root.set_solver_print(False)

    def assertEqualArrays(self, a, b):
        self.assertTrue(numpy.linalg.norm(a-b) < 1e-15)

    def assertList(self, ab_list):
        for a, b in ab_list:
            self.assertEqualArrays(a, b)

    def test_subsystems(self):
        root = self.p.root

        self.assertEqual(len(root._subsystems_allprocs), 2)
        self.assertEqual(len(root._subsystems_myproc), 2)

    def test__variable_allprocs_names(self):
        root = self.p.root
        compA = root.get_subsystem('A')
        self.assertEqual(compA._variable_allprocs_names['output'], ['x'])

    def test__variable_myproc_indices(self):
        root_inds = self.p.root._variable_myproc_indices
        compA_inds = self.p.root.get_subsystem('A')._variable_myproc_indices
        compB_inds = self.p.root.get_subsystem('B')._variable_myproc_indices

        self.assertEqualArrays(root_inds['input'], numpy.array([0]))
        self.assertEqualArrays(root_inds['output'], numpy.array([0,1]))

        self.assertEqualArrays(compA_inds['input'], numpy.array([]))
        self.assertEqualArrays(compA_inds['output'], numpy.array([0]))

        self.assertEqualArrays(compB_inds['input'], numpy.array([0]))
        self.assertEqualArrays(compB_inds['output'], numpy.array([1]))

    def test__variable_allprocs_ranges(self):
        root_rng = self.p.root._variable_allprocs_range
        compA_rng = self.p.root.get_subsystem('A')._variable_allprocs_range
        compB_rng = self.p.root.get_subsystem('B')._variable_allprocs_range

        self.assertEqualArrays(root_rng['input'], numpy.array([0,1]))
        self.assertEqualArrays(root_rng['output'], numpy.array([0,2]))

        self.assertEqualArrays(compA_rng['input'], numpy.array([0,0]))
        self.assertEqualArrays(compA_rng['output'], numpy.array([0,1]))

        self.assertEqualArrays(compB_rng['input'], numpy.array([0,1]))
        self.assertEqualArrays(compB_rng['output'], numpy.array([1,2]))

    def test_connections(self):
        root = self.p.root

        self.assertEqual(root._variable_connections_indices[0][0], 0)
        self.assertEqual(root._variable_connections_indices[0][1], 0)

    def test_GS(self):
        root = self.p.root

        if root.comm.size == 1:
            compA = root.get_subsystem('A')
            compB = root.get_subsystem('B')

            if root.comm.rank == 0:
                self.assertList([
                [root._outputs['A.x'], 0],
                [compA._outputs['x'],  0],
                ])
                compA._outputs['x'] = 10
            if root.comm.rank == 2:
                self.assertList([
                [compB._inputs['x'],   0],
                [compB._outputs['f'],  0],
                ])

            root._solve_nonlinear()

            if root.comm.rank == 0:
                self.assertList([
                [root._outputs['A.x'], 10],
                [compA._outputs['x'],  10],
                ])
            if root.comm.rank == 2:
                self.assertList([
                [compB._inputs['x'],   10],
                [compB._outputs['f'],  20],
                ])

if __name__ == '__main__':
    unittest.main()
