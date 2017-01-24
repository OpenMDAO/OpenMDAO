from __future__ import division
import numpy
import unittest

from openmdao.api import Problem, ExplicitComponent, Group, DefaultVector

try:
    from openmdao.parallel_api import PETScVector
except ImportError:
    PETScVector = None

# Systems: R > C1, C2, C3, C4
# Variables: v1, v2, v3, v4; all depend on each other


class Comp1(ExplicitComponent):

    def initialize_variables(self):
        self.add_input('v2')
        self.add_input('v3')
        self.add_input('v4')
        self.add_output('v1', var_set=1)


class Comp2(ExplicitComponent):

    def initialize_variables(self):
        self.add_input('v1')
        self.add_input('v3')
        self.add_input('v4')
        self.add_output('v2', var_set=2)


class Comp3(ExplicitComponent):

    def initialize_variables(self):
        self.add_input('v1')
        self.add_input('v2')
        self.add_input('v4')
        self.add_output('v3', var_set=3)


class Comp4(ExplicitComponent):

    def initialize_variables(self):
        self.add_input('v1')
        self.add_input('v2')
        self.add_input('v3')
        self.add_output('v4', var_set=4)


class GroupG(Group):

    def initialize(self):
        self.add_subsystem('C1', Comp1(), promotes=['*'])
        self.add_subsystem('C2', Comp2(), promotes=['*'])
        self.add_subsystem('C3', Comp3(), promotes=['*'])
        self.add_subsystem('C4', Comp4(), promotes=['*'])


class TestNumpyVec(unittest.TestCase):

    def setUp(self):
        group = GroupG()
        self.p = Problem(group).setup(DefaultVector)
        self.p.model._mpi_proc_allocator.parallel = True

    def assertEqualArrays(self, a, b):
        self.assertTrue(numpy.linalg.norm(a-b) < 1e-15)

    def assertList(self, ab_list):
        for a, b in ab_list:
            self.assertEqualArrays(a, b)

    def test__var_allprocs_names(self):
        root = self.p.model
        names = root._var_allprocs_names['output']
        self.assertEqual(names, ['v1', 'v2', 'v3', 'v4'])

    def test__variable_set_IDs(self):
        set_IDs = self.p._assembler._variable_set_IDs['output']
        self.assertEqual(set_IDs[1], 0)
        self.assertEqual(set_IDs[2], 1)
        self.assertEqual(set_IDs[3], 2)
        self.assertEqual(set_IDs[4], 3)

    def test__variable_set_indices(self):
        set_indices = self.p._assembler._variable_set_indices['output']
        array = numpy.array([[0,0],[1,0],[2,0],[3,0]])
        self.assertEqualArrays(set_indices, array)

    def test_transfer(self):
        root = self.p.model

        if root.comm.size == 1:
            comp1 = root.get_subsystem('C1')
            comp2 = root.get_subsystem('C2')
            comp3 = root.get_subsystem('C3')
            comp4 = root.get_subsystem('C4')

            comp1._outputs['v1'] = 2.0
            comp2._outputs['v2'] = 4.0
            comp3._outputs['v3'] = 6.0
            comp4._outputs['v4'] = 8.0

            self.assertList([
                [comp1._outputs['v1'], 2.0],
                [comp1._inputs['v2'],  1.0],
                [comp1._inputs['v3'],  1.0],
                [comp1._inputs['v4'],  1.0],
            ])

            self.assertList([
                [comp2._inputs['v1'],  1.0],
                [comp2._outputs['v2'], 4.0],
                [comp2._inputs['v3'],  1.0],
                [comp2._inputs['v4'],  1.0],
            ])

            root._vector_transfers['nonlinear']['fwd', 0](root._inputs, root._outputs)

            self.assertList([
                [comp1._outputs['v1'], 2.0],
                [comp1._inputs['v2'],  4.0],
                [comp1._inputs['v3'],  6.0],
                [comp1._inputs['v4'],  8.0],
            ])

            self.assertList([
                [comp2._inputs['v1'],  1.0],
                [comp2._outputs['v2'], 4.0],
                [comp2._inputs['v3'],  1.0],
                [comp2._inputs['v4'],  1.0],
            ])

            root._vector_transfers['nonlinear'][None](root._inputs, root._outputs)

            self.assertList([
                [comp1._outputs['v1'], 2.0],
                [comp1._inputs['v2'],  4.0],
                [comp1._inputs['v3'],  6.0],
                [comp1._inputs['v4'],  8.0],
            ])

            self.assertList([
                [comp2._inputs['v1'],  2.0],
                [comp2._outputs['v2'], 4.0],
                [comp2._inputs['v3'],  6.0],
                [comp2._inputs['v4'],  8.0],
            ])


@unittest.skipUnless(PETScVector, "PETSc is required.")
class TestPetscVec(TestNumpyVec):

    def setUp(self):
        group = GroupG()
        self.p = Problem(group).setup(PETScVector)
        self.p.model._mpi_proc_allocator.parallel = True


if __name__ == '__main__':
    unittest.main()
