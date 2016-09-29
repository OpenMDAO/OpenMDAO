from __future__ import division
import numpy
import unittest

from Blue.API import Problem, IndepVarComponent, ExplicitComponent, Group, PETScVector

# Systems: R > C1, C2, C3, C4
# Variables: v1, v2, v3, v4; all depend on each other


class Comp1(ExplicitComponent):

    def initialize_variables(self, comm):
        self.add_input('v2')
        self.add_input('v3')
        self.add_input('v4')
        self.add_output('v1', var_set=1)


class Comp2(ExplicitComponent):

    def initialize_variables(self, comm):
        self.add_input('v1')
        self.add_input('v3')
        self.add_input('v4')
        self.add_output('v2', var_set=2)


class Comp3(ExplicitComponent):

    def initialize_variables(self, comm):
        self.add_input('v1')
        self.add_input('v2')
        self.add_input('v4')
        self.add_output('v3', var_set=3)


class Comp4(ExplicitComponent):

    def initialize_variables(self, comm):
        self.add_input('v1')
        self.add_input('v2')
        self.add_input('v3')
        self.add_output('v4', var_set=4)


class GroupG(Group):

    def add_subsystems(self):
        self.add_subsystem(Comp1('C1', promotes_all=True))
        self.add_subsystem(Comp2('C2', promotes_all=True))
        self.add_subsystem(Comp3('C3', promotes_all=True))
        self.add_subsystem(Comp4('C4', promotes_all=True))


class Test(unittest.TestCase):

    def setUp(self):
        group = GroupG('G')
        group.add_subsystems()
        self.p = Problem(group, Vector=PETScVector).setup()
        self.p.root.mpi_proc_allocator.parallel = True

    def assertEqualArrays(self, a, b):
        self.assertTrue(numpy.linalg.norm(a-b) < 1e-15)

    def test_variable_allprocs_names(self):
        root = self.p.root
        names = root.variable_allprocs_names['output']
        self.assertEqual(names, ['v1', 'v2', 'v3', 'v4'])

    def test_variable_set_IDs(self):
        set_IDs = self.p.assembler.variable_set_IDs['output']
        self.assertEqual(set_IDs[1], 0)
        self.assertEqual(set_IDs[2], 1)
        self.assertEqual(set_IDs[3], 2)
        self.assertEqual(set_IDs[4], 3)

    def test_variable_set_indices(self):
        set_indices = self.p.assembler.variable_set_indices['output']
        array = numpy.array([[0,0],[1,0],[2,0],[3,0]])
        self.assertEqualArrays(set_indices, array)

    def test_vectors(self):
        root = self.p.root
        rank = root.mpi_comm.rank
        #print rank, self.p.assembler.variable_sizes['output'][0]
        if rank == 0:
            for key in ['v1', 'v2', 'v3', 'v4']:
                print key, root.outputs[key]

    def test_transfer(self):
        root = self.p.root

        if root.mpi_comm.size == 1:
            comp1 = root.get_subsystem('G.C1')
            comp2 = root.get_subsystem('G.C2')
            comp3 = root.get_subsystem('G.C3')
            comp4 = root.get_subsystem('G.C4')

            print
            for comp in [comp1, comp2, comp3, comp4]:
                print '*1', comp.inputs.views, comp.outputs.views

            comp1.outputs['v1'] = 1.0
            comp2.outputs['v2'] = 1.0
            comp3.outputs['v3'] = 1.0

            print
            for comp in [comp1, comp2, comp3, comp4]:
                print '*2', comp.inputs.views, comp.outputs.views

            root.vector_transfers[None]['fwd', 0](root.inputs, root.outputs)

            print
            for comp in [comp1, comp2, comp3, comp4]:
                print '*3', comp.inputs.views, comp.outputs.views

            root.vector_transfers[None][None](root.inputs, root.outputs)

            print
            for comp in [comp1, comp2, comp3, comp4]:
                print '*4', comp.inputs.views, comp.outputs.views



if __name__ == '__main__':
    unittest.main()
