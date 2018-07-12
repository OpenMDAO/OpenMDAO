from __future__ import division
import numpy as np
import unittest

from openmdao.api import Problem, ExplicitComponent, Group, ParallelGroup
from openmdao.utils.assert_utils import assert_rel_error

try:
    from openmdao.parallel_api import PETScVector
except ImportError:
    PETScVector = None

# Systems: R > C1, C2, C3, C4
# Variables: v1, v2, v3, v4; all depend on each other


class Comp1(ExplicitComponent):

    def setup(self):
        self.add_input('v2')
        self.add_input('v3')
        self.add_input('v4')
        self.add_output('v1', var_set=1)


class Comp2(ExplicitComponent):

    def setup(self):
        self.add_input('v1')
        self.add_input('v3')
        self.add_input('v4')
        self.add_output('v2', var_set=2)


class Comp3(ExplicitComponent):

    def setup(self):
        self.add_input('v1')
        self.add_input('v2')
        self.add_input('v4')
        self.add_output('v3', var_set=3)


class Comp4(ExplicitComponent):

    def setup(self):
        self.add_input('v1')
        self.add_input('v2')
        self.add_input('v3')
        self.add_output('v4', var_set=4)


class GroupG(ParallelGroup):

    def __init__(self):
        super(GroupG, self).__init__()
        self.add_subsystem('C1', Comp1(), promotes=['*'])
        self.add_subsystem('C2', Comp2(), promotes=['*'])
        self.add_subsystem('C3', Comp3(), promotes=['*'])
        self.add_subsystem('C4', Comp4(), promotes=['*'])


class TestNumpyVec(unittest.TestCase):

    def setUp(self):
        group = GroupG()
        self.p = Problem(group).setup(check=False)
        self.p.final_setup()

    def test_prom_names(self):
        root = self.p.model
        names = sorted(root._var_allprocs_prom2abs_list['output'])
        self.assertEqual(names, ['v1', 'v2', 'v3', 'v4'])

    def test__var_set_IDs(self):
        set_IDs = self.p.model._var_set2iset['output']
        self.assertEqual(set_IDs[1], 0)
        self.assertEqual(set_IDs[2], 1)
        self.assertEqual(set_IDs[3], 2)
        self.assertEqual(set_IDs[4], 3)

    def test_transfer(self):
        root = self.p.model

        if root.comm.size == 1:
            comp1 = root.C1
            comp2 = root.C2
            comp3 = root.C3
            comp4 = root.C4

            comp1._outputs['v1'] = 2.0
            comp2._outputs['v2'] = 4.0
            comp3._outputs['v3'] = 6.0
            comp4._outputs['v4'] = 8.0

            assert_rel_error(self, comp1._outputs['v1'], 2.0)
            assert_rel_error(self, comp1._inputs['v2'], 1.0)
            assert_rel_error(self, comp1._inputs['v3'], 1.0)
            assert_rel_error(self, comp1._inputs['v4'], 1.0)

            assert_rel_error(self, comp2._inputs['v1'], 1.0)
            assert_rel_error(self, comp2._outputs['v2'], 4.0)
            assert_rel_error(self, comp2._inputs['v3'], 1.0)
            assert_rel_error(self, comp2._inputs['v4'], 1.0)

            root._transfer('nonlinear', 'fwd', 0)

            assert_rel_error(self, comp1._outputs['v1'], 2.0)
            assert_rel_error(self, comp1._inputs['v2'], 4.0)
            assert_rel_error(self, comp1._inputs['v3'], 6.0)
            assert_rel_error(self, comp1._inputs['v4'], 8.0)

            assert_rel_error(self, comp2._inputs['v1'], 1.0)
            assert_rel_error(self, comp2._outputs['v2'], 4.0)
            assert_rel_error(self, comp2._inputs['v3'], 1.0)
            assert_rel_error(self, comp2._inputs['v4'], 1.0)

            root._transfer('nonlinear', 'fwd', None)

            assert_rel_error(self, comp1._outputs['v1'], 2.0)
            assert_rel_error(self, comp1._inputs['v2'], 4.0)
            assert_rel_error(self, comp1._inputs['v3'], 6.0)
            assert_rel_error(self, comp1._inputs['v4'], 8.0)

            assert_rel_error(self, comp2._inputs['v1'], 2.0)
            assert_rel_error(self, comp2._outputs['v2'], 4.0)
            assert_rel_error(self, comp2._inputs['v3'], 6.0)
            assert_rel_error(self, comp2._inputs['v4'], 8.0)


@unittest.skipUnless(PETScVector, "PETSc is required.")
class TestPETScVec(TestNumpyVec):

    def setUp(self):
        group = GroupG()
        self.p = Problem(group).setup(PETScVector, check=False)
        self.p.final_setup()


if __name__ == '__main__':
    unittest.main()
