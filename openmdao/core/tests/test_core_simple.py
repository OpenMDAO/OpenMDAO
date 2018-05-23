from __future__ import division
import numpy as np
import unittest

from openmdao.api import Problem, IndepVarComp, ExplicitComponent, Group, DefaultVector, PETScVector
from openmdao.utils.assert_utils import assert_rel_error


#      (A) -> x
# x -> (B) -> f


class CompA(ExplicitComponent):

    def setup(self):
        self.add_output('x')


class CompB(ExplicitComponent):

    def setup(self):
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
        self.p = Problem(group).setup()
        self.p.set_solver_print(level=0)
        self.p.final_setup()

    def test_subsystems(self):
        root = self.p.model

        self.assertEqual(len(root._subsystems_allprocs), 2)
        self.assertEqual(len(root._subsystems_myproc), 2)

    def test_prom_names(self):
        root = self.p.model
        self.assertEqual(list(root.A._var_allprocs_prom2abs_list['output'].keys()), ['x'])

    def test_var_indices(self):
        def get_inds(p, sname, type_):
            system = p.model._get_subsystem(sname) if sname else p.model
            idxs = p.model._var_allprocs_abs2idx['linear']
            return np.array([
                idxs[name] for name in system._var_abs_names[type_]
            ])

        assert_rel_error(self, get_inds(self.p, '', 'input'), np.array([0]))
        assert_rel_error(self, get_inds(self.p, '', 'output'), np.array([0,1]))

        assert_rel_error(self, get_inds(self.p, 'A', 'input'), np.array([]))
        assert_rel_error(self, get_inds(self.p, 'A', 'output'), np.array([0]))

        assert_rel_error(self, get_inds(self.p, 'B', 'input'), np.array([0]))
        assert_rel_error(self, get_inds(self.p, 'B', 'output'), np.array([1]))

    def test_var_allprocs_idx_range(self):
        rng = self.p.model._subsystems_var_range

        assert_rel_error(self, rng['nonlinear']['input']['A'], np.array([0,0]))
        assert_rel_error(self, rng['nonlinear']['input']['B'], np.array([0,1]))

        assert_rel_error(self, rng['nonlinear']['output']['A'], np.array([0,1]))
        assert_rel_error(self, rng['nonlinear']['output']['B'], np.array([1,2]))

    def test_GS(self):
        root = self.p.model

        if root.comm.size == 1:
            compA = root.A
            compB = root.B

            if root.comm.rank == 0:
                assert_rel_error(self, root._outputs['A.x'], 0)
                assert_rel_error(self, compA._outputs['x'], 0)
                compA._outputs['x'] = 10
            if root.comm.rank == 2:
                assert_rel_error(self, compB._inputs['x'], 0)
                assert_rel_error(self, compB._outputs['f'], 0)

            root.run_solve_nonlinear()

            if root.comm.rank == 0:
                assert_rel_error(self, root._outputs['A.x'], 10)
                assert_rel_error(self, compA._outputs['x'], 10)
            if root.comm.rank == 2:
                assert_rel_error(self, compB._inputs['x'], 10)
                assert_rel_error(self, compB._outputs['f'], 20)


@unittest.skipUnless(PETScVector, "PETSc is required.")
class TestPETScVec(Test):

    def setUp(self):
        group = GroupG()
        group.add_subsystems()
        self.p = Problem(group).setup(local_vector_class=PETScVector)
        self.p.set_solver_print(level=0)
        self.p.final_setup()


if __name__ == '__main__':
    unittest.main()
