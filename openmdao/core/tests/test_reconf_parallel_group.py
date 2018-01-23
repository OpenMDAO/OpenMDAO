from __future__ import division
import numpy as np
import unittest

from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent, DefaultVector, ExecComp
from openmdao.api import NewtonSolver, PETScKrylov, NonlinearBlockGS, LinearBlockGS
from openmdao.utils.assert_utils import assert_rel_error

try:
    from openmdao.parallel_api import PETScVector
except ImportError:
    PETScVector = None


class ReconfGroup(Group):

    def __init__(self):
        super(ReconfGroup, self).__init__()

        self.parallel = True

    def setup(self):
        self._mpi_proc_allocator.parallel = self.parallel
        if self.parallel:
            self.nonlinear_solver = NewtonSolver()
            self.linear_solver = PETScKrylov()
        else:
            self.nonlinear_solver = NonlinearBlockGS()
            self.linear_solver = LinearBlockGS()

        self.add_subsystem('C1', ExecComp('z = 1 / 3. * y + x0'), promotes=['x0'])
        self.add_subsystem('C2', ExecComp('z = 1 / 4. * y + x1'), promotes=['x1'])

        if self.parallel:
            self.connect('C1.z', 'C2.y')
            self.connect('C2.z', 'C1.y')
        else:
            self.connect('C1.z', 'C2.y', src_indices=[self.comm.rank])
            self.connect('C2.z', 'C1.y', src_indices=[self.comm.rank])

        self.parallel = not self.parallel


@unittest.skipUnless(PETScVector, "PETSc is required.")
class Test(unittest.TestCase):

    N_PROCS = 2

    def test(self):
        prob = Problem(model=Group())
        prob.model.add_subsystem('Cx0', IndepVarComp('x0'), promotes=['x0'])
        prob.model.add_subsystem('Cx1', IndepVarComp('x1'), promotes=['x1'])
        prob.model.add_subsystem('g', ReconfGroup(), promotes=['*'])
        prob.setup(vector_class=PETScVector, check=False)

        # First, run with full setup, so ReconfGroup should be a parallel group
        prob['x0'] = 6.
        prob['x1'] = 4.
        prob.run_model()
        if prob.comm.rank == 0:
            assert_rel_error(self, prob['C1.z'], 8.0)
            print(prob['C1.z'])
        elif prob.comm.rank == 1:
            assert_rel_error(self, prob['C2.z'], 6.0)
            print(prob['C2.z'])

        # Now, reconfigure so ReconfGroup is not parallel, and x0, x1 should be preserved
        prob.model.g.resetup('reconf')
        prob.model.resetup('update')
        prob.run_model()
        assert_rel_error(self, prob['C1.z'], 8.0, 1e-8)
        assert_rel_error(self, prob['C2.z'], 6.0, 1e-8)
        print(prob['C1.z'], prob['C2.z'])


if __name__ == '__main__':
    unittest.main()
