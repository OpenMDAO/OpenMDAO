"""Test the parallel groups."""

from __future__ import division, print_function

import unittest

from openmdao.api import Problem, Group, ParallelGroup, ExecComp, IndepVarComp

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from openmdao.test_suite.groups.parallel_groups import \
    FanOutGrouped, FanInGrouped, Diamond, ConvergeDiverge, \
    FanOutGroupedVarSets

from openmdao.devtools.testutil import assert_rel_error



@unittest.skipUnless(PETScVector, "PETSc is required.")
class TestParallelGroups(unittest.TestCase):

    N_PROCS = 2

    def test_fan_out_grouped(self):
        prob = Problem(FanOutGrouped())

        of=['c2.y', "c3.y"]
        wrt=['iv.x']

        prob.setup(vector_class=PETScVector, check=False, mode='fwd')
        prob.model.suppress_solver_output = True
        prob.run_model()

        J = prob.compute_total_derivs(of=['c2.y', "c3.y"], wrt=['iv.x'])

        assert_rel_error(self, J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

        assert_rel_error(self, prob['c2.y'], -6.0, 1e-6)
        assert_rel_error(self, prob['c3.y'], 15.0, 1e-6)

        prob.setup(vector_class=PETScVector, check=False, mode='rev')
        prob.run_model()

        J = prob.compute_total_derivs(of=['c2.y', "c3.y"], wrt=['iv.x'])

        assert_rel_error(self, J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

        assert_rel_error(self, prob['c2.y'], -6.0, 1e-6)
        assert_rel_error(self, prob['c3.y'], 15.0, 1e-6)

    #def test_fan_out_grouped_varsets(self):
        #prob = Problem(FanOutGroupedVarSets())

        #prob.setup(vector_class=PETScVector, check=False, mode='fwd')
        #prob.model.suppress_solver_output = True
        #prob.run_model()

        #J = prob.compute_total_derivs(of=['c2.y', "c3.y"], wrt=['iv.x'])

        #assert_rel_error(self, J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        #assert_rel_error(self, J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

        #assert_rel_error(self, prob['c2.y'], -6.0, 1e-6)
        #assert_rel_error(self, prob['c3.y'], 15.0, 1e-6)

        #prob.setup(vector_class=PETScVector, check=False, mode='rev')
        #prob.run_model()

        #J = prob.compute_total_derivs(of=['c2.y', "c3.y"], wrt=['iv.x'])

        #assert_rel_error(self, J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        #assert_rel_error(self, J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

        #assert_rel_error(self, prob['c2.y'], -6.0, 1e-6)
        #assert_rel_error(self, prob['c3.y'], 15.0, 1e-6)

    def test_fan_in_grouped(self):

        prob = Problem()
        prob.model = FanInGrouped()
        prob.setup(vector_class=PETScVector, check=False, mode='fwd')
        prob.model.suppress_solver_output = True
        prob.run_model()

        indep_list = ['iv.x1', 'iv.x2']
        unknown_list = ['c3.y']

        assert_rel_error(self, prob['c3.y'], 29.0, 1e-6)

        J = prob.compute_total_derivs(of=unknown_list, wrt=indep_list)
        assert_rel_error(self, J['c3.y', 'iv.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x2'][0][0], 35.0, 1e-6)

        assert_rel_error(self, prob['c3.y'], 29.0, 1e-6)

        prob.setup(vector_class=PETScVector, check=False, mode='rev')
        prob.run_model()

        assert_rel_error(self, prob['c3.y'], 29.0, 1e-6)

        J = prob.compute_total_derivs(of=unknown_list, wrt=indep_list)
        assert_rel_error(self, J['c3.y', 'iv.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x2'][0][0], 35.0, 1e-6)

        assert_rel_error(self, prob['c3.y'], 29.0, 1e-6)

    def test_diamond(self):

        prob = Problem()
        prob.model = Diamond()
        prob.setup(vector_class=PETScVector, check=False, mode='fwd')
        prob.model.suppress_solver_output = True
        prob.run_model()

        assert_rel_error(self, prob['c4.y1'], 46.0, 1e-6)
        assert_rel_error(self, prob['c4.y2'], -93.0, 1e-6)

        indep_list = ['iv.x']
        unknown_list = ['c4.y1', 'c4.y2']

        J = prob.compute_total_derivs(of=unknown_list, wrt=indep_list)
        assert_rel_error(self, J['c4.y1', 'iv.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['c4.y2', 'iv.x'][0][0], -40.5, 1e-6)

        prob.setup(vector_class=PETScVector, check=False, mode='rev')
        prob.run_model()

        assert_rel_error(self, prob['c4.y1'], 46.0, 1e-6)
        assert_rel_error(self, prob['c4.y2'], -93.0, 1e-6)

        J = prob.compute_total_derivs(of=unknown_list, wrt=indep_list)
        assert_rel_error(self, J['c4.y1', 'iv.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['c4.y2', 'iv.x'][0][0], -40.5, 1e-6)

    def test_converge_diverge(self):

        prob = Problem()
        prob.model = ConvergeDiverge()
        prob.setup(vector_class=PETScVector, check=False, mode='fwd')
        prob.model.suppress_solver_output = True
        prob.run_model()

        assert_rel_error(self, prob['c7.y1'], -102.7, 1e-6)

        indep_list = ['iv.x']
        unknown_list = ['c7.y1']

        J = prob.compute_total_derivs(of=unknown_list, wrt=indep_list)
        assert_rel_error(self, J['c7.y1', 'iv.x'][0][0], -40.75, 1e-6)

        prob.setup(vector_class=PETScVector, check=False, mode='rev')
        prob.run_model()

        assert_rel_error(self, prob['c7.y1'], -102.7, 1e-6)

        J = prob.compute_total_derivs(of=unknown_list, wrt=indep_list)
        assert_rel_error(self, J['c7.y1', 'iv.x'][0][0], -40.75, 1e-6)

        assert_rel_error(self, prob['c7.y1'], -102.7, 1e-6)

if __name__ == "__main__":
    from openmdao.utils.mpi import mpirun_tests
    mpirun_tests()
