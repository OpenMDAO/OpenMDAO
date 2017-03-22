"""Test the parallel groups."""

from __future__ import division, print_function

import unittest

from openmdao.api import Problem, Group, ParallelGroup, ExecComp, IndepVarComp

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from openmdao.test_suite.groups.parallel_groups import \
    FanOutGrouped, FanInGrouped, Diamond, ConvergeDiverge

from openmdao.devtools.testutil import assert_rel_error


@unittest.skipUnless(PETScVector, "PETSc is required.")
class TestParallelGroups(unittest.TestCase):

    N_PROCS = 2

    def test_fan_out_grouped(self):
        prob = Problem(FanOutGrouped())

        prob.setup(vector_class=PETScVector, check=False, mode='fwd')
        prob.model.suppress_solver_output = True
        prob.run_model()

        J = prob.compute_total_derivs(of=['c2.y', "c3.y"], wrt=['iv.x'])

        print('fwd', J)
        assert_rel_error(self, J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

        prob.setup(vector_class=PETScVector, check=False, mode='rev')

        J = prob.compute_total_derivs(of=['c2.y', "c3.y"], wrt=['iv.x'])
        print('rev', J)

        assert_rel_error(self, J['c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x'][0][0], 15.0, 1e-6)

        assert_rel_error(self, prob['c2.y'], -6.0, 1e-6)
        assert_rel_error(self, prob['c3.y'], 15.0, 1e-6)

    def test_fan_in_grouped(self):

        prob = Problem()
        prob.model = FanInGrouped()
        prob.setup(vector_class=PETScVector, check=False)
        prob.model.suppress_solver_output = True
        prob.run_model()

        assert_rel_error(self, prob['c3.y'], 29.0, 1e-6)

    def test_diamond(self):

        prob = Problem()
        prob.model = Diamond()
        prob.setup(vector_class=PETScVector, check=False)
        prob.model.suppress_solver_output = True
        prob.run_model()

        assert_rel_error(self, prob['c4.y1'], 46.0, 1e-6)
        assert_rel_error(self, prob['c4.y2'], -93.0, 1e-6)

    def test_converge_diverge(self):

        prob = Problem()
        prob.model = ConvergeDiverge()
        prob.setup(vector_class=PETScVector, check=False)
        prob.model.suppress_solver_output = True
        prob.run_model()

        assert_rel_error(self, prob['c7.y1'], -102.7, 1e-6)


if __name__ == "__main__":
    #unittest.main()

    prob = Problem()
    model = prob.model

    model.add_subsystem('iv', IndepVarComp('x', 1.0))
    model.add_subsystem('c1', ExecComp(['y=3.0*x', 'y2=4.0*xx']))

    sub = model.add_subsystem('sub', ParallelGroup())
    sub.add_subsystem('c2', ExecComp(['y=-2.0*x']))
    sub.add_subsystem('c3', ExecComp(['y=5.0*x']))

    model.add_subsystem('c2', ExecComp(['y=x']))
    model.add_subsystem('c3', ExecComp(['y=x']))

    model.connect('iv.x', 'c1.x')

    model.connect('c1.y', 'sub.c2.x')
    model.connect('c1.y', 'sub.c3.x')

    model.connect('sub.c2.y', 'c2.x')
    model.connect('sub.c3.y', 'c3.x')

    prob.setup(vector_class=PETScVector, check=False, mode='fwd')
    prob.model.suppress_solver_output = True
    prob.run_model()

    import wingdbstub
    J = prob.compute_total_derivs(of=['c2.y', "c3.y"], wrt=['iv.x'])

    print('fwd', J)

    prob.setup(vector_class=PETScVector, check=False, mode='rev')
    prob.run_model()

    J = prob.compute_total_derivs(of=['c2.y', "c3.y"], wrt=['iv.x'])
    print('rev', J)