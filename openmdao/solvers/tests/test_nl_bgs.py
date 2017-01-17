""" Unit test for the Nonlinear Block Gauss Seidel solver. """

import sys
import unittest

from six.moves import cStringIO

from openmdao.api import Problem, NonlinearBlockGS, Group, ScipyIterativeSolver
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDerivatives, \
     SellarDis1withDerivatives, SellarDis2withDerivatives


class TestNLBGaussSeidel(unittest.TestCase):

    def test_sellar(self):
        # Basic sellar test.

        prob = Problem()
        prob.root = SellarDerivatives()
        prob.root.nl_solver = NonlinearBlockGS()

        prob.setup(check=False)
        prob.root.suppress_solver_output = True
        prob.run()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        # Make sure we aren't iterating like crazy
        self.assertLess(prob.root.nl_solver._iter_count, 8)

        # Make sure we only call apply_linear on 'heads'
        #nd1 = prob.root.cycle.d1.execution_count
        #nd2 = prob.root.cycle.d2.execution_count
        #if prob.root.cycle.d1._run_apply == True:
            #self.assertEqual(nd1, 2*nd2)
        #else:
            #self.assertEqual(2*nd1, nd2)

    def test_sellar_analysis_error(self):
        # Tests Sellar behavior when AnalysisError is raised.

        raise unittest.SkipTest("AnalysisError not implemented yet")

        prob = Problem()
        prob.root = SellarDerivatives()
        prob.root.nl_solver = NonlinearBlockGS()
        prob.root.nl_solver.options['maxiter'] = 2
        prob.root.nl_solver.options['err_on_maxiter'] = True

        prob.setup(check=False)
        prob.root.suppress_solver_output = True

        try:
            prob.run()
        except AnalysisError as err:
            self.assertEqual(str(err), "Solve in '': NLGaussSeidel FAILED to converge after 2 iterations")
        else:
            self.fail("expected AnalysisError")

    def test_sellar_group_nested(self):
        # This tests true nested gs. Subsolvers solve each Sellar system. Top
        # solver couples them together through variable x.

        # This version has the indepvarcomps removed so we can connect them together.
        class SellarModified(Group):
            """ Group containing the Sellar MDA. This version uses the disciplines
            with derivatives."""

            def __init__(self):
                super(SellarModified, self).__init__()

                self.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
                self.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

                self.nl_solver = NonlinearBlockGS()
                self.ln_solver = ScipyIterativeSolver()

        prob = Problem()
        root = prob.root = Group()
        root.nl_solver = NonlinearBlockGS()
        root.nl_solver.options['maxiter'] = 20
        root.add_subsystem('g1', SellarModified())
        root.add_subsystem('g2', SellarModified())

        root.connect('g1.y2', 'g2.x')
        root.connect('g2.y2', 'g1.x')

        prob.setup(check=False)
        prob.root.suppress_solver_output = True

        prob.run()

        assert_rel_error(self, prob['g1.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g1.y2'], 0.80, .00001)
        assert_rel_error(self, prob['g2.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g2.y2'], 0.80, .00001)

    def test_run_apply(self):
        # This test makes sure that we correctly apply the "run_apply" flag
        # to all targets in the "broken" connection, even when they are
        # nested in Groups.
        # Note, this is a rather implementation-specific bug. It is not
        # certain that a new implementation will need this test.

        raise unittest.SkipTest("Test specific to implementation of double-run prevention.")

        prob = Problem()
        root = prob.root = Group()

        sub1 = root.add_subsystem('sub1', Group())
        sub2 = root.add_subsystem('sub2', Group())

        s1p1 = sub1.add_subsystem('p1', Paraboloid())
        s1p2 = sub1.add_subsystem('p2', Paraboloid())
        s2p1 = sub2.add_subsystem('p1', Paraboloid())
        s2p2 = sub2.add_subsystem('p2', Paraboloid())

        root.connect('sub1.p1.f_xy', 'sub2.p1.x')
        root.connect('sub1.p2.f_xy', 'sub2.p1.y')
        root.connect('sub1.p1.f_xy', 'sub2.p2.x')
        root.connect('sub1.p2.f_xy', 'sub2.p2.y')
        root.connect('sub2.p1.f_xy', 'sub1.p1.x')
        root.connect('sub2.p2.f_xy', 'sub1.p1.y')
        root.connect('sub2.p1.f_xy', 'sub1.p2.x')
        root.connect('sub2.p2.f_xy', 'sub1.p2.y')

        root.nl_solver = NonlinearBlockGS()
        root.ln_solver = ScipyIterativeSolver()

        prob.setup(check=False)
        prob.root.suppress_solver_output = True

        # Will be True in one group and False in the other, depending on
        # where it cuts.
        self.assertTrue(s1p1._run_apply != s2p1._run_apply)
        self.assertTrue(s1p2._run_apply != s2p2._run_apply)


if __name__ == "__main__":
    unittest.main()