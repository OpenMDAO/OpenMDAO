""" Unit test for the Nonlinear Block Gauss Seidel solver. """

import sys
import unittest

from six.moves import cStringIO

from openmdao.api import Problem, NonlinearBlockGS, Group, ScipyIterativeSolver
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDerivatives, \
     SellarDerivativesGrouped


class TestNLBGaussSeidel(unittest.TestCase):

    def test_sellar(self):

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

    def test_sellar_group_nesting_nlbgs(self):

        prob = Problem()
        prob.root = SellarDerivativesGrouped()
        prob.root.nl_solver = NonlinearBlockGS()
        mda = prob.root.get_system('mda')
        mda.nl_solver = NonlinearBlockGS()

        # So, inner solver converges loosely.
        mda.nl_solver.options['atol'] = 1e-3

        # And outer solver tightens it up.
        prob.root.nl_solver.options['atol'] = 1e-9

        prob.setup(check=False)
        prob.root.suppress_solver_output = False

        prob.run()
        #old_stdout = sys.stdout
        #sys.stdout = cStringIO() # so we don't see the iprint output during testing
        #try:
            #prob.run()
        #finally:
            #sys.stdout = old_stdout

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

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