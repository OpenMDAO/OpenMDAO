"""Test the ScipyIterativeSolver linear solver class."""

from __future__ import division, print_function
from six import iteritems

import unittest

import numpy as np

from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.devtools.testutil import assert_rel_error
from openmdao.solvers.ln_scipy import ScipyIterativeSolver, gmres
from openmdao.test_suite.groups.implicit_group import TestImplicitGroup
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped
from openmdao.test_suite.groups.parallel_groups import FanIn, FanInGrouped, \
     FanOut, FanOutGrouped, ConvergeDiverge, ConvergeDivergeFlat, \
     ConvergeDivergeGroups, Diamond, DiamondFlat

class TestScipyIterativeSolver(unittest.TestCase):

    def test_options(self):
        """Verify that the SciPy solver specific options are declared."""

        group = Group()
        group.ln_solver = ScipyIterativeSolver()

        assert(group.ln_solver.options['solver'] == gmres)

    def test_solve_linear_scipy(self):
        """Solve implicit system with ScipyIterativeSolver."""

        group = TestImplicitGroup(lnSolverClass=ScipyIterativeSolver)

        p = Problem(group)
        p.setup(check=False)
        p.model.suppress_solver_output = True

        # forward
        group._vectors['residual']['linear'].set_const(1.0)
        group._vectors['output']['linear'].set_const(0.0)
        group._solve_linear(['linear'], 'fwd')
        output = group._vectors['output']['linear']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 1e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 1e-15)

        # reverse
        group._vectors['output']['linear'].set_const(1.0)
        group._vectors['residual']['linear'].set_const(0.0)
        group._solve_linear(['linear'], 'rev')
        output = group._vectors['residual']['linear']._data
        assert_rel_error(self, output[0], group.expected_solution[0], 1e-15)
        assert_rel_error(self, output[1], group.expected_solution[1], 1e-15)

    def test_solve_linear_scipy_maxiter(self):
        """Verify that ScipyIterativeSolver abides by the 'maxiter' option."""

        group = TestImplicitGroup(lnSolverClass=ScipyIterativeSolver)
        group.ln_solver.options['maxiter'] = 2

        p = Problem(group)
        p.setup(check=False)
        p.model.suppress_solver_output = True

        # forward
        group._vectors['residual']['linear'].set_const(1.0)
        group._vectors['output']['linear'].set_const(0.0)
        group._solve_linear(['linear'], 'fwd')

        self.assertTrue(group.ln_solver._iter_count == 2)

        # reverse
        group._vectors['output']['linear'].set_const(1.0)
        group._vectors['residual']['linear'].set_const(0.0)
        group._solve_linear(['linear'], 'rev')

        self.assertTrue(group.ln_solver._iter_count == 2)

    def test_simple_matvec(self):
        group = Group()
        group.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])
        group.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        prob = Problem()
        prob.root = group
        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_simple_matvec_subbed(self):
        group = Group()
        group.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        prob = Problem()
        prob.root = Group()
        prob.root.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])
        prob.root.add('sub', group, promotes=['*'])

        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='fd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_simple_matvec_subbed_like_multipoint(self):
        group = Group()
        group.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        prob = Problem()
        prob.root = Group()
        prob.root.add('sub', group, promotes=['*'])
        prob.root.sub.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])

        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='fd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='fd', return_format='array')
        assert_rel_error(self, J[0][0], 2.0, 1e-6)

    def test_array2D(self):
        group = Group()
        group.add('x_param', IndepVarComp('x', np.ones((2, 2))), promotes=['*'])
        group.add('mycomp', ArrayComp2D(), promotes=['x', 'y'])

        prob = Problem()
        prob.root = group
        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        Jbase = prob.root.mycomp._jacobian_cache
        diff = np.linalg.norm(J['y']['x'] - Jbase['y', 'x'])
        assert_rel_error(self, diff, 0.0, 1e-8)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        diff = np.linalg.norm(J['y']['x'] - Jbase['y', 'x'])
        assert_rel_error(self, diff, 0.0, 1e-8)

    def test_double_arraycomp(self):
        # Mainly testing a bug in the array return for multiple arrays

        group = Group()
        group.add('x_param1', IndepVarComp('x1', np.ones((2))), promotes=['*'])
        group.add('x_param2', IndepVarComp('x2', np.ones((2))), promotes=['*'])
        group.add('mycomp', DoubleArrayComp(), promotes=['*'])

        prob = Problem()
        prob.root = group
        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        Jbase = group.mycomp.JJ

        J = prob.calc_gradient(['x1', 'x2'], ['y1', 'y2'], mode='fwd',
                               return_format='array')
        diff = np.linalg.norm(J - Jbase)
        assert_rel_error(self, diff, 0.0, 1e-8)

        J = prob.calc_gradient(['x1', 'x2'], ['y1', 'y2'], mode='fd',
                               return_format='array')
        diff = np.linalg.norm(J - Jbase)
        assert_rel_error(self, diff, 0.0, 1e-8)

        J = prob.calc_gradient(['x1', 'x2'], ['y1', 'y2'], mode='rev',
                               return_format='array')
        diff = np.linalg.norm(J - Jbase)
        assert_rel_error(self, diff, 0.0, 1e-8)

    def test_simple_in_group_matvec(self):
        group = Group()
        sub = group.add('sub', Group(), promotes=['x', 'y'])
        group.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])
        sub.add('mycomp', SimpleCompDerivMatVec(), promotes=['x', 'y'])

        prob = Problem()
        prob.root = group
        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_simple_jac(self):
        group = Group()
        group.add('x_param', IndepVarComp('x', 1.0), promotes=['*'])
        group.add('mycomp', ExecComp(['y=2.0*x']), promotes=['x', 'y'])

        prob = Problem()
        prob.root = group
        prob.root.ln_solver = ScipyGMRES()
        prob.setup(check=False)
        prob.run()

        J = prob.calc_gradient(['x'], ['y'], mode='fwd', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

        J = prob.calc_gradient(['x'], ['y'], mode='rev', return_format='dict')
        assert_rel_error(self, J['y']['x'][0][0], 2.0, 1e-6)

    def test_fan_out(self):
        # Test derivatives for fan-out topology.
        prob = Problem()
        prob.model = FanOut()
        prob.model.ln_solver = ScipyIterativeSolver()
        prob.model.suppress_solver_output = True

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        wrt = ['p.x']
        of = ['comp2.y', "comp3.y"]

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['comp2.y', 'p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y', 'p.x'][0][0], 15.0, 1e-6)

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['comp2.y', 'p.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y', 'p.x'][0][0], 15.0, 1e-6)

    def test_fan_out_grouped(self):
        # Test derivatives for fan-out-grouped topology.
        prob = Problem()
        prob.model = FanOutGrouped()
        prob.model.ln_solver = ScipyIterativeSolver()
        prob.model.suppress_solver_output = True

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        wrt = ['iv.x']
        of = ['sub.c2.y', "sub.c3.y"]

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['sub.c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['sub.c3.y', 'iv.x'][0][0], 15.0, 1e-6)

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['sub.c2.y', 'iv.x'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['sub.c3.y', 'iv.x'][0][0], 15.0, 1e-6)

    def test_fan_in(self):
        # Test derivatives for fan-in topology.
        prob = Problem()
        prob.model = FanIn()
        prob.model.ln_solver = ScipyIterativeSolver()
        prob.model.suppress_solver_output = True

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        wrt = ['p1.x1', 'p2.x2']
        of = ['comp3.y']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['comp3.y', 'p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y', 'p2.x2'][0][0], 35.0, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['comp3.y', 'p1.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['comp3.y', 'p2.x2'][0][0], 35.0, 1e-6)

    def test_fan_in_grouped(self):
        # Test derivatives for fan-in-grouped topology.
        prob = Problem()
        prob.model = FanInGrouped()
        prob.model.ln_solver = ScipyIterativeSolver()
        prob.model.suppress_solver_output = True

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        wrt = ['iv.x1', 'iv.x2']
        of = ['c3.y']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['c3.y', 'iv.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x2'][0][0], 35.0, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['c3.y', 'iv.x1'][0][0], -6.0, 1e-6)
        assert_rel_error(self, J['c3.y', 'iv.x2'][0][0], 35.0, 1e-6)

    def test_converge_diverge_flat(self):
        # Test derivatives for converge-diverge-flat topology.
        prob = Problem()
        prob.model = ConvergeDivergeFlat()
        prob.model.ln_solver = ScipyIterativeSolver()
        prob.model.suppress_solver_output = True

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        wrt = ['iv.x']
        of = ['c7.y1']

        # Make sure value is fine.
        assert_rel_error(self, prob['c7.y1'], -102.7, 1e-6)

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['c7.y1', 'iv.x'][0][0], -40.75, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['c7.y1', 'iv.x'][0][0], -40.75, 1e-6)

    def test_analysis_error(self):

        raise unittest.SkipTest("AnalysisError not implemented yet")

        prob = Problem()
        prob.model = ConvergeDivergeFlat()
        prob.model.ln_solver = ScipyIterativeSolver()
        prob.model.ln_solver.options['maxiter'] = 2
        prob.model.ln_solver.options['err_on_maxiter'] = True

        prob.setup(check=False)
        prob.run()

        wrt = ['iv.x']
        of = ['c7.y1']

        prob.run()

        # Make sure value is fine.
        assert_rel_error(self, prob['c7.y1'], -102.7, 1e-6)

        try:
            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        except AnalysisError as err:
            self.assertEqual(str(err), "Solve in '': ScipyGMRES failed to converge after 2 iterations")
        else:
            self.fail("expected AnalysisError")

    def test_converge_diverge_groups(self):
        # Test derivatives for converge-diverge-groups topology.
        raise unittest.SkipTest("Bug: data not being passed beyond first component.")

        prob = Problem()
        prob.model = ConvergeDivergeGroups()
        prob.model.ln_solver = ScipyIterativeSolver()
        prob.model.suppress_solver_output = True

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        wrt = ['iv.x']
        of = ['c7.y1']

        # Make sure value is fine.
        assert_rel_error(self, prob['c7.y1'], -102.7, 1e-6)

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['c7.y1', 'iv.x'][0][0], -40.75, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['c7.y1', 'iv.x'][0][0], -40.75, 1e-6)

    def test_single_diamond(self):
        # Test derivatives for flat diamond topology.
        prob = Problem()
        prob.model = DiamondFlat()
        prob.model.ln_solver = ScipyIterativeSolver()
        prob.model.suppress_solver_output = True

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        wrt = ['iv.x']
        of = ['c4.y1', 'c4.y2']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['c4.y1', 'iv.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['c4.y2', 'iv.x'][0][0], -40.5, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['c4.y1', 'iv.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['c4.y2', 'iv.x'][0][0], -40.5, 1e-6)

    def test_single_diamond_grouped(self):
        # Test derivatives for grouped diamond topology.

        prob = Problem()
        prob.model = Diamond()
        prob.model.ln_solver = ScipyIterativeSolver()
        prob.model.suppress_solver_output = True

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        wrt = ['iv.x']
        of = ['c4.y1', 'c4.y2']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['c4.y1', 'iv.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['c4.y2', 'iv.x'][0][0], -40.5, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['c4.y1', 'iv.x'][0][0], 25, 1e-6)
        assert_rel_error(self, J['c4.y2', 'iv.x'][0][0], -40.5, 1e-6)

    def test_sellar_derivs_grouped(self):
        # Test derivatives across a converged Sellar model.

        prob = Problem()
        prob.model = SellarDerivativesGrouped()
        prob.model.suppress_solver_output = True

        mda = prob.model.get_subsystem('mda')
        mda.nl_solver.options['atol'] = 1e-12

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        # Just make sure we are at the right answer
        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

        wrt = ['x', 'z']
        of = ['obj', 'con1', 'con2']

        Jbase = {}
        Jbase['con1', 'x'] = -0.98061433
        Jbase['con1', 'z'] = np.array([-9.61002285, -0.78449158])
        Jbase['con2', 'x'] = 0.09692762
        Jbase['con2', 'z'] = np.array([1.94989079, 1.0775421 ])
        Jbase['obj', 'x'] = 2.98061392
        Jbase['obj', 'z'] = np.array([9.61001155, 1.78448534])

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        for key, val in iteritems(Jbase):
            assert_rel_error(self, J[key], val, .00001)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        for key, val in iteritems(Jbase):
            assert_rel_error(self, J[key], val, .00001)

if __name__ == "__main__":
    unittest.main()
