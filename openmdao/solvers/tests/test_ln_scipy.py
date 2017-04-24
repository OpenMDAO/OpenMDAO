"""Test the ScipyIterativeSolver linear solver class."""

from __future__ import division, print_function
from six import iteritems

import unittest

import numpy as np

from openmdao.core.group import Group
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.problem import Problem
from openmdao.devtools.testutil import assert_rel_error
from openmdao.solvers.ln_bgs import LinearBlockGS
from openmdao.solvers.ln_direct import DirectSolver
from openmdao.solvers.ln_scipy import ScipyIterativeSolver, gmres
from openmdao.solvers.nl_newton import NewtonSolver
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimpleJacVec, \
     TestExplCompSimpleDense
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped, \
     SellarDerivatives
from openmdao.test_suite.components.simple_comps import DoubleArrayComp
from openmdao.test_suite.groups.implicit_group import TestImplicitGroup
from openmdao.test_suite.groups.parallel_groups import FanIn, FanInGrouped, \
     FanOut, FanOutGrouped, ConvergeDivergeFlat, \
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

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        group.run_solve_linear(['linear'], 'fwd')
        output = d_outputs._data
        assert_rel_error(self, output[1], group.expected_solution[0], 1e-15)
        assert_rel_error(self, output[5], group.expected_solution[1], 1e-15)

        # reverse
        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        group.run_solve_linear(['linear'], 'rev')
        output = d_residuals._data
        assert_rel_error(self, output[1], group.expected_solution[0], 1e-15)
        assert_rel_error(self, output[5], group.expected_solution[1], 1e-15)

    def test_solve_linear_scipy_maxiter(self):
        """Verify that ScipyIterativeSolver abides by the 'maxiter' option."""

        group = TestImplicitGroup(lnSolverClass=ScipyIterativeSolver)
        group.ln_solver.options['maxiter'] = 2

        p = Problem(group)
        p.setup(check=False)
        p.model.suppress_solver_output = True

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        group.run_solve_linear(['linear'], 'fwd')

        self.assertTrue(group.ln_solver._iter_count == 2)

        # reverse
        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        group.run_solve_linear(['linear'], 'rev')

        self.assertTrue(group.ln_solver._iter_count == 2)

    def test_simple_matvec(self):
        # Tests derivatives on a simple comp that defines compute_jacvec.
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('x_param', IndepVarComp('length', 3.0),
                            promotes=['length'])
        model.add_subsystem('mycomp', TestExplCompSimpleJacVec(),
                            promotes=['length', 'width', 'area'])

        model.ln_solver = ScipyIterativeSolver()
        model.suppress_solver_output = True

        prob.setup(check=False, mode='fwd')
        prob['width'] = 2.0
        prob.run_model()

        of = ['area']
        wrt = ['length']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['area', 'length'][0][0], 2.0, 1e-6)

        prob.setup(check=False, mode='rev')
        prob['width'] = 2.0
        prob.run_model()

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['area', 'length'][0][0], 2.0, 1e-6)

    def test_simple_matvec_subbed(self):
        # Tests derivatives on a group that contains a simple comp that
        # defines compute_jacvec.
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('x_param', IndepVarComp('length', 3.0),
                            promotes=['length'])
        sub = model.add_subsystem('sub', Group(),
                                  promotes=['length', 'width', 'area'])
        sub.add_subsystem('mycomp', TestExplCompSimpleJacVec(),
                            promotes=['length', 'width', 'area'])

        model.ln_solver = ScipyIterativeSolver()
        model.suppress_solver_output = True

        prob.setup(check=False, mode='fwd')
        prob['width'] = 2.0
        prob.run_model()

        of = ['area']
        wrt = ['length']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['area', 'length'][0][0], 2.0, 1e-6)

        prob.setup(check=False, mode='rev')
        prob['width'] = 2.0
        prob.run_model()

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['area', 'length'][0][0], 2.0, 1e-6)

    def test_simple_matvec_subbed_like_multipoint(self):
        # Tests derivatives on a group that contains a simple comp that
        # defines compute_jacvec. For this one, the indepvarcomp is also
        # in the subsystem.
        prob = Problem()
        model = prob.model = Group()
        sub = model.add_subsystem('sub', Group(),
                                  promotes=['length', 'width', 'area'])
        sub.add_subsystem('x_param', IndepVarComp('length', 3.0),
                            promotes=['length'])
        sub.add_subsystem('mycomp', TestExplCompSimpleJacVec(),
                            promotes=['length', 'width', 'area'])

        model.ln_solver = ScipyIterativeSolver()
        model.suppress_solver_output = True

        prob.setup(check=False, mode='fwd')
        prob['width'] = 2.0
        prob.run_model()

        of = ['area']
        wrt = ['length']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['area', 'length'][0][0], 2.0, 1e-6)

        prob.setup(check=False, mode='rev')
        prob['width'] = 2.0
        prob.run_model()

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['area', 'length'][0][0], 2.0, 1e-6)

    def test_double_arraycomp(self):
        # Mainly testing an old bug in the array return for multiple arrays
        group = Group()
        group.add_subsystem('x_param1', IndepVarComp('x1', np.ones((2))),
                            promotes=['x1'])
        group.add_subsystem('x_param2', IndepVarComp('x2', np.ones((2))),
                            promotes=['x2'])
        group.add_subsystem('mycomp', DoubleArrayComp(),
                            promotes=['x1', 'x2', 'y1', 'y2'])

        prob = Problem()
        model = prob.model = group
        model.ln_solver = ScipyIterativeSolver()
        model.suppress_solver_output = True

        prob.setup(check=False, mode='fwd')
        prob.run_model()

        Jbase = group.get_subsystem('mycomp').JJ
        of = ['y1', 'y2']
        wrt = ['x1', 'x2']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        diff = np.linalg.norm(J['y1', 'x1'] - Jbase[0:2, 0:2])
        assert_rel_error(self, diff, 0.0, 1e-8)
        diff = np.linalg.norm(J['y1', 'x2'] - Jbase[0:2, 2:4])
        assert_rel_error(self, diff, 0.0, 1e-8)
        diff = np.linalg.norm(J['y2', 'x1'] - Jbase[2:4, 0:2])
        assert_rel_error(self, diff, 0.0, 1e-8)
        diff = np.linalg.norm(J['y2', 'x2'] - Jbase[2:4, 2:4])
        assert_rel_error(self, diff, 0.0, 1e-8)

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
            prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        except AnalysisError as err:
            self.assertEqual(str(err), "Solve in '': ScipyGMRES failed to converge after 2 iterations")
        else:
            self.fail("expected AnalysisError")

    def test_converge_diverge_groups(self):
        # Test derivatives for converge-diverge-groups topology.
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

    def test_solve_linear_precon(self):
        """Solve implicit system with PetscKSP using a preconditioner."""

        group = TestImplicitGroup(lnSolverClass=ScipyIterativeSolver)
        precon = group.ln_solver.precon = LinearBlockGS()
        precon.options['maxiter'] = 1

        p = Problem(group)
        p.setup(check=False)
        p.model.suppress_solver_output = True

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        group.run_solve_linear(['linear'], 'fwd')

        output = d_outputs._data
        assert_rel_error(self, output[1], group.expected_solution[0], 1e-15)
        assert_rel_error(self, output[5], group.expected_solution[1], 1e-15)

        self.assertTrue(precon._iter_count > 0)

        # reverse
        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        group.run_solve_linear(['linear'], 'rev')

        output = d_residuals._data
        assert_rel_error(self, output[1], group.expected_solution[0], 3e-15)
        assert_rel_error(self, output[5], group.expected_solution[1], 3e-15)

        self.assertTrue(precon._iter_count > 0)

        # test direct solver precon and make sure the _linearize recurses correctly
        precon = group.ln_solver.precon = DirectSolver()
        p.setup(check=False)

        d_inputs, d_outputs, d_residuals = group.get_linear_vectors()

        # forward
        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        group.ln_solver._linearize()
        group.run_solve_linear(['linear'], 'fwd')

        output = d_outputs._data
        assert_rel_error(self, output[1], group.expected_solution[0], 1e-15)
        assert_rel_error(self, output[5], group.expected_solution[1], 1e-15)

        # reverse
        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        group.ln_solver._linearize()
        group.run_solve_linear(['linear'], 'rev')

        output = d_residuals._data
        assert_rel_error(self, output[1], group.expected_solution[0], 3e-15)
        assert_rel_error(self, output[5], group.expected_solution[1], 3e-15)

    def test_solve_on_subsystem(self):
        """solve an implicit system with GMRES attached to a subsystem"""

        p = Problem()
        model = p.model = Group()
        dv = model.add_subsystem('des_vars', IndepVarComp())
        # just need a dummy variable so the sizes don't match between root and g1
        dv.add_output('dummy', val=1.0, shape=10)

        g1 = model.add_subsystem('g1', TestImplicitGroup(lnSolverClass=ScipyIterativeSolver))

        p.model.ln_solver.options['maxiter'] = 1
        p.setup(check=False)

        p.model.suppress_solver_output = True

        # forward
        d_inputs, d_outputs, d_residuals = g1.get_linear_vectors()

        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        g1._solve_linear(['linear'], 'fwd')

        output = d_outputs._data
        # The empty first entry in _data is due to the dummy
        #     variable being in a different variable set not owned by g1
        assert_rel_error(self, output[1], g1.expected_solution[0], 1e-15)
        assert_rel_error(self, output[5], g1.expected_solution[1], 1e-15)

        # reverse
        d_inputs, d_outputs, d_residuals = g1.get_linear_vectors()

        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        g1.ln_solver._linearize()
        g1._solve_linear(['linear'], 'rev')

        output = d_residuals._data
        assert_rel_error(self, output[1], g1.expected_solution[0], 3e-15)
        assert_rel_error(self, output[5], g1.expected_solution[1], 3e-15)


class TestScipyIterativeSolverFeature(unittest.TestCase):

    def test_feature_simple(self):
        """Tests feature for adding a Scipy GMRES solver and calculating the
        derivatives."""
        # Tests derivatives on a simple comp that defines compute_jacvec.
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('x_param', IndepVarComp('length', 3.0),
                            promotes=['length'])
        model.add_subsystem('mycomp', TestExplCompSimpleDense(),
                            promotes=['length', 'width', 'area'])

        model.ln_solver = ScipyIterativeSolver()
        model.suppress_solver_output = True

        prob.setup(check=False, mode='fwd')
        prob['width'] = 2.0
        prob.run_model()

        of = ['area']
        wrt = ['length']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['area', 'length'][0][0], 2.0, 1e-6)

    def test_specify_solver(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = ScipyIterativeSolver()

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)

    def test_feature_maxiter(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = ScipyIterativeSolver()
        model.ln_solver.options['maxiter'] = 3

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 0.0, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 0.0, .00001)

    def test_feature_atol(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = ScipyIterativeSolver()
        model.ln_solver.options['atol'] = 1.0e-20

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001055699, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448533563, .00001)

    def test_specify_precon(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nl_solver = NewtonSolver()
        prob.model.ln_sollver = ScipyIterativeSolver()

        prob.model.ln_solver.precon = LinearBlockGS()
        prob.model.ln_solver.precon.options['maxiter'] = 2

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

if __name__ == "__main__":
    unittest.main()
