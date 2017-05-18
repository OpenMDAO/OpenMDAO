"""Common tests for linear solvers."""

from __future__ import division, print_function
from six import iteritems

import unittest

import numpy as np

from openmdao.api import Group, IndepVarComp, Problem, AssembledJacobian, ImplicitComponent
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimpleJacVec
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped, \
     SellarStateConnection, SellarDerivatives, SellarImplicitDis1, SellarImplicitDis2

from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimpleDense
from openmdao.test_suite.components.simple_comps import DoubleArrayComp
from openmdao.test_suite.groups.implicit_group import TestImplicitGroup
from openmdao.test_suite.groups.parallel_groups import FanIn, FanInGrouped, \
     FanOut, FanOutGrouped, ConvergeDivergeFlat, \
     ConvergeDivergeGroups, Diamond, DiamondFlat


class LinearSolverTests(object):
    class LinearSolverTestCase(unittest.TestCase):
        ln_solver_class = None


        def test_solve_linear_maxiter(self):
            """Verify that the linear solver abides by the 'maxiter' option."""

            group = TestImplicitGroup(lnSolverClass=self.ln_solver_class)
            group.ln_solver.options['maxiter'] = 2

            p = Problem(group)
            p.setup(check=False)
            p.set_solver_print(level=0)

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

            model.ln_solver = self.ln_solver_class()
            prob.set_solver_print(level=0)

            prob.setup(check=False, mode='fwd')
            prob['width'] = 2.0
            prob.run_model()

            of = ['area']
            wrt = ['length']

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['area', 'length'], [[2.0]], 1e-6)

            prob.setup(check=False, mode='rev')
            prob['width'] = 2.0
            prob.run_model()

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['area', 'length'], [[2.0]], 1e-6)

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

            model.ln_solver = self.ln_solver_class()
            prob.set_solver_print(level=0)

            prob.setup(check=False, mode='fwd')
            prob['width'] = 2.0
            prob.run_model()

            of = ['area']
            wrt = ['length']

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['area', 'length'], [[2.0]], 1e-6)

            prob.setup(check=False, mode='rev')
            prob['width'] = 2.0
            prob.run_model()

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['area', 'length'], [[2.0]], 1e-6)

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

            model.ln_solver = self.ln_solver_class()
            prob.set_solver_print(level=0)

            prob.setup(check=False, mode='fwd')
            prob['width'] = 2.0
            prob.run_model()

            of = ['area']
            wrt = ['length']

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['area', 'length'], [[2.0]], 1e-6)

            prob.setup(check=False, mode='rev')
            prob['width'] = 2.0
            prob.run_model()

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['area', 'length'], [[2.0]], 1e-6)

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
            model.ln_solver = self.ln_solver_class()
            prob.set_solver_print(level=0)

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
            prob.model.ln_solver = self.ln_solver_class()
            prob.set_solver_print(level=0)

            prob.setup(check=False, mode='fwd')
            prob.run_model()

            wrt = ['p.x']
            of = ['comp2.y', "comp3.y"]

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['comp2.y', 'p.x'], [[-6.0]], 1e-6)
            assert_rel_error(self, J['comp3.y', 'p.x'], [[15.0]], 1e-6)

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['comp2.y', 'p.x'], [[-6.0]], 1e-6)
            assert_rel_error(self, J['comp3.y', 'p.x'], [[15.0]], 1e-6)

        def test_fan_out_grouped(self):
            # Test derivatives for fan-out-grouped topology.
            prob = Problem()
            prob.model = FanOutGrouped()
            prob.model.ln_solver = self.ln_solver_class()
            prob.set_solver_print(level=0)

            prob.setup(check=False, mode='fwd')
            prob.run_model()

            wrt = ['iv.x']
            of = ['sub.c2.y', "sub.c3.y"]

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['sub.c2.y', 'iv.x'], [[-6.0]], 1e-6)
            assert_rel_error(self, J['sub.c3.y', 'iv.x'], [[15.0]], 1e-6)

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['sub.c2.y', 'iv.x'], [[-6.0]], 1e-6)
            assert_rel_error(self, J['sub.c3.y', 'iv.x'], [[15.0]], 1e-6)

        def test_fan_in(self):
            # Test derivatives for fan-in topology.
            prob = Problem()
            prob.model = FanIn()
            prob.model.ln_solver = self.ln_solver_class()
            prob.set_solver_print(level=0)

            prob.setup(check=False, mode='fwd')
            prob.run_model()

            wrt = ['p1.x1', 'p2.x2']
            of = ['comp3.y']

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['comp3.y', 'p1.x1'], [[-6.0]], 1e-6)
            assert_rel_error(self, J['comp3.y', 'p2.x2'], [[35.0]], 1e-6)

            prob.setup(check=False, mode='rev')
            prob.run_model()

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['comp3.y', 'p1.x1'], [[-6.0]], 1e-6)
            assert_rel_error(self, J['comp3.y', 'p2.x2'], [[35.0]], 1e-6)

        def test_fan_in_grouped(self):
            # Test derivatives for fan-in-grouped topology.
            prob = Problem()
            prob.model = FanInGrouped()
            prob.model.ln_solver = self.ln_solver_class()
            prob.set_solver_print(level=0)

            prob.setup(check=False, mode='fwd')
            prob.run_model()

            wrt = ['iv.x1', 'iv.x2']
            of = ['c3.y']

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['c3.y', 'iv.x1'], [[-6.0]], 1e-6)
            assert_rel_error(self, J['c3.y', 'iv.x2'], [[35.0]], 1e-6)

            prob.setup(check=False, mode='rev')
            prob.run_model()

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['c3.y', 'iv.x1'], [[-6.0]], 1e-6)
            assert_rel_error(self, J['c3.y', 'iv.x2'], [[35.0]], 1e-6)

        def test_converge_diverge_flat(self):
            # Test derivatives for converge-diverge-flat topology.
            prob = Problem()
            prob.model = ConvergeDivergeFlat()
            prob.model.ln_solver = self.ln_solver_class()
            prob.set_solver_print(level=0)

            prob.setup(check=False, mode='fwd')
            prob.run_model()

            wrt = ['iv.x']
            of = ['c7.y1']

            # Make sure value is fine.
            assert_rel_error(self, prob['c7.y1'], -102.7, 1e-6)

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['c7.y1', 'iv.x'], [[-40.75]], 1e-6)

            prob.setup(check=False, mode='rev')
            prob.run_model()

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['c7.y1', 'iv.x'], [[-40.75]], 1e-6)

        def test_analysis_error(self):

            raise unittest.SkipTest("AnalysisError not implemented yet")

            prob = Problem()
            prob.model = ConvergeDivergeFlat()
            prob.model.ln_solver = self.ln_solver_class()
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
            prob.model.ln_solver = self.ln_solver_class()
            prob.set_solver_print(level=0)

            prob.setup(check=False, mode='fwd')
            prob.run_model()

            wrt = ['iv.x']
            of = ['c7.y1']

            # Make sure value is fine.
            assert_rel_error(self, prob['c7.y1'], -102.7, 1e-6)

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['c7.y1', 'iv.x'], [[-40.75]], 1e-6)

            prob.setup(check=False, mode='rev')
            prob.run_model()

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['c7.y1', 'iv.x'], [[-40.75]], 1e-6)

        def test_single_diamond(self):
            # Test derivatives for flat diamond topology.
            prob = Problem()
            prob.model = DiamondFlat()
            prob.model.ln_solver = self.ln_solver_class()
            prob.set_solver_print(level=0)

            prob.setup(check=False, mode='fwd')
            prob.run_model()

            wrt = ['iv.x']
            of = ['c4.y1', 'c4.y2']

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['c4.y1', 'iv.x'], [[25]], 1e-6)
            assert_rel_error(self, J['c4.y2', 'iv.x'], [[-40.5]], 1e-6)

            prob.setup(check=False, mode='rev')
            prob.run_model()

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['c4.y1', 'iv.x'], [[25]], 1e-6)
            assert_rel_error(self, J['c4.y2', 'iv.x'], [[-40.5]], 1e-6)

        def test_single_diamond_grouped(self):
            # Test derivatives for grouped diamond topology.

            prob = Problem()
            prob.model = Diamond()
            prob.model.ln_solver = self.ln_solver_class()
            prob.set_solver_print(level=0)

            prob.setup(check=False, mode='fwd')
            prob.run_model()

            wrt = ['iv.x']
            of = ['c4.y1', 'c4.y2']

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['c4.y1', 'iv.x'], [[25]], 1e-6)
            assert_rel_error(self, J['c4.y2', 'iv.x'], [[-40.5]], 1e-6)

            prob.setup(check=False, mode='rev')
            prob.run_model()

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            assert_rel_error(self, J['c4.y1', 'iv.x'], [[25]], 1e-6)
            assert_rel_error(self, J['c4.y2', 'iv.x'], [[-40.5]], 1e-6)

        def test_sellar_derivs_grouped(self):
            # Test derivatives across a converged Sellar model.

            prob = Problem()
            prob.model = SellarDerivativesGrouped()
            prob.model.ln_solver = self.ln_solver_class()
            prob.set_solver_print(level=0)

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
            Jbase['con1', 'x'] = [[-0.98061433]]
            Jbase['con1', 'z'] = np.array([[-9.61002285, -0.78449158]])
            Jbase['con2', 'x'] = [[0.09692762]]
            Jbase['con2', 'z'] = np.array([[1.94989079, 1.0775421]])
            Jbase['obj', 'x'] = [[2.98061392]]
            Jbase['obj', 'z'] = np.array([[9.61001155, 1.78448534]])

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            for key, val in iteritems(Jbase):
                assert_rel_error(self, J[key], val, .00001)

            prob.setup(check=False, mode='rev')
            prob.run_model()

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            for key, val in iteritems(Jbase):
                assert_rel_error(self, J[key], val, .00001)

        def test_sellar_state_connection(self):
            # Test derivatives across a converged Sellar model.

            prob = Problem()
            prob.model = SellarStateConnection()
            prob.model.ln_solver = self.ln_solver_class()
            prob.set_solver_print(level=0)

            prob.model.nl_solver.options['atol'] = 1e-12

            prob.setup(check=False, mode='fwd')
            prob.run_model()

            # Just make sure we are at the right answer
            assert_rel_error(self, prob['y1'], 25.58830273, .00001)
            assert_rel_error(self, prob['d2.y2'], 12.05848819, .00001)

            wrt = ['x', 'z']
            of = ['obj', 'con1', 'con2']

            Jbase = {}
            Jbase['con1', 'x'] = [[-0.98061433]]
            Jbase['con1', 'z'] = np.array([[-9.61002285, -0.78449158]])
            Jbase['con2', 'x'] = [[0.09692762]]
            Jbase['con2', 'z'] = np.array([[1.94989079, 1.0775421]])
            Jbase['obj', 'x'] = [[2.98061392]]
            Jbase['obj', 'z'] = np.array([[9.61001155, 1.78448534]])

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            for key, val in iteritems(Jbase):
                assert_rel_error(self, J[key], val, .00001)

            prob.setup(check=False, mode='rev')
            prob.run_model()

            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
            for key, val in iteritems(Jbase):
                assert_rel_error(self, J[key], val, .00001)
