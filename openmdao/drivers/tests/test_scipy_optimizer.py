""" Unit tests for the ScipyOptimizeDriver."""

import unittest
import sys
import warnings

from six import StringIO

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ScipyOptimizeDriver, ScipyOptimizer, ExplicitComponent
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.test_suite.components.expl_comp_array import TestExplCompArrayDense
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped
from openmdao.test_suite.components.simple_comps import NonSquareArrayComp
from openmdao.test_suite.groups.sin_fitter import SineFitter


class TestScipyOptimizeDriver(unittest.TestCase):

    def test_scipyoptimizer_deprecation(self):

        msg = "'ScipyOptimizer' provides backwards compatibility " \
            + "with OpenMDAO <= 2.2 ; use 'ScipyOptimizeDriver' instead."

        # check deprecation on getter
        with warnings.catch_warnings(record=True) as w:
            driver = ScipyOptimizer()

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message), msg)

    def test_compute_totals_basic_return_array(self):
        # Make sure 'array' return_format works.

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed.")

        of = ['comp.f_xy']
        wrt = ['p1.x', 'p2.y']
        derivs = prob._compute_totals(of=of, wrt=wrt, return_format='array')

        assert_rel_error(self, derivs[0, 0], -6.0, 1e-6)
        assert_rel_error(self, derivs[0, 1], 8.0, 1e-6)

        prob.setup(check=False, mode='rev')

        failed, rel_err, abs_err = prob.run_model()

        self.assertFalse(failed, "Optimization failed.")

        of = ['comp.f_xy']
        wrt = ['p1.x', 'p2.y']
        derivs = prob._compute_totals(of=of, wrt=wrt, return_format='array')

        assert_rel_error(self, derivs[0, 0], -6.0, 1e-6)
        assert_rel_error(self, derivs[0, 1], 8.0, 1e-6)

    def test_compute_totals_return_array_non_square(self):

        prob = Problem()
        prob.model = model = Group()

        model.add_subsystem('px', IndepVarComp(name="x", val=np.ones((2, ))))
        comp = model.add_subsystem('comp', NonSquareArrayComp())
        model.connect('px.x', 'comp.x1')

        model.add_design_var('px.x')
        model.add_objective('px.x')
        model.add_constraint('comp.y1')
        model.add_constraint('comp.y2')

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed.")

        derivs = prob._compute_totals(of=['comp.y1'], wrt=['px.x'], return_format='array')

        J = comp.JJ[0:3, 0:2]
        assert_rel_error(self, J, derivs, 1.0e-3)

        # Support for a name to be in 'of' and 'wrt'

        derivs = prob._compute_totals(of=['comp.y2', 'px.x', 'comp.y1'],
                                      wrt=['px.x'],
                                      return_format='array')

        assert_rel_error(self, J, derivs[3:, :], 1.0e-3)
        assert_rel_error(self, comp.JJ[3:4, 0:2], derivs[0:1, :], 1.0e-3)
        assert_rel_error(self, np.eye(2), derivs[1:3, :], 1.0e-3)

    def test_deriv_wrt_self(self):

        prob = Problem()
        model = prob.model

        model.add_subsystem('px', IndepVarComp(name="x", val=np.ones((2, ))))

        model.add_design_var('px.x')
        model.add_objective('px.x')

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed.")

        # Support for a name to be in 'of' and 'wrt'

        J = prob.driver._compute_totals(of=['px.x'], wrt=['px.x'],
                                        return_format='array')

        assert_rel_error(self, J, np.eye(2), 1.0e-3)

    def test_scipy_optimizer_simple_paraboloid_unconstrained(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_rel_error(self, prob['x'], 6.66666667, 1e-6)
        assert_rel_error(self, prob['y'], -7.3333333, 1e-6)

    def test_simple_paraboloid_unconstrained(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_rel_error(self, prob['x'], 6.66666667, 1e-6)
        assert_rel_error(self, prob['y'], -7.3333333, 1e-6)

    def test_simple_paraboloid_unconstrained_COBYLA(self):
        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'COBYLA'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_rel_error(self, prob['x'], 6.66666667, 1e-6)
        assert_rel_error(self, prob['y'], -7.3333333, 1e-6)

    def test_simple_paraboloid_upper(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_lower(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)

        model.add_objective('f_xy')
        model.add_constraint('c', lower=15.0)

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_equality(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=-15.0)

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        # Minimum should be at (7.166667, -7.833334)
        # (Note, loose tol because of appveyor py3.4 machine.)
        assert_rel_error(self, prob['x'], 7.16667, 1e-4)
        assert_rel_error(self, prob['y'], -7.833334, 1e-4)

    def test_unsupported_equality(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'COBYLA'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=-15.0)

        prob.setup(check=False)

        with self.assertRaises(Exception) as raises_cm:
            prob.run_driver()

        exception = raises_cm.exception

        msg = "Constraints of type 'eq' not handled by COBYLA."

        self.assertEqual(exception.args[0], msg)

    def test_simple_paraboloid_double_sided_low(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=-11.0, upper=-10.0)

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_rel_error(self, prob['y'] - prob['x'], -11.0, 1e-6)

    def test_simple_paraboloid_double_sided_high(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_rel_error(self, prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_array_comp2D(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = areas - 20.0', c=np.zeros((2, 2)), areas=np.zeros((2, 2))),
                            promotes=['*'])
        model.add_subsystem('obj', ExecComp('o = areas[0, 0]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('widths', lower=-50.0, upper=50.0)
        model.add_objective('o')
        model.add_constraint('c', equals=0.0)

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        obj = prob['o']
        assert_rel_error(self, obj, 20.0, 1e-6)

    def test_simple_array_comp2D_eq_con(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('obj', ExecComp('o = areas[0, 0] + areas[1, 1]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('widths', lower=-50.0, upper=50.0)
        model.add_objective('o')
        model.add_constraint('areas', equals=np.array([24.0, 21.0, 3.5, 17.5]))

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        obj = prob['o']
        assert_rel_error(self, obj, 41.5, 1e-6)

    def test_simple_array_comp2D_dbl_sided_con(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('obj', ExecComp('o = areas[0, 0]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('widths', lower=-50.0, upper=50.0)
        model.add_objective('o')
        model.add_constraint('areas', lower=np.array([24.0, 21.0, 3.5, 17.5]), upper=np.array([24.0, 21.0, 3.5, 17.5]))

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        con = prob['areas']
        assert_rel_error(self, con, np.array([[24.0, 21.0], [3.5, 17.5]]), 1e-6)

    def test_simple_array_comp2D_dbl_sided_con_array(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('obj', ExecComp('o = areas[0, 0]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('widths', lower=-50.0, upper=50.0)
        model.add_objective('o')
        model.add_constraint('areas', lower=20.0, upper=20.0)

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        obj = prob['o']
        assert_rel_error(self, obj, 20.0, 1e-6)

    def test_simple_array_comp2D_array_lo_hi(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = areas - 20.0', c=np.zeros((2, 2)), areas=np.zeros((2, 2))),
                            promotes=['*'])
        model.add_subsystem('obj', ExecComp('o = areas[0, 0]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('widths', lower=-50.0*np.ones((2, 2)), upper=50.0*np.ones((2, 2)))
        model.add_objective('o')
        model.add_constraint('c', equals=0.0)

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        obj = prob['o']
        assert_rel_error(self, obj, 20.0, 1e-6)

    def test_simple_paraboloid_scaled_desvars_fwd(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0, ref=.02)
        model.add_design_var('y', lower=-50.0, upper=50.0, ref=.02)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup(check=False, mode='fwd')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_rel_error(self, prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_desvars_rev(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0, ref=.02)
        model.add_design_var('y', lower=-50.0, upper=50.0, ref=.02)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup(check=False, mode='rev')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_rel_error(self, prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_constraint_fwd(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0, ref=10.)

        prob.setup(check=False, mode='fwd')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_rel_error(self, prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_objective_fwd(self):

        prob = Problem()
        model = prob.model = Group()

        prob.set_solver_print(level=0)

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy', ref=10.)
        model.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup(check=False, mode='fwd')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_rel_error(self, prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_objective_rev(self):

        prob = Problem()
        model = prob.model = Group()

        prob.set_solver_print(level=0)

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy', ref=10.)
        model.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup(check=False, mode='rev')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_rel_error(self, prob['x'] - prob['y'], 11.0, 1e-6)

    def test_sellar_mdf(self):

        prob = Problem()
        model = prob.model = SellarDerivativesGrouped()

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.setup(check=False, mode='rev')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_rel_error(self, prob['z'][0], 1.9776, 1e-3)
        assert_rel_error(self, prob['z'][1], 0.0, 1e-3)
        assert_rel_error(self, prob['x'], 0.0, 1e-3)

    def test_bug_in_eq_constraints(self):
        # We were getting extra constraints created because lower and upper are maxfloat instead of
        # None when unused.
        p = Problem(model=SineFitter())
        p.driver = ScipyOptimizeDriver()

        p.setup(check=False)
        p.run_driver()

        max_defect = np.max(np.abs(p['defect.defect']))
        assert_rel_error(self, max_defect, 0.0, 1e-10)

    def test_reraise_exception_from_callbacks(self):
        class ReducedActuatorDisc(ExplicitComponent):

            def setup(self):

                # Inputs
                self.add_input('a', 0.5, desc="Induced Velocity Factor")
                self.add_input('Vu', 10.0, units="m/s", desc="Freestream air velocity, upstream of rotor")

                # Outputs
                self.add_output('Vd', 0.0, units="m/s",
                                desc="Slipstream air velocity, downstream of rotor")

            def compute(self, inputs, outputs):
                a = inputs['a']
                Vu = inputs['Vu']

                outputs['Vd'] = Vu * (1 - 2 * a)

            def compute_partials(self, inputs, J):
                Vu = inputs['Vu']

                J['Vd', 'a'] = -2.0 * Vu

        prob = Problem()
        indeps = prob.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        indeps.add_output('a', .5)
        indeps.add_output('Vu', 10.0, units='m/s')

        prob.model.add_subsystem('a_disk', ReducedActuatorDisc(),
                                 promotes_inputs=['a', 'Vu'])

        # setup the optimization
        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('a', lower=0., upper=1.)
        # negative one so we maximize the objective
        prob.model.add_objective('a_disk.Vd', scaler=-1)

        prob.setup()

        with self.assertRaises(KeyError) as context:
            prob.run_driver()

        msg = 'Variable name pair ("Vd", "a") must first be declared.'
        self.assertTrue(msg in str(context.exception))

    def test_simple_paraboloid_upper_COBYLA(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'COBYLA'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_sellar_mdf_COBYLA(self):

        prob = Problem()
        model = prob.model = SellarDerivativesGrouped()

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'COBYLA'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.setup(check=False, mode='rev')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_rel_error(self, prob['z'][0], 1.9776, 1e-3)
        assert_rel_error(self, prob['z'][1], 0.0, 1e-3)
        assert_rel_error(self, prob['x'], 0.0, 1e-3)

    def test_simple_paraboloid_lower_linear(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=15.0, linear=True)

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

        self.assertEqual(prob.driver._obj_and_nlcons, ['comp.f_xy'])

    def test_simple_paraboloid_equality_linear(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=-15.0, linear=True)

        prob.setup(check=False)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_debug_print_option(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        prob.driver.options['debug_print'] = ['desvars','ln_cons','nl_cons','objs']

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup(check=False)

        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            prob.run_driver()
        finally:
            sys.stdout = stdout

        output = strout.getvalue().split('\n')
        self.assertTrue(output.count("Design Vars") > 1,
                        "Should be more than one design vars header printed")
        self.assertTrue(output.count("Nonlinear constraints") > 1,
                        "Should be more than one nonlinear constraint header printed")
        self.assertTrue(output.count("Linear constraints") > 1,
                        "Should be more than one linear constraint header printed")
        self.assertTrue(output.count("Objectives") > 1,
                        "Should be more than one objective header printed")
        self.assertTrue(len([s for s in output if s.startswith('p1.x')]) > 1,
                        "Should be more than one p1.x printed")
        self.assertTrue(len([s for s in output if s.startswith('p2.y')]) > 1,
                        "Should be more than one p2.y printed")
        self.assertTrue(len([s for s in output if s.startswith('con.c')]) > 1,
                        "Should be more than one con.c printed")
        self.assertTrue(len([s for s in output if s.startswith('comp.f_xy')]) > 1,
                        "Should be more than one comp.f_xy printed")


class TestScipyOptimizeDriverFeatures(unittest.TestCase):

    def test_feature_basic(self):
        from openmdao.api import Problem, Group, IndepVarComp, ScipyOptimizeDriver
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup()

        prob.run_driver()

        assert_rel_error(self, prob['x'], 6.66666667, 1e-6)
        assert_rel_error(self, prob['y'], -7.3333333, 1e-6)

    def test_feature_optimizer(self):
        from openmdao.api import Problem, Group, IndepVarComp, ScipyOptimizeDriver
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'COBYLA'
        prob.driver.options['disp'] = True

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup(check=False)

        prob.run_driver()

        assert_rel_error(self, prob['x'], 6.66666667, 1e-6)
        assert_rel_error(self, prob['y'], -7.3333333, 1e-6)

    def test_feature_maxiter(self):
        from openmdao.api import Problem, Group, IndepVarComp, ScipyOptimizeDriver
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['maxiter'] = 20

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup(check=False)

        prob.run_driver()

        assert_rel_error(self, prob['x'], 6.66666667, 1e-6)
        assert_rel_error(self, prob['y'], -7.3333333, 1e-6)

    def test_feature_tol(self):
        from openmdao.api import Problem, Group, IndepVarComp, ScipyOptimizeDriver
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['tol'] = 1.0e-9

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup(check=False)

        prob.run_driver()

        assert_rel_error(self, prob['x'], 6.66666667, 1e-6)
        assert_rel_error(self, prob['y'], -7.3333333, 1e-6)

    def test_debug_print_option(self):

        from openmdao.api import Problem, Group, IndepVarComp, ScipyOptimizeDriver, ExecComp
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        prob.driver.options['debug_print'] = ['desvars','ln_cons','nl_cons','objs']

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup(check=False)

        prob.run_driver()


if __name__ == "__main__":
    unittest.main()
