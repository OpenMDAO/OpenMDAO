""" Unit tests for the system methods for modifying bounds and scalers."""
import unittest
from copy import deepcopy

import numpy as np

from openmdao.api import Problem, ExecComp, ExplicitComponent
from openmdao.components.linear_system_comp import LinearSystemComp
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver
from openmdao.solvers.linear.direct import DirectSolver
from openmdao.solvers.linear.linear_block_gs import LinearBlockGS
from openmdao.solvers.nonlinear.broyden import BroydenSolver
from openmdao.solvers.nonlinear.nonlinear_block_gs import NonlinearBlockGS
from openmdao.test_suite.components.expl_comp_array import TestExplCompArraySparse
from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStates
from openmdao.test_suite.components.sellar import SellarDerivatives, SellarProblem
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.indexer import slicer


class TestSystemSetSolverDesvarConstraintObjectiveOutputOptions(unittest.TestCase):
    # this class is for tests that cut across all set_xxxxx_options methods
    def test_set_options_called_with_nonexistant_var(self):
        prob = Problem()
        prob.model.add_subsystem('comp', ImplCompTwoStatesNoMetadataSetInAddOutput())
        prob.model.set_output_solver_options(name='comp.bad_var', lower=1.5, upper=2.5)
        with self.assertRaises(RuntimeError) as ctx:
            prob.setup()
        self.assertEqual(str(ctx.exception),
                         "Output solver options set using System.set_output_solver_options for "
                         "non-existent variable 'comp.bad_var' in System ''.")

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_design_var('x')
        prob.model.add_design_var('y')
        prob.model.add_constraint('con1')
        prob.model.add_constraint('con2')
        prob.model.add_objective('obj')
        with self.assertRaises(RuntimeError) as ctx:
            prob.model.set_design_var_options(name='bad_var', lower=-100, upper=100)
        self.assertEqual(str(ctx.exception),
                         "<class SellarDerivatives>: set_design_var_options called with design "
                         "variable 'bad_var' "
                         "that does not exist.")

        with self.assertRaises(RuntimeError) as ctx:
            prob.model.set_constraint_options(name='bad_var', ref0=-100.0, ref=100)
        self.assertEqual(str(ctx.exception),
                         "<class SellarDerivatives>: set_constraint_options called with "
                         "constraint variable 'bad_var' that does not exist.")

        with self.assertRaises(RuntimeError) as ctx:
            prob.model.set_objective_options(name='bad_var', ref0=-100.0, ref=100)
        self.assertEqual(str(ctx.exception),
                         "<class SellarDerivatives>: set_objective_options called with "
                         "objective variable 'bad_var' that does not exist.")


class TestSystemSetSolverOutputOptions(unittest.TestCase):
    # For tests specifically for set_output_solver_options
    # Some tests run models that are from other tests in which the
    #   the metadata is set using add_output arguments. Checking to see if we get same results
    #   when setting the metadata when using set_output_solver_options vs. add_output
    def _setup_model_for_lower_upper_tests(self, comp_class):
        prob = Problem()
        prob.model.add_subsystem('comp', comp_class(),
                                 promotes=['*'])
        prob.model.nonlinear_solver = BroydenSolver()
        prob.model.nonlinear_solver.options['maxiter'] = 25
        prob.model.nonlinear_solver.options['diverge_limit'] = 0.5
        prob.model.nonlinear_solver.options['state_vars'] = ['y', 'z']
        prob.model.linear_solver = DirectSolver()
        return prob

    def test_set_output_solver_options_top_model(self):
        # No metadata set on comp.z in the model. Set it after the fact. See if the results of
        #   running the model are as expected
        prob = self._setup_model_for_lower_upper_tests(ImplCompTwoStatesNoMetadataSetInAddOutput)
        prob.model.set_output_solver_options(name='comp.z', lower=1.5, upper=2.5)
        prob.setup()

        comp_z_meta = prob.model._var_allprocs_abs2meta['output']['comp.z']
        self.assertEqual(comp_z_meta['lower'], 1.5)
        self.assertEqual(comp_z_meta['upper'], 2.5)

        # Test lower bound: should go to the lower bound and stall
        prob['comp.x'] = 2.0
        prob['comp.y'] = 0.0
        prob['comp.z'] = 1.6
        prob.run_model()
        assert_near_equal(prob['comp.z'], [1.5], 1e-8)

        # Test upper bound: should go to the upper bound and stall
        prob['comp.x'] = 0.5
        prob['comp.y'] = 0.0
        prob['comp.z'] = 2.4
        prob.run_model()
        assert_near_equal(prob['comp.z'], [2.5], 1e-8)

    def test_set_output_solver_options_sub_system(self):
        # Try setting the metadata using set_output_solver_options on a subsystem, not the top
        prob = self._setup_model_for_lower_upper_tests(ImplCompTwoStatesNoMetadataSetInAddOutput)
        comp = prob.model._get_subsystem('comp')
        comp.set_output_solver_options(name='z', lower=1.5, upper=2.5)
        prob.setup()

        comp_z_meta = prob.model._var_allprocs_abs2meta['output']['comp.z']
        self.assertEqual(comp_z_meta['lower'], 1.5)
        self.assertEqual(comp_z_meta['upper'], 2.5)

        # Test lower bound: should go to the lower bound and stall
        prob['comp.x'] = 2.0
        prob['comp.y'] = 0.0
        prob['comp.z'] = 1.6
        prob.run_model()
        assert_near_equal(prob['comp.z'], [1.5], 1e-8)

        # Test upper bound: should go to the upper bound and stall
        prob['comp.x'] = 0.5
        prob['comp.y'] = 0.0
        prob['comp.z'] = 2.4
        prob.run_model()
        assert_near_equal(prob['comp.z'], [2.5], 1e-8)

    def test_set_output_solver_options_set_lower_upper_to_none(self):
        # test the ability to override values set with add_output
        #  In this case, set the lower and upper to None, to see if the code handles that
        prob = self._setup_model_for_lower_upper_tests(ImplCompTwoStates)
        prob.model.set_output_solver_options(name='comp.z', lower=None, upper=None)
        prob.setup()

        comp_z_meta = prob.model._var_allprocs_abs2meta['output']['comp.z']
        self.assertEqual(comp_z_meta['lower'], None)
        self.assertEqual(comp_z_meta['upper'], None)

        prob.set_solver_print(level=0)
        # Test lower bound: should go to the lower bound and stall
        prob['comp.x'] = 2.0
        prob['comp.y'] = 0.0
        prob['comp.z'] = 1.6
        prob.run_model()
        assert_near_equal(prob['comp.z'], [1.3333333333333333], 1e-8)

    def test_set_output_solver_options_ref_no_meta(self):
        # Test setting scaling using set_output_solver_options
        prob = Problem()
        model = prob.model
        model.add_subsystem('comp', ScalingExample3NoMetadataSetInAddOutput(),
                            promotes_inputs=['x1', 'x2'])
        prob.model.set_input_defaults('x1', 1.0)
        prob.model.set_input_defaults('x2', 1.0)
        model.set_output_solver_options(name='comp.y1', ref=1e2, res_ref=1e5)
        model.set_output_solver_options(name='comp.y2', ref=1e3, res_ref=1e-5)
        prob.setup()

        # See if values were set in the metadata
        comp_y1_meta = model._var_allprocs_abs2meta['output']['comp.y1']
        self.assertEqual(comp_y1_meta['ref'], 1e2)
        self.assertEqual(comp_y1_meta['res_ref'], 1e5)

        prob.run_model()
        model.run_apply_nonlinear()

        # are the results as expected with those values of ref and res_ref
        with model._scaled_context_all():
            val = model.comp._residuals['y1']
            assert_near_equal(val, [-.995])
            val = model.comp._residuals['y2']
            assert_near_equal(val, [(1 - 6000.) / 6000.])

    def test_set_output_solver_affect_on_has_scaling_and_bounds_attributes(self):
        # make sure bounds and scaling attributes get set correctly with different
        #  combinations of setting via add_output and set_output_solver_options

        # ref and lower not set either in add_output of set_output_solver_options
        prob = Problem()
        comp = prob.model.add_subsystem('comp', ScalingExample3ForTesting())
        prob.setup()
        self.assertFalse(comp._has_bounds)
        self.assertFalse(comp._has_output_scaling)

        # ref and lower set with set_output_solver_options
        prob = Problem()
        comp = prob.model.add_subsystem('comp', ScalingExample3ForTesting())
        prob.model.set_output_solver_options(name='comp.y1', ref=1.2, lower=2)
        prob.setup()
        self.assertTrue(comp._has_bounds)
        self.assertTrue(comp._has_output_scaling)

        # ref and lower set with add_output on both variables
        #  Unset with set_output_solver_options on just y1
        prob = Problem()
        comp = prob.model.add_subsystem('comp', ScalingExample3ForTesting(ref1=1.2, lower1=2.0,
                                                                          ref2=1.2, lower2=2.0))
        prob.model.set_output_solver_options(name='comp.y1', ref=1.0, lower=None)
        prob.setup()
        self.assertTrue(comp._has_bounds)
        self.assertTrue(comp._has_output_scaling)

        # ref and lower set with add_output on both variables
        #  Unset with set_output_solver_options on both y1 and y2
        prob = Problem()
        comp = prob.model.add_subsystem('comp', ScalingExample3ForTesting(ref1=1.2, lower1=2.0,
                                                                          ref2=1.2, lower2=2.0))
        prob.model.set_output_solver_options(name='comp.y1', ref=1.0, lower=None)
        prob.model.set_output_solver_options(name='comp.y2', ref=1.0, lower=None)
        prob.setup()
        self.assertFalse(comp._has_bounds)
        self.assertFalse(comp._has_output_scaling)

    def test_set_output_solver_options_after_initial_run(self):
        # Testing to see if the error message occurs when the user runs a model
        #   and then calls set_output_solver_options, then tries to run model again
        #   without first running setup again
        prob = Problem()
        prob.model.add_subsystem('comp', ScalingExample3())
        prob.model.set_output_solver_options(name='comp.y1', ref=3)
        prob.setup()
        prob.run_model()
        prob.model.set_output_solver_options(name='comp.y1', ref=4)
        msg = "Problem .*: Before calling `run_model`, the `setup` method must be called if " \
              "set_output_solver_options has been called."
        with self.assertRaisesRegex(RuntimeError, msg) as cm:
            prob.run_model()

        # also make sure it doesn't raise an error if done correctly with an additional call
        #  to setup
        prob = Problem()
        prob.model.add_subsystem('comp', ScalingExample3())
        prob.model.set_output_solver_options(name='comp.y1', ref=3)
        prob.setup()
        prob.run_model()
        prob.model.set_output_solver_options(name='comp.y1', ref=4)
        prob.setup()
        prob.run_model()

    def test_set_output_solver_options_override_all_options(self):
        # Just another test to check setting all metadata in one call
        prob = Problem()
        model = prob.model
        # ScalingExample3 sets values for ref, res_ref, ref0, lower, and upper
        comp = model.add_subsystem('comp', ScalingExample3())

        # override all those values
        model.set_output_solver_options(name='comp.y1',
                                        ref=3, res_ref=6, ref0=9, lower=12, upper=15)
        prob.setup()

        # check that they were all set
        comp_y1_meta = model._var_allprocs_abs2meta['output']['comp.y1']
        self.assertEqual(comp_y1_meta['ref'], 3)
        self.assertEqual(comp_y1_meta['res_ref'], 6)
        self.assertEqual(comp_y1_meta['ref0'], 9)
        self.assertEqual(comp_y1_meta['lower'], 12)
        self.assertEqual(comp_y1_meta['upper'], 15)

    def test_set_output_solver_options_no_setup_after_call_to_set_output_solver_options(self):
        prob = SellarProblem(nonlinear_solver=NonlinearBlockGS,
                             linear_solver=LinearBlockGS)
        lscomp = prob.model.add_subsystem('lscomp', LinearSystemComp())
        lscomp.nonlinear_solver = NonlinearBlockGS(maxiter=5)
        prob.model.set_output_solver_options(name='lscomp.x', ref=3)
        prob.setup()
        prob.run_driver()

        prob.model.set_output_solver_options(name='lscomp.x', ref=5)
        msg = "Problem .*: Before calling `run_driver`, the `setup` method must be called if " \
              "set_output_solver_options has been called."
        with self.assertRaisesRegex(RuntimeError, msg) as cm:
            prob.run_driver()

    def test_set_output_solver_options_vector_values(self):
        class SimpleImpComp(ImplicitComponent):
            def setup(self):
                self.add_output('y', np.array([3, 1]), upper = np.array([10,10]))
                self.declare_partials(of='*', wrt='*')
        prob = Problem()
        prob.model.add_subsystem('px', SimpleImpComp())
        prob.setup()

        new_upper_solver_option = np.array([1,1])
        prob.model.set_output_solver_options(name='px.y', upper = new_upper_solver_option)
        prob.setup()
        output_meta = prob.model._var_allprocs_abs2meta['output']
        assert_near_equal(output_meta['px.y']['upper'], new_upper_solver_option)

class TestSystemSetDesignVarOptions(unittest.TestCase):
    def test_set_design_var_options_scaler_adder_ref_ref0(self):
        # See if setting options for des vars results in the same metadata whether
        #   they are set using add_design_var or set_design_var_options

        # first set the options ref and ref0 using add_design_var
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_design_var('x', lower=-100, upper=100, ref0=-100.0, ref=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.setup()
        des_vars_using_add_design_var = prob.model.get_design_vars()

        # then set the options using set_design_var_options
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_design_var('x')
        prob.model.add_design_var('z')
        prob.model.set_design_var_options(name='x', lower=-100, upper=100, ref0=-100.0, ref=100)
        prob.model.set_design_var_options(name='z', lower=-100, upper=100)
        prob.setup()
        des_vars_using_set_design_var_options = prob.model.get_design_vars()

        self.assertEqual(des_vars_using_add_design_var, des_vars_using_set_design_var_options)

        # Now do the same using scaler and adder
        # first set the options using add_design_var
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_design_var('x', lower=-100, upper=100, adder=3, scaler=10)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.setup()
        des_vars_using_add_design_var = prob.model.get_design_vars()

        # then set the options using set_design_var_options
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_design_var('x')
        prob.model.add_design_var('z')
        prob.model.set_design_var_options(name='x', lower=-100, upper=100, adder=3, scaler=10)
        prob.model.set_design_var_options(name='z', lower=-100, upper=100)
        prob.setup()
        des_vars_using_set_design_var_options = prob.model.get_design_vars()

        self.assertEqual(des_vars_using_add_design_var, des_vars_using_set_design_var_options)

    def test_set_design_var_options_adder_scalar_ref_ref0_override(self):
        # first set the options for scaler and adder using add_design_var and then
        #   override with calls to set_design_var_options. Compare
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_design_var('x', lower=-100, upper=100, scaler=3.5, adder=77.0)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.setup()

        des_vars_using_add_design_var = prob.model.get_design_vars()

        # then set the options for scaler and adder using set_design_var_options
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_design_var('x', lower=-100, upper=100, ref0=-2.0, ref=20)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.set_design_var_options(name='x', scaler=3.5, adder=77.0)
        prob.setup()
        des_vars_using_set_design_var_options = prob.model.get_design_vars()

        assert_near_equal(des_vars_using_add_design_var, des_vars_using_set_design_var_options)

        # Now set the options for ref and ref0 using add_design_var
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_design_var('x', lower=-100, upper=100, ref0=-2.0, ref=20)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.setup()
        des_vars_using_add_design_var = prob.model.get_design_vars()

        # then set the options for ref and ref0 using set_design_var_options
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_design_var('x', lower=-100, upper=100, scaler=3.5, adder=77.0)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.set_design_var_options(name='x', ref0=-2.0, ref=20)
        prob.setup()
        des_vars_using_set_design_var_options = prob.model.get_design_vars()

        assert_near_equal(des_vars_using_add_design_var, des_vars_using_set_design_var_options)

    def test_set_design_var_options_post_setup_run_driver_compare_outputs(self):
        # Check to make sure that set_design_var_options can be called after setup
        #   and have the same effect as if setting the metadata were part of add_design_var
        prob = Problem()
        model = prob.model
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = x - y'), promotes=['*'])
        prob.set_solver_print(level=0)
        prob.driver = ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)
        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=0.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=15.0, linear=True)
        model.set_input_defaults('x', val=0.0)
        model.set_input_defaults('y', val=0.0)
        prob.setup()
        prob.run_driver()
        x_when_setting_using_add_design_var = prob['x']
        y_when_setting_using_add_design_var = prob['y']

        # now do the same but set the constraint using set_design_var_options
        prob = Problem()
        model = prob.model
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = x - y'), promotes=['*'])
        prob.set_solver_print(level=0)
        prob.driver = ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)
        model.add_design_var('x')
        model.add_design_var('y')
        model.add_objective('f_xy')
        model.add_constraint('c', lower=15.0, linear=True)
        model.set_input_defaults('x', val=0.0)
        model.set_input_defaults('y', val=0.0)
        prob.setup()
        model.set_design_var_options('x',lower=-50.0, upper=50.0 )
        model.set_design_var_options('y',lower= 0.0, upper=50.0 )
        prob.run_driver()
        assert_near_equal(prob['x'], x_when_setting_using_add_design_var, 1e-6)
        assert_near_equal(prob['y'], y_when_setting_using_add_design_var, 1e-6)

    def test_set_design_var_options_vector_values(self):
        prob = Problem(model=SellarDerivatives())
        model = prob.model
        model.nonlinear_solver = NonlinearBlockGS()

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', lower=-100, upper=100)
        model.add_constraint('con2', upper=0.0)

        prob.setup()
        new_z_lower_bound = np.array([0.0, 0.0])
        model.set_design_var_options("z", lower=np.array(new_z_lower_bound))
        des_vars_using_set_design_var_options = prob.model.get_design_vars()
        assert_near_equal(des_vars_using_set_design_var_options['z']['lower'], new_z_lower_bound)


class TestSystemSetConstraintsOptions(unittest.TestCase):

    def test_set_constraints_options_lower_upper(self):
        # first set the options lower and upper using add_constraint
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_constraint('con1', lower=-100, upper=100)
        prob.model.add_constraint('con2')
        prob.setup()
        constraints_using_add_constraint = prob.model.get_constraints()

        # then set the options using set_constraint_options
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_constraint('con1')
        prob.model.add_constraint('con2')
        prob.model.set_constraint_options(name='con1', lower=-100, upper=100)
        prob.setup()
        constraints_using_set_objective_options = prob.model.get_constraints()

        self.assertEqual(constraints_using_add_constraint, constraints_using_set_objective_options)

    def test_set_constraints_options_equals_overrides_lower_upper(self):
        # First set lower and upper in add_constraint
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_constraint('con1', equals=345.)
        prob.model.add_constraint('con2')
        prob.setup()
        constraints_using_add_constraint = prob.model.get_constraints()

        # Then set am equals constraint using an override via set_constraint_options
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_constraint('con1', lower=-100, upper=100)
        prob.model.add_constraint('con2')
        prob.model.set_constraint_options('con1', equals=345.)
        prob.setup()
        constraints_using_set_constraint_options = prob.model.get_constraints()
        self.assertEqual(constraints_using_add_constraint, constraints_using_set_constraint_options)

    def test_set_constraints_options_lower_upper_overrides_equals(self):
        # First set lower and upper in add_constraint
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_constraint('con1', lower=-100, upper=100)
        prob.model.add_constraint('con2')
        prob.setup()
        constraints_using_add_constraint = prob.model.get_constraints()

        # Then set it using an override via set_constraint_options
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_constraint('con1', equals=345.)
        prob.model.add_constraint('con2')
        prob.model.set_constraint_options('con1', lower=-100, upper=100)
        prob.setup()
        constraints_using_set_constraint_options = prob.model.get_constraints()

        self.assertEqual(constraints_using_add_constraint, constraints_using_set_constraint_options)

    def test_set_constraints_options_scaler_adder_ref_ref0(self):
        # first set the options ref and ref0 using add_constraint
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_constraint('con1', ref0=-100.0, ref=100)
        prob.model.add_constraint('con2')
        prob.setup()
        constraints_using_add_constraint = prob.model.get_constraints()

        # then set the options using set_constraint_options
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_constraint('con1')
        prob.model.add_constraint('con2')
        prob.model.set_constraint_options(name='con1', ref0=-100.0, ref=100)
        prob.setup()
        constraints_using_set_constraint_options = prob.model.get_constraints()

        self.assertEqual(constraints_using_add_constraint, constraints_using_set_constraint_options)

        # Now do the same using scaler and adder
        # first set the options using add_constraint
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_constraint('con1', adder=3, scaler=10)
        prob.model.add_constraint('con2')
        prob.setup()
        constraints_using_add_constraint = prob.model.get_constraints()

        # then set the options using set_constraint_options
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_constraint('con1')
        prob.model.add_constraint('con2')
        prob.model.set_constraint_options(name='con1', adder=3, scaler=10)
        prob.setup()
        constraints_using_set_constraint_options = prob.model.get_constraints()

        self.assertEqual(constraints_using_add_constraint, constraints_using_set_constraint_options)

    def test_set_constraints_options_adder_scalar_ref_ref0_override(self):
        # Test overriding ref values set in add_constraint with scaler and adder
        #     values set in set_constraint_options
        # Then do the opposite

        # first set the options for scaler and adder using add_constraint
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_constraint('con1', scaler=3.5, adder=77.0, lower=-100, upper=100)
        prob.model.add_constraint('con2')
        prob.setup()
        constraints_using_add_constraint = prob.model.get_constraints()

        # then set the options for scaler and adder using set_constraint_options
        #  overriding the ref0 and ref set in add_constraint
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_constraint('con1', ref0=-2.0, ref=20, lower=-100, upper=100)
        prob.model.add_constraint('con2')
        prob.model.set_constraint_options(name='con1', scaler=3.5, adder=77.0)
        prob.setup()
        constraints_using_set_constraint_options = prob.model.get_constraints()

        assert_near_equal(constraints_using_add_constraint,
                          constraints_using_set_constraint_options)

        # first set the options for ref and ref0 using add_constraint
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_constraint('con1', ref0=-2.0, ref=20, lower=-100, upper=100)
        prob.model.add_constraint('con2')
        prob.setup()
        constraints_using_add_constraint = prob.model.get_constraints()

        # then set the options for ref and ref0 using set_constraint_options
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_constraint('con1', scaler=3.5, adder=77.0, lower=-100, upper=100)
        prob.model.add_constraint('con2')
        prob.model.set_constraint_options('con1', ref0=-2.0, ref=20)
        prob.setup()
        constraints_using_set_constraint_options = prob.model.get_constraints()

        assert_near_equal(constraints_using_add_constraint,
                          constraints_using_set_constraint_options)

    def test_set_constraint_options_with_aliases(self):
        # One complication is the path to constraints in set_constraint_options, since a single
        #  variable may have multiple constraints imposed upon it provided they are aliased,
        #  but we should be able to account for this. If there is ambiguity (an alias exists),
        #  raise an error.
        prob = Problem()
        exec_comp = prob.model.add_subsystem('exec', ExecComp())
        exec_comp.add_expr('y = x**2', y={'shape': (10, 3)}, x={'shape': (10, 3)})
        prob.model.add_constraint('exec.y', alias='y0', indices=slicer[0, -1], scaler=3.5,
                                  adder=77.0)
        prob.model.add_constraint('exec.y', alias='yf', indices=slicer[-1, 0], equals=0)
        prob.setup()
        constraints_using_add_constraint = prob.model.get_constraints()

        prob = Problem()
        exec_comp = prob.model.add_subsystem('exec', ExecComp())
        exec_comp.add_expr('y = x**2', y={'shape': (10, 3)}, x={'shape': (10, 3)})
        prob.model.add_constraint('exec.y', alias='y0', indices=slicer[0, -1])
        prob.model.add_constraint('exec.y', alias='yf', indices=slicer[-1, 0], equals=0)
        prob.model.set_constraint_options('exec.y', alias='y0', scaler=3.5, adder=77.0)
        prob.setup()

        constraints_using_set_constraint_options = prob.model.get_constraints()
        assert_near_equal(constraints_using_add_constraint,
                          constraints_using_set_constraint_options)

    def test_set_constraint_options_error_calling_without_aliases(self):
        prob = Problem()
        exec_comp = prob.model.add_subsystem('exec', ExecComp())
        exec_comp.add_expr('y = x**2', y={'shape': (10, 3)}, x={'shape': (10, 3)})
        prob.model.add_constraint('exec.y', alias='y0', indices=slicer[0, -1], scaler=3.5,
                                  adder=77.0)
        prob.model.add_constraint('exec.y', alias='yf', indices=slicer[-1, 0], equals=0)

        with self.assertRaises(RuntimeError) as cm:
            prob.model.set_constraint_options('exec.y', ref0=-2.0, ref=20)
        msg = "<class Group>: set_constraint_options called with constraint variable 'exec.y' " \
              "that has multiple aliases: ['y0', 'yf']. Call set_objective_options with the " \
              "'alias' argument set to one of those aliases."
        self.assertEqual(str(cm.exception), msg)

    def test_set_constraint_options_vector_values(self):
        prob = Problem()
        model = prob.model
        model.add_subsystem('p1', IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArraySparse(), promotes=['*'])
        model.add_subsystem('obj',
                            ExecComp('o = areas[0, 0] + areas[1, 1]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('widths', lower=-50.0, upper=50.0)
        model.add_objective('o')
        model.add_constraint('areas', equals=np.array([24.0, 21.0, 3.5, 17.5]))

        prob.setup()

        new_areas_equals_bound = np.array([4.0, 1.0, 0.5, 7.5])
        prob.model.set_constraint_options(name='areas', equals=new_areas_equals_bound)
        constraints_using_set_objective_options = prob.model.get_constraints()
        assert_near_equal(constraints_using_set_objective_options['comp.areas']['equals'],
                          new_areas_equals_bound)


class TestSystemSetObjectiveOptions(unittest.TestCase):
    def test_set_objective_options_scaler_adder_ref_ref0(self):
        # See if setting options for objective results in the same metadata whether
        #   they are set using add_objective or set_objective_options

        # first set the options ref and ref0 using add_objective
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_objective('obj', ref0=-100.0, ref=100)
        prob.setup()
        objective_using_add_objective = prob.model.get_objectives()

        # then set the options using set_objective_options
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_objective('obj')
        prob.model.set_objective_options(name='obj', ref0=-100.0, ref=100)
        prob.setup()
        objective_using_set_objective_options = prob.model.get_objectives()

        self.assertEqual(objective_using_add_objective, objective_using_set_objective_options)

        # now do the same using scaler and adder
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_objective('obj', adder=3, scaler=10)
        prob.setup()
        objective_using_add_objective = prob.model.get_objectives()

        # then set the options using set_objective_options
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_objective('obj')
        prob.model.set_objective_options(name='obj', adder=3, scaler=10)
        prob.setup()
        objective_using_set_objective_options = prob.model.get_objectives()

        self.assertEqual(objective_using_add_objective, objective_using_set_objective_options)

    def test_set_objective_options_adder_scalar_ref_ref0_override(self):
        # first set the options for scaler and adder using add_objective
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_objective('obj', scaler=3.5, adder=77.0)
        prob.setup()
        objective_using_add_objective = prob.model.get_objectives()

        # then set the options for ref and ref0 using set_objective_options
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_objective('obj', ref0=-2.0, ref=20)
        prob.model.set_objective_options(name='obj', scaler=3.5, adder=77.0)
        prob.setup()
        objective_using_set_objective_options = prob.model.get_objectives()

        assert_near_equal(objective_using_add_objective, objective_using_set_objective_options)

        # first set the options for ref and ref0 using add_objective
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_objective('obj', ref0=-2.0, ref=20)
        prob.setup()
        objective_using_add_objective = prob.model.get_objectives()

        # then set the options for ref and ref0 using set_design_var_options
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.add_objective('obj', scaler=3.5, adder=77.0)
        prob.model.set_objective_options(name='obj', ref0=-2.0, ref=20)
        prob.setup()
        objective_using_set_objective_options = prob.model.get_objectives()

        assert_near_equal(objective_using_add_objective, objective_using_set_objective_options)

    def test_set_objective_options_with_alias(self):
        # Check to make sure setting options on the objective, where the objective is an alias,
        # works.
        # Testing both ref and ref0 plus scaler and adder
        # Optimize a problem with both using add_objective to set the options and also
        #   by using set_objective_options. Compare the optimization results and also the
        #   objective metadata

        def _build_model():
            prob = Problem()
            model = prob.model
            model.add_subsystem('comp', ParaboloidObjConsArray(), promotes=['*'])
            prob.driver = ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)
            prob.set_solver_print(level=0)
            model.add_design_var('x', lower=-50.0, upper=50.0)
            model.add_design_var('y', lower=-50.0, upper=50.0)
            return prob

        # Set the ref and ref0 using add_objective. Save the output values
        #   to compare results when using set_objective_options
        prob = _build_model()
        prob.model.add_constraint('fg_xy', indices=[1], alias='g_xy', lower=0, upper=10)
        prob.model.add_objective('fg_xy', index=0, alias='f_xy', ref=3.0, ref0=2.0)
        prob.setup()
        objective_using_add_objective = deepcopy(prob.model.get_objectives())
        prob.run_driver()

        fg_xy_using_add_objective = prob['fg_xy'][0]
        comp_x_using_add_objective = prob['comp.x']
        comp_y_using_add_objective = prob['comp.y']

        # Now use set_objective_options to set ref and ref0 and compare
        prob = _build_model()
        prob.model.add_constraint('fg_xy', indices=[1], alias='g_xy', lower=0, upper=10)
        prob.model.add_objective('fg_xy', index=0, alias='f_xy')
        prob.model.set_objective_options(name='fg_xy', alias='f_xy', ref=3.0, ref0=2.0)
        prob.setup()

        objective_using_set_objective_options = deepcopy(prob.model.get_objectives())
        assert_near_equal(objective_using_set_objective_options, objective_using_add_objective)

        prob.run_driver()
        assert_near_equal(prob['fg_xy'][0], fg_xy_using_add_objective, 1e-6)
        assert_near_equal(prob['comp.x'], comp_x_using_add_objective, 1e-4)
        assert_near_equal(prob['comp.y'], comp_y_using_add_objective, 1e-4)

        # Set the scaler and adder using add_objective. Save the output values
        #   to compare results when using set_objective_options
        prob = _build_model()
        prob.model.add_constraint('fg_xy', indices=[1], alias='g_xy', lower=0, upper=10)
        prob.model.add_objective('fg_xy', index=0, alias='f_xy', scaler=3.0, adder=2.0)
        prob.setup()
        objective_using_add_objective = deepcopy(prob.model.get_objectives())

        prob.run_driver()
        fg_xy_using_add_objective = prob['fg_xy'][0]
        comp_x_using_add_objective = prob['comp.x']
        comp_y_using_add_objective = prob['comp.y']

        # Now use set_objective_options to set scaler and adder and compare
        prob = _build_model()
        prob.model.add_constraint('fg_xy', indices=[1], alias='g_xy', lower=0, upper=10)
        prob.model.add_objective('fg_xy', index=0, alias='f_xy')
        prob.model.set_objective_options(name='fg_xy', alias='f_xy', scaler=3.0, adder=2.0)
        prob.setup()

        objective_using_set_objective_options = deepcopy(prob.model.get_objectives())
        assert_near_equal(objective_using_set_objective_options, objective_using_add_objective)

        prob.run_driver()
        assert_near_equal(prob['fg_xy'][0], fg_xy_using_add_objective, 1e-6)
        assert_near_equal(prob['comp.x'], comp_x_using_add_objective, 1e-4)
        assert_near_equal(prob['comp.y'], comp_y_using_add_objective, 1e-4)

    def test_set_objective_options_without_using_alias(self):
        prob = Problem()
        model = prob.model
        model.add_subsystem('comp', ParaboloidObjConsArray(), promotes=['*'])
        prob.driver = ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)
        prob.set_solver_print(level=0)
        prob.model.add_constraint('fg_xy', indices=[1], alias='g_xy', lower=0, upper=10)
        prob.model.add_objective('fg_xy', index=0, alias='f_xy')

        with self.assertRaises(RuntimeError) as cm:
            prob.model.set_objective_options(name='fg_xy', ref=3.0, ref0=2.0)
        msg = "<class Group>: set_objective_options called with objective variable 'fg_xy' that " \
              "has multiple aliases: ['g_xy', 'f_xy']. Call set_objective_options with the " \
              "'alias' argument set to one of those aliases."
        self.assertEqual(str(cm.exception), msg)


class ImplCompTwoStatesNoMetadataSetInAddOutput(ImplicitComponent):
    """
    A Simple Implicit Component with an additional output equation.

    This version has no metadata set in the calls to add_ouput
    """
    def setup(self):
        self.add_input('x', 0.5)
        self.add_output('y', 0.0)
        self.add_output('z', 2.0)

        self.maxiter = 10
        self.atol = 1.0e-12

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        x = inputs['x']
        y = outputs['y']
        z = outputs['z']

        residuals['y'] = y - x - 2.0 * z
        residuals['z'] = x * z + z - 4.0

    def linearize(self, inputs, outputs, jac):
        jac[('y', 'x')] = -1.0
        jac[('y', 'y')] = 1.0
        jac[('y', 'z')] = -2.0

        jac[('z', 'z')] = inputs['x'] + 1.0
        jac[('z', 'x')] = outputs['z']


class ScalingExample3(ImplicitComponent):
    def setup(self):
        self.add_input('x1', val=100.0)
        self.add_input('x2', val=5000.0)
        self.add_output('y1', val=200., ref=1e2, res_ref=1e5, ref0=1.0, lower=2., upper=3)
        self.add_output('y2', val=6000., ref=1e3, res_ref=1e-5)

    def apply_nonlinear(self, inputs, outputs, residuals):
        x1 = inputs['x1']
        x2 = inputs['x2']
        y1 = outputs['y1']
        y2 = outputs['y2']

        residuals['y1'] = 1e5 * (x1 - y1) / y1
        residuals['y2'] = 1e-5 * (x2 - y2) / y2


class ScalingExample3NoMetadataSetInAddOutput(ImplicitComponent):
    def setup(self):
        self.add_input('x1', val=100.0)
        self.add_input('x2', val=5000.0)
        self.add_output('y1', val=200.)
        self.add_output('y2', val=6000.)

    def apply_nonlinear(self, inputs, outputs, residuals):
        x1 = inputs['x1']
        x2 = inputs['x2']
        y1 = outputs['y1']
        y2 = outputs['y2']

        residuals['y1'] = 1e5 * (x1 - y1) / y1
        residuals['y2'] = 1e-5 * (x2 - y2) / y2

class ScalingExample3ForTesting(ImplicitComponent):
    def __init__(self, ref1=1.0, lower1=None, ref2=1.0, lower2=None):
        super().__init__()
        self.ref1 = ref1
        self.lower1 = lower1
        self.ref2 = ref2
        self.lower2 = lower2

    def setup(self):
        self.add_input('x1', val=100.0)
        self.add_input('x2', val=5000.0)
        self.add_output('y1', val=200., ref=self.ref1, lower=self.lower1)
        self.add_output('y2', val=200., ref=self.ref2, lower=self.lower2)

    def apply_nonlinear(self, inputs, outputs, residuals):
        x1 = inputs['x1']
        x2 = inputs['x2']
        y1 = outputs['y1']
        y2 = outputs['y2']
        residuals['y1'] = 1e5 * (x1 - y1) / y1
        residuals['y2'] = 1e-5 * (x2 - y2) / y2

class ParaboloidObjConsArray(ExplicitComponent):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)
        self.add_output('fg_xy', val=np.array([0., 0.]))

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute_partials(self, inputs, partials):
        x = inputs['x']
        y = inputs['y']

        partials['fg_xy', 'x'][0] = 2 * (x - 3) + y
        partials['fg_xy', 'x'][1] = 1.0

        partials['fg_xy', 'y'][0] = x + 2 * (y + 4)
        partials['fg_xy', 'y'][1] = 1.0

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        outputs['fg_xy'] = [
            (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0,
            x + y
        ]

if __name__ == "__main__":
    unittest.main()
