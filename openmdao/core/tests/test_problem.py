""" Unit tests for the problem interface."""

import sys
import unittest
import itertools

from io import StringIO
import numpy as np
from collections import defaultdict

import openmdao.api as om
from openmdao.core.problem import _default_prob_name
from openmdao.core.driver import Driver
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.misc_components import MultComp
from openmdao.test_suite.components.sellar import SellarDerivatives, SellarDerivativesConnected
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
import openmdao.utils.hooks as hooks
from openmdao.utils.units import convert_units
from openmdao.utils.om_warnings import DerivativesWarning
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.tests.test_hooks import hooks_active

try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized


class TestProblem(unittest.TestCase):
    def test_simple_component_model_with_units(self):
        class TestComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('foo', units='N')
                self.add_output('bar', units='N')
                self.declare_partials('bar', 'foo')

            def compute(self, inputs, outputs):
                outputs['bar'] = inputs['foo']

            def compute_partials(self, inputs, J):
                J['bar', 'foo'] = 1.

        p = om.Problem(model=TestComp())
        p.setup()

        p.set_val('foo', 5, units='lbf')
        p.run_model()

        lbf_val = convert_units(5, 'lbf', 'N')
        self.assertEqual(p.get_val('foo'), lbf_val)
        self.assertEqual(p.get_val('bar'), lbf_val)

    def test_feature_simple_run_once_no_promote(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid())

        model.set_input_defaults('comp.x', 3.0)
        model.set_input_defaults('comp.y', -4.0)

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['comp.f_xy'], -15.0)

    def test_feature_simple_run_once_input_input(self):

        prob = om.Problem()
        model = prob.model

        # promote the two inputs to the same name
        model.add_subsystem('comp1', Paraboloid(), promotes_inputs=['x'])
        model.add_subsystem('comp2', Paraboloid(), promotes_inputs=['x'])

        model.set_input_defaults('x', 3.0)

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['comp1.f_xy'], 13.0)
        assert_near_equal(prob['comp2.f_xy'], 13.0)

    def test_feature_simple_run_once_compute_totals(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid())

        model.set_input_defaults('comp.x', 3.0)
        model.set_input_defaults('comp.y', -4.0)

        prob.setup()

        prob.run_model()

        totals = prob.compute_totals(of=['comp.f_xy'], wrt=['comp.x', 'comp.y'])
        assert_near_equal(totals[('comp.f_xy', 'comp.x')][0][0], -4.0)
        assert_near_equal(totals[('comp.f_xy', 'comp.y')][0][0], 3.0)

        totals = prob.compute_totals(of=['comp.f_xy'], wrt=['comp.x', 'comp.y'], return_format='dict')
        assert_near_equal(totals['comp.f_xy']['comp.x'][0][0], -4.0)
        assert_near_equal(totals['comp.f_xy']['comp.y'][0][0], 3.0)

    def test_feature_simple_run_once_compute_totals_scaled(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid())

        model.set_input_defaults('comp.x', 3.0)
        model.set_input_defaults('comp.y', -4.0)

        model.add_design_var('comp.x', 3.0, ref0=50.0)
        model.add_design_var('comp.y', -4.0)
        model.add_objective('comp.f_xy')

        prob.setup()

        prob.run_model()

        totals = prob.compute_totals(of=['comp.f_xy'], wrt=['comp.x', 'comp.y'], driver_scaling=True)
        assert_near_equal(totals[('comp.f_xy', 'comp.x')][0][0], 196.0)
        assert_near_equal(totals[('comp.f_xy', 'comp.y')][0][0], 3.0)

    def test_feature_simple_run_once_set_deriv_mode(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid())

        model.set_input_defaults('comp.x', 3.0)
        model.set_input_defaults('comp.y', -4.0)

        prob.setup(mode='rev')
        # prob.setup(mode='fwd')
        prob.run_model()

        assert_near_equal(prob['comp.f_xy'], -15.0)

        prob.compute_totals(of=['comp.f_xy'], wrt=['comp.x', 'comp.y'])

    def test_single_string_wrt_of(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        model.add_subsystem('p2', om.IndepVarComp('y', -4.0))
        model.add_subsystem('comp', Paraboloid())

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        prob.setup()
        prob.run_model()

        totals = prob.compute_totals(of='comp.f_xy', wrt='p1.x')
        assert_near_equal(totals[('comp.f_xy', 'p1.x')][0][0], -4.0)

    def test_two_var_single_string_error(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
        model.add_subsystem('p2', om.IndepVarComp('y', -4.0))
        model.add_subsystem('comp', Paraboloid())

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        prob.setup()
        prob.run_model()

        with self.assertRaises(KeyError) as cm:
            totals = prob.compute_totals(of='comp.f_xy', wrt="p1.x, p2.y")
        self.assertEqual(str(cm.exception), "'p1.x, p2.y'")

    def test_compute_totals_cleanup(self):
        p = om.Problem()
        model = p.model
        model.add_subsystem('indeps1', om.IndepVarComp('x', np.ones(5)))
        model.add_subsystem('indeps2', om.IndepVarComp('x', np.ones(3)))

        model.add_subsystem('MP1', om.ExecComp('y=7*x', x=np.zeros(5), y=np.zeros(5)))
        model.add_subsystem('MP2', om.ExecComp('y=-3*x', x=np.zeros(3), y=np.zeros(3)))

        model.add_design_var('indeps1.x')
        model.add_design_var('indeps2.x')

        model.add_constraint('MP1.y')
        model.add_constraint('MP2.y')

        model.connect('indeps1.x', 'MP1.x')
        model.connect('indeps2.x', 'MP2.x')

        p.setup(mode='rev')
        p.run_model()

        J = p.compute_totals()
        assert_near_equal(J[('MP1.y', 'indeps1.x')], np.eye(5)*7., 1e-10)
        assert_near_equal(J[('MP2.y', 'indeps2.x')], np.eye(3)*-3., 1e-10)
        # before the bug fix, the following two derivs contained nonzero values even
        # though the variables involved were not dependent on each other.
        assert_near_equal(J[('MP2.y', 'indeps1.x')], np.zeros((3, 5)), 1e-10)
        assert_near_equal(J[('MP1.y', 'indeps2.x')], np.zeros((5, 3)), 1e-10)

    def test_set_2d_array(self):

        prob = om.Problem()
        model = prob.model
        model.add_subsystem(name='indeps',
                            subsys=om.IndepVarComp(name='X_c', shape=(3, 1)))
        prob.setup()

        new_val = -5*np.ones((3, 1))
        prob['indeps.X_c'] = new_val
        prob.final_setup()

        assert_near_equal(prob['indeps.X_c'], new_val, 1e-10)

        new_val = 2.5*np.ones(3)
        prob['indeps.X_c'][:, 0] = new_val
        prob.final_setup()

        assert_near_equal(prob['indeps.X_c'], new_val.reshape((3, 1)), 1e-10)
        assert_near_equal(prob['indeps.X_c'][:, 0], new_val, 1e-10)

    def test_set_checks_shape(self):

        model = om.Group()

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_output('num')
        indep.add_output('arr', shape=(10, 1))

        prob = om.Problem(model)
        prob.setup()

        msg = "Incompatible shape for '.*': Expected (.*) but got (.*)"

        # check valid scalar value
        new_val = -10.
        prob['indep.num'] = new_val
        assert_near_equal(prob['indep.num'], new_val, 1e-10)

        # check bad scalar value
        bad_val = -10*np.ones((10))
        prob['indep.num'] = bad_val
        with self.assertRaisesRegex(ValueError,
                "<model> <class Group>: Failed to set value of '.*': could not broadcast input array from shape (.*) into shape (.*)."):
            prob.final_setup()
        prob.model._initial_condition_cache = {}

        # check assign scalar to array
        arr_val = new_val*np.ones((10, 1))
        prob['indep.arr'] = new_val
        prob.final_setup()
        assert_near_equal(prob['indep.arr'], arr_val, 1e-10)

        # check valid array value
        new_val = -10*np.ones((10, 1))
        prob['indep.arr'] = new_val
        assert_near_equal(prob['indep.arr'], new_val, 1e-10)

        msg = "<model> <class Group>: Failed to set value of '.*': could not broadcast input array from shape (.*) into shape (.*)."
        # check bad array value
        bad_val = -10*np.ones((9,1))
        with self.assertRaisesRegex(ValueError, msg):
            prob['indep.arr'] = bad_val

        # check valid list value
        new_val = new_val.tolist()
        prob['indep.arr'] = new_val
        assert_near_equal(prob['indep.arr'], new_val, 1e-10)

        # check bad list value
        bad_val = bad_val.tolist()
        with self.assertRaisesRegex(ValueError, msg):
            prob['indep.arr'] = bad_val

    def test_compute_totals_basic(self):
        # Basic test for the method using default solvers on simple model.

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_near_equal(derivs['f_xy', 'x'], [[-6.0]], 1e-6)
        assert_near_equal(derivs['f_xy', 'y'], [[8.0]], 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_near_equal(derivs['f_xy', 'x'], [[-6.0]], 1e-6)
        assert_near_equal(derivs['f_xy', 'y'], [[8.0]], 1e-6)

    def test_compute_totals_basic_return_dict(self):
        # Make sure 'dict' return_format works.

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_near_equal(derivs['f_xy']['x'], [[-6.0]], 1e-6)
        assert_near_equal(derivs['f_xy']['y'], [[8.0]], 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_near_equal(derivs['f_xy']['x'], [[-6.0]], 1e-6)
        assert_near_equal(derivs['f_xy']['y'], [[8.0]], 1e-6)

    def test_compute_totals_no_args_no_desvar(self):
        p = om.Problem()

        dv = p.model.add_subsystem('des_vars', om.IndepVarComp())
        dv.add_output('x', val=2.)

        p.model.add_subsystem('calc', om.ExecComp('y=2*x'))

        p.model.connect('des_vars.x', 'calc.x')

        p.model.add_objective('calc.y')

        p.setup()
        p.run_model()

        with self.assertRaises(RuntimeError) as cm:
            p.compute_totals()

        self.assertEqual(str(cm.exception),
                         "Driver is not providing any design variables for compute_totals.")

    def test_compute_totals_no_args_no_response(self):
        p = om.Problem()

        dv = p.model.add_subsystem('des_vars', om.IndepVarComp())
        dv.add_output('x', val=2.)

        p.model.add_subsystem('calc', om.ExecComp('y=2*x'))

        p.model.connect('des_vars.x', 'calc.x')

        p.model.add_design_var('des_vars.x')

        p.setup()
        p.run_model()

        with self.assertRaises(RuntimeError) as cm:
            p.compute_totals()

        self.assertEqual(str(cm.exception),
                         "Driver is not providing any response variables for compute_totals.")

    def test_compute_totals_no_args(self):
        p = om.Problem()

        dv = p.model.add_subsystem('des_vars', om.IndepVarComp())
        dv.add_output('x', val=2.)

        p.model.add_subsystem('calc', om.ExecComp('y=2*x'))

        p.model.connect('des_vars.x', 'calc.x')

        p.model.add_design_var('des_vars.x')
        p.model.add_objective('calc.y')

        p.setup()
        p.run_model()

        derivs = p.compute_totals()

        assert_near_equal(derivs['calc.y', 'des_vars.x'], [[2.0]], 1e-6)

    def test_compute_totals_no_args_promoted(self):
        p = om.Problem()

        dv = p.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])
        dv.add_output('x', val=2.)

        p.model.add_subsystem('calc', om.ExecComp('y=2*x'), promotes=['*'])

        p.model.add_design_var('x')
        p.model.add_objective('y')

        p.setup()
        p.run_model()

        derivs = p.compute_totals()

        assert_near_equal(derivs['calc.y', 'des_vars.x'], [[2.0]], 1e-6)

    @parameterized.expand(itertools.product(['fwd', 'rev']))
    def test_compute_jacvec_product(self, mode):

        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()
        prob.model.linear_solver = om.ScipyKrylov()

        prob.setup(mode=mode)
        prob.run_model()

        of = ['obj', 'con1']
        wrt = ['_auto_ivc.v1', '_auto_ivc.v0']

        if mode == 'fwd':
            seed_names = wrt
            result_names = of
            rvec = prob.model._vectors['output']['linear']
            lvec = prob.model._vectors['residual']['linear']
        else:
            seed_names = of
            result_names = wrt
            rvec = prob.model._vectors['residual']['linear']
            lvec = prob.model._vectors['output']['linear']

        J = prob.compute_totals(of, wrt, return_format='array')

        seed = []
        for name in seed_names:
            seed.append(np.random.random(rvec[name].size))

        resdict = prob.compute_jacvec_product(of, wrt, mode, seed)
        result = []
        for name in result_names:
            result.append(resdict[name].flat)
        result = np.hstack(result)

        testvec = np.hstack(seed)

        if mode == 'fwd':
            checkvec = J.dot(testvec)
        else:
            checkvec = J.T.dot(testvec)

        np.testing.assert_allclose(checkvec, result)

    def test_feature_set_indeps(self):

        prob = om.Problem()

        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        prob.setup()

        prob['x'] = 2.
        prob['y'] = 10.
        prob.run_model()
        assert_near_equal(prob['f_xy'], 214.0, 1e-6)

    def test_feature_set_indeps_auto(self):

        prob = om.Problem()

        prob.model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        prob.setup()

        prob.set_val('x', 2.)
        prob.set_val('y', 10.)
        prob.run_model()
        assert_near_equal(prob.get_val('f_xy'), 214.0, 1e-6)

    def test_feature_basic_setup(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        prob.setup()

        prob.set_val('x', 2.)
        prob.set_val('y', 10.)
        prob.run_model()
        assert_near_equal(prob.get_val('f_xy'), 214.0, 1e-6)

        prob.set_val('x', 0.)
        prob.set_val('y', 0.)
        prob.run_model()
        assert_near_equal(prob.get_val('f_xy'), 22.0, 1e-6)

        prob.setup()

        prob.set_val('x', 4.)
        prob.set_val('y', 8.)
        prob.run_model()
        assert_near_equal(prob.get_val('f_xy'), 174.0, 1e-6)

    def test_feature_petsc_setup(self):

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        # PETScVectors will be used automatically where needed. No need to set manually.
        prob.setup()
        prob['x'] = 2.
        prob['y'] = 10.

        prob.run_model()
        assert_near_equal(prob['f_xy'], 214.0, 1e-6)

    def test_feature_check_totals_manual(self):

        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.setup()
        prob.run_model()

        # manually specify which derivatives to check
        prob.check_totals(of=['obj', 'con1'], wrt=['x', 'z'])

    def test_feature_check_totals_from_driver_compact(self):

        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.setup()

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        # check derivatives of all obj+constraints w.r.t all design variables
        prob.check_totals(compact_print=True)

    def test_feature_check_totals_from_driver(self):

        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.setup()

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        # check derivatives of all obj+constraints w.r.t all design variables
        prob.check_totals()

    def test_feature_check_totals_from_driver_scaled(self):

        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100, ref=100.0, ref0=-100.0)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0, ref=3.0)
        prob.model.add_constraint('con2', upper=0.0, ref=20.0)

        prob.setup()

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        # check derivatives of all driver vars using the declared scaling
        prob.check_totals(driver_scaling=True)

    def test_feature_check_totals_suppress(self):

        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.setup()

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        # check derivatives of all obj+constraints w.r.t all design variables
        totals = prob.check_totals(out_stream=None)
        print(totals)

    def test_feature_check_totals_cs(self):

        prob = om.Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.setup(force_alloc_complex=True)

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        prob.model.nonlinear_solver.options['atol'] = 1e-15
        prob.model.nonlinear_solver.options['rtol'] = 1e-15

        # check derivatives with complex step
        prob.check_totals(method='cs')

    def test_check_totals_user_detect(self):

        class SimpleComp(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', val=1.0)
                self.add_output('y', val=1.0)

                self.declare_partials(of='y', wrt='x')

                if not self._force_alloc_complex:
                    raise RuntimeError('force_alloc_complex not set in component.')

            def compute(self, inputs, outputs):
                outputs['y'] = 3.0*inputs['x']

                if inputs.iscomplex() and not self.under_complex_step:
                    raise RuntimeError('under_complex_step not set in component.')

            def compute_partials(self, inputs, partials):
                partials['y', 'x'] = 3.

        prob = om.Problem()
        prob.model.add_subsystem('px', om.IndepVarComp('x', 2.0))
        prob.model.add_subsystem('comp', SimpleComp())
        prob.model.connect('px.x', 'comp.x')

        prob.model.add_design_var('px.x', lower=-100, upper=100)
        prob.model.add_objective('comp.y')

        prob.setup(force_alloc_complex=True)

        prob.run_model()

        # check derivatives with complex step and a larger step size.
        prob.check_totals(method='cs', out_stream=None)
        self.assertFalse(prob.model.under_complex_step,
                         msg="The under_complex_step flag should be reset.")

    def test_feature_check_totals_user_detect_forced(self):

        class SimpleComp(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', val=1.0)
                self.add_output('y', val=1.0)

                self.declare_partials(of='y', wrt='x')

                if self._force_alloc_complex:
                    print("Vectors allocated for complex step.")

            def compute(self, inputs, outputs):
                outputs['y'] = 3.0*inputs['x']

            def compute_partials(self, inputs, partials):
                partials['y', 'x'] = 3.

        prob = om.Problem()
        prob.model.add_subsystem('comp', SimpleComp())

        prob.model.add_design_var('comp.x', lower=-100, upper=100)
        prob.model.add_objective('comp.y')

        prob.setup(force_alloc_complex=True)

        prob.run_model()

        prob.check_totals(method='cs')

    def test_set_cs_error_messages(self):
        prob = om.Problem()
        prob.model.add_subsystem('comp', Paraboloid())
        prob.setup()
        prob.run_model()
        msg = "Problem .*: To enable complex step, specify 'force_alloc_complex=True' when calling " + \
            "setup on the problem, e\.g\. 'problem\.setup\(force_alloc_complex=True\)'"
        with self.assertRaisesRegex(RuntimeError, msg):
            prob.set_complex_step_mode(True)

        prob = om.Problem()
        prob.model.add_subsystem('comp', Paraboloid())
        msg = "Problem .*: set_complex_step_mode cannot be called before `Problem\.run_model\(\)`, " + \
            "`Problem\.run_driver\(\)`, or `Problem\.final_setup\(\)`."
        with self.assertRaisesRegex(RuntimeError, msg) :
            prob.set_complex_step_mode(True)

    def test_feature_run_driver(self):

        prob = om.Problem(model=SellarDerivatives())
        model = prob.model
        model.nonlinear_solver = om.NonlinearBlockGS()

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob.get_val('x'), 0.0, 1e-5)
        assert_near_equal(prob.get_val('y1'), 3.160000, 1e-2)
        assert_near_equal(prob.get_val('y2'), 3.755278, 1e-2)
        assert_near_equal(prob.get_val('z'), [1.977639, 0.000000], 1e-2)
        assert_near_equal(prob.get_val('obj'), 3.18339395, 1e-2)

    def test_feature_promoted_sellar_set_get_outputs(self):

        prob = om.Problem(model=SellarDerivatives())
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.setup()

        prob.set_val('x', 2.75)

        prob.run_model()

        assert_near_equal(prob.get_val('y1'), 27.3049178437, 1e-6)

    def test_feature_not_promoted_sellar_set_get_outputs(self):

        prob = om.Problem(model= SellarDerivativesConnected())
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.setup()

        prob.set_val('x', 2.75)

        prob.run_model()

        assert_near_equal(prob.get_val('x'), 2.75, 1e-6)

        assert_near_equal(prob.get_val('d1.y1'), 27.3049178437, 1e-6)

    def test_feature_promoted_sellar_set_get_inputs(self):

        prob = om.Problem(model=SellarDerivatives())
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.setup()

        prob['x'] = 2.75

        prob.run_model()

        assert_near_equal(prob['x'], 2.75, 1e-6)

        # the output variable, referenced by the promoted name
        assert_near_equal(prob['y1'], 27.3049178437, 1e-6)
        # the connected input variable, referenced by the absolute path
        assert_near_equal(prob['d2.y1'], 27.3049178437, 1e-6)

    def test_get_set_with_units_exhaustive(self):

        prob = om.Problem()
        prob.model.add_subsystem('comp', om.ExecComp('y=x-25.',
                                                     x={'val': 77.0, 'units': 'degF'},
                                                     y={'val': 0.0, 'units': 'degC'}))
        prob.model.add_subsystem('prom', om.ExecComp('yy=xx-25.',
                                                     xx={'val': 77.0, 'units': 'degF'},
                                                     yy={'val': 0.0, 'units': 'degC'}),
                                 promotes=['xx', 'yy'])
        prob.model.add_subsystem('acomp', om.ExecComp('y=x-25.',
                                                      x={'val': np.array([77.0, 95.0]), 'units': 'degF'},
                                                      y={'val': 0.0, 'units': 'degC'}))
        prob.model.add_subsystem('aprom', om.ExecComp('ayy=axx-25.',
                                                      axx={'val': np.array([77.0, 95.0]), 'units': 'degF'},
                                                      ayy={'val': 0.0, 'units': 'degC'}),
                                 promotes=['axx', 'ayy'])

        prob.setup()

        # Make sure everything works before final setup with caching system.

        # Gets

        assert_near_equal(prob.get_val('comp.x'), 77.0, 1e-6)
        assert_near_equal(prob.get_val('comp.x', 'degC'), 25.0, 1e-6)
        assert_near_equal(prob.get_val('comp.y'), 0.0, 1e-6)
        assert_near_equal(prob.get_val('comp.y', 'degF'), 32.0, 1e-6)

        assert_near_equal(prob.get_val('xx'), 77.0, 1e-6)
        assert_near_equal(prob.get_val('xx', 'degC'), 25.0, 1e-6)
        assert_near_equal(prob.get_val('yy'), 0.0, 1e-6)
        assert_near_equal(prob.get_val('yy', 'degF'), 32.0, 1e-6)

        assert_near_equal(prob.get_val('acomp.x', indices=0), 77.0, 1e-6)
        assert_near_equal(prob.get_val('acomp.x', indices=[1]), 95.0, 1e-6)
        assert_near_equal(prob.get_val('acomp.x', 'degC', indices=[0]), 25.0, 1e-6)
        assert_near_equal(prob.get_val('acomp.x', 'degC', indices=1), 35.0, 1e-6)
        assert_near_equal(prob.get_val('acomp.y', indices=0), 0.0, 1e-6)
        assert_near_equal(prob.get_val('acomp.y', 'degF', indices=0), 32.0, 1e-6)

        assert_near_equal(prob.get_val('axx', indices=0), 77.0, 1e-6)
        assert_near_equal(prob.get_val('axx', indices=1), 95.0, 1e-6)
        assert_near_equal(prob.get_val('axx', 'degC', indices=0), 25.0, 1e-6)
        assert_near_equal(prob.get_val('axx', 'degC', indices=np.array([1])), 35.0, 1e-6)
        assert_near_equal(prob.get_val('ayy', indices=0), 0.0, 1e-6)
        assert_near_equal(prob.get_val('ayy', 'degF', indices=0), 32.0, 1e-6)

        # Sets

        prob.set_val('comp.x', 30.0, 'degC')
        assert_near_equal(prob['comp.x'], 86.0, 1e-6)
        assert_near_equal(prob.get_val('comp.x', 'degC'), 30.0, 1e-6)

        prob.set_val('xx', 30.0, 'degC')
        assert_near_equal(prob['xx'], 86.0, 1e-6)
        assert_near_equal(prob.get_val('xx', 'degC'), 30.0, 1e-6)

        prob.set_val('acomp.x', 30.0, 'degC', indices=[0])
        assert_near_equal(prob['acomp.x'][0], 86.0, 1e-6)
        assert_near_equal(prob.get_val('acomp.x', 'degC', indices=0), 30.0, 1e-6)

        prob.set_val('axx', 30.0, 'degC', indices=0)
        assert_near_equal(prob['axx'][0], 86.0, 1e-6)
        assert_near_equal(prob.get_val('axx', 'degC', indices=np.array([0])), 30.0, 1e-6)

        prob.final_setup()

        # Now we do it all over again for coverage.

        # Gets

        assert_near_equal(prob.get_val('comp.x'), 86.0, 1e-6)
        assert_near_equal(prob.get_val('comp.x', 'degC'), 30.0, 1e-6)
        assert_near_equal(prob.get_val('comp.y'), 0.0, 1e-6)
        assert_near_equal(prob.get_val('comp.y', 'degF'), 32.0, 1e-6)

        assert_near_equal(prob.get_val('xx'), 86.0, 1e-6)
        assert_near_equal(prob.get_val('xx', 'degC'), 30.0, 1e-6)
        assert_near_equal(prob.get_val('yy'), 0.0, 1e-6)
        assert_near_equal(prob.get_val('yy', 'degF'), 32.0, 1e-6)

        assert_near_equal(prob.get_val('acomp.x', indices=0), 86.0, 1e-6)
        assert_near_equal(prob.get_val('acomp.x', indices=[1]), 95.0, 1e-6)
        assert_near_equal(prob.get_val('acomp.x', 'degC', indices=[0]), 30.0, 1e-6)
        assert_near_equal(prob.get_val('acomp.x', 'degC', indices=1), 35.0, 1e-6)
        assert_near_equal(prob.get_val('acomp.y', indices=0), 0.0, 1e-6)
        assert_near_equal(prob.get_val('acomp.y', 'degF', indices=0), 32.0, 1e-6)

        assert_near_equal(prob.get_val('axx', indices=0), 86.0, 1e-6)
        assert_near_equal(prob.get_val('axx', indices=1), 95.0, 1e-6)
        assert_near_equal(prob.get_val('axx', 'degC', indices=0), 30.0, 1e-6)
        assert_near_equal(prob.get_val('axx', 'degC', indices=np.array([1])), 35.0, 1e-6)
        assert_near_equal(prob.get_val('ayy', indices=0), 0.0, 1e-6)
        assert_near_equal(prob.get_val('ayy', 'degF', indices=0), 32.0, 1e-6)

        # Sets

        prob.set_val('comp.x', 35.0, 'degC')
        assert_near_equal(prob['comp.x'], 95.0, 1e-6)
        assert_near_equal(prob.get_val('comp.x', 'degC'), 35.0, 1e-6)

        prob.set_val('xx', 35.0, 'degC')
        assert_near_equal(prob['xx'], 95.0, 1e-6)
        assert_near_equal(prob.get_val('xx', 'degC'), 35.0, 1e-6)

        prob.set_val('acomp.x', 35.0, 'degC', indices=[0])
        assert_near_equal(prob['acomp.x'][0], 95.0, 1e-6)
        assert_near_equal(prob.get_val('acomp.x', 'degC', indices=0), 35.0, 1e-6)

        prob.set_val('axx', 35.0, 'degC', indices=0)
        assert_near_equal(prob['axx'][0], 95.0, 1e-6)
        assert_near_equal(prob.get_val('axx', 'degC', indices=np.array([0])), 35.0, 1e-6)

    def test_feature_get_set_with_units_diff_err(self):

        prob = om.Problem(name="get_set_with_units_diff_err")
        prob.model.add_subsystem('C1', om.ExecComp('y=x*2.',
                                                     x={'val': 1.0, 'units': 'ft'},
                                                     y={'val': 0.0, 'units': 'ft'}),
                                 promotes=['x'])
        prob.model.add_subsystem('C2', om.ExecComp('y=x*3.',
                                                     x={'val': 1.0, 'units': 'inch'},
                                                     y={'val': 0.0, 'units': 'inch'}),
                                 promotes=['x'])

        try:
            prob.setup()
        except Exception as err:
            self.assertEqual(str(err),
               "\nCollected errors for problem 'get_set_with_units_diff_err':"
               "\n   <model> <class Group>: The following inputs, ['C1.x', 'C2.x'], promoted to "
               "'x', are connected but their metadata entries ['units', 'val'] differ. "
               "Call <group>.set_input_defaults('x', units=?, val=?), where <group> is the model "
               "to remove the ambiguity.")
        else:
            self.fail("Exception expected.")

    def test_feature_get_set_with_units_diff(self):

        prob = om.Problem()
        G1 = prob.model.add_subsystem('G1', om.Group())
        G1.add_subsystem('C1', om.ExecComp('y=x*2.',
                                            x={'val': 1.0, 'units': 'cm'},
                                            y={'val': 0.0, 'units': 'cm'}),
                         promotes=['x'])
        G1.add_subsystem('C2', om.ExecComp('y=x*3.',
                                            x={'val': 1.0, 'units': 'mm'},
                                            y={'val': 0.0, 'units': 'mm'}),
                         promotes=['x'])

        # units and value to use for the _auto_ivc output are ambiguous.  This fixes that.
        G1.set_input_defaults('x', units='m', val=1.0)

        prob.setup()

        # set G1.x to 2.0 m, based on the units we gave in the set_input_defaults call
        prob.set_val('G1.x', 2.)

        prob.run_model()

        # we gave 'G1.x' units of 'm' in the set_input_defaults call
        assert_near_equal(prob.get_val('G1.x'), 2.0, 1e-6)

        # using absolute value will give us the value of the input C1.x, in its units of 'cm'
        assert_near_equal(prob.get_val('G1.C1.x'), 200.0, 1e-6)

        # using absolute value will give us the value of the input C2.x, in its units of 'mm'
        assert_near_equal(prob.get_val('G1.C2.x'), 2000.0, 1e-6)

    def test_feature_get_set_with_src_indices_diff(self):

        prob = om.Problem()
        G1 = prob.model.add_subsystem('G1', om.Group())
        G1.add_subsystem('C1', om.ExecComp('y=x*2.',
                                            x={'val': 1.0, 'units': 'cm', 'src_indices': [0], 'flat_src_indices': True},
                                            y={'val': 0.0, 'units': 'cm'}),
                         promotes=['x'])
        G1.add_subsystem('C2', om.ExecComp('y=x*3.',
                                            x={'val': np.ones(2), 'units': 'mm', 'src_indices': [1,2], 'flat_src_indices': True},
                                            y={'val': np.zeros(2), 'units': 'mm'}),
                         promotes=['x'])
        G1.add_subsystem('C3', om.ExecComp('y=x*4.',
                                            x={'val': np.ones(3), 'units': 'mm'},
                                            y={'val': np.zeros(3), 'units': 'mm'}),
                         promotes=['x'])

        # units and value to use for the _auto_ivc output are ambiguous.  This fixes that.
        G1.set_input_defaults('x', units='m', val=np.ones(3))

        prob.setup()

        # set G1.x to 2.0 m, based on the units we gave in the set_input_defaults call
        prob['G1.x'] = np.ones(3) * 2.0

        prob.run_model()

        # we gave 'G1.x' units of 'm' in the set_input_defaults call
        assert_near_equal(prob['G1.x'], np.ones(3) * 2.0, 1e-6)

        # using absolute value will give us the value of the input C1.x, in its units of 'cm'
        assert_near_equal(prob['G1.C1.x'], 200.0, 1e-6)

        assert_near_equal(prob['G1.C1.y'], 400.0, 1e-6)

        # using absolute value will give us the value of the input C2.x, in its units of 'mm'
        assert_near_equal(prob['G1.C2.x'], np.ones(2) * 2000.0, 1e-6)

        assert_near_equal(prob['G1.C2.y'], np.ones(2) * 6000.0, 1e-6)

    def test_feature_get_set_with_units_prom_plus_explicit(self):

        prob = om.Problem()
        prob.model.add_subsystem('indeps', om.IndepVarComp('x', val=1.0, units='m'))
        G1 = prob.model.add_subsystem('G1', om.Group())
        G1.add_subsystem('C1', om.ExecComp('y=x*2.',
                                            x={'val': 1.0, 'units': 'cm'},
                                            y={'val': 0.0, 'units': 'cm'}),
                         promotes=['x'])
        G1.add_subsystem('C2', om.ExecComp('y=x*3.',
                                            x={'val': 1.0, 'units': 'mm'},
                                            y={'val': 0.0, 'units': 'mm'}),
                         promotes=['x'])

        # connect IVC to promoted inputs
        prob.model.connect('indeps.x', 'G1.x')

        # units and value to use for the _auto_ivc output are ambiguous.  This fixes that.
        G1.set_input_defaults('x', units='dm', val=1.0)

        prob.setup()

        prob['indeps.x'] = 2.0

        prob.run_model()

        assert_near_equal(prob['indeps.x'], 2.0, 1e-6)

        # using the promoted name of the inputs will give the value
        # in the units set in set_input_defaults, which is 'dm'
        assert_near_equal(prob['G1.x'], 20.0, 1e-6)

        # get value from lower level group
        assert_near_equal(G1.get_val('x'), 20.0, 1e-6)

        # using absolute value will give us the value of the input C1.x, in its units of 'inch'
        assert_near_equal(prob['G1.C1.x'], 200.0, 1e-6)

        # using absolute value will give us the value of the input C2.x, in its units of 'ft'
        assert_near_equal(prob['G1.C2.x'], 2000.0, 1e-6)

    def test_feature_get_set_with_units_prom_plus_explicit_err(self):

        prob = om.Problem()
        prob.model.add_subsystem('indeps', om.IndepVarComp('x', val=1.0, units='m'))
        G1 = prob.model.add_subsystem('G1', om.Group())
        G1.add_subsystem('C1', om.ExecComp('y=x*2.',
                                            x={'val': 1.0, 'units': 'cm'},
                                            y={'val': 0.0, 'units': 'cm'}),
                         promotes=['x'])
        G1.add_subsystem('C2', om.ExecComp('y=x*3.',
                                            x={'val': 1.0, 'units': 'mm'},
                                            y={'val': 0.0, 'units': 'mm'}),
                         promotes=['x'])

        # connect IVC to promoted inputs
        prob.model.connect('indeps.x', 'G1.x')

        prob.setup()

        prob['indeps.x'] = 2.0

        prob.run_model()

        assert_near_equal(prob['indeps.x'], 2.0, 1e-6)

        # using absolute value will give us the value of the input C1.x, in its units of 'inch'
        assert_near_equal(prob['G1.C1.x'], 200.0, 1e-6)

        # using absolute value will give us the value of the input C2.x, in its units of 'ft'
        assert_near_equal(prob['G1.C2.x'], 2000.0, 1e-6)

        # using the promoted name of the inputs will raise an exception because the two promoted
        # inputs have different units and set_input_defaults was not called to disambiguate.
        with self.assertRaises(RuntimeError) as cm:
            x = prob['G1.x']

        msg = "<model> <class Group>: The following inputs, ['G1.C1.x', 'G1.C2.x'], promoted to 'G1.x', are connected but their metadata entries ['units'] differ. Call <group>.set_input_defaults('x', units=?), where <group> is the Group named 'G1' to remove the ambiguity."
        self.assertEqual(cm.exception.args[0], msg)

    def test_get_set_with_units_error_messages(self):

        prob = om.Problem()
        prob.model.add_subsystem('comp', om.ExecComp('y=x+1.',
                                                     x={'val': 100.0, 'units': 'cm'},
                                                     y={'units': 'm'}))
        prob.model.add_subsystem('no_unit', om.ExecComp('y=x+1.', x={'val': 100.0}))

        prob.setup()
        prob.run_model()

        msg = "Can't express variable 'comp.x' with units of 'cm' in units of 'degK'."
        with self.assertRaisesRegex(TypeError, msg):
            prob.get_val('comp.x', 'degK')

        with self.assertRaisesRegex(TypeError, msg):
            prob.set_val('comp.x', 55.0, 'degK')

        msg = "Can't express variable 'no_unit.x' with units of 'None' in units of 'degK'."
        with self.assertRaisesRegex(TypeError, msg):
            prob.get_val('no_unit.x', 'degK')

        with self.assertRaisesRegex(TypeError, msg):
            prob.set_val('no_unit.x', 55.0, 'degK')

    def test_feature_get_set_with_units(self):

        prob = om.Problem()
        prob.model.add_subsystem('comp', om.ExecComp('y=x+1.',
                                                     x={'val': 100.0, 'units': 'cm'},
                                                     y={'units': 'm'}))

        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val('comp.x'), 100, 1e-6)
        assert_near_equal(prob.get_val('comp.x', 'm'), 1.0, 1e-6)
        prob.set_val('comp.x', 10.0, 'mm')
        assert_near_equal(prob.get_val('comp.x'), 1.0, 1e-6)
        assert_near_equal(prob.get_val('comp.x', 'm'), 1.0e-2, 1e-6)

    def test_feature_get_set_array_with_units(self):

        prob = om.Problem()
        prob.model.add_subsystem('comp', om.ExecComp('y=x+1.',
                                                     x={'val': np.array([100.0, 33.3]), 'units': 'cm'},
                                                     y={'shape': (2, ), 'units': 'm'}))

        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val('comp.x'), np.array([100, 33.3]), 1e-6)
        assert_near_equal(prob.get_val('comp.x', 'm'), np.array([1.0, 0.333]), 1e-6)
        assert_near_equal(prob.get_val('comp.x', 'km', indices=[0]), 0.001, 1e-6)

        prob.set_val('comp.x', 10.0, 'mm')
        assert_near_equal(prob.get_val('comp.x'), np.array([1.0, 1.0]), 1e-6)
        assert_near_equal(prob.get_val('comp.x', 'm', indices=0), 1.0e-2, 1e-6)

        prob.set_val('comp.x', 50.0, 'mm', indices=[1])
        assert_near_equal(prob.get_val('comp.x'), np.array([1.0, 5.0]), 1e-6)
        assert_near_equal(prob.get_val('comp.x', 'm', indices=1), 5.0e-2, 1e-6)

    def test_feature_get_set_array_with_slicer(self):

        prob = om.Problem()
        prob.model.add_subsystem('comp', om.ExecComp('y=x+1.',
                                                     x={'val': np.array([[1., 2.], [3., 4.]]), },
                                                     y={'shape': (2, 2), }))

        prob.setup()
        prob.run_model()

        assert_near_equal(prob.get_val('comp.x', indices=om.slicer[:, 0]), [1., 3.], 1e-6)
        assert_near_equal(prob.get_val('comp.x', indices=om.slicer[0, 1]), 2., 1e-6)
        assert_near_equal(prob.get_val('comp.x', indices=om.slicer[1, -1]), 4., 1e-6)

        prob.set_val('comp.x', [5., 6.], indices=om.slicer[:,0])
        assert_near_equal(prob.get_val('comp.x', indices=om.slicer[:, 0]), [5., 6.], 1e-6)
        prob.run_model()
        assert_near_equal(prob.get_val('comp.y', indices=om.slicer[:, 0]), [6., 7.], 1e-6)

    def test_feature_set_get_array(self):

        prob = om.Problem(model=SellarDerivatives())
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.setup()

        # default value from the class definition
        assert_near_equal(prob.get_val('x'), 1.0, 1e-6)

        prob.set_val('x', 2.75)
        assert_near_equal(prob.get_val('x'), 2.75, 1e-6)

        # default value from the class definition
        assert_near_equal(prob.get_val('z'), [5.0, 2.0], 1e-6)

        prob.set_val('z', [1.5, 1.5])
        assert_near_equal(prob.get_val('z'), [1.5, 1.5], 1e-6)

        prob.run_model()
        assert_near_equal(prob.get_val('y1'), 5.43379016853, 1e-6)
        assert_near_equal(prob.get_val('y2'), 5.33104915618, 1e-6)

        prob.set_val('z', np.array([2.5, 2.5])) # for convenience we convert the list to an array.
        assert_near_equal(prob.get_val('z'), [2.5, 2.5], 1e-6)

        prob.run_model()
        assert_near_equal(prob.get_val('y1'), 9.87161739688, 1e-6)
        assert_near_equal(prob.get_val('y2'), 8.14191301549, 1e-6)

    def test_feature_residuals(self):

        prob = om.Problem(model=SellarDerivatives())
        prob.model.nonlinear_solver = om.NonlinearBlockGS()

        prob.setup()

        prob.set_val('z', [1.5, 1.5])
        prob.run_model()

        inputs, outputs, residuals = prob.model.get_nonlinear_vectors()

        self.assertLess(residuals['y1'], 1e-6)
        self.assertLess(residuals['y2'], 1e-6)

    def test_setup_bad_mode(self):
        # Test error message when passing bad mode to setup.

        prob = om.Problem(name='foo')

        try:
            prob.setup(mode='junk')
        except ValueError as err:
            msg = "Problem foo: Unsupported mode: 'junk'. Use either 'fwd' or 'rev'."
            self.assertEqual(str(err), msg)
        else:
            self.fail('Expecting ValueError')

    def test_setup_bad_mode_direction_fwd(self):

        prob = om.Problem()
        prob.model.add_subsystem("indep", om.IndepVarComp("x", np.ones(99)))
        prob.model.add_subsystem("C1", om.ExecComp("y=2.0*x", x=np.zeros(10), y=np.zeros(10)))

        prob.model.connect("indep.x", "C1.x", src_indices=list(range(10)))

        prob.model.add_design_var("indep.x")
        prob.model.add_objective("C1.y")

        prob.setup(mode='fwd')

        msg = "Inefficient choice of derivative mode.  " \
              "You chose 'fwd' for a problem with 99 design variables and 10 " \
              "response variables (objectives and nonlinear constraints)."

        with assert_warning(DerivativesWarning, msg):
            prob.final_setup()

    def test_setup_bad_mode_direction_rev(self):

        prob = om.Problem()
        prob.model.add_subsystem("indep", om.IndepVarComp("x", np.ones(10)))
        prob.model.add_subsystem("C1", om.ExecComp("y=2.0*x", x=np.zeros(10), y=np.zeros(10)))
        prob.model.add_subsystem("C2", om.ExecComp("y=2.0*x", x=np.zeros(10), y=np.zeros(10)))

        prob.model.connect("indep.x", ["C1.x", "C2.x"])

        prob.model.add_design_var("indep.x")
        prob.model.add_constraint("C1.y")
        prob.model.add_constraint("C2.y")

        prob.setup(mode='rev')

        msg = "Inefficient choice of derivative mode.  " \
              "You chose 'rev' for a problem with 10 design variables and 20 " \
              "response variables (objectives and nonlinear constraints)."

        with assert_warning(DerivativesWarning, msg):
            prob.final_setup()

    def test_run_before_setup(self):
        # Test error message when running before setup.

        prob = om.Problem()

        msg = "Problem .*: The `setup` method must be called before `run_model`."
        with self.assertRaisesRegex(RuntimeError, msg):
            prob.run_model()

        msg = "Problem .*: The `setup` method must be called before `run_driver`."
        with self.assertRaisesRegex(RuntimeError, msg):
            prob.run_driver()

    def test_run_with_invalid_prefix(self):
        # Test error message when running with invalid prefix.

        prob = om.Problem()

        msg = "Problem .*: The 'case_prefix' argument should be a string."
        with self.assertRaisesRegex(TypeError, msg):
            prob.setup()
            prob.run_model(case_prefix=1234)
        with self.assertRaisesRegex(TypeError, msg):
            prob.setup()
            prob.run_driver(case_prefix=12.34)

    def test_args(self):
        # defaults
        prob = om.Problem()
        self.assertTrue(isinstance(prob.model, om.Group))
        self.assertTrue(isinstance(prob.driver, Driver))

        # model
        prob = om.Problem(SellarDerivatives())
        self.assertTrue(isinstance(prob.model, SellarDerivatives))
        self.assertTrue(isinstance(prob.driver, Driver))

        # driver
        prob = om.Problem(driver=om.ScipyOptimizeDriver())
        self.assertTrue(isinstance(prob.model, om.Group))
        self.assertTrue(isinstance(prob.driver, om.ScipyOptimizeDriver))

        # model and driver
        prob = om.Problem(model=SellarDerivatives(), driver=om.ScipyOptimizeDriver())
        self.assertTrue(isinstance(prob.model, SellarDerivatives))
        self.assertTrue(isinstance(prob.driver, om.ScipyOptimizeDriver))

        # invalid model
        msg = "Problem .*: The value provided for 'model' is not a valid System."
        with self.assertRaisesRegex(TypeError, msg) :
            prob = om.Problem(om.ScipyOptimizeDriver())

        # invalid driver
        msg = "Problem .*: The value provided for 'driver' is not a valid Driver."
        with self.assertRaisesRegex(TypeError, msg) :
            prob = om.Problem(driver=SellarDerivatives())

    def test_relevance(self):
        p = om.Problem()
        model = p.model

        model.add_subsystem("indep1", om.IndepVarComp('x', 1.0))
        G1 = model.add_subsystem('G1', om.Group())
        G1.add_subsystem('C1', om.ExecComp(['x=2.0*a', 'y=2.0*b', 'z=2.0*a']))
        G1.add_subsystem('C2', om.ExecComp(['x=2.0*a', 'y=2.0*b', 'z=2.0*b']))
        model.add_subsystem("C3", om.ExecComp(['x=2.0*a', 'y=2.0*b+3.0*c']))
        model.add_subsystem("C4", om.ExecComp(['x=2.0*a', 'y=2.0*b']))
        model.add_subsystem("indep2", om.IndepVarComp('x', 1.0))
        G2 = model.add_subsystem('G2', om.Group())
        G2.add_subsystem('C5', om.ExecComp(['x=2.0*a', 'y=2.0*b+3.0*c']))
        G2.add_subsystem('C6', om.ExecComp(['x=2.0*a', 'y=2.0*b+3.0*c']))
        G2.add_subsystem('C7', om.ExecComp(['x=2.0*a', 'y=2.0*b']))
        model.add_subsystem("C8", om.ExecComp(['y=1.5*a+2.0*b']))
        model.add_subsystem("Unconnected", om.ExecComp('y=99.*x'))

        model.connect('indep1.x', 'G1.C1.a')
        model.connect('indep2.x', 'G2.C6.a')
        model.connect('G1.C1.x', 'G1.C2.b')
        model.connect('G1.C2.z', 'C4.b')
        model.connect('G1.C1.z', ('C3.b', 'C3.c', 'G2.C5.a'))
        model.connect('C3.y', 'G2.C5.b')
        model.connect('C3.x', 'C4.a')
        model.connect('G2.C6.y', 'G2.C7.b')
        model.connect('G2.C5.x', 'C8.b')
        model.connect('G2.C7.x', 'C8.a')

        p.setup(check=False, mode='rev')

        dumb_meta = {'parallel_deriv_color': None}
        relevant = model.get_relevant_vars({'indep1.x': dumb_meta, 'indep2.x': dumb_meta},
                                           {'C8.y': dumb_meta, 'Unconnected.y': dumb_meta}, mode='rev')

        indep1_ins = set(['C3.b', 'C3.c', 'C8.b', 'G1.C1.a', 'G2.C5.a', 'G2.C5.b'])
        indep1_outs = set(['C3.y', 'C8.y', 'G1.C1.z', 'G2.C5.x', 'indep1.x'])
        indep1_sys = set(['C3', 'C8', 'G1.C1', 'G2.C5', 'indep1', 'G1', 'G2', ''])

        dct, systems = relevant['C8.y']['indep1.x']
        inputs = dct['input']
        outputs = dct['output']

        self.assertEqual(inputs, indep1_ins)
        self.assertEqual(outputs, indep1_outs)
        self.assertEqual(systems, indep1_sys)

        dct, systems = relevant['C8.y']['indep1.x']
        inputs = dct['input']
        outputs = dct['output']

        self.assertEqual(inputs, indep1_ins)
        self.assertEqual(outputs, indep1_outs)
        self.assertEqual(systems, indep1_sys)

        indep2_ins = set(['C8.a', 'G2.C6.a', 'G2.C7.b'])
        indep2_outs = set(['C8.y', 'G2.C6.y', 'G2.C7.x', 'indep2.x'])
        indep2_sys = set(['C8', 'G2.C6', 'G2.C7', 'indep2', 'G2', ''])

        dct, systems = relevant['C8.y']['indep2.x']
        inputs = dct['input']
        outputs = dct['output']

        self.assertEqual(inputs, indep2_ins)
        self.assertEqual(outputs, indep2_outs)
        self.assertEqual(systems, indep2_sys)

        dct, systems = relevant['C8.y']['indep2.x']
        inputs = dct['input']
        outputs = dct['output']

        self.assertEqual(inputs, indep2_ins)
        self.assertEqual(outputs, indep2_outs)
        self.assertEqual(systems, indep2_sys)

        dct, systems = relevant['C8.y']['@all']
        inputs = dct['input']
        outputs = dct['output']

        self.assertEqual(inputs, indep1_ins | indep2_ins)
        self.assertEqual(outputs, indep1_outs | indep2_outs)
        self.assertEqual(systems, indep1_sys | indep2_sys)

    def test_system_setup_and_configure(self):
        # Test that we can change solver settings on a subsystem in a system's setup method.
        # Also assures that highest system's settings take precedence.

        class ImplSimple(om.ImplicitComponent):

            def setup(self):
                self.add_input('a', val=1.)
                self.add_output('x', val=0.)

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['x'] = np.exp(outputs['x']) - \
                    inputs['a']**2 * outputs['x']**2

            def linearize(self, inputs, outputs, jacobian):
                jacobian['x', 'x'] = np.exp(outputs['x']) - \
                    2 * inputs['a']**2 * outputs['x']
                jacobian['x', 'a'] = -2 * inputs['a'] * outputs['x']**2

        class Sub(om.Group):

            def setup(self):
                self.add_subsystem('comp', ImplSimple())

                # This will not solve it
                self.nonlinear_solver = om.NonlinearBlockGS()

            def configure(self):
                # This will not solve it either.
                self.nonlinear_solver = om.NonlinearBlockGS()

        class Super(om.Group):

            def setup(self):
                self.add_subsystem('sub', Sub())

            def configure(self):
                # This will solve it.
                self.sub.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
                self.sub.linear_solver = om.ScipyKrylov()

        top = om.Problem(model=Super())

        top.setup()

        self.assertTrue(isinstance(top.model.sub.nonlinear_solver, om.NewtonSolver))
        self.assertTrue(isinstance(top.model.sub.linear_solver, om.ScipyKrylov))

    def test_post_setup_solver_configure(self):
        # Test that we can change solver settings after we have instantiated our model.

        class ImplSimple(om.ImplicitComponent):

            def setup(self):
                self.add_input('a', val=1.)
                self.add_output('x', val=0.)

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['x'] = np.exp(outputs['x']) - \
                    inputs['a']**2 * outputs['x']**2

            def linearize(self, inputs, outputs, jacobian):
                jacobian['x', 'x'] = np.exp(outputs['x']) - \
                    2 * inputs['a']**2 * outputs['x']
                jacobian['x', 'a'] = -2 * inputs['a'] * outputs['x']**2

        class Sub(om.Group):

            def setup(self):
                self.add_subsystem('comp', ImplSimple())

                # This solver will get over-ridden below
                self.nonlinear_solver = om.NonlinearBlockGS()

            def configure(self):
                # This solver will get over-ridden below
                self.nonlinear_solver = om.NonlinearBlockGS()

        class Super(om.Group):

            def setup(self):
                self.add_subsystem('sub', Sub())

        top = om.Problem(model=Super())

        top.setup()

        # These solvers override the ones set in the setup method of the 'sub' groups
        top.model.sub.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.sub.linear_solver = om.ScipyKrylov()

        self.assertTrue(isinstance(top.model.sub.nonlinear_solver, om.NewtonSolver))
        self.assertTrue(isinstance(top.model.sub.linear_solver, om.ScipyKrylov))

    def test_configure_add_indep_var(self):
        # add outputs to an IndepVarComp in Group configure

        class Model(om.Group):
            def initialize(self):
                self.options.declare('where_to_add', values=('setup', 'configure'))

            def setup(self):
                comp1 = self.add_subsystem('comp1', om.IndepVarComp())
                comp2 = self.add_subsystem('comp2', om.IndepVarComp())

                comp1.add_output('foo', val=1.0)
                comp2.add_output('foo', val=1.0)

                self.add_subsystem('comp3', om.ExecComp('y=a+b'))

                if self.options['where_to_add'] == 'setup':
                    comp1.add_output('a', val=2.0)
                    comp2.add_output('b', val=3.0)

                    self.connect('comp1.a', 'comp3.a')
                    self.connect('comp2.b', 'comp3.b')

            def configure(self):
                if self.options['where_to_add'] == 'configure':
                    self.comp1.add_output('a', val=2.0)
                    self.comp2.add_output('b', val=3.0)

                    self.connect('comp1.a', 'comp3.a')
                    self.connect('comp2.b', 'comp3.b')

        for where in ('setup', 'configure'):
            p = om.Problem(Model(where_to_add=where))
            p.setup()
            p.run_model()

            inputs = p.model.list_inputs(out_stream=None)
            self.assertEqual(sorted(inputs), [
                ('comp3.a', {'val': [2.]}),
                ('comp3.b', {'val': [3.]})
            ], "Inputs don't match when added in %s." % where)

            outputs = p.model.list_outputs(out_stream=None)
            self.assertEqual(sorted(outputs), [
                ('comp1.a',   {'val': [2.]}),
                ('comp1.foo', {'val': [1.]}),
                ('comp2.b',   {'val': [3.]}),
                ('comp2.foo', {'val': [1.]}),
                ('comp3.y',   {'val': [5.]})
            ], "Outputs don't match when added in %s." % where)

    def test_configure_add_input_output(self):
        # add inputs and outputs to an ExplicitComponent in Group configure

        class MyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('a', val=0.)
                self.add_output('a2', val=0.)

            def compute(self, inputs, outputs):
                outputs['a2'] = inputs['a'] * 2.
                if 'b' in inputs:
                    outputs['b2'] = inputs['b'] * 2.

        class Model(om.Group):
            def initialize(self):
                self.options.declare('add_b2', default=False)

            def setup(self):
                self.add_subsystem('indep', om.IndepVarComp(), promotes=['*'])
                self.add_subsystem('mcomp', MyComp(), promotes=['*'])

                self.add_subsystem('sub', om.Group(), promotes_inputs=['*'])
                self.sub.add_subsystem('mcomp', MyComp(), promotes=['*'])

                self.indep.add_output('a', val=2.0)

            def configure(self):
                if self.options['add_b2']:
                    self.indep.add_output('b', val=3.0)

                    self.mcomp.add_input('b', val=0.)
                    self.mcomp.add_output('b2', val=0.)

                    self.sub.mcomp.add_input('b', val=0.)
                    self.sub.mcomp.add_output('b2', val=0.)

        # add inputs/outputs in setup only
        p = om.Problem(Model(add_b2=False))
        p.setup()
        p.run_model()

        inputs = p.model.list_inputs(out_stream=None)
        self.assertEqual(sorted(inputs), [
            ('mcomp.a',     {'val': [2.]}),
            ('sub.mcomp.a', {'val': [2.]}),
        ])

        outputs = p.model.list_outputs(out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('indep.a',      {'val': [2.]}),
            ('mcomp.a2',     {'val': [4.]}),
            ('sub.mcomp.a2', {'val': [4.]}),
        ])

        # add inputs/outputs in configure
        p = om.Problem(Model(add_b2=True))
        p.setup()
        p.run_model()

        inputs = p.model.list_inputs(out_stream=None)
        self.assertEqual(sorted(inputs), [
            ('mcomp.a',     {'val': [2.]}),
            ('mcomp.b',     {'val': [3.]}),
            ('sub.mcomp.a', {'val': [2.]}),
            ('sub.mcomp.b', {'val': [3.]}),
        ])

        outputs= p.model.list_outputs(out_stream=None)
        self.assertEqual(sorted(outputs), [
            ('indep.a',      {'val': [2.]}),
            ('indep.b',      {'val': [3.]}),
            ('mcomp.a2',     {'val': [4.]}),
            ('mcomp.b2',     {'val': [6.]}),
            ('sub.mcomp.a2', {'val': [4.]}),
            ('sub.mcomp.b2', {'val': [6.]}),
        ])

    def test_feature_post_setup_solver_configure(self):

        class ImplSimple(om.ImplicitComponent):

            def setup(self):
                self.add_input('a', val=1.)
                self.add_output('x', val=0.)

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['x'] = np.exp(outputs['x']) - \
                    inputs['a']**2 * outputs['x']**2

            def linearize(self, inputs, outputs, jacobian):
                jacobian['x', 'x'] = np.exp(outputs['x']) - \
                    2 * inputs['a']**2 * outputs['x']
                jacobian['x', 'a'] = -2 * inputs['a'] * outputs['x']**2

        class Sub(om.Group):

            def setup(self):
                self.add_subsystem('comp', ImplSimple())

                # This will not solve it
                self.nonlinear_solver = om.NonlinearBlockGS()

            def configure(self):
                # This will not solve it either.
                self.nonlinear_solver = om.NonlinearBlockGS()

        class Super(om.Group):

            def setup(self):
                self.add_subsystem('sub', Sub())

        top = om.Problem(model=Super())

        top.setup()

        # This will solve it.
        top.model.sub.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        top.model.sub.linear_solver = om.ScipyKrylov()

        self.assertTrue(isinstance(top.model.sub.nonlinear_solver, om.NewtonSolver))
        self.assertTrue(isinstance(top.model.sub.linear_solver, om.ScipyKrylov))

    @hooks_active
    def test_post_final_setup_hook(self):
        def hook_func(prob):
            prob['p2.y'] = 5.0

        hooks._register_hook('final_setup', class_name='Problem', post=hook_func)
        try:
            prob = om.Problem()
            model = prob.model

            model.add_subsystem('p1', om.IndepVarComp('x', 3.0))
            model.add_subsystem('p2', om.IndepVarComp('y', -4.0))
            model.add_subsystem('comp', om.ExecComp("f_xy=2.0*x+3.0*y"))

            model.connect('p1.x', 'comp.x')
            model.connect('p2.y', 'comp.y')

            prob.setup()
            prob.run_model()

            assert_near_equal(prob['p2.y'], 5.0)
            assert_near_equal(prob['comp.f_xy'], 21.0)
        finally:
            hooks._unregister_hook('final_setup', class_name='Problem')
            hooks.use_hooks = False

    def test_list_problem_vars(self):
        model = SellarDerivatives()
        model.nonlinear_solver = om.NonlinearBlockGS()

        prob = om.Problem(model)
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.setup()
        prob.run_driver()

        # First, with no options
        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            prob.list_problem_vars()
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')
        self.assertEquals(output[1], r'Design Variables')
        self.assertRegex(output[5], r'^z +\|[0-9. e+-]+\| +2')
        self.assertEquals(output[9], r'Constraints')
        self.assertRegex(output[14], r'^con2 +\[[0-9. e+-]+\] +1')
        self.assertEquals(output[17], r'Objectives')
        self.assertRegex(output[21], r'^obj +\[[0-9. e+-]+\] +1')

        # With show_promoted_name=False
        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            prob.list_problem_vars(show_promoted_name=False)
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')
        self.assertRegex(output[5], r'^z +\|[0-9. e+-]+\| +2')
        self.assertRegex(output[14], r'^con_cmp2.con2 +\[[0-9. e+-]+\] +1')
        self.assertRegex(output[21], r'^obj_cmp.obj +\[[0-9. e+-]+\] +1')

        # With all the optional columns
        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            prob.list_problem_vars(
                desvar_opts=['lower', 'upper', 'ref', 'ref0',
                             'indices', 'adder', 'scaler',
                             'parallel_deriv_color',
                             'cache_linear_solution'],
                cons_opts=['lower', 'upper', 'equals', 'ref', 'ref0',
                           'indices', 'adder', 'scaler', 'linear',
                           'parallel_deriv_color',
                           'cache_linear_solution'],
                objs_opts=['ref', 'ref0',
                           'indices', 'adder', 'scaler',
                           'parallel_deriv_color',
                           'cache_linear_solution'],
            )
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')
        self.assertRegex(output[3],
                    r'^name\s+val\s+size\s+lower\s+upper\s+ref\s+ref0\s+'
                    r'indices\s+adder\s+scaler\s+parallel_deriv_color\s+'
                    r'cache_linear_solution')
        self.assertRegex(output[5],
                    r'^z\s+\|[0-9.e+-]+\|\s+2\s+\|10.0\|\s+\|[0-9.e+-]+\|\s+None\s+'
                    r'None\s+None\s+None\s+None\s+None\s+False')

        # With all the optional columns and print_arrays
        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            prob.list_problem_vars(print_arrays=True,
                                   desvar_opts=['lower', 'upper', 'ref', 'ref0',
                                                'indices', 'adder', 'scaler',
                                                'parallel_deriv_color',
                                                'cache_linear_solution'],
                                   cons_opts=['lower', 'upper', 'equals', 'ref', 'ref0',
                                              'indices', 'adder', 'scaler', 'linear',
                                              'parallel_deriv_color',
                                              'cache_linear_solution'],
                                   objs_opts=['ref', 'ref0',
                                              'indices', 'adder', 'scaler',
                                              'parallel_deriv_color',
                                              'cache_linear_solution'],
                                   )
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')
        self.assertRegex(output[6], r'^\s+val:')
        self.assertRegex(output[7], r'^\s+array+\(+\[[0-9., e+-]+\]+\)')
        self.assertRegex(output[9], r'^\s+lower:')
        self.assertRegex(output[10], r'^\s+array+\(+\[[0-9., e+-]+\]+\)')
        self.assertRegex(output[12], r'^\s+upper:')
        self.assertRegex(output[13], r'^\s+array+\(+\[[0-9., e+-]+\]+\)')

    def test_list_problem_vars_before_final_setup(self):
        prob = om.Problem()
        prob.model.add_subsystem('parab', Paraboloid(), promotes_inputs=['x', 'y'])
        prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])
        prob.model.set_input_defaults('x', 3.0)
        prob.model.set_input_defaults('y', -4.0)
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'COBYLA'
        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('parab.f_xy')
        prob.model.add_constraint('const.g', lower=0, upper=10.)
        prob.setup()

        msg = "Problem .*: Problem.list_problem_vars\(\) cannot be called before " \
                         "`Problem\.run_model\(\)`, `Problem\.run_driver\(\)`, or " \
                         "`Problem\.final_setup\(\)`\."
        with self.assertRaisesRegex(RuntimeError, msg):
            prob.list_problem_vars()

    def test_list_problem_w_multi_constraints(self):
        p = om.Problem()

        exec = om.ExecComp(['y = x**2',
                            'z = a + x**2'],
                        a={'shape': (1,)},
                        y={'shape': (101,)},
                        x={'shape': (101,)},
                        z={'shape': (101,)})

        p.model.add_subsystem('exec', exec)

        p.model.add_design_var('exec.a', lower=-1000, upper=1000)
        p.model.add_objective('exec.y', index=50)
        p.model.add_constraint('exec.z', indices=[0], equals=25)
        p.model.add_constraint('exec.z', indices=[-1], lower=20, alias="ALIAS_TEST")

        p.driver = om.ScipyOptimizeDriver()

        p.setup()

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_driver()

        # First, with no options
        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            p.list_problem_vars()
        finally:
            sys.stdout = stdout

        output = strout.getvalue().split('\n')
        self.assertTrue("ALIAS_TEST" in output[13])

    def test_constraint_alias_duplicate_errors(self):
        size = 7

        prob = om.Problem(name='constraint_alias_duplicate_errors')
        model = prob.model

        model.add_subsystem('comp1', om.ExecComp('f = x',
                                                f=np.ones((size, )),
                                                x=np.ones((size, ))),
                            promotes=['*'])
        model.add_subsystem('comp2', om.ExecComp('g = x',
                                                g=np.ones((size, )),
                                                x=np.ones((size, ))),
                            promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)

        model.add_constraint('f', indices=[1], flat_indices=True, lower=10.0)
        model.add_constraint('f', indices=[5], flat_indices=True, alias='g', lower=0.5)

        msg = "Constraint alias 'f' is a duplicate of an existing alias or variable name."
        with self.assertRaises(Exception) as cm:
            model.add_constraint('f', indices=[3], flat_indices=True, alias='f', lower=0.5)

        self.assertEqual(str(cm.exception), msg)

        msg = "\nCollected errors for problem 'constraint_alias_duplicate_errors':" + \
              "\n   <model> <class Group>: Constraint alias 'g' on 'comp1.f' is the same name as " + \
              "an existing variable."
        with self.assertRaises(Exception) as cm:
            prob.setup()

        self.assertEqual(str(cm.exception), msg)

    def test_constraint_ellipses_slice_as_indices(self):
        # this passes as long as no exceptions are raised
        p = om.Problem()

        exec_comp = p.model.add_subsystem('exec', om.ExecComp())

        exec_comp.add_expr('y = x**2', y={'shape': (10, 3)}, x={'shape': (10, 3)})

        p.model.add_constraint('exec.y', alias='y0', indices=om.slicer[0, ...], equals=0)
        p.model.add_constraint('exec.y', alias='yf', indices=om.slicer[1, ...], equals=0)

        p.setup()

        p.set_val('exec.x', np.random.random((10, 3)))

        p.run_model()

    def test_constraint_slice_with_negative_as_indices(self):
        # this passes as long as no exceptions are raised
        p = om.Problem()

        exec_comp = p.model.add_subsystem('exec', om.ExecComp())

        exec_comp.add_expr('y = x**2', y={'shape': (10, 3)}, x={'shape': (10, 3)})

        p.model.add_constraint('exec.y', alias='y0', indices=om.slicer[0, -1], equals=0)
        p.model.add_constraint('exec.y', alias='yf', indices=om.slicer[-1, 0], equals=0)

        p.setup()

        p.set_val('exec.x', np.random.random((10, 3)))

        p.run_model()

    def test_list_problem_vars_driver_scaling(self):
        model = SellarDerivatives()
        model.nonlinear_solver = om.NonlinearBlockGS()

        prob = om.Problem(model)
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9

        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                             upper=np.array([10.0, 10.0]), ref=1.5)
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj', ref=3.0)
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0, ref=2.0)

        prob.setup()
        prob.run_driver()

        # Driver Scaling
        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout

        try:
            prob.list_problem_vars()
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')

        self.assertTrue('1.31' in output[5]) # z
        self.assertTrue('-10.' in output[14]) # con
        self.assertTrue('1.06' in output[21]) # obj

        # Model Scaling
        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout

        try:
            prob.list_problem_vars(driver_scaling=False)
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')

        self.assertTrue('1.9' in output[5]) # z
        self.assertTrue('-20.' in output[14]) # con
        self.assertTrue('3.18' in output[21]) # obj

    def test_feature_list_problem_vars(self):

        prob = om.Problem(model=SellarDerivatives())
        model = prob.model
        model.nonlinear_solver = om.NonlinearBlockGS()

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.setup()
        prob.run_driver()

        prob.list_problem_vars(print_arrays=True,
                               desvar_opts=['lower', 'upper', 'ref', 'ref0',
                                            'indices', 'adder', 'scaler',
                                            'parallel_deriv_color'],
                               cons_opts=['lower', 'upper', 'equals', 'ref', 'ref0',
                                          'indices', 'adder', 'scaler', 'linear'],
                               objs_opts=['ref', 'ref0',
                                          'indices', 'adder', 'scaler',
                                          'parallel_deriv_color',
                                          'cache_linear_solution'])

    def test_error_msg_set_val_before_setup(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        msg = "Problem .*: 'x' Cannot call set_val before setup."
        with self.assertRaisesRegex(RuntimeError, msg):
            prob.set_val('x', 0.)

    def test_design_var_connected_to_output_as_input_err(self):
        prob = om.Problem(name='output_as_input_err')
        root = prob.model

        prob.driver = om.ScipyOptimizeDriver()

        root.add_subsystem('initial_comp', om.ExecComp(['x = 10']), promotes_outputs=['x'])

        outer_group = root.add_subsystem('outer_group', om.Group(), promotes_inputs=['x'])
        inner_group = outer_group.add_subsystem('inner_group', om.Group(), promotes_inputs=['x'])

        c1 = inner_group.add_subsystem('c1', om.ExecComp(['y = x * 2.0',  'z = x ** 2']),
                                       promotes_inputs=['x'])

        c1.add_design_var('x', lower=0, upper=5)

        with self.assertRaises(Exception) as cm:
            prob.setup()

        msg = "\nCollected errors for problem 'output_as_input_err':" + \
              "\n   <model> <class Group>: Design variable 'x' is connected to 'initial_comp.x', " + \
              "but 'initial_comp.x' is not an IndepVarComp or ImplicitComp output."
        self.assertEqual(str(cm.exception), msg)

    def test_design_var_connected_to_output(self):
        prob = om.Problem()
        root = prob.model

        prob.driver = om.ScipyOptimizeDriver()

        root.add_subsystem('ivc', om.IndepVarComp('x', val=10.))

        outer_group = root.add_subsystem('outer_group', om.Group(), promotes_inputs=['x'])
        inner_group = outer_group.add_subsystem('inner_group', om.Group(), promotes_inputs=['x'])

        c1 = inner_group.add_subsystem('c1', om.ExecComp(['y = x * 2.0',  'z = x ** 2']),
                                       promotes_inputs=['x'])

        c1.add_design_var('x', lower=0, upper=5)

        try:
            prob.setup()
        except RuntimeError:
            self.fail("'setup raised RuntimeError unexpectedly")

        prob = om.Problem()
        root = prob.model

        root.add_subsystem('ivc', om.IndepVarComp('x', val=10.))

        outer_group = root.add_subsystem('outer_group', om.Group(), promotes_inputs=['x'])
        inner_group = outer_group.add_subsystem('inner_group', om.Group(), promotes_inputs=['x'])

        c1 = inner_group.add_subsystem('c1', om.ExecComp(['y = x * 2.0',  'z = x ** 2']),
                                       promotes_inputs=['x'])

        c1.add_design_var('x', lower=0, upper=5)
        root.connect('ivc.x', 'x')

        try:
            prob.setup()
        except RuntimeError:
            self.fail("'setup raised RuntimeError unexpectedly")


@use_tempdirs
class RelevanceTestCase(unittest.TestCase):
    def _setup_relevance_problem(self):
        p = om.Problem()

        model = p.model
        indeps = model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('a')
        indeps.add_output('b')
        indeps.add_output('c')

        model.add_subsystem('C1', MultComp(2.))
        model.add_subsystem('C2', MultComp(3.))
        model.add_subsystem('C3', MultComp(5.))
        model.add_subsystem('C4', MultComp(7.))
        model.add_subsystem('C5', MultComp(9.))
        model.add_subsystem('C6', MultComp(11.))

        model.connect('indeps.a', 'C1.x')
        model.connect('indeps.b', ['C1.y', 'C2.x'])
        model.connect('indeps.c', 'C2.y')

        model.connect('C1.fxy', 'C3.x')
        model.connect('C2.fxy', ['C3.y', 'C4.x'])
        model.connect('C3.fxy', 'C5.x')
        model.connect('C4.fxy', 'C6.x')

        return p

    def _setup_relevance_problem_w_cycle(self):
        p = self._setup_relevance_problem()
        p.model.connect('C5.fxy', 'C4.y')
        p.model.connect('C6.fxy', 'C5.y')
        return p

    def _finish_setup_and_check(self, p, expected):
        p.setup()

        p['indeps.a'] = 2.
        p['indeps.b'] = 3.
        p['indeps.c'] = 4.
        p['C4.y'] = 1.
        p['C5.y'] = 1.
        p['C6.y'] = 1.

        p.run_model()

        p.run_driver()

        allcomps = [getattr(p.model, f"C{i}") for i in range(1, 7)]
        ran_linearize = [c.name for c in allcomps if c._counts['_linearize'] > 0]
        ran_compute_partials = [c.name for c in allcomps if c._counts['_compute_partials_wrapper'] > 0]
        ran_solve_linear = [c.name for c in allcomps if c._counts['_solve_linear'] > 0]

        self.assertEqual(ran_linearize, expected)
        self.assertEqual(ran_compute_partials, expected)
        self.assertEqual(ran_solve_linear, expected)

    def test_relevance(self):
        p = self._setup_relevance_problem()

        p.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9, optimizer='SLSQP')
        p.model.add_design_var('indeps.b', lower=-50., upper=50.)
        p.model.add_objective('C6.fxy')
        p.model.add_constraint('C4.fxy', upper=1000.)

        self._finish_setup_and_check(p, ['C2', 'C4', 'C6'])

    def test_relevance2(self):
        p = self._setup_relevance_problem()

        p.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9, optimizer='SLSQP')
        p.model.add_design_var('indeps.a', lower=-50., upper=50.)
        p.model.add_objective('C5.fxy')
        p.model.add_constraint('C3.fxy', upper=1000.)

        self._finish_setup_and_check(p, ['C1', 'C3', 'C5'])

    def test_relevance3(self):
        p = self._setup_relevance_problem()

        p.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9, optimizer='SLSQP')
        p.model.add_design_var('indeps.c', lower=-50., upper=50.)
        p.model.add_objective('C5.fxy')
        p.model.add_constraint('C6.fxy', upper=1000.)

        self._finish_setup_and_check(p, ['C2', 'C3', 'C4', 'C5', 'C6'])

    def test_relevance4(self):
        p = self._setup_relevance_problem_w_cycle()

        p.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9, optimizer='SLSQP')
        p.model.add_design_var('indeps.a', lower=-50., upper=50.)
        p.model.add_objective('C5.fxy')
        p.model.add_constraint('C3.fxy', upper=1000.)

        self._finish_setup_and_check(p, ['C1', 'C3', 'C4', 'C5', 'C6'])

    def test_relevance5(self):
        p = self._setup_relevance_problem_w_cycle()

        p.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9, optimizer='SLSQP')
        p.model.add_design_var('indeps.c', lower=-50., upper=50.)
        p.model.add_objective('C5.fxy')
        p.model.add_constraint('C6.fxy', upper=1e22)

        self._finish_setup_and_check(p, ['C2', 'C3', 'C4', 'C5', 'C6'])

class NestedProblemTestCase(unittest.TestCase):

    def test_nested_prob(self):

        class _ProblemSolver(om.NonlinearRunOnce):
            def solve(self):
                # create a simple subproblem and run it to test for global solver_info bug
                p = om.Problem()
                p.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
                p.model.add_subsystem('comp', om.ExecComp('y=2*x'))
                p.model.connect('indep.x', 'comp.x')
                p.setup()
                p.run_model()

                return super().solve()

        p = om.Problem()
        p.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
        G = p.model.add_subsystem('G', om.Group())
        G.add_subsystem('comp', om.ExecComp('y=2*x'))
        G.nonlinear_solver = _ProblemSolver()
        p.model.connect('indep.x', 'G.comp.x')
        p.setup()
        p.run_model()

    def test_cs_across_nested(self):

        class NestedAnalysis(om.ExplicitComponent):

            def __init__(self):
                super().__init__()
                self._problem = None

            def setup(self):
                self.add_input('x', val=0.0)
                self.add_input('y', val=0.0)

                self.add_output('f_xy', val=0.0)

                # Setup sub-problem
                self._problem = prob = om.Problem()
                model = prob.model
                model.add_subsystem('parab', Paraboloid(), promotes=['*'])
                prob.setup(force_alloc_complex=True)

            def setup_partials(self):
                self.declare_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                prob = self._problem
                under_cs = self.under_complex_step

                if under_cs:
                    prob.set_complex_step_mode(True)

                # Set inputs
                prob.set_val('x', inputs['x'])
                prob.set_val('y', inputs['y'])

                # Run model
                prob.run_model()

                # Extract outputs
                outputs['f_xy'] = prob.get_val('f_xy')

                if under_cs:
                    prob.set_complex_step_mode(False)

            def compute_partials(self, inputs, partials):
                totals = self._problem.compute_totals(of='f_xy', wrt=['x', 'y'])
                partials['f_xy', 'x'] = totals['f_xy', 'x']
                partials['f_xy', 'y'] = totals['f_xy', 'y']

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('nested', NestedAnalysis(), promotes=['*'])

        prob.setup(force_alloc_complex=True)

        prob.set_val('x', 3.5)
        prob.set_val('y', 1.5)

        prob.run_model()

        totals = prob.check_totals(of='f_xy', wrt=['x', 'y'], method='cs', out_stream=None)
        for key, val in totals.items():
            assert_near_equal(val['rel error'][0], 0.0, 1e-12)

    def test_nested_prob_default_naming(self):
        import openmdao.core.problem
        class _ProblemSolver(om.NonlinearRunOnce):

            def __init__(self, prob_name=None):
                super(_ProblemSolver, self).__init__()
                self.prob_name = prob_name
                self._problem = None

            def solve(self):
                # create a simple subproblem and run it to test for global solver_info bug
                p = om.Problem(name=self.prob_name, reports=False)
                self._problem = p
                p.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
                p.model.add_subsystem('comp', om.ExecComp('y=2*x'))
                p.model.connect('indep.x', 'comp.x')
                p.setup()
                p.run_model()

                print(p.msginfo)

                return super().solve()

        # Initially use the default names
        openmdao.core.problem._clear_problem_names()  # need to reset these to simulate separate runs
        p = om.Problem(reports=False)
        p.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
        G = p.model.add_subsystem('G', om.Group())
        G.add_subsystem('comp', om.ExecComp('y=2*x'))
        G.nonlinear_solver = _ProblemSolver()
        p.model.connect('indep.x', 'G.comp.x')
        p.setup()
        p.run_model()  # need to do run_model in this test so sub problem is created

        defname = _default_prob_name()

        self.assertEqual(p._get_inst_id(), defname)
        self.assertEqual(G.nonlinear_solver._problem._get_inst_id(), defname + '2')

        # If the second Problem uses the default name of the first
        openmdao.core.problem._clear_problem_names()  # need to reset these to simulate separate runs
        p = om.Problem(reports=False)
        p.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
        G = p.model.add_subsystem('G', om.Group())
        G.add_subsystem('comp', om.ExecComp('y=2*x'))
        G.nonlinear_solver = _ProblemSolver(prob_name=defname)
        p.model.connect('indep.x', 'G.comp.x')
        p.setup()

        with self.assertRaises(Exception) as context:
            p.run_model()
        self.assertEqual(str(context.exception), f"The problem name '{defname}' already exists")

        # If the first Problem uses the default name of 'problem2'
        openmdao.core.problem._clear_problem_names()  # need to reset these to simulate separate runs
        p = om.Problem(name=defname + '2', reports=False)
        p.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
        G = p.model.add_subsystem('G', om.Group())
        G.add_subsystem('comp', om.ExecComp('y=2*x'))
        G.nonlinear_solver = _ProblemSolver()
        p.model.connect('indep.x', 'G.comp.x')
        p.setup()
        p.run_model()

        self.assertEqual(p._get_inst_id(), defname + '2')
        self.assertEqual(G.nonlinear_solver._problem._get_inst_id(),  defname + '2.1')


class SystemInTwoProblemsTestCase(unittest.TestCase):
    def test_2problems(self):
        prob = om.Problem(reports=False)
        G1 = prob.model.add_subsystem("G1", om.Group())
        G2 = G1.add_subsystem('G2', om.Group(), promotes_inputs=['x'])
        G2.add_subsystem('C1', om.ExecComp('y = 2 * x'), promotes_inputs=['x'])
        G2.add_subsystem('C2', om.ExecComp('y = 3 * x'), promotes_inputs=['x'])

        prob.setup()
        prob.run_model()

        # 2nd problem
        prob = om.Problem(reports=False)
        prob.model = G2

        prob.setup()
        prob.run_model()

        np.testing.assert_allclose(prob['C1.y'], 2.0)
        np.testing.assert_allclose(prob['C2.y'], 3.0)


if __name__ == "__main__":
    unittest.main()
