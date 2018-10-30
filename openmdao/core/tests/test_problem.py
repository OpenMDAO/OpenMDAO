""" Unit tests for the problem interface."""

import sys
import unittest
import warnings
from six import assertRaisesRegex, StringIO, assertRegex

import numpy as np

from openmdao.core.group import get_relevant_vars
from openmdao.core.driver import Driver
from openmdao.api import Problem, IndepVarComp, NonlinearBlockGS, ScipyOptimizeDriver, \
    ExecComp, Group, NewtonSolver, ImplicitComponent, ScipyKrylov, ExplicitComponent, NonlinearRunOnce
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDerivatives


class TestProblem(unittest.TestCase):

    def test_feature_simple_run_once_no_promote(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 3.0))
        model.add_subsystem('p2', IndepVarComp('y', -4.0))
        model.add_subsystem('comp', Paraboloid())

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['comp.f_xy'], -15.0)

    def test_feature_simple_run_once_input_input(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 3.0))

        # promote the two inputs to the same name
        model.add_subsystem('comp1', Paraboloid(), promotes_inputs=['x'])
        model.add_subsystem('comp2', Paraboloid(), promotes_inputs=['x'])

        # connect the source to the common name
        model.connect('p1.x', 'x')

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['comp1.f_xy'], 13.0)
        assert_rel_error(self, prob['comp2.f_xy'], 13.0)

    def test_feature_simple_run_once_compute_totals(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 3.0))
        model.add_subsystem('p2', IndepVarComp('y', -4.0))
        model.add_subsystem('comp', Paraboloid())

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        prob.setup()
        prob.run_model()

        totals = prob.compute_totals(of=['comp.f_xy'], wrt=['p1.x', 'p2.y'])
        assert_rel_error(self, totals[('comp.f_xy', 'p1.x')][0][0], -4.0)
        assert_rel_error(self, totals[('comp.f_xy', 'p2.y')][0][0], 3.0)

        totals = prob.compute_totals(of=['comp.f_xy'], wrt=['p1.x', 'p2.y'], return_format='dict')
        assert_rel_error(self, totals['comp.f_xy']['p1.x'][0][0], -4.0)
        assert_rel_error(self, totals['comp.f_xy']['p2.y'][0][0], 3.0)

    def test_feature_simple_run_once_compute_totals_scaled(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 3.0))
        model.add_subsystem('p2', IndepVarComp('y', -4.0))
        model.add_subsystem('comp', Paraboloid())

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        model.add_design_var('p1.x', 3.0, ref0=50.0)
        model.add_design_var('p2.y', -4.0)
        model.add_objective('comp.f_xy')

        prob.setup()
        prob.run_model()

        totals = prob.compute_totals(of=['comp.f_xy'], wrt=['p1.x', 'p2.y'], driver_scaling=True)
        assert_rel_error(self, totals[('comp.f_xy', 'p1.x')][0][0], 196.0)
        assert_rel_error(self, totals[('comp.f_xy', 'p2.y')][0][0], 3.0)

    def test_feature_simple_run_once_set_deriv_mode(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 3.0))
        model.add_subsystem('p2', IndepVarComp('y', -4.0))
        model.add_subsystem('comp', Paraboloid())

        model.connect('p1.x', 'comp.x')
        model.connect('p2.y', 'comp.y')

        prob.setup(mode='rev')
        # prob.setup(mode='fwd')
        prob.run_model()

        assert_rel_error(self, prob['comp.f_xy'], -15.0)

        prob.compute_totals(of=['comp.f_xy'], wrt=['p1.x', 'p2.y'])

    def test_compute_totals_cleanup(self):
        p = Problem()
        model = p.model
        model.add_subsystem('indeps1', IndepVarComp('x', np.ones(5)))
        model.add_subsystem('indeps2', IndepVarComp('x', np.ones(3)))

        model.add_subsystem('MP1', ExecComp('y=7*x', x=np.zeros(5), y=np.zeros(5)))
        model.add_subsystem('MP2', ExecComp('y=-3*x', x=np.zeros(3), y=np.zeros(3)))

        model.add_design_var('indeps1.x')
        model.add_design_var('indeps2.x')

        model.add_constraint('MP1.y')
        model.add_constraint('MP2.y')

        model.connect('indeps1.x', 'MP1.x')
        model.connect('indeps2.x', 'MP2.x')

        p.setup(mode='rev')
        p.run_model()

        J = p.compute_totals()
        assert_rel_error(self, J[('MP1.y', 'indeps1.x')], np.eye(5)*7., 1e-10)
        assert_rel_error(self, J[('MP2.y', 'indeps2.x')], np.eye(3)*-3., 1e-10)
        # before the bug fix, the following two derivs contained nonzero values even
        # though the variables involved were not dependent on each other.
        assert_rel_error(self, J[('MP2.y', 'indeps1.x')], np.zeros((3,5)), 1e-10)
        assert_rel_error(self, J[('MP1.y', 'indeps2.x')], np.zeros((5,3)), 1e-10)

    def test_set_2d_array(self):
        import numpy as np

        from openmdao.api import Problem, IndepVarComp, Group

        prob = Problem()
        model = prob.model
        model.add_subsystem(name='indeps',
                            subsys=IndepVarComp(name='X_c', shape=(3, 1)))
        prob.setup()

        new_val = -5*np.ones((3, 1))
        prob['indeps.X_c'] = new_val
        prob.final_setup()

        assert_rel_error(self, prob['indeps.X_c'], new_val, 1e-10)

        new_val = 2.5*np.ones(3)
        prob['indeps.X_c'][:, 0] = new_val
        prob.final_setup()

        assert_rel_error(self, prob['indeps.X_c'], new_val.reshape((3, 1)), 1e-10)
        assert_rel_error(self, prob['indeps.X_c'][:, 0], new_val, 1e-10)

    def test_set_checks_shape(self):

        model = Group()

        indep = model.add_subsystem('indep', IndepVarComp())
        indep.add_output('num')
        indep.add_output('arr', shape=(10, 1))

        prob = Problem(model)
        prob.setup()

        msg = "Incompatible shape for '.*': Expected (.*) but got (.*)"

        # check valid scalar value
        new_val = -10.
        prob['indep.num'] = new_val
        assert_rel_error(self, prob['indep.num'], new_val, 1e-10)

        # check bad scalar value
        bad_val = -10*np.ones((10))
        prob['indep.num'] = bad_val
        with assertRaisesRegex(self, ValueError, msg):
            prob.final_setup()
        prob._initial_condition_cache = {}

        # check assign scalar to array
        arr_val = new_val*np.ones((10, 1))
        prob['indep.arr'] = new_val
        prob.final_setup()
        assert_rel_error(self, prob['indep.arr'], arr_val, 1e-10)

        # check valid array value
        new_val = -10*np.ones((10, 1))
        prob['indep.arr'] = new_val
        assert_rel_error(self, prob['indep.arr'], new_val, 1e-10)

        # check bad array value
        bad_val = -10*np.ones((10))
        with assertRaisesRegex(self, ValueError, msg):
            prob['indep.arr'] = bad_val

        # check valid list value
        new_val = new_val.tolist()
        prob['indep.arr'] = new_val
        assert_rel_error(self, prob['indep.arr'], new_val, 1e-10)

        # check bad list value
        bad_val = bad_val.tolist()
        with assertRaisesRegex(self, ValueError, msg):
            prob['indep.arr'] = bad_val

    def test_compute_totals_basic(self):
        # Basic test for the method using default solvers on simple model.

        prob = Problem()
        model = prob.model
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, derivs['f_xy', 'x'], [[-6.0]], 1e-6)
        assert_rel_error(self, derivs['f_xy', 'y'], [[8.0]], 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_totals(of=of, wrt=wrt)

        assert_rel_error(self, derivs['f_xy', 'x'], [[-6.0]], 1e-6)
        assert_rel_error(self, derivs['f_xy', 'y'], [[8.0]], 1e-6)

    def test_compute_totals_basic_return_dict(self):
        # Make sure 'dict' return_format works.

        prob = Problem()
        model = prob.model
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_rel_error(self, derivs['f_xy']['x'], [[-6.0]], 1e-6)
        assert_rel_error(self, derivs['f_xy']['y'], [[8.0]], 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_rel_error(self, derivs['f_xy']['x'], [[-6.0]], 1e-6)
        assert_rel_error(self, derivs['f_xy']['y'], [[8.0]], 1e-6)

    def test_compute_totals_no_args_no_desvar(self):
        p = Problem()

        dv = p.model.add_subsystem('des_vars', IndepVarComp())
        dv.add_output('x', val=2.)

        p.model.add_subsystem('calc', ExecComp('y=2*x'))

        p.model.connect('des_vars.x', 'calc.x')

        p.model.add_objective('calc.y')

        p.setup()
        p.run_model()

        with self.assertRaises(RuntimeError) as cm:
            p.compute_totals()

        self.assertEqual(str(cm.exception),
                         "Driver is not providing any design variables for compute_totals.")

    def test_compute_totals_no_args_no_response(self):
        p = Problem()

        dv = p.model.add_subsystem('des_vars', IndepVarComp())
        dv.add_output('x', val=2.)

        p.model.add_subsystem('calc', ExecComp('y=2*x'))

        p.model.connect('des_vars.x', 'calc.x')

        p.model.add_design_var('des_vars.x')

        p.setup()
        p.run_model()

        with self.assertRaises(RuntimeError) as cm:
            p.compute_totals()

        self.assertEqual(str(cm.exception),
                         "Driver is not providing any response variables for compute_totals.")

    def test_compute_totals_no_args(self):
        p = Problem()

        dv = p.model.add_subsystem('des_vars', IndepVarComp())
        dv.add_output('x', val=2.)

        p.model.add_subsystem('calc', ExecComp('y=2*x'))

        p.model.connect('des_vars.x', 'calc.x')

        p.model.add_design_var('des_vars.x')
        p.model.add_objective('calc.y')

        p.setup()
        p.run_model()

        derivs = p.compute_totals()

        assert_rel_error(self, derivs['calc.y', 'des_vars.x'], [[2.0]], 1e-6)

    def test_compute_totals_no_args_promoted(self):
        p = Problem()

        dv = p.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        dv.add_output('x', val=2.)

        p.model.add_subsystem('calc', ExecComp('y=2*x'), promotes=['*'])

        p.model.add_design_var('x')
        p.model.add_objective('y')

        p.setup()
        p.run_model()

        derivs = p.compute_totals()

        assert_rel_error(self, derivs['calc.y', 'des_vars.x'], [[2.0]], 1e-6)

    def test_feature_set_indeps(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = Problem()

        model = prob.model
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        prob.setup()

        prob['x'] = 2.
        prob['y'] = 10.
        prob.run_model()
        assert_rel_error(self, prob['f_xy'], 214.0, 1e-6)

    def test_feature_basic_setup(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = Problem()
        model = prob.model
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        prob.setup()

        prob['x'] = 2.
        prob['y'] = 10.
        prob.run_model()
        assert_rel_error(self, prob['f_xy'], 214.0, 1e-6)

        prob['x'] = 0.
        prob['y'] = 0.
        prob.run_model()
        assert_rel_error(self, prob['f_xy'], 22.0, 1e-6)

        # skip the setup error checking
        prob.setup(check=False)
        prob['x'] = 4
        prob['y'] = 8.

        prob.run_model()
        assert_rel_error(self, prob['f_xy'], 174.0, 1e-6)

    def test_feature_petsc_setup(self):
        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = Problem()
        model = prob.model
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        # PETScVectors will be used automatically where needed. No need to set manually.
        prob.setup()
        prob['x'] = 2.
        prob['y'] = 10.

        prob.run_model()
        assert_rel_error(self, prob['f_xy'], 214.0, 1e-6)

    def test_feature_check_totals_manual(self):
        from openmdao.api import Problem, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.setup()
        prob.run_model()

        # manually specify which derivatives to check
        prob.check_totals(of=['obj', 'con1'], wrt=['x', 'z'])

    def test_feature_check_totals_from_driver_compact(self):
        from openmdao.api import Problem, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

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
        from openmdao.api import Problem, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

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
        from openmdao.api import Problem, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

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
        from openmdao.api import Problem, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

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
        from openmdao.api import Problem, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nonlinear_solver = NonlinearBlockGS()

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

        class SimpleComp(ExplicitComponent):

            def setup(self):
                self.add_input('x', val=1.0)
                self.add_output('y', val=1.0)

                self.declare_partials(of='y', wrt='x')

                if not self.force_alloc_complex:
                    raise RuntimeError('force_alloc_complex not set in component.')

            def compute(self, inputs, outputs):
                outputs['y'] = 3.0*inputs['x']

                if np.iscomplex(inputs._data[0]) and not self.under_complex_step:
                    raise RuntimeError('under_complex_step not set in component.')

            def compute_partials(self, inputs, partials):
                partials['y', 'x'] = 3.


        prob = Problem()
        prob.model.add_subsystem('px', IndepVarComp('x', 2.0))
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
        from openmdao.api import Problem, ExplicitComponent

        class SimpleComp(ExplicitComponent):

            def setup(self):
                self.add_input('x', val=1.0)
                self.add_output('y', val=1.0)

                self.declare_partials(of='y', wrt='x')

                if self.force_alloc_complex:
                    print("Vectors allocated for complex step.")

            def compute(self, inputs, outputs):
                outputs['y'] = 3.0*inputs['x']

            def compute_partials(self, inputs, partials):
                partials['y', 'x'] = 3.


        prob = Problem()
        prob.model.add_subsystem('px', IndepVarComp('x', val=1.0))
        prob.model.add_subsystem('comp', SimpleComp())
        prob.model.connect('px.x', 'comp.x')

        prob.model.add_design_var('px.x', lower=-100, upper=100)
        prob.model.add_objective('comp.y')

        prob.setup(force_alloc_complex=True)

        prob.run_model()

        prob.check_totals(method='cs')

    def test_feature_run_driver(self):
        import numpy as np

        from openmdao.api import Problem, NonlinearBlockGS, ScipyOptimizeDriver
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem(model=SellarDerivatives())
        model = prob.model
        model.nonlinear_solver = NonlinearBlockGS()

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.setup()
        prob.run_driver()

        assert_rel_error(self, prob['x'], 0.0, 1e-5)
        assert_rel_error(self, prob['y1'], 3.160000, 1e-2)
        assert_rel_error(self, prob['y2'], 3.755278, 1e-2)
        assert_rel_error(self, prob['z'], [1.977639, 0.000000], 1e-2)
        assert_rel_error(self, prob['obj'], 3.18339395, 1e-2)

    def test_feature_promoted_sellar_set_get_outputs(self):
        from openmdao.api import Problem, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem(model=SellarDerivatives())
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.setup()

        prob['x'] = 2.75

        prob.run_model()

        assert_rel_error(self, prob['x'], 2.75, 1e-6)

        assert_rel_error(self, prob['y1'], 27.3049178437, 1e-6)

    def test_feature_not_promoted_sellar_set_get_outputs(self):
        from openmdao.api import Problem, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDerivativesConnected

        prob = Problem(model= SellarDerivativesConnected())
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.setup()

        prob['px.x'] = 2.75

        prob.run_model()

        assert_rel_error(self, prob['px.x'], 2.75, 1e-6)

        assert_rel_error(self, prob['d1.y1'], 27.3049178437, 1e-6)

    def test_feature_promoted_sellar_set_get_inputs(self):
        from openmdao.api import Problem, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem(model=SellarDerivatives())
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.setup()

        prob['x'] = 2.75

        prob.run_model()

        assert_rel_error(self, prob['x'], 2.75, 1e-6)

        # the output variable, referenced by the promoted name
        assert_rel_error(self, prob['y1'], 27.3049178437, 1e-6)
        # the connected input variable, referenced by the absolute path
        assert_rel_error(self, prob['d2.y1'], 27.3049178437, 1e-6)

    def test_get_set_with_units_exhaustive(self):
        from openmdao.api import Problem, ExecComp

        prob = Problem()
        prob.model.add_subsystem('comp', ExecComp('y=x-25.',
                                                  x={'value': 77.0, 'units': 'degF'},
                                                  y={'value': 0.0, 'units': 'degC'}))
        prob.model.add_subsystem('prom', ExecComp('yy=xx-25.',
                                                  xx={'value': 77.0, 'units': 'degF'},
                                                  yy={'value': 0.0, 'units': 'degC'}),
                                 promotes=['xx', 'yy'])
        prob.model.add_subsystem('acomp', ExecComp('y=x-25.',
                                                   x={'value': np.array([77.0, 95.0]), 'units': 'degF'},
                                                   y={'value': 0.0, 'units': 'degC'}))
        prob.model.add_subsystem('aprom', ExecComp('ayy=axx-25.',
                                                   axx={'value': np.array([77.0, 95.0]), 'units': 'degF'},
                                                   ayy={'value': 0.0, 'units': 'degC'}),
                                 promotes=['axx', 'ayy'])

        prob.setup(check=False)

        # Make sure everything works before final setup with caching system.

        # Gets

        assert_rel_error(self, prob.get_val('comp.x'), 77.0, 1e-6)
        assert_rel_error(self, prob.get_val('comp.x', 'degC'), 25.0, 1e-6)
        assert_rel_error(self, prob.get_val('comp.y'), 0.0, 1e-6)
        assert_rel_error(self, prob.get_val('comp.y', 'degF'), 32.0, 1e-6)

        assert_rel_error(self, prob.get_val('xx'), 77.0, 1e-6)
        assert_rel_error(self, prob.get_val('xx', 'degC'), 25.0, 1e-6)
        assert_rel_error(self, prob.get_val('yy'), 0.0, 1e-6)
        assert_rel_error(self, prob.get_val('yy', 'degF'), 32.0, 1e-6)

        assert_rel_error(self, prob.get_val('acomp.x', indices=0), 77.0, 1e-6)
        assert_rel_error(self, prob.get_val('acomp.x', indices=[1]), 95.0, 1e-6)
        assert_rel_error(self, prob.get_val('acomp.x', 'degC', indices=[0]), 25.0, 1e-6)
        assert_rel_error(self, prob.get_val('acomp.x', 'degC', indices=1), 35.0, 1e-6)
        assert_rel_error(self, prob.get_val('acomp.y', indices=0), 0.0, 1e-6)
        assert_rel_error(self, prob.get_val('acomp.y', 'degF', indices=0), 32.0, 1e-6)

        assert_rel_error(self, prob.get_val('axx', indices=0), 77.0, 1e-6)
        assert_rel_error(self, prob.get_val('axx', indices=1), 95.0, 1e-6)
        assert_rel_error(self, prob.get_val('axx', 'degC', indices=0), 25.0, 1e-6)
        assert_rel_error(self, prob.get_val('axx', 'degC', indices=np.array([1])), 35.0, 1e-6)
        assert_rel_error(self, prob.get_val('ayy', indices=0), 0.0, 1e-6)
        assert_rel_error(self, prob.get_val('ayy', 'degF', indices=0), 32.0, 1e-6)

        # Sets

        prob.set_val('comp.x', 30.0, 'degC')
        assert_rel_error(self, prob['comp.x'], 86.0, 1e-6)
        assert_rel_error(self, prob.get_val('comp.x', 'degC'), 30.0, 1e-6)

        prob.set_val('xx', 30.0, 'degC')
        assert_rel_error(self, prob['xx'], 86.0, 1e-6)
        assert_rel_error(self, prob.get_val('xx', 'degC'), 30.0, 1e-6)

        prob.set_val('acomp.x', 30.0, 'degC', indices=[0])
        assert_rel_error(self, prob['acomp.x'][0], 86.0, 1e-6)
        assert_rel_error(self, prob.get_val('acomp.x', 'degC', indices=0), 30.0, 1e-6)

        prob.set_val('axx', 30.0, 'degC', indices=0)
        assert_rel_error(self, prob['axx'][0], 86.0, 1e-6)
        assert_rel_error(self, prob.get_val('axx', 'degC', indices=np.array([0])), 30.0, 1e-6)

        prob.final_setup()

        # Now we do it all over again for coverage.

        # Gets

        assert_rel_error(self, prob.get_val('comp.x'), 86.0, 1e-6)
        assert_rel_error(self, prob.get_val('comp.x', 'degC'), 30.0, 1e-6)
        assert_rel_error(self, prob.get_val('comp.y'), 0.0, 1e-6)
        assert_rel_error(self, prob.get_val('comp.y', 'degF'), 32.0, 1e-6)

        assert_rel_error(self, prob.get_val('xx'), 86.0, 1e-6)
        assert_rel_error(self, prob.get_val('xx', 'degC'), 30.0, 1e-6)
        assert_rel_error(self, prob.get_val('yy'), 0.0, 1e-6)
        assert_rel_error(self, prob.get_val('yy', 'degF'), 32.0, 1e-6)

        assert_rel_error(self, prob.get_val('acomp.x', indices=0), 86.0, 1e-6)
        assert_rel_error(self, prob.get_val('acomp.x', indices=[1]), 95.0, 1e-6)
        assert_rel_error(self, prob.get_val('acomp.x', 'degC', indices=[0]), 30.0, 1e-6)
        assert_rel_error(self, prob.get_val('acomp.x', 'degC', indices=1), 35.0, 1e-6)
        assert_rel_error(self, prob.get_val('acomp.y', indices=0), 0.0, 1e-6)
        assert_rel_error(self, prob.get_val('acomp.y', 'degF', indices=0), 32.0, 1e-6)

        assert_rel_error(self, prob.get_val('axx', indices=0), 86.0, 1e-6)
        assert_rel_error(self, prob.get_val('axx', indices=1), 95.0, 1e-6)
        assert_rel_error(self, prob.get_val('axx', 'degC', indices=0), 30.0, 1e-6)
        assert_rel_error(self, prob.get_val('axx', 'degC', indices=np.array([1])), 35.0, 1e-6)
        assert_rel_error(self, prob.get_val('ayy', indices=0), 0.0, 1e-6)
        assert_rel_error(self, prob.get_val('ayy', 'degF', indices=0), 32.0, 1e-6)

        # Sets

        prob.set_val('comp.x', 35.0, 'degC')
        assert_rel_error(self, prob['comp.x'], 95.0, 1e-6)
        assert_rel_error(self, prob.get_val('comp.x', 'degC'), 35.0, 1e-6)

        prob.set_val('xx', 35.0, 'degC')
        assert_rel_error(self, prob['xx'], 95.0, 1e-6)
        assert_rel_error(self, prob.get_val('xx', 'degC'), 35.0, 1e-6)

        prob.set_val('acomp.x', 35.0, 'degC', indices=[0])
        assert_rel_error(self, prob['acomp.x'][0], 95.0, 1e-6)
        assert_rel_error(self, prob.get_val('acomp.x', 'degC', indices=0), 35.0, 1e-6)

        prob.set_val('axx', 35.0, 'degC', indices=0)
        assert_rel_error(self, prob['axx'][0], 95.0, 1e-6)
        assert_rel_error(self, prob.get_val('axx', 'degC', indices=np.array([0])), 35.0, 1e-6)

    def test_get_set_with_units_error_messages(self):
        from openmdao.api import Problem, ExecComp

        prob = Problem()
        prob.model.add_subsystem('comp', ExecComp('y=x+1.',
                                                  x={'value': 100.0, 'units': 'cm'},
                                                  y={'units': 'm'}))
        prob.model.add_subsystem('no_unit', ExecComp('y=x+1.', x={'value': 100.0}))

        prob.setup()
        prob.run_model()

        msg = "Incompatible units for conversion: 'cm' and 'degK'."
        with assertRaisesRegex(self, TypeError, msg):
            prob.get_val('comp.x', 'degK')

        msg = "Incompatible units for conversion: 'degK' and 'cm'."
        with assertRaisesRegex(self, TypeError, msg):
            prob.set_val('comp.x', 55.0, 'degK')

        msg = "Incompatible units for conversion: 'None' and 'degK'."
        with assertRaisesRegex(self, TypeError, msg):
            prob.get_val('no_unit.x', 'degK')

        msg = "Incompatible units for conversion: 'degK' and 'None'."
        with assertRaisesRegex(self, TypeError, msg):
            prob.set_val('no_unit.x', 55.0, 'degK')

    def test_feature_get_set_with_units(self):
        from openmdao.api import Problem, ExecComp

        prob = Problem()
        prob.model.add_subsystem('comp', ExecComp('y=x+1.',
                                                  x={'value': 100.0, 'units': 'cm'},
                                                  y={'units': 'm'}))

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob.get_val('comp.x'), 100, 1e-6)
        assert_rel_error(self, prob.get_val('comp.x', 'm'), 1.0, 1e-6)
        prob.set_val('comp.x', 10.0, 'mm')
        assert_rel_error(self, prob.get_val('comp.x'), 1.0, 1e-6)
        assert_rel_error(self, prob.get_val('comp.x', 'm'), 1.0e-2, 1e-6)

    def test_feature_get_set_array_with_units(self):
        from openmdao.api import Problem, ExecComp

        prob = Problem()
        prob.model.add_subsystem('comp', ExecComp('y=x+1.',
                                                  x={'value': np.array([100.0, 33.3]), 'units': 'cm'},
                                                  y={'shape': (2, ), 'units': 'm'}))

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob.get_val('comp.x'), np.array([100, 33.3]), 1e-6)
        assert_rel_error(self, prob.get_val('comp.x', 'm'), np.array([1.0, 0.333]), 1e-6)
        assert_rel_error(self, prob.get_val('comp.x', 'km', indices=[0]), 0.001, 1e-6)

        prob.set_val('comp.x', 10.0, 'mm')
        assert_rel_error(self, prob.get_val('comp.x'), np.array([1.0, 1.0]), 1e-6)
        assert_rel_error(self, prob.get_val('comp.x', 'm', indices=0), 1.0e-2, 1e-6)

        prob.set_val('comp.x', 50.0, 'mm', indices=[1])
        assert_rel_error(self, prob.get_val('comp.x'), np.array([1.0, 5.0]), 1e-6)
        assert_rel_error(self, prob.get_val('comp.x', 'm', indices=1), 5.0e-2, 1e-6)

    def test_feature_set_get_array(self):
        import numpy as np

        from openmdao.api import Problem, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem(model=SellarDerivatives())
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.setup()

        # default value from the class definition
        assert_rel_error(self, prob['x'], 1.0, 1e-6)

        prob['x'] = 2.75
        assert_rel_error(self, prob['x'], 2.75, 1e-6)

        # default value from the class definition
        assert_rel_error(self, prob['z'], [5.0, 2.0], 1e-6)

        prob['z'] = [1.5, 1.5]  # for convenience we convert the list to an array.
        assert_rel_error(self, prob['z'], [1.5, 1.5], 1e-6)

        prob.run_model()
        assert_rel_error(self, prob['y1'], 5.43379016853, 1e-6)
        assert_rel_error(self, prob['y2'], 5.33104915618, 1e-6)

        prob['z'] = np.array([2.5, 2.5])
        assert_rel_error(self, prob['z'], [2.5, 2.5], 1e-6)

        prob.run_model()
        assert_rel_error(self, prob['y1'], 9.87161739688, 1e-6)
        assert_rel_error(self, prob['y2'], 8.14191301549, 1e-6)

    def test_feature_residuals(self):
        from openmdao.api import Problem, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem(model=SellarDerivatives())
        prob.model.nonlinear_solver = NonlinearBlockGS()

        prob.setup()

        prob['z'] = [1.5, 1.5]  # for convenience we convert the list to an array.
        prob.run_model()

        inputs, outputs, residuals = prob.model.get_nonlinear_vectors()

        self.assertLess(residuals['y1'], 1e-6)
        self.assertLess(residuals['y2'], 1e-6)

    def test_setup_bad_mode(self):
        # Test error message when passing bad mode to setup.

        prob = Problem()

        try:
            prob.setup(mode='junk')
        except ValueError as err:
            msg = "Unsupported mode: 'junk'. Use either 'fwd' or 'rev'."
            self.assertEqual(str(err), msg)
        else:
            self.fail('Expecting ValueError')

    def test_setup_bad_mode_direction_fwd(self):

        prob = Problem()
        prob.model.add_subsystem("indep", IndepVarComp("x", np.ones(99)))
        prob.model.add_subsystem("C1", ExecComp("y=2.0*x", x=np.zeros(10), y=np.zeros(10)))

        prob.model.connect("indep.x", "C1.x", src_indices=list(range(10)))

        prob.model.add_design_var("indep.x")
        prob.model.add_objective("C1.y")

        prob.setup(mode='fwd')

        with warnings.catch_warnings(record=True) as w:
            prob.final_setup()

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, RuntimeWarning))
        self.assertEqual(str(w[0].message),
                         "Inefficient choice of derivative mode.  "
                         "You chose 'fwd' for a problem with 99 design variables and 10 "
                         "response variables (objectives and nonlinear constraints).")

    def test_setup_bad_mode_direction_rev(self):

        prob = Problem()
        prob.model.add_subsystem("indep", IndepVarComp("x", np.ones(10)))
        prob.model.add_subsystem("C1", ExecComp("y=2.0*x", x=np.zeros(10), y=np.zeros(10)))
        prob.model.add_subsystem("C2", ExecComp("y=2.0*x", x=np.zeros(10), y=np.zeros(10)))

        prob.model.connect("indep.x", ["C1.x", "C2.x"])

        prob.model.add_design_var("indep.x")
        prob.model.add_constraint("C1.y")
        prob.model.add_constraint("C2.y")

        prob.setup(mode='rev')

        with warnings.catch_warnings(record=True) as w:
            prob.final_setup()

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, RuntimeWarning))
        self.assertEqual(str(w[0].message),
                         "Inefficient choice of derivative mode.  "
                         "You chose 'rev' for a problem with 10 design variables and 20 "
                         "response variables (objectives and nonlinear constraints).")

    def test_run_before_setup(self):
        # Test error message when running before setup.

        prob = Problem()

        try:
            prob.run_model()
        except RuntimeError as err:
            msg = "The `setup` method must be called before `run_model`."
            self.assertEqual(str(err), msg)
        else:
            self.fail('Expecting RuntimeError')

        try:
            prob.run_driver()
        except RuntimeError as err:
            msg = "The `setup` method must be called before `run_driver`."
            self.assertEqual(str(err), msg)
        else:
            self.fail('Expecting RuntimeError')

    def test_run_with_invalid_prefix(self):
        # Test error message when running with invalid prefix.

        msg = "The 'case_prefix' argument should be a string."

        prob = Problem()

        try:
            prob.setup()
            prob.run_model(case_prefix=1234)
        except TypeError as err:
            self.assertEqual(str(err), msg)
        else:
            self.fail('Expecting TypeError')

        try:
            prob.setup()
            prob.run_driver(case_prefix=12.34)
        except TypeError as err:
            self.assertEqual(str(err), msg)
        else:
            self.fail('Expecting TypeError')

    def test_root_deprecated(self):
        # testing the root property
        msg = "The 'root' property provides backwards compatibility " \
            + "with OpenMDAO <= 1.x ; use 'model' instead."

        prob = Problem()

        # check deprecation on setter
        with warnings.catch_warnings(record=True) as w:
            prob.root = Group()

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message), msg)

        # check deprecation on getter
        with warnings.catch_warnings(record=True) as w:
            prob.root

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message), msg)

        # testing the root kwarg
        with self.assertRaises(ValueError) as cm:
            prob = Problem(root=Group(), model=Group())

        self.assertEqual(str(cm.exception),
                         "Cannot specify both 'root' and 'model'. "
                         "'root' has been deprecated, please use 'model'.")

        with warnings.catch_warnings(record=True) as w:
            prob = Problem(root=Group())

        self.assertEqual(str(w[0].message), "The 'root' argument provides backwards "
                         "compatibility with OpenMDAO <= 1.x ; use 'model' instead.")

    def test_args(self):
        # defaults
        prob = Problem()
        self.assertTrue(isinstance(prob.model, Group))
        self.assertTrue(isinstance(prob.driver, Driver))

        # model
        prob = Problem(SellarDerivatives())
        self.assertTrue(isinstance(prob.model, SellarDerivatives))
        self.assertTrue(isinstance(prob.driver, Driver))

        # driver
        prob = Problem(driver=ScipyOptimizeDriver())
        self.assertTrue(isinstance(prob.model, Group))
        self.assertTrue(isinstance(prob.driver, ScipyOptimizeDriver))

        # model and driver
        prob = Problem(model=SellarDerivatives(), driver=ScipyOptimizeDriver())
        self.assertTrue(isinstance(prob.model, SellarDerivatives))
        self.assertTrue(isinstance(prob.driver, ScipyOptimizeDriver))

        # invalid model
        with self.assertRaises(TypeError) as cm:
            prob = Problem(ScipyOptimizeDriver())

        self.assertEqual(str(cm.exception),
                         "The value provided for 'model' is not a valid System.")

        # invalid driver
        with self.assertRaises(TypeError) as cm:
            prob = Problem(driver=SellarDerivatives())

        self.assertEqual(str(cm.exception),
                         "The value provided for 'driver' is not a valid Driver.")

    def test_relevance(self):
        p = Problem()
        model = p.model

        model.add_subsystem("indep1", IndepVarComp('x', 1.0))
        G1 = model.add_subsystem('G1', Group())
        G1.add_subsystem('C1', ExecComp(['x=2.0*a', 'y=2.0*b', 'z=2.0*a']))
        G1.add_subsystem('C2', ExecComp(['x=2.0*a', 'y=2.0*b', 'z=2.0*b']))
        model.add_subsystem("C3", ExecComp(['x=2.0*a', 'y=2.0*b+3.0*c']))
        model.add_subsystem("C4", ExecComp(['x=2.0*a', 'y=2.0*b']))
        model.add_subsystem("indep2", IndepVarComp('x', 1.0))
        G2 = model.add_subsystem('G2', Group())
        G2.add_subsystem('C5', ExecComp(['x=2.0*a', 'y=2.0*b+3.0*c']))
        G2.add_subsystem('C6', ExecComp(['x=2.0*a', 'y=2.0*b+3.0*c']))
        G2.add_subsystem('C7', ExecComp(['x=2.0*a', 'y=2.0*b']))
        model.add_subsystem("C8", ExecComp(['y=1.5*a+2.0*b']))
        model.add_subsystem("Unconnected", ExecComp('y=99.*x'))

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

        relevant = get_relevant_vars(model._conn_global_abs_in2out, ['indep1.x', 'indep2.x'],
                                     ['C8.y', 'Unconnected.y'], mode='rev')

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

        class ImplSimple(ImplicitComponent):

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

        class Sub(Group):

            def setup(self):
                self.add_subsystem('comp', ImplSimple())

                # This will not solve it
                self.nonlinear_solver = NonlinearBlockGS()

            def configure(self):
                # This will not solve it either.
                self.nonlinear_solver = NonlinearBlockGS()

        class Super(Group):

            def setup(self):
                self.add_subsystem('sub', Sub())

            def configure(self):
                # This will solve it.
                self.sub.nonlinear_solver = NewtonSolver()
                self.sub.linear_solver = ScipyKrylov()

        top = Problem(model=Super())

        top.setup(check=False)

        self.assertTrue(isinstance(top.model.sub.nonlinear_solver, NewtonSolver))
        self.assertTrue(isinstance(top.model.sub.linear_solver, ScipyKrylov))

    def test_post_setup_solver_configure(self):
        # Test that we can change solver settings after we have instantiated our model.

        class ImplSimple(ImplicitComponent):

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

        class Sub(Group):

            def setup(self):
                self.add_subsystem('comp', ImplSimple())

                # This solver will get over-ridden below
                self.nonlinear_solver = NonlinearBlockGS()

            def configure(self):
                # This solver will get over-ridden below
                self.nonlinear_solver = NonlinearBlockGS()

        class Super(Group):

            def setup(self):
                self.add_subsystem('sub', Sub())

        top = Problem(model=Super())

        top.setup(check=False)

        # These solvers override the ones set in the setup method of the 'sub' groups
        top.model.sub.nonlinear_solver = NewtonSolver()
        top.model.sub.linear_solver = ScipyKrylov()

        self.assertTrue(isinstance(top.model.sub.nonlinear_solver, NewtonSolver))
        self.assertTrue(isinstance(top.model.sub.linear_solver, ScipyKrylov))

    def test_feature_system_configure(self):
        from openmdao.api import Problem, Group, ImplicitComponent, NewtonSolver, ScipyKrylov, NonlinearBlockGS

        class ImplSimple(ImplicitComponent):

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

        class Sub(Group):
            def setup(self):
                self.add_subsystem('comp', ImplSimple())

            def configure(self):
                # This solver won't solve the sytem. We want
                # to override it in the parent.
                self.nonlinear_solver = NonlinearBlockGS()

        class Super(Group):
            def setup(self):
                self.add_subsystem('sub', Sub())

            def configure(self):
                # This will solve it.
                self.sub.nonlinear_solver = NewtonSolver()
                self.sub.linear_solver = ScipyKrylov()

        top = Problem(model=Super())

        top.setup(check=False)

        print(isinstance(top.model.sub.nonlinear_solver, NewtonSolver))
        print(isinstance(top.model.sub.linear_solver, ScipyKrylov))

    def test_feature_post_setup_solver_configure(self):
        from openmdao.api import Problem, Group, ImplicitComponent, NewtonSolver, ScipyKrylov, NonlinearBlockGS

        class ImplSimple(ImplicitComponent):

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

        class Sub(Group):

            def setup(self):
                self.add_subsystem('comp', ImplSimple())

                # This will not solve it
                self.nonlinear_solver = NonlinearBlockGS()

            def configure(self):
                # This will not solve it either.
                self.nonlinear_solver = NonlinearBlockGS()

        class Super(Group):

            def setup(self):
                self.add_subsystem('sub', Sub())

        top = Problem(model=Super())

        top.setup(check=False)

        # This will solve it.
        top.model.sub.nonlinear_solver = NewtonSolver()
        top.model.sub.linear_solver = ScipyKrylov()

        self.assertTrue(isinstance(top.model.sub.nonlinear_solver, NewtonSolver))
        self.assertTrue(isinstance(top.model.sub.linear_solver, ScipyKrylov))

    def test_post_setup_hook(self):
        def hook_func(prob):
            prob['p2.y'] = 5.0

        prob = Problem()
        model = prob.model
        Problem._post_setup_func = hook_func

        try:
            model.add_subsystem('p1', IndepVarComp('x', 3.0))
            model.add_subsystem('p2', IndepVarComp('y', -4.0))
            model.add_subsystem('comp', ExecComp("f_xy=2.0*x+3.0*y"))

            model.connect('p1.x', 'comp.x')
            model.connect('p2.y', 'comp.y')

            prob.setup()
            prob.run_model()

            assert_rel_error(self, prob['p2.y'], 5.0)
            assert_rel_error(self, prob['comp.f_xy'], 21.0)
        finally:
            Problem._post_setup_func = None

    def test_list_problem_vars(self):

        prob = Problem(model=SellarDerivatives())
        model = prob.model
        model.nonlinear_solver = NonlinearBlockGS()

        prob.driver = ScipyOptimizeDriver()
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
        self.assertEquals(output[1], 'Design Variables')
        assertRegex(self, output[5], '^pz.z +\|[0-9. e+-]+\| +2')
        self.assertEquals(output[9], 'Constraints')
        assertRegex(self, output[14], '^con_cmp2.con2 +\[[0-9. e+-]+\] +1')
        self.assertEquals(output[17], 'Objectives')
        assertRegex(self, output[21], '^obj_cmp.obj +\[[0-9. e+-]+\] +1')

        # With show_promoted_name=False
        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            prob.list_problem_vars(show_promoted_name=False)
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')
        assertRegex(self, output[5], '^z +\|[0-9. e+-]+\| +2')
        assertRegex(self, output[14], '^con2 +\[[0-9. e+-]+\] +1')
        assertRegex(self, output[21], '^obj +\[[0-9. e+-]+\] +1')

        # With all the optional columns
        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            prob.list_problem_vars(
                desvar_opts=['lower', 'upper', 'ref', 'ref0',
                             'indices', 'adder', 'scaler',
                             'parallel_deriv_color',
                             'vectorize_derivs',
                             'cache_linear_solution'],
                cons_opts=['lower', 'upper', 'equals', 'ref', 'ref0',
                           'indices', 'adder', 'scaler', 'linear',
                           'parallel_deriv_color',
                           'vectorize_derivs',
                           'cache_linear_solution'],
                objs_opts=['ref', 'ref0',
                           'indices', 'adder', 'scaler',
                           'parallel_deriv_color',
                           'vectorize_derivs',
                           'cache_linear_solution'],
            )
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')
        assertRegex(self, output[3],
                    '^name\s+value\s+size\s+lower\s+upper\s+ref\s+ref0\s+'
                    'indices\s+adder\s+scaler\s+parallel_deriv_color\s+'
                    'vectorize_derivs\s+cache_linear_solution')
        assertRegex(self, output[5],
                    '^pz.z\s+\|[0-9.e+-]+\|\s+2\s+\|10.0\|\s+\|[0-9.e+-]+\|\s+None\s+'
                    'None\s+None\s+None\s+None\s+None\s+False\s+False')

        # With all the optional columns and print_arrays
        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            prob.list_problem_vars(print_arrays=True,
                                   desvar_opts=['lower', 'upper', 'ref', 'ref0',
                                                'indices', 'adder', 'scaler',
                                                'parallel_deriv_color',
                                                'vectorize_derivs',
                                                'cache_linear_solution'],
                                   cons_opts=['lower', 'upper', 'equals', 'ref', 'ref0',
                                              'indices', 'adder', 'scaler', 'linear',
                                              'parallel_deriv_color',
                                              'vectorize_derivs',
                                              'cache_linear_solution'],
                                   objs_opts=['ref', 'ref0',
                                              'indices', 'adder', 'scaler',
                                              'parallel_deriv_color',
                                              'vectorize_derivs',
                                              'cache_linear_solution'],
                                   )
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')
        assertRegex(self, output[6], '^\s+value:')
        assertRegex(self, output[7], '^\s+\[[0-9. e+-]+\]')
        assertRegex(self, output[9], '^\s+lower:')
        assertRegex(self, output[10], '^\s+\[[0-9. e+-]+\]')
        assertRegex(self, output[12], '^\s+upper:')
        assertRegex(self, output[13], '^\s+\[[0-9. e+-]+\]')

    def test_feature_list_problem_vars(self):
        import numpy as np
        from openmdao.api import Problem, ScipyOptimizeDriver, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem(model=SellarDerivatives())
        model = prob.model
        model.nonlinear_solver = NonlinearBlockGS()

        prob.driver = ScipyOptimizeDriver()
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
                                            'parallel_deriv_color',
                                            'vectorize_derivs'],
                               cons_opts=['lower', 'upper', 'equals', 'ref', 'ref0',
                                          'indices', 'adder', 'scaler', 'linear'],
                               objs_opts=['ref', 'ref0',
                                          'indices', 'adder', 'scaler',
                                          'parallel_deriv_color',
                                          'vectorize_derivs',
                                          'cache_linear_solution'],
                               )

class NestedProblemTestCase(unittest.TestCase):

    def test_nested_prob(self):

        class _ProblemSolver(NonlinearRunOnce):
            def solve(self):
                # create a simple subproblem and run it to test for global solver_info bug
                p = Problem()
                p.model.add_subsystem('indep', IndepVarComp('x', 1.0))
                p.model.add_subsystem('comp', ExecComp('y=2*x'))
                p.model.connect('indep.x', 'comp.x')
                p.setup()
                p.run_model()

                return super(_ProblemSolver, self).solve()

        p = Problem()
        p.model.add_subsystem('indep', IndepVarComp('x', 1.0))
        G = p.model.add_subsystem('G', Group())
        G.add_subsystem('comp', ExecComp('y=2*x'))
        G.nonlinear_solver = _ProblemSolver()
        p.model.connect('indep.x', 'G.comp.x')
        p.setup()
        p.run_model()




if __name__ == "__main__":
    unittest.main()
