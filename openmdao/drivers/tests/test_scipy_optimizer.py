""" Unit tests for the ScipyOptimizeDriver."""

import unittest
import sys
from io import StringIO

from packaging.version import Version

import numpy as np
from scipy import __version__ as scipy_version

import openmdao.api as om
from openmdao.test_suite.components.expl_comp_array import TestExplCompArrayDense, TestExplCompArraySparse, TestExplCompArrayJacVec
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.paraboloid_distributed import DistParab
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped, SellarDerivatives
from openmdao.test_suite.components.sellar_feature import SellarMDA
from openmdao.test_suite.components.simple_comps import NonSquareArrayComp
from openmdao.test_suite.groups.sin_fitter import SineFitter
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.utils.general_utils import run_driver
from openmdao.utils.testing_utils import set_env_vars_context
from openmdao.utils.mpi import MPI

try:
    from openmdao.parallel_api import PETScVector
    vector_class = PETScVector
except ImportError:
    vector_class = om.DefaultVector
    PETScVector = None

rosenbrock_size = 6  # size of the design variable

def rosenbrock(x):
    x_0 = x[:-1]
    x_1 = x[1:]
    return sum((1 - x_0) ** 2) + 100 * sum((x_1 - x_0 ** 2) ** 2)


class Rosenbrock(om.ExplicitComponent):

    def setup(self):
        self.add_input('x', np.ones(rosenbrock_size))
        self.add_output('f', 0.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x = inputs['x']
        outputs['f'] = rosenbrock(x)

def rastrigin(x):
    a = 10  # constant
    return np.sum(np.square(x) - a * np.cos(2 * np.pi * x)) + a * np.size(x)


class DummyComp(om.ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.
    """

    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('c', val=0.0)

        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3

        Optimal solution (minimum): x = 6.6667; y = -7.3333
        """
        x = inputs['x']
        y = inputs['y']

        noise = 1e-10
        if self.comm.rank == 0:
            outputs['c'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0
        if self.comm.rank == 1:
            outputs['c'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0 + noise

    def compute_partials(self, inputs, partials):
        """
        Jacobian for our paraboloid.
        """
        x = inputs['x']
        y = inputs['y']

        partials['c', 'x'] = 2.0*x - 6.0 + y
        partials['c', 'y'] = 2.0*y + 8.0 + x


@unittest.skipUnless(MPI, "MPI is required.")
class TestMPIScatter(unittest.TestCase):
    N_PROCS = 2

    def test_design_vars_on_all_procs_scipy(self):

        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', 50.0)
        model.set_input_defaults('y', 50.0)

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', DummyComp(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-6, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=-15.0)

        prob.setup()
        prob.run_driver()

        proc_vals = MPI.COMM_WORLD.allgather([prob['x'], prob['y'], prob['c'], prob['f_xy']])
        np.testing.assert_array_almost_equal(proc_vals[0], proc_vals[1])

    def test_opt_distcomp(self):
        size = 7

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', np.ones((size, )))
        ivc.add_output('y', np.ones((size, )))
        ivc.add_output('a', -3.0 + 0.6 * np.arange(size))

        model.add_subsystem('p', ivc, promotes=['*'])
        model.add_subsystem("parab", DistParab(arr_size=size, deriv_type='dense'), promotes=['*'])
        model.add_subsystem('sum', om.ExecComp('f_sum = sum(f_xy)',
                                               f_sum=np.ones((size, )),
                                               f_xy=np.ones((size, ))),
                            promotes_outputs=['*'])
        model.promotes('sum', inputs=['f_xy'], src_indices=om.slicer[:])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_constraint('f_xy', lower=0.0)
        model.add_objective('f_sum', index=-1)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP')

        prob.setup(force_alloc_complex=True)

        prob.run_driver()

        desvar = prob.driver.get_design_var_values()
        con = prob.driver.get_constraint_values()
        obj = prob.driver.get_objective_values()

        assert_near_equal(obj['sum.f_sum'], 0.0, 2e-6)
        assert_near_equal(con['parab.f_xy'],
                          np.zeros(7),
                          1e-5)


@unittest.skipUnless(MPI and  PETScVector, "MPI and PETSc are required.")
class TestScipyOptimizeDriverMPI(unittest.TestCase):
    N_PROCS = 2

    def test_optimization_output_single_proc(self):
        prob = om.Problem()
        prob.model = SellarMDA()
        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-8)

        prob.model.add_design_var('x', lower=0, upper=10)
        prob.model.add_design_var('z', lower=0, upper=10)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0)
        prob.model.add_constraint('con2', upper=0)

        # Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer
        prob.model.approx_totals()

        prob.setup()
        prob.set_solver_print(level=0)

        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            prob.run_driver()
        finally:
            sys.stdout = stdout
        output = strout.getvalue().split('\n')

        msg = "Optimization Complete"
        if MPI.COMM_WORLD.rank == 0:
            self.assertEqual(msg, output[5])
            self.assertEqual(output.count(msg), 1)
        else:
            self.assertNotEqual(msg, output[0])
            self.assertNotEqual(output.count(msg), 1)


class TestScipyOptimizeDriver(unittest.TestCase):

    def test_driver_supports(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        with self.assertRaises(KeyError) as raises_msg:
            prob.driver.supports['equality_constraints'] = False

        exception = raises_msg.exception

        msg = "ScipyOptimizeDriver: Tried to set read-only option 'equality_constraints'."

        self.assertEqual(exception.args[0], msg)

    def test_compute_totals_basic_return_array(self):
        # Make sure 'array' return_format works.

        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', val=0.)
        model.set_input_defaults('y', val=0.)

        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup(check=False, mode='fwd')
        prob.set_solver_print(level=0)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed.")

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_totals(of=of, wrt=wrt, return_format='array')

        assert_near_equal(derivs[0, 0], -6.0, 1e-6)
        assert_near_equal(derivs[0, 1], 8.0, 1e-6)

        prob.setup(check=False, mode='rev')

        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_totals(of=of, wrt=wrt, return_format='array')

        assert_near_equal(derivs[0, 0], -6.0, 1e-6)
        assert_near_equal(derivs[0, 1], 8.0, 1e-6)

    def test_compute_totals_return_array_non_square(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp(name="x", val=np.ones((2, ))))
        comp = model.add_subsystem('comp', NonSquareArrayComp())
        model.connect('px.x', 'comp.x1')

        model.add_design_var('px.x')
        model.add_objective('px.x')
        model.add_constraint('comp.y1')
        model.add_constraint('comp.y2')

        prob.setup(check=False, mode='auto')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed.")

        derivs = prob.compute_totals(of=['comp.y1'], wrt=['px.x'], return_format='array')

        J = comp.JJ[0:3, 0:2]
        assert_near_equal(J, derivs, 1.0e-3)

        # Support for a name to be in 'of' and 'wrt'

        derivs = prob.compute_totals(of=['comp.y2', 'px.x', 'comp.y1'],
                                     wrt=['px.x'],
                                     return_format='array')

        assert_near_equal(J, derivs[3:, :], 1.0e-3)
        assert_near_equal(comp.JJ[3:4, 0:2], derivs[0:1, :], 1.0e-3)
        assert_near_equal(np.eye(2), derivs[1:3, :], 1.0e-3)

    def test_deriv_wrt_self(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp(name="x", val=np.ones((2, ))))

        model.add_design_var('px.x')
        model.add_objective('px.x')

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed.")

        # Support for a name to be in 'of' and 'wrt'

        J = prob.driver._compute_totals(of=['px.x'], wrt=['px.x'],
                                        return_format='array')

        assert_near_equal(J, np.eye(2), 1.0e-3)

    def test_scipy_optimizer_simple_paraboloid_unconstrained(self):

        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', val=50.)
        model.set_input_defaults('y', val=50.)

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup()

        prob.set_val('x', 50.)
        prob.set_val('y', 50.)

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_near_equal(prob['x'], 6.66666667, 1e-6)
        assert_near_equal(prob['y'], -7.3333333, 1e-6)

    def test_simple_paraboloid_unconstrained(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_near_equal(prob['x'], 6.66666667, 1e-6)
        assert_near_equal(prob['y'], -7.3333333, 1e-6)

    def test_simple_paraboloid_unconstrained_COBYLA(self):
        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', val=50.)
        model.set_input_defaults('y', val=50.)

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='COBYLA', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_near_equal(prob['x'], 6.66666667, 1e-6)
        assert_near_equal(prob['y'], -7.3333333, 1e-6)

    def test_simple_paraboloid_upper(self):

        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', val=50.)
        model.set_input_defaults('y', val=50.)

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'], 7.16667, 1e-6)
        assert_near_equal(prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_lower(self):

        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', val=50.)
        model.set_input_defaults('y', val=50.)

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)

        model.add_objective('f_xy')
        model.add_constraint('c', lower=15.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'], 7.16667, 1e-6)
        assert_near_equal(prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_equality(self):

        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', val=50.)
        model.set_input_defaults('y', val=50.)

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=-15.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        # Minimum should be at (7.166667, -7.833334)
        # (Note, loose tol because of appveyor py3.4 machine.)
        assert_near_equal(prob['x'], 7.16667, 1e-4)
        assert_near_equal(prob['y'], -7.833334, 1e-4)

    def test_unsupported_equality(self):

        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', val=50.)
        model.set_input_defaults('y', val=50.)

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='COBYLA', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=-15.0)

        prob.setup()

        with self.assertRaises(Exception) as raises_cm:
            prob.run_driver()

        exception = raises_cm.exception

        msg = "Constraints of type 'eq' not handled by COBYLA."

        self.assertEqual(exception.args[0], msg)

    def test_scipy_missing_objective(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('x', om.IndepVarComp('x', 2.0), promotes=['*'])
        model.add_subsystem('f_x', Paraboloid(), promotes=['*'])

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP')

        prob.model.add_design_var('x', lower=0)
        # prob.model.add_constraint('x', lower=0)

        prob.setup()

        with self.assertRaises(Exception) as raises_msg:
            prob.run_driver()

        exception = raises_msg.exception

        msg = "Driver requires objective to be declared"

        self.assertEqual(exception.args[0], msg)

    def test_scipy_invalid_desvar_behavior(self):

        expected_err = ("The following design variable initial conditions are out of their specified "
                        "bounds:"
                        "\n  paraboloid.x"
                        "\n    val: [100.]"
                        "\n    lower: -50.0"
                        "\n    upper: 50.0"
                        "\n  paraboloid.y"
                        "\n    val: [-200.]"
                        "\n    lower: -50.0"
                        "\n    upper: 50.0"
                        "\nSet the initial value of the design variable to a valid value or set "
                        "the driver option['invalid_desvar_behavior'] to 'ignore'."
                        "\nThis warning will become an error by default in OpenMDAO version 3.25.")

        for option in ['warn', 'raise', 'ignore']:
            with self.subTest(f'invalid_desvar_behavior = {option}'):
                # build the model
                prob = om.Problem()

                prob.model.add_subsystem('paraboloid', om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))

                # setup the optimizationi
                prob.driver = om.ScipyOptimizeDriver(invalid_desvar_behavior=option)
                prob.driver.options['optimizer'] = 'SLSQP'

                prob.model.add_design_var('paraboloid.x', lower=-50, upper=50)
                prob.model.add_design_var('paraboloid.y', lower=-50, upper=50)
                prob.model.add_objective('paraboloid.f')

                prob.setup()

                # Set initial values.
                prob.set_val('paraboloid.x', 100.0)
                prob.set_val('paraboloid.y', -200.0)

                # run the optimization
                if option == 'ignore':
                    prob.run_driver()
                elif option == 'raise':
                    with self.assertRaises(ValueError) as ctx:
                        prob.run_driver()
                    self.assertEqual(str(ctx.exception), expected_err)
                else:
                    with assert_warning(om.DriverWarning, expected_err):
                        prob.run_driver()

                if option != 'raise':
                    assert_near_equal(prob.get_val('paraboloid.x'), 6.66666666, tolerance=1.0E-5)
                    assert_near_equal(prob.get_val('paraboloid.y'), -7.33333333, tolerance=1.0E-5)
                    assert_near_equal(prob.get_val('paraboloid.f'), -27.33333333, tolerance=1.0E-5)

    def test_scipy_invalid_desvar_behavior_env(self):

        expected_err = ("The following design variable initial conditions are out of their specified "
                        "bounds:"
                        "\n  paraboloid.x"
                        "\n    val: [100.]"
                        "\n    lower: -50.0"
                        "\n    upper: 50.0"
                        "\n  paraboloid.y"
                        "\n    val: [-200.]"
                        "\n    lower: -50.0"
                        "\n    upper: 50.0"
                        "\nSet the initial value of the design variable to a valid value or set "
                        "the driver option['invalid_desvar_behavior'] to 'ignore'."
                        "\nThis warning will become an error by default in OpenMDAO version 3.25.")

        for option in ['warn', 'raise', 'ignore']:
            with self.subTest(f'invalid_desvar_behavior = {option}'):
                with set_env_vars_context(OPENMDAO_INVALID_DESVAR_BEHAVIOR=option):
                    # build the model
                    prob = om.Problem()

                    prob.model.add_subsystem('paraboloid', om.ExecComp('f = (x-3)**2 + x*y + (y+4)**2 - 3'))

                    # setup the optimizationi
                    prob.driver = om.ScipyOptimizeDriver()
                    prob.driver.options['optimizer'] = 'SLSQP'

                    prob.model.add_design_var('paraboloid.x', lower=-50, upper=50)
                    prob.model.add_design_var('paraboloid.y', lower=-50, upper=50)
                    prob.model.add_objective('paraboloid.f')

                    prob.setup()

                    # Set initial values.
                    prob.set_val('paraboloid.x', 100.0)
                    prob.set_val('paraboloid.y', -200.0)

                    # run the optimization
                    if option == 'ignore':
                        prob.run_driver()
                    elif option == 'raise':
                        with self.assertRaises(ValueError) as ctx:
                            prob.run_driver()
                        self.assertEqual(str(ctx.exception), expected_err)
                    else:
                        with assert_warning(om.DriverWarning, expected_err):
                            prob.run_driver()

                    if option != 'raise':
                        assert_near_equal(prob.get_val('paraboloid.x'), 6.66666666, tolerance=1.0E-5)
                        assert_near_equal(prob.get_val('paraboloid.y'), -7.33333333, tolerance=1.0E-5)
                        assert_near_equal(prob.get_val('paraboloid.f'), -27.33333333, tolerance=1.0E-5)

    def test_simple_paraboloid_double_sided_low(self):

        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', val=50.)
        model.set_input_defaults('y', val=50.)

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=-11.0, upper=-10.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_near_equal(prob['y'] - prob['x'], -11.0, 1e-6)

    def test_simple_paraboloid_double_sided_high(self):

        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', val=50.)
        model.set_input_defaults('y', val=50.)

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_near_equal(prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_array_comp2D(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = areas - 20.0', c=np.zeros((2, 2)),
                                               areas=np.zeros((2, 2))),
                            promotes=['*'])
        model.add_subsystem('obj', om.ExecComp('o = areas[0, 0]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('widths', lower=-50.0, upper=50.0)
        model.add_objective('o')
        model.add_constraint('c', equals=0.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        obj = prob['o']
        assert_near_equal(obj, 20.0, 1e-6)

    def test_simple_array_comp2D_eq_con(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('obj', om.ExecComp('o = areas[0, 0] + areas[1, 1]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('widths', lower=-50.0, upper=50.0)
        model.add_objective('o')
        model.add_constraint('areas', equals=np.array([24.0, 21.0, 3.5, 17.5]))

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        obj = prob['o']
        assert_near_equal(obj, 41.5, 1e-6)

    def test_simple_array_comp2D_sparse_eq_con(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArraySparse(), promotes=['*'])
        model.add_subsystem('obj', om.ExecComp('o = areas[0, 0] + areas[1, 1]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('widths', lower=-50.0, upper=50.0)
        model.add_objective('o')
        model.add_constraint('areas', equals=np.array([24.0, 21.0, 3.5, 17.5]))

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        obj = prob['o']
        assert_near_equal(obj, 41.5, 1e-6)

    def test_simple_array_comp2D_jacvec_eq_con(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayJacVec(), promotes=['*'])
        model.add_subsystem('obj', om.ExecComp('o = areas[0, 0] + areas[1, 1]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('widths', lower=-50.0, upper=50.0)
        model.add_objective('o')
        model.add_constraint('areas', equals=np.array([24.0, 21.0, 3.5, 17.5]))

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        obj = prob['o']
        assert_near_equal(obj, 41.5, 1e-6)

    def test_simple_array_comp2D_dbl_sided_con(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('obj', om.ExecComp('o = areas[0, 0]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('widths', lower=-50.0, upper=50.0)
        model.add_objective('o')
        model.add_constraint('areas', lower=np.array([24.0, 21.0, 3.5, 17.5]), upper=np.array([24.0, 21.0, 3.5, 17.5]))

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        con = prob['areas']
        assert_near_equal(con, np.array([[24.0, 21.0], [3.5, 17.5]]), 1e-6)

    def test_simple_array_comp2D_dbl_sided_con_array(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('obj', om.ExecComp('o = areas[0, 0]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('widths', lower=-50.0, upper=50.0)
        model.add_objective('o')
        model.add_constraint('areas', lower=20.0, upper=20.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        obj = prob['o']
        assert_near_equal(obj, 20.0, 1e-6)

    def test_simple_array_comp2D_array_lo_hi(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = areas - 20.0', c=np.zeros((2, 2)), areas=np.zeros((2, 2))),
                            promotes=['*'])
        model.add_subsystem('obj', om.ExecComp('o = areas[0, 0]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('widths', lower=-50.0*np.ones((2, 2)), upper=50.0*np.ones((2, 2)))
        model.add_objective('o')
        model.add_constraint('c', equals=0.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        obj = prob['o']
        assert_near_equal(obj, 20.0, 1e-6)

    def test_simple_paraboloid_scaled_desvars_fwd(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0, ref=.02)
        model.add_design_var('y', lower=-50.0, upper=50.0, ref=.02)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup(check=False, mode='fwd')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_near_equal(prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_desvars_rev(self):

        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', val=50.)
        model.set_input_defaults('y', val=50.)

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0, ref=.02)
        model.add_design_var('y', lower=-50.0, upper=50.0, ref=.02)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup(check=False, mode='rev')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_near_equal(prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_constraint_fwd(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0, ref=10.)

        prob.setup(check=False, mode='fwd')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_near_equal(prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_objective_fwd(self):

        prob = om.Problem()
        model = prob.model

        prob.set_solver_print(level=0)

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy', ref=10.)
        model.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup(check=False, mode='fwd')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_near_equal(prob['x'] - prob['y'], 11.0, 1e-6)

    def test_simple_paraboloid_scaled_objective_rev(self):

        prob = om.Problem()
        model = prob.model

        prob.set_solver_print(level=0)

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy', ref=10.)
        model.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup(check=False, mode='rev')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_near_equal(prob['x'] - prob['y'], 11.0, 1e-6)

    def test_sellar_mdf(self):

        prob = om.Problem()
        model = prob.model = SellarDerivativesGrouped(nonlinear_solver=om.NonlinearBlockGS,
                                                      linear_solver=om.ScipyKrylov)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.setup(check=False, mode='rev')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_near_equal(prob['z'][0], 1.9776, 1e-3)
        assert_near_equal(prob['z'][1], 0.0, 1e-3)
        assert_near_equal(prob['x'], 0.0, 1e-3)

    def test_bug_in_eq_constraints(self):
        # We were getting extra constraints created because lower and upper are maxfloat instead of
        # None when unused.
        p = om.Problem(model=SineFitter())
        p.driver = om.ScipyOptimizeDriver()

        p.setup()
        p.run_driver()

        max_defect = np.max(np.abs(p['defect.defect']))
        assert_near_equal(max_defect, 0.0, 1e-10)

    def test_reraise_exception_from_callbacks(self):
        class ReducedActuatorDisc(om.ExplicitComponent):

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

        prob = om.Problem()
        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('a', .5)
        indeps.add_output('Vu', 10.0, units='m/s')

        prob.model.add_subsystem('a_disk', ReducedActuatorDisc(),
                                 promotes_inputs=['a', 'Vu'])

        # setup the optimization
        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP')

        prob.model.add_design_var('a', lower=0., upper=1.)
        # negative one so we maximize the objective
        prob.model.add_objective('a_disk.Vd', scaler=-1)

        prob.setup()

        with self.assertRaises(KeyError) as context:
            prob.run_driver()

        msg = 'Variable name pair ("Vd", "a") must first be declared.'
        self.assertTrue(msg in str(context.exception))

    def test_simple_paraboloid_upper_COBYLA(self):

        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', val=50.)
        model.set_input_defaults('y', val=50.)

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='COBYLA', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'], 7.16667, 1e-6)
        assert_near_equal(prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_desvar_indices_COBYLA(self):
        # verify indices are handled properly when creating constraints for
        # upper and lower bounds on design variables for COBYLA
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('indep', om.IndepVarComp('xy', val=[-1., 50., 50., -1]))
        model.add_subsystem('comp', Paraboloid())
        model.add_subsystem('cons', om.ExecComp('c = - x + y'))

        model.connect('indep.xy', ['comp.x', 'cons.x'], src_indices=[1])
        model.connect('indep.xy', ['comp.y', 'cons.y'], src_indices=[2])

        model.add_design_var('indep.xy', indices=[1,2], lower=[-50.0, -50.0], upper=[50.0, 50.0])
        model.add_objective('comp.f_xy')
        model.add_constraint('cons.c', upper=-15.0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='COBYLA', tol=1e-9, disp=False)
        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['indep.xy'], [-1, 7.16667, -7.833334, -1], 1e-6)

    def test_sellar_mdf_COBYLA(self):

        prob = om.Problem()
        model = prob.model = SellarDerivativesGrouped(nonlinear_solver=om.NonlinearBlockGS,
                                                      linear_solver=om.ScipyKrylov)

        prob.driver = om.ScipyOptimizeDriver(optimizer='COBYLA', tol=1e-9, disp=False)

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.setup(check=False, mode='rev')

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        assert_near_equal(prob['z'][0], 1.9776, 1e-3)
        assert_near_equal(prob['z'][1], 0.0, 1e-3)
        assert_near_equal(prob['x'], 0.0, 1e-3)

    @unittest.skipUnless(Version(scipy_version) >= Version("1.1"),
                         "scipy >= 1.1 is required.")
    def test_trust_constr(self):

        class Rosenbrock(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', np.array([1.5, 1.5, 1.5]))
                self.add_output('f', 0.0)
                self.declare_partials('f', 'x', method='fd', form='central', step=1e-2)

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                x = inputs['x']
                outputs['f'] = rosenbrock(x)

        x0 = np.array([1.2, 0.8, 1.3])

        prob = om.Problem()
        model = prob.model
        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('x', list(x0))

        prob.model.add_subsystem('rosen', Rosenbrock(), promotes=['*'])
        prob.model.add_subsystem('con', om.ExecComp('c=sum(x)', x=np.ones(3)), promotes=['*'])
        prob.driver = driver = om.ScipyOptimizeDriver()
        driver.options['optimizer'] = 'trust-constr'
        driver.options['tol'] = 1e-8
        driver.options['maxiter'] = 2000
        driver.options['disp'] = False

        model.add_design_var('x')
        model.add_objective('f', scaler=1/rosenbrock(x0))
        model.add_constraint('c', lower=0, upper=10)  # Double sided

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob['x'], np.ones(3), 2e-2)
        assert_near_equal(prob['f'], 0., 1e-2)
        self.assertTrue(prob['c'] < 10)
        self.assertTrue(prob['c'] > 0)

    @unittest.skipUnless(Version(scipy_version) >= Version("1.1"),
                         "scipy >= 1.1 is required.")
    def test_trust_constr_hess_option(self):

        class Rosenbrock(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', np.array([1.5, 1.5, 1.5]))
                self.add_output('f', 0.0)
                self.declare_partials('f', 'x', method='fd', form='central', step=1e-3)

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                x = inputs['x']
                outputs['f'] = rosenbrock(x)

        x0 = np.array([1.2, 0.8, 1.3])

        prob = om.Problem()
        model = prob.model
        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('x', list(x0))

        prob.model.add_subsystem('rosen', Rosenbrock(), promotes=['*'])
        prob.model.add_subsystem('con', om.ExecComp('c=sum(x)', x=np.ones(3)), promotes=['*'])
        prob.driver = driver = om.ScipyOptimizeDriver()
        driver.options['optimizer'] = 'trust-constr'
        driver.options['tol'] = 1e-8
        driver.options['maxiter'] = 2000
        driver.options['disp'] = False
        driver.opt_settings['hess'] = '2-point'

        model.add_design_var('x')
        model.add_objective('f', scaler=1/rosenbrock(x0))
        model.add_constraint('c', lower=0, upper=10)  # Double sided

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob['x'], np.ones(3), 2e-2)
        assert_near_equal(prob['f'], 0., 1e-2)
        self.assertTrue(prob['c'] < 10)
        self.assertTrue(prob['c'] > 0)

    @unittest.skipUnless(Version(scipy_version) >= Version("1.1"),
                         "scipy >= 1.1 is required.")
    def test_trust_constr_equality_con(self):

        class Rosenbrock(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', np.array([1.5, 1.5, 1.5]))
                self.add_output('f', 0.0)
                self.declare_partials('f', 'x', method='fd', form='central', step=1e-4)

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                x = inputs['x']
                outputs['f'] = rosenbrock(x)

        x0 = np.array([0.5, 0.8, 1.4])

        prob = om.Problem()
        model = prob.model
        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', list(x0))

        model.add_subsystem('rosen', Rosenbrock())
        model.add_subsystem('con', om.ExecComp('c=sum(x)', x=np.ones(3)))
        model.connect('indeps.x', 'rosen.x')
        model.connect('indeps.x', 'con.x')
        prob.driver = driver = om.ScipyOptimizeDriver()
        driver.options['optimizer'] = 'trust-constr'
        driver.options['tol'] = 1e-5
        driver.options['maxiter'] = 2000
        driver.options['disp'] = False

        model.add_design_var('indeps.x')
        model.add_objective('rosen.f', scaler=1/rosenbrock(x0))
        model.add_constraint('con.c', equals=1.)

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob['con.c'], 1., 1e-3)

    @unittest.skipUnless(Version(scipy_version) >= Version("1.2"),
                         "scipy >= 1.2 is required.")
    def test_trust_constr_inequality_con(self):

        class Sphere(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', np.array([1.5, 1.5]))
                self.add_output('f', 0.0)
                self.declare_partials('f', 'x', method='fd', form='central', step=1e-4)

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                x = inputs['x']
                outputs['f'] = sum(x**2)

        x0 = np.array([1.2, 1.5])

        prob = om.Problem()
        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('x', list(x0))

        prob.model.add_subsystem('sphere', Sphere(), promotes=['*'])
        prob.model.add_subsystem('con', om.ExecComp('c=sum(x)', x=np.ones(2)), promotes=['*'])
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'trust-constr'
        prob.driver.options['tol'] = 1e-5
        prob.driver.options['maxiter'] = 2000
        prob.driver.options['disp'] = False

        prob.model.add_design_var('x')
        prob.model.add_objective('f')
        prob.model.add_constraint('c', lower=1.0)

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob['c'], 1.0, 1e-2)

    @unittest.skipUnless(Version(scipy_version) >= Version("1.2"),
                         "scipy >= 1.2 is required.")
    def test_trust_constr_bounds(self):
        class Rosenbrock(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', np.array([-1.5, -1.5]))
                self.add_output('f', 1000.0)
                self.declare_partials('f', 'x', method='fd', form='central', step=1e-3)

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                x = inputs['x']
                outputs['f'] = sum(x ** 2)

        x0 = np.array([-1.5, -1.5])

        prob = om.Problem()
        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('x', list(x0))

        prob.model.add_subsystem('sphere', Rosenbrock(), promotes=['*'])
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'trust-constr'
        prob.driver.options['tol'] = 1e-7
        prob.driver.options['maxiter'] = 2000
        prob.driver.options['disp'] = False

        prob.model.add_design_var('x', lower=np.array([-2., -2.]), upper=np.array([-1., -1.2]))
        prob.model.add_objective('f', scaler=1)

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob['x'][0], -1., 1e-2)
        assert_near_equal(prob['x'][1], -1.2, 1e-2)

    def test_simple_paraboloid_lower_linear(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=15.0, linear=True)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'], 7.16667, 1e-6)
        assert_near_equal(prob['y'], -7.833334, 1e-6)

        self.assertEqual(prob.driver._obj_and_nlcons, ['comp.f_xy'])

    def test_simple_paraboloid_equality_linear(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=-15.0, linear=True)

        prob.setup()

        failed = prob.run_driver()

        self.assertFalse(failed, "Optimization failed, result =\n" +
                                 str(prob.driver.result))

        # Minimum should be at (7.166667, -7.833334)
        assert_near_equal(prob['x'], 7.16667, 1e-6)
        assert_near_equal(prob['y'], -7.833334, 1e-6)

    def test_debug_print_option_totals(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        prob.driver.options['debug_print'] = ['totals']

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup(check=False, mode='rev')

        failed, output = run_driver(prob)

        self.assertFalse(failed, "Optimization failed.")

        self.assertTrue('In mode: rev.' in output)
        self.assertTrue("('comp.f_xy', [0])" in output)
        self.assertTrue('Elapsed Time:' in output)

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        prob.driver.options['debug_print'] = ['totals']

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup(check=False, mode='fwd')

        failed, output = run_driver(prob)

        self.assertFalse(failed, "Optimization failed.")

        self.assertTrue('In mode: fwd.' in output)
        self.assertTrue("('p1.x', [0])" in output)
        self.assertTrue('Elapsed Time:' in output)

    def test_debug_print_all_options(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)
        prob.driver.options['debug_print'] = ['desvars', 'ln_cons', 'nl_cons', 'objs']

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        prob.setup()

        failed, output = run_driver(prob)

        self.assertFalse(failed, "Optimization failed.")

        output = output.split('\n')

        self.assertTrue(output.count("Design Vars") > 1,
                        "Should be more than one design vars header printed")
        self.assertTrue(output.count("Nonlinear constraints") > 1,
                        "Should be more than one nonlinear constraint header printed")
        self.assertTrue(output.count("Linear constraints") > 1,
                        "Should be more than one linear constraint header printed")
        self.assertTrue(output.count("Objectives") > 1,
                        "Should be more than one objective header printed")
        self.assertTrue(len([s for s in output if s.startswith("{'p1.x")]) > 1,
                        "Should be more than one p1.x printed")
        self.assertTrue(len([s for s in output if "'p2.y'" in s]) > 1,
                        "Should be more than one p2.y printed")
        self.assertTrue(len([s for s in output if s.startswith("{'con.c")]) > 1,
                        "Should be more than one con.c printed")
        self.assertTrue(len([s for s in output if s.startswith("{'comp.f_xy")]) > 1,
                        "Should be more than one comp.f_xy printed")

    def test_sellar_mdf_linear_con_directsolver(self):
        # This test makes sure that we call solve_nonlinear first if we have any linear constraints
        # to cache.
        prob = om.Problem()
        model = prob.model = SellarDerivatives()

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)
        model.add_constraint('x', upper=11.0, linear=True)

        prob.setup(check=False, mode='rev')
        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        failed = prob.run_driver()

        assert_near_equal(prob['z'][0], 1.9776, 1e-3)
        assert_near_equal(prob['z'][1], 0.0, 1e-3)
        assert_near_equal(prob['x'], 0.0, 4e-3)

        self.assertEqual(len(prob.driver._lincongrad_cache), 1)
        # Piggyback test: make sure we can run the driver again as a subdriver without a keyerror.
        prob.driver.run()
        self.assertEqual(len(prob.driver._lincongrad_cache), 1)

    def test_call_final_setup(self):
        # Make sure we call final setup if our model hasn't been setup.

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', equals=-15.0)

        prob.setup()

        expected_msg = \
            "Problem .*: run_model must be called before total derivatives can be checked\."
        with self.assertRaisesRegex(RuntimeError, expected_msg):
            totals = prob.check_totals(method='fd', out_stream=False)

    def test_cobyla_linear_constraint(self):
        # Bug where ScipyOptimizeDriver tried to compute and cache the constraint derivatives for the
        # lower and upper bounds of the desvars even though we were using a non-gradient optimizer.
        # This causd a KeyError.
        prob = om.Problem()
        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', 3.0)
        indeps.add_output('y', -4.0)

        prob.model.add_subsystem('parab', Paraboloid())

        prob.model.add_subsystem('const', om.ExecComp('g = x + y'))

        prob.model.connect('indeps.x', ['parab.x', 'const.x'])
        prob.model.connect('indeps.y', ['parab.y', 'const.y'])

        prob.driver = om.ScipyOptimizeDriver(optimizer='COBYLA', tol=1e-9, disp=False)

        prob.model.add_constraint('const.g', lower=0, upper=10.)
        prob.model.add_design_var('indeps.x', **{'ref0': 0, 'ref': 2, 'lower': -50, 'upper': 50})
        prob.model.add_design_var('indeps.y', **{'ref0': 0, 'ref': 2, 'lower': -50, 'upper': 50})
        prob.model.add_objective('parab.f_xy', scaler = 4.0)
        prob.setup()
        prob.run_driver()

        # minimum value
        assert_near_equal(prob['parab.f_xy'], -27, 1e-6)

    def test_multiple_objectives_error(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        self.assertFalse(prob.driver.supports['multiple_objectives'])
        prob.driver.options['debug_print'] = ['nl_cons', 'objs']

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_objective('c')  # Second objective
        prob.setup()

        with self.assertRaises(RuntimeError):
            prob.run_model()

        with self.assertRaises(RuntimeError):
            prob.run_driver()

    def test_basinhopping(self):

        class Func2d(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', np.ones(2))
                self.add_output('f', 0.0)
                self.declare_partials('f', 'x')

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                x = inputs['x']
                outputs['f'] = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]

            def compute_partials(self, inputs, partials):
                x = inputs['x']
                df = np.zeros(2)
                df[0] = -14.5 * np.sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2
                df[1] = 2. * x[1] + 0.2
                partials['f', 'x'] = df

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('indeps', om.IndepVarComp('x', np.ones(2)), promotes=['*'])
        model.add_subsystem('func2d', Func2d(), promotes=['*'])

        prob.driver = driver = om.ScipyOptimizeDriver()
        driver.options['optimizer'] = 'basinhopping'
        driver.options['disp'] = False
        driver.opt_settings['niter'] = 1000
        driver.opt_settings['seed'] = 1234

        model.add_design_var('x', lower=[-1, -1], upper=[0, 0])
        model.add_objective('f')
        prob.setup()
        prob.run_driver()
        assert_near_equal(prob['x'], np.array([-0.1951, -0.1000]), 1e-3)
        assert_near_equal(prob['f'], -1.0109, 1e-3)

    def test_basinhopping_bounded(self):
        # It should find the local minimum, which is inside the bounds

        class Func2d(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', np.ones(2))
                self.add_output('f', 0.0)
                self.declare_partials('f', 'x')

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                x = inputs['x']
                outputs['f'] = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]

            def compute_partials(self, inputs, partials):
                x = inputs['x']
                df = np.zeros(2)
                df[0] = -14.5 * np.sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2
                df[1] = 2. * x[1] + 0.2
                partials['f', 'x'] = df

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('indeps', om.IndepVarComp('x', np.ones(2)), promotes=['*'])
        model.add_subsystem('func2d', Func2d(), promotes=['*'])

        prob.driver = driver = om.ScipyOptimizeDriver()
        driver.options['optimizer'] = 'basinhopping'
        driver.options['disp'] = False
        driver.opt_settings['niter'] = 200
        driver.opt_settings['seed'] = 1234

        model.add_design_var('x', lower=[0, -1], upper=[1, 1])
        model.add_objective('f')
        prob.setup()
        prob.run_driver()
        assert_near_equal(prob['x'], np.array([0.234171, -0.1000]), 1e-3)
        assert_near_equal(prob['f'], -0.907267, 1e-3)

    @unittest.skipUnless(Version(scipy_version) >= Version("1.2"),
                         "scipy >= 1.2 is required.")
    def test_dual_annealing_rastrigin(self):
        # Example from the Scipy documentation

        size = 3  # size of the design variable

        class Rastrigin(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', 0.5 * np.ones(size))
                self.add_output('f', 0.5)

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                x = inputs['x']
                outputs['f'] = rastrigin(x)

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('indeps', om.IndepVarComp('x', np.ones(size)), promotes=['*'])
        model.add_subsystem('rastrigin', Rastrigin(), promotes=['*'])

        prob.driver = driver = om.ScipyOptimizeDriver()
        driver.options['optimizer'] = 'dual_annealing'
        driver.options['disp'] = False
        driver.options['tol'] = 1e-9
        driver.options['maxiter'] = 3000
        driver.opt_settings['seed'] = 1234
        driver.opt_settings['initial_temp'] = 5230

        model.add_design_var('x', lower=-2 * np.ones(size), upper=2 * np.ones(size))
        model.add_objective('f')
        prob.setup()
        prob.run_driver()
        assert_near_equal(prob['x'], np.zeros(size), 1e-2)
        assert_near_equal(prob['f'], 0.0, 1e-2)

    def test_differential_evolution(self):
        # Source of example:
        # https://scipy.github.io/devdocs/generated/scipy.optimize.dual_annealing.html
        np.random.seed(6)

        size = 3  # size of the design variable

        class Rastrigin(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', 0.5 * np.ones(size))
                self.add_output('f', 0.5)

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                x = inputs['x']
                outputs['f'] = rastrigin(x)

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('indeps', om.IndepVarComp('x', np.ones(size)), promotes=['*'])
        model.add_subsystem('rastrigin', Rastrigin(), promotes=['*'])

        prob.driver = driver = om.ScipyOptimizeDriver()
        driver.options['optimizer'] = 'differential_evolution'
        driver.options['disp'] = False
        driver.options['tol'] = 1e-9

        model.add_design_var('x', lower=-5.12 * np.ones(size), upper=5.12 * np.ones(size))
        model.add_objective('f')
        prob.setup()
        prob.run_driver()
        assert_near_equal(prob['x'], np.zeros(size), 1e-6)
        assert_near_equal(prob['f'], 0.0, 1e-6)

    def test_differential_evolution_bounded(self):
        # Source of example:
        # https://scipy.github.io/devdocs/generated/scipy.optimize.dual_annealing.html
        # In this example the minimum is not the unbounded global minimum.

        size = 3  # size of the design variable

        class Rastrigin(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', 0.5 * np.ones(size))
                self.add_output('f', 0.5)

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                x = inputs['x']
                outputs['f'] = rastrigin(x)

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('indeps', om.IndepVarComp('x', np.ones(size)), promotes=['*'])
        model.add_subsystem('rastrigin', Rastrigin(), promotes=['*'])

        prob.driver = driver = om.ScipyOptimizeDriver()
        driver.options['optimizer'] = 'differential_evolution'
        driver.options['disp'] = False
        driver.options['tol'] = 1e-9

        model.add_design_var('x', lower=-2.0 * np.ones(size), upper=-0.5 * np.ones(size))
        model.add_objective('f')
        prob.setup()
        prob.run_driver()
        assert_near_equal(prob['x'], -np.ones(size), 1e-2)
        assert_near_equal(prob['f'], 3.0, 1e-2)

    @unittest.skipUnless(Version(scipy_version) >= Version("1.2"),
                         "scipy >= 1.2 is required.")
    def test_shgo_rosenbrock(self):
        # Source of example:
        # https://stefan-endres.github.io/shgo/

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('indeps', om.IndepVarComp('x', np.ones(rosenbrock_size)), promotes=['*'])
        model.add_subsystem('rosen', Rosenbrock(), promotes=['*'])

        prob.driver = driver = om.ScipyOptimizeDriver()
        driver.options['optimizer'] = 'shgo'
        driver.options['disp'] = False
        driver.opt_settings['maxiter'] = None

        model.add_design_var('x', lower=np.zeros(rosenbrock_size), upper=2*np.ones(rosenbrock_size))
        model.add_objective('f')
        prob.setup()
        prob.run_driver()
        assert_near_equal(prob['x'], np.ones(rosenbrock_size), 1e-2)
        assert_near_equal(prob['f'], 0.0, 1e-2)

    def test_singular_jac_error_responses(self):
        prob = om.Problem()
        size = 3
        prob.model.add_subsystem('parab',
                                 om.ExecComp(['f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0',
                                              'z = 12.0'], shape=(size,)),
                                 promotes_inputs=['x', 'y'])

        prob.model.add_subsystem('const', om.ExecComp('g = x + y', shape=(size,)),
                                 promotes_inputs=['x', 'y'])

        prob.model.set_input_defaults('x', 3.0 * np.ones(size))
        prob.model.set_input_defaults('y', -4.0 * np.ones(size))

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['singular_jac_behavior'] = 'error'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('parab.f_xy', index=1)

        prob.model.add_constraint('const.g', lower=0, upper=10.)

        # This constraint produces a zero row.
        prob.model.add_constraint('parab.z', equals=12.)

        prob.setup()

        with self.assertRaises(RuntimeError) as msg:
            prob.run_driver()

        self.assertEqual(str(msg.exception),
                         "Constraints or objectives [('parab.z', inds=[0, 1, 2])] cannot be impacted by the design " + \
                         "variables of the problem.")

    def test_singular_jac_error_desvars(self):
        prob = om.Problem()
        prob.model.add_subsystem('parab',
                                     om.ExecComp(['f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0 - 0*z',
                                                  ]),
                                     promotes_inputs=['x', 'y', 'z'])

        prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

        prob.model.set_input_defaults('x', 3.0)
        prob.model.set_input_defaults('y', -4.0)

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['singular_jac_behavior'] = 'error'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)

        # Design var z does not affect any quantities.
        prob.model.add_design_var('z', lower=-50, upper=50)

        prob.model.add_objective('parab.f_xy')

        prob.model.add_constraint('const.g', lower=0, upper=10.)

        prob.setup()

        with self.assertRaises(RuntimeError) as msg:
            prob.run_driver()

        self.assertEqual(str(msg.exception),
                         "Design variables [('z', inds=[0])] have no impact on the constraints or objective.")

    def test_singular_jac_ignore(self):
        prob = om.Problem()
        prob.model.add_subsystem('parab',
                                 om.ExecComp(['f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0',
                                              'z = 12.0'],),
                                 promotes_inputs=['x', 'y'])

        prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

        prob.model.set_input_defaults('x', 3.0)
        prob.model.set_input_defaults('y', -4.0)

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['singular_jac_behavior'] = 'ignore'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('parab.f_xy')

        prob.model.add_constraint('const.g', lower=0, upper=10.)

        # This constraint produces a zero row.
        prob.model.add_constraint('parab.z', equals=12.)

        prob.setup()

        # Will not raise an exception.
        prob.run_driver()

    def test_singular_jac_warn(self):
        prob = om.Problem()
        prob.model.add_subsystem('parab',
                                 om.ExecComp(['f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0',
                                              'z = 12.0'],),
                                 promotes_inputs=['x', 'y'])

        prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

        prob.model.set_input_defaults('x', 3.0)
        prob.model.set_input_defaults('y', -4.0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP')
        # Default behavior is 'warn'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('parab.f_xy')

        prob.model.add_constraint('const.g', lower=0, upper=10.)

        # This constraint produces a zero row.
        prob.model.add_constraint('parab.z', equals=12.)

        prob.setup()

        msg = "Constraints or objectives [('parab.z', inds=[0])] cannot be impacted by the design variables of the problem."

        with assert_warning(UserWarning, msg):
            prob.run_driver()

    def test_singular_jac_error_desvars_multidim_indices_dv(self):
        prob = om.Problem()
        prob.model.add_subsystem('parab',
                                 om.ExecComp(['f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0 - 0*z'], shape=(3,2,2)),
                                 promotes_inputs=['x', 'y', 'z'])

        prob.model.add_subsystem('const', om.ExecComp('g = x + y', shape=(3,2,2)), promotes_inputs=['x', 'y'])

        prob.model.set_input_defaults('x', np.ones((3,2,2)) * 3.0)
        prob.model.set_input_defaults('y', np.ones((3,2,2)) * -4.0)

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['singular_jac_behavior'] = 'error'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)

        # Design var z does not affect any quantities.
        prob.model.add_design_var('z', lower=-50, upper=50, indices=[2,5,6], flat_indices=True)

        prob.model.add_objective('parab.f_xy', index=6, flat_indices=True)

        prob.model.add_constraint('const.g', lower=0, upper=10.)

        prob.setup()

        with self.assertRaises(RuntimeError) as msg:
            prob.run_driver()

        self.assertEqual(str(msg.exception),
                         "Design variables [('z', inds=[(0, 1, 0), (1, 0, 1), (1, 1, 0)])] have no impact on the constraints or objective.")

    def test_singular_jac_error_desvars_multidim_indices_con(self):
        prob = om.Problem()
        prob.model.add_subsystem('parab',
                                 om.ExecComp(['f_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0 - z',
                                              'f_z = z * 0.0'], shape=(3,2,2)),
                                 promotes_inputs=['x', 'y', 'z'])

        prob.model.add_subsystem('const', om.ExecComp('g = x + y', shape=(3,2,2)), promotes_inputs=['x', 'y'])

        prob.model.set_input_defaults('x', np.ones((3,2,2)) * 3.0)
        prob.model.set_input_defaults('y', np.ones((3,2,2)) * -4.0)

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['singular_jac_behavior'] = 'error'

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_design_var('z', lower=-50, upper=50)

        # objective parab.f_z is not impacted by any quantities.
        prob.model.add_objective('parab.f_z', index=6, flat_indices=True)

        prob.model.add_constraint('const.g', lower=0, upper=10.)

        prob.setup()

        with self.assertRaises(RuntimeError) as msg:
            prob.run_driver()

        self.assertEqual(str(msg.exception),
                         "Constraints or objectives [('parab.f_z', inds=[(1, 1, 0)])] cannot be impacted by the design variables of the problem.")

    @unittest.skipUnless(Version(scipy_version) >= Version("1.2"),
                         "scipy >= 1.2 is required.")
    def test_feature_shgo_rastrigin(self):
        # Source of example: https://stefan-endres.github.io/shgo/

        size = 3  # size of the design variable

        def rastrigin(x):
            a = 10  # constant
            return np.sum(np.square(x) - a * np.cos(2 * np.pi * x)) + a * np.size(x)

        class Rastrigin(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', np.ones(size))
                self.add_output('f', 0.0)

            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                x = inputs['x']
                outputs['f'] = rastrigin(x)

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('rastrigin', Rastrigin(), promotes=['*'])

        prob.driver = driver = om.ScipyOptimizeDriver()
        driver.options['optimizer'] = 'shgo'
        driver.options['disp'] = False
        driver.opt_settings['maxtime'] = 10  # seconds
        driver.opt_settings['iters'] = 3
        driver.opt_settings['maxiter'] = None

        model.add_design_var('x', lower=-5.12*np.ones(size), upper=5.12*np.ones(size))
        model.add_objective('f')
        prob.setup()

        prob.set_val('x', np.ones(size))
        prob.run_driver()

        assert_near_equal(prob.get_val('x'), np.zeros(size), 1e-6)
        assert_near_equal(prob.get_val('f'), 0.0, 1e-6)

    def test_multiple_constraints_scipy(self):

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
        p.model.add_constraint('exec.z', indices=[10], upper=0)
        p.model.add_constraint('exec.z', indices=[-1], equals=25, alias="ALIAS_TEST")

        p.driver = om.ScipyOptimizeDriver()

        p.setup()

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_driver()

        assert_near_equal(p.get_val('exec.z')[0], 25)
        assert_near_equal(p.get_val('exec.z')[10], -11)

    def test_con_and_obj_duplicate(self):

        p = om.Problem()

        exec = om.ExecComp(['y = x**2',
                            'z = a + x**2'],
                            a={'shape': (1,)},
                            y={'shape': (101,)},
                            x={'shape': (101,)},
                            z={'shape': (101,)})

        p.model.add_subsystem('exec', exec)

        p.model.add_design_var('exec.a', lower=-1000, upper=1000)
        p.model.add_objective('exec.z', index=50)
        p.model.add_constraint('exec.z', indices=[0], equals=25, alias='ALIAS_TEST')

        p.driver = om.ScipyOptimizeDriver()

        p.setup()

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_driver()

        assert_near_equal(p.get_val('exec.z')[0], 25)
        assert_near_equal(p.get_val('exec.z')[50], -75)


if __name__ == "__main__":
    unittest.main()
