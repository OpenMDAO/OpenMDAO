""" Unit tests for DifferentialEvolutionDriver."""

import unittest
import os

import numpy as np

import openmdao.api as om

from openmdao.core.constants import INF_BOUND

from openmdao.drivers.differential_evolution_driver import DifferentialEvolution

from openmdao.test_suite.components.branin import Branin
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.paraboloid_distributed import DistParab
from openmdao.test_suite.components.sellar_feature import SellarMDA

from openmdao.utils.general_utils import run_driver
from openmdao.utils.testing_utils import use_tempdirs, set_env_vars_context
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.utils.mpi import MPI
try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

try:
    import pyDOE2
except ImportError:
    pyDOE2 = None

extra_prints = False  # enable printing results


def _test_func_name(func, num, param):
    args = []
    for p in param.args:
        if p and p == INF_BOUND:
            args.append('INF_BOUND')
        elif p and p == -INF_BOUND:
            args.append('-INF_BOUND')
        else:
            args.append(str(p))
    return func.__name__ + '_' + '_'.join(args)


class TestErrors(unittest.TestCase):

    @unittest.skipIf(pyDOE2, "only runs if 'pyDOE2' is not installed")
    def test_no_pyDOE2(self):
        with self.assertRaises(RuntimeError) as err:
            DifferentialEvolution(lambda: 0)

        self.assertEqual(str(err.exception),
                         "DifferentialEvolution requires the 'pyDOE2' package, "
                         "which can be installed with one of the following commands:\n"
                         "    pip install openmdao[doe]\n"
                         "    pip install pyDOE2")

        with self.assertRaises(RuntimeError) as err:
            om.DifferentialEvolutionDriver()

        self.assertEqual(str(err.exception),
                         "DifferentialEvolutionDriver requires the 'pyDOE2' package, "
                         "which can be installed with one of the following commands:\n"
                         "    pip install openmdao[doe]\n"
                         "    pip install pyDOE2")


@unittest.skipUnless(pyDOE2, "requires 'pyDOE2', install openmdao[doe]")
class TestDifferentialEvolution(unittest.TestCase):

    def setUp(self):
        import os  # import needed in setup for tests in documentation
        os.environ['DifferentialEvolutionDriver_seed'] = '11'  # make RNG repeatable

    def tearDown(self):
        del os.environ['DifferentialEvolutionDriver_seed']  # clean up environment

    def test_basic_with_assert(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Branin(), promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_design_var('xI', lower=-5.0, upper=10.0)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.DifferentialEvolutionDriver()

        prob.setup()
        prob.run_driver()

        # Optimal solution (actual optimum, not the optimal with integer inputs as found by SimpleGA)
        assert_near_equal(prob['comp.f'], 0.397887, 1e-4)

    def test_rastrigin(self):

        ORDER = 6  # dimension of problem
        span = 5   # upper and lower limits

        class RastriginComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.zeros(ORDER))
                self.add_output('y', 0.0)

            def compute(self, inputs, outputs):
                x = inputs['x']

                # nth dimensional Rastrigin function, array input and scalar output
                # global minimum at f(0,0,0...) = 0
                n = len(x)
                s = 10 * n
                for i in range(n):
                    if np.abs(x[i]) < 1e-200:  # avoid underflow runtime warnings from squaring tiny numbers
                        x[i] = 0.0
                    s += x[i] * x[i] - 10 * np.cos(2 * np.pi * x[i])

                outputs['y'] = s

        prob = om.Problem()

        prob.model.add_subsystem('rastrigin', RastriginComp(), promotes_inputs=['x'])
        prob.model.add_design_var('x',
                                  lower=-span * np.ones(ORDER),
                                  upper=span * np.ones(ORDER))
        prob.model.add_objective('rastrigin.y')

        prob.driver = om.DifferentialEvolutionDriver()
        prob.driver.options['max_gen'] = 400
        prob.driver.options['Pc'] = 0.5
        prob.driver.options['F'] = 0.5

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob['rastrigin.y'], 0.0, 1e-6)
        assert_near_equal(prob['x'], np.zeros(ORDER), 1e-6)

    def test_rosenbrock(self):
        ORDER = 6  # dimension of problem
        span = 2   # upper and lower limits

        class RosenbrockComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.zeros(ORDER))
                self.add_output('y', 0.0)

            def compute(self, inputs, outputs):
                x = inputs['x']

                # nth dimensional Rosenbrock function, array input and scalar output
                # global minimum at f(1,1,1...) = 0
                n = len(x)
                assert (n > 1)
                s = 0
                for i in range(n - 1):
                    s += 100 * (x[i + 1] - x[i] * x[i]) ** 2 + (1 - x[i]) ** 2

                outputs['y'] = s

        prob = om.Problem()

        prob.model.add_subsystem('rosenbrock', RosenbrockComp(), promotes_inputs=['x'])
        prob.model.add_design_var('x',
                                  lower=-span * np.ones(ORDER),
                                  upper=span * np.ones(ORDER))
        prob.model.add_objective('rosenbrock.y')

        prob.driver = om.DifferentialEvolutionDriver()
        prob.driver.options['max_gen'] = 800

        prob.setup()
        prob.run_driver()

        # show results
        if extra_prints:
            print('rosenbrock.y', prob['rosenbrock.y'])
            print('x', prob['x'])
            print('objective function calls', prob.driver.iter_count, '\n')

        assert_near_equal(prob['rosenbrock.y'], 0.0, 1e-5)
        assert_near_equal(prob['x'], np.ones(ORDER), 1e-3)

    def test_simple_test_func(self):
        class MyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.zeros((2, )))

                self.add_output('a', 0.0)
                self.add_output('b', 0.0)
                self.add_output('c', 0.0)
                self.add_output('d', 0.0)

            def compute(self, inputs, outputs):
                x = inputs['x']

                outputs['a'] = (2.0*x[0] - 3.0*x[1])**2
                outputs['b'] = 18.0 - 32.0*x[0] + 12.0*x[0]**2 + 48.0*x[1] - 36.0*x[0]*x[1] + 27.0*x[1]**2
                outputs['c'] = (x[0] + x[1] + 1.0)**2
                outputs['d'] = 19.0 - 14.0*x[0] + 3.0*x[0]**2 - 14.0*x[1] + 6.0*x[0]*x[1] + 3.0*x[1]**2

        prob = om.Problem()

        prob.model.add_subsystem('comp', MyComp(), promotes_inputs=['x'])
        prob.model.add_subsystem('obj', om.ExecComp('f=(30 + a*b)*(1 + c*d)'))

        prob.model.connect('comp.a', 'obj.a')
        prob.model.connect('comp.b', 'obj.b')
        prob.model.connect('comp.c', 'obj.c')
        prob.model.connect('comp.d', 'obj.d')

        # Played with bounds so we don't get subtractive cancellation of tiny numbers.
        prob.model.add_design_var('x', lower=np.array([0.2, -1.0]), upper=np.array([1.0, -0.2]))
        prob.model.add_objective('obj.f')

        prob.driver = om.DifferentialEvolutionDriver()
        prob.driver.options['max_gen'] = 75

        prob.setup()

        prob.set_val('x', [0.3, -0.3])

        prob.run_driver()

        if extra_prints:
            print('obj.f', prob['obj.f'])
            print('x', prob['x'])

        assert_near_equal(prob['obj.f'], 12.37306086, 1e-4)
        assert_near_equal(prob['x'][0], 0.2, 1e-4)
        assert_near_equal(prob['x'][1], -0.88653391, 1e-4)

    def test_analysis_error(self):
        class ValueErrorComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', 1.0)
                self.add_output('f', 1.0)

            def compute(self, inputs, outputs):
                raise ValueError

        prob = om.Problem()

        prob.model.add_subsystem('comp', ValueErrorComp(), promotes_inputs=['x'])
        prob.model.add_design_var('x', lower=-5.0, upper=10.0)
        prob.model.add_objective('comp.f')

        prob.driver = om.DifferentialEvolutionDriver(max_gen=75, pop_size=25)

        prob.setup()
        # prob.run_driver()
        self.assertRaises(ValueError, prob.run_driver)

    def test_vector_desvars_multiobj(self):
        prob = om.Problem()

        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', 3)
        indeps.add_output('y', [4.0, 1.0])

        prob.model.add_subsystem('paraboloid1',
                                 om.ExecComp('f = (x+5)**2- 3'))
        prob.model.add_subsystem('paraboloid2',
                                 om.ExecComp('f = (y[0]-3)**2 + (y[1]-1)**2 - 3',
                                             y=[0, 0]))
        prob.model.connect('indeps.x', 'paraboloid1.x')
        prob.model.connect('indeps.y', 'paraboloid2.y')

        prob.driver = om.DifferentialEvolutionDriver()

        prob.model.add_design_var('indeps.x', lower=-5, upper=5)
        prob.model.add_design_var('indeps.y', lower=[-10, 0], upper=[10, 3])
        prob.model.add_objective('paraboloid1.f')
        prob.model.add_objective('paraboloid2.f')
        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('indeps.x', prob['indeps.x'])
            print('indeps.y', prob['indeps.y'])

        np.testing.assert_array_almost_equal(prob['indeps.x'], -5)
        np.testing.assert_array_almost_equal(prob['indeps.y'], [3, 1])

    def test_missing_objective(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('x', om.IndepVarComp('x', 2.0), promotes=['*'])
        model.add_subsystem('f_x', Paraboloid(), promotes=['*'])

        model.add_design_var('x', lower=-50, upper=50)

        prob.driver = om.DifferentialEvolutionDriver()

        prob.setup()

        with self.assertRaises(Exception) as raises_msg:
            prob.run_driver()

        exception = raises_msg.exception

        msg = "Driver requires objective to be declared"

        self.assertEqual(exception.args[0], msg)

    def test_invalid_desvar_values(self):

        expected_err = ("The following design variable initial conditions are out of their specified "
                        "bounds:"
                        "\n  indeps.y"
                        "\n    val: [4.  3.1]"
                        "\n    lower: [-10.   0.]"
                        "\n    upper: [10.  3.]"
                        "\nSet the initial value of the design variable to a valid value or set "
                        "the driver option['invalid_desvar_behavior'] to 'ignore'."
                        "\nThis warning will become an error by default in OpenMDAO version 3.25.")

        for option in ['warn', 'raise', 'ignore']:
            with self.subTest(f'invalid_desvar_behavior = {option}'):

                prob = om.Problem()

                indeps = prob.model.add_subsystem('indeps', om.IndepVarComp())
                indeps.add_output('x', 3)
                indeps.add_output('y', [4.0, 3.1])

                prob.model.add_subsystem('paraboloid1',
                                         om.ExecComp('f = (x+5)**2- 3'))
                prob.model.add_subsystem('paraboloid2',
                                         om.ExecComp('f = (y[0]-3)**2 + (y[1]-1)**2 - 3',
                                                     y=[0, 0]))
                prob.model.connect('indeps.x', 'paraboloid1.x')
                prob.model.connect('indeps.y', 'paraboloid2.y')

                prob.driver = om.DifferentialEvolutionDriver(invalid_desvar_behavior=option)

                prob.model.add_design_var('indeps.x', lower=-5, upper=5)
                prob.model.add_design_var('indeps.y', lower=[-10, 0], upper=[10, 3])
                prob.model.add_objective('paraboloid1.f')
                prob.model.add_objective('paraboloid2.f')
                prob.setup()

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

                    if extra_prints:
                        print('indeps.x', prob['indeps.x'])
                        print('indeps.y', prob['indeps.y'])

                    np.testing.assert_array_almost_equal(prob['indeps.x'], -5)
                    np.testing.assert_array_almost_equal(prob['indeps.y'], [3, 1])

    @parameterized.expand([
        (None, None),
        (INF_BOUND, INF_BOUND),
        (None, INF_BOUND),
        (None, -INF_BOUND),
        (INF_BOUND, None),
        (-INF_BOUND, None),
    ],
    name_func=_test_func_name)
    def test_inf_desvar(self, lower, upper):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('x', om.IndepVarComp('x', 2.0), promotes=['*'])
        model.add_subsystem('f_x', Paraboloid(), promotes=['*'])

        model.add_objective('f_xy')
        model.add_design_var('x', lower=lower, upper=upper)

        prob.driver = om.DifferentialEvolutionDriver()

        prob.setup()

        with self.assertRaises(ValueError) as err:
            prob.final_setup()

        # A value of None for lower and upper is changed to +/- INF_BOUND in add_design_var()
        if lower == None:
            lower = -INF_BOUND
        if upper == None:
            upper = INF_BOUND

        msg = ("Invalid bounds for design variable 'x.x'. When using "
               "DifferentialEvolutionDriver, values for both 'lower' and 'upper' "
               f"must be specified between +/-INF_BOUND ({INF_BOUND}), "
               f"but they are: lower={lower}, upper={upper}.")

        self.maxDiff = None
        self.assertEqual(err.exception.args[0], msg)

    def test_vectorized_constraints(self):
        prob = om.Problem()

        dim = 2
        prob.model.add_subsystem('x', om.IndepVarComp('x', np.ones(dim)), promotes=['*'])
        prob.model.add_subsystem('f_x', om.ExecComp('f_x = sum(x * x)', x=np.ones(dim), f_x=1.0), promotes=['*'])
        prob.model.add_subsystem('g_x', om.ExecComp('g_x = 1 - x', x=np.ones(dim), g_x=np.zeros(dim)), promotes=['*'])

        prob.driver = om.DifferentialEvolutionDriver()

        prob.model.add_design_var('x', lower=-10, upper=10)
        prob.model.add_objective('f_x')
        prob.model.add_constraint('g_x', upper=np.zeros(dim))

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('x', prob['x'])

        # Check that the constraint is approximately satisfied (x >= 1)
        for i in range(dim):
            self.assertLessEqual(1.0 - 1e-6, prob["x"][i])


@unittest.skipUnless(pyDOE2, "requires 'pyDOE2', install openmdao[doe]")
class TestDriverOptionsDifferentialEvolution(unittest.TestCase):

    def setUp(self):
        os.environ['DifferentialEvolutionDriver_seed'] = '11'

    def tearDown(self):
        del os.environ['DifferentialEvolutionDriver_seed']  # clean up environment

    def test_driver_options(self):
        """Tests if F and Pc options can be set."""
        prob = om.Problem()

        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('x', 1.)

        prob.model.add_subsystem('model', om.ExecComp('y=x**2'), promotes=['*'])

        prob.driver = om.DifferentialEvolutionDriver()
        prob.driver.options['F'] = 0.123
        prob.driver.options['Pc'] = 0.0123
        prob.driver.options['max_gen'] = 5

        prob.model.add_design_var('x', lower=-10., upper=10.)
        prob.model.add_objective('y')

        prob.setup()
        prob.run_driver()

        self.assertEqual(prob.driver.options['F'], 0.123)
        self.assertEqual(prob.driver.options['Pc'], 0.0123)


@unittest.skipUnless(pyDOE2, "requires 'pyDOE2', install openmdao[doe]")
class TestMultiObjectiveDifferentialEvolution(unittest.TestCase):

    def setUp(self):
        os.environ['DifferentialEvolutionDriver_seed'] = '11'

    def tearDown(self):
        del os.environ['DifferentialEvolutionDriver_seed']  # clean up environment

    def test_multi_obj(self):
        class Box(om.ExplicitComponent):
            def setup(self):
                self.add_input('length', val=1.)
                self.add_input('width', val=1.)
                self.add_input('height', val=1.)

                self.add_output('front_area', val=1.0)
                self.add_output('top_area', val=1.0)
                self.add_output('area', val=1.0)
                self.add_output('volume', val=1.)

            def compute(self, inputs, outputs):
                length = inputs['length']
                width = inputs['width']
                height = inputs['height']

                outputs['top_area'] = length * width
                outputs['front_area'] = length * height
                outputs['area'] = 2*length*height + 2*length*width + 2*height*width
                outputs['volume'] = length*height*width

        prob = om.Problem()
        prob.model.add_subsystem('box', Box(), promotes=['*'])

        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('length', 1.5)
        indeps.add_output('width', 1.5)
        indeps.add_output('height', 1.5)

        # setup the optimization
        prob.driver = om.DifferentialEvolutionDriver()
        prob.driver.options['max_gen'] = 100
        prob.driver.options['multi_obj_exponent'] = 1.
        prob.driver.options['penalty_parameter'] = 10.
        prob.driver.options['multi_obj_weights'] = {'box.front_area': 0.1,
                                                    'box.top_area': 0.9}
        prob.driver.options['multi_obj_exponent'] = 1

        prob.model.add_design_var('length', lower=0.1, upper=2.)
        prob.model.add_design_var('width', lower=0.1, upper=2.)
        prob.model.add_design_var('height', lower=0.1, upper=2.)
        prob.model.add_objective('front_area', scaler=-1)  # maximize
        prob.model.add_objective('top_area', scaler=-1)  # maximize
        prob.model.add_constraint('volume', upper=1.)

        # run #1
        prob.setup()
        prob.run_driver()
        front = prob['front_area']
        top = prob['top_area']
        l1 = prob['length']
        w1 = prob['width']
        h1 = prob['height']

        if extra_prints:
            print('Box dims: ', l1, w1, h1)
            print('Front and top area: ', front, top)
            print('Volume: ', prob['volume'])  # should be around 1

        # run #2
        # weights changed
        prob2 = om.Problem()
        prob2.model.add_subsystem('box', Box(), promotes=['*'])

        indeps2 = prob2.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps2.add_output('length', 1.5)
        indeps2.add_output('width', 1.5)
        indeps2.add_output('height', 1.5)

        # setup the optimization
        prob2.driver = om.DifferentialEvolutionDriver()
        prob2.driver.options['max_gen'] = 100
        prob2.driver.options['multi_obj_exponent'] = 1.
        prob2.driver.options['penalty_parameter'] = 10.
        prob2.driver.options['multi_obj_weights'] = {'box.front_area': 0.9,
                                                     'box.top_area': 0.1}
        prob2.driver.options['multi_obj_exponent'] = 1

        prob2.model.add_design_var('length', lower=0.1, upper=2.)
        prob2.model.add_design_var('width', lower=0.1, upper=2.)
        prob2.model.add_design_var('height', lower=0.1, upper=2.)
        prob2.model.add_objective('front_area', scaler=-1)  # maximize
        prob2.model.add_objective('top_area', scaler=-1)  # maximize
        prob2.model.add_constraint('volume', upper=1.)

        # run #1
        prob2.setup()
        prob2.run_driver()
        front2 = prob2['front_area']
        top2 = prob2['top_area']
        l2 = prob2['length']
        w2 = prob2['width']
        h2 = prob2['height']

        if extra_prints:
            print('Box dims: ', l2, w2, h2)
            print('Front and top area: ', front2, top2)
            print('Volume: ', prob['volume'])  # should be around 1

        self.assertGreater(w1, w2)  # front area does not depend on width
        self.assertGreater(h2, h1)  # top area does not depend on height


@unittest.skipUnless(pyDOE2, "requires 'pyDOE2', install openmdao[doe]")
class TestConstrainedDifferentialEvolution(unittest.TestCase):

    def setUp(self):
        # This env var was changed from '11' to '0' to avoid having to change test results.
        # The old implementation of the seed calculation erroneously set the seed to 0
        # regardless of the value of the random_state passed in (in the non-MPI case only).
        os.environ['DifferentialEvolutionDriver_seed'] = '0'

    def tearDown(self):
        del os.environ['DifferentialEvolutionDriver_seed']  # clean up environment

    def tearDown(self):
        del os.environ['DifferentialEvolutionDriver_seed']  # clean up environment

    def test_constrained_with_penalty(self):
        class Cylinder(om.ExplicitComponent):
            def setup(self):
                self.add_input('radius', val=1.0)
                self.add_input('height', val=1.0)

                self.add_output('Area', val=1.0)
                self.add_output('Volume', val=1.0)

            def compute(self, inputs, outputs):
                radius = inputs['radius']
                height = inputs['height']

                area = height * radius * 2 * 3.14 + 3.14 * radius ** 2 * 2
                volume = 3.14 * radius ** 2 * height
                outputs['Area'] = area
                outputs['Volume'] = volume

        prob = om.Problem()
        prob.model.add_subsystem('cylinder', Cylinder(), promotes=['*'])

        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('radius', 2.)  # height
        indeps.add_output('height', 3.)  # radius

        # setup the optimization
        driver = prob.driver = om.DifferentialEvolutionDriver()
        prob.driver.options['penalty_parameter'] = 3.
        prob.driver.options['penalty_exponent'] = 1.
        prob.driver.options['max_gen'] = 50

        prob.model.add_design_var('radius', lower=0.5, upper=5.)
        prob.model.add_design_var('height', lower=0.5, upper=5.)
        prob.model.add_objective('Area')
        prob.model.add_constraint('Volume', lower=10.)

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('radius', prob['radius'])  # exact solution is (5/pi)^(1/3) ~= 1.167
            print('height', prob['height'])  # exact solution is 2*radius
            print('Area', prob['Area'])
            print('Volume', prob['Volume'])  # should be around 10

        self.assertTrue(driver.supports["equality_constraints"], True)
        self.assertTrue(driver.supports["inequality_constraints"], True)
        # check that it is not going to the unconstrained optimum
        self.assertGreater(prob['radius'], 1.)
        self.assertGreater(prob['height'], 1.)

    def test_driver_supports(self):
        prob = om.Problem()

        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])

        # setup the optimization
        driver = prob.driver = om.DifferentialEvolutionDriver()

        with self.assertRaises(KeyError) as raises_msg:
            prob.driver.supports['equality_constraints'] = False

        exception = raises_msg.exception

        msg = "DifferentialEvolutionDriver: Tried to set read-only option 'equality_constraints'."

        self.assertEqual(exception.args[0], msg)

    def test_constrained_without_penalty(self):
        class Cylinder(om.ExplicitComponent):
            def setup(self):
                self.add_input('radius', val=1.0)
                self.add_input('height', val=1.0)

                self.add_output('Area', val=1.0)
                self.add_output('Volume', val=1.0)

            def compute(self, inputs, outputs):
                radius = inputs['radius']
                height = inputs['height']

                area = height * radius * 2 * 3.14 + 3.14 * radius ** 2 * 2
                volume = 3.14 * radius ** 2 * height
                outputs['Area'] = area
                outputs['Volume'] = volume

        prob = om.Problem()
        prob.model.add_subsystem('cylinder', Cylinder(), promotes=['*'])

        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('radius', 2.)  # height
        indeps.add_output('height', 3.)  # radius

        # setup the optimization
        driver = prob.driver = om.DifferentialEvolutionDriver()
        prob.driver.options['penalty_parameter'] = 0.  # no penalty, same as unconstrained
        prob.driver.options['penalty_exponent'] = 1.
        prob.driver.options['max_gen'] = 50

        prob.model.add_design_var('radius', lower=0.5, upper=5.)
        prob.model.add_design_var('height', lower=0.5, upper=5.)
        prob.model.add_objective('Area')
        prob.model.add_constraint('Volume', lower=10.)

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('radius', prob['radius'])  # exact solution is (5/pi)^(1/3) ~= 1.167
            print('height', prob['height'])  # exact solution is 2*radius
            print('Area', prob['Area'])
            print('Volume', prob['Volume'])  # should be around 10

        self.assertTrue(driver.supports["equality_constraints"], True)
        self.assertTrue(driver.supports["inequality_constraints"], True)
        # it is going to the unconstrained optimum
        self.assertAlmostEqual(prob['radius'], 0.5, 1)
        self.assertAlmostEqual(prob['height'], 0.5, 1)

    def test_no_constraint(self):
        class Cylinder(om.ExplicitComponent):
            def setup(self):
                self.add_input('radius', val=1.0)
                self.add_input('height', val=1.0)

                self.add_output('Area', val=1.0)
                self.add_output('Volume', val=1.0)

            def compute(self, inputs, outputs):
                radius = inputs['radius']
                height = inputs['height']

                area = height * radius * 2 * 3.14 + 3.14 * radius ** 2 * 2
                volume = 3.14 * radius ** 2 * height
                outputs['Area'] = area
                outputs['Volume'] = volume

        prob = om.Problem()
        prob.model.add_subsystem('cylinder', Cylinder(), promotes=['*'])

        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('radius', 2.)  # height
        indeps.add_output('height', 3.)  # radius

        # setup the optimization
        driver = prob.driver = om.DifferentialEvolutionDriver()
        prob.driver.options['penalty_parameter'] = 10.  # will have no effect
        prob.driver.options['penalty_exponent'] = 1.
        prob.driver.options['max_gen'] = 50

        prob.model.add_design_var('radius', lower=0.5, upper=5.)
        prob.model.add_design_var('height', lower=0.5, upper=5.)
        prob.model.add_objective('Area')

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('radius', prob['radius'])  # exact solution is (5/pi)^(1/3) ~= 1.167
            print('height', prob['height'])  # exact solution is 2*radius
            print('Area', prob['Area'])
            print('Volume', prob['Volume'])  # should be around 10

        self.assertTrue(driver.supports["equality_constraints"], True)
        self.assertTrue(driver.supports["inequality_constraints"], True)
        self.assertAlmostEqual(prob['radius'], 0.5, 1)  # it is going to the unconstrained optimum
        self.assertAlmostEqual(prob['height'], 0.5, 1)  # it is going to the unconstrained optimum

    def test_multiple_constraints(self):

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
        p.model.add_constraint('exec.z', indices=[-1], lower=0)
        p.model.add_constraint('exec.z', indices=[0], upper=300, alias="ALIAS_TEST")

        p.driver = om.DifferentialEvolutionDriver()

        p.setup()

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_driver()

        assert_near_equal(p.get_val('exec.z')[0], 100.0, tolerance=1e-6)
        assert_near_equal(p.get_val('exec.z')[-1], 100.0, tolerance=1e-6)

    def test_same_cons_and_obj(self):

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
        p.model.add_constraint('exec.z', indices=[0], upper=300, alias="ALIAS_TEST")

        p.driver = om.DifferentialEvolutionDriver()

        p.setup()

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_driver()

        assert_near_equal(p.get_val('exec.z')[0], -900)
        assert_near_equal(p.get_val('exec.z')[50], -1000)

    @parameterized.expand([
        (None, -INF_BOUND, INF_BOUND),
        (INF_BOUND, None, None),
        (-INF_BOUND, None, None),
    ],
    name_func=_test_func_name)
    def test_inf_constraints(self, equals, lower, upper):
        # define paraboloid problem with constraint
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('parab', Paraboloid(), promotes_inputs=['x', 'y'])
        model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])
        model.set_input_defaults('x', 3.0)
        model.set_input_defaults('y', -4.0)

        # setup the optimization
        prob.driver = om.DifferentialEvolutionDriver()
        model.add_objective('parab.f_xy')
        model.add_design_var('x', lower=-50, upper=50)
        model.add_design_var('y', lower=-50, upper=50)
        model.add_constraint('const.g', equals=equals, lower=lower, upper=upper)

        prob.setup()

        with self.assertRaises(ValueError) as err:
            prob.final_setup()

        # A value of None for lower and upper is changed to +/- INF_BOUND in add_constraint()
        if lower == None:
            lower = -INF_BOUND
        if upper == None:
            upper = INF_BOUND

        msg = ("Invalid bounds for constraint 'const.g'. "
               "When using DifferentialEvolutionDriver, the value for "
               "'equals', 'lower' or 'upper' must be specified between "
               f"+/-INF_BOUND ({INF_BOUND}), but they are: "
               f"equals={equals}, lower={lower}, upper={upper}.")

        self.maxDiff = None
        self.assertEqual(err.exception.args[0], msg)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
@unittest.skipUnless(pyDOE2, "requires 'pyDOE2', install openmdao[doe]")
class MPITestDifferentialEvolution(unittest.TestCase):
    N_PROCS = 2

    def setUp(self):
        os.environ['DifferentialEvolutionDriver_seed'] = '11'

    def tearDown(self):
        del os.environ['DifferentialEvolutionDriver_seed']  # clean up environment

    def test_mpi_bug_solver(self):
        # This test verifies that mpi doesn't hang due to collective calls in the solver.
        prob = om.Problem()
        prob.model = SellarMDA()

        prob.model.add_design_var('x', lower=0, upper=10)
        prob.model.add_design_var('z', lower=0, upper=10)
        prob.model.add_objective('obj')

        prob.driver = om.DifferentialEvolutionDriver(run_parallel=True)

        # Set these low because we don't need to run long.
        prob.driver.options['max_gen'] = 2
        prob.driver.options['pop_size'] = 5

        prob.setup()
        prob.set_solver_print(level=0)

        prob.run_driver()

    def test_random_state_bug(self):
        # this test passes if it raises no exceptions
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Branin(), promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_design_var('xI', lower=-5.0, upper=10.0)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.DifferentialEvolutionDriver()
        prob.driver.options['max_gen'] = 5
        prob.driver.options['run_parallel'] = True

        prob.setup()
        prob.run_driver()


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
@unittest.skipUnless(pyDOE2, "requires 'pyDOE2', install openmdao[doe]")
class MPITestDifferentialEvolutionNoSetSeed(unittest.TestCase):
    N_PROCS = 2

    def test_random_state_bug(self):
        # this test passes if it raises no exceptions
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Branin(), promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_design_var('xI', lower=-5.0, upper=10.0)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.DifferentialEvolutionDriver()
        prob.driver.options['max_gen'] = 5
        prob.driver.options['run_parallel'] = True

        prob.setup()
        prob.run_driver()


class D1(om.ExplicitComponent):
    def setup(self):
        comm = self.comm
        rank = comm.rank

        if rank == 1:
            start = 1
            end = 2
        else:
            start = 0
            end = 1

        self.add_input('y2', np.ones((1, ), float), distributed=True,
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('x', np.ones((1, ), float), distributed=True)

        self.add_output('y1', np.ones((1, ), float), distributed=True)

        self.declare_partials('y1', ['y2', 'x'])

    def compute(self, inputs, outputs):
        y2 = inputs['y2']
        x = inputs['x']

        if self.comm.rank == 1:
            outputs['y1'] = 18.0 - 0.2*y2 + 2*x
        else:
            outputs['y1'] = 28.0 - 0.2*y2 + x

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        y2 = inputs['y2']
        x = inputs['x']

        partials['y1', 'y2'] = -0.2
        if self.comm.rank == 1:
            partials['y1', 'x'] = 2.0
        else:
            partials['y1', 'x'] = 1.0


class D2(om.ExplicitComponent):
    def setup(self):
        comm = self.comm
        rank = comm.rank

        if rank == 1:
            start = 1
            end = 2
        else:
            start = 0
            end = 1

        self.add_input('y1', np.ones((1, ), float), distributed=True,
                       src_indices=np.arange(start, end, dtype=int))

        self.add_output('y2', np.ones((1, ), float), distributed=True)

        self.declare_partials('y2', ['y1'])

    def compute(self, inputs, outputs):
        y1 = inputs['y1']

        if self.comm.rank == 1:
            outputs['y2'] = y2 = y1**.5 - 3
        else:
            outputs['y2'] = y1**.5 + 7

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        y1 = inputs['y1']

        partials['y2', 'y1'] = 0.5 / y1**.5


class Summer(om.ExplicitComponent):
    def setup(self):
        self.add_input('y1', val=np.zeros((2, )))
        self.add_input('y2', val=np.zeros((2, )))
        self.add_output('obj', 0.0, shape=1)

        self.declare_partials('obj', 'y1', rows=np.array([0, 0]), cols=np.array([0, 1]), val=np.ones((2, )))

    def compute(self, inputs, outputs):
        outputs['obj'] = np.sum(inputs['y1']) + np.sum(inputs['y2'])


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
@unittest.skipUnless(pyDOE2, "requires 'pyDOE2', install openmdao[doe]")
@use_tempdirs
class MPITestDifferentialEvolution4Procs(unittest.TestCase):
    N_PROCS = 4

    def setUp(self):
        os.environ['DifferentialEvolutionDriver_seed'] = '11'

    def tearDown(self):
        del os.environ['DifferentialEvolutionDriver_seed']  # clean up environment

    def test_indivisible_error(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('par', om.ParallelGroup())

        prob.driver = om.DifferentialEvolutionDriver()
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = 3

        with self.assertRaises(RuntimeError) as context:
            prob.setup()

        self.assertEqual(str(context.exception),
                         "The total number of processors is not evenly divisible by the "
                         "specified number of processors per model.\n Provide a number of "
                         "processors that is a multiple of 3, or specify a number "
                         "of processors per model that divides into 4.")

    def test_concurrent_eval_padded(self):
        # This test only makes sure we don't lock up if we overallocate our integer desvar space
        # to the next power of 2.

        class GAGroup(om.Group):

            def setup(self):
                self.add_subsystem('p1', om.IndepVarComp('x', 1.0))
                self.add_subsystem('p2', om.IndepVarComp('y', 1.0))
                self.add_subsystem('p3', om.IndepVarComp('z', 1.0))

                self.add_subsystem('comp', om.ExecComp(['f = x + y + z']))

                self.add_design_var('p1.x', lower=-100, upper=100)
                self.add_design_var('p2.y', lower=-100, upper=100)
                self.add_design_var('p3.z', lower=-100, upper=100)
                self.add_objective('comp.f')

        prob = om.Problem()
        prob.model = GAGroup()

        driver = prob.driver = om.DifferentialEvolutionDriver()
        driver.options['max_gen'] = 5
        driver.options['pop_size'] = 40
        driver.options['run_parallel'] = True

        prob.setup()

        # No meaningful result from a short run; just make sure we don't hang.
        prob.run_driver()

    def test_proc_per_model(self):
        # Test that we can run a GA on a distributed component without lockups.
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p', om.IndepVarComp('x', 3.0), promotes=['x'])

        model.add_subsystem('d1', D1(), promotes=['*'])
        model.add_subsystem('d2', D2(), promotes=['*'])

        model.add_subsystem('obj_comp', Summer(), promotes_outputs=['*'])
        model.promotes('obj_comp', inputs=['*'], src_indices=om.slicer[:])
        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        model.linear_solver = om.LinearBlockGS()

        model.add_design_var('x', lower=-0.5, upper=0.5)
        model.add_objective('obj')

        driver = prob.driver = om.DifferentialEvolutionDriver()
        driver.options['pop_size'] = 4
        driver.options['max_gen'] = 3
        driver.options['run_parallel'] = True
        driver.options['procs_per_model'] = 2

        # also check that parallel recording works
        driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.set_solver_print(level=0)

        failed, output = run_driver(prob)

        self.assertFalse(failed)

        # we will have run 2 models in parallel on our 4 procs
        num_models = prob.comm.size // driver.options['procs_per_model']
        self.assertEqual(num_models, 2)

        # a separate case file should have been written by rank 0 of each parallel model
        # (the top two global ranks)
        rank = prob.comm.rank
        filename = "cases.sql_%d" % rank

        if rank < num_models:
            expect_msg = "Cases from rank %d are being written to %s." % (rank, filename)
            self.assertTrue(expect_msg in output)

            cr = om.CaseReader(filename)
            cases = cr.list_cases('driver')

            # check that cases were recorded on this proc
            num_cases = len(cases)
            self.assertTrue(num_cases > 0)
        else:
            self.assertFalse("Cases from rank %d are being written" % rank in output)
            self.assertFalse(os.path.exists(filename))

    def test_distributed_obj(self):
        size = 3
        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('x', np.ones((size, )))
        ivc.add_output('y', np.ones((size, )))
        ivc.add_output('a', -3.0 + 0.6 * np.arange(size))

        model.add_subsystem('p', ivc, promotes=['*'])
        model.add_subsystem("parab", DistParab(arr_size=size, deriv_type='dense'),
                            promotes=['*'])
        model.add_subsystem('sum', om.ExecComp('f_sum = sum(f_xy)',
                                               f_sum=np.ones((size, )),
                                               f_xy=np.ones((size, ))),
                            promotes_outputs=['*'])
        model.promotes('sum', inputs=['f_xy'], src_indices=om.slicer[:])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.driver = om.DifferentialEvolutionDriver()
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = 2
        prob.driver.options['max_gen'] = 250
        prob.driver.options['pop_size'] = 10

        prob.setup()
        prob.run_driver()

        # optimal solution for minimize (x-a)^2 +x*y +(y+4)^2 - 3 for a=[-3, -2.4, -1.8] is:
        # x =    [ 6.66667,  5.86667,  5.06667]
        # y =    [-7.33333, -6.93333, -6.53333]
        # f_xy = [-27.3333, -23.0533, -19.0133]  mean f_xy = -23.1333
        assert_near_equal(prob.get_val('x', get_remote=True),    [ 6.66667,  5.86667,  5.06667], 1e-3)
        assert_near_equal(prob.get_val('y', get_remote=True),    [-7.33333, -6.93333, -6.53333], 1e-3)
        assert_near_equal(prob.get_val('f_xy', get_remote=True), [-27.3333, -23.0533, -19.0133], 1e-3)
        assert_near_equal(np.sum(prob.get_val('f_xy', get_remote=True))/3, -23.1333, 1e-4)


if __name__ == "__main__":
    unittest.main()
