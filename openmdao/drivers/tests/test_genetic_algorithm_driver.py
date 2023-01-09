""" Unit tests for SimpleGADriver."""

import unittest
import os

import numpy as np

import openmdao.api as om

from openmdao.core.constants import INF_BOUND

from openmdao.drivers.genetic_algorithm_driver import GeneticAlgorithm

from openmdao.test_suite.components.branin import Branin, BraninDiscrete
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.paraboloid_distributed import DistParab
from openmdao.test_suite.components.sellar_feature import SellarMDA
from openmdao.test_suite.components.three_bar_truss import ThreeBarTruss

from openmdao.utils.general_utils import run_driver
from openmdao.utils.testing_utils import use_tempdirs, set_env_vars_context
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized

from openmdao.utils.mpi import MPI

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
            GeneticAlgorithm(lambda: 0)

        self.assertEqual(str(err.exception),
                         "GeneticAlgorithm requires the 'pyDOE2' package, "
                         "which can be installed with one of the following commands:\n"
                         "    pip install openmdao[doe]\n"
                         "    pip install pyDOE2")

        with self.assertRaises(RuntimeError) as err:
            om.SimpleGADriver()

        self.assertEqual(str(err.exception),
                         "SimpleGADriver requires the 'pyDOE2' package, "
                         "which can be installed with one of the following commands:\n"
                         "    pip install openmdao[doe]\n"
                         "    pip install pyDOE2")


@unittest.skipUnless(pyDOE2, "requires 'pyDOE2', install openmdao[doe]")
class TestSimpleGA(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        os.environ['SimpleGADriver_seed'] = '11'

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
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', np.array([0.2, -0.2])))
        model.add_subsystem('comp', MyComp())
        model.add_subsystem('obj', om.ExecComp('f=(30 + a*b)*(1 + c*d)'))

        model.connect('px.x', 'comp.x')
        model.connect('comp.a', 'obj.a')
        model.connect('comp.b', 'obj.b')
        model.connect('comp.c', 'obj.c')
        model.connect('comp.d', 'obj.d')

        # Played with bounds so we don't get subtractive cancellation of tiny numbers.
        model.add_design_var('px.x', lower=np.array([0.2, -1.0]), upper=np.array([1.0, -0.2]))
        model.add_objective('obj.f')

        prob.driver = om.SimpleGADriver()
        prob.driver.options['bits'] = {'px.x': 16}
        prob.driver.options['max_gen'] = 75

        prob.driver._randomstate = 11

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('obj.f', prob['obj.f'])
            print('px.x', prob['px.x'])

        # TODO: Satadru listed this solution, but I get a way better one.
        # Solution: xopt = [0.2857, -0.8571], fopt = 23.2933
        assert_near_equal(prob['obj.f'], 12.37306086, 1e-4)
        assert_near_equal(prob['px.x'][0], 0.2, 1e-4)
        assert_near_equal(prob['px.x'][1], -0.88653391, 1e-4)

    def test_mixed_integer_branin(self):
        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('xC', 7.5)
        model.set_input_defaults('xI', 0.0)

        model.add_subsystem('comp', Branin(),
                            promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_design_var('xI', lower=-5.0, upper=10.0)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.SimpleGADriver(max_gen=75, pop_size=25)
        prob.driver.options['bits'] = {'xC': 8}

        prob.driver._randomstate = 1

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('comp.f', prob['comp.f'])

        # Optimal solution
        assert_near_equal(prob['comp.f'], 0.49399549, 1e-4)
        self.assertTrue(int(prob['xI']) in [3, -3])

    def test_mixed_integer_branin_discrete(self):
        prob = om.Problem(reports=('optimizer',))
        model = prob.model

        indep = om.IndepVarComp()
        indep.add_output('xC', val=7.5)
        indep.add_discrete_output('xI', val=0)

        model.add_subsystem('p', indep)
        model.add_subsystem('comp', BraninDiscrete())

        model.connect('p.xI', 'comp.x0')
        model.connect('p.xC', 'comp.x1')

        model.add_design_var('p.xI', lower=-5, upper=10)
        model.add_design_var('p.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.SimpleGADriver(max_gen=75, pop_size=25)
        prob.driver.options['bits'] = {'p.xC': 8}

        prob.driver._randomstate = 1

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('comp.f', prob['comp.f'])
            print('p.xI', prob['p.xI'])

        # Optimal solution
        assert_near_equal(prob['comp.f'], 0.49399549, 1e-4)
        self.assertTrue(prob['p.xI'] in [3, -3])
        self.assertTrue(isinstance(prob['p.xI'], int))

    def test_mixed_integer_3bar(self):
        class ObjPenalty(om.ExplicitComponent):
            """
            Weight objective with penalty on stress constraint.
            """
            def setup(self):
                self.add_input('obj', 0.0)
                self.add_input('stress', val=np.zeros((3, )))

                self.add_output('weighted', 0.0)

            def compute(self, inputs, outputs):
                obj = inputs['obj']
                stress = inputs['stress']

                pen = 0.0
                for j in range(len(stress)):
                    if stress[j] > 1.0:
                        pen += 10.0*(stress[j] - 1.0)**2

                outputs['weighted'] = obj + pen

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('xc_a1', om.IndepVarComp('area1', 5.0, units='cm**2'), promotes=['*'])
        model.add_subsystem('xc_a2', om.IndepVarComp('area2', 5.0, units='cm**2'), promotes=['*'])
        model.add_subsystem('xc_a3', om.IndepVarComp('area3', 5.0, units='cm**2'), promotes=['*'])
        model.add_subsystem('xi_m1', om.IndepVarComp('mat1', 1), promotes=['*'])
        model.add_subsystem('xi_m2', om.IndepVarComp('mat2', 1), promotes=['*'])
        model.add_subsystem('xi_m3', om.IndepVarComp('mat3', 1), promotes=['*'])
        model.add_subsystem('comp', ThreeBarTruss(), promotes=['*'])
        model.add_subsystem('obj_with_penalty', ObjPenalty(), promotes=['*'])

        model.add_design_var('area1', lower=1.2, upper=1.3)
        model.add_design_var('area2', lower=2.0, upper=2.1)
        model.add_design_var('mat1', lower=1, upper=4)
        model.add_design_var('mat2', lower=1, upper=4)
        model.add_design_var('mat3', lower=1, upper=4)
        model.add_objective('weighted')

        prob.driver = om.SimpleGADriver()
        prob.driver.options['bits'] = {'area1': 6,
                                       'area2': 6}
        prob.driver.options['max_gen'] = 75

        prob.driver._randomstate = 1

        prob.setup()
        prob['area3'] = 0.0005
        prob.run_driver()

        if extra_prints:
            print('mass', prob['mass'])
            print('mat1', prob['mat1'])
            print('mat2', prob['mat2'])

        # Note, GA doesn't do so well with the continuous vars, naturally, so we reduce the space
        # as much as we can. Objective is still rather random, but it is close. GA does a great job
        # of picking the correct values for the integer desvars though.
        self.assertLess(prob['mass'], 6.0)
        assert_near_equal(prob['mat1'], 3, 1e-5)
        assert_near_equal(prob['mat2'], 3, 1e-5)
        # Material 3 can be anything

    def test_mixed_integer_3bar_default_bits(self):
        # Tests bug where letting openmdao calculate the bits didn't preserve
        # integer status unless range was a power of 2.

        class ObjPenalty(om.ExplicitComponent):
            """
            Weight objective with penalty on stress constraint.
            """
            def setup(self):
                self.add_input('obj', 0.0)
                self.add_input('stress', val=np.zeros((3, )))

                self.add_output('weighted', 0.0)

            def compute(self, inputs, outputs):
                obj = inputs['obj']
                stress = inputs['stress']

                pen = 0.0
                for j in range(len(stress)):
                    if stress[j] > 1.0:
                        pen += 10.0*(stress[j] - 1.0)**2

                outputs['weighted'] = obj + pen

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('xc_a1', om.IndepVarComp('area1', 5.0, units='cm**2'), promotes=['*'])
        model.add_subsystem('xc_a2', om.IndepVarComp('area2', 5.0, units='cm**2'), promotes=['*'])
        model.add_subsystem('xc_a3', om.IndepVarComp('area3', 5.0, units='cm**2'), promotes=['*'])
        model.add_subsystem('xi_m1', om.IndepVarComp('mat1', 1), promotes=['*'])
        model.add_subsystem('xi_m2', om.IndepVarComp('mat2', 1), promotes=['*'])
        model.add_subsystem('xi_m3', om.IndepVarComp('mat3', 1), promotes=['*'])
        model.add_subsystem('comp', ThreeBarTruss(), promotes=['*'])
        model.add_subsystem('obj_with_penalty', ObjPenalty(), promotes=['*'])

        model.add_design_var('area1', lower=1.2, upper=1.3)
        model.add_design_var('area2', lower=2.0, upper=2.1)
        model.add_design_var('mat1', lower=2, upper=4)
        model.add_design_var('mat2', lower=2, upper=4)
        model.add_design_var('mat3', lower=1, upper=4)
        model.add_objective('weighted')

        prob.driver = om.SimpleGADriver()
        prob.driver.options['bits'] = {'area1': 6,
                                       'area2': 6}
        prob.driver.options['max_gen'] = 75

        prob.driver._randomstate = 1

        prob.setup()
        prob['area3'] = 0.0005
        prob.run_driver()

        if extra_prints:
            print('mass', prob['mass'])
            print('mat1', prob['mat1'])
            print('mat2', prob['mat2'])

        # Note, GA doesn't do so well with the continuous vars, naturally, so we reduce the space
        # as much as we can. Objective is still rather random, but it is close. GA does a great job
        # of picking the correct values for the integer desvars though.
        self.assertLess(prob['mass'], 6.0)
        assert_near_equal(prob['mat1'], 3, 1e-5)
        assert_near_equal(prob['mat2'], 3, 1e-5)
        # Material 3 can be anything

    def test_analysis_error(self):
        class ValueErrorComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', 1.0)
                self.add_output('f', 1.0)

            def compute(self, inputs, outputs):
                raise ValueError

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p', om.IndepVarComp('x', 0.0))
        model.add_subsystem('comp', ValueErrorComp())

        model.connect('p.x', 'comp.x')

        model.add_design_var('p.x', lower=-5.0, upper=10.0)
        model.add_objective('comp.f')

        prob.driver = om.SimpleGADriver(max_gen=75, pop_size=25)
        prob.driver._randomstate = 1
        prob.setup()
        # prob.run_driver()
        self.assertRaises(ValueError, prob.run_driver)

    def test_encode_and_decode(self):
        ga = GeneticAlgorithm(None)
        gen = np.array([[0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1,
                         0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1,
                         1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0],
                        [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
                         0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0,
                         1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0]])
        vlb = np.array([-170.0, -170.0, -170.0, -170.0, -170.0, -170.0])
        vub = np.array([255.0, 255.0, 255.0, 170.0, 170.0, 170.0])
        bits = np.array([9, 9, 9, 9, 9, 9])
        x = np.array([[-69.36399217, 22.12328767, -7.81800391, -66.86888454, 116.77103718, 76.18395303],
                      [248.34637965, 191.79060665, -31.93737769, 97.47553816, 118.76712329, 92.15264188]])

        ga.npop = 2
        ga.lchrom = int(np.sum(bits))
        np.testing.assert_array_almost_equal(x, ga.decode(gen, vlb, vub, bits))
        np.testing.assert_array_almost_equal(gen[0], ga.encode(x[0], vlb, vub, bits))
        np.testing.assert_array_almost_equal(gen[1], ga.encode(x[1], vlb, vub, bits))

        dec = ga.decode(gen, vlb, vub, bits)
        enc0 = ga.encode(dec[0], vlb, vub, bits)
        enc1 = ga.encode(dec[1], vlb, vub, bits)
        np.testing.assert_array_almost_equal(gen[0], enc0)  # decode followed by encode gives original array
        np.testing.assert_array_almost_equal(gen[1], enc1)

    def test_encode_and_decode_gray_code(self):
        ga = GeneticAlgorithm(None)
        gen = np.array([[0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0,
                         1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0,
                         0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1],
                        [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0,
                         1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1,
                         1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1]])
        vlb = np.array([-170.0, -170.0, -170.0, -170.0, -170.0, -170.0])
        vub = np.array([255.0, 255.0, 255.0, 170.0, 170.0, 170.0])
        bits = np.array([9, 9, 9, 9, 9, 9])
        x = np.array([[-69.36399217, 22.12328767, -7.81800391, -66.86888454, 116.77103718, 76.18395303],
                      [248.34637965, 191.79060665, -31.93737769, 97.47553816, 118.76712329, 92.15264188]])

        ga.npop = 2
        ga.lchrom = int(np.sum(bits))
        ga.gray_code = True
        np.testing.assert_array_almost_equal(x, ga.decode(gen, vlb, vub, bits))
        np.testing.assert_array_almost_equal(gen[0], ga.encode(x[0], vlb, vub, bits))
        np.testing.assert_array_almost_equal(gen[1], ga.encode(x[1], vlb, vub, bits))

        dec = ga.decode(gen, vlb, vub, bits)
        enc0 = ga.encode(dec[0], vlb, vub, bits)
        enc1 = ga.encode(dec[1], vlb, vub, bits)
        np.testing.assert_array_almost_equal(gen[0], enc0)  # decode followed by encode gives original array
        np.testing.assert_array_almost_equal(gen[1], enc1)

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

        prob.driver = om.SimpleGADriver()

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

        prob.driver = om.SimpleGADriver()

        prob.setup()

        with self.assertRaises(Exception) as raises_msg:
            prob.run_driver()

        exception = raises_msg.exception

        msg = "Driver requires objective to be declared"

        self.assertEqual(exception.args[0], msg)

    def test_scipy_invalid_desvar_values(self):

        expected_err = ("The following design variable initial conditions are out of their specified "
                        "bounds:"
                        "\n  xI"
                        "\n    val: [-6.]"
                        "\n    lower: -5.0"
                        "\n    upper: 10.0"
                        "\n  xC"
                        "\n    val: [15.1]"
                        "\n    lower: 0.0"
                        "\n    upper: 15.0"
                        "\nSet the initial value of the design variable to a valid value or set "
                        "the driver option['invalid_desvar_behavior'] to 'ignore'."
                        "\nThis warning will become an error by default in OpenMDAO version 3.25.")

        for option in ['warn', 'raise', 'ignore']:
            with self.subTest(f'invalid_desvar_behavior = {option}'):

                # build the model
                prob = om.Problem()
                model = prob.model

                model.add_subsystem('comp', Branin(),
                                    promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

                model.add_design_var('xI', lower=-5.0, upper=10.0)
                model.add_design_var('xC', lower=0.0, upper=15.0)
                model.add_objective('comp.f')

                prob.driver = om.SimpleGADriver(invalid_desvar_behavior=option)
                prob.driver.options['bits'] = {'xC': 8}
                prob.driver.options['pop_size'] = 10

                prob.setup()

                prob.set_val('xC', 15.1)
                prob.set_val('xI', -6)

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

        prob.driver = om.SimpleGADriver()

        prob.setup()

        with self.assertRaises(ValueError) as err:
            prob.final_setup()

        # A value of None for lower and upper is changed to +/- INF_BOUND in add_design_var()
        if lower == None:
            lower = -INF_BOUND
        if upper == None:
            upper = INF_BOUND

        msg = ("Invalid bounds for design variable 'x.x'. When using "
               "SimpleGADriver, values for both 'lower' and 'upper' "
               f"must be specified between +/-INF_BOUND ({INF_BOUND}), "
               f"but they are: lower={lower}, upper={upper}.")

        self.maxDiff = None
        self.assertEqual(err.exception.args[0], msg)

    def test_vectorized_constraints(self):
        prob = om.Problem()
        model = prob.model

        dim = 2
        model.add_subsystem('x', om.IndepVarComp('x', np.ones(dim)), promotes=['*'])
        model.add_subsystem('f_x', om.ExecComp('f_x = sum(x * x)', x=np.ones(dim), f_x=1.0), promotes=['*'])
        model.add_subsystem('g_x', om.ExecComp('g_x = 1 - x', x=np.ones(dim), g_x=np.zeros(dim)), promotes=['*'])

        prob.driver = om.SimpleGADriver()

        prob.model.add_design_var('x', lower=-10, upper=10)
        prob.model.add_objective('f_x')
        prob.model.add_constraint('g_x', upper=np.zeros(dim))

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('x', prob['x'])

        # Check that the constraint is satisfied (x >= 1)
        for i in range(dim):
            self.assertLessEqual(1.0, prob["x"][i])


@unittest.skipUnless(pyDOE2, "requires 'pyDOE2', install openmdao[doe]")
class TestDriverOptionsSimpleGA(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        os.environ['SimpleGADriver_seed'] = '11'

    def test_driver_options(self):
        """Tests if Pm and Pc options can be set."""
        prob = om.Problem()
        model = prob.model
        indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('x', 1.)
        model.add_subsystem('model', om.ExecComp('y=x**2'), promotes=['*'])
        driver = prob.driver = om.SimpleGADriver()
        driver.options['Pm'] = 0.123
        driver.options['Pc'] = 0.0123
        driver.options['max_gen'] = 5
        driver.options['bits'] = {'x': 8}
        prob.model.add_design_var('x', lower=-10., upper=10.)
        prob.model.add_objective('y')
        prob.setup()
        prob.run_driver()
        self.assertEqual(driver.options['Pm'], 0.123)
        self.assertEqual(driver.options['Pc'], 0.0123)


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


@unittest.skipUnless(pyDOE2, "requires 'pyDOE2', install openmdao[doe]")
class TestMultiObjectiveSimpleGA(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        os.environ['SimpleGADriver_seed'] = '11'

    def test_multi_obj(self):

        prob = om.Problem()
        prob.model.add_subsystem('box', Box(), promotes=['*'])

        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('length', 1.5)
        indeps.add_output('width', 1.5)
        indeps.add_output('height', 1.5)

        # setup the optimization
        prob.driver = om.SimpleGADriver()
        prob.driver.options['max_gen'] = 100
        prob.driver.options['bits'] = {'length': 8, 'width': 8, 'height': 8}
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
        prob2.driver = om.SimpleGADriver()
        prob2.driver.options['max_gen'] = 100
        prob2.driver.options['bits'] = {'length': 8, 'width': 8, 'height': 8}
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

    def test_pareto(self):
        np.random.seed(11)

        prob = om.Problem()

        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('length', 1.5)
        indeps.add_output('width', 1.5)
        indeps.add_output('height', 1.5)

        prob.model.add_subsystem('box', Box(), promotes=['*'])

        # setup the optimization
        prob.driver = om.SimpleGADriver()
        prob.driver.options['max_gen'] = 20
        prob.driver.options['bits'] = {'length': 8, 'width': 8, 'height': 8}
        prob.driver.options['penalty_parameter'] = 10.
        prob.driver.options['compute_pareto'] = True

        prob.driver._randomstate = 11

        prob.model.add_design_var('length', lower=0.1, upper=2.)
        prob.model.add_design_var('width', lower=0.1, upper=2.)
        prob.model.add_design_var('height', lower=0.1, upper=2.)
        prob.model.add_objective('front_area', scaler=-1)  # maximize
        prob.model.add_objective('top_area', scaler=-1)  # maximize
        prob.model.add_constraint('volume', upper=1.)

        prob.setup()
        prob.run_driver()

        nd_obj = prob.driver.obj_nd
        sorted_obj = nd_obj[nd_obj[:, 0].argsort()]

        # We have sorted the pareto points by col 1, so col 1 should be ascending.
        # Col 2 should be descending because the points are non-dominated.
        self.assertTrue(np.all(sorted_obj[:-1, 0] <= sorted_obj[1:, 0]))
        self.assertTrue(np.all(sorted_obj[:-1, 1] >= sorted_obj[1:, 1]))


@unittest.skipUnless(pyDOE2, "requires 'pyDOE2', install openmdao[doe]")
class TestConstrainedSimpleGA(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)
        os.environ['SimpleGADriver_seed'] = '11'

    def test_constrained_with_penalty(self):

        class Cylinder(om.ExplicitComponent):
            """Main class"""

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
        driver = prob.driver = om.SimpleGADriver()
        prob.driver.options['penalty_parameter'] = 3.
        prob.driver.options['penalty_exponent'] = 1.
        prob.driver.options['max_gen'] = 50
        prob.driver.options['bits'] = {'radius': 8, 'height': 8}

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
        driver = prob.driver = om.SimpleGADriver()

        with self.assertRaises(KeyError) as raises_msg:
            prob.driver.supports['equality_constraints'] = False

        exception = raises_msg.exception

        msg = "SimpleGADriver: Tried to set read-only option 'equality_constraints'."

        self.assertEqual(exception.args[0], msg)

    def test_constrained_without_penalty(self):

        class Cylinder(om.ExplicitComponent):
            """Main class"""

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
        driver = prob.driver = om.SimpleGADriver()
        prob.driver.options['penalty_parameter'] = 0.  # no penalty, same as unconstrained
        prob.driver.options['penalty_exponent'] = 1.
        prob.driver.options['max_gen'] = 50
        prob.driver.options['bits'] = {'radius': 8, 'height': 8}

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
            """Main class"""

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
        driver = prob.driver = om.SimpleGADriver()
        prob.driver.options['penalty_parameter'] = 10.  # will have no effect
        prob.driver.options['penalty_exponent'] = 1.
        prob.driver.options['max_gen'] = 50
        prob.driver.options['bits'] = {'radius': 8, 'height': 8}

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

    def test_two_constraints(self):

        import openmdao.api as om

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
        p.model.add_constraint('exec.z', indices=[50], equals=30, alias="ALIAS_TEST")

        p.driver = om.SimpleGADriver()

        p.setup()

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_driver()

        assert_near_equal(p.get_val('exec.z')[-1], 130)
        assert_near_equal(p.get_val('exec.z')[50], 30)

    def test_con_and_obj_same_var_name(self):

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
        p.model.add_constraint('exec.z', indices=[0], equals=-300, alias="ALIAS_TEST")

        p.driver = om.SimpleGADriver()

        p.setup()

        p.set_val('exec.x', np.linspace(-10, 10, 101))

        p.run_driver()

        assert_near_equal(p.get_val('exec.z')[0], -300)
        assert_near_equal(p.get_val('exec.z')[50], -400)

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
        prob.driver = om.SimpleGADriver()
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
               "When using SimpleGADriver, the value for 'equals', "
               "'lower' or 'upper' must be specified between "
               f"+/-INF_BOUND ({INF_BOUND}), but they are: "
               f"equals={equals}, lower={lower}, upper={upper}.")

        self.maxDiff = None
        self.assertEqual(err.exception.args[0], msg)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
@unittest.skipUnless(pyDOE2, "requires 'pyDOE2', install openmdao[doe]")
class MPITestSimpleGA(unittest.TestCase):

    N_PROCS = 2

    def setUp(self):
        np.random.seed(1)
        os.environ['SimpleGADriver_seed'] = '11'

    def test_mixed_integer_branin(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('xC', 7.5))
        model.add_subsystem('p2', om.IndepVarComp('xI', 0.0))
        model.add_subsystem('comp', Branin())

        model.connect('p2.xI', 'comp.x0')
        model.connect('p1.xC', 'comp.x1')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.SimpleGADriver()
        prob.driver.options['bits'] = {'p1.xC': 8}
        prob.driver.options['max_gen'] = 50
        prob.driver.options['pop_size'] = 25
        prob.driver.options['run_parallel'] = True

        prob.driver._randomstate = 1

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('comp.f', prob['comp.f'])
            print('p2.xI', prob['p2.xI'])

        # Optimal solution
        assert_near_equal(prob['comp.f'], 0.49399549, 1e-4)
        self.assertTrue(int(prob['p2.xI']) in [3, -3])

    def test_two_branin_parallel_model(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('xC', 7.5))
        model.add_subsystem('p2', om.IndepVarComp('xI', 0.0))
        par = model.add_subsystem('par', om.ParallelGroup())

        par.add_subsystem('comp1', Branin())
        par.add_subsystem('comp2', Branin())

        model.connect('p2.xI', 'par.comp1.x0')
        model.connect('p1.xC', 'par.comp1.x1')
        model.connect('p2.xI', 'par.comp2.x0')
        model.connect('p1.xC', 'par.comp2.x1')

        model.add_subsystem('comp', om.ExecComp('f = f1 + f2'))
        model.connect('par.comp1.f', 'comp.f1')
        model.connect('par.comp2.f', 'comp.f2')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.SimpleGADriver()
        prob.driver.options['bits'] = {'p1.xC': 8}
        prob.driver.options['max_gen'] = 40
        prob.driver.options['pop_size'] = 25
        prob.driver.options['run_parallel'] = False
        prob.driver.options['procs_per_model'] = 2

        prob.driver._randomstate = 1

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('comp.f', prob['comp.f'])
            print('p2.xI', prob['p2.xI'])

        # Optimal solution
        assert_near_equal(prob['comp.f'], 0.98799098, 1e-4)
        self.assertTrue(int(prob['p2.xI']) in [3, -3])

    def test_mixed_integer_3bar_default_bits(self):
        # Tests bug where letting openmdao calculate the bits didn't preserve
        # integer status unless range was a power of 2.

        class ObjPenalty(om.ExplicitComponent):
            """
            Weight objective with penalty on stress constraint.
            """

            def setup(self):
                self.add_input('obj', 0.0)
                self.add_input('stress', val=np.zeros((3, )))

                self.add_output('weighted', 0.0)

            def compute(self, inputs, outputs):
                obj = inputs['obj']
                stress = inputs['stress']

                pen = 0.0
                for j in range(len(stress)):
                    if stress[j] > 1.0:
                        pen += 10.0*(stress[j] - 1.0)**2

                outputs['weighted'] = obj + pen

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('xc_a1', om.IndepVarComp('area1', 5.0, units='cm**2'), promotes=['*'])
        model.add_subsystem('xc_a2', om.IndepVarComp('area2', 5.0, units='cm**2'), promotes=['*'])
        model.add_subsystem('xc_a3', om.IndepVarComp('area3', 5.0, units='cm**2'), promotes=['*'])
        model.add_subsystem('xi_m1', om.IndepVarComp('mat1', 1), promotes=['*'])
        model.add_subsystem('xi_m2', om.IndepVarComp('mat2', 1), promotes=['*'])
        model.add_subsystem('xi_m3', om.IndepVarComp('mat3', 1), promotes=['*'])
        model.add_subsystem('comp', ThreeBarTruss(), promotes=['*'])
        model.add_subsystem('obj_with_penalty', ObjPenalty(), promotes=['*'])

        model.add_design_var('area1', lower=1.2, upper=1.3)
        model.add_design_var('area2', lower=2.0, upper=2.1)
        model.add_design_var('mat1', lower=2, upper=4)
        model.add_design_var('mat2', lower=2, upper=4)
        model.add_design_var('mat3', lower=1, upper=4)
        model.add_objective('weighted')

        prob.driver = om.SimpleGADriver()
        prob.driver.options['bits'] = {'area1': 6,
                                       'area2': 6}
        prob.driver.options['max_gen'] = 75

        prob.driver._randomstate = 1

        prob.setup()
        prob['area3'] = 0.0005
        prob.run_driver()

        if extra_prints:
            print('mass', prob['mass'])
            print('mat1', prob['mat1'])
            print('mat2', prob['mat2'])

        # Note, GA doesn't do so well with the continuous vars, naturally, so we reduce the space
        # as much as we can. Objective is still rather random, but it is close. GA does a great job
        # of picking the correct values for the integer desvars though.
        self.assertLess(prob['mass'], 6.0)
        assert_near_equal(prob['mat1'], 3, 1e-5)
        assert_near_equal(prob['mat2'], 3, 1e-5)
        # Material 3 can be anything

    def test_mpi_bug_solver(self):
        # This test verifies that mpi doesn't hang due to collective calls in the solver.

        prob = om.Problem()
        prob.model = SellarMDA()

        prob.model.add_design_var('x', lower=0, upper=10)
        prob.model.add_design_var('z', lower=0, upper=10)
        prob.model.add_objective('obj')

        prob.driver = om.SimpleGADriver(run_parallel=True)

        # Set these low because we don't need to run long.
        prob.driver.options['max_gen'] = 2
        prob.driver.options['pop_size'] = 5

        prob.setup()
        prob.set_solver_print(level=0)

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
class MPITestSimpleGA4Procs(unittest.TestCase):

    N_PROCS = 4

    def setUp(self):
        np.random.seed(1)
        os.environ['SimpleGADriver_seed'] = '11'

    def test_two_branin_parallel_model(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('xC', 5))
        model.add_subsystem('p2', om.IndepVarComp('xI', 0.0))
        par = model.add_subsystem('par', om.ParallelGroup())

        par.add_subsystem('comp1', Branin())
        par.add_subsystem('comp2', Branin())

        model.connect('p2.xI', 'par.comp1.x0')
        model.connect('p1.xC', 'par.comp1.x1')
        model.connect('p2.xI', 'par.comp2.x0')
        model.connect('p1.xC', 'par.comp2.x1')

        model.add_subsystem('comp', om.ExecComp('f = f1 + f2'))
        model.connect('par.comp1.f', 'comp.f1')
        model.connect('par.comp2.f', 'comp.f2')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.SimpleGADriver()
        prob.driver.options['bits'] = {'p1.xC': 8}
        prob.driver.options['max_gen'] = 10
        prob.driver.options['pop_size'] = 25
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = 2

        prob.driver._randomstate = 1

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('comp.f', prob['comp.f'])
            print('p2.xI', prob['p2.xI'])

        # Optimal solution
        assert_near_equal(prob.get_val('comp.f'), 0.98799098, 1e-6)
        self.assertTrue(int(prob['p2.xI']) in [3, -3])
        assert_near_equal(prob.get_val('p1.xC'), 11.94117647, 1e-6)

    def test_indivisible_error(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('par', om.ParallelGroup())

        prob.driver = om.SimpleGADriver()
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

        driver = prob.driver = om.SimpleGADriver()
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

        driver = prob.driver = om.SimpleGADriver()
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

        prob.driver = om.SimpleGADriver()
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = 2
        prob.driver.options['bits'] = {'x': 16, 'y': 16}  # use enough bits to get accurate answer
        prob.driver.options['max_gen'] = 30
        prob.driver.options['pop_size'] = 150

        prob.setup()
        prob.run_driver()

        # optimal solution for minimize (x-a)^2 +x*y +(y+4)^2 - 3 for a=[-3, -2.4, -1.8] is:
        # x =    [ 6.66667,  5.86667,  5.06667]
        # y =    [-7.33333, -6.93333, -6.53333]
        # f_xy = [-27.3333, -23.0533, -19.0133]  mean f_xy = -23.1333
        # loose tests so that few generations are required
        # assert_near_equal(prob.get_val('x', get_remote=True),    [ 6.66667,  5.86667,  5.06667], 0.2)
        # assert_near_equal(prob.get_val('y', get_remote=True),    [-7.33333, -6.93333, -6.53333], 0.25)
        # assert_near_equal(prob.get_val('f_xy', get_remote=True), [-27.3333, -23.0533, -19.0133], 0.2)
        assert_near_equal(np.sum(prob.get_val('f_xy', get_remote=True))/3, -23.1333, 0.15)


@unittest.skipUnless(pyDOE2, "requires 'pyDOE2', install openmdao[doe]")
class TestFeatureSimpleGA(unittest.TestCase):

    def setUp(self):
        import numpy as np
        np.random.seed(1)

        import os
        os.environ['SimpleGADriver_seed'] = '11'

    def test_basic_with_assert(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Branin(),
                            promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_design_var('xI', lower=-5.0, upper=10.0)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.SimpleGADriver()
        prob.driver.options['bits'] = {'xC': 8}

        prob.driver._randomstate = 1

        prob.setup()

        prob.set_val('xC', 7.5)
        prob.set_val('xI', 0.0)

        prob.run_driver()

        # Optimal solution
        assert_near_equal(prob['comp.f'], 0.49399549, 1e-4)

    def test_option_max_gen(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Branin(),
                            promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_design_var('xI', lower=-5.0, upper=10.0)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.SimpleGADriver()
        prob.driver.options['bits'] = {'xC': 8}
        prob.driver.options['max_gen'] = 5

        prob.setup()

        prob.set_val('xC', 7.5)
        prob.set_val('xI', 0.0)

        prob.run_driver()

    def test_option_pop_size(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Branin(),
                            promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_design_var('xI', lower=-5.0, upper=10.0)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.SimpleGADriver()
        prob.driver.options['bits'] = {'xC': 8}
        prob.driver.options['pop_size'] = 10

        prob.setup()

        prob.set_val('xC', 7.5)
        prob.set_val('xI', 0.0)

        prob.run_driver()

    def test_constrained_with_penalty(self):

        class Cylinder(om.ExplicitComponent):
            """Main class"""

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

        # setup the optimization
        prob.driver = om.SimpleGADriver()
        prob.driver.options['penalty_parameter'] = 3.
        prob.driver.options['penalty_exponent'] = 1.
        prob.driver.options['max_gen'] = 50
        prob.driver.options['bits'] = {'radius': 8, 'height': 8}

        prob.model.add_design_var('radius', lower=0.5, upper=5.)
        prob.model.add_design_var('height', lower=0.5, upper=5.)
        prob.model.add_objective('Area')
        prob.model.add_constraint('Volume', lower=10.)

        prob.setup()

        prob.set_val('radius', 2.)
        prob.set_val('height', 3.)

        prob.run_driver()

        # These go to 0.5 for unconstrained problem. With constraint and penalty, they
        # will be above 1.0 (actual values will vary.)
        self.assertGreater(prob.get_val('radius'), 1.)
        self.assertGreater(prob.get_val('height'), 1.)

    def test_pareto(self):

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

        # setup the optimization
        prob.driver = om.SimpleGADriver()
        prob.driver.options['max_gen'] = 20
        prob.driver.options['bits'] = {'length': 8, 'width': 8, 'height': 8}
        prob.driver.options['penalty_parameter'] = 10.
        prob.driver.options['compute_pareto'] = True

        prob.model.add_design_var('length', lower=0.1, upper=2.)
        prob.model.add_design_var('width', lower=0.1, upper=2.)
        prob.model.add_design_var('height', lower=0.1, upper=2.)
        prob.model.add_objective('front_area', scaler=-1)  # maximize
        prob.model.add_objective('top_area', scaler=-1)  # maximize
        prob.model.add_constraint('volume', upper=1.)

        prob.setup()

        prob.set_val('length', 1.5)
        prob.set_val('width', 1.5)
        prob.set_val('height', 1.5)

        prob.run_driver()

        desvar_nd = prob.driver.desvar_nd
        nd_obj = prob.driver.obj_nd

        assert_near_equal(desvar_nd,
                          np.array([[1.83607843, 0.54705882, 0.95686275],
                                    [1.50823529, 0.28627451, 1.95529412],
                                    [1.45607843, 1.90313725, 0.34588235],
                                    [1.76156863, 0.54705882, 1.01647059],
                                    [1.85098039, 1.03137255, 0.49490196],
                                    [1.87333333, 0.57686275, 0.90470588],
                                    [1.38156863, 1.87333333, 0.38313725],
                                    [1.68705882, 0.39803922, 1.47098039],
                                    [1.86588235, 0.50980392, 1.01647059],
                                    [2.        , 0.42784314, 1.13568627],
                                    [1.99254902, 0.69607843, 0.70352941],
                                    [1.74666667, 0.39803922, 1.40392157],
                                    [1.99254902, 0.30117647, 1.38901961],
                                    [1.97764706, 0.20431373, 1.95529412],
                                    [1.99254902, 0.57686275, 0.84509804],
                                    [1.9254902 , 0.30117647, 1.50823529],
                                    [1.75411765, 0.57686275, 0.97176471],
                                    [1.94039216, 0.92705882, 0.5545098 ],
                                    [1.74666667, 0.81529412, 0.70352941],
                                    [1.75411765, 0.42039216, 1.35176471]]),
                          1e-6)

        sorted_obj = nd_obj[nd_obj[:, 0].argsort()]

        assert_near_equal(sorted_obj,
                          np.array([[-3.86688166, -0.40406044],
                                    [-2.9490436 , -0.43176932],
                                    [-2.90409227, -0.57991234],
                                    [-2.76768966, -0.60010888],
                                    [-2.48163045, -0.67151557],
                                    [-2.45218301, -0.69524183],
                                    [-2.37115433, -0.7374173 ],
                                    [-2.27137255, -0.85568627],
                                    [-1.89661453, -0.95123414],
                                    [-1.7905827 , -0.96368166],
                                    [-1.75687505, -1.00444291],
                                    [-1.70458962, -1.01188512],
                                    [-1.69481569, -1.08065621],
                                    [-1.68389927, -1.1494273 ],
                                    [-1.40181684, -1.3869704 ],
                                    [-1.21024148, -1.40545716],
                                    [-1.07596647, -1.79885767],
                                    [-0.91605383, -1.90905037],
                                    [-0.52933041, -2.58813856],
                                    [-0.50363183, -2.77111711]]),
                          1e-6)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
@unittest.skipUnless(pyDOE2, "requires 'pyDOE2', install openmdao[doe]")
class MPIFeatureTests(unittest.TestCase):
    N_PROCS = 2

    def setUp(self):
        np.random.seed(1)

        import os
        os.environ['SimpleGADriver_seed'] = '11'

    def test_option_parallel(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Branin(),
                            promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_design_var('xI', lower=-5.0, upper=10.0)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.SimpleGADriver()
        prob.driver.options['bits'] = {'xC': 8}
        prob.driver.options['max_gen'] = 10
        prob.driver.options['run_parallel'] = True

        prob.setup()

        prob.set_val('xC', 7.5)
        prob.set_val('xI', 0.0)

        prob.run_driver()

        # Optimal solution
        assert_near_equal(prob.get_val('comp.f'), 1.25172426, 1e-6)
        assert_near_equal(prob.get_val('xI'), 9.0, 1e-6)
        assert_near_equal(prob.get_val('xC'), 2.11764706, 1e-6)


if __name__ == "__main__":
    unittest.main()
