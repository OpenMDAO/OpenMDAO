""" Unit tests for the SimpleGADriver Driver."""

import unittest
import os

import numpy as np

import openmdao.api as om
from openmdao.drivers.genetic_algorithm_driver import GeneticAlgorithm
from openmdao.test_suite.components.branin import Branin, BraninDiscrete
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.paraboloid_distributed import DistParab
from openmdao.test_suite.components.sellar_feature import SellarMDA
from openmdao.test_suite.components.three_bar_truss import ThreeBarTruss

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.mpi import MPI

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

extra_prints = False  # enable printing results

class TestSimpleGA(unittest.TestCase):

    def setUp(self):
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
        prob = om.Problem()
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
        indeps.add_output('y', [4.0, -4])

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

    def test_SimpleGADriver_missing_objective(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('x', om.IndepVarComp('x', 2.0), promotes=['*'])
        model.add_subsystem('f_x', Paraboloid(), promotes=['*'])

        prob.driver = om.SimpleGADriver()

        prob.model.add_design_var('x', lower=0)
        prob.model.add_constraint('x', lower=0)

        prob.setup()

        with self.assertRaises(Exception) as raises_msg:
            prob.run_driver()

        exception = raises_msg.exception

        msg = "Driver requires objective to be declared"

        self.assertEqual(exception.args[0], msg)

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


class TestDriverOptionsSimpleGA(unittest.TestCase):

    def setUp(self):
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


class TestMultiObjectiveSimpleGA(unittest.TestCase):

    def setUp(self):
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


class TestConstrainedSimpleGA(unittest.TestCase):

    def setUp(self):
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


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class MPITestSimpleGA(unittest.TestCase):

    N_PROCS = 2

    def setUp(self):
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
    def initialize(self):
        self.options['distributed'] = True

    def setup(self):
        comm = self.comm
        rank = comm.rank

        if rank == 1:
            start = 1
            end = 2
        else:
            start = 0
            end = 1

        self.add_input('y2', np.ones((1, ), float),
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('x', np.ones((1, ), float))

        self.add_output('y1', np.ones((1, ), float))

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
    def initialize(self):
        self.options['distributed'] = True

    def setup(self):
        comm = self.comm
        rank = comm.rank

        if rank == 1:
            start = 1
            end = 2
        else:
            start = 0
            end = 1

        self.add_input('y1', np.ones((1, ), float),
                       src_indices=np.arange(start, end, dtype=int))

        self.add_output('y2', np.ones((1, ), float))

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
class MPITestSimpleGA4Procs(unittest.TestCase):

    N_PROCS = 4

    def setUp(self):
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
        assert_near_equal(prob['comp.f'], 0.98799098, 1e-4)
        self.assertTrue(int(prob['p2.xI']) in [3, -3])

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

        model.add_subsystem('obj_comp', Summer(), promotes=['*'])
        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        model.linear_solver = om.LinearBlockGS()

        model.add_design_var('x', lower=-0.5, upper=0.5)
        model.add_objective('obj')

        driver = prob.driver = om.SimpleGADriver()
        prob.driver.options['pop_size'] = 4
        prob.driver.options['max_gen'] = 3
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = 2

        prob.setup()
        prob.set_solver_print(level=0)

        prob.run_driver()

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
                            promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')

        prob.driver = om.SimpleGADriver()
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = 2
        prob.driver.options['bits'] = {'x': 12, 'y': 12}  # use enough bits and generations to get close
        prob.driver.options['max_gen'] = 40

        prob.setup()
        prob.run_driver()

        # optimal solution for minimize (x-a)^2 +x*y +(y+4)^2 - 3 for a=[-3, -2.4, -1.8] is:
        # x =    [ 6.66667,  5.86667,  5.06667]
        # y =    [-7.33333, -6.93333, -6.53333]
        # f_xy = [-27.3333, -23.0533, -19.0133]  mean f_xy = -23.1333
        # very loose tests to allow Travis testing to suceed
        assert_near_equal(prob.get_val('x', get_remote=True),    [ 6.66667,  5.86667,  5.06667], 2.0)
        assert_near_equal(prob.get_val('y', get_remote=True),    [-7.33333, -6.93333, -6.53333], 2.0)
        assert_near_equal(prob.get_val('f_xy', get_remote=True), [-27.3333, -23.0533, -19.0133], 3.0)
        assert_near_equal(np.sum(prob.get_val('f_xy', get_remote=True))/3, -23.1333, 3.0)


class TestFeatureSimpleGA(unittest.TestCase):

    def setUp(self):
        import numpy as np
        import os
        os.environ['SimpleGADriver_seed'] = '11'

    def test_basic(self):
        import openmdao.api as om
        from openmdao.test_suite.components.branin import Branin

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Branin(),
                            promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_design_var('xI', lower=-5.0, upper=10.0)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.SimpleGADriver()
        prob.driver.options['bits'] = {'xC': 8}

        prob.setup()

        prob.set_val('xC', 7.5)
        prob.set_val('xI', 0.0)

        prob.run_driver()

    def test_basic_with_assert(self):
        import openmdao.api as om
        from openmdao.test_suite.components.branin import Branin

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
        import openmdao.api as om
        from openmdao.test_suite.components.branin import Branin

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
        import openmdao.api as om
        from openmdao.test_suite.components.branin import Branin

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
        import openmdao.api as om

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
        import openmdao.api as om

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
        sorted_obj = nd_obj[nd_obj[:, 0].argsort()]

        assert_near_equal(desvar_nd,
                          np.array([[1.70941176, 1.79882353, 0.22666667],
                                    [1.83607843, 0.54705882, 0.99411765],
                                    [1.97764706, 0.20431373, 1.71686275],
                                    [1.85843137, 1.03137255, 0.49490196],
                                    [1.68705882, 1.98509804, 0.1372549 ],
                                    [1.50823529, 1.60509804, 0.37568627],
                                    [1.82862745, 1.63490196, 0.24901961],
                                    [1.95529412, 0.33843137, 1.46352941],
                                    [1.98509804, 0.66627451, 0.71843137],
                                    [1.82862745, 0.4427451 , 1.22509804],
                                    [1.94784314, 0.33843137, 1.48588235],
                                    [1.85843137, 0.45019608, 1.19529412],
                                    [1.86588235, 1.15058824, 0.4427451 ],
                                    [1.95529412, 0.56196078, 0.81529412],
                                    [1.86588235, 0.56196078, 0.91960784],
                                    [1.79882353, 0.4054902 , 1.29960784],
                                    [1.97764706, 0.45019608, 1.07607843],
                                    [1.94784314, 0.91215686, 0.56196078],
                                    [1.87333333, 0.73333333, 0.72588235],
                                    [1.87333333, 0.8972549 , 0.59176471],
                                    [1.87333333, 0.35333333, 1.36666667],
                                    [1.9627451 , 0.66627451, 0.75568627],
                                    [1.74666667, 0.39803922, 1.44117647],
                                    [1.73921569, 0.33843137, 1.6945098 ]]),
                          1e-6)

        assert_near_equal(sorted_obj,
                          np.array([[-3.39534856, -0.40406044],
                                    [-2.94711803, -0.58860515],
                                    [-2.89426574, -0.65921123],
                                    [-2.86163045, -0.66173287],
                                    [-2.56022222, -0.66191111],
                                    [-2.49759323, -0.67558016],
                                    [-2.33776517, -0.72940531],
                                    [-2.2402479 , -0.80961584],
                                    [-2.22084206, -0.83612849],
                                    [-2.12810334, -0.89032895],
                                    [-1.82527797, -1.00444291],
                                    [-1.71588005, -1.04855271],
                                    [-1.59413979, -1.09879862],
                                    [-1.48321953, -1.30772703],
                                    [-1.42615671, -1.32262022],
                                    [-1.35981961, -1.37377778],
                                    [-1.10857255, -1.68085752],
                                    [-1.09461146, -1.77673849],
                                    [-0.91974133, -1.9167351 ],
                                    [-0.82611027, -2.14686228],
                                    [-0.5666233 , -2.42086551],
                                    [-0.45536409, -2.98962661],
                                    [-0.38746667, -3.0749301 ],
                                    [-0.23155709, -3.34897716]]),
                          1e-6)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class MPIFeatureTests(unittest.TestCase):
    N_PROCS = 2

    def setUp(self):
        import numpy as np
        import os
        os.environ['SimpleGADriver_seed'] = '11'

    def test_option_parallel(self):
        import openmdao.api as om
        from openmdao.test_suite.components.branin import Branin

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


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class MPIFeatureTests4(unittest.TestCase):
    N_PROCS = 4

    def setUp(self):
        import numpy as np
        import os
        os.environ['SimpleGADriver_seed'] = '11'

    def test_option_procs_per_model(self):
        import openmdao.api as om
        from openmdao.test_suite.components.branin import Branin

        prob = om.Problem()
        model = prob.model

        par = model.add_subsystem('par', om.ParallelGroup(),
                                  promotes_inputs=['*'])

        par.add_subsystem('comp1', Branin(),
                          promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])
        par.add_subsystem('comp2', Branin(),
                          promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_subsystem('comp', om.ExecComp('f = f1 + f2'))
        model.connect('par.comp1.f', 'comp.f1')
        model.connect('par.comp2.f', 'comp.f2')

        model.add_design_var('xI', lower=-5.0, upper=10.0)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = om.SimpleGADriver()
        prob.driver.options['bits'] = {'xC': 8}
        prob.driver.options['max_gen'] = 10
        prob.driver.options['pop_size'] = 25
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = 2

        prob.driver._randomstate = 1

        prob.setup()

        prob.set_val('xC', 7.5)
        prob.set_val('xI', 0.0)

        prob.run_driver()

        # Optimal solution
        assert_near_equal(prob.get_val('comp.f'), 0.98799098, 1e-6)
        assert_near_equal(prob.get_val('xI'),-3.0, 1e-6)
        assert_near_equal(prob.get_val('xC'), 11.94117647, 1e-6)


if __name__ == "__main__":
    unittest.main()
