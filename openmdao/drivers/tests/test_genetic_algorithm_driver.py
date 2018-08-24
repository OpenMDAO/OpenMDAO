""" Unit tests for the SimpleGADriver Driver."""

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent, ExecComp, PETScVector, \
                         ParallelGroup
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver
from openmdao.test_suite.components.branin import Branin
from openmdao.test_suite.components.three_bar_truss import ThreeBarTruss
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.utils.mpi import MPI


class TestSimpleGA(unittest.TestCase):

    def test_simple_test_func(self):
        np.random.seed(11)

        class MyComp(ExplicitComponent):

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


        prob = Problem()
        prob.model = model = Group()

        model.add_subsystem('px', IndepVarComp('x', np.array([0.2, -0.2])))
        model.add_subsystem('comp', MyComp())
        model.add_subsystem('obj', ExecComp('f=(30 + a*b)*(1 + c*d)'))

        model.connect('px.x', 'comp.x')
        model.connect('comp.a', 'obj.a')
        model.connect('comp.b', 'obj.b')
        model.connect('comp.c', 'obj.c')
        model.connect('comp.d', 'obj.d')

        # Played with bounds so we don't get subtractive cancellation of tiny numbers.
        model.add_design_var('px.x', lower=np.array([0.2, -1.0]), upper=np.array([1.0, -0.2]))
        model.add_objective('obj.f')

        prob.driver = SimpleGADriver()
        prob.driver.options['bits'] = {'px.x': 16}
        prob.driver.options['max_gen'] = 75

        prob.driver._randomstate = 11

        prob.setup(check=False)
        prob.run_driver()

        # TODO: Satadru listed this solution, but I get a way better one.
        # Solution: xopt = [0.2857, -0.8571], fopt = 23.2933
        assert_rel_error(self, prob['obj.f'], 12.37341703, 1e-4)
        assert_rel_error(self, prob['px.x'][0], 0.2, 1e-4)
        assert_rel_error(self, prob['px.x'][1], -0.88654333, 1e-4)

    def test_mixed_integer_branin(self):
        np.random.seed(1)


        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('xC', 7.5))
        model.add_subsystem('p2', IndepVarComp('xI', 0.0))
        model.add_subsystem('comp', Branin())

        model.connect('p2.xI', 'comp.x0')
        model.connect('p1.xC', 'comp.x1')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = SimpleGADriver(max_gen=75, pop_size=25)
        prob.driver.options['bits'] = {'p1.xC': 8}

        prob.driver._randomstate = 1

        prob.setup(check=False)
        prob.run_driver()

        # Optimal solution
        assert_rel_error(self, prob['comp.f'], 0.49398, 1e-4)
        self.assertTrue(int(prob['p2.xI']) in [3, -3])

    def test_mixed_integer_3bar(self):
        np.random.seed(1)

        class ObjPenalty(ExplicitComponent):
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

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('xc_a1', IndepVarComp('area1', 5.0, units='cm**2'), promotes=['*'])
        model.add_subsystem('xc_a2', IndepVarComp('area2', 5.0, units='cm**2'), promotes=['*'])
        model.add_subsystem('xc_a3', IndepVarComp('area3', 5.0, units='cm**2'), promotes=['*'])
        model.add_subsystem('xi_m1', IndepVarComp('mat1', 1), promotes=['*'])
        model.add_subsystem('xi_m2', IndepVarComp('mat2', 1), promotes=['*'])
        model.add_subsystem('xi_m3', IndepVarComp('mat3', 1), promotes=['*'])
        model.add_subsystem('comp', ThreeBarTruss(), promotes=['*'])
        model.add_subsystem('obj_with_penalty', ObjPenalty(), promotes=['*'])

        model.add_design_var('area1', lower=1.2, upper=1.3)
        model.add_design_var('area2', lower=2.0, upper=2.1)
        model.add_design_var('mat1', lower=1, upper=4)
        model.add_design_var('mat2', lower=1, upper=4)
        model.add_design_var('mat3', lower=1, upper=4)
        model.add_objective('weighted')

        prob.driver = SimpleGADriver()
        prob.driver.options['bits'] = {'area1': 6,
                                       'area2': 6}
        prob.driver.options['max_gen'] = 75

        prob.driver._randomstate = 1

        prob.setup(check=False)
        prob['area3'] = 0.0005
        prob.run_driver()

        # Note, GA doesn't do so well with the continuous vars, naturally, so we reduce the space
        # as much as we can. Objective is still rather random, but it is close. GA does a great job
        # of picking the correct values for the integer desvars though.
        self.assertLess(prob['mass'], 6.0)
        assert_rel_error(self, prob['mat1'], 3, 1e-5)
        assert_rel_error(self, prob['mat2'], 3, 1e-5)
        #Material 3 can be anything


class TestDriverOptionsSimpleGA(unittest.TestCase):

    def test_driver_options(self):
        """Tests if Pm and Pc options can be set."""
        prob = Problem()
        model = prob.model
        indeps = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        indeps.add_output('x', 1.)
        model.add_subsystem('model', ExecComp('y=x**2'), promotes=['*'])
        driver = prob.driver = SimpleGADriver()
        driver.options['Pm'] = 0.1
        driver.options['Pc'] = 0.01
        driver.options['max_gen'] = 5
        driver.options['bits'] = {'x': 8}
        prob.model.add_design_var('x', lower=-10., upper=10.)
        prob.model.add_objective('y')
        prob.setup(check=False)
        prob.run_driver()
        self.assertEqual(driver.options['Pm'], 0.1)
        self.assertEqual(driver.options['Pc'], 0.01)


class TestMultiObjectiveSimpleGA(unittest.TestCase):

    def test_multi_obj(self):
        from math import sqrt, pi

        class Cone(ExplicitComponent):

            def setup(self):
                self.add_input('radius', val=1.)
                self.add_input('height', val=1.)

                self.add_output('base_area', val=1.0)
                self.add_output('lateral_surf_area', val=1.0)
                self.add_output('area', val=1.0)
                self.add_output('volume', val=1.)

            def compute(self, inputs, outputs):
                r = inputs['radius']
                h = inputs['height']
                s = sqrt(r**2 + h**2)

                outputs['base_area'] = base_area = pi * r**2
                outputs['lateral_surf_area'] = lateral_area = pi * r * s
                outputs['area'] = base_area + lateral_area
                outputs['volume'] = base_area * h / 3.

        prob = Problem()
        cone = prob.model.add_subsystem('cone', Cone(), promotes=['*'])

        indeps = prob.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        indeps.add_output('radius', 2.)  # height
        indeps.add_output('height', 3.)  # radius

        # setup the optimization
        driver = prob.driver = SimpleGADriver()
        prob.driver.options['max_gen'] = 100
        prob.driver.options['bits'] = {'radius': 8, 'height': 8}
        # prob.driver.options['multi_obj_exponent'] = 1.
        prob.driver.options['multi_obj_weights'] = {'cone.base_area': 0.5,
                                                    'cone.lateral_surf_area': 0.5}
        prob.driver.options['multi_obj_exponent'] = 1

        prob.model.add_design_var('radius', lower=5., upper=15.)
        prob.model.add_design_var('height', lower=1., upper=15.)
        prob.model.add_objective('base_area')
        prob.model.add_objective('lateral_surf_area')
        prob.model.add_constraint('volume', equals=200)

        prob.setup()
        prob.run_driver()
        print('Areas: ', prob['base_area'], prob['lateral_surf_area'])
        print('Volume: ', prob['volume'])  #


class TestConstrainedSimpleGA(unittest.TestCase):

    def test_constrained_with_penalty(self):

        class Cylinder(ExplicitComponent):
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

        prob = Problem()
        cylinder = prob.model.add_subsystem('cylinder', Cylinder(), promotes=['*'])

        indeps = prob.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        indeps.add_output('radius', 2.)  # height
        indeps.add_output('height', 3.)  # radius

        # setup the optimization
        driver = prob.driver = SimpleGADriver()
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
        print('radius', prob['radius'])  # exact solution is (5/pi)^(1/3) ~= 1.167
        print('height', prob['height'])  # exact solution is 2*radius
        print('Area', prob['Area'])
        print('Volume', prob['Volume'])  # should be around 10
        self.assertTrue(driver.supports["equality_constraints"], True)
        self.assertTrue(driver.supports["inequality_constraints"], True)
        # check that it is not going to the unconstrained optimum
        self.assertGreater(prob['radius'], 1.)
        self.assertGreater(prob['height'], 1.)

    def test_constrained_without_penalty(self):

        class Cylinder(ExplicitComponent):
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

        prob = Problem()
        cylinder = prob.model.add_subsystem('cylinder', Cylinder(), promotes=['*'])

        indeps = prob.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        indeps.add_output('radius', 2.)  # height
        indeps.add_output('height', 3.)  # radius

        # setup the optimization
        driver = prob.driver = SimpleGADriver()
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

        class Cylinder(ExplicitComponent):
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

        prob = Problem()
        cylinder = prob.model.add_subsystem('cylinder', Cylinder(), promotes=['*'])

        indeps = prob.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        indeps.add_output('radius', 2.)  # height
        indeps.add_output('height', 3.)  # radius

        # setup the optimization
        driver = prob.driver = SimpleGADriver()
        prob.driver.options['penalty_parameter'] = 10.  # will have no effect
        prob.driver.options['penalty_exponent'] = 1.
        prob.driver.options['max_gen'] = 50
        prob.driver.options['bits'] = {'radius': 8, 'height': 8}

        prob.model.add_design_var('radius', lower=0.5, upper=5.)
        prob.model.add_design_var('height', lower=0.5, upper=5.)
        prob.model.add_objective('Area')

        prob.setup()
        prob.run_driver()
        print('radius', prob['radius'])  # exact solution is (5/pi)^(1/3) ~= 1.167
        print('height', prob['height'])  # exact solution is 2*radius
        print('Area', prob['Area'])
        print('Volume', prob['Volume'])  # should be around 10
        self.assertTrue(driver.supports["equality_constraints"], True)
        self.assertTrue(driver.supports["inequality_constraints"], True)
        self.assertAlmostEqual(prob['radius'], 0.5, 1)  # it is going to the unconstrained optimum
        self.assertAlmostEqual(prob['height'], 0.5, 1)  # it is going to the unconstrained optimum


@unittest.skipUnless(PETScVector, "PETSc is required.")
class MPITestSimpleGA(unittest.TestCase):

    N_PROCS = 2

    def test_mixed_integer_branin(self):
        np.random.seed(1)

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('xC', 7.5))
        model.add_subsystem('p2', IndepVarComp('xI', 0.0))
        model.add_subsystem('comp', Branin())

        model.connect('p2.xI', 'comp.x0')
        model.connect('p1.xC', 'comp.x1')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = SimpleGADriver()
        prob.driver.options['bits'] = {'p1.xC': 8}
        prob.driver.options['max_gen'] = 50
        prob.driver.options['pop_size'] = 25
        prob.driver.options['run_parallel'] = True

        prob.driver._randomstate = 1

        prob.setup(check=False)
        prob.run_driver()

        # Optimal solution
        assert_rel_error(self, prob['comp.f'], 0.49398, 1e-4)
        self.assertTrue(int(prob['p2.xI']) in [3, -3])

    def test_two_branin_parallel_model(self):
        np.random.seed(1)

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('xC', 7.5))
        model.add_subsystem('p2', IndepVarComp('xI', 0.0))
        par = model.add_subsystem('par', ParallelGroup())

        par.add_subsystem('comp1', Branin())
        par.add_subsystem('comp2', Branin())

        model.connect('p2.xI', 'par.comp1.x0')
        model.connect('p1.xC', 'par.comp1.x1')
        model.connect('p2.xI', 'par.comp2.x0')
        model.connect('p1.xC', 'par.comp2.x1')

        model.add_subsystem('comp', ExecComp('f = f1 + f2'))
        model.connect('par.comp1.f', 'comp.f1')
        model.connect('par.comp2.f', 'comp.f2')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = SimpleGADriver()
        prob.driver.options['bits'] = {'p1.xC': 8}
        prob.driver.options['max_gen'] = 40
        prob.driver.options['pop_size'] = 25
        prob.driver.options['run_parallel'] = False
        prob.driver.options['procs_per_model'] = 2

        prob.driver._randomstate = 1

        prob.setup(check=False)
        prob.run_driver()

        # Optimal solution
        assert_rel_error(self, prob['comp.f'], 0.98799098, 1e-4)
        self.assertTrue(int(prob['p2.xI']) in [3, -3])


@unittest.skipUnless(PETScVector, "PETSc is required.")
class MPITestSimpleGA4Procs(unittest.TestCase):

    N_PROCS = 4

    def test_two_branin_parallel_model(self):
        np.random.seed(1)

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('xC', 7.5))
        model.add_subsystem('p2', IndepVarComp('xI', 0.0))
        par = model.add_subsystem('par', ParallelGroup())

        par.add_subsystem('comp1', Branin())
        par.add_subsystem('comp2', Branin())

        model.connect('p2.xI', 'par.comp1.x0')
        model.connect('p1.xC', 'par.comp1.x1')
        model.connect('p2.xI', 'par.comp2.x0')
        model.connect('p1.xC', 'par.comp2.x1')

        model.add_subsystem('comp', ExecComp('f = f1 + f2'))
        model.connect('par.comp1.f', 'comp.f1')
        model.connect('par.comp2.f', 'comp.f2')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = SimpleGADriver()
        prob.driver.options['bits'] = {'p1.xC': 8}
        prob.driver.options['max_gen'] = 10
        prob.driver.options['pop_size'] = 25
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = 2

        prob.driver._randomstate = 1

        prob.setup(check=False)
        prob.run_driver()

        # Optimal solution
        assert_rel_error(self, prob['comp.f'], 0.98799098, 1e-4)
        self.assertTrue(int(prob['p2.xI']) in [3, -3])

    def test_indivisible_error(self):
        prob = Problem()
        model = prob.model = Group()
        par = model.add_subsystem('par', ParallelGroup())

        prob.driver = SimpleGADriver()
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = 3

        with self.assertRaises(RuntimeError) as context:
            prob.setup()

        self.assertEqual(str(context.exception),
                         "The total number of processors is not evenly divisible by the "
                         "specified number of processors per model.\n Provide a number of "
                         "processors that is a multiple of 3, or specify a number "
                         "of processors per model that divides into 4.")


class TestFeatureSimpleGA(unittest.TestCase):

    def setUp(self):
        import numpy as np
        np.random.seed(1)

    def test_basic(self):
        from openmdao.api import Problem, Group, IndepVarComp, SimpleGADriver
        from openmdao.test_suite.components.branin import Branin

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('xC', 7.5))
        model.add_subsystem('p2', IndepVarComp('xI', 0.0))
        model.add_subsystem('comp', Branin())

        model.connect('p2.xI', 'comp.x0')
        model.connect('p1.xC', 'comp.x1')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = SimpleGADriver()
        prob.driver.options['bits'] = {'p1.xC': 8}

        prob.setup()
        prob.run_driver()

        # Optimal solution
        print('comp.f', prob['comp.f'])
        print('p2.xI', prob['p2.xI'])
        print('p1.xC', prob['p1.xC'])

    def test_basic_with_assert(self):
        from openmdao.api import Problem, Group, IndepVarComp, SimpleGADriver
        from openmdao.test_suite.components.branin import Branin

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('xC', 7.5))
        model.add_subsystem('p2', IndepVarComp('xI', 0.0))
        model.add_subsystem('comp', Branin())

        model.connect('p2.xI', 'comp.x0')
        model.connect('p1.xC', 'comp.x1')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = SimpleGADriver()
        prob.driver.options['bits'] = {'p1.xC': 8}

        prob.driver._randomstate = 1

        prob.setup()
        prob.run_driver()

        # Optimal solution
        assert_rel_error(self, prob['comp.f'], 0.49399549, 1e-4)

    def test_option_max_gen(self):
        from openmdao.api import Problem, Group, IndepVarComp, SimpleGADriver
        from openmdao.test_suite.components.branin import Branin

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('xC', 7.5))
        model.add_subsystem('p2', IndepVarComp('xI', 0.0))
        model.add_subsystem('comp', Branin())

        model.connect('p2.xI', 'comp.x0')
        model.connect('p1.xC', 'comp.x1')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = SimpleGADriver()
        prob.driver.options['bits'] = {'p1.xC': 8}
        prob.driver.options['max_gen'] = 5

        prob.setup()
        prob.run_driver()

        # Optimal solution
        print('comp.f', prob['comp.f'])
        print('p2.xI', prob['p2.xI'])
        print('p1.xC', prob['p1.xC'])

    def test_option_pop_size(self):
        from openmdao.api import Problem, Group, IndepVarComp, SimpleGADriver
        from openmdao.test_suite.components.branin import Branin

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('xC', 7.5))
        model.add_subsystem('p2', IndepVarComp('xI', 0.0))
        model.add_subsystem('comp', Branin())

        model.connect('p2.xI', 'comp.x0')
        model.connect('p1.xC', 'comp.x1')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = SimpleGADriver()
        prob.driver.options['bits'] = {'p1.xC': 8}
        prob.driver.options['pop_size'] = 10

        prob.setup()
        prob.run_driver()

        # Optimal solution
        print('comp.f', prob['comp.f'])
        print('p2.xI', prob['p2.xI'])
        print('p1.xC', prob['p1.xC'])


@unittest.skipUnless(PETScVector, "PETSc is required.")
@unittest.skipUnless(MPI, "MPI is required.")
class MPIFeatureTests(unittest.TestCase):
    N_PROCS = 2

    def test_option_parallel(self):
        from openmdao.api import Problem, Group, IndepVarComp, SimpleGADriver
        from openmdao.test_suite.components.branin import Branin

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('xC', 7.5))
        model.add_subsystem('p2', IndepVarComp('xI', 0.0))
        model.add_subsystem('comp', Branin())

        model.connect('p2.xI', 'comp.x0')
        model.connect('p1.xC', 'comp.x1')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = SimpleGADriver()
        prob.driver.options['bits'] = {'p1.xC': 8}
        prob.driver.options['max_gen'] = 10
        prob.driver.options['run_parallel'] = True

        prob.setup(check=False)
        prob.run_driver()

        # Optimal solution
        print('comp.f', prob['comp.f'])
        print('p2.xI', prob['p2.xI'])
        print('p1.xC', prob['p1.xC'])


@unittest.skipUnless(PETScVector, "PETSc is required.")
@unittest.skipUnless(MPI, "MPI is required.")
class MPIFeatureTests4(unittest.TestCase):
    N_PROCS = 4

    def test_option_procs_per_model(self):
        from openmdao.api import Problem, Group, IndepVarComp, SimpleGADriver, ParallelGroup
        from openmdao.test_suite.components.branin import Branin

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('xC', 7.5))
        model.add_subsystem('p2', IndepVarComp('xI', 0.0))
        par = model.add_subsystem('par', ParallelGroup())

        par.add_subsystem('comp1', Branin())
        par.add_subsystem('comp2', Branin())

        model.connect('p2.xI', 'par.comp1.x0')
        model.connect('p1.xC', 'par.comp1.x1')
        model.connect('p2.xI', 'par.comp2.x0')
        model.connect('p1.xC', 'par.comp2.x1')

        model.add_subsystem('comp', ExecComp('f = f1 + f2'))
        model.connect('par.comp1.f', 'comp.f1')
        model.connect('par.comp2.f', 'comp.f2')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = SimpleGADriver()
        prob.driver.options['bits'] = {'p1.xC': 8}
        prob.driver.options['max_gen'] = 10
        prob.driver.options['pop_size'] = 25
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = 2

        prob.driver._randomstate = 1

        prob.setup(check=False)
        prob.run_driver()

        # Optimal solution
        print('comp.f', prob['comp.f'])
        print('p2.xI', prob['p2.xI'])
        print('p1.xC', prob['p1.xC'])

if __name__ == "__main__":
    unittest.main()
