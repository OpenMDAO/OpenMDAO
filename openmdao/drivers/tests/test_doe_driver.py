"""
Test DOE Driver and Generators.
"""
from __future__ import division

import unittest

import os
import shutil
import tempfile
import platform

import numpy as np

from openmdao.api import Problem, ExplicitComponent, IndepVarComp, ExecComp, \
    SqliteRecorder, CaseReader, PETScVector

from openmdao.drivers.doe_driver import DOEDriver
from openmdao.drivers.doe_generators import UniformGenerator, FullFactorialGenerator, \
    PlackettBurmanGenerator, BoxBehnkenGenerator, LatinHypercubeGenerator

from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.utils.assert_utils import assert_rel_error


class ParaboloidArray(ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + x*y + (y+4)^2 - 3.

    Where x and y are xy[0] and xy[1] repectively.
    """

    def __init__(self):
        super(ParaboloidArray, self).__init__()

        self.add_input('xy', val=np.array([0., 0.]))
        self.add_output('f_xy', val=0.0)

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
        """
        x = inputs['xy'][0]
        y = inputs['xy'][1]
        outputs['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0


class TestErrors(unittest.TestCase):

    def test_generator_check(self):
        prob = Problem()

        with self.assertRaises(TypeError) as err:
            prob.driver = DOEDriver(FullFactorialGenerator)

        self.assertEqual(str(err.exception),
                         "DOEDriver requires an instance of DOEGenerator, "
                         "but a class object was found: FullFactorialGenerator")

        with self.assertRaises(TypeError) as err:
            prob.driver = DOEDriver(Problem())

        self.assertEqual(str(err.exception),
                         "DOEDriver requires an instance of DOEGenerator, "
                         "but an instance of Problem was found.")

    def test_lhc_criterion(self):
        with self.assertRaises(ValueError) as err:
            LatinHypercubeGenerator(criterion='foo')

        self.assertEqual(str(err.exception),
                         "Invalid criterion 'foo' specified for LatinHypercubeGenerator. "
                         "Must be one of ['center', 'c', 'maximin', 'm', 'centermaximin', "
                         "'cm', 'correlation', 'corr', None].")


class TestDOEDriver(unittest.TestCase):

    def setUp(self):
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='TestDOEDriver-')
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    def test_uniform_generator(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        model.add_design_var('x', lower=-10, upper=10)
        model.add_design_var('y', lower=-10, upper=10)
        model.add_objective('f_xy')

        prob.driver = DOEDriver(UniformGenerator(num_samples=5, seed=0))
        prob.driver.add_recorder(SqliteRecorder("CASES.sql"))

        prob.setup(check=False)
        prob.run_driver()
        prob.cleanup()

        # all values should be between -10 and 10, check expected values for seed = 0
        expected = {
            0: {'x': np.array([ 0.97627008]), 'y': np.array([ 4.30378733])},
            1: {'x': np.array([ 2.05526752]), 'y': np.array([ 0.89766366])},
            2: {'x': np.array([-1.52690401]), 'y': np.array([ 2.91788226])},
            3: {'x': np.array([-1.24825577]), 'y': np.array([ 7.83546002])},
            4: {'x': np.array([ 9.27325521]), 'y': np.array([-2.33116962])},
        }

        cases = CaseReader("CASES.sql").driver_cases

        self.assertEqual(cases.num_cases, 5)

        for n in range(cases.num_cases):
            assert_rel_error(self, cases.get_case(n).desvars['x'], expected[n]['x'], 1e-4)
            assert_rel_error(self, cases.get_case(n).desvars['y'], expected[n]['y'], 1e-4)

    def test_full_factorial(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = DOEDriver(FullFactorialGenerator(levels=3))
        prob.driver.add_recorder(SqliteRecorder("CASES.sql"))

        prob.setup(check=False)
        prob.run_driver()
        prob.cleanup()

        expected = {
            0: {'x': np.array([0.]), 'y': np.array([0.]), 'f_xy': np.array([22.00])},
            1: {'x': np.array([.5]), 'y': np.array([0.]), 'f_xy': np.array([19.25])},
            2: {'x': np.array([1.]), 'y': np.array([0.]), 'f_xy': np.array([17.00])},

            3: {'x': np.array([0.]), 'y': np.array([.5]), 'f_xy': np.array([26.25])},
            4: {'x': np.array([.5]), 'y': np.array([.5]), 'f_xy': np.array([23.75])},
            5: {'x': np.array([1.]), 'y': np.array([.5]), 'f_xy': np.array([21.75])},

            6: {'x': np.array([0.]), 'y': np.array([1.]), 'f_xy': np.array([31.00])},
            7: {'x': np.array([.5]), 'y': np.array([1.]), 'f_xy': np.array([28.75])},
            8: {'x': np.array([1.]), 'y': np.array([1.]), 'f_xy': np.array([27.00])},
        }

        cases = CaseReader("CASES.sql").driver_cases

        self.assertEqual(cases.num_cases, 9)

        for n in range(cases.num_cases):
            self.assertEqual(cases.get_case(n).desvars['x'], expected[n]['x'])
            self.assertEqual(cases.get_case(n).desvars['y'], expected[n]['y'])
            self.assertEqual(cases.get_case(n).objectives['f_xy'], expected[n]['f_xy'])

    def test_full_factorial_array(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('xy', np.array([50., 50.])), promotes=['*'])
        model.add_subsystem('comp', ParaboloidArray(), promotes=['*'])

        model.add_design_var('xy', lower=np.array([-50., -50.]), upper=np.array([50., 50.]))
        model.add_objective('f_xy')

        prob.driver = DOEDriver(FullFactorialGenerator(levels=3))
        prob.driver.add_recorder(SqliteRecorder("CASES.sql"))

        prob.setup(check=False)
        prob.run_driver()
        prob.cleanup()

        expected = {
            0: {'xy': np.array([-50., -50.])},
            1: {'xy': np.array([  0., -50.])},
            2: {'xy': np.array([ 50., -50.])},
            3: {'xy': np.array([-50.,   0.])},
            4: {'xy': np.array([  0.,   0.])},
            5: {'xy': np.array([ 50.,   0.])},
            6: {'xy': np.array([-50.,  50.])},
            7: {'xy': np.array([  0.,  50.])},
            8: {'xy': np.array([ 50.,  50.])},
        }

        cases = CaseReader("CASES.sql").driver_cases

        self.assertEqual(cases.num_cases, 9)

        for n in range(cases.num_cases):
            self.assertEqual(cases.get_case(n).desvars['xy'][0], expected[n]['xy'][0])
            self.assertEqual(cases.get_case(n).desvars['xy'][1], expected[n]['xy'][1])

    def test_plackett_burman(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = DOEDriver(PlackettBurmanGenerator())
        prob.driver.add_recorder(SqliteRecorder("CASES.sql"))

        prob.setup(check=False)
        prob.run_driver()
        prob.cleanup()

        expected = {
            0: {'x': np.array([0.]), 'y': np.array([0.]), 'f_xy': np.array([22.00])},
            1: {'x': np.array([1.]), 'y': np.array([0.]), 'f_xy': np.array([17.00])},
            2: {'x': np.array([0.]), 'y': np.array([1.]), 'f_xy': np.array([31.00])},
            3: {'x': np.array([1.]), 'y': np.array([1.]), 'f_xy': np.array([27.00])},
        }

        cases = CaseReader("CASES.sql").driver_cases

        self.assertEqual(cases.num_cases, 4)

        for n in range(cases.num_cases):
            self.assertEqual(cases.get_case(n).desvars['x'], expected[n]['x'])
            self.assertEqual(cases.get_case(n).desvars['y'], expected[n]['y'])
            self.assertEqual(cases.get_case(n).objectives['f_xy'], expected[n]['f_xy'])

    def test_box_behnken(self):
        upper = 10.
        center = 1

        prob = Problem()
        model = prob.model

        indep = model.add_subsystem('indep', IndepVarComp(), promotes=['*'])
        indep.add_output('x', 0.0)
        indep.add_output('y', 0.0)
        indep.add_output('z', 0.0)

        model.add_subsystem('comp', ExecComp('a = x**2 + y - z'), promotes=['*'])

        model.add_design_var('x', lower=0., upper=upper)
        model.add_design_var('y', lower=0., upper=upper)
        model.add_design_var('z', lower=0., upper=upper)

        model.add_objective('a')

        prob.driver = DOEDriver(BoxBehnkenGenerator(center=center))
        prob.driver.add_recorder(SqliteRecorder("CASES.sql"))

        prob.setup(check=False)
        prob.run_driver()
        prob.cleanup()

        cases = CaseReader("CASES.sql").driver_cases

        # The Box-Behnken design for 3 factors involves three blocks, in each of
        # which 2 factors are varied thru the 4 possible combinations of high & low.
        # It also includes centre points (all factors at their central values).
        # ref: https://en.wikipedia.org/wiki/Box-Behnken_design
        self.assertEqual(cases.num_cases, (3*4)+center)

        expected = {
            0:  {'x': np.array([ 0.]), 'y': np.array([ 0.]), 'z': np.array([ 5.])},
            1:  {'x': np.array([10.]), 'y': np.array([ 0.]), 'z': np.array([ 5.])},
            2:  {'x': np.array([ 0.]), 'y': np.array([10.]), 'z': np.array([ 5.])},
            3:  {'x': np.array([10.]), 'y': np.array([10.]), 'z': np.array([ 5.])},

            4:  {'x': np.array([ 0.]), 'y': np.array([ 5.]), 'z': np.array([ 0.])},
            5:  {'x': np.array([10.]), 'y': np.array([ 5.]), 'z': np.array([ 0.])},
            6:  {'x': np.array([ 0.]), 'y': np.array([ 5.]), 'z': np.array([10.])},
            7:  {'x': np.array([10.]), 'y': np.array([ 5.]), 'z': np.array([10.])},

            8:  {'x': np.array([ 5.]), 'y': np.array([ 0.]), 'z': np.array([ 0.])},
            9:  {'x': np.array([ 5.]), 'y': np.array([10.]), 'z': np.array([ 0.])},
            10: {'x': np.array([ 5.]), 'y': np.array([ 0.]), 'z': np.array([10.])},
            11: {'x': np.array([ 5.]), 'y': np.array([10.]), 'z': np.array([10.])},

            12: {'x': np.array([ 5.]), 'y': np.array([ 5.]), 'z': np.array([ 5.])},
        }

        for n in range(cases.num_cases):
            self.assertEqual(cases.get_case(n).desvars['x'], expected[n]['x'])
            self.assertEqual(cases.get_case(n).desvars['y'], expected[n]['y'])
            self.assertEqual(cases.get_case(n).desvars['z'], expected[n]['z'])

    def test_latin_hypercube(self):
        samples = 4

        bounds = np.array([
            [-1, -10],  # lower bounds for x and y
            [ 1,  10]   # upper bounds for x and y
        ])
        xlb, xub = bounds[0][0], bounds[1][0]
        ylb, yub = bounds[0][1], bounds[1][1]

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=xlb, upper=xub)
        model.add_design_var('y', lower=ylb, upper=yub)
        model.add_objective('f_xy')

        prob.driver = DOEDriver(LatinHypercubeGenerator(samples=4, seed=0))
        prob.driver.add_recorder(SqliteRecorder("CASES.sql"))

        prob.setup(check=False)
        prob.run_driver()
        prob.cleanup()

        # the sample space for each variable should be divided into equal
        # size buckets and each variable should have a value in each bucket
        all_buckets = set(range(samples))

        xlb, xub = bounds[0][0], bounds[1][0]
        x_offset = 0 - xlb
        x_bucket_size = xub - xlb
        x_buckets_filled = set()

        ylb, yub = bounds[0][1], bounds[1][1]
        y_offset = 0 - ylb
        y_bucket_size = yub - ylb
        y_buckets_filled = set()

        # expected values for seed = 0
        expected = {
            0: {'x': np.array([-0.19861831]), 'y': np.array([-6.42405317])},
            1: {'x': np.array([ 0.2118274]),  'y': np.array([ 9.458865])},
            2: {'x': np.array([ 0.71879361]), 'y': np.array([ 3.22947057])},
            3: {'x': np.array([-0.72559325]), 'y': np.array([-2.27558409])},
        }

        cases = CaseReader("CASES.sql").driver_cases

        self.assertEqual(cases.num_cases, 4)

        for n in range(cases.num_cases):
            x = cases.get_case(n).desvars['x']
            y = cases.get_case(n).desvars['y']

            bucket = int((x+x_offset)/(x_bucket_size/samples))
            x_buckets_filled.add(bucket)

            bucket = int((y+y_offset)/(y_bucket_size/samples))
            y_buckets_filled.add(bucket)

            assert_rel_error(self, x, expected[n]['x'], 1e-4)
            assert_rel_error(self, y, expected[n]['y'], 1e-4)

        self.assertEqual(x_buckets_filled, all_buckets)
        self.assertEqual(y_buckets_filled, all_buckets)

    def test_latin_hypercube_array(self):
        samples = 4

        bounds = np.array([
            [-10, -50],  # lower bounds for x and y
            [ 10,  50]   # upper bounds for x and y
        ])

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('xy', np.array([50., 50.])), promotes=['*'])
        model.add_subsystem('comp', ParaboloidArray(), promotes=['*'])

        model.add_design_var('xy', lower=bounds[0], upper=bounds[1])
        model.add_objective('f_xy')

        prob.driver = DOEDriver(LatinHypercubeGenerator(samples=4, seed=0))
        prob.driver.add_recorder(SqliteRecorder("CASES.sql"))

        prob.setup(check=False)
        prob.run_driver()
        prob.cleanup()

        # the sample space for each variable should be divided into equal
        # size buckets and each variable should have a value in each bucket
        all_buckets = set(range(samples))

        xlb, xub = bounds[0][0], bounds[1][0]
        x_offset = 0 - xlb
        x_bucket_size = xub - xlb
        x_buckets_filled = set()

        ylb, yub = bounds[0][1], bounds[1][1]
        y_offset = 0 - ylb
        y_bucket_size = yub - ylb
        y_buckets_filled = set()

        # expected values for seed = 0
        expected = {
            0: {'xy': np.array([-1.98618312, -32.12026584])},
            1: {'xy': np.array([ 2.118274,    47.29432502])},
            2: {'xy': np.array([ 7.18793606,  16.14735283])},
            3: {'xy': np.array([-7.25593248, -11.37792043])},
        }

        cases = CaseReader("CASES.sql").driver_cases

        self.assertEqual(cases.num_cases, 4)

        for n in range(cases.num_cases):
            x = cases.get_case(n).desvars['xy'][0]
            y = cases.get_case(n).desvars['xy'][1]

            bucket = int((x+x_offset)/(x_bucket_size/samples))
            x_buckets_filled.add(bucket)

            bucket = int((y+y_offset)/(y_bucket_size/samples))
            y_buckets_filled.add(bucket)

            assert_rel_error(self, x, expected[n]['xy'][0], 1e-4)
            assert_rel_error(self, y, expected[n]['xy'][1], 1e-4)

        self.assertEqual(x_buckets_filled, all_buckets)
        self.assertEqual(y_buckets_filled, all_buckets)

    def test_latin_hypercube_center(self):
        samples = 4
        upper = 10.

        prob = Problem()
        model = prob.model

        indep = model.add_subsystem('indep', IndepVarComp())
        indep.add_output('x', 0.0)
        indep.add_output('y', 0.0)

        model.add_subsystem('comp', Paraboloid())

        model.connect('indep.x', 'comp.x')
        model.connect('indep.y', 'comp.y')

        model.add_design_var('indep.x', lower=0., upper=upper)
        model.add_design_var('indep.y', lower=0., upper=upper)

        model.add_objective('comp.f_xy')

        prob.driver = DOEDriver(LatinHypercubeGenerator(samples=samples, criterion='c'))
        prob.driver.add_recorder(SqliteRecorder("CASES.sql"))

        prob.setup(check=False)
        prob.run_driver()
        prob.cleanup()

        cases = CaseReader("CASES.sql").driver_cases

        self.assertEqual(cases.num_cases, samples)

        # the sample space for each variable (0 to upper) should be divided into
        # equal size buckets and each variable should have a value in each bucket
        bucket_size = upper/samples
        all_buckets = set(range(samples))

        x_buckets_filled = set()
        y_buckets_filled = set()

        # with criterion of 'center', each value should be in the center of it's bucket
        valid_values = [round(bucket_size*(bucket + 1/2), 3) for bucket in all_buckets]

        for n in range(cases.num_cases):
            x = float(cases.get_case(n).desvars['indep.x'])
            y = float(cases.get_case(n).desvars['indep.y'])

            x_buckets_filled.add(int(x/bucket_size))
            y_buckets_filled.add(int(y/bucket_size))

            self.assertTrue(round(x, 3) in valid_values, '%f not in %s' % (x, valid_values))
            self.assertTrue(round(y, 3) in valid_values, '%f not in %s' % (y, valid_values))

        self.assertEqual(x_buckets_filled, all_buckets)
        self.assertEqual(y_buckets_filled, all_buckets)


@unittest.skipUnless(PETScVector, "PETSc is required.")
class TestParallelDOE(unittest.TestCase):

    N_PROCS = 2

    def setUp(self):
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='TestDOEDriver-')
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    def test_full_factorial(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = DOEDriver(FullFactorialGenerator(levels=3))
        prob.driver.add_recorder(SqliteRecorder("CASES.sql"))
        prob.driver.options['run_parallel'] =  True

        prob.setup(vector_class=PETScVector)
        prob.run_driver()
        prob.cleanup()

        expected = {
            0: {'x': np.array([0.]), 'y': np.array([0.]), 'f_xy': np.array([22.00])},
            1: {'x': np.array([.5]), 'y': np.array([0.]), 'f_xy': np.array([19.25])},
            2: {'x': np.array([1.]), 'y': np.array([0.]), 'f_xy': np.array([17.00])},

            3: {'x': np.array([0.]), 'y': np.array([.5]), 'f_xy': np.array([26.25])},
            4: {'x': np.array([.5]), 'y': np.array([.5]), 'f_xy': np.array([23.75])},
            5: {'x': np.array([1.]), 'y': np.array([.5]), 'f_xy': np.array([21.75])},

            6: {'x': np.array([0.]), 'y': np.array([1.]), 'f_xy': np.array([31.00])},
            7: {'x': np.array([.5]), 'y': np.array([1.]), 'f_xy': np.array([28.75])},
            8: {'x': np.array([1.]), 'y': np.array([1.]), 'f_xy': np.array([27.00])},
        }

        if model.comm.rank == 0:
            cases = CaseReader("CASES.sql").driver_cases
            print('# cases:', cases.num_cases)

            # self.assertEqual(cases.num_cases, 9)

            for n in range(cases.num_cases):
                case = cases.get_case(n)
                print('-----------')
                print('filename:', case.filename)
                print('counter:', case.counter)
                print('iteration_coordinate:', case.iteration_coordinate)
                print('timestamp:', case.timestamp)
                print('success:', case.success)
                print('msg:', case.msg)
                print(cases.get_case(n).desvars['x'],
                      cases.get_case(n).desvars['y'])
                      # cases.get_case(n).desvars['f_xy'])
                # self.assertEqual(cases.get_case(n).desvars['x'], expected[n]['x'])
                # self.assertEqual(cases.get_case(n).desvars['y'], expected[n]['y'])
                # self.assertEqual(cases.get_case(n).objectives['f_xy'], expected[n]['f_xy'])


if __name__ == "__main__":
    unittest.main()
