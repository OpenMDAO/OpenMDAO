"""
Test DOE Driver and Generators.
"""
from __future__ import print_function, division

from six import PY3

import unittest

import os
import shutil
import tempfile
import json

import numpy as np

from openmdao.api import Problem, ExplicitComponent, IndepVarComp, ExecComp, \
    SqliteRecorder, CaseReader, PETScVector

from openmdao.drivers.doe_driver import DOEDriver
from openmdao.drivers.doe_generators import UniformGenerator, FullFactorialGenerator, \
    PlackettBurmanGenerator, BoxBehnkenGenerator, LatinHypercubeGenerator, JSONGenerator

from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.groups.parallel_groups import FanInGrouped

from openmdao.utils.assert_utils import assert_rel_error
from openmdao.utils.general_utils import run_driver

from openmdao.utils.mpi import MPI


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

    def test_no_generator(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 0.), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 0.), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        model.add_design_var('x', lower=-10, upper=10)
        model.add_design_var('y', lower=-10, upper=10)
        model.add_objective('f_xy')

        prob.driver = DOEDriver()
        prob.driver.add_recorder(SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cases = CaseReader("cases.sql").driver_cases

        self.assertEqual(cases.num_cases, 0)

    def test_uniform(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 0.), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 0.), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        model.add_design_var('x', lower=-10, upper=10)
        model.add_design_var('y', lower=-10, upper=10)
        model.add_objective('f_xy')

        prob.driver = DOEDriver(UniformGenerator(num_samples=5, seed=0))
        prob.driver.add_recorder(SqliteRecorder("cases.sql"))

        prob.setup()
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

        cases = CaseReader("cases.sql").driver_cases

        self.assertEqual(cases.num_cases, 5)

        for n in range(cases.num_cases):
            outputs = cases.get_case(n).outputs
            assert_rel_error(self, outputs['x'], expected[n]['x'], 1e-4)
            assert_rel_error(self, outputs['y'], expected[n]['y'], 1e-4)

    def test_full_factorial(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = DOEDriver(generator=FullFactorialGenerator(levels=3))
        prob.driver.add_recorder(SqliteRecorder("cases.sql"))

        prob.setup()
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

        cases = CaseReader("cases.sql").driver_cases

        self.assertEqual(cases.num_cases, 9)

        for n in range(cases.num_cases):
            outputs = cases.get_case(n).outputs
            self.assertEqual(outputs['x'], expected[n]['x'])
            self.assertEqual(outputs['y'], expected[n]['y'])
            self.assertEqual(outputs['f_xy'], expected[n]['f_xy'])

    def test_full_factorial_array(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('xy', np.array([0., 0.])), promotes=['*'])
        model.add_subsystem('comp', ParaboloidArray(), promotes=['*'])

        model.add_design_var('xy', lower=np.array([-50., -50.]), upper=np.array([50., 50.]))
        model.add_objective('f_xy')

        prob.driver = DOEDriver(FullFactorialGenerator(levels=3))
        prob.driver.add_recorder(SqliteRecorder("cases.sql"))

        prob.setup()
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

        cases = CaseReader("cases.sql").driver_cases

        self.assertEqual(cases.num_cases, 9)

        for n in range(cases.num_cases):
            outputs = cases.get_case(n).outputs
            self.assertEqual(outputs['xy'][0], expected[n]['xy'][0])
            self.assertEqual(outputs['xy'][1], expected[n]['xy'][1])

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
        prob.driver.add_recorder(SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        expected = {
            0: {'x': np.array([0.]), 'y': np.array([0.]), 'f_xy': np.array([22.00])},
            1: {'x': np.array([1.]), 'y': np.array([0.]), 'f_xy': np.array([17.00])},
            2: {'x': np.array([0.]), 'y': np.array([1.]), 'f_xy': np.array([31.00])},
            3: {'x': np.array([1.]), 'y': np.array([1.]), 'f_xy': np.array([27.00])},
        }

        cases = CaseReader("cases.sql").driver_cases

        self.assertEqual(cases.num_cases, 4)

        for n in range(cases.num_cases):
            outputs = cases.get_case(n).outputs
            self.assertEqual(outputs['x'], expected[n]['x'])
            self.assertEqual(outputs['y'], expected[n]['y'])
            self.assertEqual(outputs['f_xy'], expected[n]['f_xy'])

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
        prob.driver.add_recorder(SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cases = CaseReader("cases.sql").driver_cases

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
            outputs = cases.get_case(n).outputs
            self.assertEqual(outputs['x'], expected[n]['x'])
            self.assertEqual(outputs['y'], expected[n]['y'])
            self.assertEqual(outputs['z'], expected[n]['z'])

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

        prob.driver = DOEDriver()
        prob.driver.options['generator'] = LatinHypercubeGenerator(samples=4, seed=0)

        prob.driver.add_recorder(SqliteRecorder("cases.sql"))

        prob.setup()
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

        cases = CaseReader("cases.sql").driver_cases

        self.assertEqual(cases.num_cases, 4)

        for n in range(cases.num_cases):
            outputs = cases.get_case(n).outputs
            x = outputs['x']
            y = outputs['y']

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
        prob.driver.add_recorder(SqliteRecorder("cases.sql"))

        prob.setup()
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

        cases = CaseReader("cases.sql").driver_cases

        self.assertEqual(cases.num_cases, 4)

        for n in range(cases.num_cases):
            outputs = cases.get_case(n).outputs
            x = outputs['xy'][0]
            y = outputs['xy'][1]

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
        prob.driver.add_recorder(SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cases = CaseReader("cases.sql").driver_cases

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
            outputs = cases.get_case(n).outputs
            x = float(outputs['indep.x'])
            y = float(outputs['indep.y'])

            x_buckets_filled.add(int(x/bucket_size))
            y_buckets_filled.add(int(y/bucket_size))

            self.assertTrue(round(x, 3) in valid_values, '%f not in %s' % (x, valid_values))
            self.assertTrue(round(y, 3) in valid_values, '%f not in %s' % (y, valid_values))

        self.assertEqual(x_buckets_filled, all_buckets)
        self.assertEqual(y_buckets_filled, all_buckets)

    def test_json(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.setup()

        # save DOE case data to a JSON string
        cases = []
        case_gen = FullFactorialGenerator(levels=3)
        for case in case_gen(model.get_design_vars(recurse=True)):
            cases.append([(var, list(val)) for (var, val) in case])

        json_data = json.dumps(cases)

        # create DOEDriver using JSON data
        prob.driver = DOEDriver(generator=JSONGenerator(json_data))
        prob.driver.add_recorder(SqliteRecorder("cases.sql"))

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

        cases = CaseReader("cases.sql").driver_cases

        self.assertEqual(cases.num_cases, 9)

        for n in range(cases.num_cases):
            outputs = cases.get_case(n).outputs
            self.assertEqual(outputs['x'], expected[n]['x'])
            self.assertEqual(outputs['y'], expected[n]['y'])
            self.assertEqual(outputs['f_xy'], expected[n]['f_xy'])

    def test_json_errors(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.setup()

        # data does not contain a list
        cases ={'desvar': 1.0}

        with self.assertRaises(RuntimeError) as err:
            prob.driver = DOEDriver(generator=JSONGenerator(cases))
        self.assertEqual(str(err.exception),
                         "JSONGenerator was not provided valid DOE case data.")

        # data contains a list of non-list
        cases = [{'desvar': 1.0}]
        prob.driver = DOEDriver(generator=JSONGenerator(cases))

        with self.assertRaises(RuntimeError) as err:
            prob.run_driver()
        self.assertEqual(str(err.exception), "Invalid DOE case found, "
                         "expecting a list:\n{'desvar': 1.0}")

        # data contains a list of list, but one has the wrong length
        cases = [
            [['p1.x', 0.], ['p2.y', 0.]],
            [['p1.x', 1.], ['p2.y', 1., 'foo']]
        ]
        with open('cases.json', 'w') as f:
            f.write(json.dumps(cases))

            prob.driver = DOEDriver(generator=JSONGenerator(cases))

        with self.assertRaises(RuntimeError) as err:
            prob.run_driver()
        self.assertEqual(str(err.exception), "Invalid DOE case found, "
                         "expecting a list of name/value pairs:\n"
                         "[['p1.x', 1.0], ['p2.y', 1.0, 'foo']]")

        # data contains a list of list, but one case has an invalid design var
        cases = [
            [['p1.x', 0.], ['p2.y', 0.]],
            [['p1.x', 1.], ['p2.z', 1.]]
        ]
        with open('cases.json', 'w') as f:
            f.write(json.dumps(cases))

            prob.driver = DOEDriver(generator=JSONGenerator(cases))

        with self.assertRaises(RuntimeError) as err:
            prob.run_driver()
        self.assertEqual(str(err.exception), "Invalid DOE case found, "
                         "'p2.z' is not a valid design variable:\n"
                         "[['p1.x', 1.0], ['p2.z', 1.0]]")

        # data contains a list of list, but one case has multiple invalid design vars
        cases = [
            [['p1.x', 0.], ['p2.y', 0.]],
            [['p1.y', 1.], ['p2.z', 1.]]
        ]
        with open('cases.json', 'w') as f:
            f.write(json.dumps(cases))

            prob.driver = DOEDriver(generator=JSONGenerator(cases))

        with self.assertRaises(RuntimeError) as err:
            prob.run_driver()
        self.assertEqual(str(err.exception), "Invalid DOE case found, "
                         "['p1.y', 'p2.z'] are not valid design variables:\n"
                         "[['p1.y', 1.0], ['p2.z', 1.0]]")


@unittest.skipUnless(PETScVector, "PETSc is required.")
class TestParallelDOE(unittest.TestCase):

    N_PROCS = 4

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

    def test_indivisable_error(self):
        prob = Problem()

        prob.driver = DOEDriver(FullFactorialGenerator(levels=3))
        prob.driver.options['parallel'] =  3

        with self.assertRaises(RuntimeError) as context:
            prob.setup()

        self.assertEqual(str(context.exception),
                         "The number of processors is not evenly divisable by the "
                         "specified number of parallel cases.\n Provide a number of "
                         "processors that is a multiple of 3, or specify a number "
                         "of parallel cases that divides into 4.")

    def test_minprocs_error(self):
        prob = Problem(FanInGrouped())

        # require 2 procs for the ParallelGroup
        prob.model._proc_info['sub'] = (2, None, 1.0)

        # run cases on all procs
        prob.driver = DOEDriver(FullFactorialGenerator(levels=3))
        prob.driver.options['parallel'] =  True

        with self.assertRaises(RuntimeError) as context:
            prob.setup()

        self.assertEqual(str(context.exception),
                         ": MPI process allocation failed: can't meet min_procs "
                         "required for the following subsystems: ['sub']")

    def test_full_factorial(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = DOEDriver(FullFactorialGenerator(levels=3), parallel=True)
        prob.driver.add_recorder(SqliteRecorder("cases.sql"))

        prob.setup()

        failed, output = run_driver(prob)
        self.assertFalse(failed)

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

        size = prob.comm.size
        rank = prob.comm.rank

        # cases will be split across files for each proc
        filename = "cases.sql_%d" % rank

        expect_msg = "Cases from rank %d are being written to %s." % (rank, filename)
        self.assertTrue(expect_msg in output)

        cases = CaseReader(filename).driver_cases

        # cases recorded on this proc
        num_cases = cases.num_cases
        self.assertEqual(num_cases, len(expected)//size+(rank<len(expected)%size))

        for n in range(num_cases):
            outputs = cases.get_case(n).outputs
            idx = n * size + rank  # index of expected case

            self.assertEqual(outputs['x'], expected[idx]['x'])
            self.assertEqual(outputs['y'], expected[idx]['y'])
            self.assertEqual(outputs['f_xy'], expected[idx]['f_xy'])

        # total number of cases recorded across all procs
        num_cases = prob.comm.allgather(num_cases)
        self.assertEqual(sum(num_cases), len(expected))

    def test_fan_in_grouped(self):
        # run 2 cases at a time, each using 2 of our 4 procs
        doe_parallel = 2

        prob = Problem(FanInGrouped())
        model = prob.model

        model.add_design_var('iv.x1', lower=0.0, upper=1.0)
        model.add_design_var('iv.x2', lower=0.0, upper=1.0)

        model.add_objective('c3.y')

        prob.driver = DOEDriver(FullFactorialGenerator(levels=3))
        prob.driver.add_recorder(SqliteRecorder("cases.sql"))
        prob.driver.options['parallel'] =  doe_parallel

        prob.setup()

        failed, output = run_driver(prob)
        self.assertFalse(failed)

        prob.cleanup()

        expected = {
            0: {'iv.x1': np.array([0.]), 'iv.x2': np.array([0.]), 'c3.y': np.array([ 0.0])},
            1: {'iv.x1': np.array([.5]), 'iv.x2': np.array([0.]), 'c3.y': np.array([-3.0])},
            2: {'iv.x1': np.array([1.]), 'iv.x2': np.array([0.]), 'c3.y': np.array([-6.0])},

            3: {'iv.x1': np.array([0.]), 'iv.x2': np.array([.5]), 'c3.y': np.array([17.5])},
            4: {'iv.x1': np.array([.5]), 'iv.x2': np.array([.5]), 'c3.y': np.array([14.5])},
            5: {'iv.x1': np.array([1.]), 'iv.x2': np.array([.5]), 'c3.y': np.array([11.5])},

            6: {'iv.x1': np.array([0.]), 'iv.x2': np.array([1.]), 'c3.y': np.array([35.0])},
            7: {'iv.x1': np.array([.5]), 'iv.x2': np.array([1.]), 'c3.y': np.array([32.0])},
            8: {'iv.x1': np.array([1.]), 'iv.x2': np.array([1.]), 'c3.y': np.array([29.0])},
        }

        rank = prob.comm.rank
        size = prob.comm.size // doe_parallel

        num_cases = 0

        # cases will be split across files for each proc up to the number requested
        if rank < doe_parallel:
            filename = "cases.sql_%d" % rank

            expect_msg = "Cases from rank %d are being written to %s." % (rank, filename)
            self.assertTrue(expect_msg in output)

            cases = CaseReader(filename).driver_cases

            # cases recorded on this proc
            num_cases = cases.num_cases
            self.assertEqual(num_cases, len(expected)//size+(rank<len(expected)%size))

            for n in range(num_cases):
                idx = n * size + rank  # index of expected case

                outputs = cases.get_case(n).outputs

                self.assertEqual(outputs['iv.x1'], expected[idx]['iv.x1'])
                self.assertEqual(outputs['iv.x2'], expected[idx]['iv.x2'])
                self.assertEqual(outputs['c3.y'], expected[idx]['c3.y'])
        else:
            self.assertFalse("Cases from rank %d are being written" % rank in output)

        # total number of cases recorded across all requested procs
        num_cases = prob.comm.allgather(num_cases)
        self.assertEqual(sum(num_cases), len(expected))

    def test_fan_in_grouped_serial(self):
        # run cases on all procs (parallel model will run on single proc)
        doe_parallel = True

        prob = Problem(FanInGrouped())
        model = prob.model

        model.add_design_var('iv.x1', lower=0.0, upper=1.0)
        model.add_design_var('iv.x2', lower=0.0, upper=1.0)

        model.add_objective('c3.y')

        prob.driver = DOEDriver(FullFactorialGenerator(levels=3))
        prob.driver.add_recorder(SqliteRecorder("cases.sql"))
        prob.driver.options['parallel'] =  doe_parallel

        prob.setup()

        failed, output = run_driver(prob)
        self.assertFalse(failed)

        prob.cleanup()

        expected = {
            0: {'iv.x1': np.array([0.]), 'iv.x2': np.array([0.]), 'c3.y': np.array([ 0.0])},
            1: {'iv.x1': np.array([.5]), 'iv.x2': np.array([0.]), 'c3.y': np.array([-3.0])},
            2: {'iv.x1': np.array([1.]), 'iv.x2': np.array([0.]), 'c3.y': np.array([-6.0])},

            3: {'iv.x1': np.array([0.]), 'iv.x2': np.array([.5]), 'c3.y': np.array([17.5])},
            4: {'iv.x1': np.array([.5]), 'iv.x2': np.array([.5]), 'c3.y': np.array([14.5])},
            5: {'iv.x1': np.array([1.]), 'iv.x2': np.array([.5]), 'c3.y': np.array([11.5])},

            6: {'iv.x1': np.array([0.]), 'iv.x2': np.array([1.]), 'c3.y': np.array([35.0])},
            7: {'iv.x1': np.array([.5]), 'iv.x2': np.array([1.]), 'c3.y': np.array([32.0])},
            8: {'iv.x1': np.array([1.]), 'iv.x2': np.array([1.]), 'c3.y': np.array([29.0])},
        }

        rank = prob.comm.rank
        size = prob.comm.size // doe_parallel

        num_cases = 0

        # cases will be split across files for each proc up to the number requested
        filename = "cases.sql_%d" % rank

        expect_msg = "Cases from rank %d are being written to %s." % (rank, filename)
        self.assertTrue(expect_msg in output)

        cases = CaseReader(filename).driver_cases

        # cases recorded on this proc
        num_cases = cases.num_cases
        self.assertEqual(num_cases, len(expected)//size+(rank<len(expected)%size))

        for n in range(num_cases):
            idx = n * size + rank  # index of expected case

            outputs = cases.get_case(n).outputs

            self.assertEqual(outputs['iv.x1'], expected[idx]['iv.x1'])
            self.assertEqual(outputs['iv.x2'], expected[idx]['iv.x2'])
            self.assertEqual(outputs['c3.y'], expected[idx]['c3.y'])

        # total number of cases recorded across all requested procs
        num_cases = prob.comm.allgather(num_cases)
        self.assertEqual(sum(num_cases), len(expected))


class TestDOEDriverFeature(unittest.TestCase):

    def setUp(self):
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='TestDOEDriverFeature-')
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    def test_uniform(self):
        from openmdao.api import Problem, IndepVarComp
        from openmdao.test_suite.components.paraboloid import Paraboloid

        from openmdao.api import DOEDriver, UniformGenerator, SqliteRecorder, CaseReader

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 0.), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 0.), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        model.add_design_var('x', lower=-10, upper=10)
        model.add_design_var('y', lower=-10, upper=10)
        model.add_objective('f_xy')

        prob.driver = DOEDriver(UniformGenerator(num_samples=5))
        prob.driver.add_recorder(SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cases = CaseReader("cases.sql").driver_cases

        self.assertEqual(cases.num_cases, 5)

        values = []
        for n in range(cases.num_cases):
            outputs = cases.get_case(n).outputs
            values.append((outputs['x'], outputs['y'], outputs['f_xy']))

        print("\n".join(["x: %5.2f, y: %5.2f, f_xy: %6.2f" % (x, y, f_xy) for x, y, f_xy in values]))


@unittest.skipUnless(PETScVector, "PETSc is required.")
class TestParallelDOEFeature(unittest.TestCase):

    N_PROCS = 2

    def setUp(self):
        rank = MPI.COMM_WORLD.rank

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

        # expect odd cases on rank 0 and even cases on rank 1
        values = []
        for idx in range(len(expected)):
            if idx % 2 == rank:
                case = expected[idx]
                values.append((case['x'], case['y'], case['f_xy']))

        self.expect_text = "\n"+"\n".join([
            "x: %5.2f, y: %5.2f, f_xy: %6.2f" % (x, y, f_xy) for x, y, f_xy in values
        ])

        # run in temp dir
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='TestParallelDOEFeature-')
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    def test_full_factorial(self):
        from openmdao.api import Problem, IndepVarComp
        from openmdao.test_suite.components.paraboloid import Paraboloid

        from openmdao.api import DOEDriver, FullFactorialGenerator
        from openmdao.api import SqliteRecorder, CaseReader

        from mpi4py import MPI

        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = DOEDriver(FullFactorialGenerator(levels=3))
        prob.driver.options['parallel'] =  True

        prob.driver.add_recorder(SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        self.assertEqual(MPI.COMM_WORLD.size, 2)

        # check recorded cases from each case file
        rank = MPI.COMM_WORLD.rank
        filename = "cases.sql_%d" % rank
        self.assertEqual(filename, "cases.sql_%d" % rank)

        cases = CaseReader(filename).driver_cases
        self.assertEqual(cases.num_cases, 5 if rank == 0 else 4)

        values = []
        for n in range(cases.num_cases):
            outputs = cases.get_case(n).outputs
            values.append((outputs['x'], outputs['y'], outputs['f_xy']))

        self.assertEqual("\n"+"\n".join(["x: %5.2f, y: %5.2f, f_xy: %6.2f" % (x, y, f_xy) for x, y, f_xy in values]),
            self.expect_text)


@unittest.skipUnless(PETScVector, "PETSc is required.")
class TestParallelDOEFeature2(unittest.TestCase):

    N_PROCS = 4

    def setUp(self):
        rank = MPI.COMM_WORLD.rank

        expected = {
            0: {'iv.x1': np.array([0.]), 'iv.x2': np.array([0.]), 'c3.y': np.array([ 0.00])},
            1: {'iv.x1': np.array([.5]), 'iv.x2': np.array([0.]), 'c3.y': np.array([-3.00])},
            2: {'iv.x1': np.array([1.]), 'iv.x2': np.array([0.]), 'c3.y': np.array([-6.00])},

            3: {'iv.x1': np.array([0.]), 'iv.x2': np.array([.5]), 'c3.y': np.array([17.50])},
            4: {'iv.x1': np.array([.5]), 'iv.x2': np.array([.5]), 'c3.y': np.array([14.50])},
            5: {'iv.x1': np.array([1.]), 'iv.x2': np.array([.5]), 'c3.y': np.array([11.50])},

            6: {'iv.x1': np.array([0.]), 'iv.x2': np.array([1.]), 'c3.y': np.array([35.00])},
            7: {'iv.x1': np.array([.5]), 'iv.x2': np.array([1.]), 'c3.y': np.array([32.00])},
            8: {'iv.x1': np.array([1.]), 'iv.x2': np.array([1.]), 'c3.y': np.array([29.00])},
        }

        # expect odd cases on rank 0 and even cases on rank 1
        values = []
        for idx in range(len(expected)):
            if idx % 2 == rank:
                case = expected[idx]
                values.append((case['iv.x1'], case['iv.x2'], case['c3.y']))

        self.expect_text = "\n"+"\n".join(list([
            "iv.x1: %5.2f, iv.x2: %5.2f, c3.y: %6.2f" % (x1, x2, y) for x1, x2, y in values
        ]))

        # run in temp dir
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='TestParallelDOEFeature2-')
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    def test_fan_in_grouped(self):
        from openmdao.api import Problem
        from openmdao.test_suite.groups.parallel_groups import FanInGrouped

        from openmdao.api import DOEDriver, FullFactorialGenerator
        from openmdao.api import SqliteRecorder, CaseReader

        from mpi4py import MPI

        prob = Problem(FanInGrouped())
        model = prob.model

        model.add_design_var('iv.x1', lower=0.0, upper=1.0)
        model.add_design_var('iv.x2', lower=0.0, upper=1.0)

        model.add_objective('c3.y')

        prob.driver = DOEDriver(FullFactorialGenerator(levels=3))
        prob.driver.add_recorder(SqliteRecorder("cases.sql"))

        # run 2 cases at a time, each using 2 of our 4 procs
        doe_parallel = prob.driver.options['parallel'] = 2

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        rank = MPI.COMM_WORLD.rank

        # check recorded cases from each case file
        if rank < doe_parallel:
            filename = "cases.sql_%d" % rank

            cases = CaseReader(filename).driver_cases

            values = []
            for n in range(cases.num_cases):
                outputs = cases.get_case(n).outputs
                values.append((outputs['iv.x1'], outputs['iv.x2'], outputs['c3.y']))

            self.assertEqual("\n"+"\n".join(["iv.x1: %5.2f, iv.x2: %5.2f, c3.y: %6.2f" % (x1, x2, y) for x1, x2, y in values]),
                self.expect_text)


if __name__ == "__main__":
    unittest.main()
