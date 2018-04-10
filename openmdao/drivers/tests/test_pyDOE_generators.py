"""Test DOEDriver"""

import unittest

import os
import shutil
import tempfile
import platform

import numpy as np

from openmdao.api import Problem, ExplicitComponent, IndepVarComp, SqliteRecorder, CaseReader
from openmdao.drivers.doe_driver import DOEDriver
from openmdao.drivers.uniform_generator import UniformGenerator
# from openmdao.drivers.full_factorial_generator import FullFactorialGenerator
from openmdao.drivers.latin_hypercube_generator import OptimizedLatinHypercubeGenerator

from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.utils.assert_utils import assert_rel_error

from openmdao.drivers.factorial_generators import FullFactorialGenerator, PlackettBurmanGenerator


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


class TestDOEDriverErrors(unittest.TestCase):

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

        # all values should be between -10 and 10, specific values don't really
        # matter, we'll just check them to make sure they are not all zeros, etc.
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

    def test_optimized_latin_hypercube(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('xy', np.array([50., 50.])), promotes=['*'])
        model.add_subsystem('comp', ParaboloidArray(), promotes=['*'])

        model.add_design_var('xy', lower=np.array([-50., -50.]), upper=np.array([50., 50.]))
        model.add_objective('f_xy')

        prob.driver = DOEDriver(OptimizedLatinHypercubeGenerator(
            num_samples=4, seed=0, population=20, generations=4, norm_method=2
        ))
        prob.driver.add_recorder(SqliteRecorder("CASES.sql"))

        prob.setup(check=False)
        prob.run_driver()
        prob.cleanup()

        # the integer methods in the random module changed in Python 3.2
        # https://docs.python.org/dev/whatsnew/3.2.html#random
        if platform.python_version() < '3.2':
            expected = {
                0: {'xy': np.array([-11.279662,  -32.120265])},
                1: {'xy': np.array([ 40.069084,  -11.377920])},
                2: {'xy': np.array([ 10.5913699,  41.147352826])},
                3: {'xy': np.array([-39.06031971, 22.29432501])},
            }
        else:
            expected = {
                0: {'xy': np.array([ 38.7203376,  17.87973416])},
                1: {'xy': np.array([-34.9309156, -11.37792043])},
                2: {'xy': np.array([-14.40863002, 41.14735283])},
                3: {'xy': np.array([ 10.93968028,-27.70567498])},
            }

        cases = CaseReader("CASES.sql").driver_cases

        self.assertEqual(cases.num_cases, 4)

        for n in range(cases.num_cases):
            assert_rel_error(self, cases.get_case(n).desvars['xy'], expected[n]['xy'], 1e-4)


if __name__ == "__main__":
    unittest.main()
