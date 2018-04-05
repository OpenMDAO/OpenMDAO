"""Test DOEDriver"""

import unittest

import os
import shutil
import tempfile

import numpy as np

from openmdao.api import Problem, ExplicitComponent, IndepVarComp, SqliteRecorder, CaseReader
from openmdao.drivers.doe_driver import DOEDriver
from openmdao.drivers.full_factorial_generator import FullFactorialGenerator
from openmdao.drivers.latin_hypercube_generator import OptimizedLatinHypercubeGenerator

from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.utils.assert_utils import assert_rel_error

from pprint import pprint


class TestDOEDriver(unittest.TestCase):

    def test_generator_check(self):
        prob = Problem()

        with self.assertRaises(TypeError) as err:
            prob.driver = DOEDriver(FullFactorialGenerator)

        self.assertEqual(str(err.exception),
            "DOEDriver requires an instance of DOEGenerator, but a class "
            "object was found: FullFactorialGenerator")

        with self.assertRaises(TypeError) as err:
            prob.driver = DOEDriver(Problem())

        self.assertEqual(str(err.exception),
            "DOEDriver requires an instance of DOEGenerator, but an instance "
            "of Problem was found.")


class TestDOEDriverData(unittest.TestCase):

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

        prob.driver = DOEDriver(FullFactorialGenerator(levels=2))
        prob.driver.add_recorder(SqliteRecorder("CASES.sql"))

        prob.setup(check=False)
        prob.run_driver()
        prob.cleanup()

        expected = {
            0: {'x': np.array([0.]), 'y': np.array([0.]), 'f_xy': np.array([22.])},
            1: {'x': np.array([0.]), 'y': np.array([1.]), 'f_xy': np.array([31.])},
            2: {'x': np.array([1.]), 'y': np.array([0.]), 'f_xy': np.array([17.])},
            3: {'x': np.array([1.]), 'y': np.array([1.]), 'f_xy': np.array([27.])},
        }

        cases = CaseReader("CASES.sql").driver_cases

        self.assertEqual(cases.num_cases, 4)

        for n in range(cases.num_cases):
            self.assertEqual(cases.get_case(n).desvars['x'], expected[n]['x'])
            self.assertEqual(cases.get_case(n).desvars['y'], expected[n]['y'])
            self.assertEqual(cases.get_case(n).objectives['f_xy'], expected[n]['f_xy'])

    def test_optimized_latin_hypercube(self):
        class ParaboloidArray(ExplicitComponent):
            """
            Evaluates the equation f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3.

            Where x and y and xy[0] and xy[1] repectively.
            """

            def __init__(self):
                super(ParaboloidArray, self).__init__()

                self.add_input('xy', val=np.array([0., 0.]))
                self.add_output('f_xy', val=0.0)

            def compute(self, inputs, outputs):
                """f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
                """
                x = inputs['xy'][0]
                y = inputs['xy'][1]
                outputs['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

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

        cases = CaseReader("CASES.sql").driver_cases

        expected = {
            0: {'xy': np.array([-11.279662, -32.120265])},
            1: {'xy': np.array([ 40.069084, -11.377920])},
            2: {'xy': np.array([ 10.5913699, 41.147352826])},
            3: {'xy': np.array([-39.06031971, 22.29432501])},
        }

        cases = CaseReader("CASES.sql").driver_cases

        self.assertEqual(cases.num_cases, 4)

        for n in range(cases.num_cases):
            assert_rel_error(self, cases.get_case(n).desvars['xy'], expected[n]['xy'], 1e-4)


if __name__ == "__main__":
    unittest.main()
