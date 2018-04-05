"""Testing FullFactorialDriver"""

import unittest

import os
import shutil
import tempfile

import numpy as np

from openmdao.api import Problem, IndepVarComp, SqliteRecorder, CaseReader
from openmdao.drivers.doe_driver import DOEDriver, FullFactorialGenerator

from openmdao.test_suite.components.paraboloid import Paraboloid

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

    def test_fullfactorial(self):
        prob = Problem()
        model = prob.model

        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = DOEDriver(FullFactorialGenerator(num_levels=2))
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


if __name__ == "__main__":
    unittest.main()
