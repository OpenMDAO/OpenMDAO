"""
Test DOE Driver and Generators.
"""
import unittest

import os
import shutil
import tempfile
import csv
import json

import numpy as np

import openmdao.api as om

from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.paraboloid_distributed import DistParab
from openmdao.test_suite.groups.parallel_groups import FanInGrouped

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.general_utils import run_driver, printoptions
from openmdao.utils.testing_utils import use_tempdirs

from openmdao.utils.mpi import MPI

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


class ParaboloidArray(om.ExplicitComponent):
    """
    Evaluates the equation f(x,y) = (x-3)^2 + x*y + (y+4)^2 - 3.

    Where x and y are xy[0] and xy[1] respectively.
    """

    def setup(self):
        self.add_input('xy', val=np.array([0., 0.]))
        self.add_output('f_xy', val=0.0)

    def compute(self, inputs, outputs):
        """
        f(x,y) = (x-3)^2 + xy + (y+4)^2 - 3
        """
        x = inputs['xy'][0]
        y = inputs['xy'][1]
        outputs['f_xy'] = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0


class ParaboloidDiscrete(om.ExplicitComponent):

    def setup(self):
        self.add_discrete_input('x', val=10, tags='xx')
        self.add_discrete_input('y', val=0, tags='yy')
        self.add_discrete_output('f_xy', val=0, tags='ff')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        x = discrete_inputs['x']
        y = discrete_inputs['y']
        f_xy = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0
        discrete_outputs['f_xy'] = int(f_xy)


class ParaboloidDiscreteArray(om.ExplicitComponent):

    def setup(self):
        self.add_discrete_input('x', val=np.ones((2, )), tags='xx')
        self.add_discrete_input('y', val=np.ones((2, )), tags='yy')
        self.add_discrete_output('f_xy', val=np.ones((2, )), tags='ff')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        x = discrete_inputs['x']
        y = discrete_inputs['y']
        f_xy = (x - 3.0)**2 + x * y + (y + 4.0)**2 - 3.0
        discrete_outputs['f_xy'] = f_xy.astype(np.int)


class TestErrors(unittest.TestCase):

    def test_generator_check(self):
        prob = om.Problem()

        with self.assertRaises(TypeError) as err:
            prob.driver = om.DOEDriver(om.FullFactorialGenerator)

        self.assertEqual(str(err.exception),
                         "DOEDriver requires an instance of DOEGenerator, "
                         "but a class object was found: FullFactorialGenerator")

        with self.assertRaises(TypeError) as err:
            prob.driver = om.DOEDriver(om.Problem())

        self.assertEqual(str(err.exception),
                         "DOEDriver requires an instance of DOEGenerator, "
                         "but an instance of Problem was found.")

    def test_lhc_criterion(self):
        with self.assertRaises(ValueError) as err:
            om.LatinHypercubeGenerator(criterion='foo')

        self.assertEqual(str(err.exception),
                         "Invalid criterion 'foo' specified for LatinHypercubeGenerator. "
                         "Must be one of ['center', 'c', 'maximin', 'm', 'centermaximin', "
                         "'cm', 'correlation', 'corr', None].")


@use_tempdirs
class TestDOEDriver(unittest.TestCase):

    def setUp(self):
        self.expected_fullfact3 = [
            {'x': np.array([0.]), 'y': np.array([0.]), 'f_xy': np.array([22.00])},
            {'x': np.array([.5]), 'y': np.array([0.]), 'f_xy': np.array([19.25])},
            {'x': np.array([1.]), 'y': np.array([0.]), 'f_xy': np.array([17.00])},

            {'x': np.array([0.]), 'y': np.array([.5]), 'f_xy': np.array([26.25])},
            {'x': np.array([.5]), 'y': np.array([.5]), 'f_xy': np.array([23.75])},
            {'x': np.array([1.]), 'y': np.array([.5]), 'f_xy': np.array([21.75])},

            {'x': np.array([0.]), 'y': np.array([1.]), 'f_xy': np.array([31.00])},
            {'x': np.array([.5]), 'y': np.array([1.]), 'f_xy': np.array([28.75])},
            {'x': np.array([1.]), 'y': np.array([1.]), 'f_xy': np.array([27.00])},
        ]

    def test_no_generator(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        model.add_design_var('x', lower=-10, upper=10)
        model.add_design_var('y', lower=-10, upper=10)
        model.add_objective('f_xy')

        prob.driver = om.DOEDriver()
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), 0)

    def test_list(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.setup()

        # create a list of DOE cases
        case_gen = om.FullFactorialGenerator(levels=3)
        cases = list(case_gen(model.get_design_vars(recurse=True)))

        # create DOEDriver using provided list of cases
        prob.driver = om.DOEDriver(cases)
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.run_driver()
        prob.cleanup()

        expected = self.expected_fullfact3

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), 9)

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            for name in ('x', 'y', 'f_xy'):
                self.assertEqual(outputs[name], expected_case[name])

    def test_list_errors(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.setup()

        # data does not contain a list
        cases = {'desvar': 1.0}

        with self.assertRaises(RuntimeError) as err:
            prob.driver = om.DOEDriver(generator=om.ListGenerator(cases))
        self.assertEqual(str(err.exception), "Invalid DOE case data, "
                         "expected a list but got a dict.")

        # data contains a list of non-list
        cases = [{'desvar': 1.0}]
        prob.driver = om.DOEDriver(generator=om.ListGenerator(cases))

        with self.assertRaises(RuntimeError) as err:
            prob.run_driver()
        self.assertEqual(str(err.exception), "Invalid DOE case found, "
                         "expecting a list of name/value pairs:\n{'desvar': 1.0}")

        # data contains a list of list, but one has the wrong length
        cases = [
            [['p1.x', 0.], ['p2.y', 0.]],
            [['p1.x', 1.], ['p2.y', 1., 'foo']]
        ]

        prob.driver = om.DOEDriver(generator=om.ListGenerator(cases))

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

        prob.driver = om.DOEDriver(generator=om.ListGenerator(cases))

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

        prob.driver = om.DOEDriver(generator=om.ListGenerator(cases))

        with self.assertRaises(RuntimeError) as err:
            prob.run_driver()
        self.assertEqual(str(err.exception), "Invalid DOE case found, "
                         "['p1.y', 'p2.z'] are not valid design variables:\n"
                         "[['p1.y', 1.0], ['p2.z', 1.0]]")

    def test_csv(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])
        model.set_input_defaults('x', 0.0)
        model.set_input_defaults('y', 0.0)
        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.setup()

        # create a list of DOE cases
        case_gen = om.FullFactorialGenerator(levels=3)
        cases = list(case_gen(model.get_design_vars(recurse=True)))

        # generate CSV file with cases
        header = [var for (var, val) in cases[0]]
        with open('cases.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for case in cases:
                writer.writerow([val for _, val in case])

        # create DOEDriver using generated CSV file
        prob.driver = om.DOEDriver(om.CSVGenerator('cases.csv'))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.run_driver()
        prob.cleanup()

        expected = self.expected_fullfact3

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), 9)

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            for name in ('x', 'y', 'f_xy'):
                self.assertEqual(outputs[name], expected_case[name])

    def test_csv_array(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', [0., 1.]))
        model.add_subsystem('p2', om.IndepVarComp('y', [0., 1.]))
        model.add_subsystem('comp1', Paraboloid())
        model.add_subsystem('comp2', Paraboloid())

        model.connect('p1.x', 'comp1.x', src_indices=[0])
        model.connect('p2.y', 'comp1.y', src_indices=[0])

        model.connect('p1.x', 'comp2.x', src_indices=[1])
        model.connect('p2.y', 'comp2.y', src_indices=[1])

        model.add_design_var('p1.x', lower=0.0, upper=1.0)
        model.add_design_var('p2.y', lower=0.0, upper=1.0)

        prob.setup()

        # create a list of DOE cases
        case_gen = om.FullFactorialGenerator(levels=2)
        cases = list(case_gen(model.get_design_vars(recurse=True)))

        # generate CSV file with cases
        header = [var for var, _ in cases[0]]
        with open('cases.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for case in cases:
                writer.writerow([val for _, val in case])

        # create DOEDriver using generated CSV file
        prob.driver = om.DOEDriver(om.CSVGenerator('cases.csv'))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.run_driver()
        prob.cleanup()

        expected = [
            {'p1.x': np.array([0., 0.]), 'p2.y': np.array([0., 0.])},
            {'p1.x': np.array([1., 0.]), 'p2.y': np.array([0., 0.])},
            {'p1.x': np.array([0., 1.]), 'p2.y': np.array([0., 0.])},
            {'p1.x': np.array([1., 1.]), 'p2.y': np.array([0., 0.])},
            {'p1.x': np.array([0., 0.]), 'p2.y': np.array([1., 0.])},
            {'p1.x': np.array([1., 0.]), 'p2.y': np.array([1., 0.])},
            {'p1.x': np.array([0., 1.]), 'p2.y': np.array([1., 0.])},
            {'p1.x': np.array([1., 1.]), 'p2.y': np.array([1., 0.])},
            {'p1.x': np.array([0., 0.]), 'p2.y': np.array([0., 1.])},
            {'p1.x': np.array([1., 0.]), 'p2.y': np.array([0., 1.])},
            {'p1.x': np.array([0., 1.]), 'p2.y': np.array([0., 1.])},
            {'p1.x': np.array([1., 1.]), 'p2.y': np.array([0., 1.])},
            {'p1.x': np.array([0., 0.]), 'p2.y': np.array([1., 1.])},
            {'p1.x': np.array([1., 0.]), 'p2.y': np.array([1., 1.])},
            {'p1.x': np.array([0., 1.]), 'p2.y': np.array([1., 1.])},
            {'p1.x': np.array([1., 1.]), 'p2.y': np.array([1., 1.])},
        ]

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), 16)

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            self.assertEqual(outputs['p1.x'][0], expected_case['p1.x'][0])
            self.assertEqual(outputs['p2.y'][0], expected_case['p2.y'][0])
            self.assertEqual(outputs['p1.x'][1], expected_case['p1.x'][1])
            self.assertEqual(outputs['p2.y'][1], expected_case['p2.y'][1])

    def test_csv_errors(self):
        # test invalid file name
        with self.assertRaises(RuntimeError) as err:
            om.CSVGenerator(1.23)
        self.assertEqual(str(err.exception),
                         "'1.23' is not a valid file name.")

        # test file not found
        with self.assertRaises(RuntimeError) as err:
            om.CSVGenerator('nocases.csv')
        self.assertEqual(str(err.exception),
                         "File not found: nocases.csv")

        # create problem and a list of DOE cases
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.setup()

        case_gen = om.FullFactorialGenerator(levels=2)
        cases = list(case_gen(model.get_design_vars(recurse=True)))

        # test CSV file with an invalid design var
        header = [var for var, _ in cases[0]]
        header[-1] = 'foobar'
        with open('cases.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for case in cases:
                writer.writerow([val for _, val in case])

        prob.driver = om.DOEDriver(om.CSVGenerator('cases.csv'))
        with self.assertRaises(RuntimeError) as err:
            prob.run_driver()
        self.assertEqual(str(err.exception), "Invalid DOE case file, "
                         "'foobar' is not a valid design variable.")

        # test CSV file with invalid design vars
        header = [var + '_bad' for var, _ in cases[0]]
        with open('cases.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for case in cases:
                writer.writerow([val for _, val in case])

        with self.assertRaises(RuntimeError) as err:
            prob.run_driver()
        self.assertEqual(str(err.exception), "Invalid DOE case file, "
                         "%s are not valid design variables." %
                         str(header))

        # test CSV file with invalid values
        header = [var for var, _ in cases[0]]
        with open('cases.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for case in cases:
                writer.writerow([np.ones((2, 2)) * val for _, val in case])

        from distutils.version import LooseVersion
        if LooseVersion(np.__version__) >= LooseVersion("1.14"):
            opts = {'legacy': '1.13'}
        else:
            opts = {}

        with printoptions(**opts):
            with self.assertRaises(ValueError) as err:
                prob.run_driver()
            self.assertEqual(str(err.exception),
                             "Error assigning p1.x = [ 0.  0.  0.  0.]: "
                             "could not broadcast input array from shape (4) into shape (1)")

    def test_uniform(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.set_input_defaults('x', 0.0)
        model.set_input_defaults('y', 0.0)

        model.add_design_var('x', lower=-10, upper=10)
        model.add_design_var('y', lower=-10, upper=10)
        model.add_objective('f_xy')

        prob.driver = om.DOEDriver(om.UniformGenerator(num_samples=5, seed=0))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        # all values should be between -10 and 10, check expected values for seed = 0
        expected = [
            {'x': np.array([0.97627008]), 'y': np.array([4.30378733])},
            {'x': np.array([2.05526752]), 'y': np.array([0.89766366])},
            {'x': np.array([-1.52690401]), 'y': np.array([2.91788226])},
            {'x': np.array([-1.24825577]), 'y': np.array([7.83546002])},
            {'x': np.array([9.27325521]), 'y': np.array([-2.33116962])},
        ]

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), 5)

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            for name in ('x', 'y'):
                assert_near_equal(outputs[name], expected_case[name], 1e-4)

    def test_full_factorial(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])
        model.set_input_defaults('x', 0.0)
        model.set_input_defaults('y', 0.0)
        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = om.DOEDriver(generator=om.FullFactorialGenerator(levels=3))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        expected = self.expected_fullfact3

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), 9)

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            for name in ('x', 'y', 'f_xy'):
                self.assertEqual(outputs[name], expected_case[name])

    def test_full_factorial_factoring(self):

        class Digits2Num(om.ExplicitComponent):
            """
            Makes from two vectors with 2 elements a 4 digit number.
            For singe digit integers always gives a unique output number.
            """

            def setup(self):
                self.add_input('x', val=np.array([0., 0.]))
                self.add_input('y', val=np.array([0., 0.]))
                self.add_output('f', val=0.0)

            def compute(self, inputs, outputs):
                x = inputs['x']
                y = inputs['y']
                outputs['f'] = x[0] * 1000 + x[1] * 100 + y[0] * 10 + y[1]

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', np.array([0.0, 0.0])), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', np.array([0.0, 0.0])), promotes=['*'])
        model.add_subsystem('comp', Digits2Num(), promotes=['*'])

        model.add_design_var('x', lower=0.0, upper=np.array([1.0, 2.0]))
        model.add_design_var('y', lower=0.0, upper=np.array([3.0, 4.0]))
        model.add_objective('f')

        prob.driver = om.DOEDriver(generator=om.FullFactorialGenerator(levels=2))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        objs = [int(cr.get_case(case).outputs['f']) for case in cases]

        self.assertEqual(len(objs), 16)
        # Testing uniqueness. If all elements are unique, it should be the same length as the
        # number of cases
        self.assertEqual(len(set(objs)), 16)

    def test_full_factorial_array(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('xy', np.array([0., 0.])), promotes=['*'])
        model.add_subsystem('comp', ParaboloidArray(), promotes=['*'])

        model.add_design_var('xy', lower=np.array([-10., -50.]), upper=np.array([10., 50.]))
        model.add_objective('f_xy')

        prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=3))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        expected = [
            {'xy': np.array([-10., -50.])},
            {'xy': np.array([0., -50.])},
            {'xy': np.array([10., -50.])},

            {'xy': np.array([-10.,   0.])},
            {'xy': np.array([0.,   0.])},
            {'xy': np.array([10.,   0.])},

            {'xy': np.array([-10.,  50.])},
            {'xy': np.array([0.,  50.])},
            {'xy': np.array([10.,  50.])},
        ]

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), 9)

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            self.assertEqual(outputs['xy'][0], expected_case['xy'][0])
            self.assertEqual(outputs['xy'][1], expected_case['xy'][1])

    def test_plackett_burman(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = om.DOEDriver(om.PlackettBurmanGenerator())
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        expected = [
            {'x': np.array([0.]), 'y': np.array([0.]), 'f_xy': np.array([22.00])},
            {'x': np.array([1.]), 'y': np.array([0.]), 'f_xy': np.array([17.00])},
            {'x': np.array([0.]), 'y': np.array([1.]), 'f_xy': np.array([31.00])},
            {'x': np.array([1.]), 'y': np.array([1.]), 'f_xy': np.array([27.00])},
        ]

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), 4)

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            for name in ('x', 'y', 'f_xy'):
                self.assertEqual(outputs[name], expected_case[name])

    def test_box_behnken(self):
        upper = 10.
        center = 1

        prob = om.Problem()
        model = prob.model

        indep = model.add_subsystem('indep', om.IndepVarComp(), promotes=['*'])
        indep.add_output('x', 0.0)
        indep.add_output('y', 0.0)
        indep.add_output('z', 0.0)

        model.add_subsystem('comp', om.ExecComp('a = x**2 + y - z'), promotes=['*'])

        model.add_design_var('x', lower=0., upper=upper)
        model.add_design_var('y', lower=0., upper=upper)
        model.add_design_var('z', lower=0., upper=upper)

        model.add_objective('a')

        prob.driver = om.DOEDriver(om.BoxBehnkenGenerator(center=center))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        # The Box-Behnken design for 3 factors involves three blocks, in each of
        # which 2 factors are varied thru the 4 possible combinations of high & low.
        # It also includes centre points (all factors at their central values).
        # ref: https://en.wikipedia.org/wiki/Box-Behnken_design
        self.assertEqual(len(cases), (3*4)+center)

        expected = [
            {'x': np.array([0.]), 'y': np.array([0.]), 'z': np.array([5.])},
            {'x': np.array([10.]), 'y': np.array([0.]), 'z': np.array([5.])},
            {'x': np.array([0.]), 'y': np.array([10.]), 'z': np.array([5.])},
            {'x': np.array([10.]), 'y': np.array([10.]), 'z': np.array([5.])},

            {'x': np.array([0.]), 'y': np.array([5.]), 'z': np.array([0.])},
            {'x': np.array([10.]), 'y': np.array([5.]), 'z': np.array([0.])},
            {'x': np.array([0.]), 'y': np.array([5.]), 'z': np.array([10.])},
            {'x': np.array([10.]), 'y': np.array([5.]), 'z': np.array([10.])},

            {'x': np.array([5.]), 'y': np.array([0.]), 'z': np.array([0.])},
            {'x': np.array([5.]), 'y': np.array([10.]), 'z': np.array([0.])},
            {'x': np.array([5.]), 'y': np.array([0.]), 'z': np.array([10.])},
            {'x': np.array([5.]), 'y': np.array([10.]), 'z': np.array([10.])},

            {'x': np.array([5.]), 'y': np.array([5.]), 'z': np.array([5.])},
        ]

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            for name in ('x', 'y', 'z'):
                self.assertEqual(outputs[name], expected_case[name])

    def test_latin_hypercube(self):
        samples = 4

        bounds = np.array([
            [-1, -10],  # lower bounds for x and y
            [1,  10]   # upper bounds for x and y
        ])
        xlb, xub = bounds[0][0], bounds[1][0]
        ylb, yub = bounds[0][1], bounds[1][1]

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=xlb, upper=xub)
        model.add_design_var('y', lower=ylb, upper=yub)
        model.add_objective('f_xy')

        prob.driver = om.DOEDriver()
        prob.driver.options['generator'] = om.LatinHypercubeGenerator(samples=4, seed=0)

        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        # the sample space for each variable should be divided into equal
        # size buckets and each variable should have a value in each bucket
        all_buckets = set(range(samples))

        x_offset = - xlb
        x_bucket_size = xub - xlb
        x_buckets_filled = set()

        y_offset = - ylb
        y_bucket_size = yub - ylb
        y_buckets_filled = set()

        # expected values for seed = 0
        expected = [
            {'x': np.array([-0.19861831]), 'y': np.array([-6.42405317])},
            {'x': np.array([0.2118274]),  'y': np.array([9.458865])},
            {'x': np.array([0.71879361]), 'y': np.array([3.22947057])},
            {'x': np.array([-0.72559325]), 'y': np.array([-2.27558409])},
        ]

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), 4)

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            x = outputs['x']
            y = outputs['y']

            bucket = int((x + x_offset) / (x_bucket_size / samples))
            x_buckets_filled.add(bucket)

            bucket = int((y + y_offset) / (y_bucket_size / samples))
            y_buckets_filled.add(bucket)

            assert_near_equal(x, expected_case['x'], 1e-4)
            assert_near_equal(y, expected_case['y'], 1e-4)

        self.assertEqual(x_buckets_filled, all_buckets)
        self.assertEqual(y_buckets_filled, all_buckets)

    def test_latin_hypercube_array(self):
        samples = 4

        bounds = np.array([
            [-10, -50],  # lower bounds for x and y
            [10,  50]   # upper bounds for x and y
        ])

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('xy', np.array([50., 50.])), promotes=['*'])
        model.add_subsystem('comp', ParaboloidArray(), promotes=['*'])

        model.add_design_var('xy', lower=bounds[0], upper=bounds[1])
        model.add_objective('f_xy')

        prob.driver = om.DOEDriver(om.LatinHypercubeGenerator(samples=4, seed=0))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        # the sample space for each variable should be divided into equal
        # size buckets and each variable should have a value in each bucket
        all_buckets = set(range(samples))

        xlb, xub = bounds[0][0], bounds[1][0]
        x_offset = - xlb
        x_bucket_size = xub - xlb
        x_buckets_filled = set()

        ylb, yub = bounds[0][1], bounds[1][1]
        y_offset = - ylb
        y_bucket_size = yub - ylb
        y_buckets_filled = set()

        # expected values for seed = 0
        expected = [
            {'xy': np.array([-1.98618312, -32.12026584])},
            {'xy': np.array([2.118274,    47.29432502])},
            {'xy': np.array([7.18793606,  16.14735283])},
            {'xy': np.array([-7.25593248, -11.37792043])},
        ]

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), 4)

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            x = outputs['xy'][0]
            y = outputs['xy'][1]

            bucket = int((x + x_offset) / (x_bucket_size / samples))
            x_buckets_filled.add(bucket)

            bucket = int((y + y_offset) / (y_bucket_size / samples))
            y_buckets_filled.add(bucket)

            assert_near_equal(x, expected_case['xy'][0], 1e-4)
            assert_near_equal(y, expected_case['xy'][1], 1e-4)

        self.assertEqual(x_buckets_filled, all_buckets)
        self.assertEqual(y_buckets_filled, all_buckets)

    def test_latin_hypercube_center(self):
        samples = 4
        upper = 10.

        prob = om.Problem()
        model = prob.model

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_output('x', 0.0)
        indep.add_output('y', 0.0)

        model.add_subsystem('comp', Paraboloid())

        model.connect('indep.x', 'comp.x')
        model.connect('indep.y', 'comp.y')

        model.add_design_var('indep.x', lower=0., upper=upper)
        model.add_design_var('indep.y', lower=0., upper=upper)

        model.add_objective('comp.f_xy')

        prob.driver = om.DOEDriver(om.LatinHypercubeGenerator(samples=samples, criterion='c'))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), samples)

        # the sample space for each variable (0 to upper) should be divided into
        # equal size buckets and each variable should have a value in each bucket
        bucket_size = upper / samples
        all_buckets = set(range(samples))

        x_buckets_filled = set()
        y_buckets_filled = set()

        # with criterion of 'center', each value should be in the center of it's bucket
        valid_values = [round(bucket_size * (bucket + 1 / 2), 3) for bucket in all_buckets]

        for case in cases:
            outputs = cr.get_case(case).outputs
            x = float(outputs['indep.x'])
            y = float(outputs['indep.y'])

            x_buckets_filled.add(int(x/bucket_size))
            y_buckets_filled.add(int(y/bucket_size))

            self.assertTrue(round(x, 3) in valid_values, '%f not in %s' % (x, valid_values))
            self.assertTrue(round(y, 3) in valid_values, '%f not in %s' % (y, valid_values))

        self.assertEqual(x_buckets_filled, all_buckets)
        self.assertEqual(y_buckets_filled, all_buckets)

    def test_record_bug(self):
        # There was a bug that caused values to be recorded in driver_scaled form.

        prob = om.Problem()
        model = prob.model

        ivc = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('x', val=1.)

        model.add_subsystem('obj_comp', om.ExecComp('y=2*x'), promotes=['*'])
        model.add_subsystem('con_comp', om.ExecComp('z=3*x'), promotes=['*'])

        prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=3))

        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))
        prob.driver.recording_options['includes'] = ['*']

        model.add_design_var('x', lower=0., upper=10., ref=3.0)
        model.add_constraint('z', lower=2.0, scaler=13.0)
        model.add_objective('y', scaler=-1)

        prob.setup(check=True)

        prob.run_driver()

        cr = om.CaseReader("cases.sql")
        final_case = cr.list_cases('driver', out_stream=None)[-1]
        outputs = cr.get_case(final_case).outputs

        assert_near_equal(outputs['x'], 10.0, 1e-7)
        assert_near_equal(outputs['y'], 20.0, 1e-7)
        assert_near_equal(outputs['z'], 30.0, 1e-7)

    def test_discrete_desvar_list(self):
        prob = om.Problem()
        model = prob.model

        # Add independent variables
        indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_discrete_output('x', 4)
        indeps.add_discrete_output('y', 3)

        # Add components
        model.add_subsystem('parab', ParaboloidDiscrete(), promotes=['*'])

        # Specify design variable range and objective
        model.add_design_var('x')
        model.add_design_var('y')
        model.add_objective('f_xy')

        samples = [[('x', 5), ('y', 1)],
                   [('x', 3), ('y', 6)],
                   [('x', -1), ('y', 3)],
        ]

        # Setup driver for 3 cases at a time
        prob.driver = om.DOEDriver(om.ListGenerator(samples))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        expected = [{'x': 5, 'y': 1, 'f_xy': 31},
                    {'x': 3, 'y': 6, 'f_xy': 115},
                    {'x': -1, 'y': 3, 'f_xy': 59},
        ]
        self.assertEqual(len(cases), len(expected))

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            for name in ('x', 'y', 'f_xy'):
                self.assertEqual(outputs[name], expected_case[name])
                self.assertTrue(isinstance(outputs[name], int))

    def test_discrete_desvar_alltypes(self):
        # Make sure we can handle any allowed type for discrete variables.

        class PassThrough(om.ExplicitComponent):

            def setup(self):
                self.add_discrete_input('x', val='abc')
                self.add_discrete_output('y', val='xyz')

            def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
                discrete_outputs['y'] = discrete_inputs['x']

        prob = om.Problem()
        model = prob.model

        indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_discrete_output('x', 'abc')

        model.add_subsystem('parab', PassThrough(), promotes=['*'])

        model.add_design_var('x')
        model.add_constraint('y')

        my_obj = Paraboloid()
        samples = [[('x', 'abc'), ],
                   [('x', None), ],
                   [('x', my_obj, ), ]
        ]

        prob.driver = om.DOEDriver(om.ListGenerator(samples))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        expected = ['abc', None]

        for case, expected_value in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            self.assertEqual(outputs['x'], expected_value)

        # Can't read/write objects through SQL case.
        self.assertEqual(prob['y'], my_obj)

    def test_discrete_array_output(self):
        prob = om.Problem()
        model = prob.model

        # Add independent variables
        indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_discrete_output('x', np.ones((2, ), dtype=np.int))
        indeps.add_discrete_output('y', np.ones((2, ), dtype=np.int))

        # Add components
        model.add_subsystem('parab', ParaboloidDiscreteArray(), promotes=['*'])

        # Specify design variable range and objective
        model.add_design_var('x', np.array([5, 1]))
        model.add_design_var('y', np.array([1, 4]))
        model.add_objective('f_xy')

        recorder = om.SqliteRecorder("cases.sql")
        prob.driver.add_recorder(recorder)
        prob.add_recorder(recorder)
        prob.recording_options['record_inputs'] = True

        prob.setup()
        prob.run_driver()
        prob.record("end")
        prob.cleanup()

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('problem', out_stream=None)

        case = cr.get_case('end')
        inputs = case.inputs
        outputs = case.outputs
        for name in ('x', 'y'):
            self.assertTrue(isinstance(inputs[name], np.ndarray))
            self.assertTrue(inputs[name].shape, (2,))
            self.assertTrue(isinstance(outputs[name], np.ndarray))
            self.assertTrue(outputs[name].shape, (2,))

    def test_discrete_arraydesvar_list(self):
        prob = om.Problem()
        model = prob.model

        # Add components
        model.add_subsystem('parab', ParaboloidDiscreteArray(), promotes=['*'])

        # Specify design variable range and objective
        model.add_design_var('x')
        model.add_design_var('y')
        model.add_objective('f_xy')

        samples = [[('x', np.array([5, 1])), ('y', np.array([1, 4]))],
                   [('x', np.array([3, 2])), ('y', np.array([6, -3]))],
                   [('x', np.array([-1, 0])), ('y', np.array([3, 5]))],
        ]

        # Setup driver for 3 cases at a time
        prob.driver = om.DOEDriver(om.ListGenerator(samples))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()

        prob.set_val('x', np.ones((2, ), dtype=np.int))
        prob.set_val('y', np.ones((2, ), dtype=np.int))

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        expected = [{'x': np.array([5, 1]), 'y': np.array([1, 4]), 'f_xy': np.array([31, 69])},
                    {'x': np.array([3, 2]), 'y': np.array([6, -3]), 'f_xy': np.array([115, -7])},
                    {'x': np.array([-1, 0]), 'y': np.array([3, 5]), 'f_xy': np.array([59, 87])},
        ]
        self.assertEqual(len(cases), len(expected))

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            for name in ('x', 'y', 'f_xy'):
                self.assertEqual(outputs[name][0], expected_case[name][0])
                self.assertEqual(outputs[name][1], expected_case[name][1])

    def test_discrete_desvar_csv(self):
        prob = om.Problem()
        model = prob.model

        # Add independent variables
        indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_discrete_output('x', 4)
        indeps.add_discrete_output('y', 3)

        # Add components
        model.add_subsystem('parab', ParaboloidDiscrete(), promotes=['*'])

        # Specify design variable range and objective
        model.add_design_var('x')
        model.add_design_var('y')
        model.add_objective('f_xy')

        samples = '\n'.join([" x ,   y",
                             "5,  1",
                             "3,  6",
                             "-1,  3",
                             ])

        # this file contains design variable inputs in CSV format
        with open('cases.csv', 'w') as f:
            f.write(samples)

        # Setup driver for 3 cases at a time
        prob.driver = om.DOEDriver(om.CSVGenerator('cases.csv'))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        expected = [{'x': 5, 'y': 1, 'f_xy': 31},
                    {'x': 3, 'y': 6, 'f_xy': 115},
                    {'x': -1, 'y': 3, 'f_xy': 59},
        ]
        self.assertEqual(len(cases), len(expected))

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            for name in ('x', 'y', 'f_xy'):
                self.assertEqual(outputs[name], expected_case[name])
                self.assertTrue(isinstance(outputs[name], int))


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
@use_tempdirs
class TestParallelDOE(unittest.TestCase):

    N_PROCS = 4

    def setUp(self):
        self.expected_fullfact3 = [
            {'x': np.array([0.]), 'y': np.array([0.]), 'f_xy': np.array([22.00])},
            {'x': np.array([.5]), 'y': np.array([0.]), 'f_xy': np.array([19.25])},
            {'x': np.array([1.]), 'y': np.array([0.]), 'f_xy': np.array([17.00])},

            {'x': np.array([0.]), 'y': np.array([.5]), 'f_xy': np.array([26.25])},
            {'x': np.array([.5]), 'y': np.array([.5]), 'f_xy': np.array([23.75])},
            {'x': np.array([1.]), 'y': np.array([.5]), 'f_xy': np.array([21.75])},

            {'x': np.array([0.]), 'y': np.array([1.]), 'f_xy': np.array([31.00])},
            {'x': np.array([.5]), 'y': np.array([1.]), 'f_xy': np.array([28.75])},
            {'x': np.array([1.]), 'y': np.array([1.]), 'f_xy': np.array([27.00])},
        ]

    def test_indivisible_error(self):
        prob = om.Problem()

        prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=3))
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = 3

        with self.assertRaises(RuntimeError) as context:
            prob.setup()

        self.assertEqual(str(context.exception),
                         "The total number of processors is not evenly divisible by the "
                         "specified number of processors per model.\n Provide a number of "
                         "processors that is a multiple of 3, or specify a number "
                         "of processors per model that divides into 4.")

    def test_minprocs_error(self):
        prob = om.Problem(FanInGrouped())

        # require 2 procs for the ParallelGroup
        prob.model._proc_info['sub'] = (2, None, 1.0)

        # run cases on all procs
        prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=3))
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = 1

        with self.assertRaises(RuntimeError) as context:
            prob.setup()

        self.assertEqual(str(context.exception),
                         "<model> <class FanInGrouped>: MPI process allocation failed: can't meet "
                         "min_procs required for the following subsystems: ['sub']")

    def test_full_factorial(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=3), procs_per_model=1,
                                   run_parallel=True)
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()

        failed, output = run_driver(prob)
        self.assertFalse(failed)

        prob.cleanup()

        expected = self.expected_fullfact3

        size = prob.comm.size
        rank = prob.comm.rank

        # cases will be split across files for each proc
        filename = "cases.sql_%d" % rank

        expect_msg = "Cases from rank %d are being written to %s." % (rank, filename)
        self.assertTrue(expect_msg in output)

        cr = om.CaseReader(filename)
        cases = cr.list_cases('driver', out_stream=None)

        # cases recorded on this proc
        num_cases = len(cases)
        self.assertEqual(num_cases, len(expected) // size + (rank < len(expected) % size))

        for n in range(num_cases):
            outputs = cr.get_case(cases[n]).outputs
            idx = n * size + rank  # index of expected case

            self.assertEqual(outputs['x'], expected[idx]['x'])
            self.assertEqual(outputs['y'], expected[idx]['y'])
            self.assertEqual(outputs['f_xy'], expected[idx]['f_xy'])

        # total number of cases recorded across all procs
        num_cases = prob.comm.allgather(num_cases)
        self.assertEqual(sum(num_cases), len(expected))

    def test_fan_in_grouped_parallel_2x2(self):
        # run cases in parallel with 2 procs per model
        # (cases will be split between the 2 parallel model instances)
        run_parallel = True
        procs_per_model = 2

        prob = om.Problem(FanInGrouped())
        model = prob.model

        model.add_design_var('x1', lower=0.0, upper=1.0)
        model.add_design_var('x2', lower=0.0, upper=1.0)

        model.add_objective('c3.y')

        prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=3))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.driver.options['run_parallel'] = run_parallel
        prob.driver.options['procs_per_model'] = procs_per_model

        prob.setup()

        failed, output = run_driver(prob)

        from openmdao.utils.mpi import multi_proc_exception_check

        with multi_proc_exception_check(prob.comm):
            self.assertFalse(failed)

            prob.cleanup()

            expected = [
                {'x1': np.array([0.]), 'x2': np.array([0.]), 'c3.y': np.array([0.0])},
                {'x1': np.array([.5]), 'x2': np.array([0.]), 'c3.y': np.array([-3.0])},
                {'x1': np.array([1.]), 'x2': np.array([0.]), 'c3.y': np.array([-6.0])},

                {'x1': np.array([0.]), 'x2': np.array([.5]), 'c3.y': np.array([17.5])},
                {'x1': np.array([.5]), 'x2': np.array([.5]), 'c3.y': np.array([14.5])},
                {'x1': np.array([1.]), 'x2': np.array([.5]), 'c3.y': np.array([11.5])},

                {'x1': np.array([0.]), 'x2': np.array([1.]), 'c3.y': np.array([35.0])},
                {'x1': np.array([.5]), 'x2': np.array([1.]), 'c3.y': np.array([32.0])},
                {'x1': np.array([1.]), 'x2': np.array([1.]), 'c3.y': np.array([29.0])},
            ]

            num_cases = 0

            # we can run two models in parallel on our 4 procs
            num_models = prob.comm.size // procs_per_model

            # a separate case file will be written by rank 0 of each parallel model
            # (the top two global ranks)
            rank = prob.comm.rank

            if rank < num_models:
                filename = "cases.sql_%d" % rank

                expect_msg = "Cases from rank %d are being written to %s." % (rank, filename)
                self.assertTrue(expect_msg in output)

                cr = om.CaseReader(filename)
                cases = cr.list_cases('driver')

                # cases recorded on this proc
                num_cases = len(cases)
                self.assertEqual(num_cases, len(expected) // num_models+(rank < len(expected) % num_models))

                for n, case in enumerate(cases):
                    idx = n * num_models + rank  # index of expected case

                    outputs = cr.get_case(case).outputs

                    for name in ('x1', 'x2', 'c3.y'):
                        self.assertEqual(outputs[name], expected[idx][name])
            else:
                self.assertFalse("Cases from rank %d are being written" % rank in output)

        # total number of cases recorded across all requested procs
        num_cases = prob.comm.allgather(num_cases)
        self.assertEqual(sum(num_cases), len(expected))

    def test_fan_in_grouped_parallel_4x1(self):
        # run cases in parallel with 1 proc per model
        # (cases will be split between the 4 serial model instances)
        run_parallel = True
        procs_per_model = 1

        prob = om.Problem(FanInGrouped())
        model = prob.model

        model.add_design_var('x1', lower=0.0, upper=1.0)
        model.add_design_var('x2', lower=0.0, upper=1.0)

        model.add_objective('c3.y')

        prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=3))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.driver.options['run_parallel'] = run_parallel
        prob.driver.options['procs_per_model'] = procs_per_model

        prob.setup()

        failed, output = run_driver(prob)
        self.assertFalse(failed)

        prob.cleanup()

        expected = [
            {'x1': np.array([0.]), 'x2': np.array([0.]), 'c3.y': np.array([0.0])},
            {'x1': np.array([.5]), 'x2': np.array([0.]), 'c3.y': np.array([-3.0])},
            {'x1': np.array([1.]), 'x2': np.array([0.]), 'c3.y': np.array([-6.0])},

            {'x1': np.array([0.]), 'x2': np.array([.5]), 'c3.y': np.array([17.5])},
            {'x1': np.array([.5]), 'x2': np.array([.5]), 'c3.y': np.array([14.5])},
            {'x1': np.array([1.]), 'x2': np.array([.5]), 'c3.y': np.array([11.5])},

            {'x1': np.array([0.]), 'x2': np.array([1.]), 'c3.y': np.array([35.0])},
            {'x1': np.array([.5]), 'x2': np.array([1.]), 'c3.y': np.array([32.0])},
            {'x1': np.array([1.]), 'x2': np.array([1.]), 'c3.y': np.array([29.0])},
        ]

        rank = prob.comm.rank

        # there will be a separate case file for each proc, containing the cases
        # run by the instance of the model that runs in serial mode on that proc
        filename = "cases.sql_%d" % rank

        expect_msg = "Cases from rank %d are being written to %s." % (rank, filename)
        self.assertTrue(expect_msg in output)

        # we are running 4 models in parallel, each using 1 proc
        num_models = prob.comm.size // procs_per_model

        cr = om.CaseReader(filename)
        cases = cr.list_cases('driver', out_stream=None)

        # cases recorded on this proc
        num_cases = len(cases)
        self.assertEqual(num_cases, len(expected) // num_models + (rank < len(expected) % num_models))

        for n, case in enumerate(cases):
            idx = n * num_models + rank  # index of expected case

            outputs = cr.get_case(case).outputs

            self.assertEqual(outputs['x1'], expected[idx]['x1'])
            self.assertEqual(outputs['x2'], expected[idx]['x2'])
            self.assertEqual(outputs['c3.y'], expected[idx]['c3.y'])

        # total number of cases recorded across all requested procs
        num_cases = prob.comm.allgather(num_cases)
        self.assertEqual(sum(num_cases), len(expected))

    def test_fan_in_grouped_serial_2x2(self):
        # do not run cases in parallel, but with 2 procs per model
        # (all cases will run on each of the 2 parallel model instances)
        run_parallel = False
        procs_per_model = 2

        prob = om.Problem(FanInGrouped())
        model = prob.model

        model.add_design_var('x1', lower=0.0, upper=1.0)
        model.add_design_var('x2', lower=0.0, upper=1.0)
        model.add_objective('c3.y')

        prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=3))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.driver.options['run_parallel'] = run_parallel
        prob.driver.options['procs_per_model'] = procs_per_model

        prob.setup()

        failed, output = run_driver(prob)
        self.assertFalse(failed)

        prob.cleanup()

        expected = [
            {'x1': np.array([0.]), 'x2': np.array([0.]), 'c3.y': np.array([0.0])},
            {'x1': np.array([.5]), 'x2': np.array([0.]), 'c3.y': np.array([-3.0])},
            {'x1': np.array([1.]), 'x2': np.array([0.]), 'c3.y': np.array([-6.0])},

            {'x1': np.array([0.]), 'x2': np.array([.5]), 'c3.y': np.array([17.5])},
            {'x1': np.array([.5]), 'x2': np.array([.5]), 'c3.y': np.array([14.5])},
            {'x1': np.array([1.]), 'x2': np.array([.5]), 'c3.y': np.array([11.5])},

            {'x1': np.array([0.]), 'x2': np.array([1.]), 'c3.y': np.array([35.0])},
            {'x1': np.array([.5]), 'x2': np.array([1.]), 'c3.y': np.array([32.0])},
            {'x1': np.array([1.]), 'x2': np.array([1.]), 'c3.y': np.array([29.0])},
        ]

        num_cases = 0

        rank = prob.comm.rank

        # we are running the model on two sets of two procs
        num_models = prob.comm.size // procs_per_model

        if rank < num_models:
            # a separate case file will be written by rank 0 of each parallel model
            # (the top two global ranks)
            filename = "cases.sql_%d" % rank

            expect_msg = "Cases from rank %d are being written to %s." % (rank, filename)
            self.assertTrue(expect_msg in output)

            cr = om.CaseReader(filename)
            cases = cr.list_cases('driver', out_stream=None)

            # cases recorded on this proc... each proc will run all cases
            num_cases = len(cases)
            self.assertEqual(num_cases, len(expected))

            for idx, case in enumerate(cases):
                outputs = cr.get_case(case).outputs

                self.assertEqual(outputs['x1'], expected[idx]['x1'])
                self.assertEqual(outputs['x2'], expected[idx]['x2'])
                self.assertEqual(outputs['c3.y'], expected[idx]['c3.y'])

        # total number of cases recorded will be twice the number of cases
        # (every case will be recorded on all pairs of procs)
        num_cases = prob.comm.allgather(num_cases)
        self.assertEqual(sum(num_cases), num_models*len(expected))

    def test_fan_in_grouped_serial_4x1(self):
        # do not run cases in parallel, with 1 proc per model
        # (all cases will run on each of the 4 serial model instances)
        run_parallel = False
        procs_per_model = 1

        prob = om.Problem(FanInGrouped())
        model = prob.model

        model.add_design_var('x1', lower=0.0, upper=1.0)
        model.add_design_var('x2', lower=0.0, upper=1.0)
        model.add_objective('c3.y')

        prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=3))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.driver.options['run_parallel'] = run_parallel
        prob.driver.options['procs_per_model'] = procs_per_model

        prob.setup()

        failed, output = run_driver(prob)
        self.assertFalse(failed)

        prob.cleanup()

        expected = [
            {'x1': np.array([0.]), 'x2': np.array([0.]), 'c3.y': np.array([0.0])},
            {'x1': np.array([.5]), 'x2': np.array([0.]), 'c3.y': np.array([-3.0])},
            {'x1': np.array([1.]), 'x2': np.array([0.]), 'c3.y': np.array([-6.0])},

            {'x1': np.array([0.]), 'x2': np.array([.5]), 'c3.y': np.array([17.5])},
            {'x1': np.array([.5]), 'x2': np.array([.5]), 'c3.y': np.array([14.5])},
            {'x1': np.array([1.]), 'x2': np.array([.5]), 'c3.y': np.array([11.5])},

            {'x1': np.array([0.]), 'x2': np.array([1.]), 'c3.y': np.array([35.0])},
            {'x1': np.array([.5]), 'x2': np.array([1.]), 'c3.y': np.array([32.0])},
            {'x1': np.array([1.]), 'x2': np.array([1.]), 'c3.y': np.array([29.0])},
        ]

        rank = prob.comm.rank

        # we are running the model on all four procs
        num_models = prob.comm.size // procs_per_model

        # there will be a separate case file for each proc, containing the cases
        # run by the instance of the model that runs in serial mode on that proc
        filename = "cases.sql_%d" % rank

        expect_msg = "Cases from rank %d are being written to %s." % (rank, filename)
        self.assertTrue(expect_msg in output)

        # we are running 4 models in parallel, each using 1 proc
        num_models = prob.comm.size // procs_per_model

        cr = om.CaseReader(filename)
        cases = cr.list_cases('driver', out_stream=None)

        # cases recorded on this proc
        num_cases = len(cases)
        self.assertEqual(num_cases, len(expected))

        for idx, case in enumerate(cases):
            outputs = cr.get_case(case).outputs

            self.assertEqual(outputs['x1'], expected[idx]['x1'])
            self.assertEqual(outputs['x2'], expected[idx]['x2'])
            self.assertEqual(outputs['c3.y'], expected[idx]['c3.y'])

        # total number of cases recorded will be 4x the number of cases
        # (every case will be recorded on all procs)
        num_cases = prob.comm.allgather(num_cases)
        self.assertEqual(sum(num_cases), num_models*len(expected))


@use_tempdirs
class TestDOEDriverFeature(unittest.TestCase):

    def setUp(self):
        import json
        import numpy as np

        self.expected_csv = '\n'.join([
            " x ,   y",
            "0.0,  0.0",
            "0.5,  0.0",
            "1.0,  0.0",
            "0.0,  0.5",
            "0.5,  0.5",
            "1.0,  0.5",
            "0.0,  1.0",
            "0.5,  1.0",
            "1.0,  1.0",
        ])

        with open('cases.csv', 'w') as f:
            f.write(self.expected_csv)

        expected = [
            {'x': np.array([0.]), 'y': np.array([0.]), 'f_xy': np.array([22.00])},
            {'x': np.array([.5]), 'y': np.array([0.]), 'f_xy': np.array([19.25])},
            {'x': np.array([1.]), 'y': np.array([0.]), 'f_xy': np.array([17.00])},

            {'x': np.array([0.]), 'y': np.array([.5]), 'f_xy': np.array([26.25])},
            {'x': np.array([.5]), 'y': np.array([.5]), 'f_xy': np.array([23.75])},
            {'x': np.array([1.]), 'y': np.array([.5]), 'f_xy': np.array([21.75])},

            {'x': np.array([0.]), 'y': np.array([1.]), 'f_xy': np.array([31.00])},
            {'x': np.array([.5]), 'y': np.array([1.]), 'f_xy': np.array([28.75])},
            {'x': np.array([1.]), 'y': np.array([1.]), 'f_xy': np.array([27.00])},
        ]

        values = []
        cases = []

        for case in expected:
            values.append((case['x'], case['y'], case['f_xy']))
            # converting ndarray to list enables JSON serialization
            cases.append((('x', list(case['x'])), ('y', list(case['y']))))

        self.expected_text = "\n".join([
            "x: %5.2f, y: %5.2f, f_xy: %6.2f" % vals_i for vals_i in values
        ])

        self.expected_json = json.dumps(cases).replace(']]],', ']]],\n')
        with open('cases.json', 'w') as f:
            f.write(self.expected_json)

    def test_uniform(self):
        import openmdao.api as om
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        model.add_design_var('x', lower=-10, upper=10)
        model.add_design_var('y', lower=-10, upper=10)
        model.add_objective('f_xy')

        prob.driver = om.DOEDriver(om.UniformGenerator(num_samples=5))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()

        prob.set_val('x', 0.0)
        prob.set_val('y', 0.0)

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver')

        self.assertEqual(len(cases), 5)

        values = []
        for case in cases:
            outputs = cr.get_case(case).outputs
            values.append((outputs['x'], outputs['y'], outputs['f_xy']))

        print("\n".join(["x: %5.2f, y: %5.2f, f_xy: %6.2f" % xyf for xyf in values]))

    def test_csv(self):
        import openmdao.api as om
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.setup()

        prob.set_val('x', 0.0)
        prob.set_val('y', 0.0)

        # this file contains design variable inputs in CSV format
        with open('cases.csv', 'r') as f:
            self.assertEqual(f.read(), self.expected_csv)

        # run problem with DOEDriver using the CSV file
        prob.driver = om.DOEDriver(om.CSVGenerator('cases.csv'))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver')

        values = []
        for case in cases:
            outputs = cr.get_case(case).outputs
            values.append((outputs['x'], outputs['y'], outputs['f_xy']))

        self.assertEqual("\n".join(["x: %5.2f, y: %5.2f, f_xy: %6.2f" % xyf for xyf in values]),
                         self.expected_text)

    def test_list(self):
        import openmdao.api as om
        from openmdao.test_suite.components.paraboloid import Paraboloid

        import json

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.setup()

        prob.set_val('x', 0.0)
        prob.set_val('y', 0.0)

        # load design variable inputs from JSON file and decode into list
        with open('cases.json', 'r') as f:
            json_data = f.read()

        self.assertEqual(json_data, self.expected_json)

        case_list = json.loads(json_data)

        self.assertEqual(case_list, json.loads(json_data))

        # create DOEDriver using provided list of cases
        prob.driver = om.DOEDriver(case_list)

        # a ListGenerator was created
        self.assertEqual(type(prob.driver.options['generator']), om.ListGenerator)

        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")
        cases = cr.list_cases('driver')

        values = []
        for case in cases:
            outputs = cr.get_case(case).outputs
            values.append((outputs['x'], outputs['y'], outputs['f_xy']))

        self.assertEqual("\n".join(["x: %5.2f, y: %5.2f, f_xy: %6.2f" % xyf for xyf in values]),
                         self.expected_text)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
@use_tempdirs
class TestParallelDOEFeature(unittest.TestCase):

    N_PROCS = 2

    def setUp(self):
        import numpy as np

        from mpi4py import MPI
        rank = MPI.COMM_WORLD.rank

        expected = [
            {'x': np.array([0.]), 'y': np.array([0.]), 'f_xy': np.array([22.00])},
            {'x': np.array([.5]), 'y': np.array([0.]), 'f_xy': np.array([19.25])},
            {'x': np.array([1.]), 'y': np.array([0.]), 'f_xy': np.array([17.00])},

            {'x': np.array([0.]), 'y': np.array([.5]), 'f_xy': np.array([26.25])},
            {'x': np.array([.5]), 'y': np.array([.5]), 'f_xy': np.array([23.75])},
            {'x': np.array([1.]), 'y': np.array([.5]), 'f_xy': np.array([21.75])},

            {'x': np.array([0.]), 'y': np.array([1.]), 'f_xy': np.array([31.00])},
            {'x': np.array([.5]), 'y': np.array([1.]), 'f_xy': np.array([28.75])},
            {'x': np.array([1.]), 'y': np.array([1.]), 'f_xy': np.array([27.00])},
        ]

        # expect odd cases on rank 0 and even cases on rank 1
        values = []
        for idx, case in enumerate(expected):
            if idx % 2 == rank:
                values.append((case['x'], case['y'], case['f_xy']))

        self.expect_text = "\n"+"\n".join([
            "x: %5.2f, y: %5.2f, f_xy: %6.2f" % xyf for xyf in values
        ])

    def test_full_factorial(self):
        import openmdao.api as om
        from openmdao.test_suite.components.paraboloid import Paraboloid

        from mpi4py import MPI

        prob = om.Problem()

        prob.model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])
        prob.model.add_design_var('x', lower=0.0, upper=1.0)
        prob.model.add_design_var('y', lower=0.0, upper=1.0)
        prob.model.add_objective('f_xy')

        prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=3))
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = 1

        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        self.assertEqual(MPI.COMM_WORLD.size, 2)

        # check recorded cases from each case file
        rank = MPI.COMM_WORLD.rank
        filename = "cases.sql_%d" % rank
        self.assertEqual(filename, "cases.sql_%d" % rank)

        cr = om.CaseReader(filename)
        cases = cr.list_cases('driver')
        self.assertEqual(len(cases), 5 if rank == 0 else 4)

        values = []
        for case in cases:
            outputs = cr.get_case(case).outputs
            values.append((outputs['x'], outputs['y'], outputs['f_xy']))

        self.assertEqual("\n"+"\n".join(["x: %5.2f, y: %5.2f, f_xy: %6.2f" % xyf for xyf in values]),
                         self.expect_text)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
@use_tempdirs
class TestParallelDOEFeature2(unittest.TestCase):

    N_PROCS = 4

    def setUp(self):
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.rank

        expected = [
            {'x1': np.array([0.]), 'x2': np.array([0.]), 'c3.y': np.array([0.00])},
            {'x1': np.array([.5]), 'x2': np.array([0.]), 'c3.y': np.array([-3.00])},
            {'x1': np.array([1.]), 'x2': np.array([0.]), 'c3.y': np.array([-6.00])},

            {'x1': np.array([0.]), 'x2': np.array([.5]), 'c3.y': np.array([17.50])},
            {'x1': np.array([.5]), 'x2': np.array([.5]), 'c3.y': np.array([14.50])},
            {'x1': np.array([1.]), 'x2': np.array([.5]), 'c3.y': np.array([11.50])},

            {'x1': np.array([0.]), 'x2': np.array([1.]), 'c3.y': np.array([35.00])},
            {'x1': np.array([.5]), 'x2': np.array([1.]), 'c3.y': np.array([32.00])},
            {'x1': np.array([1.]), 'x2': np.array([1.]), 'c3.y': np.array([29.00])},
        ]

        # expect odd cases on rank 0 and even cases on rank 1
        values = []
        for idx, case in enumerate(expected):
            if idx % 2 == rank:
                values.append((case['x1'], case['x2'], case['c3.y']))

        self.expect_text = "\n"+"\n".join([
            "x1: %5.2f, x2: %5.2f, c3.y: %6.2f" % vals_i for vals_i in values
        ])

    def test_fan_in_grouped(self):
        import openmdao.api as om
        from openmdao.test_suite.groups.parallel_groups import FanInGrouped

        from mpi4py import MPI

        prob = om.Problem(FanInGrouped())

        prob.model.add_design_var('x1', lower=0.0, upper=1.0)
        prob.model.add_design_var('x2', lower=0.0, upper=1.0)
        prob.model.add_objective('c3.y')

        prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=3))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        # the FanInGrouped model uses 2 processes, so we can run
        # two instances of the model at a time, each using 2 of our 4 procs
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = procs_per_model = 2

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        # a separate case file will be written by rank 0 of each parallel model
        # (the top two global ranks)
        rank = prob.comm.rank

        num_models = prob.comm.size // procs_per_model

        if rank < num_models:
            filename = "cases.sql_%d" % rank

            cr = om.CaseReader(filename)
            cases = cr.list_cases('driver')

            values = []
            for case in cases:
                outputs = cr.get_case(case).outputs
                values.append((outputs['x1'], outputs['x2'], outputs['c3.y']))

            self.assertEqual("\n"+"\n".join(["x1: %5.2f, x2: %5.2f, c3.y: %6.2f" % (x1, x2, y) for x1, x2, y in values]),
                self.expect_text)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
@use_tempdirs
class TestParallelDistribDOE(unittest.TestCase):

    N_PROCS = 4

    def test_doe_distributed_var(self):
        size = 3

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
                            promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_objective('f_sum', index=-1)

        prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=2))
        prob.driver.options['run_parallel'] = True
        prob.driver.options['procs_per_model'] = 2

        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        # check recorded cases from each case file
        rank = prob.comm.rank
        if rank == 0:
            filename0 = "cases.sql_0"
            values = []

            cr = om.CaseReader(filename0)
            cases = cr.list_cases('driver')
            for case in cases:
                outputs = cr.get_case(case).outputs
                values.append(outputs)

            # 2**6 cases, half on each rank
            self.assertEqual(len(values), 32)
            x_inputs = [list(val['x']) for val in values]
            for n1 in [-50.]:
                for n2 in [-50., 50.]:
                    for n3 in [-50., 50.]:
                        self.assertEqual(x_inputs.count([n1, n2, n3]), 8)

        elif rank == 1:
            filename0 = "cases.sql_1"
            values = []

            cr = om.CaseReader(filename0)
            cases = cr.list_cases('driver')
            for case in cases:
                outputs = cr.get_case(case).outputs
                values.append(outputs)

            # 2**6 cases, half on each rank
            self.assertEqual(len(values), 32)
            x_inputs = [list(val['x']) for val in values]
            for n1 in [50.]:
                for n2 in [-50., 50.]:
                    for n3 in [-50., 50.]:
                        self.assertEqual(x_inputs.count([n1, n2, n3]), 8)


if __name__ == "__main__":
    unittest.main()
