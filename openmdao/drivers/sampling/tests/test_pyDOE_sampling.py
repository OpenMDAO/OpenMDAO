"""
Test Sampling Generators.
"""
import unittest

import numpy as np

from packaging.version import Version

import openmdao.api as om

from openmdao.test_suite.components.paraboloid import Paraboloid

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from openmdao.drivers.sampling.pyDOE_generators import FullFactorialGenerator, \
    GeneralizedSubsetGenerator, PlackettBurmanGenerator, \
    BoxBehnkenGenerator, LatinHypercubeGenerator


try:
    import pyDOE3
except ImportError:
    pyDOE3 = None


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


class TestPyDOEErrors(unittest.TestCase):

    @unittest.skipIf(pyDOE3, "only runs if 'pyDOE3' is not installed")
    def test_no_pyDOE3(self):
        with self.assertRaises(RuntimeError) as err:
            FullFactorialGenerator(var_dict={}, levels=3)

        self.assertEqual(str(err.exception),
                         "FullFactorialGenerator requires the 'pyDOE3' package, "
                         "which can be installed with one of the following commands:\n"
                         "    pip install openmdao[doe]\n"
                         "    pip install pyDOE3")

        with self.assertRaises(RuntimeError) as err:
            om.AnalysisDriver(samples=FullFactorialGenerator(var_dict={}, levels=3))

        self.assertEqual(str(err.exception),
                         "FullFactorialGenerator requires the 'pyDOE3' package, "
                         "which can be installed with one of the following commands:\n"
                         "    pip install openmdao[doe]\n"
                         "    pip install pyDOE3")

    @unittest.skipUnless(pyDOE3, "requires 'pyDOE3', pip install openmdao[doe]")
    def test_lhc_criterion(self):
        with self.assertRaises(ValueError) as err:
            LatinHypercubeGenerator(var_dict={}, criterion='foo')

        self.assertEqual(str(err.exception),
                         "Invalid criterion 'foo' specified for LatinHypercubeGenerator. "
                         "Must be one of ['center', 'c', 'maximin', 'm', 'centermaximin', "
                         "'cm', 'correlation', 'corr', None].")

    @unittest.skipUnless(pyDOE3, "requires 'pyDOE3', pip install openmdao[doe]")
    def test_missing_bounds(self):
        with self.assertRaises(RuntimeError) as err:
            factors = {
                'x': {'upper': 1.0},
            }
            FullFactorialGenerator(factors, levels=3)

        self.assertEqual(str(err.exception),
                         "Unable to determine levels for factor 'x'. Factors dictionary must "
                         "contain both 'lower' and 'upper' keys.")

    @unittest.skipUnless(pyDOE3, "requires 'pyDOE3', pip install openmdao[doe]")
    def test_mismatched_bounds(self):
        with self.assertRaises(ValueError) as err:
            factors = {
                'x': {'lower': 0.0, 'upper': np.array([1.0, 2.0, 3.0])},
            }
            FullFactorialGenerator(factors, levels=3)

        self.assertEqual(str(err.exception),
                         "Size mismatch for factor 'x': "
                         "'lower' bound size (1) does not match 'upper' bound size (3).")


@use_tempdirs
@unittest.skipUnless(pyDOE3, "requires 'pyDOE3', pip install openmdao[doe]")
class TestPyDOEGenerators(unittest.TestCase):

    def setUp(self):
        self.NOfullfact3 = [
            [('x', 0.), ('y', 0.)],
            [('x', .5), ('y', 0.)],
            [('x', 1.), ('y', 0.)],

            [('x', 0.), ('y', .5)],
            [('x', .5), ('y', .5)],
            [('x', 1.), ('y', .5)],

            [('x', 0.), ('y', 1.)],
            [('x', .5), ('y', 1.)],
            [('x', 1.), ('y', 1.)],
        ]

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

    def test_full_factorial(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])
        model.set_input_defaults('x', 0.0)
        model.set_input_defaults('y', 0.0)

        factors = {
            'x': {'lower': 0.0, 'upper': 1.0},
            'y': {'lower': 0.0, 'upper': 1.0}
        }

        prob.driver = om.AnalysisDriver(samples=FullFactorialGenerator(factors, levels=3))
        prob.driver.add_response('f_xy')
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        expected = self.expected_fullfact3

        cr = om.CaseReader(prob.get_outputs_dir() / "cases.sql")
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

        model.set_input_defaults('x', np.array([0.0, 0.0]))
        model.set_input_defaults('y', np.array([0.0, 0.0]))
        model.add_subsystem('comp', Digits2Num(), promotes=['*'])

        factors = {
            'x': {'lower': np.array([0., 0.]), 'upper': np.array([1.0, 2.0])},
            'y': {'lower': np.array([0., 0.]), 'upper': np.array([3.0, 4.0])}
        }

        prob.driver = om.AnalysisDriver(samples=FullFactorialGenerator(factors, levels=2))
        prob.driver.add_response('f')
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(prob.get_outputs_dir() / "cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        objs = [cr.get_case(case).outputs['f'].item() for case in cases]

        self.assertEqual(len(objs), 16)
        # Testing uniqueness. If all elements are unique, it should be the same length as the
        # number of cases
        self.assertEqual(len(set(objs)), 16)

    def test_full_factorial_array(self):
        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('xy', np.array([0., 0.]))
        model.add_subsystem('comp', ParaboloidArray(), promotes=['*'])

        model.add_design_var('xy', lower=np.array([-10., -50.]), upper=np.array([10., 50.]))

        factors = {
            'xy': {'lower': np.array([-10., -50.]), 'upper': np.array([10., 50.])},
        }

        prob.driver = om.AnalysisDriver(FullFactorialGenerator(factors, levels=3))
        prob.driver.add_response('f_xy')
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

        cr = om.CaseReader(prob.get_outputs_dir() / "cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), 9)

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            self.assertEqual(outputs['xy'][0], expected_case['xy'][0])
            self.assertEqual(outputs['xy'][1], expected_case['xy'][1])

    def test_full_fact_dict_levels(self):
        # Specifying levels only for one factor, the other is defaulted
        prob = om.Problem()
        model = prob.model

        expected = [
            {'x': np.array([0.]), 'y': np.array([0.]), 'f_xy': np.array([22.00])},
            {'x': np.array([1.]), 'y': np.array([0.]), 'f_xy': np.array([17.00])},

            {'x': np.array([0.]), 'y': np.array([.5]), 'f_xy': np.array([26.25])},
            {'x': np.array([1.]), 'y': np.array([.5]), 'f_xy': np.array([21.75])},

            {'x': np.array([0.]), 'y': np.array([1.]), 'f_xy': np.array([31.00])},
            {'x': np.array([1.]), 'y': np.array([1.]), 'f_xy': np.array([27.00])},
        ]

        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])
        model.set_input_defaults('x', 0.0)
        model.set_input_defaults('y', 0.0)

        factors = {
            'x': {'lower': 0.0, 'upper': 1.0},
            'y': {'lower': 0.0, 'upper': 1.0}
        }

        prob.driver = om.AnalysisDriver(samples=FullFactorialGenerator(factors, levels={"y": 3}))
        prob.driver.add_response('f_xy')
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(prob.get_outputs_dir() / "cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), 6)

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            self.assertEqual(outputs['x'], expected_case['x'])
            self.assertEqual(outputs['y'], expected_case['y'])
            self.assertEqual(outputs['f_xy'], expected_case['f_xy'])

    def test_generalized_subset(self):
        # All factors have the same number of levels
        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', 0.0)
        model.set_input_defaults('y', 0.0)
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        factors = {
            'x': {'lower': 0.0, 'upper': 1.0},
            'y': {'lower': 0.0, 'upper': 1.0}
        }

        prob.driver = om.AnalysisDriver(GeneralizedSubsetGenerator(factors, levels=2, reduction=2))
        prob.driver.add_response('f_xy')
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        expected = [
            {'x': np.array([0.0]), 'y': np.array([0.0]), 'f_xy': np.array([22.0])},
            {'x': np.array([1.0]), 'y': np.array([1.0]), 'f_xy': np.array([27.0])},
        ]

        cr = om.CaseReader(prob.get_outputs_dir() / "cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), 2)

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            for name in ('x', 'y', 'f_xy'):
                self.assertEqual(outputs[name], expected_case[name])

    def test_generalized_subset_dict_levels(self):
        # Number of levels specified individually for all factors (scalars).
        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', 0.0)
        model.set_input_defaults('y', 0.0)
        model.add_subsystem('comp', Paraboloid(default_shape=()), promotes=['x', 'y', 'f_xy'])

        factors = {
            'x': {'lower': 0.0, 'upper': 1.0},
            'y': {'lower': 0.0, 'upper': 1.0}
        }

        prob.driver = om.AnalysisDriver(GeneralizedSubsetGenerator(factors, levels={'x': 3, 'y': 6}, reduction=2))
        prob.driver.add_response('f_xy')
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        expected = [
            {'x': np.array([0.]), 'y': np.array([0.]), 'f_xy': 22.},
            {'x': np.array([0.]), 'y': np.array([0.4]), 'f_xy': 25.36},
            {'x': np.array([0.]), 'y': np.array([0.8]), 'f_xy': 29.04},
            {'x': np.array([1.]), 'y': np.array([0.]), 'f_xy': 17.},
            {'x': np.array([1.]), 'y': np.array([0.4]), 'f_xy': 20.76},
            {'x': np.array([1.]), 'y': np.array([0.8]), 'f_xy': 24.84},
            {'x': np.array([0.5]), 'y': np.array([0.2]), 'f_xy': 20.99},
            {'x': np.array([0.5]), 'y': np.array([0.6]), 'f_xy': 24.71},
            {'x': np.array([0.5]), 'y': np.array([1.]), 'f_xy': 28.75},
        ]

        cr = om.CaseReader(prob.get_outputs_dir() / "cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), 9)

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            for name in ('x', 'y'):
                self.assertAlmostEqual(outputs[name][0], expected_case[name][0])
            self.assertAlmostEqual(outputs['f_xy'], expected_case['f_xy'])

    def test_generalized_subset_array(self):
        # Number of levels specified individually for all factors (arrays).

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

        model.set_input_defaults('x', np.array([0.0, 0.0]))
        model.set_input_defaults('y', np.array([0.0, 0.0]))
        model.add_subsystem('comp', Digits2Num(), promotes=['*'])

        factors = {
            'x': {'lower': np.array([0., 0.]), 'upper': np.array([1.0, 2.0])},
            'y': {'lower': np.array([0., 0.]), 'upper': np.array([3.0, 4.0])}
        }

        prob.driver = om.AnalysisDriver(GeneralizedSubsetGenerator(factors, levels={'x': 5, 'y': 8}, reduction=14))
        prob.driver.add_response('f')
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(prob.get_outputs_dir() / "cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        objs = [cr.get_case(case).outputs['f'].item() for case in cases]

        self.assertEqual(len(objs), 104)  # The number can be verified with standalone pyDOE3
        # Testing uniqueness. If all elements are unique, it should be the same length as the number of cases
        self.assertEqual(len(set(objs)), 104)

    def test_plackett_burman(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        factors = {
            'x': {'lower': 0.0, 'upper': 1.0},
            'y': {'lower': 0.0, 'upper': 1.0}
        }

        prob.driver = om.AnalysisDriver(PlackettBurmanGenerator(factors))
        prob.driver.add_response('f_xy')
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

        cr = om.CaseReader(prob.get_outputs_dir() / "cases.sql")
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

        factors = {
            'x': {'lower': 0.0, 'upper': upper},
            'y': {'lower': 0.0, 'upper': upper},
            'z': {'lower': 0.0, 'upper': upper}
        }
        prob.driver = om.AnalysisDriver(BoxBehnkenGenerator(factors, center=center))
        prob.driver.add_response('a')

        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(prob.get_outputs_dir() / "cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        # The Box-Behnken design for 3 factors involves three blocks, in each of
        # which 2 factors are varied thru the 4 possible combinations of high & low.
        # It also includes centre points (all factors at their central values).
        # ref: https://en.wikipedia.org/wiki/Box-Behnken_design
        self.assertEqual(len(cases), (3*4)+center)

        # slight change in order from refactor in pyDOE3 v1.0.4 (PR #15)
        if Version(pyDOE3.__version__) >= Version("1.0.4"):
            expected = [
                {'x': np.array([0.]), 'y': np.array([0.]), 'z': np.array([5.])},
                {'x': np.array([0.]), 'y': np.array([10.]), 'z': np.array([5.])},
                {'x': np.array([10.]), 'y': np.array([0.]), 'z': np.array([5.])},
                {'x': np.array([10.]), 'y': np.array([10.]), 'z': np.array([5.])},

                {'x': np.array([0.]), 'y': np.array([5.]), 'z': np.array([0.])},
                {'x': np.array([0.]), 'y': np.array([5.]), 'z': np.array([10.])},
                {'x': np.array([10.]), 'y': np.array([5.]), 'z': np.array([0.])},
                {'x': np.array([10.]), 'y': np.array([5.]), 'z': np.array([10.])},

                {'x': np.array([5.]), 'y': np.array([0.]), 'z': np.array([0.])},
                {'x': np.array([5.]), 'y': np.array([0.]), 'z': np.array([10.])},
                {'x': np.array([5.]), 'y': np.array([10.]), 'z': np.array([0.])},
                {'x': np.array([5.]), 'y': np.array([10.]), 'z': np.array([10.])},

                {'x': np.array([5.]), 'y': np.array([5.]), 'z': np.array([5.])}
            ]
        else:
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

        factors = {
            'x': {'lower': xlb, 'upper': xub},
            'y': {'lower': ylb, 'upper': yub}
        }

        prob.driver = om.AnalysisDriver(LatinHypercubeGenerator(factors, samples=4, seed=0))
        prob.driver.add_response('f_xy')

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

        cr = om.CaseReader(prob.get_outputs_dir() / "cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), 4)

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            x = outputs['x'].item()
            y = outputs['y'].item()

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

        factors = {
            'xy': {'lower': bounds[0], 'upper': bounds[1]},
        }

        prob.driver = om.AnalysisDriver(LatinHypercubeGenerator(factors, samples=4, seed=0))
        prob.driver.add_response('f_xy')
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

        cr = om.CaseReader(prob.get_outputs_dir() / "cases.sql")
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

        factors = {
            'indep.x': {'lower': 0.0, 'upper': upper},
            'indep.y': {'lower': 0.0, 'upper': upper}
        }

        prob.driver = om.AnalysisDriver(LatinHypercubeGenerator(factors, samples=samples, criterion='c'))
        prob.driver.add_response('comp.f_xy')
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))
        prob.driver.recording_options['includes'] = ['indep.x', 'indep.y']

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(prob.get_outputs_dir() / "cases.sql")
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
            x = outputs['indep.x'].item()
            y = outputs['indep.y'].item()

            x_buckets_filled.add(int(x/bucket_size))
            y_buckets_filled.add(int(y/bucket_size))

            self.assertTrue(round(x, 3) in valid_values, '%f not in %s' % (x, valid_values))
            self.assertTrue(round(y, 3) in valid_values, '%f not in %s' % (y, valid_values))

        self.assertEqual(x_buckets_filled, all_buckets)
        self.assertEqual(y_buckets_filled, all_buckets)


if __name__ == "__main__":
    unittest.main()
