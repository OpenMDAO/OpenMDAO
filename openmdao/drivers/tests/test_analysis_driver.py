"""
Test Analysis Driver and Generators.
"""
import csv
import glob
import unittest

import numpy as np
import openmdao.api as om

from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.utils.assert_utils import assert_warning, assert_warnings, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.mpi import MPI


try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


fullfact3 = [
    {'x': {'val': 0.}, 'y': {'val': 0.}},
    {'x': {'val': .5}, 'y': {'val': 0.}},
    {'x': {'val': 1.}, 'y': {'val': 0.}},

    {'x': {'val': 0.}, 'y': {'val': 0.5}},
    {'x': {'val': .5}, 'y': {'val': 0.5}},
    {'x': {'val': 1.}, 'y': {'val': 0.5}},

    {'x': {'val': 0.}, 'y': {'val': 1.}},
    {'x': {'val': .5}, 'y': {'val': 1.}},
    {'x': {'val': 1.}, 'y': {'val': 1.}},
]

expected_fullfact3 = [
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

expected_fullfact3_derivs = [
    {('f_xy', 'x'): np.array(-6.), ('f_xy', 'y'): np.array(8.)},
    {('f_xy', 'x'): np.array(-5.), ('f_xy', 'y'): np.array(8.5)},
    {('f_xy', 'x'): np.array(-4.), ('f_xy', 'y'): np.array(9.)},

    {('f_xy', 'x'): np.array(-5.5), ('f_xy', 'y'): np.array(9.)},
    {('f_xy', 'x'): np.array(-4.5), ('f_xy', 'y'): np.array(9.5)},
    {('f_xy', 'x'): np.array(-3.5), ('f_xy', 'y'): np.array(10.)},

    {('f_xy', 'x'): np.array(-5.), ('f_xy', 'y'): np.array(10.)},
    {('f_xy', 'x'): np.array(-4.), ('f_xy', 'y'): np.array(10.5)},
    {('f_xy', 'x'): np.array(-3.), ('f_xy', 'y'): np.array(11.)},
]


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
        discrete_outputs['f_xy'] = f_xy.astype(int)


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
@use_tempdirs
class TestAnalysisDriverParallel(unittest.TestCase):

    N_PROCS = 4

    def test_simple_fullfact3(self):
        """
        Test AnalysisDriver with an explicit list of samples to be run.
        """
        prob = om.Problem()

        prob.model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = om.AnalysisDriver(samples=fullfact3, run_parallel=True)
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.driver.add_response('f_xy', units=None, indices=[0])

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        if prob.comm.rank == 0:
            num_recorded_cases = 0
            for file in glob.glob(str(prob.get_outputs_dir() / "cases.sql*")):
                if file.endswith('meta'):
                    continue
                cr = om.CaseReader(file)
                for case in cr.get_cases(source='driver'):
                    case_number = int(case.name.split('|')[-1])
                    assert_near_equal(case.get_val('f_xy'),
                                      expected_fullfact3[case_number]['f_xy'])
                num_recorded_cases += len(cr.list_cases(out_stream=None))
            self.assertEqual(num_recorded_cases, 9)

    def test_fullfact3_derivs(self):
        """
        Test AnalysisDriver with an explicit list of samples to be run
        and record the derivatives.
        """
        prob = om.Problem(reports=None)

        prob.model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = om.AnalysisDriver(samples=fullfact3, run_parallel=True)
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))
        prob.driver.recording_options.set(record_derivatives=True)

        prob.driver.add_response('f_xy', units=None, indices=[0])

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        if prob.comm.rank == 0:
            num_recorded_cases = 0
            for file in glob.glob(str(prob.get_outputs_dir() / "cases.sql*")):
                if file.endswith('meta'):
                    continue
                cr = om.CaseReader(file)
                for case in cr.get_cases(source='driver'):
                    case_number = int(case.name.split('|')[-1])
                    assert_near_equal(case.get_val('f_xy'),
                                      expected_fullfact3[case_number]['f_xy'])
                    for wrt in ['x', 'y']:
                        expected_deriv = expected_fullfact3_derivs[case_number]['f_xy', wrt]
                        assert_near_equal(case.derivatives['f_xy', wrt],
                                          np.atleast_2d(expected_deriv))
                num_recorded_cases += len(cr.list_cases(out_stream=None))
            self.assertEqual(num_recorded_cases, 9)

    def test_product_generator(self):
        """
        Test AnalysisDriver with an explicit list of samples to be run.
        """
        samples = {'x': {'val': [0.0, 0.5, 1.0]},
                   'y': {'val': [0.0, 0.5, 1.0]}}

        prob = om.Problem()

        prob.model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = om.AnalysisDriver(samples=om.ProductGenerator(samples), run_parallel=True)
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.driver.add_response('f_xy', units=None, indices=[0])

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        if prob.comm.rank == 0:
            num_recorded_cases = 0
            for file in glob.glob(str(prob.get_outputs_dir() / "cases.sql*")):
                if file.endswith('meta'):
                    continue
                cr = om.CaseReader(file)
                num_recorded_cases += len(cr.list_cases(out_stream=None))
            self.assertEqual(num_recorded_cases, 9)

    def test_list_parallel(self):
        """
        Test AnalysisDriver with an explicit list of samples to be run.
        """
        samples = {'x': {'val': [0.0, 0.5, 1.0]},
                   'y': {'val': [0.0, 0.5, 1.0]}}

        generator = om.ProductGenerator(samples)

        samples_list = [s for s in generator]

        prob = om.Problem()

        prob.model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = om.AnalysisDriver(samples=samples_list, run_parallel=True, batch_size=4)
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.driver.add_response('f_xy', units=None, indices=[0])

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        if prob.comm.rank == 0:
            num_recorded_cases = 0
            for file in glob.glob(str(prob.get_outputs_dir() / "cases.sql*")):
                if file.endswith('meta'):
                    continue
                cr = om.CaseReader(file)
                num_recorded_cases += len(cr.list_cases(out_stream=None))
            self.assertEqual(num_recorded_cases, 9)

    def test_large_sample_set(self):
        """
        Test AnalysisDriver with an explicit list of samples to be run.
        """
        samples = {'x': {'val': np.linspace(-10, 10, 50)},
                   'y': {'val': np.linspace(-10, 10, 50)}}

        prob = om.Problem()

        prob.model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = om.AnalysisDriver(samples=om.ProductGenerator(samples), run_parallel=True)
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))
        # from openmdao.recorders.stream_recorder import StreamRecorder
        # prob.driver.add_recorder(StreamRecorder())
        prob.driver.add_response('f_xy', units=None, indices=[0])

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        # prob.comm.barrier()

        if prob.comm.rank == 0:
            num_recorded_cases = 0
            for file in glob.glob(str(prob.get_outputs_dir() / "cases.sql*")):
                if file.endswith('meta'):
                    continue
                cr = om.CaseReader(file)
                num_recorded_cases += len(cr.list_cases(out_stream=None))
            self.assertEqual(num_recorded_cases,
                             samples['x']['val'].size * samples['y']['val'].size)

    def test_zip_generator(self):
        """
        Test AnalysisDriver with an explicit list of samples to be run.
        """
        samples = {'x': {'val': [0.0, 0.5, 1.0, 1.5, 2.0]},
                   'y': {'val': [0.0, 0.5, 1.0, 1.5, 2.0]}}

        prob = om.Problem()

        prob.model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = om.AnalysisDriver(samples=om.ZipGenerator(samples), run_parallel=True)
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.driver.add_response('f_xy', units=None, indices=[0])

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        if prob.comm.rank == 0:
            num_recorded_cases = 0
            for file in glob.glob(str(prob.get_outputs_dir() / "cases.sql*")):
                if file.endswith('meta'):
                    continue
                cr = om.CaseReader(file)
                num_recorded_cases += len(cr.list_cases(out_stream=None))
            self.assertEqual(num_recorded_cases, 5)

    def test_parallel_system(self):
        import numpy as np
        import openmdao.api as om

        class FanInGrouped(om.Group):
            """
            Topology where two components in a Group feed a single component
            outside of that Group.
            """

            def __init__(self):
                super().__init__()

                self.set_input_defaults('x1', 1.0)
                self.set_input_defaults('x2', 1.0)

                self.sub = self.add_subsystem('sub', om.ParallelGroup(),
                                            promotes_inputs=['x1', 'x2'])

                self.sub.add_subsystem('c1', om.ExecComp(['y=-2.0*x']),
                                    promotes_inputs=[('x', 'x1')])
                self.sub.add_subsystem('c2', om.ExecComp(['y=5.0*x']),
                                    promotes_inputs=[('x', 'x2')])

                self.add_subsystem('c3', om.ExecComp(['y=3.0*x1+7.0*x2']))

                self.connect("sub.c1.y", "c3.x1")
                self.connect("sub.c2.y", "c3.x2")

        prob = om.Problem(FanInGrouped())

        # Note the absense of adding design varaibles here, compared to DOEGenerator

        # the FanInGrouped model uses 2 processes, so we can run
        # two instances of the model at a time, each using 2 of our 4 procs
        procs_per_model = 2
        prob.driver = om.AnalysisDriver(om.ProductGenerator({'x1': {'val': np.linspace(0.0, 1.0, 10)},
                                                            'x2': {'val': np.linspace(0.0, 1.0, 10)}}),
                                        run_parallel=True,
                                        procs_per_model=procs_per_model)
        # prob.driver.add_response('c3.y')
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))
        prob.driver.recording_options['includes'].append('sub.c1.x')
        prob.driver.recording_options['includes'].append('sub.c2.x')

        prob.setup()
        prob.final_setup()
        prob.run_driver()
        prob.cleanup()

        num_models = prob.comm.size // procs_per_model
        rank = prob.comm.rank

        if rank < num_models:
            filename = f'cases.sql_{rank}'
            cr = om.CaseReader(prob.get_outputs_dir() / filename)
            cases = cr.list_cases(source='driver', out_stream=None)

            case_nums = {int(s.split('|')[-1]) for s in cases}

            # On rank 0 we should have even cases, with odd cases on rank 1
            self.assertSetEqual(case_nums, {i for i in range(rank, 100, 2)})

@use_tempdirs
class TestAnalysisDriver(unittest.TestCase):

    def test_changing_sample_vars(self):
        """
        Test AnalysisDriver when the variables changed in the samples are changing.
        """
        samples = [
            {'x': {'val': 0.}, 'y': {'val': 0.}},
            {'x': {'val': .5}},
            {'x': {'val': 1.}, 'y': {'val': 0.}},
            {'y': {'val': 0.5}},
        ]

        prob = om.Problem()

        prob.model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = om.AnalysisDriver(samples=samples)
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.driver.add_response('f_xy', units=None, indices=[0])

        prob.setup()

        expected_warnings = [(om.DriverWarning,
                              ("The variables in sample 1 differ from\n"
                               "the previous sample's variables.\n"
                               "New variables: {'y'}\n")),
                             (om.DriverWarning,
                              ("The variables in sample 2 differ from\n"
                              "the previous sample's variables.\n"
                              "Missing variables: {'y'}\n")),
                             (om.DriverWarning,
                              ("The variables in sample 3 differ from\n"
                               "the previous sample's variables.\n"
                               "New variables: {'x'}\n"))]

        with assert_warnings(expected_warnings):
            prob.run_driver()

        prob.cleanup()

        if prob.comm.rank == 0:
            num_recorded_cases = 0
            for file in glob.glob(str(prob.get_outputs_dir() / "cases.sql*")):
                if file.endswith('meta'):
                    continue
                cr = om.CaseReader(file)
                num_recorded_cases += len(cr.list_cases(out_stream=None))
            self.assertEqual(num_recorded_cases, 4)

    def test_output_in_sample(self):
        """
        Test AnalysisDriver when the variables changed in the samples are changing.
        """
        samples = [
            {'x': {'val': 0.}, 'y': {'val': 0.}, 'f_xy': {'val': 0.0}},
        ]

        prob = om.Problem()

        prob.model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = om.AnalysisDriver(samples=samples)
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.driver.add_response('f_xy', units=None, indices=[0])

        prob.setup()

        expected_warning = ("Variable `f_xy` is neither an independent variable\n"
                            "nor an implicit output in the model on rank 0.\n"
                            "Setting its value in the case data will have no\n"
                            "impact on the outputs of the model after execution.")
        with assert_warning(om.DriverWarning, expected_warning):
            prob.run_driver()
        prob.cleanup()

        if prob.comm.rank == 0:
            num_recorded_cases = 0
            for file in glob.glob(str(prob.get_outputs_dir() / "cases.sql*")):
                if file.endswith('meta'):
                    continue
                cr = om.CaseReader(file)
                num_recorded_cases += len(cr.list_cases(out_stream=None))
            self.assertEqual(num_recorded_cases, 1)

    def test_csv(self):

        ### Part 1 - Create a CSV file of the cases we want to run

        var_dict = {'x': {'val': [0.0, 0.5, 1.0], 'units': None, 'indices': [0]},
                    'y': {'val': [0.0, 0.5, 1.0], 'units': None, 'indices': [0]}}

        case_gen = om.ProductGenerator(var_dict)
        cases_csv_data = []

        row_0 = {v: data['units'] for v, data in var_dict.items()}
        row_1 = {v: data['indices'] for v, data in var_dict.items()}

        cases_csv_data = [row_0, row_1]

        for c in case_gen:
            case_i = {}
            for k, v in c.items():
                case_i[k] = v['val']
            cases_csv_data.append(case_i)

        with open('samples.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=var_dict.keys())
            writer.writeheader()
            writer.writerows(cases_csv_data)

        ### Part 2 - Run the CSVGenerator on the file we just created

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('paraboloid', Paraboloid(), promotes_inputs=['x', 'y'], promotes_outputs=['f_xy'])

        prob.setup()

        # # create DOEDriver using generated CSV file
        prob.driver = om.AnalysisDriver(om.CSVAnalysisGenerator('samples.csv', has_units=True, has_indices=True))
        prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

        prob.run_driver()
        prob.cleanup()

        expected = [
            {'x': np.array([0.0]), 'y': np.array([0.0])},
            {'x': np.array([0.0]), 'y': np.array([0.5])},
            {'x': np.array([0.0]), 'y': np.array([1.0])},
            {'x': np.array([0.5]), 'y': np.array([0.0])},
            {'x': np.array([0.5]), 'y': np.array([0.5])},
            {'x': np.array([0.5]), 'y': np.array([1.0])},
            {'x': np.array([1.0]), 'y': np.array([0.0])},
            {'x': np.array([1.0]), 'y': np.array([0.5])},
            {'x': np.array([1.0]), 'y': np.array([1.0])},
        ]

        cr = om.CaseReader(prob.get_outputs_dir() / "cases.sql")
        cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(len(cases), len(expected))

        for case, expected_case in zip(cases, expected):
            outputs = cr.get_case(case).outputs
            self.assertEqual(outputs['x'][0], expected_case['x'][0])
            self.assertEqual(outputs['y'][0], expected_case['y'][0])

    def test_zip_generator_incompatible_sizes(self):
        """
        Test that ZipGenerator raises if the given value lists do not agree in shape.
        """
        samples = {'x': {'val': [0.0, 0.5, 1.0, 1.5, 2.0]},
                   'y': {'val': [0.0, 0.5, 1.0, 1.5]}}

        with self.assertRaises(ValueError) as e:
            om.ZipGenerator(samples)

        expected = ("ZipGenerator requires that val for all var_dict have the same length:\n"
                   "{'x': 5, 'y': 4}")

        self.assertEqual(expected, str(e.exception))


class TestErrors(unittest.TestCase):

    def test_generator_check(self):
        prob = om.Problem()

        with self.assertRaises(ValueError) as err:
            prob.driver = om.AnalysisDriver(om.Problem())

        self.assertEqual(str(err.exception),
                         "samples must be a list, tuple, or derived from AnalysisGenerator "
                         "but got <class 'openmdao.core.problem.Problem'>")


if __name__ == "__main__":
    unittest.main()
