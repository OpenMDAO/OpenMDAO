"""
Test Analysis Driver and Generators.
"""
import glob
import unittest

import numpy as np
import openmdao.api as om

from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.utils.assert_utils import assert_warning, assert_warnings, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.mpi import MPI
from openmdao.drivers.analysis_driver import AnalysisDriver
from openmdao.drivers.analysis_generators import ProductGenerator, ZipGenerator

from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_group import MultipointBeamGroup


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

        prob.driver = AnalysisDriver(samples=fullfact3)
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
        Test AnalysisDriver with an explicit list of samples to be run.
        """
        prob = om.Problem(reports=None)

        prob.model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        prob.driver = AnalysisDriver(samples=fullfact3)
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

        prob.driver = AnalysisDriver(samples=ProductGenerator(samples))
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

        prob.driver = AnalysisDriver(samples=ProductGenerator(samples))
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

        prob.driver = AnalysisDriver(samples=ZipGenerator(samples))
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

    def test_beam_np4(self):

        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01

        num_elements = 50 * 32
        num_cp = 4
        num_load_cases = 32

        beam_model = MultipointBeamGroup(E=E, L=L, b=b, volume=volume,
                                         num_elements=num_elements, num_cp=num_cp,
                                         num_load_cases=num_load_cases)

        prob = om.Problem(model=beam_model, driver=AnalysisDriver(procs_per_model=2))

        prob.set_solver_print(2)

        prob.setup()

        prob.run_model()

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

        prob.driver = AnalysisDriver(samples=samples)
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

        prob.driver = AnalysisDriver(samples=samples)
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


if __name__ == "__main__":
    unittest.main()
