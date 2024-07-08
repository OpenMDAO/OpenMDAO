""" Unit tests that exercise Case via the Problem.load_case method. """

import unittest

import numpy as np

import openmdao.api as om
from openmdao.recorders.tests.recorder_test_utils import assert_model_matches_case
from openmdao.core.tests.test_units import SpeedComp
from openmdao.test_suite.components.expl_comp_array import TestExplCompArray
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.paraboloid_problem import ParaboloidProblem
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped, SellarProblem
from openmdao.utils.assert_utils import assert_near_equal, assert_warnings, assert_no_warning
from openmdao.utils.om_warnings import OpenMDAOWarning
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.units import convert_units
from openmdao.utils.mpi import MPI, multi_proc_exception_check

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


@use_tempdirs
class TestLoadCase(unittest.TestCase):

    def setUp(self):
        self.filename = "sqlite_test"
        self.recorder = om.SqliteRecorder(self.filename, record_viewer_data=False)

    def test_simple_load_system_cases(self):
        prob = SellarProblem()

        model = prob.model
        model.recording_options['record_inputs'] = True
        model.recording_options['record_outputs'] = True
        model.recording_options['record_residuals'] = True
        model.add_recorder(self.recorder)

        prob.setup()

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        system_cases = cr.list_cases('root', out_stream=None)
        case = cr.get_case(system_cases[0])

        # Add one to all the inputs and outputs just to change the model
        #   so we can see if loading the case values really changes the model
        for name in model._inputs:
            model._inputs[name] += 1.0
        for name in model._outputs:
            model._outputs[name] += 1.0

        # Now load in the case we recorded
        prob.load_case(case)

        assert_model_matches_case(case, model)

    def test_load_incompatible_model(self):
        prob = SellarProblem(SellarDerivativesGrouped)

        prob.model.add_recorder(self.recorder)

        driver = prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        system_cases = cr.list_cases('root', out_stream=None)
        case = cr.get_case(system_cases[0])

        # try to load it into a completely different model
        prob = ParaboloidProblem()
        prob.setup()

        expected_warnings = [
            (OpenMDAOWarning, "<model> <class Group>: Input variable, 'y1', recorded in the case is not found in the model."),
            (OpenMDAOWarning, "<model> <class Group>: Input variable, 'y2', recorded in the case is not found in the model."),
            (OpenMDAOWarning, "<model> <class Group>: Input variable, 'z', recorded in the case is not found in the model."),
            (OpenMDAOWarning, "<model> <class Group>: Output variable, 'z', recorded in the case is not found in the model."),
            (OpenMDAOWarning, "<model> <class Group>: Output variable, 'con1', recorded in the case is not found in the model."),
            (OpenMDAOWarning, "<model> <class Group>: Output variable, 'con2', recorded in the case is not found in the model."),
            (OpenMDAOWarning, "<model> <class Group>: Output variable, 'y1', recorded in the case is not found in the model."),
            (OpenMDAOWarning, "<model> <class Group>: Output variable, 'y2', recorded in the case is not found in the model."),
            (OpenMDAOWarning, "<model> <class Group>: Output variable, 'obj', recorded in the case is not found in the model.")
        ]
        with assert_warnings(expected_warnings):
            prob.load_case(case)

    def test_load_equivalent_model(self):
        prob = SellarProblem(SellarDerivativesGrouped)

        prob.model.add_recorder(self.recorder)

        driver = prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        system_cases = cr.list_cases('root', out_stream=None)
        case = cr.get_case(system_cases[0])

        # try to load it into a different version of the Sellar model
        # this should succeed with no warnings due to the two models
        # having the same promoted inputs/outputs, even though the
        # underlying model heierarchy has changed
        prob = SellarProblem()
        prob.setup()

        with assert_no_warning(UserWarning):
            prob.load_case(case)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

    def test_subsystem_load_system_cases(self):
        prob = SellarProblem()
        prob.setup()

        model = prob.model
        model.recording_options['record_inputs'] = True
        model.recording_options['record_outputs'] = True
        model.recording_options['record_residuals'] = True

        # Only record a subsystem
        model.d2.add_recorder(self.recorder)

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        system_cases = cr.list_cases('root.d2', out_stream=None)
        case = cr.get_case(system_cases[0])

        # Add one to all the inputs just to change the model
        #   so we can see if loading the case values really changes the model
        for name in prob.model._inputs:
            model._inputs[name] += 1.0
        for name in prob.model._outputs:
            model._outputs[name] += 1.0

        # Now load in the case we recorded
        prob.load_case(case)

        assert_model_matches_case(case, model.d2)

    def test_load_system_cases_with_units(self):
        comp = om.IndepVarComp()
        comp.add_output('distance', val=1., units='m')
        comp.add_output('time', val=1., units='s')

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('c1', comp)
        model.add_subsystem('c2', SpeedComp())
        model.add_subsystem('c3', om.ExecComp('f=speed', speed={'units': 'm/s'}, f={'units': 'm/s'}))
        model.connect('c1.distance', 'c2.distance')
        model.connect('c1.time', 'c2.time')
        model.connect('c2.speed', 'c3.speed')

        model.add_recorder(self.recorder)

        prob.setup()
        prob.run_model()

        cr = om.CaseReader(self.filename)

        system_cases = cr.list_cases('root', out_stream=None)
        case = cr.get_case(system_cases[0])

        # Add one to all the inputs just to change the model
        # so we can see if loading the case values really changes the model
        for name in model._inputs:
            model._inputs[name] += 1.0
        for name in model._outputs:
            model._outputs[name] += 1.0

        # Now load in the case we recorded
        prob.load_case(case)

        assert_model_matches_case(case, model)

        # make sure it still runs with loaded values
        prob.run_model()

        # make sure the loaded unit strings are compatible with `convert_units`
        outputs = case.list_outputs(explicit=True, implicit=True, val=True,
                                    units=True, shape=True, out_stream=None)
        meta = {}
        for name, vals in outputs:
            meta[name] = vals

        from_units = meta['c2.speed']['units']
        to_units = meta['c3.f']['units']

        self.assertEqual(from_units, 'km/h')
        self.assertEqual(to_units, 'm/s')

        self.assertEqual(convert_units(10., from_units, to_units), 10000./3600.)

    def test_load_system_with_discrete_values(self):
        # Defines a test class with discrete inputs and outputs
        class ParaboloidWithDiscreteOutput(Paraboloid):

            def setup(self):
                super().setup()
                self.add_discrete_input('disc_in', val='in')
                self.add_discrete_output('disc_out', val='out')
                self.add_design_var('x', lower=-50, upper=50)
                self.add_design_var('y', lower=-50, upper=50)
                self.add_objective('f_xy')

            def compute(self, inputs, outputs, d_ins, d_outs):
                super().compute(inputs, outputs)

            def compute_partials(self, inputs, outputs, d_ins):
                super().compute_partials(inputs, outputs)


        # Setup the optimization 
        prob = om.Problem() 
        prob.model.add_subsystem(
            'paraboloid', ParaboloidWithDiscreteOutput())
        prob.driver = om.ScipyOptimizeDriver() 
        prob.driver.options['optimizer'] = 'SLSQP'

        # Setup Recorder
        prob.model.recording_options['record_inputs'] = True
        prob.model.recording_options['record_outputs'] = True
        recorder = om.SqliteRecorder('cases.sql')
        prob.add_recorder(recorder)

        # Run the Opt
        prob.setup()
        prob.run_driver()
        prob.record("after_run_driver")

        # Set the discrete value to something arbritrary
        prob.set_val('paraboloid.disc_in', 'INCORRECT_VAL')
        prob.set_val('paraboloid.disc_out', 'INCORRECT_VAL')

        # Load the Case from the recorder
        cr = om.CaseReader("cases.sql")
        case = cr.get_case('after_run_driver')
        prob.load_case(case)

        # Assert that the values have returned to those following the opt
        self.assertEqual(prob.get_val('paraboloid.disc_in'), 'in')
        self.assertEqual(prob.get_val('paraboloid.disc_out'), 'out')

    def test_optimization_load_system_cases(self):
        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=om.NonlinearRunOnce,
                                                       linear_solver=om.ScipyKrylov,
                                                       mda_nonlinear_solver=om.NonlinearBlockGS)

        prob.model.add_recorder(self.recorder)

        driver = prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        inputs_before = prob.model.list_inputs(val=True, units=True, out_stream=None)
        outputs_before = prob.model.list_outputs(val=True, units=True, out_stream=None)

        cr = om.CaseReader(self.filename)

        # get third case
        system_cases = cr.list_cases('root', out_stream=None)
        third_case = cr.get_case(system_cases[2])

        iter_count_before = driver.iter_count

        # recreate the Problem with a fresh model
        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=om.NonlinearRunOnce,
                                                       linear_solver=om.ScipyKrylov,
                                                       mda_nonlinear_solver=om.NonlinearBlockGS)

        driver = prob.driver = om.ScipyOptimizeDriver()
        driver.options['optimizer'] = 'SLSQP'
        driver.options['tol'] = 1e-9
        driver.options['disp'] = False

        prob.setup()
        prob.load_case(third_case)
        prob.run_driver()
        prob.cleanup()

        inputs_after = prob.model.list_inputs(val=True, units=True, out_stream=None)
        outputs_after = prob.model.list_outputs(val=True, units=True, out_stream=None)

        iter_count_after = driver.iter_count

        for before, after in zip(inputs_before, inputs_after):
            np.testing.assert_almost_equal(before[1]['val'], after[1]['val'])

        for before, after in zip(outputs_before, outputs_after):
            np.testing.assert_almost_equal(before[1]['val'], after[1]['val'])

        # Should take one less iteration since we gave it a head start in the second run
        self.assertEqual(iter_count_before, iter_count_after + 1)

    def test_load_solver_cases(self):
        prob = SellarProblem()
        prob.setup()

        model = prob.model
        model.nonlinear_solver.add_recorder(self.recorder)

        fail = not prob.run_driver().success
        prob.cleanup()

        self.assertFalse(fail, 'Problem failed to converge')

        cr = om.CaseReader(self.filename)

        solver_cases = cr.list_cases('root.nonlinear_solver', out_stream=None)
        case = cr.get_case(solver_cases[0])

        # Add one to all the inputs just to change the model
        # so we can see if loading the case values really changes the model
        for name in prob.model._inputs:
            model._inputs[name] += 1.0
        for name in prob.model._outputs:
            model._outputs[name] += 1.0

        # Now load in the case we recorded
        prob.load_case(case)

        assert_model_matches_case(case, model)

    def test_load_driver_cases(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)

        model.add_objective('f_xy')
        model.add_constraint('c', lower=15.0)

        prob.driver.add_recorder(self.recorder)
        prob.driver.recording_options['includes'] = ['*']

        prob.set_solver_print(0)

        prob.setup()
        fail = not prob.run_driver().success
        prob.cleanup()

        self.assertFalse(fail, 'Problem failed to converge')

        cr = om.CaseReader(self.filename)

        driver_cases = cr.list_cases('driver', out_stream=None)
        case = cr.get_case(driver_cases[0])

        # Add one to all the inputs just to change the model
        #   so we can see if loading the case values really changes the model
        for name in prob.model._inputs:
            prob.model._inputs[name] += 1.0
        for name in prob.model._outputs:
            prob.model._outputs[name] += 1.0

        # Now load in the case we recorded
        prob.load_case(case)

        assert_model_matches_case(case, model)

    def test_reading_driver_cases_with_indices(self):
        # note: size must be an even number
        SIZE = 10
        prob = om.Problem()

        driver = prob.driver = om.ScipyOptimizeDriver(disp=False)

        prob.driver.add_recorder(self.recorder)
        driver.recording_options['includes'] = ['*']

        model = prob.model
        indeps = model.add_subsystem('indeps', om.IndepVarComp(), promotes_outputs=['*'])

        # the following were randomly generated using np.random.random(10)*2-1 to randomly
        # disperse them within a unit circle centered at the origin.
        # Also converted this array to > 1D array to test that capability of case recording
        x_vals = np.array([
            0.55994437, -0.95923447, 0.21798656, -0.02158783, 0.62183717,
            0.04007379, 0.46044942, -0.10129622, 0.27720413, -0.37107886
        ]).reshape((-1, 1))

        indeps.add_output('x', x_vals)
        indeps.add_output('y', np.array([
            0.52577864, 0.30894559, 0.8420792, 0.35039912, -0.67290778,
            -0.86236787, -0.97500023, 0.47739414, 0.51174103, 0.10052582
        ]))
        indeps.add_output('r', .7)

        model.add_subsystem('circle', om.ExecComp('area = pi * r**2'))

        model.add_subsystem('r_con', om.ExecComp('g = x**2 + y**2 - r**2',
                                                 g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)))

        thetas = np.linspace(0, np.pi/4, SIZE)

        model.add_subsystem('theta_con', om.ExecComp('g=arctan(y/x) - theta',
                                                     g=np.ones(SIZE), x=np.ones(SIZE),
                                                     y=np.ones(SIZE), theta=thetas))
        model.add_subsystem('delta_theta_con', om.ExecComp('g = arctan(y/x)[::2]-arctan(y/x)[1::2]',
                                                           g=np.ones(SIZE//2), x=np.ones(SIZE),
                                                           y=np.ones(SIZE)))

        model.add_subsystem('l_conx', om.ExecComp('g=x-1', g=np.ones(SIZE), x=np.ones(SIZE)))

        model.connect('r', ('circle.r', 'r_con.r'))
        model.connect('x', ['r_con.x', 'theta_con.x', 'delta_theta_con.x'])
        model.connect('x', 'l_conx.x')
        model.connect('y', ['r_con.y', 'theta_con.y', 'delta_theta_con.y'])

        model.add_design_var('x', indices=[0, 3], flat_indices=True)
        model.add_design_var('y')
        model.add_design_var('r', lower=.5, upper=10)

        # nonlinear constraints
        model.add_constraint('r_con.g', equals=0)

        IND = np.arange(SIZE, dtype=int)
        EVEN_IND = IND[0::2]  # all even indices
        model.add_constraint('theta_con.g', lower=-1e-5, upper=1e-5, indices=EVEN_IND)
        model.add_constraint('delta_theta_con.g', lower=-1e-5, upper=1e-5)

        # this constrains x[0] to be 1 (see definition of l_conx)
        model.add_constraint('l_conx.g', equals=0, linear=False, indices=[0, ])

        # linear constraint
        model.add_constraint('y', equals=0, indices=[0], linear=True)

        model.add_objective('circle.area', ref=-1)

        prob.setup(mode='fwd')
        prob.run_driver()
        prob.cleanup()

        # get the case we recorded
        cr = om.CaseReader(self.filename)
        case = cr.get_case(0)

        # check 'use_indices' option, default is to use indices
        dvs = case.get_design_vars()
        assert_near_equal(dvs['x'], x_vals[[0, 3]], 1e-12)

        dvs = case.get_design_vars(use_indices=False)
        assert_near_equal(dvs['x'], x_vals, 1e-12)

        cons = case.get_constraints()
        self.assertEqual(len(cons['theta_con.g']), len(EVEN_IND))

        cons = case.get_constraints(use_indices=False)
        self.assertEqual(len(cons['theta_con.g']), SIZE)

        # add one to all the inputs just to change the model, so we
        # can see if loading the case values really changes the model
        for name in prob.model._inputs:
            model._inputs[name] += 1.0
        for name in prob.model._outputs:
            model._outputs[name] += 1.0

        # load in the case we recorded and check that the model then matches
        prob.load_case(case)
        assert_model_matches_case(case, model)

    def test_multidimensional_arrays(self):
        prob = om.Problem()
        model = prob.model

        comp = TestExplCompArray(thickness=1.)  # has 2D arrays as inputs and outputs
        model.add_subsystem('comp', comp, promotes=['*'])
        # just to add a connection, otherwise an exception is thrown in recording viewer data.
        # must be a bug
        model.add_subsystem('double_area',
                            om.ExecComp('double_area = 2 * areas',
                                        areas=np.zeros((2, 2)),
                                        double_area=np.zeros((2, 2))),
                            promotes=['*'])

        prob.driver.add_recorder(self.recorder)
        prob.driver.recording_options['includes'] = ['*']

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        # Add one to all the inputs just to change the model
        #   so we can see if loading the case values really changes the model
        for name in prob.model._inputs:
            model._inputs[name] += 1.0
        for name in prob.model._outputs:
            model._outputs[name] += 1.0

        # Now load in the case we recorded
        cr = om.CaseReader(self.filename)

        driver_cases = cr.list_cases('driver', out_stream=None)
        case = cr.get_case(driver_cases[0])

        prob.load_case(case)

        assert_model_matches_case(case, model)


class Adder(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', shape_by_conn=True, distributed=True)
        self.add_input('y')
        self.add_output('x_sum', shape=1)
        self.add_output('out_dist', copy_shape='x', distributed=True)

    def compute(self, inputs, outputs):
        outputs['x_sum'] = np.sum(inputs['x']) + inputs['y']**2.0
        outputs['out_dist'] = inputs['x'] + 1.


@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
@use_tempdirs
class TestLoadCaseMPI(unittest.TestCase):

    N_PROCS = 2

    def test_distrib_var_load(self):
        prob = om.Problem()
        ivc = prob.model.add_subsystem('ivc',om.IndepVarComp(), promotes=['*'])
        ivc.add_output('x', val = np.ones(100 if prob.comm.rank == 0 else 10), distributed=True)
        ivc.add_output('y', val = 1.0)

        prob.model.add_subsystem('adder', Adder(), promotes=['*'])
        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9)
        prob.model.add_design_var('y', lower=-1, upper=1)
        prob.model.add_objective('x_sum')

        recorder_file = 'distributed.sql'
        prob.driver.add_recorder(om.SqliteRecorder(recorder_file))
        prob.driver.recording_options['includes'] = ['*']
        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_constraints'] = True
        prob.driver.recording_options['record_desvars'] = True

        prob.setup()
        prob.run_driver()

        reader = om.CaseReader(recorder_file)
        prob.load_case(reader.get_case(-1))

        with multi_proc_exception_check(prob.comm):
            val = prob.get_val('adder.x', get_remote=False)
            if prob.comm.rank == 0:
                assert_near_equal(val, np.ones(100, dtype=float))
            else:
                assert_near_equal(val, np.ones(10, dtype=float))

            val = prob.get_val('adder.out_dist', get_remote=False)
            if prob.comm.rank == 0:
                assert_near_equal(val, np.ones(100, dtype=float) + 1.)
            else:
                assert_near_equal(val, np.ones(10, dtype=float) + 1.)


if __name__ == "__main__":
    unittest.main()
