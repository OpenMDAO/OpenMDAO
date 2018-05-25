""" Unit test for the SqliteRecorder. """
import errno
import os
import unittest
import numpy as np

import sqlite3

from shutil import rmtree
from six import PY2, PY3
from tempfile import mkdtemp

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, SqliteRecorder, \
    NonlinearRunOnce, NonlinearBlockGS, NonlinearBlockJac, NewtonSolver, \
    LinearRunOnce, LinearBlockGS, LinearBlockJac, DirectSolver, ScipyKrylov, PETScKrylov, \
    BoundsEnforceLS, ArmijoGoldsteinLS

from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.recorders.recording_iteration_stack import recording_iteration

from openmdao.test_suite.components.sellar import SellarDerivatives, SellarDerivativesGrouped, \
    SellarDis1withDerivatives, SellarDis2withDerivatives, SellarProblem
from openmdao.test_suite.components.paraboloid import Paraboloid

from openmdao.recorders.tests.sqlite_recorder_test_utils import assertMetadataRecorded, \
    assertDriverIterDataRecorded, assertSystemIterDataRecorded, assertSolverIterDataRecorded, \
    assertDriverMetadataRecorded, assertSystemMetadataIdsRecorded, assertSystemIterCoordsRecorded
from openmdao.recorders.tests.recorder_test_utils import run_driver
from openmdao.utils.assert_utils import assert_rel_error

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

if PY2:
    import cPickle as pickle
if PY3:
    import pickle

# check that pyoptsparse is installed. if it is, try to use SLSQP.
OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class ParaboloidProblem(Problem):
    """
    Paraboloid problem with Constraint.
    """
    def __init__(self):
        super(ParaboloidProblem, self).__init__()

        model = self.model
        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)


class TestSqliteRecorder(unittest.TestCase):

    def setUp(self):
        recording_iteration.stack = []  # reset to avoid problems from earlier tests

        self.orig_dir = os.getcwd()
        self.temp_dir = mkdtemp()
        os.chdir(self.temp_dir)

        self.filename = os.path.join(self.temp_dir, "sqlite_test")
        self.recorder = SqliteRecorder(self.filename)

        self.eps = 1e-3

    def tearDown(self):
        os.chdir(self.orig_dir)
        try:
            rmtree(self.temp_dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def test_only_desvars_recorded(self):
        prob = SellarProblem()

        driver = prob.driver
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_responses'] = False
        driver.recording_options['record_objectives'] = False
        driver.recording_options['record_constraints'] = False
        driver.recording_options['includes'] = []
        driver.add_recorder(self.recorder)

        prob.setup()
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0, )]
        expected_outputs = {"px.x": [1.0, ], "pz.z": [5.0, 2.0]}

        expected_data = ((coordinate, (t0, t1), expected_outputs, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_add_recorder_after_setup(self):
        prob = SellarProblem()

        driver = prob.driver
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_responses'] = False
        driver.recording_options['record_objectives'] = False
        driver.recording_options['record_constraints'] = False
        driver.recording_options['includes'] = []

        driver.add_recorder(self.recorder)

        prob.setup()
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0, )]
        expected_outputs = {"px.x": [1.0, ], "pz.z": [5.0, 2.0]}

        expected_data = ((coordinate, (t0, t1), expected_outputs, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_only_objectives_recorded(self):
        prob = SellarProblem()

        driver = prob.driver
        driver.recording_options['record_desvars'] = False
        driver.recording_options['record_responses'] = False
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = False
        driver.recording_options['includes'] = []
        driver.add_recorder(self.recorder)

        prob.setup()
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0, )]
        expected_objectives = {"obj_cmp.obj": [28.58830817, ]}
        expected_outputs = expected_objectives

        expected_data = ((coordinate, (t0, t1), expected_outputs, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_only_constraints_recorded(self):
        prob = SellarProblem()

        driver = prob.driver
        driver.recording_options['record_desvars'] = False
        driver.recording_options['record_responses'] = False
        driver.recording_options['record_objectives'] = False
        driver.recording_options['record_constraints'] = True
        driver.recording_options['includes'] = []
        driver.add_recorder(self.recorder)

        prob.setup()
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0, )]
        expected_constraints = {
            "con_cmp1.con1": [-22.42830237, ],
            "con_cmp2.con2": [-11.94151185, ],
        }
        expected_outputs = expected_constraints

        expected_data = ((coordinate, (t0, t1), expected_outputs, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP")
    def test_simple_driver_recording(self):
        prob = ParaboloidProblem()

        driver = prob.driver = pyOptSparseDriver(optimizer='SLSQP')
        driver.options['print_results'] = False
        driver.opt_settings['ACC'] = 1e-9
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_responses'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'SLSQP', (3, )]

        expected_desvars = {"p1.x": [7.16706813], "p2.y": [-7.83293187]}
        expected_objectives = {"comp.f_xy": [-27.0833]}
        expected_constraints = {"con.c": [-15.0]}

        expected_outputs = expected_desvars
        expected_outputs.update(expected_objectives)
        expected_outputs.update(expected_constraints)

        expected_inputs = {
            "con.x": 7.1666667,
            "comp.y": -7.83333333,
            "comp.x": 7.1666667,
            "con.y": -7.8333333
        }

        expected_data = ((coordinate, (t0, t1), expected_outputs, expected_inputs),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP")
    def test_simple_driver_recording_with_prefix(self):
        prob = ParaboloidProblem()

        driver = prob.driver = pyOptSparseDriver(optimizer='SLSQP')
        driver.options['print_results'] = False
        driver.opt_settings['ACC'] = 1e-9
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_responses'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        run1_t0, run1_t1 = run_driver(prob, case_prefix='Run1')
        run2_t0, run2_t1 = run_driver(prob, case_prefix='Run2')
        prob.cleanup()

        run1_coord = [0, 'SLSQP', (4, )]  # 1st run, 5 iterations
        run2_coord = [0, 'SLSQP', (1, )]  # 2nd run, 2 iterations

        expected_desvars = {"p1.x": [7.16706813], "p2.y": [-7.83293187]}
        expected_objectives = {"comp.f_xy": [-27.0833]}
        expected_constraints = {"con.c": [-15.0]}

        expected_outputs = expected_desvars
        expected_outputs.update(expected_objectives)
        expected_outputs.update(expected_constraints)

        expected_inputs = {
            "con.x": 7.1666667,
            "comp.y": -7.83333333,
            "comp.x": 7.1666667,
            "con.y": -7.8333333
        }

        expected_data = (
            (run1_coord, (run1_t0, run1_t1), expected_outputs, expected_inputs),
        )
        assertDriverIterDataRecorded(self, expected_data, self.eps, prefix='Run1')

        expected_data = (
            (run2_coord, (run2_t0, run2_t1), expected_outputs, expected_inputs),
        )
        assertDriverIterDataRecorded(self, expected_data, self.eps, prefix='Run2')

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP")
    def test_driver_everything_recorded_by_default(self):
        prob = ParaboloidProblem()

        driver = prob.driver = pyOptSparseDriver(optimizer='SLSQP')
        driver.options['print_results'] = False
        driver.opt_settings['ACC'] = 1e-9
        driver.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'SLSQP', (3, )]

        expected_desvars = {"p1.x": [7.16706813, ], "p2.y": [-7.83293187]}
        expected_objectives = {"comp.f_xy": [-27.0833]}
        expected_constraints = {"con.c": [-15.0]}

        expected_inputs = {
            "con.x": 7.1666667,
            "comp.y": -7.83333333,
            "comp.x": 7.1666667,
            "con.y": -7.8333333
        }

        expected_outputs = expected_desvars
        expected_outputs.update(expected_objectives)
        expected_outputs.update(expected_constraints)

        expected_data = ((coordinate, (t0, t1), expected_outputs, expected_inputs),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_driver_records_metadata(self):
        prob = SellarProblem()

        driver = prob.driver
        driver.recording_options['includes'] = ["p1.x"]
        driver.recording_options['record_metadata'] = True
        driver.add_recorder(self.recorder)

        prob.setup()
        prob.final_setup()  # Conclude setup but don't run model.
        prob.cleanup()

        prom2abs = {
            'input': {
                'z': ['d1.z', 'd2.z', 'obj_cmp.z'],
                'x': ['d1.x', 'obj_cmp.x'],
                'y2': ['d1.y2', 'obj_cmp.y2', 'con_cmp2.y2'],
                'y1': ['d2.y1', 'obj_cmp.y1', 'con_cmp1.y1']
            },
            'output': {
                'x': ['px.x'],
                'z': ['pz.z'],
                'y1': ['d1.y1'],
                'y2': ['d2.y2'],
                'obj': ['obj_cmp.obj'],
                'con1': ['con_cmp1.con1'],
                'con2': ['con_cmp2.con2']
            }
        }

        abs2prom = {
            'input': {
                'd1.z': 'z',
                'd1.x': 'x',
                'd1.y2': 'y2',
                'd2.z': 'z',
                'd2.y1': 'y1',
                'obj_cmp.x': 'x',
                'obj_cmp.y1': 'y1',
                'obj_cmp.y2': 'y2',
                'obj_cmp.z': 'z',
                'con_cmp1.y1': 'y1',
                'con_cmp2.y2': 'y2'
            },
            'output': {
                'px.x': 'x',
                'pz.z': 'z',
                'd1.y1': 'y1',
                'd2.y2': 'y2',
                'obj_cmp.obj': 'obj',
                'con_cmp1.con1': 'con1',
                'con_cmp2.con2': 'con2'
            }
        }

        assertMetadataRecorded(self, prom2abs, abs2prom)
        expected_driver_metadata = {
            'connections_list_length': 11,
            'tree_length': 4,
            'tree_children_length': 7,
        }
        assertDriverMetadataRecorded(self, expected_driver_metadata)

    def test_driver_without_n2_data(self):
        prob = SellarProblem()

        prob.driver.recording_options['record_n2_data'] = False
        prob.driver.add_recorder(self.recorder)

        prob.setup()
        prob.final_setup()  # Conclude setup but don't run model.
        prob.cleanup()

        assertDriverMetadataRecorded(self, None, True)

    def test_driver_doesnt_record_metadata(self):
        prob = SellarProblem()

        prob.driver.recording_options['record_metadata'] = False
        prob.driver.add_recorder(self.recorder)

        prob.setup()
        prob.final_setup()  # Conclude setup but don't run model.
        prob.cleanup()

        assertDriverMetadataRecorded(self, None)

    @unittest.skipIf(PETScVector is None, "PETSc is required.")
    # @unittest.skipIf(os.environ.get("TRAVIS"), "Unreliable on Travis CI.")
    def test_record_system(self):
        prob = SellarProblem()
        prob.setup()

        model = prob.model
        model.recording_options['record_inputs'] = True
        model.recording_options['record_outputs'] = True
        model.recording_options['record_residuals'] = True
        model.recording_options['record_metadata'] = True
        model.add_recorder(self.recorder)

        d1 = model.d1  # SellarDis1withDerivatives, an ExplicitComp
        d1.recording_options['record_inputs'] = True
        d1.recording_options['record_outputs'] = True
        d1.recording_options['record_residuals'] = True
        d1.recording_options['record_metadata'] = True
        d1.add_recorder(self.recorder)

        obj_cmp = model.obj_cmp  # an ExecComp
        obj_cmp.recording_options['record_inputs'] = True
        obj_cmp.recording_options['record_outputs'] = True
        obj_cmp.recording_options['record_residuals'] = True
        obj_cmp.recording_options['record_metadata'] = True
        obj_cmp.add_recorder(self.recorder)

        t0, t1 = run_driver(prob)
        prob.cleanup()

        expected_data = [
            # data from 'd1'
            [
                # coords
                [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ),
                 'NonlinearBlockGS', (6, ), 'd1._solve_nonlinear', (6, )],
                # timestamps
                (t0, t1),
                # inputs
                {"d1.y2": [12.05848815], "d1.z": [5.0, 2.0], "d1.x": [1.0, ]},
                # outputs
                {"d1.y1": [25.58830237]},
                # residuals
                {"d1.y1": [0.0]}
            ],

            # data from 'obj_cmp'
            [
                # coords
                [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ),
                 'NonlinearBlockGS', (6, ), 'obj_cmp._solve_nonlinear', (6, )],
                # timestamps
                (t0, t1),
                # inputs
                {"obj_cmp.z": [5.0, 2.0],
                 "obj_cmp.y1": [25.58830236],
                 "obj_cmp.x": [1.0, ],
                 "obj_cmp.y2": [12.05857185],
                },
                # outputs
                {"obj_cmp.obj": [28.58830816]},
                # residuals
                {"obj_cmp.obj": [0.0]}
            ],
        ]
        assertSystemIterDataRecorded(self, expected_data, self.eps)

        # run again with prefix, only changes should be iter count reset and timestamps
        t0, t1 = run_driver(prob, case_prefix='Run#2')
        prob.cleanup()

        expected_data[0][0] = [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ),
                               'NonlinearBlockGS', (0, ), 'd1._solve_nonlinear', (0, )]
        expected_data[0][1] = (t0, t1)

        expected_data[1][0] = [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ),
                               'NonlinearBlockGS', (0, ), 'obj_cmp._solve_nonlinear', (0, )]
        expected_data[1][1] = (t0, t1)

        assertSystemIterDataRecorded(self, expected_data, self.eps, prefix='Run#2')

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP")
    def test_includes(self):
        prob = ParaboloidProblem()

        driver = prob.driver = pyOptSparseDriver(optimizer='SLSQP')
        driver.options['print_results'] = False
        driver.opt_settings['ACC'] = 1e-9
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_responses'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['includes'] = ['*']
        driver.recording_options['excludes'] = ['p2*']
        driver.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)

        prob.cleanup()

        coordinate = [0, 'SLSQP', (4, )]

        expected_desvars = {"p1.x": prob["p1.x"]}
        expected_objectives = {"comp.f_xy": prob['comp.f_xy']}
        expected_constraints = {"con.c": prob['con.c']}

        expected_responses = expected_objectives.copy()
        expected_responses.update(expected_constraints)

        expected_outputs = expected_desvars
        expected_outputs.update(expected_objectives)
        expected_outputs.update(expected_constraints)

        expected_inputs = {
            "con.x": 7.1666667,
            "comp.y": -7.83333333,
            "comp.x": 7.1666667,
            "con.y": -7.8333333
        }

        expected_data = ((coordinate, (t0, t1), expected_outputs, expected_inputs),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP")
    def test_includes_post_setup(self):
        prob = ParaboloidProblem()

        driver = prob.driver = pyOptSparseDriver(optimizer='SLSQP')
        driver.options['print_results'] = False
        driver.opt_settings['ACC'] = 1e-9

        prob.setup()

        # Set up recorder after intitial setup.
        driver.add_recorder(self.recorder)
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_responses'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['includes'] = ['*']
        driver.recording_options['excludes'] = ['p2*']

        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'SLSQP', (4, )]

        expected_desvars = {"p1.x": prob["p1.x"]}
        expected_objectives = {"comp.f_xy": prob['comp.f_xy']}
        expected_constraints = {"con.c": prob['con.c']}

        expected_outputs = expected_desvars
        expected_outputs.update(expected_objectives)
        expected_outputs.update(expected_constraints)

        expected_inputs = {
            "con.x": 7.1666667,
            "comp.y": -7.83333333,
            "comp.x": 7.1666667,
            "con.y": -7.8333333
        }

        expected_data = ((coordinate, (t0, t1), expected_outputs, expected_inputs),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP")
    def test_record_system_with_hierarchy(self):
        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=NonlinearRunOnce)
        prob.setup(mode='rev')

        model = prob.model
        model.recording_options['record_inputs'] = True
        model.recording_options['record_outputs'] = True
        model.recording_options['record_residuals'] = True
        model.recording_options['record_metadata'] = True
        model.add_recorder(self.recorder)

        pz = model.pz  # IndepVarComp which is an ExplicitComponent
        pz.recording_options['record_inputs'] = True
        pz.recording_options['record_outputs'] = True
        pz.recording_options['record_residuals'] = True
        pz.recording_options['record_metadata'] = True
        pz.add_recorder(self.recorder)

        d1 = model.mda.d1
        d1.recording_options['record_inputs'] = True
        d1.recording_options['record_outputs'] = True
        d1.recording_options['record_residuals'] = True
        d1.recording_options['record_metadata'] = True
        d1.add_recorder(self.recorder)

        driver = prob.driver = pyOptSparseDriver(optimizer='SLSQP')
        driver.options['print_results'] = False
        driver.opt_settings['ACC'] = 1e-9

        t0, t1 = run_driver(prob)
        prob.cleanup()

        #
        # check data for 'd1'
        #
        coordinate = [0, 'SLSQP', (0, ), 'root._solve_nonlinear', (0, ), 'NLRunOnce', (0, ),
                      'mda._solve_nonlinear', (0, ), 'NonlinearBlockGS', (4,), 'mda.d1._solve_nonlinear', (4, )]

        expected_inputs = {
            "mda.d1.z": [5.0, 2.0],
            "mda.d1.x": [1.0, ],
            "mda.d1.y2": [12.0584865, ],
        }
        expected_outputs = {"mda.d1.y1": [25.5883027, ], }
        expected_residuals = {"mda.d1.y1": [0.0, ], }

        expected_data = ((coordinate, (t0, t1), expected_inputs, expected_outputs, expected_residuals),)
        assertSystemIterDataRecorded(self, expected_data, self.eps)

        #
        # check data for 'pz'
        #
        coordinate = [0, 'SLSQP', (1, ), 'root._solve_nonlinear', (1, ), 'NLRunOnce', (0, ),
                      'pz._solve_nonlinear', (1, )]

        expected_inputs = None
        expected_outputs = {"pz.z": [2.8640616, 0.825643, ], }
        expected_residuals = {"pz.z": [0.0, 0.0], }

        expected_data = ((coordinate, (t0, t1), expected_inputs, expected_outputs, expected_residuals),)
        assertSystemIterDataRecorded(self, expected_data, self.eps)

    def test_record_solver(self):
        prob = SellarProblem()
        prob.setup()

        nl = prob.model.nonlinear_solver
        nl.recording_options['record_abs_error'] = True
        nl.recording_options['record_rel_error'] = True
        nl.recording_options['record_solver_residuals'] = True
        nl.add_recorder(self.recorder)

        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ), 'NonlinearBlockGS', (6, )]

        expected_abs_error = 1.318802844707e-10
        expected_rel_error = 3.62990740e-12

        expected_solver_output = {
            "con_cmp1.con1": [-22.42830237],
            "d1.y1": [25.58830237],
            "con_cmp2.con2": [-11.941511849],
            "pz.z": [5.0, 2.0],
            "obj_cmp.obj": [28.588308165],
            "d2.y2": [12.058488150],
            "px.x": [1.0]
        }

        expected_solver_residuals = {
            "con_cmp1.con1": [0.0],
            "d1.y1": [1.318802844707534e-10],
            "con_cmp2.con2": [0.0],
            "pz.z": [0.0, 0.0],
            "obj_cmp.obj": [0.0],
            "d2.y2": [0.0],
            "px.x": [0.0]
        }

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps)

        #
        # run again with a prefix, iter_counts should be reset
        #
        t0, t1 = run_driver(prob, case_prefix='run_again')
        prob.cleanup()

        coordinate = [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ), 'NonlinearBlockGS', (0, )]

        expected_abs_error = 2.60769184e-12
        expected_rel_error = 0.01977317421405673

        expected_solver_residuals = {
            "con_cmp1.con1": [0.0],
            "d1.y1": [-2.60769184e-12],
            "con_cmp2.con2": [0.0],
            "pz.z": [0.0, 0.0],
            "obj_cmp.obj": [0.0],
            "d2.y2": [0.0],
            "px.x": [0.0]
        }

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps, prefix='run_again')

    def test_record_line_search_armijo_goldstein(self):
        prob = SellarProblem()
        prob.setup()

        model = prob.model
        model.linear_solver = ScipyKrylov()

        nl = model.nonlinear_solver = NewtonSolver()
        nl.options['solve_subsystems'] = True
        nl.options['max_sub_solves'] = 4

        ls = nl.linesearch = ArmijoGoldsteinLS(bound_enforcement='vector')
        ls.options['c'] = 100.0  # This is pretty bogus, but it ensures that we get a few LS iterations.
        ls.add_recorder(self.recorder)

        t0, t1 = run_driver(prob)

        prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (3,), 'ArmijoGoldsteinLS', (4,)]

        expected_abs_error = 3.49773898733e-9
        expected_rel_error = expected_abs_error / 2.9086436370e-08

        expected_solver_output = {
            "con_cmp1.con1": [-22.42830237],
            "d1.y1": [25.58830237],
            "con_cmp2.con2": [-11.941511849],
            "pz.z": [5.0, 2.0],
            "obj_cmp.obj": [28.58830816516],
            "d2.y2": [12.058488150],
            "px.x": [1.0]
        }

        expected_solver_residuals = None

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps)

    def test_record_line_search_bounds_enforce(self):
        prob = SellarProblem()
        prob.setup()

        model = prob.model
        model.linear_solver = ScipyKrylov()

        nl = model.nonlinear_solver = NewtonSolver()
        nl.options['solve_subsystems'] = True
        nl.options['max_sub_solves'] = 4

        ls = nl.linesearch = BoundsEnforceLS(bound_enforcement='vector')
        ls.add_recorder(self.recorder)

        t0, t1 = run_driver(prob)

        prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (1,), 'BoundsEnforceLS', (0,)]

        expected_abs_error = 7.02783609310096e-10
        expected_rel_error = 8.078674883382422e-07

        expected_solver_output = {
            "con_cmp1.con1": [-22.42830237],
            "d1.y1": [25.58830237],
            "con_cmp2.con2": [-11.941511849],
            "pz.z": [5.0, 2.0],
            "obj_cmp.obj": [28.588308165],
            "d2.y2": [12.058488150],
            "px.x": [1.0]
        }

        expected_solver_residuals = None

        expected_solver_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                                 expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_solver_data, self.eps)

    def test_record_solver_nonlinear_block_gs(self):
        prob = SellarProblem(linear_solver=LinearBlockGS, nonlinear_solver=NonlinearBlockGS)
        prob.setup()

        prob.model.nonlinear_solver.add_recorder(self.recorder)
        prob.model.nonlinear_solver.recording_options['record_solver_residuals'] = True

        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NonlinearBlockGS', (6, )]
        expected_abs_error = 1.31880284470753394998e-10
        expected_rel_error = 3.6299074030587596e-12

        expected_solver_output = {
            'px.x': [1.],
            'pz.z': [5., 2.],
            'd1.y1': [25.58830237],
            'd2.y2': [12.05848815],
            'obj_cmp.obj': [28.58830817],
            'con_cmp1.con1': [-22.42830237],
            'con_cmp2.con2': [-11.94151185]
        }

        expected_solver_residuals = {
            'px.x': [-0.],
            'pz.z': [-0., -0.],
            'd1.y1': [1.31880284e-10],
            'd2.y2': [0.],
            'obj_cmp.obj': [0.],
            'con_cmp1.con1': [0.],
            'con_cmp2.con2': [0.]
        }

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps)

    def test_record_solver_nonlinear_block_jac(self):
        prob = SellarProblem(linear_solver=LinearBlockGS, nonlinear_solver=NonlinearBlockJac)
        prob.setup()

        prob.model.nonlinear_solver.add_recorder(self.recorder)

        prob.set_solver_print(-1)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NonlinearBlockJac', (9,)]

        expected_abs_error = 7.234027587097439e-07
        expected_rel_error = 1.991112651729199e-08

        expected_solver_output = {
            'px.x': [1.],
            'pz.z': [5., 2.],
            'd1.y1': [25.58830237],
            'd2.y2': [12.05848815],
            'obj_cmp.obj': [28.58830817],
            'con_cmp1.con1': [-22.42830237],
            'con_cmp2.con2': [-11.94151185]
        }

        expected_solver_residuals = None

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps)

    def test_record_solver_nonlinear_newton(self):
        prob = SellarProblem(linear_solver=LinearBlockGS, nonlinear_solver=NewtonSolver)
        prob.setup()

        prob.model.nonlinear_solver.add_recorder(self.recorder)

        prob.set_solver_print(-1)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (2,)]

        expected_abs_error = 2.1677810075550974e-10
        expected_rel_error = 5.966657077752565e-12

        expected_solver_output = {
            'px.x': [1.],
            'pz.z': [5., 2.],
            'd1.y1': [25.58830237],
            'd2.y2': [12.05848815],
            'obj_cmp.obj': [28.58830817],
            'con_cmp1.con1': [-22.42830237],
            'con_cmp2.con2': [-11.94151185]
        }

        expected_solver_residuals = None

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps)

    def test_record_solver_nonlinear_nonlinear_run_once(self):
        prob = SellarProblem(nonlinear_solver=NonlinearRunOnce)
        prob.setup()

        prob.model.nonlinear_solver.add_recorder(self.recorder)

        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        # No norms so no expected norms
        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NLRunOnce', (0,)]

        expected_abs_error = None
        expected_rel_error = None

        expected_solver_output = {
            'px.x': [1.],
            'pz.z': [5., 2.],
            'd1.y1': [27.8],
            'd2.y2': [12.27257053],
            'obj_cmp.obj': [30.80000468],
            'con_cmp1.con1': [-24.64],
            'con_cmp2.con2': [-11.72742947]
        }

        expected_solver_residuals = None

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps)

    def test_record_solver_linear_direct_solver(self):
        prob = SellarProblem()
        prob.setup()

        nl = prob.model.nonlinear_solver = NewtonSolver()

        ln = nl.linear_solver = DirectSolver()  # used for analytic derivatives
        ln.recording_options['record_abs_error'] = True
        ln.recording_options['record_rel_error'] = True
        ln.recording_options['record_solver_residuals'] = True
        ln.add_recorder(self.recorder)

        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        # No norms so no expected norms
        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (2,), 'DirectSolver', (0,)]

        expected_abs_error = None
        expected_rel_error = None

        expected_solver_output = {
            'px.x': [0.],
            'pz.z': [0.0, 0.00000000e+00],
            'd1.y1': [0.00045069],
            'd2.y2': [-0.00225346],
            'obj_cmp.obj': [0.00045646],
            'con_cmp1.con1': [-0.00045069],
            'con_cmp2.con2': [-0.00225346]
        }

        expected_solver_residuals = {
            'px.x': [0.],
            'pz.z': [-0., -0.],
            'd1.y1': [0.0],
            'd2.y2': [-0.00229801],
            'obj_cmp.obj': [5.75455956e-06],
            'con_cmp1.con1': [-0.],
            'con_cmp2.con2': [-0.]
        }

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self,expected_data, self.eps)

    def test_record_solver_linear_scipy_iterative_solver(self):
        prob = SellarProblem()
        prob.setup()

        nl = prob.model.nonlinear_solver = NewtonSolver()

        ln = nl.linear_solver = ScipyKrylov()  # used for analytic derivatives
        ln.recording_options['record_abs_error'] = True
        ln.recording_options['record_rel_error'] = True
        ln.recording_options['record_solver_residuals'] = True
        ln.add_recorder(self.recorder)

        prob.set_solver_print(0)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (2,), 'ScipyKrylov', (1,)]

        expected_abs_error = 0.0
        expected_rel_error = 0.0

        expected_solver_output = {
            'px.x': [0.],
            'pz.z': [0., 0.],
            'd1.y1': [0.0],
            'd2.y2': [-0.41168147],
            'obj_cmp.obj': [-0.48667678],
            'con_cmp1.con1': [0.770496],
            'con_cmp2.con2': [-2.70578793e-06]
        }

        expected_solver_residuals = {
            'px.x': [0.],
            'pz.z': [0., 0.],
            'd1.y1': [-0.08233575],
            'd2.y2': [-0.41168152],
            'obj_cmp.obj': [-0.4866797],
            'con_cmp1.con1': [0.77049654],
            'con_cmp2.con2': [0.41167877]
        }

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps)

    @unittest.skipIf(PETScVector is None, "PETSc is required.")
    @unittest.skipIf(os.environ.get("TRAVIS"), "Unreliable on Travis CI.")
    def test_record_solver_linear_petsc_ksp(self):
        prob = SellarProblem()
        prob.setup()

        nl = prob.model.nonlinear_solver = NewtonSolver()

        ln = nl.linear_solver = PETScKrylov()  # used for analytic derivatives
        ln.recording_options['record_abs_error'] = True
        ln.recording_options['record_rel_error'] = True
        ln.recording_options['record_solver_residuals'] = True
        ln.add_recorder(self.recorder)

        t0, t1 = run_driver(prob)

        prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (2,), 'PETScKrylov', (3,)]

        expected_abs_error = 0.0
        expected_rel_error = 0.0

        expected_solver_output = {
            'px.x': [0.],
            'pz.z': [0., 0.],
            'd1.y1': [5.41157587e-07],
            'd2.y2': [-0.41168147],
            'obj_cmp.obj': [-0.48667678],
            'con_cmp1.con1': [0.770496],
            'con_cmp2.con2': [-2.70578793e-06]
        }

        expected_solver_residuals = {
            'px.x': [0.],
            'pz.z': [0., 0.],
            'd1.y1': [-0.08233575],
            'd2.y2': [-0.41168152],
            'obj_cmp.obj': [-0.4866797],
            'con_cmp1.con1': [0.77049654],
            'con_cmp2.con2': [0.41167877]
        }

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps)

    def test_record_solver_linear_block_gs(self):
        prob = SellarProblem()
        prob.setup()

        nl = prob.model.nonlinear_solver = NewtonSolver()

        ln = nl.linear_solver = LinearBlockGS()  # used for analytic derivatives
        ln.recording_options['record_abs_error'] = True
        ln.recording_options['record_rel_error'] = True
        ln.recording_options['record_solver_residuals'] = True
        ln.add_recorder(self.recorder)

        prob.set_solver_print(-1)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (2,), 'LinearBlockGS', (6,)]

        expected_abs_error = 9.109083208861876e-11
        expected_rel_error = 9.114367543620551e-12

        expected_solver_output = {
            'px.x': [-0.],
            'pz.z': [-0., -0.],
            'd1.y1': [0.00045069],
            'd2.y2': [-0.00225346],
            'obj_cmp.obj': [0.00045646],
            'con_cmp1.con1': [-0.00045069],
            'con_cmp2.con2': [-0.00225346]
        }

        expected_solver_residuals = {
            'px.x': [0.],
            'pz.z': [0., 0.],
            'd1.y1': [9.10908321e-11],
            'd2.y2': [0.],
            'obj_cmp.obj': [-2.03287907e-20],
            'con_cmp1.con1': [0.],
            'con_cmp2.con2': [0.]
        }

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps)

    def test_record_solver_linear_linear_run_once(self):
        prob = SellarProblem()
        prob.setup()

        nl = prob.model.nonlinear_solver = NewtonSolver()

        ln = nl.linear_solver = LinearRunOnce() # used for analytic derivatives
        ln.recording_options['record_abs_error'] = True
        ln.recording_options['record_rel_error'] = True
        ln.recording_options['record_solver_residuals'] = True
        ln.add_recorder(self.recorder)

        prob.set_solver_print(-1)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (9,), 'LinearRunOnce', (0,)]

        expected_abs_error = None
        expected_rel_error = None

        expected_solver_output = {
            'px.x': [-0.],
            'pz.z': [-0., -0.],
            'd1.y1': [-4.15366975e-05],
            'd2.y2': [-4.10568454e-06],
            'obj_cmp.obj': [-4.15366737e-05],
            'con_cmp1.con1': [4.15366975e-05],
            'con_cmp2.con2': [-4.10568454e-06]
        }

        expected_solver_residuals = {
            'px.x': [-0.],
            'pz.z': [0., 0.],
            'd1.y1': [-4.15366975e-05],
            'd2.y2': [4.10564051e-06],
            'obj_cmp.obj': [-4.15366737e-05],
            'con_cmp1.con1': [-4.15366975e-05],
            'con_cmp2.con2': [-4.10568454e-06]
        }

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps)

    def test_record_solver_linear_block_jac(self):
        prob = SellarProblem()
        prob.setup()

        nl = prob.model.nonlinear_solver = NewtonSolver()

        ln = nl.linear_solver = LinearBlockJac() # used for analytic derivatives
        ln.recording_options['record_abs_error'] = True
        ln.recording_options['record_rel_error'] = True
        ln.recording_options['record_solver_residuals'] = True
        ln.add_recorder(self.recorder)

        prob.set_solver_print(-1)
        t0, t1 = run_driver(prob)
        prob.cleanup()

        coordinate = [0, 'Driver', (0,), 'root._solve_nonlinear', (0,), 'NewtonSolver', (3,), 'LinearBlockJac', (9,)]

        expected_abs_error = 9.947388408259769e-11
        expected_rel_error = 4.330301334141486e-08

        expected_solver_output = {
            'px.x': [-0.],
            'pz.z': [-0., -0.],
            'd1.y1': [4.55485639e-09],
            'd2.y2': [-2.27783334e-08],
            'obj_cmp.obj': [-2.28447051e-07],
            'con_cmp1.con1': [2.28461863e-07],
            'con_cmp2.con2': [-2.27742837e-08]
        }

        expected_solver_residuals = {
            'px.x': [-0.],
            'pz.z': [0., 0.],
            'd1.y1': [-2.84055951e-16],
            'd2.y2': [6.93561782e-12],
            'obj_cmp.obj': [7.01674811e-11],
            'con_cmp1.con1': [-7.01674811e-11],
            'con_cmp2.con2': [1.42027975e-15]
        }

        expected_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                          expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_data, self.eps)

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP")
    def test_record_driver_system_solver(self):
        # Test what happens when all three types are recorded: Driver, System, and Solver

        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=NonlinearRunOnce)
        prob.setup(mode='rev')

        driver = prob.driver = pyOptSparseDriver(optimizer='SLSQP')
        driver.options['print_results'] = False
        driver.opt_settings['ACC'] = 1e-9

        #
        # Add recorders
        #

        # Driver
        driver.recording_options['includes'] = []
        driver.recording_options['record_metadata'] = True
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_responses'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.add_recorder(self.recorder)

        # System
        pz = prob.model.pz  # IndepVarComp which is an ExplicitComponent
        pz.recording_options['record_metadata'] = True
        pz.recording_options['record_inputs'] = True
        pz.recording_options['record_outputs'] = True
        pz.recording_options['record_residuals'] = True
        pz.add_recorder(self.recorder)

        # Solver
        nl = prob.model.mda.nonlinear_solver = NonlinearBlockGS()
        nl.recording_options['record_metadata'] = True
        nl.recording_options['record_abs_error'] = True
        nl.recording_options['record_rel_error'] = True
        nl.recording_options['record_solver_residuals'] = True
        nl.add_recorder(self.recorder)

        t0, t1 = run_driver(prob)

        prob.cleanup()

        #
        # Driver recording test
        #
        coordinate = [0, 'SLSQP', (6, )]

        expected_desvars = {
            "pz.z": prob['pz.z'],
            "px.x": prob['px.x']
        }
        expected_objectives = {
            "obj_cmp.obj": prob['obj_cmp.obj']
        }
        expected_constraints = {
            "con_cmp1.con1": prob['con_cmp1.con1'],
            "con_cmp2.con2": prob['con_cmp2.con2']
        }

        expected_outputs = expected_desvars
        expected_outputs.update(expected_objectives)
        expected_outputs.update(expected_constraints)

        expected_driver_data = ((coordinate, (t0, t1), expected_outputs, None),)
        assertDriverIterDataRecorded(self, expected_driver_data, self.eps)

        #
        # System recording test
        #
        coordinate = [0, 'SLSQP', (1, ), 'root._solve_nonlinear', (1, ), 'NLRunOnce', (0, ),
                      'pz._solve_nonlinear', (1, )]

        expected_inputs = None
        expected_outputs = {"pz.z": [2.8640616, 0.825643, ], }
        expected_residuals = {"pz.z": [0.0, 0.0], }

        expected_system_data = (
            (coordinate, (t0, t1), expected_inputs, expected_outputs, expected_residuals),
        )
        assertSystemIterDataRecorded(self, expected_system_data, self.eps)

        #
        # Solver recording test
        #
        coordinate = [0, 'SLSQP', (5, ), 'root._solve_nonlinear', (5, ), 'NLRunOnce', (0, ),
                      'mda._solve_nonlinear', (5, ), 'NonlinearBlockGS', (3, )]

        expected_abs_error = 0.0,
        expected_rel_error = 0.0,

        expected_solver_output = {
            "mda.d2.y2": [3.75610187],
            "mda.d1.y1": [3.16],
        }

        expected_solver_residuals = {
            "mda.d2.y2": [0.0],
            "mda.d1.y1": [0.0],
        }

        expected_solver_data = ((coordinate, (t0, t1), expected_abs_error, expected_rel_error,
                                 expected_solver_output, expected_solver_residuals),)
        assertSolverIterDataRecorded(self, expected_solver_data, self.eps)

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP")
    def test_global_counter(self):
        # The case recorder maintains a global counter across all recordings

        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=NonlinearRunOnce)
        prob.setup(mode='rev')

        driver = prob.driver = pyOptSparseDriver(optimizer='SLSQP')
        driver.options['print_results'] = False
        driver.opt_settings['ACC'] = 1e-2  # to speed the test up
        driver.opt_settings['ACC'] = 1e-9

        # Add recorders for Driver, System, Solver
        driver.add_recorder(self.recorder)
        prob.model.add_recorder(self.recorder)
        prob.model.mda.nonlinear_solver.add_recorder(self.recorder)

        t0, t1 = run_driver(prob)

        prob.cleanup()

        # get global counter values from driver, system, and solver recording
        con = sqlite3.connect(self.filename)
        cur = con.cursor()
        cur.execute("SELECT counter FROM driver_iterations")
        counters_driver = set(i[0] for i in cur.fetchall())
        cur.execute("SELECT counter FROM system_iterations")
        counters_system = set(i[0] for i in cur.fetchall())
        cur.execute("SELECT counter FROM solver_iterations")
        counters_solver = set(i[0] for i in cur.fetchall())
        cur.execute("SELECT COUNT(rowid) FROM global_iterations")
        global_iterations_records = cur.fetchone()[0]
        con.close()

        # Check to see that they make sense
        self.assertEqual(self.recorder._counter, global_iterations_records)
        self.assertEqual(self.recorder._counter, len(counters_driver) + len(counters_system) +
                         len(counters_solver))
        self.assertTrue(counters_driver.isdisjoint(counters_system))
        self.assertTrue(counters_driver.isdisjoint(counters_solver))
        self.assertTrue(counters_system.isdisjoint(counters_solver))

    def test_implicit_component(self):
        from openmdao.core.tests.test_impl_comp import QuadraticLinearize, QuadraticJacVec

        indeps = IndepVarComp()
        indeps.add_output('a', 1.0)
        indeps.add_output('b', 1.0)
        indeps.add_output('c', 1.0)

        group = Group()
        group.add_subsystem('comp1', indeps)
        group.add_subsystem('comp2', QuadraticLinearize())
        group.add_subsystem('comp3', QuadraticJacVec())
        group.connect('comp1.a', 'comp2.a')
        group.connect('comp1.b', 'comp2.b')
        group.connect('comp1.c', 'comp2.c')
        group.connect('comp1.a', 'comp3.a')
        group.connect('comp1.b', 'comp3.b')
        group.connect('comp1.c', 'comp3.c')

        prob = Problem(model=group)
        prob.setup()

        prob['comp1.a'] = 1.
        prob['comp1.b'] = -4.
        prob['comp1.c'] = 3.

        comp2 = prob.model.comp2  # ImplicitComponent
        comp2.recording_options['record_metadata'] = False
        comp2.add_recorder(self.recorder)

        t0, t1 = run_driver(prob)

        prob.cleanup()

        coordinate = [0, 'Driver', (0, ), 'root._solve_nonlinear', (0, ), 'NLRunOnce', (0, ),
                      'comp2._solve_nonlinear', (0, )]

        expected_inputs = {
            "comp2.a": [1.0, ],
            "comp2.b": [-4.0, ],
            "comp2.c": [3.0, ],
        }
        expected_outputs = {"comp2.x": [3.0, ], }
        expected_residuals = {"comp2.x": [0.0, ], }

        expected_data = ((coordinate, (t0, t1), expected_inputs, expected_outputs, expected_residuals),)
        assertSystemIterDataRecorded(self, expected_data, self.eps)

    def test_multidimensional_arrays(self):
        # component TestExplCompArray, put in a model and run it; its outputs are multi-d-arrays.
        from openmdao.test_suite.components.expl_comp_array import TestExplCompArray
        comp = TestExplCompArray(thickness=1.)
        prob = Problem(comp).setup()

        prob['lengths'] = 3.
        prob['widths'] = 2.

        comp.add_recorder(self.recorder)
        comp.recording_options['record_inputs'] = True
        comp.recording_options['record_outputs'] = True
        comp.recording_options['record_residuals'] = True
        comp.recording_options['record_metadata'] = False

        t0, t1 = run_driver(prob)

        prob.run_model()

        prob.cleanup()

        # coordinate = rank0:._solve_nonlinear | 0
        coordinate = [0, 'Driver', (0,), '._solve_nonlinear', (0,)]

        expected_inputs = {
            'lengths': [[3.,  3.], [3., 3.]],
            'widths': [[2.,  2.], [2., 2.]],
        }

        expected_outputs = {
            'total_volume': [24.],
            'areas': [[6., 6.], [6., 6.]],
        }

        expected_residuals = {
            'total_volume': [0.],
            'areas': [[0., 0.], [0., 0.]],
        }

        expected_data = ((coordinate, (t0, t1), expected_inputs, expected_outputs, expected_residuals),)
        assertSystemIterDataRecorded(self, expected_data, self.eps)

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP")
    def test_record_system_recursively(self):
        # Test adding recorders to all Systems using the recurse option to add_recorder

        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=NonlinearRunOnce)
        prob.setup(mode='rev')

        # Need to do recursive adding of recorders AFTER setup
        prob.model.add_recorder(self.recorder, recurse=True)

        driver = prob.driver = pyOptSparseDriver(optimizer='SLSQP')
        driver.options['print_results'] = False
        driver.opt_settings['ACC'] = 1e-9

        t0, t1 = run_driver(prob)
        prob.cleanup()

        # Just make sure all Systems had some metadata recorded
        assertSystemMetadataIdsRecorded(self, [
            'root',
            'px',
            'pz',
            'mda',
            'mda.d1',
            'mda.d2',
            'obj_cmp',
            'con_cmp1',
            'con_cmp2'
        ])

        # Make sure all the Systems are recorded at least once
        assertSystemIterCoordsRecorded(self, [
            'rank0:SLSQP|0|root._solve_nonlinear|0',
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0|con_cmp1._solve_nonlinear|0',
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0|con_cmp2._solve_nonlinear|0',
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0',
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|0|mda.d1._solve_nonlinear|0',
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|0|mda.d2._solve_nonlinear|0',
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0|obj_cmp._solve_nonlinear|0',
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0|px._solve_nonlinear|0',
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0|pz._solve_nonlinear|0',
        ])

    def test_record_system_with_prefix(self):
        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=NonlinearRunOnce)
        prob.setup(mode='rev')

        prob.model.add_recorder(self.recorder, recurse=True)

        prob.run_model('Run1')
        prob.run_model('Run2')
        prob.cleanup()

        # Just make sure all Systems had some metadata recorded
        assertSystemMetadataIdsRecorded(self, [
            'root',
            'px',
            'pz',
            'mda',
            'mda.d1',
            'mda.d2',
            'obj_cmp',
            'con_cmp1',
            'con_cmp2'
        ])

        # Make sure all the Systems are recorded at least once
        assertSystemIterCoordsRecorded(self, [
            'Run1_rank0:root._solve_nonlinear|0',
            'Run1_rank0:root._solve_nonlinear|0|NLRunOnce|0|con_cmp1._solve_nonlinear|0',
            'Run1_rank0:root._solve_nonlinear|0|NLRunOnce|0|con_cmp2._solve_nonlinear|0',
            'Run1_rank0:root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0',
            'Run1_rank0:root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|0|mda.d1._solve_nonlinear|0',
            'Run1_rank0:root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|0|mda.d2._solve_nonlinear|0',
            'Run1_rank0:root._solve_nonlinear|0|NLRunOnce|0|obj_cmp._solve_nonlinear|0',
            'Run1_rank0:root._solve_nonlinear|0|NLRunOnce|0|px._solve_nonlinear|0',
            'Run1_rank0:root._solve_nonlinear|0|NLRunOnce|0|pz._solve_nonlinear|0',

            'Run2_rank0:root._solve_nonlinear|0',
            'Run2_rank0:root._solve_nonlinear|0|NLRunOnce|0|con_cmp1._solve_nonlinear|0',
            'Run2_rank0:root._solve_nonlinear|0|NLRunOnce|0|con_cmp2._solve_nonlinear|0',
            'Run2_rank0:root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0',
            'Run2_rank0:root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|0|mda.d1._solve_nonlinear|0',
            'Run2_rank0:root._solve_nonlinear|0|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|0|mda.d2._solve_nonlinear|0',
            'Run2_rank0:root._solve_nonlinear|0|NLRunOnce|0|obj_cmp._solve_nonlinear|0',
            'Run2_rank0:root._solve_nonlinear|0|NLRunOnce|0|px._solve_nonlinear|0',
            'Run2_rank0:root._solve_nonlinear|0|NLRunOnce|0|pz._solve_nonlinear|0',
        ])

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP")
    def test_driver_recording_with_system_vars(self):
        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=NonlinearRunOnce)

        driver = prob.driver = pyOptSparseDriver(optimizer='SLSQP')
        driver.options['print_results'] = False
        driver.opt_settings['ACC'] = 1e-9
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_responses'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['record_inputs'] = False
        driver.recording_options['includes'] = ['mda.d2.y2',]
        driver.add_recorder(self.recorder)

        prob.setup()

        t0, t1 = run_driver(prob)

        prob.cleanup()

        # Driver recording test
        coordinate = [0, 'SLSQP', (6, )]

        expected_desvars = {
            "pz.z": prob['pz.z'],
            "px.x": prob['px.x']
        }
        expected_objectives = {
            "obj_cmp.obj": prob['obj_cmp.obj']
        }

        expected_constraints = {
            "con_cmp1.con1": prob['con_cmp1.con1'],
            "con_cmp2.con2": prob['con_cmp2.con2']
        }

        expected_sysincludes = {
            'mda.d2.y2': prob['mda.d2.y2']
        }

        expected_outputs = expected_desvars
        expected_outputs.update(expected_objectives)
        expected_outputs.update(expected_constraints)
        expected_outputs.update(expected_sysincludes)

        expected_data = ((coordinate, (t0, t1), expected_outputs, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)

    def test_recorder_file_already_exists_no_append(self):
        prob = SellarProblem()

        driver = prob.driver
        driver.recording_options['record_metadata'] = True
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_responses'] = False
        driver.recording_options['record_objectives'] = False
        driver.recording_options['record_constraints'] = False
        driver.recording_options['includes'] = []
        driver.add_recorder(self.recorder)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        # Open up a new instance of the recorder but with the same filename
        prob = SellarProblem()

        driver = prob.driver
        driver.recording_options['record_metadata'] = True
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_responses'] = False
        driver.recording_options['record_objectives'] = False
        driver.recording_options['record_constraints'] = False
        driver.recording_options['includes'] = []
        driver.add_recorder(SqliteRecorder(self.filename))

        prob.setup()
        t0, t1 = run_driver(prob)
        prob.cleanup()

        # Do a simple test to see if recording second time was OK
        coordinate = [0, 'Driver', (0, )]

        expected_outputs = {"px.x": [1.0, ], "pz.z": [5.0, 2.0]}

        expected_data = ((coordinate, (t0, t1), expected_outputs, None),)
        assertDriverIterDataRecorded(self, expected_data, self.eps)


class TestFeatureSqliteRecorder(unittest.TestCase):
    def setUp(self):
        self.dir = mkdtemp()
        self.original_path = os.getcwd()
        os.chdir(self.dir)

    def tearDown(self):
        os.chdir(self.original_path)
        try:
            rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def test_feature_simple_driver_recording(self):
        from openmdao.api import Problem, Group, IndepVarComp, ExecComp, \
            ScipyOptimizeDriver, SqliteRecorder, CaseReader
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = Problem()

        model = prob.model
        model.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', ExecComp('c = - x + y'), promotes=['*'])

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', upper=-15.0)

        driver = prob.driver = ScipyOptimizeDriver()
        driver.options['optimizer'] = 'SLSQP'
        driver.options['tol'] = 1e-9

        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_responses'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True

        case_recorder_filename = 'cases.sql'

        recorder = SqliteRecorder(case_recorder_filename)
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = CaseReader(case_recorder_filename)
        case = cr.driver_cases.get_case('rank0:SLSQP|3')

        assert_rel_error(self, case.outputs['x'], 7.16666667, 1e-6)
        assert_rel_error(self, case.outputs['y'], -7.83333333, 1e-6)

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP")
    def test_feature_driver_metadata(self):
        from openmdao.api import Problem, SqliteRecorder, CaseReader
        from openmdao.test_suite.components.sellar import SellarDerivatives
        from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver

        prob = Problem(SellarDerivatives())

        prob.model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                       upper=np.array([10.0, 10.0]))
        prob.model.add_design_var('x', lower=0.0, upper=10.0)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0.0)
        prob.model.add_constraint('con2', upper=0.0)

        prob.driver = pyOptSparseDriver(optimizer='SLSQP')
        prob.driver.options['print_results'] = False

        # make sure we record metadata
        prob.driver.recording_options['record_metadata'] = True

        recorder = SqliteRecorder("cases.sql")
        prob.driver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = CaseReader("cases.sql")

        # access list of connections stored in metadata
        self.assertEqual(len(cr.driver_metadata['connections_list']), 11)

        # access the model tree stored in metadata
        self.assertEqual(len(cr.driver_metadata['tree']), 4)

    def test_feature_solver_metadata(self):
        from openmdao.api import Problem, SqliteRecorder, CaseReader, IndepVarComp, ExecComp,\
            NonlinearBlockGS, LinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives,\
            SellarDis2withDerivatives
        import numpy as np

        prob = Problem()

        model = prob.model
        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])
        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])
        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                            z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = NonlinearBlockGS()
        model.linear_solver = LinearBlockGS()

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        # create recorder
        recorder = SqliteRecorder("cases.sql")

        # add recorder to the nonlinear solver
        nonlinear_solver = model.nonlinear_solver
        nonlinear_solver.add_recorder(recorder)

        # add recorder to the linear solver as well
        linear_solver = model.linear_solver
        linear_solver.add_recorder(recorder)

        # add recorder to the nonlinear solver for Component 'd1' (SellarDis1withDerivatives)
        d1 = prob.model.d1
        d1.nonlinear_solver = NonlinearBlockGS()
        d1.nonlinear_solver.options['maxiter'] = 5
        d1.nonlinear_solver.add_recorder(recorder)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = CaseReader("cases.sql")

        self.assertEqual(
            sorted(cr.solver_metadata.keys()),
            sorted(['root.LinearBlockGS', 'root.NonlinearBlockGS', 'd1.NonlinearBlockGS'])
        )
        self.assertEqual(cr.solver_metadata['d1.NonlinearBlockGS']['solver_options']['maxiter'], 5)
        self.assertEqual(cr.solver_metadata['root.NonlinearBlockGS']['solver_options']['maxiter'],10)
        self.assertEqual(cr.solver_metadata['root.LinearBlockGS']['solver_class'],'LinearBlockGS')

    def test_feature_system_metadata(self):
        from openmdao.api import Problem, SqliteRecorder, CaseReader, IndepVarComp, ExecComp,\
            NonlinearBlockGS, LinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives,\
            SellarDis2withDerivatives
        import numpy as np

        prob = Problem()

        model = prob.model
        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])
        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])
        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                            z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = NonlinearBlockGS()
        model.linear_solver = LinearBlockGS()

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        nonlinear_solver = prob.model.nonlinear_solver
        linear_solver = prob.model.linear_solver

        d1 = prob.model.d1  # SellarDis1withDerivatives
        d1.nonlinear_solver = NonlinearBlockGS()
        d1.nonlinear_solver.options['maxiter'] = 5

        # declare two options
        d1.options.declare('options value 1', 1)
        d1.options.declare('options value to ignore', 2)

        # create recorder and attach to d1
        recorder = SqliteRecorder("cases.sql")
        d1.add_recorder(recorder)

        # don't record the second option on d1
        d1.recording_options['options_excludes'] = ['options value to ignore']

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = CaseReader("cases.sql")

        d1_options = cr.system_metadata['d1']['component_options']

        # option 1 is recorded
        self.assertEqual(d1_options['options value 1'], 1)

        # option 2 is not recorded
        self.assertFalse('options value to ignore' in d1_options)

    def test_feature_system_options(self):
        import numpy as np

        from openmdao.api import Problem, IndepVarComp, ExecComp, NonlinearBlockGS, SqliteRecorder, CaseReader
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = Problem()
        model = prob.model

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])
        model.nonlinear_solver = NonlinearBlockGS()

        obj_cmp = model.obj_cmp

        recorder = SqliteRecorder("cases.sql")
        obj_cmp.add_recorder(recorder)
        obj_cmp.recording_options['includes'] = ['*']
        obj_cmp.recording_options['excludes'] = ['obj_cmp.x']

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = CaseReader("cases.sql")
        first_system_case = cr.system_cases.get_case(0)
        recorded_inputs = first_system_case.inputs.keys
        self.assertEqual(set(recorded_inputs), {'y2', 'y1', 'z'})

    def test_feature_driver_options(self):
        from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder, CaseReader
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem()
        prob.model = SellarDerivatives()
        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.driver = ScipyOptimizeDriver()
        driver = prob.driver
        driver.options['optimizer'] = 'SLSQP'
        driver.options['tol'] = 1e-9
        driver.options['disp'] = False

        recorder = SqliteRecorder("cases.sql")
        driver.add_recorder(recorder)
        driver.recording_options['includes'] = []
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['record_desvars'] = True

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = CaseReader("cases.sql")
        first_driver_case = cr.driver_cases.get_case(0)
        recorded_objectives = first_driver_case.get_objectives().keys
        recorded_constraints = first_driver_case.get_constraints().keys
        recorded_desvars = first_driver_case.get_desvars().keys

        self.assertEqual(set(recorded_objectives), {'obj'})
        self.assertEqual(set(recorded_constraints), {'con1', 'con2'})
        self.assertEqual(set(recorded_desvars), {'x', 'z'})

    def test_feature_solver_options(self):
        import numpy as np

        from openmdao.api import Problem, IndepVarComp, ExecComp, NonlinearBlockGS, SqliteRecorder, CaseReader
        from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

        prob = Problem()

        model = prob.model
        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                z=np.array([0.0, 0.0]), x=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        model.nonlinear_solver = NonlinearBlockGS()
        model.nonlinear_solver.recording_options['record_abs_error'] = True

        recorder = SqliteRecorder("cases.sql")
        model.nonlinear_solver.add_recorder(recorder)

        prob.setup()
        prob.run_model()
        prob.cleanup()

        cr = CaseReader("cases.sql")
        first_solver_case = cr.solver_cases.get_case(0)

        recorded_abs_error = first_solver_case.abs_err
        self.assertAlmostEqual(recorded_abs_error, 2.2545141)

    def test_feature_circuit_with_recorder(self):
        from openmdao.api import Group, NewtonSolver, DirectSolver, Problem, IndepVarComp, CaseReader, SqliteRecorder
        from openmdao.test_suite.test_examples.test_circuit_analysis import Resistor, Diode, Node

        class Circuit(Group):

            def setup(self):
                self.add_subsystem('n1', Node(n_in=1, n_out=2), promotes_inputs=[('I_in:0', 'I_in')])
                self.add_subsystem('n2', Node())  # leaving defaults

                self.add_subsystem('R1', Resistor(R=100.), promotes_inputs=[('V_out', 'Vg')])
                self.add_subsystem('R2', Resistor(R=10000.))
                self.add_subsystem('D1', Diode(), promotes_inputs=[('V_out', 'Vg')])

                self.connect('n1.V', ['R1.V_in', 'R2.V_in'])
                self.connect('R1.I', 'n1.I_out:0')
                self.connect('R2.I', 'n1.I_out:1')

                self.connect('n2.V', ['R2.V_out', 'D1.V_in'])
                self.connect('R2.I', 'n2.I_in:0')
                self.connect('D1.I', 'n2.I_out:0')

                self.nonlinear_solver = NewtonSolver()
                self.nonlinear_solver.options['iprint'] = 2
                self.nonlinear_solver.options['maxiter'] = 20
                self.linear_solver = DirectSolver()

        prob = Problem()

        model = prob.model
        model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        recorder = SqliteRecorder("cases.sql")
        prob.driver.add_recorder(recorder)

        prob.setup()

        # set some initial guesses
        prob['circuit.n1.V'] = 10.
        prob['circuit.n2.V'] = 1.

        prob.run_driver()
        prob.cleanup()

        # create the case reader
        cr = CaseReader("cases.sql")

        # grab the data recorded in the first driver iteration
        first_driver_case = cr.driver_cases.get_case(0)

        self.assertAlmostEqual(first_driver_case.inputs['circuit.R1.V_in'][0], 9.90830282)
        self.assertAlmostEqual(first_driver_case.outputs['circuit.R1.I'][0], 0.09908303)

    def test_feature_driver_options_with_values(self):
        from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder, CaseReader
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem(model=SellarDerivatives())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.driver = ScipyOptimizeDriver()
        driver = prob.driver
        driver.options['optimizer'] = 'SLSQP'
        driver.options['tol'] = 1e-9
        driver.options['disp'] = False

        recorder = SqliteRecorder("cases.sql")
        driver.add_recorder(recorder)

        driver.recording_options['includes'] = []
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['record_desvars'] = True

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = CaseReader("cases.sql")
        first_driver_case = cr.driver_cases.get_case(0)
        recorded_objectives = first_driver_case.get_objectives()
        recorded_constraints = first_driver_case.get_constraints()
        recorded_desvars = first_driver_case.get_desvars()

        self.assertEqual(set(recorded_objectives.keys), {'obj'})
        self.assertEqual(set(recorded_constraints.keys), {'con1', 'con2'})
        self.assertEqual(set(recorded_desvars.keys), {'x', 'z'})

        self.assertAlmostEqual(recorded_objectives['obj'][0], 28.58830817)
        self.assertAlmostEqual(recorded_desvars['x'][0], 1.)

    def test_feature_load_system_case_for_restart(self):
        #######################################################################
        # Do the initial optimization run
        #######################################################################
        from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem(model=SellarDerivatives())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        recorder = SqliteRecorder('cases.sql')
        model.add_recorder(recorder)

        model.recording_options['record_inputs'] = True
        model.recording_options['record_outputs'] = True
        model.recording_options['record_residuals'] = True
        model.recording_options['record_metadata'] = False
        model.recording_options['options_excludes'] = ['*']

        driver = prob.driver = ScipyOptimizeDriver()
        driver.options['optimizer'] = 'SLSQP'
        driver.options['tol'] = 1e-9
        driver.options['disp'] = False

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        #######################################################################
        # Assume that the optimization given above failed before it finished.
        # To debug the problem, we can run the script again, but this time using
        # the last recorded case as a starting point.
        #######################################################################
        from openmdao.api import Problem, ScipyOptimizeDriver, CaseReader
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem(model=SellarDerivatives())
        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        model.recording_options['record_inputs'] = True
        model.recording_options['record_outputs'] = True
        model.recording_options['record_residuals'] = True
        model.recording_options['record_metadata'] = False
        model.recording_options['options_excludes'] = ['*']

        prob.driver = ScipyOptimizeDriver()
        driver = prob.driver
        driver.options['optimizer'] = 'SLSQP'
        driver.options['tol'] = 1e-9
        driver.options['disp'] = False

        prob.setup()

        cr = CaseReader('cases.sql')

        # Load the last case written
        last_case = cr.system_cases.get_case(-1)
        prob.load_case(last_case)

        prob.run_driver()
        prob.cleanup()


if __name__ == "__main__":
    unittest.main()
