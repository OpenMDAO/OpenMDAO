""" Unit tests for the SqliteCaseReader. """
from __future__ import print_function

import errno
import os
import unittest
from shutil import rmtree
from tempfile import mkdtemp, mkstemp

import numpy as np

from openmdao.test_suite.components.sellar import SellarDerivatives
from openmdao.api import Problem, Group, IndepVarComp, ExecComp, NonlinearBlockGS, ScipyKrylov, LinearBlockGS
from openmdao.recorders.sqlite_recorder import SqliteRecorder, format_version
from openmdao.recorders.case_reader import CaseReader
from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.recorders.recording_iteration_stack import recording_iteration
from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, \
    SellarDis2withDerivatives
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.utils.general_utils import set_pyoptsparse_opt

try:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
except ImportError:
    pyOptSparseDriver = None

# Test that pyoptsparse SLSQP is a viable option
try:
    import pyoptsparse.pySLSQP.slsqp as slsqp
except ImportError:
    slsqp = None

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')
if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
    optimizers = {'pyoptsparse': pyOptSparseDriver}


class TestSqliteCaseReader(unittest.TestCase):

    def setup_sellar_model_with_optimization(self):
        self.prob = Problem(SellarDerivatives())

        optimizer = 'pyoptsparse'
        self.prob.driver = optimizers[optimizer]()

        self.prob.model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                       upper=np.array([10.0, 10.0]))
        self.prob.model.add_design_var('x', lower=0.0, upper=10.0)
        self.prob.model.add_objective('obj')
        self.prob.model.add_constraint('con1', upper=0.0)
        self.prob.model.add_constraint('con2', upper=0.0)
        self.prob.model.suppress_solver_output = True

        self.prob.driver.options['print_results'] = False

    def setup_sellar_model(self):
        self.prob = Problem()

        model = self.prob.model
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

    def setup_sellar_grouped_model(self):
        self.prob = Problem()

        model = self.prob.model = Group()

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        mda = model.add_subsystem('mda', Group(), promotes=['x', 'z', 'y1', 'y2'])
        mda.linear_solver = ScipyKrylov()
        mda.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        mda.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                            z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        mda.nonlinear_solver = NonlinearBlockGS()
        model.linear_solver = LinearBlockGS()

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

    def setup_sellar_grouped_scaled_model(self):
        self.prob = Problem()

        model = self.prob.model = Group()

        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0]), ref=2.0), promotes=['z'])

        mda = model.add_subsystem('mda', Group(), promotes=['x', 'z', 'y1', 'y2'])
        mda.linear_solver = ScipyKrylov()
        mda.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        mda.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                            z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        mda.nonlinear_solver = NonlinearBlockGS()
        model.linear_solver = LinearBlockGS()

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

    def setup_sellar_model_with_units(self):
        self.prob = Problem()

        model = self.prob.model = Group()
        model.add_subsystem('px', IndepVarComp('x', 1.0, units='m', lower=-1000, upper=1000), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])
        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])
        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                            z=np.array([0.0, 0.0]), x={'value': 0.0, 'units': 'm'},
                            y1={'units': 'm'}, y2={'units': 'cm'}),
                            promotes=['obj', 'x', 'z', 'y1', 'y2'])

        model.add_subsystem('con_cmp1', ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        model.add_subsystem('con_cmp2', ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])
        self.prob.model.nonlinear_solver = NonlinearBlockGS()
        self.prob.model.linear_solver = LinearBlockGS()

        self.prob.model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        self.prob.model.add_design_var('x', lower=0.0, upper=10.0)
        self.prob.model.add_objective('obj')
        self.prob.model.add_constraint('con1', upper=0.0)
        self.prob.model.add_constraint('con2', upper=0.0)

    def setUp(self):

        recording_iteration.stack = []  # reset to avoid problems from earlier tests
        self.dir = mkdtemp()
        self.filename = os.path.join(self.dir, "sqlite_test")
        self.recorder = SqliteRecorder(self.filename)
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

    def test_bad_filetype(self):
        # Pass it a plain text file.
        fd, filepath = mkstemp()
        with os.fdopen(fd, 'w') as tmp:
            tmp.write("Lorem ipsum")
            tmp.close()

        with self.assertRaises(IOError):
            _ = CaseReader(filepath)

    def test_format_version(self):
        self.setup_sellar_model()
        self.prob.model.add_recorder(self.recorder)
        self.prob.setup(check=False)
        self.prob.run_driver()
        self.prob.cleanup()

        cr = CaseReader(self.filename)
        self.assertEqual(cr.format_version, format_version,
                         msg='format version not read correctly')

    def test_reader_instantiates(self):
        """ Test that CaseReader returns an SqliteCaseReader. """
        self.setup_sellar_model()
        self.prob.model.add_recorder(self.recorder)
        self.prob.setup(check=False)
        self.prob.run_driver()
        self.prob.cleanup()

        cr = CaseReader(self.filename)
        self.assertTrue(isinstance(cr, SqliteCaseReader), msg='CaseReader not'
                        ' returning the correct subclass.')

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed" )
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP" )
    def test_reading_driver_cases(self):
        """ Tests that the reader returns params correctly. """
        self.setup_sellar_model_with_optimization()

        self.prob.driver.recording_options['record_desvars'] = True
        self.prob.driver.recording_options['record_responses'] = True
        self.prob.driver.recording_options['record_objectives'] = True
        self.prob.driver.recording_options['record_constraints'] = True
        self.prob.driver.add_recorder(self.recorder)

        self.prob.setup(check=False)
        self.prob.run_driver()
        self.prob.cleanup()

        cr = CaseReader(self.filename)

        # Test to see if we got the correct number of cases
        self.assertEqual(cr.driver_cases.num_cases, 7)
        self.assertEqual(cr.system_cases.num_cases, 0)
        self.assertEqual(cr.solver_cases.num_cases, 0)

        # Test to see if the access by case keys works:
        seventh_slsqp_iteration_case = cr.driver_cases.get_case('rank0:SLSQP|5')
        np.testing.assert_almost_equal(seventh_slsqp_iteration_case.outputs['z'], [1.97846296,  -2.21388305e-13],
                                       decimal=2,
                                       err_msg='Case reader gives '
                                       'incorrect Parameter value'
                                       ' for {0}'.format('pz.z'))

        # Test values from one case, the last case
        last_case = cr.driver_cases.get_case(-1)
        np.testing.assert_almost_equal(last_case.outputs['z'], self.prob['z'],
                                       err_msg='Case reader gives '
                                       'incorrect Parameter value'
                                       ' for {0}'.format('pz.z'))
        np.testing.assert_almost_equal(last_case.outputs['x'], [-0.00309521],
                                       decimal=2,
                                       err_msg='Case reader gives '
                                       'incorrect Parameter value'
                                       ' for {0}'.format('px.x'))

        # Test to see if the case keys (iteration coords) come back correctly
        case_keys = cr.driver_cases.list_cases()
        print (case_keys)
        for i, iter_coord in enumerate(case_keys):
            self.assertEqual(iter_coord, 'rank0:SLSQP|{}'.format(i))

    def test_reading_system_cases(self):

        self.setup_sellar_model()

        self.prob.model.recording_options['record_inputs'] = True
        self.prob.model.recording_options['record_outputs'] = True
        self.prob.model.recording_options['record_residuals'] = True
        self.prob.model.recording_options['record_metadata'] = False

        self.prob.model.add_recorder(self.recorder)

        d1 = self.prob.model.d1  # instance of SellarDis1withDerivatives, a Group
        d1.add_recorder(self.recorder)

        obj_cmp = self.prob.model.obj_cmp  # an ExecComp
        obj_cmp.add_recorder(self.recorder)

        self.prob.setup(check=False)
        self.prob.run_driver()
        self.prob.cleanup()

        cr = CaseReader(self.filename)

        # Test to see if we got the correct number of cases
        self.assertEqual(cr.driver_cases.num_cases, 0)
        self.assertEqual(cr.system_cases.num_cases, 15)
        self.assertEqual(cr.solver_cases.num_cases, 0)

        # Test values from cases
        second_last_case = cr.system_cases.get_case(-2)
        np.testing.assert_almost_equal(second_last_case.inputs['y2'], [12.05848815, ],
                                       err_msg='Case reader gives '
                                       'incorrect input value for {0}'.format('obj_cmp.y2'))
        np.testing.assert_almost_equal(second_last_case.outputs['obj'], [28.58830817, ],
                                       err_msg='Case reader gives '
                                       'incorrect output value for {0}'.format('obj_cmp.obj'))
        np.testing.assert_almost_equal(second_last_case.residuals['obj'], [0.0, ],
                                       err_msg='Case reader gives '
                                       'incorrect residual value for {0}'.format('obj_cmp.obj'))

        # Test to see if the case keys ( iteration coords ) come back correctly
        case_keys = cr.system_cases.list_cases()[:-1]  # Skip the last one
        for i, iter_coord in enumerate(case_keys):
            if i % 2 == 0:
                last_solver = 'd1._solve_nonlinear'
            else:
                last_solver = 'obj_cmp._solve_nonlinear'
            solver_iter_count = i // 2
            self.assertEqual(iter_coord,
                             'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|{iter}|'
                             '{solver}|{iter}'.format(iter=solver_iter_count, solver=last_solver))

    def test_reading_solver_cases(self):
        self.setup_sellar_model()

        self.prob.model._nonlinear_solver.recording_options['record_abs_error'] = True
        self.prob.model._nonlinear_solver.recording_options['record_rel_error'] = True
        self.prob.model._nonlinear_solver.recording_options['record_solver_residuals'] = True
        self.prob.model._nonlinear_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)

        self.prob.run_driver()
        self.prob.cleanup()

        cr = CaseReader(self.filename)

        # Test to see if we got the correct number of cases
        self.assertEqual(cr.driver_cases.num_cases, 0)
        self.assertEqual(cr.system_cases.num_cases, 0)
        self.assertEqual(cr.solver_cases.num_cases, 7)

        # Test values from cases
        last_case = cr.solver_cases.get_case(-1)
        np.testing.assert_almost_equal(last_case.abs_err, [0.0, ],
                                       err_msg='Case reader gives incorrect value for abs_err')
        np.testing.assert_almost_equal(last_case.rel_err, [0.0, ],
                                       err_msg='Case reader gives incorrect value for rel_err')
        np.testing.assert_almost_equal(last_case.outputs['x'], [1.0, ],
                                       err_msg='Case reader gives '
                                       'incorrect output value for {0}'.format('x'))
        np.testing.assert_almost_equal(last_case.residuals['con2'], [0.0, ],
                                       err_msg='Case reader gives '
                                       'incorrect residual value for {0}'.format('con_cmp2.con2'))

        # Test to see if the case keys ( iteration coords ) come back correctly
        case_keys = cr.system_cases.list_cases()
        for i, iter_coord in enumerate(case_keys):
            self.assertEqual(iter_coord,
                             'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|{}'
                             .format(i))

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP")
    def test_reading_driver_metadata(self):
        self.setup_sellar_model_with_optimization()

        self.prob.driver.recording_options['record_desvars'] = True
        self.prob.driver.recording_options['record_responses'] = True
        self.prob.driver.recording_options['record_objectives'] = True
        self.prob.driver.recording_options['record_constraints'] = True
        self.prob.driver.add_recorder(self.recorder)

        self.prob.setup(check=False)
        self.prob.run_driver()
        self.prob.cleanup()

        cr = CaseReader(self.filename)

        self.assertEqual(len(cr.driver_metadata['connections_list']), 11)
        self.assertEqual(len(cr.driver_metadata['tree']), 4)

    def test_reading_metadata(self):
        self.setup_sellar_model_with_units()
        self.prob.driver.add_recorder(self.recorder)

        self.prob.setup(check=False)
        self.prob.run_driver()
        self.prob.cleanup()

        cr = CaseReader(self.filename)

        self.assertEqual(cr.output2meta['x']['units'], 'm')
        self.assertEqual(cr.input2meta['obj_cmp.y1']['units'], 'm')
        self.assertEqual(cr.input2meta['obj_cmp.y2']['units'], 'cm')
        self.assertEqual(cr.input2meta['d1.x']['units'], None)
        self.assertEqual(cr.input2meta['d1.y1']['units'], None)
        self.assertEqual(cr.input2meta['d1.y2']['units'], None)
        self.assertEqual(cr.output2meta['x']['explicit'], True)
        self.assertEqual(cr.output2meta['x']['type'], {'output', 'desvar'})
        self.assertEqual(cr.input2meta['obj_cmp.y1']['explicit'], True)
        self.assertEqual(cr.input2meta['obj_cmp.y2']['explicit'], True)
        self.assertEqual(cr.output2meta['x']['lower'], -1000)
        self.assertEqual(cr.output2meta['x']['upper'], 1000)
        self.assertEqual(cr.output2meta['y2']['upper'], None)
        self.assertEqual(cr.output2meta['y2']['lower'], None)

    def test_reading_system_metadata(self):

        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

        self.setup_sellar_grouped_scaled_model()

        self.prob.driver = pyOptSparseDriver()
        self.prob.driver.options['optimizer'] = OPTIMIZER

        self.prob.model.options.declare("test1", 1)

        self.prob.model.mda.d1.options.declare("test2", "2")

        self.prob.model.pz.options.declare("test3", True)
        self.prob.model.pz.recording_options['metadata_excludes'] = ['*']

        if OPTIMIZER == 'SLSQP':
            self.prob.driver.opt_settings['ACC'] = 1e-9

        self.prob.model.recording_options['record_inputs'] = True
        self.prob.model.recording_options['record_outputs'] = True
        self.prob.model.recording_options['record_residuals'] = True
        self.prob.model.recording_options['record_metadata'] = True

        self.prob.model.add_recorder(self.recorder)

        pz = self.prob.model.pz # IndepVarComp which is an ExplicitComponent
        pz.add_recorder(self.recorder)

        mda = self.prob.model.mda  # Group
        d1 = mda.d1
        d1.add_recorder(self.recorder)

        self.prob.setup(check=False, mode='rev')

        self.prob.run_driver()

        self.prob.cleanup()

        cr = CaseReader(self.filename)

        self.assertEqual(
                sorted(cr.system_metadata.keys()),
                sorted(['root', 'mda.d1', 'pz'])
        )

        self.assertEqual(cr.system_metadata['root']['component_metadata']['test1'], 1)
        self.assertEqual(cr.system_metadata['mda.d1']['component_metadata']['test2'], "2")
        self.assertFalse('test3' in cr.system_metadata['pz']['component_metadata'])

        assert_rel_error(
            self, cr.system_metadata['pz']['scaling_factors']['output']['nonlinear']['phys'][0][1], [2.0, 2.0], 1.0e-3)

    def test_reading_solver_metadata(self):
        self.setup_sellar_model()

        nonlinear_solver = self.prob.model.nonlinear_solver
        nonlinear_solver.add_recorder(self.recorder)

        linear_solver = self.prob.model.linear_solver
        linear_solver.add_recorder(self.recorder)

        d1 = self.prob.model.d1  # instance of SellarDis1withDerivatives, a Group
        d1.nonlinear_solver = NonlinearBlockGS()
        d1.nonlinear_solver.options['maxiter'] = 5
        d1.nonlinear_solver.add_recorder(self.recorder)

        self.prob.setup(check=False)
        self.prob.run_driver()
        self.prob.cleanup()

        cr = CaseReader(self.filename)

        self.assertEqual(
                sorted(cr.solver_metadata.keys()),
                sorted(['root.LinearBlockGS', 'root.NonlinearBlockGS', 'd1.NonlinearBlockGS'])
        )
        self.assertEqual(cr.solver_metadata['d1.NonlinearBlockGS']['solver_options']['maxiter'], 5)
        self.assertEqual(cr.solver_metadata['root.NonlinearBlockGS']['solver_options']['maxiter'],10)
        self.assertEqual(cr.solver_metadata['root.LinearBlockGS']['solver_class'],'LinearBlockGS')

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed" )
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP" )
    def test_reading_driver_recording_with_system_vars(self):

        self.setup_sellar_grouped_model()

        self.prob.driver = pyOptSparseDriver()
        self.prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            self.prob.driver.opt_settings['ACC'] = 1e-9

        self.prob.driver.add_recorder(self.recorder)
        driver = self.prob.driver
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_responses'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['includes'] = ['mda.d2.y2',]

        self.prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            self.prob.driver.opt_settings['ACC'] = 1e-9

        self.prob.setup(check=False)
        self.prob.run_driver()
        self.prob.cleanup()

        cr = CaseReader(self.filename)

        # Test values from one case, the last case
        last_case = cr.driver_cases.get_case(-1)
        np.testing.assert_almost_equal(last_case.outputs['z'],
                                       self.prob['pz.z'],
                                       err_msg='Case reader gives '
                                       'incorrect Parameter value'
                                       ' for {0}'.format('pz.z'))
        np.testing.assert_almost_equal(last_case.outputs['x'],
                                       self.prob['px.x'],
                                       err_msg='Case reader gives '
                                       'incorrect Parameter value'
                                       ' for {0}'.format('px.x'))
        np.testing.assert_almost_equal(last_case.outputs['y2'],
                                       self.prob['mda.d2.y2'],
                                       err_msg='Case reader gives '
                                       'incorrect Parameter value'
                                       ' for {0}'.format('mda.d2.y2'))

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP")
    def test_get_child_cases(self):
        self.setup_sellar_grouped_model()

        self.prob.driver = pyOptSparseDriver()
        self.prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            self.prob.driver.opt_settings['ACC'] = 1e-9

        self.prob.driver.add_recorder(self.recorder)
        self.prob.model.add_recorder(self.recorder)
        driver = self.prob.driver
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_responses'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True

        self.prob.model._nonlinear_solver.add_recorder(self.recorder)
        self.prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            self.prob.driver.opt_settings['ACC'] = 1e-9

        self.prob.setup(check=False)
        self.prob.run_driver()
        self.prob.cleanup()

        cr = CaseReader(self.filename)

        expected_coords = [
            'rank0:SLSQP|0',
            'rank0:SLSQP|1',
            'rank0:SLSQP|2',
            'rank0:SLSQP|3',
            'rank0:SLSQP|4',
            'rank0:SLSQP|5',
            'rank0:SLSQP|6',
        ]
        ind = 0
        for c in cr.get_cases():
            self.assertEqual(c.iteration_coordinate, expected_coords[ind])
            ind += 1

        coords_2 = [
            'rank0:SLSQP|0',
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0',
            'rank0:SLSQP|1',
            'rank0:SLSQP|1|root._solve_nonlinear|1|NLRunOnce|0',
            'rank0:SLSQP|2',
            'rank0:SLSQP|2|root._solve_nonlinear|2|NLRunOnce|0',
            'rank0:SLSQP|3',
            'rank0:SLSQP|3|root._solve_nonlinear|3|NLRunOnce|0',
            'rank0:SLSQP|4',
            'rank0:SLSQP|4|root._solve_nonlinear|4|NLRunOnce|0',
            'rank0:SLSQP|5',
            'rank0:SLSQP|5|root._solve_nonlinear|5|NLRunOnce|0',
            'rank0:SLSQP|6',
            'rank0:SLSQP|6|root._solve_nonlinear|6|NLRunOnce|0'
        ]
        ind = 0
        for c in cr.get_cases(recursive=True):
            self.assertEqual(c.iteration_coordinate, coords_2[ind])
            ind += 1

        coord_children = [
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0'
        ]
        for c in cr.get_cases('rank0:SLSQP|0', True):
            self.assertEqual(c.iteration_coordinate, coord_children[0])

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP")
    def test_get_child_cases_system(self):
        self.setup_sellar_grouped_model()

        self.prob.driver = pyOptSparseDriver()
        self.prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            self.prob.driver.opt_settings['ACC'] = 1e-9

        self.prob.model.add_recorder(self.recorder)
        driver = self.prob.driver

        self.prob.model._nonlinear_solver.add_recorder(self.recorder)
        self.prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            self.prob.driver.opt_settings['ACC'] = 1e-9

        self.prob.setup(check=False)
        self.prob.run_driver()
        self.prob.cleanup()

        cr = CaseReader(self.filename)

        coords = [
            'rank0:SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0',
            'rank0:SLSQP|1|root._solve_nonlinear|1|NLRunOnce|0',
            'rank0:SLSQP|2|root._solve_nonlinear|2|NLRunOnce|0',
            'rank0:SLSQP|3|root._solve_nonlinear|3|NLRunOnce|0',
            'rank0:SLSQP|4|root._solve_nonlinear|4|NLRunOnce|0',
            'rank0:SLSQP|5|root._solve_nonlinear|5|NLRunOnce|0',
            'rank0:SLSQP|6|root._solve_nonlinear|6|NLRunOnce|0'
        ]
        ind = 0
        for c in cr.get_cases(recursive=True):
            self.assertEqual(c.iteration_coordinate, coords[ind])
            ind += 1

    def test_list_outputs(self):
        self.setup_sellar_model()

        nonlinear_solver = self.prob.model.nonlinear_solver
        linear_solver = self.prob.model.linear_solver

        d1 = self.prob.model.d1  # instance of SellarDis1withDerivatives, a Group
        d1.nonlinear_solver = NonlinearBlockGS()
        d1.nonlinear_solver.options['maxiter'] = 5
        self.prob.model.add_recorder(self.recorder)
        self.prob.model.d1.add_recorder(self.recorder)
        self.prob.model.recording_options['record_residuals'] = True

        self.prob.setup(check=False)
        self.prob.run_driver()
        self.prob.cleanup()

        cr = CaseReader(self.filename)

        outputs = cr.list_outputs(None, True, True, True, True, None, True, True, True,
                                  True, True, True)

        expected_outputs = {
            'd2.y2': {'lower': None, 'ref': 1.0, 'resids': [0.], 'shape': (1,), 'values': [12.0584882]},
            'con_cmp1.con1': {'lower': None, 'ref': 1.0, 'resids': [0.], 'shape': (1,), 'values': [-22.4283024]},
            'pz.z': {'lower': None, 'ref': 1.0, 'resids': [0., 0.], 'shape': (2,), 'values': [5., 2.]},
            'obj_cmp.obj': {'lower': None, 'ref': 1.0, 'resids': [0.], 'shape': (1,), 'values': [28.5883082]},
            'px.x': {'lower': None, 'ref': 1.0, 'resids': [0.], 'shape': (1,), 'values': [1.]},
            'con_cmp2.con2': {'lower': None, 'ref': 1.0, 'resids': [0.], 'shape': (1,), 'values': [-11.9415118]},
            'd1.y1': {'lower': None, 'ref': 1.0, 'resids': [1.318e-10], 'shape': (1,), 'values': [25.5883024]}
        }

        self.assertEqual(len(outputs), 7)
        for o in outputs:
            vals = o[1]
            name = o[0]
            expected = expected_outputs[name]
            self.assertEqual(vals['lower'], expected['lower'])
            self.assertEqual(vals['ref'], expected['ref'])
            self.assertEqual(vals['shape'], expected['shape'])
            np.testing.assert_almost_equal(vals['resids'], expected['resids'])
            np.testing.assert_almost_equal(vals['value'], expected['values'])

        expected_outputs_case = {
            'd1.y1': {'lower': None, 'ref': 1.0, 'resids': [1.318e-10], 'shape': (1,), 'values': [25.5454859]}
        }

        sys_case = cr.system_cases.get_case(1)
        outputs_case = cr.list_outputs(sys_case, True, True, True, True, None, True, True, True,
                                       True, True, True)

        for o in outputs_case:
            vals = o[1]
            name = o[0]
            expected = expected_outputs_case[name]
            self.assertEqual(vals['lower'], expected['lower'])
            self.assertEqual(vals['ref'], expected['ref'])
            self.assertEqual(vals['shape'], expected['shape'])
            np.testing.assert_almost_equal(vals['resids'], expected['resids'])
            np.testing.assert_almost_equal(vals['value'], expected['values'])

        expected_explicit_outputs_case = expected_outputs_case
        expl_outputs_case = cr.list_outputs(sys_case, True, False, True, True, None, True, True, True,
                                            True, True, True)
        for o in outputs_case:
            vals = o[1]
            name = o[0]
            expected = expected_explicit_outputs_case[name]
            self.assertEqual(vals['lower'], expected['lower'])
            self.assertEqual(vals['ref'], expected['ref'])
            self.assertEqual(vals['shape'], expected['shape'])
            np.testing.assert_almost_equal(vals['resids'], expected['resids'])
            np.testing.assert_almost_equal(vals['value'], expected['values'])

        impl_outputs_case = cr.list_outputs(sys_case, False, True)
        self.assertEqual(len(impl_outputs_case), 0)

    def test_list_inputs(self):
        self.setup_sellar_model()

        nonlinear_solver = self.prob.model.nonlinear_solver
        linear_solver = self.prob.model.linear_solver

        d1 = self.prob.model.d1  # instance of SellarDis1withDerivatives, a Group
        d1.nonlinear_solver = NonlinearBlockGS()
        d1.nonlinear_solver.options['maxiter'] = 5
        self.prob.model.add_recorder(self.recorder)
        self.prob.model.d1.add_recorder(self.recorder)
        self.prob.model.recording_options['record_residuals'] = True

        self.prob.setup(check=False)
        self.prob.run_driver()
        self.prob.cleanup()

        cr = CaseReader(self.filename)

        inputs = cr.list_inputs(None, True, True, True)

        expected_inputs = {
            'obj_cmp.x': {'value': [1.]},
            'd2.z': {'value': [5., 2.]},
            'con_cmp2.y2': {'value': [12.05848815]},
            'obj_cmp.y2': {'value': [12.05848815]},
            'obj_cmp.z': {'value': [5., 2.]},
            'd1.x': {'value': [1.]},
            'd1.z': {'value': [5., 2.]},
            'd1.y2': {'value': [12.05848815]},
            'con_cmp1.y1': {'value': [25.58830237]},
            'obj_cmp.y1': {'value': [25.58830237]},
            'd2.y1': {'value': [25.58830237]}
        }

        self.assertEqual(len(inputs), 11)
        for o in inputs:
            vals = o[1]
            name = o[0]
            expected = expected_inputs[name]
            np.testing.assert_almost_equal(vals['value'], expected['value'])

        expected_inputs_case = {
            'd1.z': {'value': [5., 2.]},
            'd1.x': {'value': [1.]},
            'd1.y2': {'value': [12.27257053]}
        }

        sys_case = cr.system_cases.get_case(1)
        inputs_case = cr.list_inputs(sys_case, True, True, True, None)

        for o in inputs_case:
            vals = o[1]
            name = o[0]
            expected = expected_inputs_case[name]
            np.testing.assert_almost_equal(vals['value'], expected['value'])

    def test_get_vars(self):
        self.setup_sellar_model()

        d1 = self.prob.model.d1  # instance of SellarDis1withDerivatives, a Group
        d1.nonlinear_solver = NonlinearBlockGS()
        d1.nonlinear_solver.options['maxiter'] = 5
        self.prob.model.add_recorder(self.recorder)
        self.prob.model.d1.add_recorder(self.recorder)
        self.prob.model.recording_options['record_residuals'] = True
        self.prob.driver.add_recorder(self.recorder)

        self.prob.setup(check=False)
        self.prob.run_driver()
        self.prob.cleanup()

        cr = CaseReader(self.filename)
        driver_case = cr.driver_cases.get_case(0)
        desvars = driver_case.get_desvars()
        objectives = driver_case.get_objectives()
        constraints = driver_case.get_constraints()
        responses = driver_case.get_responses()

        expected_desvars = {
            "x": [1.],
            "z": [5., 2.]
        }

        expected_objectives = { "obj": [28.58830817], }

        expected_constraints = { "con1": [-22.42830237], "con2": [-11.94151185] }

        expected_responses = expected_objectives.copy()
        expected_responses.update(expected_constraints)

        for expected_set, actual_set in (
            (expected_desvars, desvars),
            (expected_objectives, objectives),
            (expected_constraints, constraints),
            (expected_responses, responses)):

            self.assertEqual(len(expected_set), len(actual_set.keys))
            for k in actual_set:
                np.testing.assert_almost_equal(expected_set[k], actual_set[k])


if __name__ == "__main__":
    unittest.main()
