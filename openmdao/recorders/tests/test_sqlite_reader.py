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
        self.prob = Problem()
        self.prob.model = SellarDerivatives()

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

        model = self.prob.model = Group()
        model.add_subsystem('px', IndepVarComp('x', 1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])
        model.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])
        model.add_subsystem('obj_cmp', ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                            z=np.array([0.0, 0.0]), x=0.0),
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
        self.prob.setup(check=False)
        self.prob.run_driver()
        self.prob.cleanup()

        cr = CaseReader(self.filename)
        self.assertEqual(cr.format_version, format_version,
                         msg='format version not read correctly')

    def test_reader_instantiates(self):
        """ Test that CaseReader returns an SqliteCaseReader. """
        self.setup_sellar_model()
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
        self.assertEqual(cr.driver_cases.num_cases, 8)
        self.assertEqual(cr.system_cases.num_cases, 0)
        self.assertEqual(cr.solver_cases.num_cases, 0)

        # Test to see if the access by case keys works:
        seventh_slsqp_iteration_case = cr.driver_cases.get_case('rank0:SLSQP|6')
        np.testing.assert_almost_equal(seventh_slsqp_iteration_case.desvars['z'], [1.97846296,  -2.21388305e-13],
                                       decimal=2,
                                       err_msg='Case reader gives '
                                       'incorrect Parameter value'
                                       ' for {0}'.format('pz.z'))

        # Test values from one case, the last case
        last_case = cr.driver_cases.get_case(-1)
        np.testing.assert_almost_equal(last_case.desvars['z'], self.prob['z'],
                                       err_msg='Case reader gives '
                                       'incorrect Parameter value'
                                       ' for {0}'.format('pz.z'))
        np.testing.assert_almost_equal(last_case.desvars['x'], [-0.00309521],
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

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed" )
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP" )
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

    def test_reading_system_metadata(self):

        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

        self.setup_sellar_grouped_scaled_model()

        self.prob.driver = pyOptSparseDriver()
        self.prob.driver.options['optimizer'] = OPTIMIZER
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
        
        assert_rel_error(
            self, cr.system_metadata['pz']['scaling_factors']['output']['nonlinear']['phys'][0][1], [2.0, 2.0], 1.0e-3)

    def test_reading_solver_metadata(self):
        self.setup_sellar_model()

        nonlinear_solver = self.prob.model.nonlinear_solver
        self.prob.model.nonlinear_solver.add_recorder(self.recorder)

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
        np.testing.assert_almost_equal(last_case.desvars['z'],
                                       self.prob['pz.z'],
                                       err_msg='Case reader gives '
                                       'incorrect Parameter value'
                                       ' for {0}'.format('pz.z'))
        np.testing.assert_almost_equal(last_case.desvars['x'],
                                       self.prob['px.x'],
                                       err_msg='Case reader gives '
                                       'incorrect Parameter value'
                                       ' for {0}'.format('px.x'))
        np.testing.assert_almost_equal(last_case.sysincludes['y2'],
                                       self.prob['mda.d2.y2'],
                                       err_msg='Case reader gives '
                                       'incorrect Parameter value'
                                       ' for {0}'.format('mda.d2.y2'))



if __name__ == "__main__":
    unittest.main()
