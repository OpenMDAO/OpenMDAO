""" Unit tests for the SqliteCaseReader. """

import errno
import sys
import os
import sys
import unittest

from io import StringIO
from tempfile import mkstemp
from collections import OrderedDict

import numpy as np

import openmdao.api as om
from openmdao import __version__ as openmdao_version
from openmdao.recorders.sqlite_recorder import format_version
from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.recorders.tests.test_sqlite_recorder import ParaboloidProblem
from openmdao.recorders.case import PromAbsDict
from openmdao.core.tests.test_discrete import ModCompEx, ModCompIm
from openmdao.core.tests.test_expl_comp import RectangleComp, RectangleCompWithTags
from openmdao.core.tests.test_units import SpeedComp
from openmdao.test_suite.components.eggcrate import EggCrate
from openmdao.test_suite.components.expl_comp_array import TestExplCompArray
from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStates
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.paraboloid_problem import ParaboloidProblem
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped, \
    SellarDis1withDerivatives, SellarDis2withDerivatives, SellarProblem, SellarDerivatives
from openmdao.test_suite.components.sellar_feature import SellarMDA
from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_group import \
    MultipointBeamGroup
from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.utils.general_utils import set_pyoptsparse_opt, determine_adder_scaler, printoptions
from openmdao.utils.general_utils import remove_whitespace
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.mpi import MPI, multi_proc_exception_check
from openmdao.utils.units import convert_units
from openmdao.utils.om_warnings import OMDeprecationWarning

# check that pyoptsparse is installed
OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')
if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None


def count_keys(d):
    """
    Count the number of keys in the nested dictionary.

    Parameters
    ----------
    d: nested OrderedDict
        The dictionary of cases to be counted.
    """
    count = 0

    for k in d:
        count += 1
        if isinstance(d[k], OrderedDict):
            count += count_keys(d[k])

    return count


class SellarDerivativesGroupedPreAutoIVC(om.Group):
    """
    This version is needed for testing backwards compatibility for load_case on pre-3.2
    models.
    """

    def initialize(self):
        self.options.declare('nonlinear_solver', default=om.NonlinearBlockGS,
                             desc='Nonlinear solver (class or instance) for Sellar MDA')
        self.options.declare('nl_atol', default=None,
                             desc='User-specified atol for nonlinear solver.')
        self.options.declare('nl_maxiter', default=None,
                             desc='Iteration limit for nonlinear solver.')
        self.options.declare('linear_solver', default=om.ScipyKrylov,
                             desc='Linear solver (class or instance)')
        self.options.declare('ln_atol', default=None,
                             desc='User-specified atol for linear solver.')
        self.options.declare('ln_maxiter', default=None,
                             desc='Iteration limit for linear solver.')

    def setup(self):
        self.add_subsystem('px', om.IndepVarComp('x', 1.0), promotes=['x'])
        self.add_subsystem('pz', om.IndepVarComp('z', np.array([5.0, 2.0])), promotes=['z'])

        self.mda = mda = self.add_subsystem('mda', om.Group(), promotes=['x', 'z', 'y1', 'y2'])
        mda.add_subsystem('d1', SellarDis1withDerivatives(), promotes=['x', 'z', 'y1', 'y2'])
        mda.add_subsystem('d2', SellarDis2withDerivatives(), promotes=['z', 'y1', 'y2'])

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                  z=np.array([0.0, 0.0]), x=0.0, y1=0.0, y2=0.0),
                           promotes=['obj', 'x', 'z', 'y1', 'y2'])

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'), promotes=['con1', 'y1'])
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'), promotes=['con2', 'y2'])

        nl = self.options['nonlinear_solver']
        self.nonlinear_solver = nl()
        if self.options['nl_atol']:
            self.nonlinear_solver.options['atol'] = self.options['nl_atol']
        if self.options['nl_maxiter']:
            self.nonlinear_solver.options['maxiter'] = self.options['nl_maxiter']

        ln = self.options['linear_solver']
        self.linear_solver = ln()
        if self.options['ln_atol']:
            self.linear_solver.options['atol'] = self.options['ln_atol']
        if self.options['ln_maxiter']:
            self.linear_solver.options['maxiter'] = self.options['ln_maxiter']

    def configure(self):
        self.mda.linear_solver = om.ScipyKrylov()
        self.mda.nonlinear_solver = om.NonlinearBlockGS()


@use_tempdirs
class TestSqliteCaseReader(unittest.TestCase):

    def setUp(self):
        self.filename = "sqlite_test"
        self.recorder = om.SqliteRecorder(self.filename, record_viewer_data=False)

    def test_bad_filetype(self):
        # Pass a plain text file.
        fd, filepath = mkstemp()
        with os.fdopen(fd, 'w') as tmp:
            tmp.write("Lorem ipsum")
            tmp.close()

        with self.assertRaises(IOError) as cm:
            om.CaseReader(filepath)

        msg = 'File does not contain a valid sqlite database'
        self.assertTrue(str(cm.exception).startswith(msg))

    def test_bad_filename(self):
        # Pass a nonexistent file.
        with self.assertRaises(IOError) as cm:
            om.CaseReader('junk.sql')

        self.assertTrue(str(cm.exception).startswith('File does not exist'))

    def test_format_version(self):
        prob = SellarProblem()
        prob.model.add_recorder(self.recorder)
        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)
        self.assertEqual(cr._format_version, format_version,
                         msg='format version not read correctly')

    def test_reader_instantiates(self):
        """ Test that CaseReader returns an SqliteCaseReader. """
        prob = SellarProblem()
        prob.model.add_recorder(self.recorder)
        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)
        self.assertTrue(isinstance(cr, SqliteCaseReader),
                        msg='CaseReader not returning the correct subclass.')

    def test_case_attributes(self):
        """ Check that a Case object has all the expected attributes. """
        prob = SellarProblem()
        prob.setup()

        prob.driver.add_recorder(self.recorder)
        prob.run_driver()

        prob.cleanup()

        cr = om.CaseReader(self.filename)
        case = cr.get_case(0)

        self.assertEqual(case.source, 'driver')
        self.assertEqual(case.name, 'rank0:Driver|0')
        self.assertEqual(case.counter, 1)
        self.assertTrue(isinstance(case.timestamp, float))
        self.assertEqual(case.success, True)
        self.assertEqual(case.msg, '')
        self.assertTrue(isinstance(case.outputs, PromAbsDict))
        self.assertEqual(case.inputs, None)
        self.assertEqual(case.residuals, None)
        self.assertEqual(case.derivatives, None)
        self.assertEqual(case.parent, None)
        self.assertEqual(case.abs_err, None)
        self.assertEqual(case.rel_err, None)

    def test_invalid_source(self):
        """ Tests that the reader returns params correctly. """
        prob = SellarProblem(SellarDerivativesGrouped)

        driver = prob.driver

        driver.recording_options['record_desvars'] = False
        driver.recording_options['record_objectives'] = False
        driver.recording_options['record_constraints'] = False
        driver.recording_options['record_derivatives'] = False
        driver.add_recorder(self.recorder)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        # check that driver is our only source
        self.assertEqual(cr.list_sources(out_stream=None), ['driver'])

        # check source vars
        source_vars = cr.list_source_vars('driver', out_stream=None)
        self.assertEqual(sorted(source_vars['inputs']), [])
        self.assertEqual(sorted(source_vars['outputs']), [])

        with self.assertRaisesRegex(RuntimeError, "No cases recorded for problem"):
            cr.list_source_vars('problem', out_stream=None)

        with self.assertRaisesRegex(RuntimeError, "Source not found: root"):
            cr.list_source_vars('root', out_stream=None)

        with self.assertRaisesRegex(RuntimeError, "Source not found: root.nonlinear_solver"):
            cr.list_source_vars('root.nonlinear_solver', out_stream=None)

        # check list cases
        with self.assertRaisesRegex(RuntimeError, "Source not found: foo"):
            cr.list_cases('foo')

        with self.assertRaisesRegex(TypeError, "Source parameter must be a string, 999 is type int"):
            cr.list_cases(999)

    def test_reading_driver_cases(self):
        """ Tests that the reader returns params correctly. """
        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=om.NonlinearRunOnce,
                                                       linear_solver=om.ScipyKrylov,
                                                       mda_nonlinear_solver=om.NonlinearBlockGS)

        driver = prob.driver = om.ScipyOptimizeDriver(tol=1e-9, disp=False)

        driver.recording_options['record_desvars'] = False
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['record_derivatives'] = True
        driver.recording_options['record_inputs'] = True
        driver.recording_options['includes'] = ['*']
        driver.add_recorder(self.recorder)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        # check that we only have driver cases
        self.assertEqual(cr.list_sources(out_stream=None), ['driver'])

        # check source vars
        source_vars = cr.list_source_vars('driver', out_stream=None)
        self.assertEqual(sorted(source_vars['inputs']), ['x', 'y1', 'y2', 'z'])
        self.assertEqual(sorted(source_vars['outputs']), ['con1', 'con2', 'obj', 'x', 'y1', 'y2', 'z'])

        # check that we got the correct number of cases
        driver_cases = cr.list_cases('driver', out_stream=None)

        self.assertEqual(driver_cases, [
            'rank0:ScipyOptimize_SLSQP|0', 'rank0:ScipyOptimize_SLSQP|1', 'rank0:ScipyOptimize_SLSQP|2',
            'rank0:ScipyOptimize_SLSQP|3', 'rank0:ScipyOptimize_SLSQP|4', 'rank0:ScipyOptimize_SLSQP|5',
            'rank0:ScipyOptimize_SLSQP|6'
        ])

        # Test to see if the access by case keys works:
        seventh_slsqp_iteration_case = cr.get_case('rank0:ScipyOptimize_SLSQP|6')
        np.testing.assert_almost_equal(seventh_slsqp_iteration_case.outputs['z'],
                                       [1.97846296, -2.21388305e-13], decimal=2)

        deriv_case = cr.get_case('rank0:ScipyOptimize_SLSQP|4')
        np.testing.assert_almost_equal(deriv_case.derivatives['obj', 'z'],
                                       [[3.8178954, 1.73971323]], decimal=2)

        # While thinking about derivatives, let's get them all.
        derivs = deriv_case.derivatives

        self.assertEqual(set(derivs.keys()), set([
            ('obj', 'z'), ('con2', 'z'), ('con1', 'x'),
            ('obj', 'x'), ('con2', 'x'), ('con1', 'z')
        ]))

        # Test values from the last case
        last_case = cr.get_case(driver_cases[-1])
        np.testing.assert_almost_equal(last_case.outputs['z'], prob['z'])
        np.testing.assert_almost_equal(last_case.outputs['x'], [-0.00309521], decimal=2)

        # Test to see if the case keys (iteration coords) come back correctly
        for i, iter_coord in enumerate(driver_cases):
            self.assertEqual(iter_coord, 'rank0:ScipyOptimize_SLSQP|{}'.format(i))

    def test_driver_reading_outputs(self):

        prob = ParaboloidProblem()
        driver = prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)
        driver.recording_options['record_desvars'] = False
        driver.recording_options['record_objectives'] = False
        driver.recording_options['record_constraints'] = False
        driver.recording_options['record_inputs'] = False
        driver.recording_options['record_outputs'] = True
        driver.recording_options['record_residuals'] = False
        driver.recording_options['includes'] = ['*']
        driver.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()

        cr = om.CaseReader(self.filename)

        # check that we only have driver cases
        self.assertEqual(cr.list_sources(out_stream=None), ['driver'])

        # check source vars
        source_vars = cr.list_source_vars('driver', out_stream=None)
        self.assertEqual(sorted(source_vars['inputs']), [])
        self.assertEqual(sorted(source_vars['outputs']), ['c', 'f_xy', 'x', 'y'])

        # Test values from the last case
        driver_cases = cr.list_cases('driver', out_stream=None)
        last_case = cr.get_case(driver_cases[-1])
        np.testing.assert_almost_equal(last_case.outputs['f_xy'], prob['f_xy'])
        np.testing.assert_almost_equal(last_case.outputs['x'], prob['x'])

    def test_driver_reading_residuals(self):

        prob = ParaboloidProblem()
        driver = prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)
        driver.recording_options['record_desvars'] = False
        driver.recording_options['record_objectives'] = False
        driver.recording_options['record_constraints'] = False
        driver.recording_options['record_inputs'] = False
        driver.recording_options['record_outputs'] = False
        driver.recording_options['record_residuals'] = True
        driver.recording_options['includes'] = ['*']
        driver.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()

        cr = om.CaseReader(self.filename)

        # check that we only have driver cases
        self.assertEqual(cr.list_sources(out_stream=None), ['driver'])

        # check source vars
        source_vars = cr.list_source_vars('driver', out_stream=None)
        self.assertEqual(sorted(source_vars['inputs']), [])
        self.assertEqual(sorted(source_vars['residuals']), ['c', 'f_xy', 'x', 'y'])

        # Test values from the last case
        driver_cases = cr.list_cases('driver', out_stream=None)
        last_case = cr.get_case(driver_cases[-1])
        np.testing.assert_almost_equal(last_case.residuals['f_xy'], 0.0)
        np.testing.assert_almost_equal(last_case.residuals['x'], 0.0)

    def test_reading_system_cases(self):
        prob = SellarProblem(nonlinear_solver=om.NonlinearBlockGS,
                             linear_solver=om.ScipyKrylov)

        model = prob.model

        model.recording_options['record_inputs'] = True
        model.recording_options['record_outputs'] = True
        model.recording_options['record_residuals'] = True

        model.add_recorder(self.recorder)

        prob.setup()

        model.nonlinear_solver.options['use_apply_nonlinear'] = True

        model.d1.add_recorder(self.recorder)  # SellarDis1withDerivatives (an ExplicitComp)
        model.obj_cmp.add_recorder(self.recorder)  # an ExecComp

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        # check that we only have the three system sources
        self.assertEqual(sorted(cr.list_sources(out_stream=None)), ['root', 'root.d1', 'root.obj_cmp'])

        # check source vars
        source_vars = cr.list_source_vars('root', out_stream=None)
        self.assertEqual(sorted(source_vars['inputs']), ['x', 'y1', 'y2', 'z'])
        self.assertEqual(sorted(source_vars['outputs']), ['con1', 'con2', 'obj', 'x', 'y1', 'y2', 'z'])

        source_vars = cr.list_source_vars('root.d1', out_stream=None)
        self.assertEqual(sorted(source_vars['inputs']), ['x', 'y2', 'z'])
        self.assertEqual(sorted(source_vars['outputs']), ['y1'])

        source_vars = cr.list_source_vars('root.obj_cmp', out_stream=None)
        self.assertEqual(sorted(source_vars['inputs']), ['x', 'y1', 'y2', 'z'])
        self.assertEqual(sorted(source_vars['outputs']), ['obj'])

        # Test to see if we got the correct number of cases
        self.assertEqual(len(cr.list_cases('root', recurse=False, out_stream=None)), 1)
        self.assertEqual(len(cr.list_cases('root.d1', recurse=False, out_stream=None)), 7)
        self.assertEqual(len(cr.list_cases('root.obj_cmp', recurse=False, out_stream=None)), 7)

        # Test values from cases
        case = cr.get_case('rank0:Driver|0|root._solve_nonlinear|0')
        np.testing.assert_almost_equal(case.inputs['d1.y2'], [12.05848815, ])
        np.testing.assert_almost_equal(case.outputs['obj'], [28.58830817, ])
        np.testing.assert_almost_equal(case.residuals['obj'], [0.0, ],)

        # Test to see if the case keys (iteration coords) come back correctly
        for i, iter_coord in enumerate(cr.list_cases('root.d1', recurse=False, out_stream=None)):
            self.assertEqual(iter_coord,
                             'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|{iter}|'
                             'd1._solve_nonlinear|{iter}'.format(iter=i))

        for i, iter_coord in enumerate(cr.list_cases('root.obj_cmp', recurse=False, out_stream=None)):
            self.assertEqual(iter_coord,
                             'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|{iter}|'
                             'obj_cmp._solve_nonlinear|{iter}'.format(iter=i))

    def test_reading_solver_cases(self):
        prob = SellarProblem(nonlinear_solver=om.NonlinearBlockGS,
                             linear_solver=om.ScipyKrylov)
        prob.setup()

        solver = prob.model.nonlinear_solver
        solver.add_recorder(self.recorder)

        solver.recording_options['record_abs_error'] = True
        solver.recording_options['record_rel_error'] = True
        solver.recording_options['record_solver_residuals'] = True

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        # check that we only have the one solver source
        self.assertEqual(sorted(cr.list_sources(out_stream=None)), ['root.nonlinear_solver'])

        # check source vars
        source_vars = cr.list_source_vars('root.nonlinear_solver', out_stream=None)
        self.assertEqual(sorted(source_vars['inputs']), ['x', 'y1', 'y2', 'z'])
        self.assertEqual(sorted(source_vars['outputs']), ['con1', 'con2', 'obj', 'x', 'y1', 'y2', 'z'])

        # Test to see if we got the correct number of cases
        solver_cases = cr.list_cases('root.nonlinear_solver', out_stream=None)
        self.assertEqual(len(solver_cases), 7)

        # Test values from cases
        last_case = cr.get_case(solver_cases[-1])
        np.testing.assert_almost_equal(last_case.abs_err, [0.0, ])
        np.testing.assert_almost_equal(last_case.rel_err, [0.0, ])
        np.testing.assert_almost_equal(last_case.outputs['x'], [1.0, ])
        np.testing.assert_almost_equal(last_case.residuals['con2'], [0.0, ])

        # check that the case keys (iteration coords) come back correctly
        for i, iter_coord in enumerate(solver_cases):
            iter_num = i + 1
            self.assertEqual(iter_coord,
                             'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|%d' % iter_num)

    def test_reading_solver_metadata(self):
        prob = SellarProblem(nonlinear_solver=om.NonlinearBlockGS,
                             linear_solver=om.LinearBlockGS)

        prob.model.nonlinear_solver.add_recorder(self.recorder)

        # add an implicitcomp so we can verify that nonlinear solver metadata is saved
        # at component level
        lscomp = prob.model.add_subsystem('lscomp', om.LinearSystemComp())
        lscomp.nonlinear_solver = om.NonlinearBlockGS(maxiter=5)
        lscomp.nonlinear_solver.add_recorder(self.recorder)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        metadata = om.CaseReader(self.filename).solver_metadata

        self.assertEqual(
            sorted(metadata.keys()),
            ['lscomp.NonlinearBlockGS', 'root.LinearBlockGS', 'root.NonlinearBlockGS']
        )
        self.assertEqual(metadata['lscomp.NonlinearBlockGS']['solver_options']['maxiter'], 5)
        self.assertEqual(metadata['root.LinearBlockGS']['solver_options']['maxiter'], 10)
        self.assertEqual(metadata['root.NonlinearBlockGS']['solver_options']['maxiter'], 10)

    def test_reading_driver_recording_with_system_vars(self):
        prob = SellarProblem(SellarDerivativesGrouped)

        driver = prob.driver = om.ScipyOptimizeDriver(tol=1e-9, disp=False)

        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['includes'] = ['y2']
        driver.add_recorder(self.recorder)

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        # Test values from the last case
        driver_cases = cr.list_cases('driver', out_stream=None)
        last_case = cr.get_case(driver_cases[-1])

        np.testing.assert_almost_equal(last_case.outputs['z'], prob['z'])
        np.testing.assert_almost_equal(last_case.outputs['x'], prob['x'])
        np.testing.assert_almost_equal(last_case.outputs['y2'], prob['mda.d2.y2'])

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP")
    def test_get_child_cases(self):
        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=om.NonlinearRunOnce,
                                                       linear_solver=om.ScipyKrylov,
                                                       mda_nonlinear_solver=om.NonlinearBlockGS)

        driver = prob.driver = pyOptSparseDriver(optimizer='SLSQP', print_results=False)
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.add_recorder(self.recorder)

        prob.setup()

        model = prob.model
        model.add_recorder(self.recorder)
        model.nonlinear_solver.add_recorder(self.recorder)

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        # check driver cases
        expected_coords = [
            'rank0:pyOptSparse_SLSQP|0',
            'rank0:pyOptSparse_SLSQP|1',
            'rank0:pyOptSparse_SLSQP|2',
            'rank0:pyOptSparse_SLSQP|3',
            'rank0:pyOptSparse_SLSQP|4',
            'rank0:pyOptSparse_SLSQP|5',
            'rank0:pyOptSparse_SLSQP|6'
        ]

        last_counter = 0
        for i, c in enumerate(cr.get_cases(flat=False)):
            self.assertEqual(c.name, expected_coords[i])
            self.assertTrue(c.counter > last_counter)
            last_counter = c.counter

        self.assertEqual(i+1, len(expected_coords))

        # check driver cases with recursion, flat
        expected_coords = [
            'rank0:pyOptSparse_SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0',
            'rank0:pyOptSparse_SLSQP|0|root._solve_nonlinear|0',
            'rank0:pyOptSparse_SLSQP|0',
            'rank0:pyOptSparse_SLSQP|1|root._solve_nonlinear|1|NLRunOnce|0',
            'rank0:pyOptSparse_SLSQP|1|root._solve_nonlinear|1',
            'rank0:pyOptSparse_SLSQP|1',
            'rank0:pyOptSparse_SLSQP|2|root._solve_nonlinear|2|NLRunOnce|0',
            'rank0:pyOptSparse_SLSQP|2|root._solve_nonlinear|2',
            'rank0:pyOptSparse_SLSQP|2',
            'rank0:pyOptSparse_SLSQP|3|root._solve_nonlinear|3|NLRunOnce|0',
            'rank0:pyOptSparse_SLSQP|3|root._solve_nonlinear|3',
            'rank0:pyOptSparse_SLSQP|3',
            'rank0:pyOptSparse_SLSQP|4|root._solve_nonlinear|4|NLRunOnce|0',
            'rank0:pyOptSparse_SLSQP|4|root._solve_nonlinear|4',
            'rank0:pyOptSparse_SLSQP|4',
            'rank0:pyOptSparse_SLSQP|5|root._solve_nonlinear|5|NLRunOnce|0',
            'rank0:pyOptSparse_SLSQP|5|root._solve_nonlinear|5',
            'rank0:pyOptSparse_SLSQP|5',
            'rank0:pyOptSparse_SLSQP|6|root._solve_nonlinear|6|NLRunOnce|0',
            'rank0:pyOptSparse_SLSQP|6|root._solve_nonlinear|6',
            'rank0:pyOptSparse_SLSQP|6',
        ]

        last_counter = 0
        for i, c in enumerate(cr.get_cases(recurse=True, flat=True)):
            self.assertEqual(c.name, expected_coords[i])
            if len(c.name.split('|')) > 2:
                self.assertEqual(c.parent, expected_coords[i+1])
            else:
                self.assertEqual(c.parent, None)
            self.assertTrue(c.counter > last_counter)
            last_counter = c.counter

        self.assertEqual(i+1, len(expected_coords))

        # check child cases with recursion, flat
        expected_coords = [
            'rank0:pyOptSparse_SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0',
            'rank0:pyOptSparse_SLSQP|0|root._solve_nonlinear|0',
            'rank0:pyOptSparse_SLSQP|0',
        ]

        last_counter = 0
        for i, c in enumerate(cr.get_cases('rank0:pyOptSparse_SLSQP|0', recurse=True, flat=True)):
            self.assertEqual(c.name, expected_coords[i])
            self.assertTrue(c.counter > last_counter)
            last_counter = c.counter

        self.assertEqual(i+1, len(expected_coords))

        # check child cases with recursion, nested
        expected_coords = {
            'rank0:pyOptSparse_SLSQP|0': {
                'rank0:pyOptSparse_SLSQP|0|root._solve_nonlinear|0': {
                    'rank0:pyOptSparse_SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0': {}
                },
            }
        }

        cases = cr.get_cases('rank0:pyOptSparse_SLSQP|0', recurse=True, flat=False)

        count = 0
        for case in cases:
            count += 1
            coord = case.name
            self.assertTrue(coord in list(expected_coords.keys()))
            for child_case in cases[case]:
                count += 1
                child_coord = child_case.name
                self.assertTrue(child_coord in expected_coords[coord].keys())
                for grandchild_case in cases[case][child_case]:
                    count += 1
                    grandchild_coord = grandchild_case.name
                    self.assertTrue(grandchild_coord in expected_coords[coord][child_coord].keys())

        self.assertEqual(count, 3)

    def test_get_child_cases_system(self):
        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=om.NonlinearRunOnce,
                                                       linear_solver=om.ScipyKrylov,
                                                       mda_nonlinear_solver=om.NonlinearBlockGS)
        prob.driver = om.ScipyOptimizeDriver(tol=1e-9, disp=False)
        prob.setup()

        model = prob.model
        model.add_recorder(self.recorder)
        model.nonlinear_solver.add_recorder(self.recorder)
        model.mda.nonlinear_solver.add_recorder(self.recorder)

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        parent_coord = 'rank0:ScipyOptimize_SLSQP|2|root._solve_nonlinear|2'
        coord = parent_coord + '|NLRunOnce|0'

        # user scenario: given a case (with coord), get all cases with same parent
        case = cr.get_case(coord)
        self.assertEqual(case.parent, parent_coord)

        expected_coords = [
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|2|NonlinearBlockGS|1',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|2|NonlinearBlockGS|2',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|2|NonlinearBlockGS|3',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|2|NonlinearBlockGS|4',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|2|NonlinearBlockGS|5',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|2|NonlinearBlockGS|6',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|2|NonlinearBlockGS|7',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|2|NonlinearBlockGS|8',
            parent_coord + '|NLRunOnce|0',
            parent_coord
        ]

        last_counter = 0
        for i, c in enumerate(cr.get_cases(source=case.parent, recurse=True, flat=True)):
            self.assertEqual(c.name, expected_coords[i])
            self.assertTrue(c.counter > last_counter)
            last_counter = c.counter
            i += 1

        self.assertEqual(i, len(expected_coords))

    def test_list_cases_recurse(self):
        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=om.NonlinearRunOnce,
                                                       linear_solver=om.ScipyKrylov,
                                                       mda_nonlinear_solver=om.NonlinearBlockGS)
        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)
        prob.driver.add_recorder(self.recorder)
        prob.setup()

        model = prob.model
        model.add_recorder(self.recorder)
        model.mda.add_recorder(self.recorder)
        model.nonlinear_solver.add_recorder(self.recorder)
        model.mda.nonlinear_solver.add_recorder(self.recorder)

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        # get total iteration count to check against
        global_iterations = len(cr._global_iterations)

        #
        # get a recursive list of all cases (flat)
        #
        cases = cr.list_cases(recurse=True, flat=True, out_stream=None)

        # verify the cases are all there
        self.assertEqual(len(cases), global_iterations)

        # verify the cases are in proper order
        counter = 0
        for i, c in enumerate(cr.get_case(case) for case in cases):
            counter += 1
            self.assertEqual(c.counter, counter)

        #
        # get a recursive dict of all cases (nested)
        #
        cases = cr.list_cases(recurse=True, flat=False, out_stream=None)

        num_cases = count_keys(cases)

        self.assertEqual(num_cases, global_iterations)

        #
        # get a recursive list of child cases
        #
        parent_coord = 'rank0:ScipyOptimize_SLSQP|0|root._solve_nonlinear|0'

        expected_coords = [
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|1',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|2',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|3',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|4',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|5',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|6',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|7',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|0',
            parent_coord + '|NLRunOnce|0',
            parent_coord
        ]

        cases = cr.list_cases(parent_coord, recurse=True, flat=True, out_stream=None)

        # verify the cases are all there and are as expected
        self.assertEqual(len(cases), len(expected_coords))
        for i, c in enumerate(cases):
            self.assertEqual(c, expected_coords[i])

        #
        # get a list of cases for each source
        #
        sources = cr.list_sources(out_stream=None)
        self.assertEqual(sorted(sources), [
            'driver', 'root', 'root.mda', 'root.mda.nonlinear_solver', 'root.nonlinear_solver'
        ])

        # verify the coordinates of the returned cases are all there as expected
        expected_coord = {
            'driver':                    r'rank0:ScipyOptimize_SLSQP\|\d',
            'root':                      r'rank0:ScipyOptimize_SLSQP\|\d\|root._solve_nonlinear\|\d',
            'root.nonlinear_solver':     r'rank0:ScipyOptimize_SLSQP\|\d\|root._solve_nonlinear\|\d\|NLRunOnce\|0',
            'root.mda':                  r'rank0:ScipyOptimize_SLSQP\|\d\|root._solve_nonlinear\|\d\|NLRunOnce\|0\|mda._solve_nonlinear\|\d',
            'root.mda.nonlinear_solver': r'rank0:ScipyOptimize_SLSQP\|\d\|root._solve_nonlinear\|\d\|NLRunOnce\|0\|mda._solve_nonlinear\|\d\|NonlinearBlockGS\|\d',
        }
        counter = 0
        mda_counter = 0
        root_counter = 0
        for source in sources:
            expected = expected_coord[source]
            cases = cr.list_cases(source, recurse=False, out_stream=None)
            for case in cases:
                counter += 1
                if source.startswith('root.mda'):  # count all cases for/under mda system
                    mda_counter += 1
                if source.startswith('root.'):     # count all cases for/under root solver
                    root_counter += 1
                self.assertRegexpMatches(case, expected)

        self.assertEqual(counter, global_iterations)

        #
        # get a recursive list of child cases for the mda system
        #
        counter = 0

        cases = cr.list_cases('root.mda', recurse=True, flat=True, out_stream=None)
        for case in cases:
            self.assertTrue(case.index('|mda._solve_nonlinear|') > 0)
            counter += 1

        self.assertEqual(counter, mda_counter)

        #
        # get a recursive list of child cases for the root solver
        #
        counter = 0

        cases = cr.list_cases('root.nonlinear_solver', recurse=True, flat=True, out_stream=None)
        for case in cases:
            self.assertTrue(case.index('|NLRunOnce|') > 0)
            counter += 1

        self.assertEqual(counter, root_counter)

    def test_list_cases_nested_model(self):
        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=om.NonlinearRunOnce)
        prob.driver = om.ScipyOptimizeDriver(tol=1e-9, disp=False)
        prob.setup()

        model = prob.model
        model.add_recorder(self.recorder)
        model.mda.add_recorder(self.recorder)
        model.nonlinear_solver.add_recorder(self.recorder)
        model.mda.nonlinear_solver.add_recorder(self.recorder)

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        # get total iteration count to check against
        global_iterations = len(cr._global_iterations)

        #
        # get a recursive list of all cases (flat)
        #
        cases = cr.list_cases(recurse=True, flat=True, out_stream=None)

        # verify the cases are all there
        self.assertEqual(len(cases), global_iterations)

        # verify the cases are in proper order
        counter = 0
        for i, c in enumerate(cr.get_case(case) for case in cases):
            counter += 1
            self.assertEqual(c.counter, counter)

        #
        # get a recursive dict of all cases (nested)
        #
        cases = cr.list_cases(recurse=True, flat=False, out_stream=None)

        num_cases = count_keys(cases)

        self.assertEqual(num_cases, global_iterations)

    def test_list_cases_nested_no_source(self):
        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=om.NonlinearRunOnce)
        prob.driver = om.ScipyOptimizeDriver(tol=1e-9, disp=False)
        prob.setup()

        model = prob.model
        model.mda.add_recorder(self.recorder)
        model.nonlinear_solver.add_recorder(self.recorder)
        model.mda.nonlinear_solver.add_recorder(self.recorder)

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        # get total iteration count to check against
        global_iterations = len(cr._global_iterations)

        #
        # get a recursive list of all cases (flat)
        #
        cases = cr.list_cases(recurse=True, flat=True, out_stream=None)

        # verify the cases are all there
        self.assertEqual(len(cases), global_iterations)

        # verify the cases are in proper order
        counter = 0
        for i, c in enumerate(cr.get_case(case) for case in cases):
            counter += 1
            self.assertEqual(c.counter, counter)

        #
        # try to get a recursive dict of all cases (nested), without driver or model
        #
        expected_err = ("A nested dictionary of all cases was requested, but "
                        "neither the driver or the model was recorded. Please "
                        "specify another source (system or solver) for the cases "
                        "you want to see.")

        with self.assertRaises(RuntimeError) as cm:
            cases = cr.list_cases(recurse=True, flat=False)

        self.assertEqual(str(cm.exception), expected_err)

    def test_get_cases_recurse(self):
        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=om.NonlinearRunOnce,
                                                       linear_solver=om.ScipyKrylov,
                                                       mda_nonlinear_solver=om.NonlinearBlockGS)
        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)
        prob.driver.add_recorder(self.recorder)
        prob.setup()

        model = prob.model
        model.add_recorder(self.recorder)
        model.mda.add_recorder(self.recorder)
        model.nonlinear_solver.add_recorder(self.recorder)
        model.mda.nonlinear_solver.add_recorder(self.recorder)

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        # get total iteration count to check against
        global_iterations = len(cr._global_iterations)

        #
        # get a recursive list of all cases (flat)
        #
        cases = cr.get_cases(recurse=True, flat=True)

        # verify the cases are all there
        self.assertEqual(len(cases), global_iterations)

        # verify the cases are in proper order
        counter = 0
        for i, c in enumerate(cases):
            counter += 1
            self.assertEqual(c.counter, counter)

        #
        # get a recursive dict of all cases (nested)
        #
        cases = cr.get_cases(recurse=True, flat=False)

        num_cases = count_keys(cases)

        self.assertEqual(num_cases, global_iterations)

        #
        # get a recursive list of child cases
        #
        parent_coord = 'rank0:ScipyOptimize_SLSQP|0|root._solve_nonlinear|0'

        expected_coords = [
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|1',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|2',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|3',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|4',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|5',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|6',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|0|NonlinearBlockGS|7',
            parent_coord + '|NLRunOnce|0|mda._solve_nonlinear|0',
            parent_coord + '|NLRunOnce|0',
            parent_coord
        ]

        cases = cr.get_cases(parent_coord, recurse=True, flat=True)

        # verify the cases are all there and are as expected
        self.assertEqual(len(cases), len(expected_coords))
        for i, c in enumerate(cases):
            self.assertEqual(c.name, expected_coords[i])

        #
        # get a list of cases for each source
        #
        sources = cr.list_sources(out_stream=None)
        self.assertEqual(sorted(sources), [
            'driver', 'root', 'root.mda', 'root.mda.nonlinear_solver', 'root.nonlinear_solver'
        ])

        # verify the coordinates of the returned cases are as expected and that the cases are all there
        expected_coord = {
            'driver':                    r'rank0:ScipyOptimize_SLSQP\|\d',
            'root':                      r'rank0:ScipyOptimize_SLSQP\|\d\|root._solve_nonlinear\|\d',
            'root.nonlinear_solver':     r'rank0:ScipyOptimize_SLSQP\|\d\|root._solve_nonlinear\|\d\|NLRunOnce\|0',
            'root.mda':                  r'rank0:ScipyOptimize_SLSQP\|\d\|root._solve_nonlinear\|\d\|NLRunOnce\|0\|mda._solve_nonlinear\|\d',
            'root.mda.nonlinear_solver': r'rank0:ScipyOptimize_SLSQP\|\d\|root._solve_nonlinear\|\d\|NLRunOnce\|0\|mda._solve_nonlinear\|\d\|NonlinearBlockGS\|\d',
        }
        counter = 0
        mda_counter = 0
        root_counter = 0
        for source in sources:
            expected = expected_coord[source]
            cases = cr.get_cases(source, recurse=False)
            for case in cases:
                counter += 1
                if source.startswith('root.mda'):  # count all cases for/under mda system
                    mda_counter += 1
                if source.startswith('root.'):     # count all cases for/under root solver
                    root_counter += 1
                self.assertRegexpMatches(case.name, expected)

        self.assertEqual(counter, global_iterations)

        #
        # get a recursive list of child cases for the mda system
        #
        counter = 0
        cases = cr.get_cases('root.mda', recurse=True, flat=True)
        for case in cases:
            counter += 1

        self.assertEqual(counter, mda_counter)

        #
        # get a recursive list of child cases for the root solver
        #
        counter = 0
        cases = cr.get_cases('root.nonlinear_solver', recurse=True, flat=True)
        for case in cases:
            counter += 1

        self.assertEqual(counter, root_counter)

    def test_list_outputs(self):
        prob = SellarProblem(nonlinear_solver=om.NonlinearBlockGS,
                             linear_solver=om.ScipyKrylov)

        prob.model.add_recorder(self.recorder)
        prob.model.recording_options['record_residuals'] = True

        prob.setup()

        d1 = prob.model.d1  # SellarDis1withDerivatives (an ExplicitComp)
        d1.add_recorder(self.recorder)

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        # check the system case for 'd1' (there should be only one output, 'd1.y1')
        system_cases = cr.list_cases('root.d1', out_stream=None)
        case = cr.get_case(system_cases[-1])

        outputs = case.list_outputs(explicit=True, implicit=True, val=True,
                                    residuals=True, residuals_tol=None,
                                    units=True, shape=True, bounds=True, desc=True,
                                    scaling=True, hierarchical=True, print_arrays=True,
                                    out_stream=None)

        expected_outputs = {
            'd1.y1': {
                'lower': 0.1,
                'upper': 1000.,
                'ref': 1.0,
                'resids': [1.318e-10],
                'shape': (1,),
                'values': [25.5883024],
                'desc': ''
            }
        }

        self.assertEqual(len(outputs), 1)
        [name, vals] = outputs[0]
        self.assertEqual(name, 'd1.y1')

        expected = expected_outputs[name]
        self.assertEqual(vals['lower'], expected['lower'])
        self.assertEqual(vals['ref'], expected['ref'])
        self.assertEqual(vals['shape'], expected['shape'])
        self.assertEqual(vals['desc'], expected['desc'])
        np.testing.assert_almost_equal(vals['resids'], expected['resids'])
        np.testing.assert_almost_equal(vals['val'], expected['values'])

        # check implicit outputs, there should not be any
        impl_outputs_case = case.list_outputs(explicit=False, implicit=True,
                                              out_stream=None)
        self.assertEqual(len(impl_outputs_case), 0)

        # check that output from the Case method matches output from the System method
        # the system for the case should be properly identified as 'd1'
        stream = StringIO()
        d1.list_outputs(prom_name=True, desc=True, out_stream=stream)
        expected = stream.getvalue().split('\n')

        stream = StringIO()
        case.list_outputs(prom_name=True, desc=True, out_stream=stream)
        text = stream.getvalue().split('\n')

        for i, line in enumerate(expected):
            self.assertEqual(text[i], line)

    def test_list_residuals_tol(self):

        class EComp(om.ExplicitComponent):

            def setup(self):
                self.add_input('x', val=1)
                self.add_output('y', val=1)

            def compute(self, inputs, outputs):
                outputs['y'] = 2*inputs['x']

        class IComp(om.ImplicitComponent):

            def setup(self):
                self.add_input('y', val=1)
                self.add_output('z1', val=1)
                self.add_output('z2', val=1)
                self.add_output('z3', val=1)

            def solve_nonlinear(self, inputs, outputs):
                # only solving z1 so that one specific residual goes to 0
                outputs['z1'] = 2*inputs['y']

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['z1'] = outputs['z1'] - 2*inputs['y']
                residuals['z2'] = outputs['z2'] - 2*inputs['y']
                residuals['z3'] = 2*inputs['y'] - outputs['z3']


        p = om.Problem()
        p.model.add_subsystem('ec', EComp(), promotes=['*'])
        p.model.add_subsystem('ic', IComp(), promotes=['*'])

        p.setup()
        p.add_recorder(self.recorder)
        p.recording_options['record_residuals'] = True

        p.run_model()
        p.model.run_apply_nonlinear()
        p.record('final')

        cr = om.CaseReader(self.filename)
        case = cr.get_case('final')

        # list outputs with residuals
        sysout = sys.stdout
        try:
            capture_stdout = StringIO()
            sys.stdout = capture_stdout
            case.list_outputs(residuals=True)
        finally:
            sys.stdout = sysout

        expected_text = [
            "1 Explicit Output(s) in 'model'",
            "",
            "varname  val   resids",
            "-------  ----  ------",
            "ec",
            "  y      [2.]  [0.]  ",
            "",
            "",
            "3 Implicit Output(s) in 'model'",
            "",
            "varname  val   resids",
            "-------  ----  ------",
            "ic",
            "  z1     [4.]  [0.]  ",
            "  z2     [1.]  [-3.] ",
            "  z3     [1.]  [3.]  ",
            "",
            "",
            "",
        ]
        captured_output = capture_stdout.getvalue()
        for i, line in enumerate(captured_output.split('\n')):
            self.assertEqual(line.strip(), expected_text[i].strip())

        # list outputs filtered by residuals_tol
        sysout = sys.stdout
        try:
            capture_stdout = StringIO()
            sys.stdout = capture_stdout
            case.list_outputs(residuals=True, residuals_tol=1e-2)
        finally:
            sys.stdout = sysout

        # Note: Explicit output has 0 residual, so it should not be included.
        # Note: Implicit outputs Z2 and Z3 should both be shown, because the
        #       tolerance check uses the norm, which is always gives positive.
        expected_text = [
            "0 Explicit Output(s) in 'model'",
            "",
            "",
            "2 Implicit Output(s) in 'model'",
            "",
            "varname  val   resids",
            "-------  ----  ------",
            "ic",
              "z2     [1.]  [-3.]",
              "z3     [1.]  [3.]",
            "",
            "",
            "",
        ]
        captured_output = capture_stdout.getvalue()
        for i, line in enumerate(captured_output.split('\n')):
            self.assertEqual(line.strip(), expected_text[i].strip())

    def test_list_inputs(self):
        prob = SellarProblem(nonlinear_solver=om.NonlinearBlockGS,
                             linear_solver=om.ScipyKrylov)

        prob.model.add_recorder(self.recorder)
        prob.model.recording_options['record_residuals'] = True

        prob.setup()

        d1 = prob.model.d1  # SellarDis1withDerivatives (an ExplicitComp)
        d1.add_recorder(self.recorder)

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        expected_inputs_case = {
            'd1.z': {'val': [5., 2.], 'desc': ''},
            'd1.x': {'val': [1.], 'desc': ''},
            'd1.y2': {'val': [12.0584882], 'desc': ''}
        }

        system_cases = cr.list_cases('root.d1', out_stream=None)

        case = cr.get_case(system_cases[-1])

        inputs = case.list_inputs(val=True, desc=True, out_stream=None)

        for name, meta in inputs:
            expected = expected_inputs_case[name]
            np.testing.assert_almost_equal(meta['val'], expected['val'])
            self.assertEqual(meta['desc'], expected['desc'])

        # check that output from the Case method matches output from the System method
        # the system for the case should be properly identified as 'd1'
        stream = StringIO()
        d1.list_inputs(prom_name=True, desc=True, out_stream=stream)
        expected = stream.getvalue().split('\n')

        stream = StringIO()
        case.list_inputs(prom_name=True, desc=True, out_stream=stream)
        text = stream.getvalue().split('\n')

        for i, line in enumerate(expected):
            self.assertEqual(text[i], line)

    def test_list_inputs_outputs_solver_case(self):
        prob = SellarProblem(SellarDerivativesGrouped)
        prob.setup()

        mda = prob.model.mda
        mda.nonlinear_solver = om.NonlinearBlockGS(maxiter=5)
        mda.nonlinear_solver.add_recorder(self.recorder)

        prob.set_solver_print(-1)
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)
        case = cr.get_case(-1)

        # check that output from the Case methods match output from the System methods
        # the system for the solver case should be properly identified as 'mda'
        stream = StringIO()
        mda.list_inputs(prom_name=True, out_stream=stream)
        expected = stream.getvalue().split('\n')

        stream = StringIO()
        case.list_inputs(prom_name=True, out_stream=stream)
        text = stream.getvalue().split('\n')

        for i, line in enumerate(expected):
            self.assertEqual(text[i], line)

        stream = StringIO()
        mda.list_outputs(prom_name=True, out_stream=stream)
        expected = stream.getvalue().split('\n')

        stream = StringIO()
        case.list_outputs(prom_name=True, out_stream=stream)
        text = stream.getvalue().split('\n')

        for i, line in enumerate(expected):
            self.assertEqual(text[i], line)

    def test_list_inputs_outputs_indep_desvar(self):
        prob = SellarProblem(SellarDerivativesGrouped)
        prob.setup()

        prob.model.recording_options['record_inputs'] = True
        prob.model.add_recorder(self.recorder)

        prob.set_solver_print(-1)
        prob.run_model()
        prob.cleanup()

        cr = om.CaseReader(self.filename)
        case = cr.get_case(-1)

        indeps = case.list_inputs(is_indep_var=True, prom_name=True, out_stream=None)
        self.assertEqual(sorted([name for name, _ in indeps]),
                         ['mda.d1.x', 'mda.d1.z', 'mda.d2.z', 'obj_cmp.x', 'obj_cmp.z'])

        desvars = case.list_inputs(is_design_var=True, out_stream=None)
        self.assertEqual(sorted([name for name, _ in desvars]),
                          ['mda.d1.x', 'mda.d1.z', 'mda.d2.z', 'obj_cmp.x', 'obj_cmp.z'])

        non_desvars = case.list_inputs(is_design_var=False, out_stream=None)
        self.assertEqual(sorted([name for name, _ in non_desvars]),
                         ['con_cmp1.y1', 'con_cmp2.y2',
                          'mda.d1.y2', 'mda.d2.y1',
                          'obj_cmp.y1', 'obj_cmp.y2'])

        nonDV_indeps = case.list_inputs(is_indep_var=True, is_design_var=False, out_stream=None)
        self.assertEqual(sorted([name for name, _ in nonDV_indeps]),
                         [])

        indeps = case.list_outputs(is_indep_var=True, list_autoivcs=True, out_stream=None)
        self.assertEqual(sorted([name for name, _ in indeps]),
                         ['_auto_ivc.v0', '_auto_ivc.v1'])

        desvars = case.list_outputs(is_design_var=True, list_autoivcs=True, out_stream=None)
        self.assertEqual(sorted([name for name, _ in desvars]),
                         ['_auto_ivc.v0', '_auto_ivc.v1'])

        non_desvars = case.list_outputs(is_design_var=False, list_autoivcs=True, out_stream=None)
        self.assertEqual(sorted([name for name, _ in non_desvars]),
                         ['con_cmp1.con1', 'con_cmp2.con2',
                          'mda.d1.y1', 'mda.d2.y2', 'obj_cmp.obj'])

        nonDV_indeps = case.list_outputs(is_indep_var=True, is_design_var=False, list_autoivcs=True, out_stream=None)
        self.assertEqual(sorted([name for name, _ in nonDV_indeps]),
                         [])

    def test_list_input_and_outputs_with_tags(self):
        prob = om.Problem(RectangleCompWithTags())

        recorder = om.SqliteRecorder("cases.sql")
        prob.model.add_recorder(recorder)

        prob.setup(check=False)
        prob.run_model()

        prob.cleanup()

        cr = om.CaseReader("cases.sql")
        cases = cr.get_cases()
        case = cases[0]

        # Inputs no tags
        inputs = case.list_inputs(out_stream=None)
        self.assertEqual(sorted([inp[0] for inp in inputs]), ['length', 'width'])

        # Inputs with tag that matches
        inputs = case.list_inputs(out_stream=None, tags="tag2")
        self.assertEqual([inp[0] for inp in inputs], ['width',])

        # Inputs with tag that does not match
        inputs = case.list_inputs(out_stream=None, tags="tag3")
        self.assertEqual([inp[0] for inp in inputs], [])

        # Inputs with multiple tags
        inputs = case.list_inputs(out_stream=None, tags=["tag2", "tag3"])
        self.assertEqual([inp[0] for inp in inputs], ['width',])

        # Outputs no tags
        outputs = case.list_outputs(out_stream=None)
        self.assertEqual(sorted([outp[0] for outp in outputs]), ['area',])

        # Outputs with tag that does match
        outputs = case.list_outputs(out_stream=None, tags="tag1")
        self.assertEqual(sorted([outp[0] for outp in outputs]), ['area',])

        # Outputs with tag that do not match any vars
        outputs = case.list_outputs(out_stream=None, tags="tag3")
        self.assertEqual(sorted([outp[0] for outp in outputs]), [])

        # Outputs with multiple tags
        outputs = case.list_outputs(out_stream=None, tags=["tag1", "tag3"])
        self.assertEqual(sorted([outp[0] for outp in outputs]), ['area',])

    def test_list_inputs_with_includes_excludes(self):
        prob = SellarProblem()

        prob.model.add_recorder(self.recorder)

        prob.setup()

        d1 = prob.model.d1  # SellarDis1withDerivatives (an ExplicitComp)
        d1.add_recorder(self.recorder)

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        system_cases = cr.list_cases('root.d1', out_stream=None)
        case = cr.get_case(system_cases[-1])

        # inputs with no includes or excludes. Should get d1.z, d1.x, and d1.y2
        inputs = case.list_inputs(out_stream=None)
        self.assertEqual( len(inputs), 3)

        # inputs with includes
        inputs = case.list_inputs(includes = ['*z'], out_stream=None)
        self.assertEqual(len(inputs), 1)

        # inputs with excludes
        inputs = case.list_inputs(excludes = ['*z'], out_stream=None)
        self.assertEqual(len(inputs), 2)

        # inputs with includes and excludes
        inputs = case.list_inputs(includes = ['*z'], excludes = ['d1*'], out_stream=None)
        self.assertEqual(len(inputs), 0)

        # outputs with no includes or excludes. Should get d1.y1
        outputs = case.list_outputs(out_stream=None)
        self.assertEqual( len(outputs), 1)

        # outputs with includes
        outputs = case.list_outputs(includes = ['*z'], out_stream=None)
        self.assertEqual( len(outputs), 0)

        # outputs with excludes
        outputs = case.list_outputs(excludes = ['*z'], out_stream=None)
        self.assertEqual( len(outputs), 1)

        # outputs with includes and excludes
        outputs = case.list_outputs(includes = ['d1*'], excludes = ['*z'], out_stream=None)
        self.assertEqual( len(outputs), 1)

    def test_list_discrete(self):
        model = om.Group()

        model.add_subsystem('expl', ModCompEx(3),
                            promotes_inputs=['x'])
        model.add_subsystem('impl', ModCompIm(3),
                            promotes_inputs=['x'])

        model.add_recorder(self.recorder)

        prob = om.Problem(model)

        prob.setup()

        prob.set_val('x', 11)

        prob.run_model()
        prob.cleanup()

        cr = om.CaseReader(self.filename)
        case = cr.get_case(0)

        #
        # list inputs, not hierarchical
        #
        stream = StringIO()
        model.list_inputs(hierarchical=False, out_stream=stream)
        expected = stream.getvalue().split('\n')

        stream = StringIO()
        case.list_inputs(hierarchical=False, out_stream=stream)
        text = stream.getvalue().split('\n')

        for i, line in enumerate(expected):
            if line and not line.startswith('-'):
                self.assertEqual(text[i], line)

        #
        # list inputs, hierarchical
        #
        stream = StringIO()
        model.list_inputs(hierarchical=True, out_stream=stream)
        expected = stream.getvalue().split('\n')

        stream = StringIO()
        case.list_inputs(hierarchical=True, out_stream=stream)
        text = stream.getvalue().split('\n')

        for i, line in enumerate(expected):
            if line and not line.startswith('-'):
                self.assertEqual(text[i], line)

        #
        # list outputs, not hierarchical, with residuals
        #
        expected = [
            "2 Explicit Output(s) in 'model'",
            "",
            "varname  val    resids      ",
            "-------  -----  ------------",
            "expl.b   [20.]  [0.]        ",
            "expl.y   2      Not Recorded",
        ]

        stream = StringIO()
        case.list_outputs(residuals=True, hierarchical=False, out_stream=stream)
        text = stream.getvalue().split('\n')

        for i, line in enumerate(expected):
            if line and not line.startswith('-'):
                self.assertEqual(remove_whitespace(text[i]), remove_whitespace(line))

        #
        # list outputs, hierarchical
        #
        stream = StringIO()
        model.list_outputs(hierarchical=True, out_stream=stream)
        expected = stream.getvalue().split('\n')

        stream = StringIO()
        case.list_outputs(hierarchical=True, out_stream=stream)
        text = stream.getvalue().split('\n')

        for i, line in enumerate(expected):
            if line and not line.startswith('-'):
                self.assertEqual(text[i], line)

    def test_list_discrete_filtered(self):
        model = om.Group()

        indep = model.add_subsystem('indep', om.IndepVarComp())
        indep.add_discrete_output('x', 11)

        sub = model.add_subsystem('sub', om.Group())
        sub.add_subsystem('expl', ModCompEx(3))
        sub.add_subsystem('impl', ModCompIm(3))

        model.connect('indep.x', 'sub.expl.x')
        model.connect('indep.x', 'sub.impl.x')

        sub.add_recorder(self.recorder)

        # exclude one discrete input (abs_name) and one discrete output (prom_name)
        sub.recording_options['excludes'] = ['sub.impl.x', 'expl.y']

        prob = om.Problem(model)

        prob.setup()
        prob.run_model()
        prob.cleanup()

        cr = om.CaseReader(self.filename)
        case = cr.get_case(0)

        #
        # list inputs
        #
        expected = [
            "2 Input(s) in 'sub'",
            "",
            "varname     val",
            "----------  -----",
            "expl.a  [10.]",
            "expl.x  11   ",
            # sub.impl.x is not recorded (excluded)
        ]

        stream = StringIO()
        case.list_inputs(hierarchical=False, out_stream=stream)
        text = stream.getvalue().split('\n')

        for i, line in enumerate(expected):
            if line and not line.startswith('-'):
                self.assertEqual(remove_whitespace(text[i]), remove_whitespace(line))

        #
        # list outputs
        #
        expected = [
            "1 Explicit Output(s) in 'sub'",
            "",
            "varname   val",
            "--------  -----",
            "expl",
            "  b   [20.]",
            #      y is not recorded (excluded)
            "",
            "",
            "1 Implicit Output(s) in 'sub'",
            "",
            "varname   val",
            "-------   -----",
            "impl",
            "  y   2    ",
        ]

        stream = StringIO()
        case.list_outputs(out_stream=stream)
        text = stream.getvalue().split('\n')

        for i, line in enumerate(expected):
            if line and not line.startswith('-'):
                self.assertEqual(remove_whitespace(text[i]), remove_whitespace(line))

    def test_list_discrete_promoted(self):
        model = om.Group()

        indep = om.IndepVarComp()
        indep.add_discrete_output('x', 11)

        model.add_subsystem('indep', indep, promotes_outputs=['x'])

        model.add_subsystem('expl', ModCompEx(3), promotes_inputs=['x'])
        model.add_subsystem('impl', ModCompIm(3), promotes_inputs=['x'])

        model.add_recorder(self.recorder)

        prob = om.Problem(model)

        prob.setup()
        prob.run_model()
        prob.cleanup()

        cr = om.CaseReader(self.filename)
        case = cr.get_case(0)

        #
        # list inputs
        #
        stream = StringIO()
        model.list_inputs(hierarchical=False, prom_name=True, out_stream=stream)
        expected = stream.getvalue().split('\n')

        stream = StringIO()
        case.list_inputs(hierarchical=False, prom_name=True, out_stream=stream)
        text = stream.getvalue().split('\n')

        for i, line in enumerate(expected):
            if line and not line.startswith('-'):
                self.assertEqual(text[i], line)

        #
        # list outputs
        #
        stream = StringIO()
        model.list_outputs(prom_name=True, out_stream=stream)
        expected = stream.getvalue().split('\n')

        stream = StringIO()
        case.list_outputs(prom_name=True, out_stream=stream)
        text = stream.getvalue().split('\n')

        for i, line in enumerate(expected):
            if line and not line.startswith('-'):
                self.assertEqual(text[i], line)

    def test_getitem(self):
        prob = SellarProblem(nonlinear_solver=om.NonlinearBlockGS,
                             linear_solver=om.ScipyKrylov)
        prob.setup()

        prob.add_recorder(self.recorder)
        prob.driver.add_recorder(self.recorder)
        prob.model.d1.add_recorder(self.recorder)

        prob.run_driver()

        prob.record('final')
        prob.cleanup()

        # expected input and output values after run_once
        expected = {
            # promoted names
            "x": 1.,
            "y1": 25.58830237,
            "y2": 12.05848815,
            "z": [5., 2.],
            "obj": 28.58830817,
            "con1": -22.42830237,
            "con2": -11.94151185,
            # unpromoted output names
            "_auto_ivc.v1": 1.,
            "_auto_ivc.v0": [5., 2.],
            "obj_cmp.obj": 28.58830817,
            "con_cmp1.con1": -22.42830237,
            "con_cmp2.con2": -11.94151185,
            # unpromoted system names
            "d1.x": 1.,
            "d1.y1": 25.58830237,
            "d1.y2": 12.05848815,
            "d1.z": [5., 2.],
        }

        cr = om.CaseReader(self.filename)

        # driver will record design vars, objectives and constraints
        cases = cr.list_cases('driver', recurse=False, out_stream=None)
        case = cr.get_case(cases[0])

        for name in expected:
            if name[0] in ['y', 'd']:
                # driver does not record coupling vars y1 & y2
                # or the lower level inputs and outputs of d1
                msg = "'Variable name \"%s\" not found.'" % name
                with self.assertRaises(KeyError) as cm:
                    case[name]
                self.assertEqual(str(cm.exception), msg)
            else:
                np.testing.assert_almost_equal(case[name], expected[name])

        # problem will record all inputs and outputs at the problem level
        case = cr.get_case('final')

        for name in expected:
            if name in ['d1.x', 'd1.y2', 'd1.z']:
                # problem does not record lower level inputs
                msg = "'Variable name \"%s\" not found.'" % name
                with self.assertRaises(KeyError) as cm:
                    case[name]
                self.assertEqual(str(cm.exception), msg)
            else:
                np.testing.assert_almost_equal(case[name], expected[name])

        # system will record inputs and outputs at the system level
        cases = cr.list_cases('root.d1', out_stream=None)
        case = cr.get_case(cases[-1])

        for name in expected:
            if name[0] in ['p', 'o', 'c']:
                # system d1 does not record params, obj and cons
                msg = "'Variable name \"%s\" not found.'" % name
                with self.assertRaises(KeyError) as cm:
                    case[name]
                self.assertEqual(str(cm.exception), msg)
            else:
                np.testing.assert_almost_equal(case[name], expected[name])

    def test_get_val_exhaustive(self):

        model = om.Group()
        model.add_subsystem('comp', om.ExecComp('y=x-25.',
                                                x={'val': 77.0, 'units': 'degF'},
                                                y={'val': 0.0, 'units': 'degC'}))
        model.add_subsystem('prom', om.ExecComp('yy=xx-25.',
                                                xx={'val': 77.0, 'units': 'degF'},
                                                yy={'val': 0.0, 'units': 'degC'}),
                            promotes=['xx', 'yy'])
        model.add_subsystem('acomp', om.ExecComp('y=x-25.',
                                                 x={'val': np.array([77.0, 95.0]), 'units': 'degF'},
                                                 y={'val': np.array([0., 0.]), 'units': 'degC'}))
        model.add_subsystem('aprom', om.ExecComp('ayy=axx-25.',
                                                 axx={'val': np.array([77.0, 95.0]), 'units': 'degF'},
                                                 ayy={'val': np.array([0., 0.]), 'units': 'degC'}),
                            promotes=['axx', 'ayy'])

        model.add_recorder(self.recorder)

        prob = om.Problem(model)
        prob.setup()
        prob.run_model()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        case = cr.get_case(0)

        assert_near_equal(case.get_val('comp.x'), 77.0, 1e-6)
        assert_near_equal(case.get_val('comp.x', 'degC'), 25.0, 1e-6)
        assert_near_equal(case.get_val('comp.y'), 52., 1e-6)
        assert_near_equal(case.get_val('comp.y', 'degF'), 125.6, 1e-6)

        assert_near_equal(case.get_val('xx'), 77.0, 1e-6)
        assert_near_equal(case.get_val('xx', 'degC'), 25.0, 1e-6)
        assert_near_equal(case.get_val('yy'), 52., 1e-6)
        assert_near_equal(case.get_val('yy', 'degF'), 125.6, 1e-6)

        assert_near_equal(case.get_val('acomp.x', indices=0), 77.0, 1e-6)
        assert_near_equal(case.get_val('acomp.x', indices=[1]), 95.0, 1e-6)
        assert_near_equal(case.get_val('acomp.x', 'degC', indices=[0]), 25.0, 1e-6)
        assert_near_equal(case.get_val('acomp.x', 'degC', indices=1), 35.0, 1e-6)
        assert_near_equal(case.get_val('acomp.y', indices=0), 52., 1e-6)
        assert_near_equal(case.get_val('acomp.y', 'degF', indices=0), 125.6, 1e-6)

        assert_near_equal(case.get_val('axx', indices=0), 77.0, 1e-6)
        assert_near_equal(case.get_val('axx', indices=1), 95.0, 1e-6)
        assert_near_equal(case.get_val('axx', 'degC', indices=0), 25.0, 1e-6)
        assert_near_equal(case.get_val('axx', 'degC', indices=np.array([1])), 35.0, 1e-6)
        assert_near_equal(case.get_val('ayy', indices=0), 52., 1e-6)
        assert_near_equal(case.get_val('ayy', 'degF', indices=0), 125.6, 1e-6)

    def test_get_val_reducable_units(self):

        model = om.Group()
        model.add_subsystem('comp', om.ExecComp('y=x-25.',
                                                x={'val': 77.0, 'units': 'm'},
                                                y={'val': 0.0, 'units': 'm'}))
        model.add_subsystem('acomp', om.ExecComp('tout=tin-25.',
                                                 tin={'val': np.array([77.0, 95.0]), 'units': 'degC'},
                                                 tout={'val': np.array([0., 0.]), 'units': 'degF'}))

        model.add_recorder(self.recorder)

        prob = om.Problem(model)
        prob.setup()
        prob.run_model()

        cr = om.CaseReader(self.filename)

        case = cr.get_case(0)

        for datasrc in [case, prob, prob.model]:
            assert_near_equal(datasrc.get_val('comp.x', units='m/s*s'), 77.0, 1e-6)
            assert_near_equal(datasrc.get_val('comp.x', units='ft/s*s'),
                              om.convert_units(77, 'm', 'ft'), 1e-6)

            assert_near_equal(datasrc.get_val('comp.y', units='m'), 52., 1e-6)
            assert_near_equal(datasrc.get_val('comp.y', units='s*ft/s'),
                              om.convert_units(52, 'm', 'ft'), 1e-6)

            assert_near_equal(datasrc.get_val('acomp.tin', units='degC'), [77., 95.], 1e-6)
            assert_near_equal(datasrc.get_val('acomp.tin', units='degF'),
                              om.convert_units(np.array([77., 95.]), 'degC', 'degF'), 1e-6)
            assert_near_equal(datasrc.get_val('acomp.tin', units='s*degK/s'),
                              om.convert_units(np.array([77., 95.]), 'degC', 'degK'), 1e-6)

            with self.assertRaises(expected_exception=ValueError) as e:
                datasrc.get_val('acomp.tin', units='not_a_unit')
            self.assertEqual("The units 'not_a_unit' are invalid.", str(e.exception))

        prob.set_val('comp.x', val=100.0, units='s*ft/s')
        prob.run_model()
        prob.cleanup()
        cr = om.CaseReader(self.filename)
        case = cr.get_case(0)

        for datasrc in [case, prob, prob.model]:
            assert_near_equal(datasrc.get_val('comp.x', units='m/s*s'),
                              om.convert_units(100.0, 'ft', 'm'),
                              1e-6)
            assert_near_equal(datasrc.get_val('comp.y', units='m*s/s'),
                              om.convert_units(100.0, 'ft', 'm')-25.0,
                              1e-6)

    def test_get_ambiguous_input(self):
        model = om.Group()
        model.add_recorder(self.recorder)

        G1 = model.add_subsystem("G1", om.Group(), promotes=['x'])
        G1.add_subsystem("C0", om.IndepVarComp('x', 1.0, units='m'), promotes=['x'])

        G2 = model.add_subsystem("G2", om.Group(), promotes=['a'])
        G2.add_subsystem("C1", om.ExecComp('y=m*2.0', m={'units': 'm'}), promotes=[('m', 'a')])
        G2.add_subsystem("C2", om.ExecComp('y=f*2.0', f={'units': 'ft'}), promotes=[('f', 'a')])

        model.connect('x', 'a')

        prob = om.Problem(model)
        prob.setup()
        prob.run_model()
        prob.cleanup()

        cr = om.CaseReader(self.filename)
        case = cr.get_case(0)

        assert_near_equal(case.get_val('x'), 1., 1e-6)
        assert_near_equal(case.get_val('x', units='ft'), 3.280839895, 1e-6)

        assert_near_equal(case.get_val('G1.C0.x'), 1., 1e-6)
        assert_near_equal(case.get_val('G1.C0.x', units='ft'), 3.280839895, 1e-6)

        assert_near_equal(case.get_val('G2.C1.m'), 1., 1e-6)
        assert_near_equal(case.get_val('G2.C2.f'), 3.280839895, 1e-6)

        # 'a' is ambiguous.. which input do you want when accessing 'a'?
        msg = "The promoted name 'a' is invalid because it refers to multiple inputs:" + \
              " ['G2.C1.m', 'G2.C2.f']. Access the value using an absolute path name " + \
              "or the connected output variable instead."

        with self.assertRaises(RuntimeError) as cm:
            case['a']
        self.assertEquals(str(cm.exception), msg)

        with self.assertRaises(RuntimeError) as cm:
            case.get_val('a')
        self.assertEquals(str(cm.exception), msg)

        with self.assertRaises(RuntimeError) as cm:
            case.get_val('a', units='m')
        self.assertEquals(str(cm.exception), msg)

        with self.assertRaises(RuntimeError) as cm:
            case.get_val('a', units='ft')
        self.assertEquals(str(cm.exception), msg)

        # 'a' is ambiguous.. which input's units do you want when accessing 'a'?
        # (test the underlying function, currently only called from inside get_val)
        msg = "Can't get units for the promoted name 'a' because it refers to " + \
              "multiple inputs: ['G2.C1.m', 'G2.C2.f']. Access the units using " + \
              "an absolute path name."

        with self.assertRaises(RuntimeError) as cm:
            case._get_units('a')
        self.assertEquals(str(cm.exception), msg)

    def test_get_vars(self):
        prob = SellarProblem(nonlinear_solver=om.NonlinearBlockGS,
                             linear_solver=om.ScipyKrylov)
        prob.setup()

        prob.model.add_recorder(self.recorder)
        prob.model.recording_options['record_residuals'] = True

        prob.driver.add_recorder(self.recorder)

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        driver_cases = cr.list_cases('driver', out_stream=None)
        driver_case = cr.get_case(driver_cases[0])

        desvars = driver_case.get_design_vars()
        objectives = driver_case.get_objectives()
        constraints = driver_case.get_constraints()
        responses = driver_case.get_responses()

        expected_desvars = {"x": 1., "z": [5., 2.]}
        expected_objectives = {"obj": 28.58830817, }
        expected_constraints = {"con1": -22.42830237, "con2": -11.94151185}

        expected_responses = expected_objectives.copy()
        expected_responses.update(expected_constraints)

        for expected_set, actual_set in ((expected_desvars, desvars),
                                         (expected_objectives, objectives),
                                         (expected_constraints, constraints),
                                         (expected_responses, responses)):

            self.assertEqual(len(expected_set), len(actual_set))
            for k in expected_set:
                np.testing.assert_almost_equal(expected_set[k], actual_set[k])

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

        _assert_model_matches_case(case, model)

    def test_load_bad_system_case(self):
        prob = SellarProblem(SellarDerivativesGrouped)

        prob.model.add_recorder(self.recorder)

        driver = prob.driver = om.ScipyOptimizeDriver()
        driver.options['optimizer'] = 'SLSQP'
        driver.options['tol'] = 1e-9
        driver.options['disp'] = False
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
        prob = SellarProblem()
        prob.setup()

        error_msg = "Input variable, '[^']+', recorded in the case is not found in the model"
        with self.assertRaisesRegex(KeyError, error_msg):
            prob.load_case(case)

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

        _assert_model_matches_case(case, model.d2)

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

        _assert_model_matches_case(case, model)

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

    def test_optimization_load_system_cases(self):
        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=om.NonlinearRunOnce,
                                                       linear_solver=om.ScipyKrylov,
                                                       mda_nonlinear_solver=om.NonlinearBlockGS)

        prob.model.add_recorder(self.recorder)

        driver = prob.driver = om.ScipyOptimizeDriver()
        driver.options['optimizer'] = 'SLSQP'
        driver.options['tol'] = 1e-9
        driver.options['disp'] = False
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

        # run the model again with a fresh model
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

        fail = prob.run_driver()
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

        _assert_model_matches_case(case, model)

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
        fail = prob.run_driver()
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

        _assert_model_matches_case(case, model)

    def test_system_options_pickle_fail(self):
        # simple paraboloid model
        model = om.Group()
        ivc = om.IndepVarComp()
        ivc.add_output('x', 3.0)
        model.add_subsystem('subs', ivc)
        subs = model.subs

        # declare two options
        subs.options.declare('options value 1', 1)
        # Given object which can't be pickled
        subs.options.declare('options value to fail', (i for i in []))
        subs.add_recorder(self.recorder)

        prob = om.Problem(model)
        prob.setup()

        msg = ("'subs' <class IndepVarComp>: Trying to record option 'options value to fail' which "
               "cannot be pickled on this system. Set option 'recordable' to False. Skipping "
               "recording options for this system.")
        with assert_warning(om.CaseRecorderWarning, msg):
            prob.run_model()

        prob.cleanup()
        cr = om.CaseReader(self.filename)
        subs_options = cr._system_options['subs']['component_options']

        # no options should have been recorded for d1
        self.assertEqual(len(subs_options._dict), 0)

    def test_pre_load(self):
        prob = SellarProblem(nonlinear_solver=om.NonlinearBlockGS,
                             linear_solver=om.ScipyKrylov)
        prob.setup()

        recorder = om.SqliteRecorder(self.filename)

        prob.add_recorder(recorder)
        prob.driver.add_recorder(recorder)
        prob.model.add_recorder(recorder)
        prob.model.nonlinear_solver.add_recorder(recorder)

        prob.run_driver()
        prob.record('c_1')
        prob.record('c_2')
        prob.cleanup()

        # without pre_load, we should get format_version and metadata but no cases
        cr = om.CaseReader(self.filename, pre_load=False)

        num_driver_cases = len(cr.list_cases('driver', recurse=False, out_stream=None))
        num_system_cases = len(cr.list_cases('root', recurse=False, out_stream=None))
        num_solver_cases = len(cr.list_cases('root.nonlinear_solver', recurse=False, out_stream=None))
        num_problem_cases = len(cr.list_cases('problem', out_stream=None))

        self.assertEqual(num_driver_cases, 1)
        self.assertEqual(num_system_cases, 1)
        self.assertEqual(num_solver_cases, 7)
        self.assertEqual(num_problem_cases, 2)

        self.assertEqual(cr._format_version, format_version)

        self.assertEqual(set(cr.system_options.keys()),
                         set(['root'] + list(prob.model._subsystems_allprocs)))

        self.assertEqual(set(cr.problem_metadata.keys()), {
            'tree', 'sys_pathnames_list', 'connections_list', 'variables', 'abs2prom',
            'driver', 'design_vars', 'responses', 'declare_partials_list', 'md5_hash'
        })

        self.assertEqual(len(cr._driver_cases._cases), 0)
        self.assertEqual(len(cr._system_cases._cases), 0)
        self.assertEqual(len(cr._solver_cases._cases), 0)
        self.assertEqual(len(cr._problem_cases._cases), 0)

        # with pre_load, we should get format_version, metadata and all cases
        cr = om.CaseReader(self.filename, pre_load=True)

        num_driver_cases = len(cr.list_cases('driver', recurse=False, out_stream=None))
        num_system_cases = len(cr.list_cases('root', recurse=False, out_stream=None))
        num_solver_cases = len(cr.list_cases('root.nonlinear_solver', recurse=False, out_stream=None))
        num_problem_cases = len(cr.list_cases('problem', out_stream=None))

        self.assertEqual(num_driver_cases, 1)
        self.assertEqual(num_system_cases, 1)
        self.assertEqual(num_solver_cases, 7)
        self.assertEqual(num_problem_cases, 2)

        self.assertEqual(cr._format_version, format_version)

        self.assertEqual(set(cr.system_options.keys()),
                         set(['root'] + list(prob.model._subsystems_allprocs)))

        self.assertEqual(set(cr.problem_metadata.keys()), {
            'tree', 'sys_pathnames_list', 'connections_list', 'variables', 'abs2prom',
            'driver', 'design_vars', 'responses', 'declare_partials_list', 'md5_hash'
        })

        self.assertEqual(len(cr._driver_cases._cases), num_driver_cases)
        self.assertEqual(len(cr._system_cases._cases), num_system_cases)
        self.assertEqual(len(cr._solver_cases._cases), num_solver_cases)
        self.assertEqual(len(cr._problem_cases._cases), num_problem_cases)

        for case_type in (cr._driver_cases, cr._solver_cases,
                          cr._system_cases, cr._problem_cases):
            for key in case_type.list_cases():
                self.assertTrue(key in case_type._cases)
                self.assertEqual(key, case_type._cases[key].name)

    def test_caching_cases(self):
        prob = SellarProblem(nonlinear_solver=om.NonlinearBlockGS,
                             linear_solver=om.ScipyKrylov)
        prob.setup()

        prob.add_recorder(self.recorder)
        prob.driver.add_recorder(self.recorder)
        prob.model.add_recorder(self.recorder)
        prob.model.nonlinear_solver.add_recorder(self.recorder)

        prob.run_driver()
        prob.record('c_1')
        prob.record('c_2')
        prob.cleanup()

        cr = om.CaseReader(self.filename, pre_load=False)

        self.assertEqual(len(cr._driver_cases._cases), 0)
        self.assertEqual(len(cr._system_cases._cases), 0)
        self.assertEqual(len(cr._solver_cases._cases), 0)
        self.assertEqual(len(cr._problem_cases._cases), 0)

        # get cases without caching them
        for case_type in (cr._driver_cases, cr._solver_cases,
                          cr._system_cases, cr._problem_cases):
            for key in case_type.list_cases():
                case_type.get_case(key)

        self.assertEqual(len(cr._driver_cases._cases), 0)
        self.assertEqual(len(cr._system_cases._cases), 0)
        self.assertEqual(len(cr._solver_cases._cases), 0)
        self.assertEqual(len(cr._problem_cases._cases), 0)

        # get cases and cache them
        for case_type in (cr._driver_cases, cr._solver_cases,
                          cr._system_cases, cr._problem_cases):
            for key in case_type.list_cases():
                case_type.get_case(key, cache=True)

        # assert that we have now stored each of the cases
        self.assertEqual(len(cr._driver_cases._cases), 1)
        self.assertEqual(len(cr._system_cases._cases), 1)
        self.assertEqual(len(cr._solver_cases._cases), 7)
        self.assertEqual(len(cr._problem_cases._cases), 2)

        for case_type in (cr._driver_cases, cr._solver_cases,
                          cr._system_cases, cr._problem_cases):
            for key in case_type.list_cases():
                self.assertTrue(key in case_type._cases)
                self.assertEqual(key, case_type._cases[key].name)

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
        _assert_model_matches_case(case, model)

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

        _assert_model_matches_case(case, model)

    def test_simple_paraboloid_scaled_desvars(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['record_objectives'] = True
        prob.driver.recording_options['record_constraints'] = True
        recorder = om.SqliteRecorder("cases.sql")
        prob.driver.add_recorder(recorder)

        ref = 5.0
        ref0 = -5.0
        model.add_design_var('x', lower=-50.0, upper=50.0, ref=ref, ref0=ref0)
        model.add_design_var('y', lower=-50.0, upper=50.0, ref=ref, ref0=ref0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=10.0, upper=11.0)

        prob.setup(check=False, mode='fwd')

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        # Test values from the last case
        driver_cases = cr.list_cases('driver', out_stream=None)
        last_case = cr.get_case(driver_cases[-1])

        dvs = last_case.get_design_vars(scaled=False)
        unscaled_x = dvs['x'][0]
        unscaled_y = dvs['y'][0]

        dvs = last_case.get_design_vars(scaled=True)
        scaled_x = dvs['x'][0]
        scaled_y = dvs['y'][0]

        adder, scaler = determine_adder_scaler(ref0, ref, None, None)
        self.assertAlmostEqual((unscaled_x + adder) * scaler, scaled_x, places=12)
        self.assertAlmostEqual((unscaled_y + adder) * scaler, scaled_y, places=12)

    def test_reading_all_case_types(self):
        prob = SellarProblem(SellarDerivativesGrouped, nonlinear_solver=om.NonlinearRunOnce,
                                                       linear_solver=om.ScipyKrylov,
                                                       mda_nonlinear_solver=om.NonlinearBlockGS)
        prob.setup(mode='rev')

        driver = prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)

        #
        # Add recorders
        #

        # driver
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.add_recorder(self.recorder)

        # root solver
        nl = prob.model.nonlinear_solver
        nl.recording_options['record_abs_error'] = True
        nl.recording_options['record_rel_error'] = True
        nl.recording_options['record_solver_residuals'] = True
        nl.add_recorder(self.recorder)

        # system
        pz = prob.model.obj_cmp
        pz.recording_options['record_inputs'] = True
        pz.recording_options['record_outputs'] = True
        pz.recording_options['record_residuals'] = True
        pz.add_recorder(self.recorder)

        # mda solver
        nl = prob.model.mda.nonlinear_solver = om.NonlinearBlockGS()
        nl.recording_options['record_abs_error'] = True
        nl.recording_options['record_rel_error'] = True
        nl.recording_options['record_solver_residuals'] = True
        nl.add_recorder(self.recorder)

        # problem
        prob.recording_options['includes'] = []
        prob.recording_options['record_objectives'] = True
        prob.recording_options['record_constraints'] = True
        prob.recording_options['record_desvars'] = True
        prob.add_recorder(self.recorder)

        fail = prob.run_driver()

        prob.record('final')
        prob.cleanup()

        self.assertFalse(fail, 'Problem optimization failed.')

        cr = om.CaseReader(self.filename)

        #
        # check sources
        #

        self.assertEqual(sorted(cr.list_sources(out_stream=None)), [
            'driver', 'problem', 'root.mda.nonlinear_solver', 'root.nonlinear_solver', 'root.obj_cmp'
        ])

        #
        # check system cases
        #

        system_cases = cr.list_cases('root.obj_cmp', recurse=False, out_stream=None)
        expected_cases = [
            'rank0:ScipyOptimize_SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0|obj_cmp._solve_nonlinear|0',
            'rank0:ScipyOptimize_SLSQP|1|root._solve_nonlinear|1|NLRunOnce|0|obj_cmp._solve_nonlinear|1',
            'rank0:ScipyOptimize_SLSQP|2|root._solve_nonlinear|2|NLRunOnce|0|obj_cmp._solve_nonlinear|2',
            'rank0:ScipyOptimize_SLSQP|3|root._solve_nonlinear|3|NLRunOnce|0|obj_cmp._solve_nonlinear|3',
            'rank0:ScipyOptimize_SLSQP|4|root._solve_nonlinear|4|NLRunOnce|0|obj_cmp._solve_nonlinear|4',
            'rank0:ScipyOptimize_SLSQP|5|root._solve_nonlinear|5|NLRunOnce|0|obj_cmp._solve_nonlinear|5',
            'rank0:ScipyOptimize_SLSQP|6|root._solve_nonlinear|6|NLRunOnce|0|obj_cmp._solve_nonlinear|6'
        ]
        self.assertEqual(len(system_cases), len(expected_cases))
        for i, coord in enumerate(system_cases):
            self.assertEqual(coord, expected_cases[i])

        # check inputs, outputs and residuals for last case
        case = cr.get_case(system_cases[-1])

        self.assertEqual(list(case.inputs.keys()), ['x', 'y1', 'y2', 'z'])
        self.assertEqual(case.inputs['y1'], prob['y1'])
        self.assertEqual(case.inputs['y2'], prob['y2'])

        self.assertEqual(list(case.outputs.keys()), ['obj'])
        self.assertEqual(case.outputs['obj'], prob['obj'])

        self.assertEqual(list(case.residuals.keys()), ['obj'])
        self.assertEqual(case.residuals['obj'][0], 0.)

        #
        # check solver cases
        #

        root_solver_cases = cr.list_cases('root.nonlinear_solver', recurse=False, out_stream=None)
        expected_cases = [
            'rank0:ScipyOptimize_SLSQP|0|root._solve_nonlinear|0|NLRunOnce|0',
            'rank0:ScipyOptimize_SLSQP|1|root._solve_nonlinear|1|NLRunOnce|0',
            'rank0:ScipyOptimize_SLSQP|2|root._solve_nonlinear|2|NLRunOnce|0',
            'rank0:ScipyOptimize_SLSQP|3|root._solve_nonlinear|3|NLRunOnce|0',
            'rank0:ScipyOptimize_SLSQP|4|root._solve_nonlinear|4|NLRunOnce|0',
            'rank0:ScipyOptimize_SLSQP|5|root._solve_nonlinear|5|NLRunOnce|0',
            'rank0:ScipyOptimize_SLSQP|6|root._solve_nonlinear|6|NLRunOnce|0'
        ]
        self.assertEqual(len(root_solver_cases), len(expected_cases))
        for i, coord in enumerate(root_solver_cases):
            self.assertEqual(coord, expected_cases[i])

        case = cr.get_case(root_solver_cases[-1])

        expected_inputs = ['x', 'y1', 'y2', 'z']
        expected_outputs = ['con1', 'con2', 'obj', 'x', 'y1', 'y2', 'z']

        # input values must be accessed using absolute path names
        expected_inputs_abs = [
            'mda.d1.x', 'obj_cmp.x',
            'mda.d2.y1', 'obj_cmp.y1', 'con_cmp1.y1',
            'mda.d1.y2', 'obj_cmp.y2', 'con_cmp2.y2',
            'mda.d1.z', 'mda.d2.z', 'obj_cmp.z'
        ]

        self.assertEqual(sorted(case.inputs.keys()), expected_inputs)
        self.assertEqual(sorted(case.outputs.keys()), expected_outputs)
        self.assertEqual(sorted(case.residuals.keys()), expected_outputs)

        for key in expected_inputs_abs:
            np.testing.assert_almost_equal(case.inputs[key], prob[key])

        for key in expected_outputs:
            np.testing.assert_almost_equal(case.outputs[key], prob[key])

        np.testing.assert_almost_equal(case.abs_err, 0, decimal=6)
        np.testing.assert_almost_equal(case.rel_err, 0, decimal=6)

        #
        # check mda solver cases
        #

        # check that there are multiple iterations and mda solver is part of the coordinate
        mda_solver_cases = cr.list_cases('root.mda.nonlinear_solver', recurse=False, out_stream=None)
        self.assertTrue(len(mda_solver_cases) > 1)
        for coord in mda_solver_cases:
            self.assertTrue('mda._solve_nonlinear' in coord)

        case = cr.get_case(mda_solver_cases[-1])

        expected_inputs = ['x', 'y1', 'y2', 'z']
        expected_outputs = ['y1', 'y2']

        # input values must be accessed using absolute path names
        expected_inputs_abs = [
            'mda.d1.x',
            'mda.d1.y2',
            'mda.d1.z',
            'mda.d2.y1',
            'mda.d2.z'
        ]

        self.assertEqual(sorted(case.inputs.keys()), expected_inputs)
        self.assertEqual(sorted(case.outputs.keys()), expected_outputs)
        self.assertEqual(sorted(case.residuals.keys()), expected_outputs)

        # check that output from the Case method matches output from the System method
        # the system for the case should be properly identified as 'd1'
        stream = StringIO()
        prob.model.mda.list_inputs(prom_name=True, out_stream=stream)
        expected = stream.getvalue().split('\n')

        stream = StringIO()
        case.list_inputs(prom_name=True, out_stream=stream)
        text = stream.getvalue().split('\n')

        for i, line in enumerate(expected):
            self.assertEqual(text[i], line)

        for key in expected_inputs_abs:
            np.testing.assert_almost_equal(case.inputs[key], prob[key])

        for key in expected_outputs:
            np.testing.assert_almost_equal(case.outputs[key], prob[key])

        np.testing.assert_almost_equal(case.abs_err, 0, decimal=6)
        np.testing.assert_almost_equal(case.rel_err, 0, decimal=6)

        # check that the recurse option returns root and mda solver cases plus child system cases
        all_solver_cases = cr.list_cases('root.nonlinear_solver', recurse=True, flat=True, out_stream=None)
        self.assertEqual(len(all_solver_cases),
                         len(root_solver_cases) + len(mda_solver_cases) + len(system_cases))

        #
        # check driver cases
        #

        driver_cases = cr.list_cases('driver', recurse=False, out_stream=None)
        expected_cases = [
            'rank0:ScipyOptimize_SLSQP|0',
            'rank0:ScipyOptimize_SLSQP|1',
            'rank0:ScipyOptimize_SLSQP|2',
            'rank0:ScipyOptimize_SLSQP|3',
            'rank0:ScipyOptimize_SLSQP|4',
            'rank0:ScipyOptimize_SLSQP|5',
            'rank0:ScipyOptimize_SLSQP|6'
        ]
        # check that there are multiple iterations and they have the expected coordinates
        self.assertTrue(len(driver_cases), len(expected_cases))
        for i, coord in enumerate(driver_cases):
            self.assertEqual(coord, expected_cases[i])

        # check VOI values from last driver iteration
        case = cr.get_case(driver_cases[-1])

        expected_dvs = {
            "z": prob['z'],
            "x": prob['x']
        }
        expected_obj = {
            "obj": prob['obj_cmp.obj']
        }
        expected_con = {
            "con1": prob['con_cmp1.con1'],
            "con2": prob['con_cmp2.con2']
        }

        dvs = case.get_design_vars()
        obj = case.get_objectives()
        con = case.get_constraints()

        self.assertEqual(dvs.keys(), expected_dvs.keys())
        for key in expected_dvs:
            np.testing.assert_almost_equal(dvs[key], expected_dvs[key])

        self.assertEqual(obj.keys(), expected_obj.keys())
        for key in expected_obj:
            np.testing.assert_almost_equal(obj[key], expected_obj[key])

        self.assertEqual(con.keys(), expected_con.keys())
        for key in expected_con:
            np.testing.assert_almost_equal(con[key], expected_con[key])

        # check accessing values via outputs attribute
        expected_outputs = expected_dvs
        expected_outputs.update(expected_obj)
        expected_outputs.update(expected_con)

        self.assertEqual(set(case.outputs.keys()), set(expected_outputs.keys()))
        for key in expected_outputs:
            np.testing.assert_almost_equal(case.outputs[key], expected_outputs[key])

        # check that the recurse option also returns system and solver cases (all_solver_cases)
        all_driver_cases = cr.list_cases('driver', recurse=True, flat=True, out_stream=None)

        expected_cases = driver_cases + \
            [c for c in all_solver_cases if c.startswith('rank0:ScipyOptimize_SLSQP')]

        self.assertEqual(len(all_driver_cases), len(expected_cases))
        for case in expected_cases:
            self.assertTrue(case in all_driver_cases)

    def test_abs_rel_error(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', ImplCompTwoStates())

        # mda solver
        nl = model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        nl.options['maxiter'] = 2
        nl.recording_options['record_abs_error'] = True
        nl.recording_options['record_rel_error'] = True
        nl.recording_options['record_solver_residuals'] = True
        nl.add_recorder(self.recorder)

        prob.setup()
        prob.set_val('comp.y', 8.0)
        prob.set_val('comp.z', 5.0)
        prob.run_model()
        prob.cleanup()

        cr = om.CaseReader(self.filename)
        case = cr.get_case(cr.list_cases()[-1])

        norm =  nl._iter_get_norm()
        norm0 = nl._norm0
        self.assertEqual(case.abs_err, norm)
        self.assertEqual(case.rel_err, norm/norm0)

    def test_linesearch(self):
        prob = om.Problem()

        model = prob.model

        model.add_subsystem('px', om.IndepVarComp('x', 1.0))
        model.add_subsystem('comp', ImplCompTwoStates())
        model.connect('px.x', 'comp.x')

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        model.nonlinear_solver.options['maxiter'] = 3
        model.nonlinear_solver.options['iprint'] = 2
        model.linear_solver = om.ScipyKrylov()

        ls = model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='vector')
        ls.options['maxiter'] = 3
        ls.options['alpha'] = 1.0

        # add recorder to nonlinear solver, linesearch solver and model
        model.nonlinear_solver.add_recorder(self.recorder)
        model.nonlinear_solver.linesearch.add_recorder(self.recorder)
        model.comp.add_recorder(self.recorder)
        model.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(0)

        prob['px.x'] = 2.0
        prob['comp.y'] = 0.0
        prob['comp.z'] = 1.6
        prob.run_model()
        prob.cleanup()

        expected = [
            'rank0:root._solve_nonlinear|0|Newton_subsolve|0',
            'rank0:root._solve_nonlinear|0|NewtonSolver|0',
            'rank0:root._solve_nonlinear|0|NewtonSolver|1|ArmijoGoldsteinLS|0',
            'rank0:root._solve_nonlinear|0|NewtonSolver|1|ArmijoGoldsteinLS|1',
            'rank0:root._solve_nonlinear|0|NewtonSolver|1|ArmijoGoldsteinLS|2',
            'rank0:root._solve_nonlinear|0|NewtonSolver|1',
            'rank0:root._solve_nonlinear|0|NewtonSolver|2|ArmijoGoldsteinLS|0',
            'rank0:root._solve_nonlinear|0|NewtonSolver|2|ArmijoGoldsteinLS|1',
            'rank0:root._solve_nonlinear|0|NewtonSolver|2|ArmijoGoldsteinLS|2',
            'rank0:root._solve_nonlinear|0|NewtonSolver|2',
            'rank0:root._solve_nonlinear|0'
        ]

        cr = om.CaseReader(self.filename)

        for i, c in enumerate(cr.list_cases(out_stream=None)):
            case = cr.get_case(c)

            coord = case.name
            self.assertEqual(coord, expected[i])

            # check the source
            if 'ArmijoGoldsteinLS' in coord:
                self.assertEqual(case.source, 'root.nonlinear_solver.linesearch')
            elif 'Newton' in coord:
                self.assertEqual(case.source, 'root.nonlinear_solver')
            else:
                self.assertEqual(case.source, 'root')

    def test_case_array_list_vars_options(self):

        class ArrayAdder(om.ExplicitComponent):
            """
            A simple component that has array inputs and outputs
            """
            def __init__(self, size):
                super().__init__()
                self.size = size

            def setup(self):
                self.add_input('x', val=np.zeros(self.size), units='inch')
                self.add_output('y', val=np.zeros(self.size), units='ft')

            def compute(self, inputs, outputs):
                outputs['y'] = inputs['x'] + 10.0

        size = 100  # how many items in the array

        prob = om.Problem()

        prob.model.add_subsystem('des_vars', om.IndepVarComp('x', np.ones(size), units='inch'),
                                 promotes=['x'])
        prob.model.add_subsystem('mult', ArrayAdder(size), promotes=['x', 'y'])

        recorder = om.SqliteRecorder("cases.sql")
        prob.model.add_recorder(recorder)

        prob.model.recording_options['record_inputs'] = True
        prob.model.recording_options['record_outputs'] = True
        prob.model.recording_options['record_residuals'] = True

        prob.setup()

        prob['x'] = np.ones(size)

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")
        system_cases = cr.list_cases(out_stream=None)
        case = cr.get_case(system_cases[0])

        # list inputs
        # out_stream - not hierarchical - extras - no print_arrays
        stream = StringIO()
        case.list_inputs(val=True,
                         units=True,
                         prom_name=True,
                         hierarchical=False,
                         print_arrays=False,
                         out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(1, text.count("1 Input(s) in 'model'"))
        self.assertEqual(1, text.count('mult.x'))
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(4, num_non_empty_lines)
        self.assertEqual(1, text.count('mult.x   |10.0|  inch   x'))

        # out_stream - hierarchical - extras - no print_arrays
        stream = StringIO()
        case.list_inputs(val=True,
                         units=True,
                         shape=True,
                         hierarchical=True,
                         print_arrays=False,
                         out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(1, text.count("1 Input(s) in 'model'"))
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(5, num_non_empty_lines)
        self.assertEqual(1, text.count('\nmult'))
        self.assertEqual(1, text.count('\n  x      |10.0|  inch   (100'))

        # list outputs
        # out_stream - not hierarchical - extras - no print_arrays
        stream = StringIO()
        case.list_outputs(val=True,
                          units=True,
                          shape=True,
                          bounds=True,
                          residuals=True,
                          scaling=True,
                          hierarchical=False,
                          print_arrays=False,
                          out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(text.count('2 Explicit Output'), 1)
        # make sure they are in the correct order
        # FIXME: disabled until Case orders outputs
        # self.assertTrue(text.find("des_vars.x") < text.find('mult.y'))
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(6, num_non_empty_lines)

        # Promoted names - no print arrays
        stream = StringIO()
        case.list_outputs(val=True,
                          prom_name=True,
                          print_arrays=False,
                          out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(text.count('  x       |10.0|   x'), 1)
        self.assertEqual(text.count('  y       |110.0|  y'), 1)
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(num_non_empty_lines, 8)

        # Hierarchical - no print arrays
        stream = StringIO()
        case.list_outputs(val=True,
                          units=True,
                          shape=True,
                          bounds=True,
                          residuals=True,
                          scaling=True,
                          hierarchical=True,
                          print_arrays=False,
                          out_stream=stream)
        text = stream.getvalue()
        self.assertEqual(text.count('\ndes_vars'), 1)
        self.assertEqual(text.count('\n  x'), 1)
        self.assertEqual(text.count('\nmult'), 1)
        self.assertEqual(text.count('\n  y'), 1)
        num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
        self.assertEqual(num_non_empty_lines, 8)

        # Need to explicitly set this to make sure all ways of running this test
        #   result in the same format of the output. When running this test from the
        #   top level via testflo, the format comes out different than if the test is
        #   run individually
        opts = {
            'edgeitems': 3,
            'infstr': 'inf',
            'linewidth': 75,
            'nanstr': 'nan',
            'precision': 8,
            'suppress': False,
            'threshold': 1000,
        }

        from packaging.version import Version
        if Version(np.__version__) >= Version("1.14"):
            opts['legacy'] = '1.13'

        with printoptions(**opts):
            # list outputs
            # out_stream - not hierarchical - extras - print_arrays
            stream = StringIO()
            case.list_outputs(val=True,
                              units=True,
                              shape=True,
                              bounds=True,
                              residuals=True,
                              scaling=True,
                              hierarchical=False,
                              print_arrays=True,
                              out_stream=stream)
            text = stream.getvalue()
            self.assertEqual(text.count('2 Explicit Output'), 1)
            self.assertEqual(text.count('val:'), 2)
            self.assertEqual(text.count('resids:'), 2)
            self.assertEqual(text.count('['), 4)
            # make sure they are in the correct order
            # FIXME: disabled until Case orders outputs
            # self.assertTrue(text.find("des_vars.x") < text.find('mult.y'))
            num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
            self.assertEqual(44, num_non_empty_lines)

            # Hierarchical
            stream = StringIO()
            case.list_outputs(val=True,
                              units=True,
                              shape=True,
                              bounds=True,
                              residuals=True,
                              scaling=True,
                              hierarchical=True,
                              print_arrays=True,
                              out_stream=stream)
            text = stream.getvalue()
            self.assertEqual(text.count('2 Explicit Output'), 1)
            self.assertEqual(text.count('val:'), 2)
            self.assertEqual(text.count('resids:'), 2)
            self.assertEqual(text.count('['), 4)
            self.assertEqual(text.count('\ndes_vars'), 1)
            self.assertEqual(text.count('\n  x'), 1)
            self.assertEqual(text.count('\nmult'), 1)
            self.assertEqual(text.count('\n  y'), 1)
            num_non_empty_lines = sum([1 for s in text.splitlines() if s.strip()])
            self.assertEqual(num_non_empty_lines, 46)

    def test_system_metadata_attribute_deprecated(self):
        model = om.Group()
        model.add_recorder(self.recorder)
        prob = om.Problem(model)
        prob.setup()
        prob.run_model()
        prob.cleanup()

        cr = om.CaseReader(self.filename)
        msg = "The BaseCaseReader.system_metadata attribute is deprecated. " \
        "Use `list_model_options` instead."
        with assert_warning(OMDeprecationWarning, msg):
            options = cr.system_metadata

    def test_system_options_attribute_deprecated(self):
        model = om.Group()
        model.add_recorder(self.recorder)
        prob = om.Problem(model)
        prob.setup()
        prob.run_model()
        prob.cleanup()

        cr = om.CaseReader(self.filename)
        msg = "The system_options attribute is deprecated. Use `list_model_options` instead."
        with assert_warning(OMDeprecationWarning, msg):
            options = cr.system_options

    def test_sqlite_reader_problem_derivatives(self):

        class UglyParaboloid(om.ExplicitComponent):
            """
            A version of the Paraboloid component with some odd but valid
            variable names... for testing the parsing of derivative keys.
            """
            def setup(self):
                self.add_input('x,1', val=0.0)
                self.add_input('y:2', val=0.0)
                self.add_output('f(xy)', val=0.0)
                self.declare_partials('*', '*')

            def compute(self, inputs, outputs):
                x = inputs['x,1']
                y = inputs['y:2']
                outputs['f(xy)'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0

            def compute_partials(self, inputs, partials):
                x = inputs['x,1']
                y = inputs['y:2']
                partials['f(xy)', 'x,1'] = 2.0*x - 6.0 + y
                partials['f(xy)', 'y:2'] = 2.0*y + 8.0 + x

        # make a model with some messy var names
        model = om.Group()
        model.add_subsystem('p1', om.IndepVarComp('x,1', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y:2', 50.0), promotes=['*'])
        model.add_subsystem('comp', UglyParaboloid(), promotes=['*'])

        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes_outputs=['c'])
        model.connect('x,1', 'con.x')
        model.connect('y:2', 'con.y')

        model.add_design_var('x,1', lower=-50.0, upper=50.0)
        model.add_design_var('y:2', lower=-50.0, upper=50.0)
        model.add_objective('f(xy)')
        model.add_constraint('c', upper=-15.0)

        prob = om.Problem(model)
        prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)
        prob.recording_options['record_derivatives'] = True
        recorder = om.SqliteRecorder('cases.sql')

        prob.add_recorder(recorder)

        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()

        case_name = "c1"
        prob.record(case_name)

        prob.cleanup()

        cr = om.CaseReader('cases.sql')

        num_problem_cases = len(cr.list_cases('problem', out_stream=None))
        self.assertEqual(num_problem_cases, 1)

        c1 = cr.get_case('c1')

        J = prob.compute_totals()
        np.testing.assert_almost_equal(c1.derivatives[('f(xy)', 'x,1')], J[('comp.f(xy)', 'p1.x,1')])
        np.testing.assert_almost_equal(c1.derivatives[('f(xy)', 'y:2')], J[('comp.f(xy)', 'p2.y:2')])
        np.testing.assert_almost_equal(c1.derivatives[('c', 'x,1')], J[('con.c', 'p1.x,1')])
        np.testing.assert_almost_equal(c1.derivatives[('c', 'y:2')], J[('con.c', 'p2.y:2')])

    def test_comma_comp(self):
        class CommaComp(om.ExplicitComponent):

            def setup(self):
                self.add_input('some_{input,withcommas}', val=3)
                self.add_output('an_{output,withcommas}', val=10)
                self.declare_partials('*', '*', method='fd')

            def compute(self, inputs, outputs):
                outputs['an_{output,withcommas}'] = 2*inputs['some_{input,withcommas}']**2

        p = om.Problem()

        p.model.add_subsystem('dv', om.IndepVarComp('some_{input,withcommas}', val=26.), promotes=['*'])
        p.model.add_subsystem('comma_comp', CommaComp(), promotes=['*'])

        recorder = om.SqliteRecorder('cases.sql')
        p.add_recorder(recorder)

        p.recording_options['record_derivatives'] = True

        p.model.add_design_var('some_{input,withcommas}', upper=100, lower=-100)
        p.model.add_objective('an_{output,withcommas}')

        p.setup()
        p.run_driver()
        p.record('final')

        J = p.compute_totals()

        cr = om.CaseReader("cases.sql")
        case = cr.get_case('final')

        for deriv_key in J:
            np.testing.assert_almost_equal(case.derivatives[deriv_key], J[deriv_key])

    def test_list_cases(self):
        prob = SellarProblem(nonlinear_solver=om.NonlinearBlockGS,
                             linear_solver=om.ScipyKrylov)
        prob.setup()

        prob.add_recorder(self.recorder)
        prob.driver.add_recorder(self.recorder)
        prob.model.d1.add_recorder(self.recorder)

        prob.run_driver()

        prob.record('final')
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        cases_set = set(cr.list_cases(out_stream=None))

        expected_set = {'rank0:Driver|0|root._solve_nonlinear|0|d1._solve_nonlinear|0',
            'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|1|d1._solve_nonlinear|1',
            'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|2|d1._solve_nonlinear|2',
            'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|3|d1._solve_nonlinear|3',
            'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|4|d1._solve_nonlinear|4',
            'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|5|d1._solve_nonlinear|5',
            'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|6|d1._solve_nonlinear|6',
            'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|7|d1._solve_nonlinear|7',
            'rank0:Driver|0',
            'final'}

        self.assertSetEqual(cases_set, expected_set)

    def test_list_cases_format(self):
        prob = SellarProblem(nonlinear_solver=om.NonlinearBlockGS,
                             linear_solver=om.ScipyKrylov)
        prob.setup()

        prob.add_recorder(self.recorder)
        prob.driver.add_recorder(self.recorder)
        prob.model.d1.add_recorder(self.recorder)

        prob.run_driver()

        prob.record('final')
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        expected_cases = [
            'system',
            '    rank0:Driver|0|root._solve_nonlinear|0|d1._solve_nonlinear|0',
            '    rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|1|d1._solve_nonlinear|1',
            '    rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|2|d1._solve_nonlinear|2',
            '    rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|3|d1._solve_nonlinear|3',
            '    rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|4|d1._solve_nonlinear|4',
            '    rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|5|d1._solve_nonlinear|5',
            '    rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|6|d1._solve_nonlinear|6',
            '    rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|7|d1._solve_nonlinear|7',
            'driver',
            '    rank0:Driver|0',
            'problem',
            '    final',
        ]

        stream = StringIO()
        cases = cr.list_cases(out_stream=stream)
        text = stream.getvalue().split('\n')
        for i, line in enumerate(expected_cases):
            self.assertEqual(text[i], line)

    def test_list_sources_format(self):
        prob = SellarProblem()
        prob.setup()

        prob.add_recorder(self.recorder)
        prob.driver.add_recorder(self.recorder)
        prob.model.d1.add_recorder(self.recorder)

        prob.run_driver()

        prob.record('final')
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        expected_sources = [
            'driver',
            'root.d1',
            'problem'
        ]

        with self.assertRaises(TypeError) as cm:
            cr.list_sources('problem')
        self.assertTrue(str(cm.exception), "Invalid output stream specified for 'out_stream'.")

        stream = StringIO()
        cases = cr.list_sources(out_stream=stream)
        text = stream.getvalue().split('\n')
        for i, line in enumerate(expected_sources):
            self.assertEqual(text[i], line)

    def test_list_source_vars_format(self):
        prob = SellarProblem()
        prob.setup()

        prob.add_recorder(self.recorder)
        prob.driver.add_recorder(self.recorder)
        prob.model.d1.add_recorder(self.recorder)

        prob.run_driver()

        prob.record('final')
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        expected_cases = [
            'outputs',
            '    z',
            '    x',
            '    obj',
            '    con2',
            '    con1'
        ]

        stream = StringIO()
        cases = cr.list_source_vars('driver', out_stream=stream)
        text = sorted(stream.getvalue().split('\n'), reverse=True)
        for i, line in enumerate(expected_cases):
            self.assertEqual(text[i], line)

    def test_get_openmdao_version(self):
        prob = SellarProblem()
        prob.setup()

        prob.add_recorder(self.recorder)
        prob.driver.add_recorder(self.recorder)
        prob.model.d1.add_recorder(self.recorder)

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        print(cr.openmdao_version)
        self.assertEqual(openmdao_version, cr.openmdao_version)

    def test_hierarchical_model(self):
        # test with model that has a deep system hierarchy
        model = MultipointBeamGroup(E=1., L=1., b=0.1, volume=0.01,
                                    num_elements=50, num_cp=5,
                                    num_load_cases=2)

        prob = om.Problem(model)
        prob.setup()

        model.add_recorder(self.recorder)
        model.parallel.sub_0.add_recorder(self.recorder)
        model.parallel.sub_0.compliance_comp.add_recorder(self.recorder)

        prob.run_model()

        reader = om.CaseReader(self.filename)

        # check that sources are recorded properly
        sources = sorted(reader.list_sources(out_stream=None), reverse=True)
        self.assertEqual(sources, [
            'root.parallel.sub_0.compliance_comp',
            'root.parallel.sub_0',
            'root'
        ])

        # there should be one case from each source
        cases = reader.list_cases(out_stream=None)
        self.assertEqual(cases, [
            'rank0:root._solve_nonlinear|0|NLRunOnce|0|parallel._solve_nonlinear|0|NLRunOnce|0|parallel.sub_0._solve_nonlinear|0|NLRunOnce|0|parallel.sub_0.compliance_comp._solve_nonlinear|0',
            'rank0:root._solve_nonlinear|0|NLRunOnce|0|parallel._solve_nonlinear|0|NLRunOnce|0|parallel.sub_0._solve_nonlinear|0',
            'rank0:root._solve_nonlinear|0'
        ])

        # check that we can properly list cases for each source
        for i, src in enumerate(sources):
            self.assertEqual(reader.list_cases(src, recurse=False, out_stream=None), [cases[i]])

    def test_hierarchical_solvers(self):
        # test with model that has a deep solver hierarchy
        p = om.Problem()

        cycle1 = p.model.add_subsystem('cycle1', om.Group(), promotes=['*'])
        cycle1_1 = cycle1.add_subsystem('cycle1_1', om.Group(), promotes=['*'])
        cycle1_1.add_subsystem('comp', om.ExecComp('x1 = 3 + x2'), promotes=["*"])

        cycle1_2 = cycle1.add_subsystem('cycle1_2', om.Group(), promotes=['*'])
        cycle1_2.add_subsystem('comp',  om.ExecComp('x2 = 3 + x1 + y'), promotes=["*"])

        cycle2 = p.model.add_subsystem('cycle2', om.Group(), promotes=['*'])
        cycle2.add_subsystem('comp', om.ExecComp('y = x1 + 2'), promotes=['*'])

        cycle1.nonlinear_solver.add_recorder(self.recorder)
        cycle1_1.nonlinear_solver.add_recorder(self.recorder)
        cycle1_2.nonlinear_solver.add_recorder(self.recorder)
        cycle2.nonlinear_solver.add_recorder(self.recorder)

        p.setup()
        p.run_model()

        reader = om.CaseReader(self.filename)

        # check that sources are recorded properly
        sources = sorted(reader.list_sources(out_stream=None))
        self.assertEqual(sources, [
            'root.cycle1.cycle1_1.nonlinear_solver',
            'root.cycle1.cycle1_2.nonlinear_solver',
            'root.cycle1.nonlinear_solver',
            'root.cycle2.nonlinear_solver'
        ])

        # there should be one case from each source
        cases = reader.list_cases(out_stream=None)
        self.assertEqual(cases, [
            'rank0:root._solve_nonlinear|0|NLRunOnce|0|cycle1._solve_nonlinear|0|NLRunOnce|0|cycle1.cycle1_1._solve_nonlinear|0|NLRunOnce|0',
            'rank0:root._solve_nonlinear|0|NLRunOnce|0|cycle1._solve_nonlinear|0|NLRunOnce|0|cycle1.cycle1_2._solve_nonlinear|0|NLRunOnce|0',
            'rank0:root._solve_nonlinear|0|NLRunOnce|0|cycle1._solve_nonlinear|0|NLRunOnce|0',
            'rank0:root._solve_nonlinear|0|NLRunOnce|0|cycle2._solve_nonlinear|0|NLRunOnce|0'
        ])

        # check that we can properly list cases for each source
        for i, src in enumerate(sources):
            self.assertEqual(reader.list_cases(src, recurse=False, out_stream=None), [cases[i]])

    @unittest.skipIf(OPT is None, "pyoptsparse is not installed")
    @unittest.skipIf(OPTIMIZER is None, "pyoptsparse is not providing SNOPT or SLSQP")
    def test_constraints_with_aliases(self):
        p = om.Problem()

        exec = om.ExecComp(['y = x**2',
                            'z = a * x**2'],
                        a={'shape': (1,)},
                        y={'shape': (3,)},
                        x={'shape': (3,)},
                        z={'shape': (3,)})

        p.model.add_subsystem('exec', exec)

        p.model.add_design_var('exec.a', lower=-1000, upper=1000)
        p.model.add_objective('exec.y', index=1)
        p.model.add_constraint('exec.z', indices=[0], equals=25)
        p.model.add_constraint('exec.z', indices=[-1], lower=20, alias="ALIAS_TEST")
        p.model.add_constraint('exec.z', indices=[1], lower=0, alias="con|with->scaling", ref=11.0)

        driver = p.driver = om.pyOptSparseDriver()
        driver.recording_options['record_derivatives'] = True
        driver.add_recorder(om.SqliteRecorder('cases.sql'))

        p.setup()

        p.set_val('exec.x', np.array([7.0, 2.0, 4.0]))

        p.run_driver()

        p.cleanup()
        cr = om.CaseReader('cases.sql')
        case = cr.get_case(1)
        derivs = case.derivatives

        assert_near_equal(derivs['exec.z', 'exec.a'].ravel(), 49.0)
        assert_near_equal(derivs['ALIAS_TEST', 'exec.a'].ravel(), 16.0)
        assert_near_equal(derivs['con|with->scaling', 'exec.a'].ravel(), 4.0/11.0)

        cons = case.get_constraints()
        con_vals = driver.get_constraint_values()

        assert_near_equal(cons['exec.z'], con_vals['exec.z'])
        assert_near_equal(cons['ALIAS_TEST'], con_vals['ALIAS_TEST'])
        assert_near_equal(cons['con|with->scaling'], con_vals['con|with->scaling'])


@use_tempdirs
class TestFeatureSqliteReader(unittest.TestCase):

    def test_feature_list_cases(self):

        prob = om.Problem(model=SellarMDA())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        driver = prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9)

        driver.add_recorder(om.SqliteRecorder('cases.sql'))

        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader('cases.sql')

        case_names = cr.list_cases(out_stream=None)

        self.assertEqual(len(case_names), driver.iter_count)
        self.assertEqual(case_names, ['rank0:ScipyOptimize_SLSQP|%d' % i for i in range(driver.iter_count)])
        self.assertEqual('', '')

        for name in case_names:
            case = cr.get_case(name)
            self.assertEqual(case, case)

    def test_feature_get_cases(self):

        prob = om.Problem(model=SellarMDA())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        driver = prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9)
        driver.add_recorder(om.SqliteRecorder('cases.sql'))

        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader('cases.sql')

        cases = cr.get_cases()

        self.assertEqual(len(cases), driver.iter_count)

        for case in cases:
            self.assertEqual(case, case)

    def test_feature_get_cases_nested(self):

        # define Sellar MDA problem
        prob = om.Problem(model=SellarMDA())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        # add recorder to the driver, model and solver
        recorder = om.SqliteRecorder('cases.sql')

        prob.driver.add_recorder(recorder)
        model.add_recorder(recorder)
        model.nonlinear_solver.add_recorder(recorder)

        # run the problem
        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()
        prob.cleanup()

        # get the last driver case
        cr = om.CaseReader('cases.sql')

        driver_cases = cr.list_cases('driver')
        last_driver_case = driver_cases[-1]

        # get a recursive dict of all child cases of the last driver case
        cases = cr.get_cases(last_driver_case, recurse=True, flat=False)

        # access the last driver case and it's children
        for case in cases:
            self.assertEqual(case, case)
            for child_case in cases[case]:
                self.assertEqual(child_case, child_case)
                for grandchild in cases[case][child_case]:
                    self.assertEqual(grandchild, grandchild)

    def test_feature_list_sources(self):

        # define Sellar MDA problem
        prob = om.Problem(model=SellarMDA())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        prob.setup()

        # add recorder to the driver, model and solver
        recorder = om.SqliteRecorder('cases.sql')

        prob.driver.add_recorder(recorder)
        model.add_recorder(recorder)
        model.cycle.nonlinear_solver.add_recorder(recorder)
        model.nonlinear_solver.add_recorder(recorder)

        # run the problem
        prob.set_solver_print(0)
        prob.run_driver()
        prob.cleanup()

        # examine cases to see what was recorded
        cr = om.CaseReader('cases.sql')

        sources = cr.list_sources()
        self.assertEqual(sorted(sources), ['driver', 'root', 'root.cycle.nonlinear_solver', 'root.nonlinear_solver'])

        driver_vars = cr.list_source_vars('driver')
        self.assertEqual(('inputs:', sorted(driver_vars['inputs']), 'outputs:', sorted(driver_vars['outputs'])),
                         ('inputs:', [], 'outputs:', ['con1', 'con2', 'obj', 'x', 'z']))

        model_vars = cr.list_source_vars('root')
        self.assertEqual(('inputs:', sorted(model_vars['inputs']), 'outputs:', sorted(model_vars['outputs'])),
                         ('inputs:', ['x', 'y1', 'y2', 'z'], 'outputs:', ['con1', 'con2', 'obj', 'x', 'y1', 'y2', 'z']))

        solver_vars = cr.list_source_vars('root.nonlinear_solver')
        self.assertEqual(('inputs:', sorted(solver_vars['inputs']), 'outputs:', sorted(solver_vars['outputs'])),
                         ('inputs:', ['x', 'y1', 'y2', 'z'], 'outputs:', ['con1', 'con2', 'obj', 'x', 'y1', 'y2', 'z']))

    def test_feature_reading_driver_derivatives(self):

        prob = om.Problem(model=SellarMDA())

        model = prob.model
        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        driver = prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)
        driver.recording_options['record_derivatives'] = True

        driver.add_recorder(om.SqliteRecorder('cases.sql'))

        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader('cases.sql')

        # Get derivatives associated with the last iteration.
        derivs = cr.get_case(-1).derivatives

        # check that derivatives have been recorded.
        self.assertEqual(set(derivs.keys()), set([
            ('obj', 'z'), ('con2', 'z'), ('con1', 'x'),
            ('obj', 'x'), ('con2', 'x'), ('con1', 'z')
        ]))

        # Get specific derivative.
        assert_near_equal(derivs['obj', 'z'], derivs['obj', 'z'])

    def test_feature_recording_option_precedence(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

        model.set_input_defaults('x', val=50.0)
        model.set_input_defaults('y', val=50.0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=15.0)

        filename = "cases.sql"
        recorder = om.SqliteRecorder(filename)

        prob.driver.add_recorder(recorder)
        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['record_outputs'] = True
        prob.driver.recording_options['includes'] = []
        prob.driver.recording_options['excludes'] = []

        prob.set_solver_print(0)
        prob.setup()
        prob.run_driver()
        prob.cleanup()

        # First case with record_desvars = True and includes = []
        cr = om.CaseReader(filename)
        case = cr.get_case(-1)

        self.assertEqual(sorted(case.outputs.keys()), ['c', 'f_xy', 'x', 'y'])

        # Second case with record_desvars = False and includes = []
        recorder = om.SqliteRecorder(filename)
        prob.driver.add_recorder(recorder)
        prob.driver.recording_options['record_desvars'] = False
        prob.driver.recording_options['record_outputs'] = True
        prob.driver.recording_options['includes'] = []

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(filename)
        case = cr.get_case(0)

        self.assertEqual(sorted(case.outputs.keys()), ['c', 'f_xy'])

        # Third case with record_desvars = True and includes = ['*']
        recorder = om.SqliteRecorder(filename)
        prob.driver.add_recorder(recorder)
        prob.driver.recording_options['record_desvars'] = True
        prob.driver.recording_options['record_outputs'] = True
        prob.driver.recording_options['includes'] = ['*']

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(filename)
        case = cr.get_case(0)

        self.assertEqual(sorted(case.outputs.keys()), ['c', 'f_xy', 'x', 'y'])

        # Fourth case with record_desvars = False, record_outputs = True, and includes = ['*']
        recorder = om.SqliteRecorder(filename)
        prob.driver.add_recorder(recorder)
        prob.driver.recording_options['record_desvars'] = False
        prob.driver.recording_options['record_outputs'] = True
        prob.driver.recording_options['includes'] = ['*']

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader(filename)
        case = cr.get_case(0)

        self.assertEqual(sorted(case.outputs.keys()), ['c', 'f_xy', 'x', 'y'])

    def test_feature_driver_options_with_values(self):

        model = SellarDerivatives(nonlinear_solver=om.NonlinearBlockGS,
                                  linear_solver=om.ScipyKrylov)

        model.add_design_var('z', lower=np.array([-10.0, 0.0]),
                                  upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob = om.Problem(model)

        driver = prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        driver.add_recorder(om.SqliteRecorder("cases.sql"))

        driver.recording_options['includes'] = []
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['record_desvars'] = True

        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        driver_cases = cr.list_cases('driver')
        case = cr.get_case(driver_cases[0])

        self.assertEqual(sorted(case.outputs.keys()), ['con1', 'con2', 'obj', 'x', 'z'])

        objs = case.get_objectives()
        cons = case.get_constraints()
        dvs = case.get_design_vars()
        rsps = case.get_responses()

        # keys() will give you the promoted variable names
        self.assertEqual((sorted(objs.keys()), sorted(cons.keys()), sorted(dvs.keys())),
                         (['obj'], ['con1', 'con2'], ['x', 'z']))

        # alternatively, you can get the absolute names
        self.assertEqual((sorted(objs.absolute_names()), sorted(cons.absolute_names()), sorted(dvs.absolute_names())),
                         (['obj_cmp.obj'], ['con_cmp1.con1', 'con_cmp2.con2'], ['x', 'z']))

        # you can access variable values using either the promoted or the absolute name
        self.assertEqual((objs['obj'], objs['obj_cmp.obj']), (objs['obj_cmp.obj'], objs['obj']))
        self.assertEqual((dvs['x'], dvs['_auto_ivc.v1']), (dvs['_auto_ivc.v1'], dvs['x']))
        self.assertEqual((rsps['obj'], rsps['obj_cmp.obj']), (rsps['obj_cmp.obj'], rsps['obj']))

        # you can also access the variables directly from the case object
        self.assertEqual((case['obj'], case['obj_cmp.obj']), (objs['obj_cmp.obj'], objs['obj']))
        self.assertEqual((case['x'], case['_auto_ivc.v1']), (dvs['_auto_ivc.v1'], dvs['x']))

    def test_feature_list_inputs_and_outputs(self):
        prob = SellarProblem(nonlinear_solver=om.NonlinearBlockGS,
                             linear_solver=om.ScipyKrylov)

        recorder = om.SqliteRecorder("cases.sql")
        prob.model.add_recorder(recorder)
        prob.model.recording_options['record_residuals'] = True

        prob.setup()

        d1 = prob.model.d1
        d1.add_recorder(recorder)

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        system_cases = cr.list_cases('root.d1')

        case = cr.get_case(system_cases[1])

        # list_inputs will print a report to the screen
        case_inputs = sorted(case.list_inputs())

        assert_near_equal(case_inputs[0][1]['val'], [1.], tolerance=1e-10) # d1.x
        assert_near_equal(case_inputs[1][1]['val'], [12.27257053], tolerance=1e-10) # d1.y2
        assert_near_equal(case_inputs[2][1]['val'], [5., 2.], tolerance=1e-10) # d1.z

        case_outputs = case.list_outputs(prom_name=True)

        assert_near_equal(case_outputs[0][1]['val'], [25.545485893882876], tolerance=1e-10) # d1.y1

    def test_feature_list_inputs_and_outputs_with_tags(self):

        class RectangleCompWithTags(om.ExplicitComponent):
            """
            A simple Explicit Component that also has input and output with tags.
            """

            def setup(self):
                self.add_input('length', val=1., tags=["tag1", "tag2"])
                self.add_input('width', val=1., tags=["tag2"])
                self.add_output('area', val=1., tags="tag1")

            def setup_partials(self):
                self.declare_partials('*', '*')

            def compute(self, inputs, outputs):
                outputs['area'] = inputs['length'] * inputs['width']

        model = om.Group()
        prob = om.Problem(model)
        model.add_recorder(om.SqliteRecorder('cases.sql'))

        model.add_subsystem('rect', RectangleCompWithTags(), promotes=['length', 'width', 'area'])

        prob.setup(check=False)

        prob.set_val('length', 100.0)
        prob.set_val('width', 60.0)

        prob.run_model()

        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        cases = cr.get_cases()
        case = cases[0]

        # Inputs with tag that matches
        inputs = case.list_inputs(out_stream=None, tags="tag1")
        self.assertEqual(sorted([inp[0] for inp in inputs]), sorted(['rect.length',]))

        # Inputs with multiple tags
        inputs = case.list_inputs(out_stream=None, tags=["tag1", "tag2"])
        self.assertEqual(sorted([inp[0] for inp in inputs]), sorted(['rect.width', 'rect.length']))

        # Outputs with tag that does match
        outputs = case.list_outputs(tags="tag1")
        self.assertEqual(sorted([outp[0] for outp in outputs]), ['rect.area',])

    def test_feature_list_inputs_and_outputs_with_includes_excludes(self):

        model = om.Group()
        prob = om.Problem(model)
        model.add_recorder(om.SqliteRecorder('cases.sql'))

        model.add_subsystem('rect', RectangleComp(), promotes=['length', 'width', 'area'])

        prob.setup(check=False)

        prob.set_val('length', 100.)
        prob.set_val('width', 60.0)

        prob.run_model()

        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        cases = cr.get_cases()
        case = cases[0]

        # Inputs with includes
        inputs = case.list_inputs(includes=['*length'], out_stream=None)
        self.assertEqual(sorted([inp[0] for inp in inputs]), sorted(['rect.length',]))

        # Inputs with multiple tags
        inputs = case.list_inputs(excludes=['*length'], out_stream=None)
        self.assertEqual(sorted([inp[0] for inp in inputs]), sorted(['rect.width',]))

        # Outputs with includes
        outputs = case.list_outputs(includes=['*area'], out_stream=None)
        self.assertEqual(sorted([outp[0] for outp in outputs]), ['rect.area',])

        # Inputs with excludes
        inputs = case.list_inputs(excludes=['*length'], out_stream=None)
        self.assertEqual(sorted(['rect.width']), sorted([inp[0] for inp in inputs]))

    def test_feature_get_val(self):

        model = om.Group()
        model.add_recorder(om.SqliteRecorder('cases.sql'))

        speed = om.ExecComp('v=x/t', x={'units': 'm'}, t={'units': 's'}, v={'units': 'm/s'})

        model.add_subsystem('speed', speed, promotes=['x', 't', 'v'])

        prob = om.Problem(model)
        prob.setup()

        prob.set_val('x', 100., units='m')
        prob.set_val('t', 60., units='s')

        prob.run_model()
        prob.cleanup()

        cr = om.CaseReader('cases.sql')
        case = cr.get_case(0)

        assert_near_equal(case['x'], 100., 1e-6)
        assert_near_equal(case.get_val('x', units='ft'), 328.0839895, 1e-6)

        assert_near_equal(case['v'], 100./60., 1e-6)
        assert_near_equal(case.get_val('v', units='ft/s'), 5.46807, 1e-6)

    def test_feature_sqlite_reader_read_problem_derivatives_multiple_recordings(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('egg_crate', EggCrate(), promotes=['x', 'y', 'f_xy'])
        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)

        prob.recording_options['record_derivatives'] = True
        recorder = om.SqliteRecorder('cases.sql')
        prob.add_recorder(recorder)

        prob.setup()

        prob.set_solver_print(0)

        prob.set_val('x', 2.5)
        prob.set_val('y', 2.5)

        prob.run_driver()
        assert_near_equal([prob.get_val('x'), prob.get_val('y'), prob.get_val('f_xy')],
                          [[3.01960159], [3.01960159], [18.97639468]], 1e-6)
        case_name_1 = "c1"
        prob.record(case_name_1)


        prob.set_val('x', 0.1)
        prob.set_val('y', -0.1)
        prob.run_driver()
        assert_near_equal([prob.get_val('x'), prob.get_val('y'), prob.get_val('f_xy')],
                          [[-2.14311975e-08], [2.14312031e-08], [2.388341e-14]], 1e-6)

        case_name_2 = "c2"
        prob.record(case_name_2)
        prob.cleanup()

        cr = om.CaseReader('cases.sql')

        num_problem_cases = len(cr.list_cases('problem'))
        self.assertEqual(num_problem_cases, 2)

        c1 = cr.get_case(case_name_1)
        c2 = cr.get_case(case_name_2)

        # check that derivatives have been recorded properly.
        assert_near_equal(c1.derivatives[('f_xy', 'x')][0], 0.0, 1e-4)
        assert_near_equal(c1.derivatives[('f_xy', 'y')][0], 0.0, 1e-4)

        assert_near_equal(c2.derivatives[('f_xy', 'x')][0], 0.0, 1e-4)
        assert_near_equal(c2.derivatives[('f_xy', 'y')][0], 0.0, 1e-4)


@use_tempdirs
class TestPromAbsDict(unittest.TestCase):

    def test_dict_functionality_pre_autoivc(self):
        prob = SellarProblem(SellarDerivativesGroupedPreAutoIVC)
        driver = prob.driver = om.ScipyOptimizeDriver(disp=False)

        recorder = om.SqliteRecorder("cases.sql")

        driver.add_recorder(recorder)
        driver.recording_options['includes'] = []
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_derivatives'] = True

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        driver_cases = cr.list_cases('driver', out_stream=None)
        driver_case = cr.get_case(driver_cases[-1])

        dvs = driver_case.get_design_vars()
        derivs = driver_case.derivatives

        # verify that map looks and acts like a regular dict
        self.assertTrue(isinstance(dvs, dict))
        self.assertEqual(sorted(dvs.keys()), ['x', 'z'])
        self.assertEqual(sorted(dvs.items()), [('x', dvs['x']), ('z', dvs['z'])])

        # verify that using absolute names works the same as using promoted names
        self.assertEqual(sorted(dvs.absolute_names()), ['px.x', 'pz.z'])
        self.assertEqual(dvs['x'], dvs['x'])
        self.assertEqual(dvs['pz.z'][0], dvs['z'][0])
        self.assertEqual(dvs['pz.z'][1], dvs['z'][1])

        # verify we can set the value using either promoted or absolute name as key
        # (although users wouldn't normally do this, it's used when copying or scaling)
        dvs['x'] = 111.
        self.assertEqual(dvs['x'], 111.)
        self.assertEqual(dvs['px.x'], 111.)

        dvs['px.x'] = 222.
        self.assertEqual(dvs['x'], 222.)
        self.assertEqual(dvs['px.x'], 222.)

        # verify deriv keys are tuples as expected, both promoted and absolute
        self.assertEqual(set(derivs.keys()), set([
            ('obj', 'z'), ('con2', 'z'), ('con1', 'x'),
            ('obj', 'x'), ('con2', 'x'), ('con1', 'z')
        ]))
        self.assertEqual(set(derivs.absolute_names()), set([
            ('obj_cmp.obj', 'pz.z'), ('con_cmp2.con2', 'pz.z'), ('con_cmp1.con1', 'px.x'),
            ('obj_cmp.obj', 'px.x'), ('con_cmp2.con2', 'px.x'), ('con_cmp1.con1', 'pz.z')
        ]))

        # verify we can access derivs via tuple or string, with promoted or absolute names
        J = prob.compute_totals(of=['obj'], wrt=['x'])
        expected = J[('obj', 'x')]
        np.testing.assert_almost_equal(derivs[('obj', 'x')], expected, decimal=6)
        np.testing.assert_almost_equal(derivs[('obj', 'px.x')], expected, decimal=6)
        np.testing.assert_almost_equal(derivs[('obj_cmp.obj', 'px.x')], expected, decimal=6)
        np.testing.assert_almost_equal(derivs['obj!x'], expected, decimal=6)
        np.testing.assert_almost_equal(derivs['obj!px.x'], expected, decimal=6)
        np.testing.assert_almost_equal(derivs['obj_cmp.obj!x'], expected, decimal=6)

        # verify we can set derivs via tuple or string, with promoted or absolute names
        # (although users wouldn't normally do this, it's used when copying)
        for key, value in [(('obj', 'x'), 111.), (('obj', 'px.x'), 222.),
                           ('obj_cmp.obj!x', 333.), ('obj_cmp.obj!px.x', 444.)]:
            derivs[key] = value
            self.assertEqual(derivs[('obj', 'x')], value)
            self.assertEqual(derivs[('obj', 'px.x')], value)
            self.assertEqual(derivs[('obj_cmp.obj', 'px.x')], value)
            self.assertEqual(derivs['obj!x'], value)
            self.assertEqual(derivs['obj!px.x'], value)
            self.assertEqual(derivs['obj_cmp.obj!x'], value)

        # verify that we didn't mess up deriv keys by setting values
        self.assertEqual(set(derivs.keys()), set([
            ('obj', 'z'), ('con2', 'z'), ('con1', 'x'),
            ('obj', 'x'), ('con2', 'x'), ('con1', 'z')
        ]))
        self.assertEqual(set(derivs.absolute_names()), set([
            ('obj_cmp.obj', 'pz.z'), ('con_cmp2.con2', 'pz.z'), ('con_cmp1.con1', 'px.x'),
            ('obj_cmp.obj', 'px.x'), ('con_cmp2.con2', 'px.x'), ('con_cmp1.con1', 'pz.z')
        ]))

    def test_dict_functionality(self):
        prob = SellarProblem(SellarDerivativesGrouped)
        driver = prob.driver = om.ScipyOptimizeDriver(disp=False)

        recorder = om.SqliteRecorder("cases.sql")

        driver.add_recorder(recorder)
        driver.recording_options['includes'] = []
        driver.recording_options['record_objectives'] = True
        driver.recording_options['record_constraints'] = True
        driver.recording_options['record_desvars'] = True
        driver.recording_options['record_derivatives'] = True

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader("cases.sql")

        driver_cases = cr.list_cases('driver', out_stream=None)
        driver_case = cr.get_case(driver_cases[-1])

        dvs = driver_case.get_design_vars()
        derivs = driver_case.derivatives

        # verify that map looks and acts like a regular dict
        self.assertTrue(isinstance(dvs, dict))
        self.assertEqual(sorted(dvs.keys()), ['x', 'z'])
        self.assertEqual(sorted(dvs.items()), [('x', dvs['x']), ('z', dvs['z'])])

        # verify that using absolute names works the same as using promoted names
        self.assertEqual(sorted(dvs.absolute_names()), ['x', 'z'])
        self.assertEqual(dvs['x'], dvs['x'])
        self.assertEqual(dvs['z'][0], dvs['z'][0])
        self.assertEqual(dvs['z'][1], dvs['z'][1])

        dvs['x'] = 111.
        self.assertEqual(dvs['x'], 111.)

        dvs['x'] = 222.
        self.assertEqual(dvs['x'], 222.)

        # verify deriv keys are tuples as expected, both promoted and absolute
        self.assertEqual(set(derivs.keys()), set([
            ('obj', 'z'), ('con2', 'z'), ('con1', 'x'),
            ('obj', 'x'), ('con2', 'x'), ('con1', 'z')
        ]))
        self.assertEqual(set(derivs.absolute_names()), set([
            ('obj_cmp.obj', 'z'), ('con_cmp2.con2', 'z'), ('con_cmp1.con1', 'x'),
            ('obj_cmp.obj', 'x'), ('con_cmp2.con2', 'x'), ('con_cmp1.con1', 'z')
        ]))

        # verify we can access derivs via tuple or string, with promoted or absolute names
        J = prob.compute_totals(of=['obj'], wrt=['x'])
        expected = J[('obj', 'x')]
        np.testing.assert_almost_equal(derivs[('obj', 'x')], expected, decimal=6)
        np.testing.assert_almost_equal(derivs[('obj_cmp.obj', 'x')], expected, decimal=6)
        np.testing.assert_almost_equal(derivs['obj!x'], expected, decimal=6)
        np.testing.assert_almost_equal(derivs['obj_cmp.obj!x'], expected, decimal=6)

        # verify we can set derivs via tuple or string, with promoted or absolute names
        # (although users wouldn't normally do this, it's used when copying)
        for key, value in [(('obj', 'x'), 111.), ('obj_cmp.obj!x', 444.)]:
            derivs[key] = value
            self.assertEqual(derivs[('obj', 'x')], value)
            self.assertEqual(derivs[('obj_cmp.obj', 'x')], value)
            self.assertEqual(derivs['obj!x'], value)
            self.assertEqual(derivs['obj_cmp.obj!x'], value)

        # verify that we didn't mess up deriv keys by setting values
        self.assertEqual(set(derivs.keys()), set([
            ('obj', 'z'), ('con2', 'z'), ('con1', 'x'),
            ('obj', 'x'), ('con2', 'x'), ('con1', 'z')
        ]))
        self.assertEqual(set(derivs.absolute_names()), set([
            ('obj_cmp.obj', 'z'), ('con_cmp2.con2', 'z'), ('con_cmp1.con1', 'x'),
            ('obj_cmp.obj', 'x'), ('con_cmp2.con2', 'x'), ('con_cmp1.con1', 'z')
        ]))


def _assert_model_matches_case(case, system):
    """
    Check to see if the values in the case match those in the model.

    Parameters
    ----------
    case: Case object
        Case to be used for the comparison.
    system: System object
        System to be used for the comparison.
    """
    case_inputs = case.inputs
    model_inputs = system._inputs
    for name, model_input in model_inputs._abs_item_iter(flat=False):
        np.testing.assert_almost_equal(case_inputs[name], model_input)

    case_outputs = case.outputs
    model_outputs = system._outputs
    for name, model_output in model_outputs._abs_item_iter(flat=False):
        np.testing.assert_almost_equal(case_outputs[name], model_output)


@use_tempdirs
class TestSqliteCaseReaderLegacy(unittest.TestCase):

    legacy_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'legacy_sql')

    # the change from v12 to v13 is just adding the openmdao version.
    # the tests below should already be able to test the ability to read a file without that

    def test_options_v12(self):

        # The case reader should handle an old database that does not have
        # the system and solver options recorded
        filename = os.path.join(self.legacy_dir, 'case_problem_driver_v8.sql')

        cr = om.CaseReader(filename)

        with assert_warning(UserWarning, 'System options not recorded.'):
            options = cr.list_model_options()

        with assert_warning(UserWarning, 'Solver options not recorded.'):
            options = cr.list_solver_options()

        # The case reader should handle a v11 database that had a
        # different separator for runs in the model option keys
        filename = os.path.join(self.legacy_dir, 'case_problem_v11.sql')

        cr = om.CaseReader(filename)

        stream = StringIO()

        cr.list_model_options(run_number=1, out_stream=stream)

        text = stream.getvalue().split('\n')

        expected = [
            "Run Number: 1",
            "    Subsystem : root",
            "        assembled_jac_type: dense",
            "    Subsystem : p1",
            "        distributed: False",
            "        name: UNDEFINED",
            "        val: 1.0",
            "        shape: None",
            "        units: None",
            "        res_units: None",
            "        desc: None",
            "        lower: None",
            "        upper: None",
            "        ref: 1.0",
            "        ref0: 0.0",
            "        res_ref: None",
            "        tags: None",
            "    Subsystem : p2",
            "        distributed: False",
            "        name: UNDEFINED",
            "        val: 1.0",
            "        shape: None",
            "        units: None",
            "        res_units: None",
            "        desc: None",
            "        lower: None",
            "        upper: None",
            "        ref: 1.0",
            "        ref0: 0.0",
            "        res_ref: None",
            "        tags: None",
            "    Subsystem : comp",
            "        distributed: False",
            "    Subsystem : con",
            "        distributed: False",
            "        has_diag_partials: False",
            "        units: None",
            "        shape: None",
            ""
        ]

        for i, line in enumerate(text):
            self.assertEqual(line, expected[i])

    def test_problem_v9(self):

        # The change from v9 to v10 was changing adding the abs_err and rel_err
        #   data from the top level solver to the Problem recording.
        # The case reader code should seamlessly handle reading Problem cases
        #   that do not contain the abs_err and rel_err values. The Case will have
        #   values of None for those two.
        # We can re-use the v8 legacy sql file since all we need is a case recorder
        #   file with a Problem case in it

        filename = os.path.join(self.legacy_dir, 'case_problem_driver_v8.sql')

        cr = om.CaseReader(filename)

        case = cr.get_case('final')

        self.assertIsNone(case.abs_err)
        self.assertIsNone(case.rel_err)

    def test_problem_v8(self):

        # The change from v8 to v9 was changing the character to split the derivatives from
        # 'of,wrt' to 'of!wrt' to allow for commas

        # The v8 case file used in this test was created with this code:

        # class CommaComp(om.ExplicitComponent):

        #     def setup(self):
        #         self.add_input('input_var', val=3)
        #         self.add_output('output_var', val=10)
        #         self.declare_partials('*', '*', method='fd')

        #     def compute(self, inputs, outputs):
        #         outputs['output_var'] = 2*inputs['input_var']**2

        # p = om.Problem()

        # p.model.add_subsystem('dv', om.IndepVarComp('input_var', val=26.),
        #                       promotes=['*'])
        # p.model.add_subsystem('comma_comp', CommaComp(), promotes=['*'])

        # recorder = om.SqliteRecorder('case_problem_driver_v8.sql')
        # p.add_recorder(recorder)

        # p.recording_options['record_derivatives'] = True

        # p.model.add_design_var('input_var', upper=100, lower=-100)
        # p.model.add_objective('output_var')

        # p.setup()
        # p.run_driver()
        # p.record('final')

        filename = os.path.join(self.legacy_dir, 'case_problem_driver_v8.sql')

        cr = om.CaseReader(filename)

        case = cr.get_case('final')

        np.testing.assert_almost_equal(case.derivatives._values[0], 104.000002011162)

    def test_problem_v7(self):

        # the change from v7 to v8 was adding the recording of input, output, and residuals to problem
        # and also recording of output and residuals to Driver.
        # check to make sure reading a v7 file works when reading problem and driver cases

        # The v7 case file used in this test was created with this code:

        # prob = SellarProblem(SellarDerivativesGrouped)
        # prob.setup()
        #
        # driver = prob.driver = om.ScipyOptimizeDriver(disp=False, tol=1e-9)
        #
        # recorder = om.SqliteRecorder('case_problem_driver_v7.sql')
        #
        # driver.recording_options['record_desvars'] = True
        # driver.recording_options['record_objectives'] = True
        # driver.recording_options['record_constraints'] = True
        # driver.add_recorder(recorder)
        #
        # prob.recording_options['includes'] = []
        # prob.recording_options['record_objectives'] = True
        # prob.recording_options['record_constraints'] = True
        # prob.recording_options['record_desvars'] = True
        # prob.add_recorder(recorder)
        #
        # fail = prob.run_driver()
        #
        # prob.record('final')
        # prob.cleanup()

        filename = os.path.join(self.legacy_dir, 'case_problem_driver_v7.sql')

        cr = om.CaseReader(filename)

        # check that we can get correct values from the driver iterations:
        seventh_slsqp_iteration_case = cr.get_case('rank0:ScipyOptimize_SLSQP|6')
        np.testing.assert_almost_equal(seventh_slsqp_iteration_case.outputs['z'],
                                      [1.97846296, -2.21388305e-13], decimal=2)

        # check that we can get correct values from the problem cases
        problem_case = cr.get_case('final')
        self.assertEqual(sorted(problem_case.outputs.keys()), sorted(['con1', 'con2', 'obj',
                                                                      'x', 'z']))
        np.testing.assert_almost_equal(problem_case.outputs['z'],
                                      [1.97846296, -2.21388305e-13], decimal=2)

    def test_problem_v6(self):
        # the change from v6 to v7 was adding the derivatives to problem
        # check to make sure reading a v6 file works when reading problem cases

        # The case file was created with this code:

        # prob = om.Problem(model=SellarDerivatives())
        #
        # model = prob.model
        # model.add_design_var('z', lower=np.array([-10.0, 0.0]),
        #                      upper=np.array([10.0, 10.0]))
        # model.add_design_var('x', lower=0.0, upper=10.0)
        # model.add_objective('obj')
        # model.add_constraint('con1', upper=0.0)
        # model.add_constraint('con2', upper=0.0)
        #
        # prob.add_recorder(om.SqliteRecorder("case_problem_v6.sql"))
        #
        # prob.setup()
        # prob.run_driver()
        #
        # prob.record('final')
        # prob.cleanup()

        filename = os.path.join(self.legacy_dir, 'case_problem_v6.sql')

        cr = om.CaseReader(filename)

        #
        # check sources
        #

        self.assertEqual(sorted(cr.list_sources(out_stream=None)), [
            'problem',
        ])

        case = cr.get_case('final')

        q = case.outputs.keys()

        self.assertEqual(sorted(q), sorted(['con1', 'con2', 'obj', 'x', 'y1', 'y2', 'z']))

    def test_driver_v5(self):
        """ Not a big change to v6 but make sure reading of driver data from v5 works. """

        # Case file created using this code

        # prob = om.Problem()
        #
        # model = prob.model
        # model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        # model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        # model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        # model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])
        #
        # model.add_design_var('x', lower=-50.0, upper=50.0)
        # model.add_design_var('y', lower=-50.0, upper=50.0)
        # model.add_objective('f_xy')
        # model.add_constraint('c', upper=-15.0)
        #
        # driver = prob.driver = om.ScipyOptimizeDriver()
        # driver.options['optimizer'] = 'SLSQP'
        # driver.options['tol'] = 1e-9
        #
        # driver.recording_options['record_desvars'] = True
        # driver.recording_options['record_objectives'] = True
        # driver.recording_options['record_constraints'] = True
        #
        # case_recorder_filename = 'case_driver_v5.sql'
        #
        # recorder = om.SqliteRecorder(case_recorder_filename)
        # prob.driver.add_recorder(recorder)
        #
        # prob.setup()
        # prob.run_driver()
        # prob.cleanup()

        filename = os.path.join(self.legacy_dir, 'case_driver_v5.sql')
        cr = om.CaseReader(filename)

        # recorded data from driver only
        self.assertEqual(cr.list_sources(out_stream=None), ['driver'])

        # check that we got the correct number of cases
        driver_cases = cr.list_cases('driver', out_stream=None)
        self.assertEqual(len(driver_cases), 5)

        case = cr.get_case('rank0:ScipyOptimize_SLSQP|4')

        assert_near_equal(case.outputs['x'], 7.16666667, 1e-6)
        assert_near_equal(case.outputs['y'], -7.83333333, 1e-6)

    def test_database_v4(self):
        # the change between v4 and v5 was the addition of the 'source' information
        # this tests the proper determination of the case source without that data
        #
        # the legacy database was created with the same setup as the above test named
        # "test_reading_all_case_types" so this is a slimmed down version of that test
        #
        # NOTE: ScipyOptimizeDriver did not record its initial run under its own
        #       recording context prior to V5, so the initial case does not reflect
        #       the driver as the source

        filename = os.path.join(self.legacy_dir, 'case_database_v4.sql')

        cr = om.CaseReader(filename)

        #
        # check sources
        #

        self.assertEqual(sorted(cr.list_sources(out_stream=None)), [
            'driver', 'problem', 'root.mda.nonlinear_solver', 'root.nonlinear_solver', 'root.pz'
        ])

        driver_vars = cr.list_source_vars('driver', out_stream=None)
        self.assertEqual(('inputs:', sorted(driver_vars['inputs']), 'outputs:', sorted(driver_vars['outputs'])),
                         ('inputs:', [], 'outputs:', ['con1', 'con2', 'obj', 'x', 'z']))

        model_vars = cr.list_source_vars('root.pz', out_stream=None)
        self.assertEqual(('inputs:', sorted(model_vars['inputs']), 'outputs:', sorted(model_vars['outputs'])),
                         ('inputs:', [], 'outputs:', ['z']))

        solver_vars = cr.list_source_vars('root.mda.nonlinear_solver', out_stream=None)
        self.assertEqual(('inputs:', sorted(solver_vars['inputs']), 'outputs:', sorted(solver_vars['outputs'])),
                         ('inputs:', ['x', 'y1', 'y2', 'z'], 'outputs:', ['y1', 'y2']))

        #
        # check system cases
        #

        system_cases = cr.list_cases('root.pz', recurse=False, out_stream=None)
        expected_cases = [
            'rank0:root._solve_nonlinear|0|NLRunOnce|0|pz._solve_nonlinear|0',
            'rank0:SLSQP|0|root._solve_nonlinear|1|NLRunOnce|0|pz._solve_nonlinear|1',
            'rank0:SLSQP|1|root._solve_nonlinear|2|NLRunOnce|0|pz._solve_nonlinear|2',
            'rank0:SLSQP|2|root._solve_nonlinear|3|NLRunOnce|0|pz._solve_nonlinear|3',
            'rank0:SLSQP|3|root._solve_nonlinear|4|NLRunOnce|0|pz._solve_nonlinear|4',
            'rank0:SLSQP|4|root._solve_nonlinear|5|NLRunOnce|0|pz._solve_nonlinear|5',
            'rank0:SLSQP|5|root._solve_nonlinear|6|NLRunOnce|0|pz._solve_nonlinear|6',
        ]
        self.assertEqual(len(system_cases), len(expected_cases))
        for i, coord in enumerate(system_cases):
            self.assertEqual(coord, expected_cases[i])

        #
        # check solver cases
        #

        root_solver_cases = cr.list_cases('root.nonlinear_solver', recurse=False, out_stream=None)
        expected_cases = [
            'rank0:root._solve_nonlinear|0|NLRunOnce|0',
            'rank0:SLSQP|0|root._solve_nonlinear|1|NLRunOnce|0',
            'rank0:SLSQP|1|root._solve_nonlinear|2|NLRunOnce|0',
            'rank0:SLSQP|2|root._solve_nonlinear|3|NLRunOnce|0',
            'rank0:SLSQP|3|root._solve_nonlinear|4|NLRunOnce|0',
            'rank0:SLSQP|4|root._solve_nonlinear|5|NLRunOnce|0',
            'rank0:SLSQP|5|root._solve_nonlinear|6|NLRunOnce|0'
        ]
        self.assertEqual(len(root_solver_cases), len(expected_cases))
        for i, coord in enumerate(root_solver_cases):
            self.assertEqual(coord, expected_cases[i])

        case = cr.get_case(root_solver_cases[-1])

        expected_inputs = ['x', 'y1', 'y2', 'z']
        expected_outputs = ['con1', 'con2', 'obj', 'x', 'y1', 'y2', 'z']

        self.assertEqual(sorted(case.inputs.keys()), expected_inputs)
        self.assertEqual(sorted(case.outputs.keys()), expected_outputs)
        self.assertEqual(sorted(case.residuals.keys()), expected_outputs)

        #
        # check mda solver cases
        #

        # check that there are multiple iterations and mda solver is part of the coordinate
        mda_solver_cases = cr.list_cases('root.mda.nonlinear_solver', recurse=False, out_stream=None)
        self.assertTrue(len(mda_solver_cases) > 1)
        for coord in mda_solver_cases:
            self.assertTrue('mda._solve_nonlinear' in coord)

        case = cr.get_case(mda_solver_cases[-1])

        expected_inputs = ['x', 'y1', 'y2', 'z']
        expected_outputs = ['y1', 'y2']

        self.assertEqual(sorted(case.inputs.keys()), expected_inputs)
        self.assertEqual(sorted(case.outputs.keys()), expected_outputs)
        self.assertEqual(sorted(case.residuals.keys()), expected_outputs)

        # check that inputs & outputs are in sorted order, since exec/setup order is not available
        expected = [
            "5 Input(s) in 'mda'",
            "",
            "varname    value               ",
            "---------  --------------------",
            "d1.x   [3.43977636e-15]    ",
            "d1.y2  [3.75527777]        ",
            "d1.z   |1.9776388835080063|",
            "d2.y1  [3.16]              ",
            "d2.z   |1.9776388835080063|",
         ]

        stream = StringIO()
        case.list_inputs(hierarchical=False, out_stream=stream)
        text = stream.getvalue().split('\n')
        for i, line in enumerate(expected):
            if i == 0:
                self.assertEqual(text[i], line)
            elif line and not line.startswith('-'):
                self.assertTrue(text[i].startswith(line.split()[0]))

        expected = [
            "2 Explicit Output(s) in 'mda'",
            "",
            "varname    value       ",
            "---------  ------------",
            "d1.y1  [3.16]      ",
            "d2.y2  [3.75527777]",
            "",
            "",
            "0 Implicit Output(s) in 'mda'",
            "-----------------------------",
         ]

        stream = StringIO()
        case.list_outputs(hierarchical=False, out_stream=stream)
        text = stream.getvalue().split('\n')
        for i, line in enumerate(expected):
            if i == 0:
                self.assertEqual(text[i], line)
            elif line and not line.startswith('-'):
                self.assertTrue(text[i].startswith(line.split()[0]))

        np.testing.assert_almost_equal(case.abs_err, 0, decimal=6)
        np.testing.assert_almost_equal(case.rel_err, 0, decimal=6)

        # check that the recurse option returns root and mda solver cases plus child system cases
        all_solver_cases = cr.list_cases('root.nonlinear_solver', recurse=True, flat=True, out_stream=None)
        self.assertEqual(len(all_solver_cases),
                         len(root_solver_cases) + len(mda_solver_cases) + len(system_cases))

        #
        # check driver cases
        #

        driver_cases = cr.list_cases('driver', recurse=False, out_stream=None)
        expected_cases = [
            'rank0:SLSQP|0',
            'rank0:SLSQP|1',
            'rank0:SLSQP|2',
            'rank0:SLSQP|3',
            'rank0:SLSQP|4',
            'rank0:SLSQP|5',
        ]
        # check that there are multiple iterations and they have the expected coordinates
        self.assertTrue(len(driver_cases), len(expected_cases))
        for i, coord in enumerate(driver_cases):
            self.assertEqual(coord, expected_cases[i])

    def test_driver_v3(self):
        """
        Backwards compatibility version 3.
        Legacy case recording file generated using code from test_record_driver_system_solver
        test in test_sqlite_recorder.py
        """
        prob = SellarProblem(SellarDerivativesGroupedPreAutoIVC)
        prob.driver = om.ScipyOptimizeDriver(tol=1e-9, disp=False)
        prob.setup()
        prob.run_driver()
        prob.cleanup()

        filename = os.path.join(self.legacy_dir, 'case_driver_solver_system_03.sql')

        cr = om.CaseReader(filename)

        # list just the driver cases
        driver_cases = cr.list_cases('driver', recurse=False, out_stream=None)

        # check that we got the correct number of cases
        self.assertEqual(len(driver_cases), 6)

        # check that the access by case keys works:
        seventh_slsqp_iteration_case = cr.get_case('rank0:SLSQP|5')
        np.testing.assert_almost_equal(seventh_slsqp_iteration_case.outputs['z'],
                                       [1.97846296, -2.21388305e-13], decimal=2)

        # Test values from the last case
        last_case = cr.get_case(driver_cases[-1])
        np.testing.assert_almost_equal(last_case.outputs['z'], prob['z'])
        np.testing.assert_almost_equal(last_case.outputs['x'], [-0.00309521], decimal=2)

        # check that the case keys (iteration coords) come back correctly
        for i, iter_coord in enumerate(driver_cases):
            self.assertEqual(iter_coord, 'rank0:SLSQP|{}'.format(i))

        # Test problem metadata
        self.assertIsNotNone(cr.problem_metadata)
        self.assertTrue('connections_list' in cr.problem_metadata)
        self.assertTrue('tree' in cr.problem_metadata)
        self.assertTrue('variables' in cr.problem_metadata)

        # While we are here, make sure we can load this case.

        # Add one to all the inputs just to change the model
        #   so we can see if loading the case values really changes the model
        for name in prob.model._inputs:
            prob.model._inputs[name] += 1.0
        for name in prob.model._outputs:
            prob.model._outputs[name] += 1.0

        # Now load in the case we recorded
        prob.load_case(seventh_slsqp_iteration_case)

        _assert_model_matches_case(seventh_slsqp_iteration_case, prob.model)

    def test_driver_v2(self):
        """ Backwards compatibility version 2. """
        prob = SellarProblem(SellarDerivativesGroupedPreAutoIVC)
        prob.driver = om.ScipyOptimizeDriver(tol=1e-9, disp=False)
        prob.setup()
        prob.run_driver()
        prob.cleanup()

        filename = os.path.join(self.legacy_dir, 'case_driver_solver_system_02.sql')

        cr = om.CaseReader(filename)

        # list just the driver cases
        driver_cases = cr.list_cases('driver', recurse=False, out_stream=None)

        # check that we got the correct number of cases
        self.assertEqual(len(driver_cases), 7)

        # check that the access by case keys works:
        seventh_slsqp_iteration_case = cr.get_case('rank0:SLSQP|5')
        np.testing.assert_almost_equal(seventh_slsqp_iteration_case.outputs['z'],
                                       [1.97846296, -2.21388305e-13], decimal=2)

        # Test values from the last case
        last_case = cr.get_case(driver_cases[-1])
        np.testing.assert_almost_equal(last_case.outputs['z'], prob['z'])
        np.testing.assert_almost_equal(last_case.outputs['x'], [-0.00309521], decimal=2)

        # check that the case keys (iteration coords) come back correctly
        for i, iter_coord in enumerate(driver_cases):
            self.assertEqual(iter_coord, 'rank0:SLSQP|{}'.format(i))

        # Test driver metadata
        self.assertIsNotNone(cr.problem_metadata)
        self.assertTrue('connections_list' in cr.problem_metadata)
        self.assertTrue('tree' in cr.problem_metadata)
        self.assertTrue('variables' in cr.problem_metadata)

        # While we are here, make sure we can load this case.

        # Add one to all the inputs just to change the model
        #   so we can see if loading the case values really changes the model
        for name in prob.model._inputs:
            prob.model._inputs[name] += 1.0
        for name in prob.model._outputs:
            prob.model._outputs[name] += 1.0

        # Now load in the case we recorded
        prob.load_case(seventh_slsqp_iteration_case)

        _assert_model_matches_case(seventh_slsqp_iteration_case, prob.model)

    def test_solver_v2(self):
        """ Backwards compatibility version 2. """
        filename = os.path.join(self.legacy_dir, 'case_driver_solver_system_02.sql')

        cases = om.CaseReader(filename)

        # list just the solver cases
        solver_cases = cases.list_cases('root.nonlinear_solver', recurse=False, out_stream=None)

        # check that we got the correct number of cases
        self.assertEqual(len(solver_cases), 7)

        # check that the access by case keys works:
        sixth_solver_case_id = solver_cases[5]
        self.assertEqual(sixth_solver_case_id, 'rank0:SLSQP|5|root._solve_nonlinear|5|NLRunOnce|0')

        sixth_solver_iteration = cases.get_case(sixth_solver_case_id)

        np.testing.assert_almost_equal(sixth_solver_iteration.outputs['z'],
                                       [1.97846296, -2.21388305e-13], decimal=2)

        # Test values from the last case
        last_case = cases.get_case(solver_cases[-1])
        np.testing.assert_almost_equal(last_case.outputs['x'], [-0.00309521], decimal=2)

        # check that the case keys (iteration coords) come back correctly
        coord = 'rank0:SLSQP|{}|root._solve_nonlinear|{}|NLRunOnce|0'
        for i, iter_coord in enumerate(solver_cases):
            self.assertEqual(iter_coord, coord.format(i, i))

    def test_system_v2(self):
        """ Backwards compatibility version 2. """
        filename = os.path.join(self.legacy_dir, 'case_driver_solver_system_02.sql')

        cr = om.CaseReader(filename)

        # list just the system cases
        system_cases = cr.list_cases('root', recurse=False, out_stream=None)

        # check that we got the correct number of cases
        self.assertEqual(len(system_cases), 7)

        # check that the access by case keys works:
        sixth_system_case_id = system_cases[5]
        self.assertEqual(sixth_system_case_id, 'rank0:SLSQP|5|root._solve_nonlinear|5')

        sixth_system_case = cr.get_case(system_cases[5])

        np.testing.assert_almost_equal(sixth_system_case.outputs['z'],
                                       [1.97846296, -2.21388305e-13], decimal=2)

        last_case = cr.get_case(system_cases[-1])
        np.testing.assert_almost_equal(last_case.outputs['x'], [-0.00309521], decimal=2)

        # check that the case keys (iteration coords) come back correctly
        for i, iter_coord in enumerate(system_cases):
            self.assertEqual(iter_coord, 'rank0:SLSQP|{}|root._solve_nonlinear|{}'.format(i, i))

    def test_driver_v1(self):
        """ Backwards compatibility oldest version. """
        prob = SellarProblem(SellarDerivativesGroupedPreAutoIVC)
        prob.driver = om.ScipyOptimizeDriver(tol=1e-9, disp=False)
        prob.setup()
        prob.run_driver()
        prob.cleanup()

        filename = os.path.join(self.legacy_dir, 'case_driver_01.sql')

        cr = om.CaseReader(filename)

        # recorded data from driver only
        self.assertEqual(cr.list_sources(out_stream=None), ['driver'])

        # check that we got the correct number of cases
        driver_cases = cr.list_cases('driver', out_stream=None)
        self.assertEqual(len(driver_cases), 7)

        # check that the access by case keys works:
        seventh_slsqp_iteration_case = cr.get_case('rank0:SLSQP|5')
        np.testing.assert_almost_equal(seventh_slsqp_iteration_case.outputs['z'],
                                       [1.97846296, -2.21388305e-13], decimal=2)

        # Test values from the last case
        last_case = cr.get_case(driver_cases[-1])
        np.testing.assert_almost_equal(last_case.outputs['z'], prob['z'],)
        np.testing.assert_almost_equal(last_case.outputs['x'], [-0.00309521], decimal=2)

        # check that the case keys (iteration coords) come back correctly
        for i, iter_coord in enumerate(driver_cases):
            self.assertEqual(iter_coord, 'rank0:SLSQP|{}'.format(i))

        # While we are here, make sure we can load this case.

        # Add one to all the inputs just to change the model
        #   so we can see if loading the case values really changes the model
        for name in prob.model._inputs:
            prob.model._inputs[name] += 1.0
        for name in prob.model._outputs:
            prob.model._outputs[name] += 1.0

        # Now load in the case we recorded
        prob.load_case(seventh_slsqp_iteration_case)

        _assert_model_matches_case(seventh_slsqp_iteration_case, prob.model)

    def test_driver_v1_pre_problem(self):
        """ Backwards compatibility oldest version. """
        prob = SellarProblem(SellarDerivativesGroupedPreAutoIVC)
        prob.driver = om.ScipyOptimizeDriver(tol=1e-9, disp=False)
        prob.setup()
        prob.run_driver()
        prob.cleanup()

        filename = os.path.join(self.legacy_dir, 'case_driver_pre01.sql')

        cr = om.CaseReader(filename)

        # recorded data from driver only
        self.assertEqual(cr.list_sources(out_stream=None), ['driver'])

        # check that we got the correct number of cases
        driver_cases = cr.list_cases('driver', out_stream=None)
        self.assertEqual(len(driver_cases), 7)

        # check that the access by case keys works:
        seventh_slsqp_iteration_case = cr.get_case('rank0:SLSQP|5')
        np.testing.assert_almost_equal(seventh_slsqp_iteration_case.outputs['z'],
                                       [1.97846296, -2.21388305e-13], decimal=2)

        # Test values from the last case
        last_case = cr.get_case(driver_cases[-1])
        np.testing.assert_almost_equal(last_case.outputs['z'], prob['z'])
        np.testing.assert_almost_equal(last_case.outputs['x'], [-0.00309521], decimal=2)

        # check that the case keys (iteration coords) come back correctly
        for i, iter_coord in enumerate(driver_cases):
            self.assertEqual(iter_coord, 'rank0:SLSQP|{}'.format(i))

        # While we are here, make sure we can load this case.

        # Add one to all the inputs just to change the model
        #   so we can see if loading the case values really changes the model
        for name in prob.model._inputs:
            prob.model._inputs[name] += 1.0
        for name in prob.model._outputs:
            prob.model._outputs[name] += 1.0

        # Now load in the case we recorded
        prob.load_case(seventh_slsqp_iteration_case)

        _assert_model_matches_case(seventh_slsqp_iteration_case, prob.model)


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
class SqliteCaseReaderMPI(unittest.TestCase):
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
        # print(reader.list_cases())
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
