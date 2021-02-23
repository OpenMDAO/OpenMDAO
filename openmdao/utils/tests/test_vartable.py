import sys
import unittest

import numpy as np
from io import StringIO

import openmdao
import openmdao.api as om

from openmdao.core.notebook_mode import notebook_mode, tabulate

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.double_sellar import DoubleSellar
from openmdao.test_suite.components.sellar import SellarProblem


@unittest.skipUnless(tabulate, "Tabulate is required")
@use_tempdirs
class TestNotebookFormat(unittest.TestCase):
    """
    The following System methods should generate HTML using tabulate
    when called with the default output stream in a jupyter notebook.

    x list_outputs()
    x list_inputs()

    The following Case methods should generate HTML using tabulate
    when called with the default output stream in a jupyter notebook.

    - list_outputs()
    - list_inputs()

    The following CaseReader methods should generate HTML using tabulate
    when called with the default output stream in a jupyter notebook.

    - list_sources()
    - list_source_vars()
    x list_cases()
    - list_cases(problem)
    x list_cases(driver|system|solver, recurse=False)
    - list_cases(driver|system|solver, recurse=True, flat=True)
    - list_cases(coord, recurse=True, flat=True)
    """

    def setUp(self):
        # override notebook flag for variable table and sqlite_reader
        from openmdao.core import system
        from openmdao.utils import variable_table
        from openmdao.recorders import sqlite_reader
        system.notebook = variable_table.notebook = sqlite_reader.notebook = True

        # capture HTML output
        self.html_stream = StringIO()

        variable_table.HTML = lambda x : self.html_stream.write(x)

        self.filename = "sqlite_test"
        self.recorder = om.SqliteRecorder(self.filename, record_viewer_data=False)

    def tearDown(self):
        openmdao.core.system.notebook = False
        openmdao.recorders.sqlite_reader.notebook = False

    def test_list_inputs_and_outputs(self):

        prob = om.Problem()
        model = prob.model = DoubleSellar()

        driver = prob.driver

        recorder = om.SqliteRecorder("cases.sql")
        prob.model.add_recorder(recorder)

        driver.recording_options['record_desvars'] = False
        driver.recording_options['record_objectives'] = False
        driver.recording_options['record_constraints'] = False
        driver.recording_options['record_derivatives'] = False
        driver.add_recorder(recorder)

        # each SubSellar group converges itself
        g1 = model.g1
        g1.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        g1.linear_solver = om.DirectSolver()  # used for derivatives

        g2 = model.g2
        g2.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        g2.linear_solver = om.DirectSolver()

        # Converge the outer loop with Gauss Seidel, with a looser tolerance.
        model.nonlinear_solver = om.NonlinearBlockGS(rtol=1.0e-5)
        model.linear_solver = om.ScipyKrylov()
        model.linear_solver.precon = om.LinearBlockGS()

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()

        # when calling list_inputs() with the default out_stream, the notebook
        # should get an HTML table and no results should be written to stdout.
        sysout = sys.stdout
        try:
            capture_stdout = StringIO()
            sys.stdout = capture_stdout
            inputs = prob.model.list_inputs(units=True, shape=True, prom_name=True, hierarchical=False)
        finally:
            sys.stdout = sysout

        # we should still get the expected return value
        expected_inputs = [
            ('g1.d1.z', {'value': [0., 0.]}),
            ('g1.d2.z', {'value': [0., 0.]}),
            ('g1.d1.x', {'value': [0.80000448]}),
            ('g1.d1.y2', {'value': [0.80000249]}),
            ('g1.d2.y1', {'value': [0.64000398]}),
            ('g2.d1.z', {'value': [0., 0.]}),
            ('g2.d2.z', {'value': [0., 0.]}),
            ('g2.d1.x', {'value': [0.80000249]}),
            ('g2.d1.y2', {'value': [0.80000138]}),
            ('g2.d2.y1', {'value': [0.64000221]})
        ]
        for i, tup in enumerate(inputs):
            name, meta = tup
            expected_name, expected_meta = expected_inputs[i]
            self.assertEqual(name, expected_name)
            assert_near_equal(meta['value'], expected_meta['value'], 1e-6)

        # no results should have been written to stdout
        captured_output = capture_stdout.getvalue()
        for name, meta in expected_inputs:
            self.assertFalse(name in captured_output)

        # generated HTML should be a table and have all expected inputs with metadata
        captured_HTML = self.html_stream.getvalue()
        self.assertTrue(captured_HTML.startswith('<table>'))
        self.assertTrue(captured_HTML.endswith('</table>'))
        for column in ['varname', 'value', 'units', 'shape', 'prom_name']:
            self.assertTrue(column in captured_HTML)
        for name, meta in expected_inputs:
            self.assertTrue(name in captured_HTML)

        # when calling list_outputs() with the default out_stream, the notebook
        # should get an HTML table and no results should be written to stdout.
        sysout = sys.stdout
        try:
            capture_stdout = StringIO()
            sys.stdout = capture_stdout
            outputs = prob.model.list_outputs(units=True, shape=True, prom_name=True, hierarchical=False)
        finally:
            sys.stdout = sysout

        # we should still get the expected return value
        expected_outputs = [
            ('g1.d1.y1', {'value': [0.64000398]}),
            ('g1.d2.y2', {'value': [0.80000249]}),
            ('g2.d1.y1', {'value': [0.64000221]}),
            ('g2.d2.y2', {'value': [0.80000138]})
        ]
        for i, tup in enumerate(outputs):
            name, meta = tup
            expected_name, expected_meta = expected_outputs[i]
            self.assertEqual(name, expected_name)
            assert_near_equal(meta['value'], expected_meta['value'], 1e-6)

        # no results should have been written to stdout
        captured_output = capture_stdout.getvalue()
        for name, meta in expected_outputs:
            self.assertFalse(name in captured_output)

        # generated HTML should be a table and have all expected inputs with metadata
        captured_HTML = self.html_stream.getvalue()
        self.assertTrue(captured_HTML.startswith('<table>'))
        self.assertTrue(captured_HTML.endswith('</table>'))
        for column in ['varname', 'value', 'units', 'shape', 'prom_name']:
            self.assertTrue(column in captured_HTML)
        for name, meta in expected_outputs:
            self.assertTrue(name in captured_HTML)

    def test_list_cases(self):
        prob = SellarProblem()
        prob.setup()

        prob.add_recorder(self.recorder)
        prob.driver.add_recorder(self.recorder)
        prob.model.d1.add_recorder(self.recorder)

        prob.run_driver()

        prob.record('final')
        prob.cleanup()

        cr = om.CaseReader(self.filename)

        # when calling list_cases() with the default out_stream, the notebook
        # should get an HTML table and no results should be written to stdout.
        sysout = sys.stdout
        try:
            capture_stdout = StringIO()
            sys.stdout = capture_stdout
            cases = cr.list_cases()
        finally:
            sys.stdout = sysout

        expected_sources = ['problem', 'driver', 'system']
        expected_cases = [
            'rank0:Driver|0|root._solve_nonlinear|0|d1._solve_nonlinear|0',
            'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|1|d1._solve_nonlinear|1',
            'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|2|d1._solve_nonlinear|2',
            'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|3|d1._solve_nonlinear|3',
            'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|4|d1._solve_nonlinear|4',
            'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|5|d1._solve_nonlinear|5',
            'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|6|d1._solve_nonlinear|6',
            'rank0:Driver|0|root._solve_nonlinear|0|NonlinearBlockGS|7|d1._solve_nonlinear|7',
            'rank0:Driver|0',
            'final',
        ]

        # we should still get the expected return value
        for i, line in enumerate(expected_cases):
            self.assertEqual(cases[i], line, f'case not found "{line}"')

        # no results should have been written to stdout
        captured_output = capture_stdout.getvalue()
        for source in expected_sources:
            self.assertFalse(source in captured_output)
        for case in expected_cases:
            self.assertFalse(case in captured_output)

        # generated HTML should be a table and have all expected sources and cases
        captured_HTML = self.html_stream.getvalue()
        self.assertTrue(captured_HTML.startswith('<table>'))
        self.assertTrue(captured_HTML.endswith('</table>'))
        for source in expected_sources:
            self.assertTrue(source in captured_HTML)
        for case in expected_cases:
            self.assertTrue(case in captured_HTML)

    def test_cases_driver_recurseFalse(self):
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
        prob.driver.add_recorder(om.SqliteRecorder('cases.sql'))

        prob.run_driver()
        prob.cleanup()

        cr = om.CaseReader('cases.sql')

        # when calling list_cases() with the default out_stream, the notebook
        # should get an HTML table and no results should be written to stdout.
        sysout = sys.stdout
        try:
            capture_stdout = StringIO()
            sys.stdout = capture_stdout
            cases = cr.list_cases('driver', recurse=False)
        finally:
            sys.stdout = sysout

        # we should still get the expected return value
        expected_sources = ['driver']
        expected_cases = [
            'rank0:DOEDriver_List|0',
            'rank0:DOEDriver_List|1',
            'rank0:DOEDriver_List|2',
            'rank0:DOEDriver_List|3',
            'rank0:DOEDriver_List|4',
            'rank0:DOEDriver_List|5',
            'rank0:DOEDriver_List|6',
            'rank0:DOEDriver_List|7',
            'rank0:DOEDriver_List|8'
        ]
        for i, expected_case in enumerate(expected_cases):
            self.assertEqual(cases[i], expected_case)

        # no results should have been written to stdout
        captured_output = capture_stdout.getvalue()
        for source in expected_sources:
            self.assertFalse(source in captured_output)
        for case in expected_cases:
            self.assertFalse(case in captured_output)

        # generated HTML should be a table and have all expected sources and cases
        captured_HTML = self.html_stream.getvalue()
        self.assertTrue(captured_HTML.startswith('<table>'))
        self.assertTrue(captured_HTML.endswith('</table>'))
        for source in expected_sources:
            self.assertTrue(source in captured_HTML)
        for case in expected_cases:
            self.assertTrue(case in captured_HTML)


if __name__ == "__main__":
    unittest.main()
