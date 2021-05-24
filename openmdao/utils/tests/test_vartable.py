import sys
import unittest

import numpy as np
from io import StringIO

import openmdao.api as om

from openmdao.utils.notebook_utils import notebook_mode, tabulate
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.double_sellar import DoubleSellar
from openmdao.test_suite.components.sellar import SellarProblem

"""
The following System methods should generate HTML using tabulate
when called with the default output stream in a jupyter notebook.

- list_outputs()
- list_inputs()

The following Case methods should generate HTML using tabulate
when called with the default output stream in a jupyter notebook.

- list_outputs()
- list_inputs()

The following CaseReader methods should generate HTML using tabulate
when called with the default output stream in a jupyter notebook.

- list_sources()
- list_source_vars()
- list_cases()
- list_cases(problem)
- list_cases(driver|system|solver, recurse=True, flat=True)
- list_cases(driver|system|solver, recurse=False)
- list_cases(coord, recurse=True, flat=True)
"""

@unittest.skipUnless(tabulate, "Tabulate is required")
@use_tempdirs
class TestNotebookListIO(unittest.TestCase):

    def setUp(self):
        # override notebook flag for system, variable table and sqlite_reader
        from openmdao.core import system
        from openmdao.utils import variable_table
        from openmdao.recorders import sqlite_reader
        system.notebook = variable_table.notebook = sqlite_reader.notebook = True

        # capture HTML output from variable_table
        self.html_stream = StringIO()
        variable_table.HTML = lambda x: self.html_stream.write(x)
        sqlite_reader.HTML = lambda x: self.html_stream.write(x)

        # create recorder
        self.filename = "cases.sql"
        self.recorder = om.SqliteRecorder(self.filename, record_viewer_data=False)

        # create & run problem, generate case
        prob = om.Problem(model=DoubleSellar())

        prob.model.add_recorder(self.recorder)

        prob.setup()
        prob.set_solver_print(level=0)
        prob.run_model()
        prob.cleanup()

        # expected results
        self.prob = prob

        self.expected_inputs = [
            ('g1.d1.x',  {'value': [0.80000]}),
            ('g1.d1.y2', {'value': [0.80000]}),
            ('g1.d1.z',  {'value': [0., 0.]}),
            ('g1.d2.y1', {'value': [0.64000]}),
            ('g1.d2.z',  {'value': [0., 0.]}),
            ('g2.d1.x',  {'value': [0.80000]}),
            ('g2.d1.y2', {'value': [0.80000]}),
            ('g2.d1.z',  {'value': [0., 0.]}),
            ('g2.d2.y1', {'value': [0.64000]}),
            ('g2.d2.z',  {'value': [0., 0.]})
        ]

        self.expected_outputs = [
            ('g1.d1.y1', {'value': [0.64000]}),
            ('g1.d2.y2', {'value': [0.80000]}),
            ('g2.d1.y1', {'value': [0.64000]}),
            ('g2.d2.y2', {'value': [0.80000]})
        ]

    def test_invalid_stream(self):
        with self.assertRaises(TypeError) as cm:
            self.prob.model.list_outputs(out_stream='stdout')
        self.assertTrue(str(cm.exception), "Invalid output stream specified for 'out_stream'.")

    def test_system_list_inputs(self):
        # when calling <System>.list_inputs() with the default out_stream,
        # the notebook should get an HTML table and no results should be
        # written to stdout.
        sysout = sys.stdout
        try:
            capture_stdout = StringIO()
            sys.stdout = capture_stdout
            inputs = self.prob.model.list_inputs(units=True, shape=True,
                                                 prom_name=True, hierarchical=False)
        finally:
            sys.stdout = sysout

        # we should still get the expected return value
        for i, tup in enumerate(sorted(inputs)):
            name, meta = tup
            expected_name, expected_meta = self.expected_inputs[i]
            self.assertEqual(name, expected_name)
            assert_near_equal(meta['value'], expected_meta['value'], 1e-5)

        # no results should have been written to stdout
        captured_output = capture_stdout.getvalue()
        for name, meta in self.expected_inputs:
            self.assertFalse(name in captured_output)

        # generated HTML should be a table and have all expected inputs with metadata
        captured_HTML = self.html_stream.getvalue()
        self.assertTrue(captured_HTML.startswith('<table>'))
        self.assertTrue(captured_HTML.endswith('</table>'))
        self.assertEqual(captured_HTML.count('<tr><td'), len(self.expected_inputs))
        for column in ['varname', 'value', 'units', 'shape', 'prom_name']:
            self.assertTrue(column in captured_HTML)
        for name, meta in self.expected_inputs:
            self.assertTrue(name in captured_HTML)

    def test_system_list_outputs(self):
        # when calling <System>.list_outputs() with the default out_stream,
        # the notebook should get an HTML table and no results should be
        # written to stdout.
        sysout = sys.stdout
        try:
            capture_stdout = StringIO()
            sys.stdout = capture_stdout
            outputs = self.prob.model.list_outputs(units=True, shape=True,
                                                   prom_name=True, hierarchical=False)
        finally:
            sys.stdout = sysout

        # we should still get the expected return value
        for i, tup in enumerate(sorted(outputs)):
            name, meta = tup
            expected_name, expected_meta = self.expected_outputs[i]
            self.assertEqual(name, expected_name)
            assert_near_equal(meta['value'], expected_meta['value'], 1e-5)

        # no results should have been written to stdout
        captured_output = capture_stdout.getvalue()
        for name, meta in self.expected_outputs:
            self.assertFalse(name in captured_output)

        # generated HTML should be a table and have all expected inputs with metadata
        captured_HTML = self.html_stream.getvalue()
        self.assertTrue(captured_HTML.startswith('<table>'))
        self.assertTrue(captured_HTML.endswith('</table>'))
        self.assertEqual(captured_HTML.count('<tr><td'), len(self.expected_outputs))
        for column in ['varname', 'value', 'units', 'shape', 'prom_name']:
            self.assertTrue(column in captured_HTML)
        for name, meta in self.expected_outputs:
            self.assertTrue(name in captured_HTML)

    def test_case_list_inputs(self):
        last_case = om.CaseReader(self.filename).get_case(-1)

        # when calling <Case>.list_inputs() with the default out_stream,
        # the notebook should get an HTML table and no results should be
        # written to stdout.
        sysout = sys.stdout
        try:
            capture_stdout = StringIO()
            sys.stdout = capture_stdout
            inputs = last_case.list_inputs(units=True, shape=True, prom_name=True, hierarchical=False)
        finally:
            sys.stdout = sysout

        # we should still get the expected return value
        for i, tup in enumerate(sorted(inputs)):
            name, meta = tup
            expected_name, expected_meta = self.expected_inputs[i]
            self.assertEqual(name, expected_name)
            assert_near_equal(meta['value'], expected_meta['value'], 1e-5)

        # no results should have been written to stdout
        captured_output = capture_stdout.getvalue()
        for name, meta in self.expected_inputs:
            self.assertFalse(name in captured_output)

        # generated HTML should be a table and have all expected inputs with metadata
        captured_HTML = self.html_stream.getvalue()
        self.assertTrue(captured_HTML.startswith('<table>'))
        self.assertTrue(captured_HTML.endswith('</table>'))
        self.assertEqual(captured_HTML.count('<tr><td'), len(self.expected_inputs))
        for column in ['varname', 'value', 'units', 'shape', 'prom_name']:
            self.assertTrue(column in captured_HTML)
        for name, meta in self.expected_inputs:
            self.assertTrue(name in captured_HTML)

    def test_case_list_outputs(self):
        last_case = om.CaseReader(self.filename).get_case(-1)

        # when calling <Case>.list_outputs() with the default out_stream,
        # the notebook should get an HTML table and no results should be
        # written to stdout.
        sysout = sys.stdout
        try:
            capture_stdout = StringIO()
            sys.stdout = capture_stdout
            outputs = last_case.list_outputs(units=True, shape=True,
                                             prom_name=True, hierarchical=False)
        finally:
            sys.stdout = sysout

        # we should still get the expected return value
        for i, tup in enumerate(sorted(outputs)):
            name, meta = tup
            expected_name, expected_meta = self.expected_outputs[i]
            self.assertEqual(name, expected_name)
            assert_near_equal(meta['value'], expected_meta['value'], 1e-5)

        # no results should have been written to stdout
        captured_output = capture_stdout.getvalue()
        for name, meta in self.expected_outputs:
            self.assertFalse(name in captured_output)

        # generated HTML should be a table and have all expected inputs with metadata
        captured_HTML = self.html_stream.getvalue()
        self.assertTrue(captured_HTML.startswith('<table>'))
        self.assertTrue(captured_HTML.endswith('</table>'))
        self.assertEqual(captured_HTML.count('<tr><td'), len(self.expected_outputs))
        for column in ['varname', 'value', 'units', 'shape', 'prom_name']:
            self.assertTrue(column in captured_HTML)
        for name, meta in self.expected_outputs:
            self.assertTrue(name in captured_HTML)


@unittest.skipUnless(tabulate, "Tabulate is required")
@use_tempdirs
class TestNotebookCaseReader(unittest.TestCase):

    def setUp(self):
        # override notebook flag for system, variable table and sqlite_reader
        from openmdao.core import system
        from openmdao.utils import variable_table
        from openmdao.recorders import sqlite_reader
        system.notebook = variable_table.notebook = sqlite_reader.notebook = True

        # capture HTML output from variable_table
        self.html_stream = StringIO()
        variable_table.HTML = lambda x: self.html_stream.write(x)
        sqlite_reader.HTML = lambda x: self.html_stream.write(x)

        # create recorder
        self.filename = "cases.sql"
        self.recorder = om.SqliteRecorder(self.filename, record_viewer_data=False)

        # create & run problem, generate case
        prob = SellarProblem()
        prob.setup()

        prob.add_recorder(self.recorder)
        prob.driver.add_recorder(self.recorder)
        prob.model.d1.add_recorder(self.recorder)

        prob.run_driver()

        prob.record('final')
        prob.cleanup()

        # expected results
        self.prob = prob

        self.expected_headers = ['driver', 'problem', 'system']

        self.expected_sources = ['driver', 'problem', 'root.d1']

        self.expected_source_vars = {
            'driver': {
                'inputs': [],
                'outputs': ['con1', 'con2', 'obj', 'x', 'z'],
                'residuals': []
            },
            'problem': {
                'inputs': [],
                'outputs': ['con1', 'con2', 'obj', 'x', 'y1', 'y2', 'z'],
                'residuals': []
            },
            'root.d1': {
                'inputs': ['x', 'y2', 'z'],
                'outputs': ['y1'],
                'residuals': ['y1']
            }
         }

        self.expected_cases = [
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

    def test_list_sources(self):
        cr = om.CaseReader(self.filename)

        # when calling list_sources() with the default out_stream, the notebook
        # should get an HTML table and no results should be written to stdout.
        sysout = sys.stdout
        try:
            capture_stdout = StringIO()
            sys.stdout = capture_stdout
            sources = cr.list_sources()
        finally:
            sys.stdout = sysout

        # we should still get the expected return value
        for i, source in enumerate(sorted(sources)):
            self.assertEqual(source, self.expected_sources[i])

        # no results should have been written to stdout
        captured_output = capture_stdout.getvalue()
        for source in self.expected_sources:
            self.assertFalse(source in captured_output)

        # generated HTML should be a table and have all expected sources and cases
        captured_HTML = self.html_stream.getvalue()
        self.assertTrue(captured_HTML.startswith('<table>'))
        self.assertTrue(captured_HTML.endswith('</table>'))
        self.assertEqual(captured_HTML.count('<tr><td'), len(self.expected_sources))
        for source in self.expected_sources:
            self.assertTrue(source in captured_HTML)

    def test_list_source_vars(self):
        cr = om.CaseReader(self.filename)

        # when calling list_source_vars() with the default out_stream, the notebook
        # should get an HTML table and no results should be written to stdout.
        for source, expected_vars in self.expected_source_vars.items():
            sysout = sys.stdout
            try:
                capture_stdout = StringIO()
                sys.stdout = capture_stdout
                source_vars = cr.list_source_vars(source)
            finally:
                sys.stdout = sysout

            # we should still get the expected return value
            self.assertTrue(sorted(source_vars['inputs']) == expected_vars['inputs'])
            self.assertTrue(sorted(source_vars['outputs']) == expected_vars['outputs'])
            self.assertTrue(sorted(source_vars['residuals']) == expected_vars['residuals'])

            # no results should have been written to stdout
            captured_output = capture_stdout.getvalue()
            for source in self.expected_sources:
                self.assertFalse(source in captured_output)

            # generated HTML should be a table and have all expected source vars
            captured_HTML = self.html_stream.getvalue()
            self.assertTrue(captured_HTML.startswith('<table>'))
            self.assertTrue(captured_HTML.endswith('</table>'))
            for header in ['inputs', 'outputs', 'residuals']:
                self.assertTrue(header in captured_HTML)
            for var_name in source_vars['inputs'] + source_vars['outputs'] + source_vars['residuals']:
                self.assertTrue(var_name in captured_HTML)

    def test_list_cases(self):
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

        # we should still get the expected return value
        for i, line in enumerate(self.expected_cases):
            self.assertEqual(cases[i], line, f'case not found "{line}"')

        # no results should have been written to stdout
        captured_output = capture_stdout.getvalue()
        for source in self.expected_headers:
            self.assertFalse(source in captured_output)
        for case in self.expected_cases:
            self.assertFalse(case in captured_output)

        # generated HTML should be a table and have all expected sources and cases
        captured_HTML = self.html_stream.getvalue()
        self.assertTrue(captured_HTML.startswith('<table>'))
        self.assertTrue(captured_HTML.endswith('</table>'))
        self.assertEqual(captured_HTML.count('<tr><td'), len(self.expected_cases))
        for source in self.expected_headers:
            self.assertTrue(source in captured_HTML)
        for case in self.expected_cases:
            self.assertTrue(case in captured_HTML)

    def test_list_cases_problem(self):
        cr = om.CaseReader(self.filename)

        # when calling list_cases() with the default out_stream, the notebook
        # should get an HTML table and no results should be written to stdout.
        sysout = sys.stdout
        try:
            capture_stdout = StringIO()
            sys.stdout = capture_stdout
            cases = cr.list_cases('problem')
        finally:
            sys.stdout = sysout

        # we should still get the expected return value
        self.assertEqual(cases, ['final'])

        # no results should have been written to stdout
        captured_output = capture_stdout.getvalue()
        self.assertFalse('problem' in captured_output)
        self.assertFalse('final' in captured_output)

        # generated HTML should be a table and have all expected sources and cases
        captured_HTML = self.html_stream.getvalue()
        self.assertTrue(captured_HTML.startswith('<table>'))
        self.assertTrue(captured_HTML.endswith('</table>'))
        self.assertEqual(captured_HTML.count('<tr><td'), 1)
        self.assertTrue('problem' in captured_HTML)  # table header
        self.assertTrue('final' in captured_HTML)    # row value

    def test_list_cases_coord(self):
        cr = om.CaseReader(self.filename)

        # when calling list_cases() with the default out_stream, the notebook
        # should get an HTML table and no results should be written to stdout.
        sysout = sys.stdout
        try:
            capture_stdout = StringIO()
            sys.stdout = capture_stdout
            cases = cr.list_cases('rank0:Driver|0')
        finally:
            sys.stdout = sysout

        # all but the final problem case
        expected_headers = ['driver', 'system']
        expected_cases = [case for case in self.expected_cases
                          if case.startswith('rank0:Driver|0')]

        # we should still get the expected return value
        for i, line in enumerate(expected_cases):
            self.assertEqual(cases[i], line, f'case not found "{line}"')

        # no results should have been written to stdout
        captured_output = capture_stdout.getvalue()
        for source in self.expected_headers:
            self.assertFalse(source in captured_output)
        for case in expected_cases:
            self.assertFalse(case in captured_output)

        # generated HTML should be a table and have all expected sources and cases
        captured_HTML = self.html_stream.getvalue()
        self.assertTrue(captured_HTML.startswith('<table>'))
        self.assertTrue(captured_HTML.endswith('</table>'))
        self.assertEqual(captured_HTML.count('<tr><td'), len(expected_cases))
        for source in expected_headers:
            self.assertTrue(source in captured_HTML)
        for case in expected_cases:
            self.assertTrue(case in captured_HTML)


@unittest.skipUnless(tabulate, "Tabulate is required")
@use_tempdirs
class TestNotebookDriverCases(unittest.TestCase):

    def setUp(self):
        # override notebook flag for system, variable table and sqlite_reader
        from openmdao.core import system
        from openmdao.utils import variable_table
        from openmdao.recorders import sqlite_reader
        system.notebook = variable_table.notebook = sqlite_reader.notebook = True

        # capture HTML output from variable_table
        self.html_stream = StringIO()
        variable_table.HTML = lambda x: self.html_stream.write(x)
        sqlite_reader.HTML = lambda x: self.html_stream.write(x)

        # create & run problem, generate cases
        model = om.Group()
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])
        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob = om.Problem(model)
        prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=3))
        prob.driver.add_recorder(om.SqliteRecorder('cases.sql', record_viewer_data=False))

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        # expected results
        self.expected_sources = ['driver']

        self.expected_cases = [
            'rank0:DOEDriver_FullFactorial|0',
            'rank0:DOEDriver_FullFactorial|1',
            'rank0:DOEDriver_FullFactorial|2',
            'rank0:DOEDriver_FullFactorial|3',
            'rank0:DOEDriver_FullFactorial|4',
            'rank0:DOEDriver_FullFactorial|5',
            'rank0:DOEDriver_FullFactorial|6',
            'rank0:DOEDriver_FullFactorial|7',
            'rank0:DOEDriver_FullFactorial|8'
        ]

    def test_cases_driver_recurseFalse(self):
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
        for i, expected_case in enumerate(self.expected_cases):
            self.assertEqual(cases[i], expected_case)

        # no results should have been written to stdout
        captured_output = capture_stdout.getvalue()
        for source in self.expected_sources:
            self.assertFalse(source in captured_output)
        for case in self.expected_cases:
            self.assertFalse(case in captured_output)

        # generated HTML should be a table and have all expected sources and cases
        captured_HTML = self.html_stream.getvalue()
        self.assertTrue(captured_HTML.startswith('<table>'))
        self.assertTrue(captured_HTML.endswith('</table>'))
        self.assertEqual(captured_HTML.count('<tr><td'), len(self.expected_cases))
        for source in self.expected_sources:
            self.assertEqual(captured_HTML.count(source), 1)  # header occurs only once
        for case in self.expected_cases:
            self.assertTrue(case in captured_HTML)            # all case rows are there


if __name__ == "__main__":
    unittest.main()
