""" Unit tests for the problem interface."""

import unittest
import os
import json

import errno
from shutil import rmtree
from tempfile import mkdtemp

from openmdao.core.problem import Problem
from openmdao.test_suite.components.sellar import SellarStateConnection
from openmdao.devtools.problem_viewer.problem_viewer import _get_viewer_data, view_model
from openmdao.recorders.sqlite_recorder import SqliteRecorder


class TestViewModelData(unittest.TestCase):

    def setUp(self):
        self.dir = mkdtemp()
        self.sqlite_db_filename = os.path.join(self.dir, "sellarstate_model.sqlite")
        self.sqlite_db_filename2 = os.path.join(self.dir, "sellarstate_model_view.sqlite")
        self.sqlite_filename = os.path.join(self.dir, "sqlite_n2")
        self.sqlite_html_filename = self.sqlite_filename
        self.problem_filename = os.path.join(self.dir, "problem_n2")
        self.problem_html_filename = self.problem_filename
        self.expected_tree_json = '{"name": "root", "type": "root", "subsystem_type": "group", "children": [{"name": "px", "type": "subsystem", "subsystem_type": "component", "children": [{"name": "x", "type": "unknown", "implicit": false, "dtype": "ndarray"}]}, {"name": "pz", "type": "subsystem", "subsystem_type": "component", "children": [{"name": "z", "type": "unknown", "implicit": false, "dtype": "ndarray"}]}, {"name": "sub", "type": "subsystem", "subsystem_type": "group", "children": [{"name": "state_eq_group", "type": "subsystem", "subsystem_type": "group", "children": [{"name": "state_eq", "type": "subsystem", "subsystem_type": "component", "children": [{"name": "y2_actual", "type": "param", "dtype": "ndarray"}, {"name": "y2_command", "type": "unknown", "implicit": true, "dtype": "ndarray"}]}]}, {"name": "d1", "type": "subsystem", "subsystem_type": "component", "children": [{"name": "z", "type": "param", "dtype": "ndarray"}, {"name": "x", "type": "param", "dtype": "ndarray"}, {"name": "y2", "type": "param", "dtype": "ndarray"}, {"name": "y1", "type": "unknown", "implicit": false, "dtype": "ndarray"}]}, {"name": "d2", "type": "subsystem", "subsystem_type": "component", "children": [{"name": "z", "type": "param", "dtype": "ndarray"}, {"name": "y1", "type": "param", "dtype": "ndarray"}, {"name": "y2", "type": "unknown", "implicit": false, "dtype": "ndarray"}]}]}, {"name": "obj_cmp", "type": "subsystem", "subsystem_type": "component", "children": [{"name": "x", "type": "param", "dtype": "ndarray"}, {"name": "y1", "type": "param", "dtype": "ndarray"}, {"name": "y2", "type": "param", "dtype": "ndarray"}, {"name": "z", "type": "param", "dtype": "ndarray"}, {"name": "obj", "type": "unknown", "implicit": false, "dtype": "ndarray"}]}, {"name": "con_cmp1", "type": "subsystem", "subsystem_type": "component", "children": [{"name": "y1", "type": "param", "dtype": "ndarray"}, {"name": "con1", "type": "unknown", "implicit": false, "dtype": "ndarray"}]}, {"name": "con_cmp2", "type": "subsystem", "subsystem_type": "component", "children": [{"name": "y2", "type": "param", "dtype": "ndarray"}, {"name": "con2", "type": "unknown", "implicit": false, "dtype": "ndarray"}]}]}'
        self.expected_conns_json = '[{"src": "sub.d1.y1", "tgt": "con_cmp1.y1"}, {"src": "sub.d2.y2", "tgt": "con_cmp2.y2"}, {"src": "px.x", "tgt": "obj_cmp.x"}, {"src": "sub.d1.y1", "tgt": "obj_cmp.y1"}, {"src": "sub.d2.y2", "tgt": "obj_cmp.y2"}, {"src": "pz.z", "tgt": "obj_cmp.z"}, {"src": "px.x", "tgt": "sub.d1.x"}, {"src": "sub.state_eq_group.state_eq.y2_command", "tgt": "sub.d1.y2"}, {"src": "pz.z", "tgt": "sub.d1.z"}, {"src": "sub.d1.y1", "tgt": "sub.d2.y1"}, {"src": "pz.z", "tgt": "sub.d2.z"}, {"src": "sub.d2.y2", "tgt": "sub.state_eq_group.state_eq.y2_actual", "cycle_arrows": ["sub.d1 sub.d2", "sub.state_eq_group.state_eq sub.d1"]}]'

    def tearDown(self):
        try:
            rmtree(self.dir)
        except OSError as e:
            # If directory already deleted, keep going
            if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                raise e

    def test_model_viewer_has_correct_data_from_problem(self):
        """
        Verify that the correct model structure data exists when stored as compared
        to the expected structure, using the SellarStateConnection model.
        """
        p = Problem()
        p.model = SellarStateConnection()
        p.setup(check=False)
        model_viewer_data = _get_viewer_data(p)
        tree_json = json.dumps(model_viewer_data['tree'])
        conns_json = json.dumps(model_viewer_data['connections_list'])

        self.assertEqual(self.expected_tree_json, tree_json)
        self.assertEqual(self.expected_conns_json, conns_json)

    def test_model_viewer_has_correct_data_from_sqlite(self):
        """
        Verify that the correct data exists when a model structure is recorded
        and then pulled out of a sqlite db file and compared to the expected
        structure.  Uses the SellarStateConnection model.
        """
        p = Problem()
        p.model = SellarStateConnection()
        r = SqliteRecorder(self.sqlite_db_filename)
        p.driver.add_recorder(r)
        p.setup(check=False)
        p.final_setup()
        r.shutdown()

        model_viewer_data = _get_viewer_data(self.sqlite_db_filename)

        expected_tree = json.loads(self.expected_tree_json)
        expected_conns = json.loads(self.expected_conns_json)

        self.assertDictEqual(expected_tree, model_viewer_data['tree'])
        self.assertEqual(expected_conns, model_viewer_data['connections_list'])

    def test_view_model_from_problem(self):
        """
        Test that an n2 html file is generated from a Problem.
        """
        p = Problem()
        p.model = SellarStateConnection()
        p.setup(check=False)
        view_model(p, outfile=self.problem_filename, show_browser=False)

        # Check that the html file has been created and has something in it.
        self.assertTrue(os.path.isfile(self.problem_html_filename), (self.problem_html_filename + " is not a valid file."))
        self.assertGreater(os.path.getsize(self.problem_html_filename), 100)

    def test_view_model_from_sqlite(self):
        """
        Test that an n2 html file is generated from a sqlite file.
        """
        p = Problem()
        p.model = SellarStateConnection()
        r = SqliteRecorder(self.sqlite_db_filename2)
        p.driver.add_recorder(r)
        p.setup(check=False)
        p.final_setup()
        r.shutdown()
        view_model(self.sqlite_db_filename2, outfile=self.sqlite_filename, show_browser=False)

        # Check that the html file has been created and has something in it.
        self.assertTrue(os.path.isfile(self.sqlite_html_filename), (self.problem_html_filename + " is not a valid file."))
        self.assertGreater(os.path.getsize(self.sqlite_html_filename), 100)


if __name__ == "__main__":
    unittest.main()
