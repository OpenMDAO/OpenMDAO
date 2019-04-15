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


# set DEBUG to True if you want to view the generated HTML file(s)
DEBUG = False


class TestViewModelData(unittest.TestCase):

    def setUp(self):
        if not DEBUG:
            self.dir = mkdtemp()
        else:
            self.dir = os.getcwd()

        self.sqlite_db_filename = os.path.join(self.dir, "sellarstate_model.sqlite")
        self.sqlite_db_filename2 = os.path.join(self.dir, "sellarstate_model_view.sqlite")
        self.sqlite_html_filename = os.path.join(self.dir, "sqlite_n2.html")
        self.problem_html_filename = os.path.join(self.dir, "problem_n2.html")

        self.expected_tree = json.loads('{"linear_solver": "LN: SCIPY", "name": "root", "component_type": null, "nonlinear_solver": "NL: Newton", "subsystem_type": "group", "children": [{"linear_solver": "", "name": "px", "component_type": "indep", "nonlinear_solver": "", "subsystem_type": "component", "children": [{"dtype": "ndarray", "type": "unknown", "name": "x", "implicit": false}], "type": "subsystem", "is_parallel": false}, {"linear_solver": "", "name": "pz", "component_type": "indep", "nonlinear_solver": "", "subsystem_type": "component", "children": [{"dtype": "ndarray", "type": "unknown", "name": "z", "implicit": false}], "type": "subsystem", "is_parallel": false}, {"linear_solver": "LN: SCIPY", "name": "sub", "component_type": null, "nonlinear_solver": "NL: RUNONCE", "subsystem_type": "group", "children": [{"linear_solver": "LN: SCIPY", "name": "state_eq_group", "component_type": null, "nonlinear_solver": "NL: RUNONCE", "subsystem_type": "group", "children": [{"linear_solver": "", "name": "state_eq", "component_type": "implicit", "nonlinear_solver": "", "subsystem_type": "component", "children": [{"dtype": "ndarray", "type": "param", "name": "y2_actual"}, {"dtype": "ndarray", "type": "unknown", "name": "y2_command", "implicit": true}], "type": "subsystem", "is_parallel": false}], "type": "subsystem", "is_parallel": false}, {"linear_solver": "", "name": "d1", "component_type": "explicit", "nonlinear_solver": "", "subsystem_type": "component", "children": [{"dtype": "ndarray", "type": "param", "name": "z"}, {"dtype": "ndarray", "type": "param", "name": "x"}, {"dtype": "ndarray", "type": "param", "name": "y2"}, {"dtype": "ndarray", "type": "unknown", "name": "y1", "implicit": false}], "type": "subsystem", "is_parallel": false}, {"linear_solver": "", "name": "d2", "component_type": "explicit", "nonlinear_solver": "", "subsystem_type": "component", "children": [{"dtype": "ndarray", "type": "param", "name": "z"}, {"dtype": "ndarray", "type": "param", "name": "y1"}, {"dtype": "ndarray", "type": "unknown", "name": "y2", "implicit": false}], "type": "subsystem", "is_parallel": false}], "type": "subsystem", "is_parallel": false}, {"linear_solver": "", "name": "obj_cmp", "component_type": "exec", "nonlinear_solver": "", "subsystem_type": "component", "children": [{"dtype": "ndarray", "type": "param", "name": "x"}, {"dtype": "ndarray", "type": "param", "name": "y1"}, {"dtype": "ndarray", "type": "param", "name": "y2"}, {"dtype": "ndarray", "type": "param", "name": "z"}, {"dtype": "ndarray", "type": "unknown", "name": "obj", "implicit": false}], "type": "subsystem", "is_parallel": false}, {"linear_solver": "", "name": "con_cmp1", "component_type": "exec", "nonlinear_solver": "", "subsystem_type": "component", "children": [{"dtype": "ndarray", "type": "param", "name": "y1"}, {"dtype": "ndarray", "type": "unknown", "name": "con1", "implicit": false}], "type": "subsystem", "is_parallel": false}, {"linear_solver": "", "name": "con_cmp2", "component_type": "exec", "nonlinear_solver": "", "subsystem_type": "component", "children": [{"dtype": "ndarray", "type": "param", "name": "y2"}, {"dtype": "ndarray", "type": "unknown", "name": "con2", "implicit": false}], "type": "subsystem", "is_parallel": false}], "type": "root", "is_parallel": false}')
        self.expected_pathnames = json.loads('["sub.d1", "sub.d2", "sub.state_eq_group.state_eq"]')
        self.expected_conns = json.loads("""
            [
                {"src": "sub.d1.y1", "tgt": "con_cmp1.y1"},
                {"src": "sub.d2.y2", "tgt": "con_cmp2.y2"},
                {"src": "px.x", "tgt": "obj_cmp.x"},
                {"src": "sub.d1.y1", "tgt": "obj_cmp.y1"},
                {"src": "sub.d2.y2", "tgt": "obj_cmp.y2"},
                {"src": "pz.z", "tgt": "obj_cmp.z"},
                {"src": "px.x", "tgt": "sub.d1.x"},
                {"src": "sub.state_eq_group.state_eq.y2_command", "tgt": "sub.d1.y2"},
                {"src": "pz.z", "tgt": "sub.d1.z"},
                {"src": "sub.d1.y1", "tgt": "sub.d2.y1"},
                {"src": "pz.z", "tgt": "sub.d2.z"},
                {"src": "sub.d2.y2", "tgt": "sub.state_eq_group.state_eq.y2_actual", "cycle_arrows": [[0, 1], [2, 0]]}
            ]
        """)
        self.expected_abs2prom = json.loads("""
            {
                "input": {
                    "sub.state_eq_group.state_eq.y2_actual": "state_eq.y2_actual",
                    "sub.d1.z": "z",
                    "sub.d1.x": "x",
                    "sub.d1.y2": "d1.y2",
                    "sub.d2.z": "z",
                    "sub.d2.y1": "y1",
                    "obj_cmp.x": "x",
                    "obj_cmp.y1": "y1",
                    "obj_cmp.y2": "obj_cmp.y2",
                    "obj_cmp.z": "z",
                    "con_cmp1.y1": "y1",
                    "con_cmp2.y2": "con_cmp2.y2"
                },
                "output": {
                    "px.x": "x",
                    "pz.z": "z",
                    "sub.state_eq_group.state_eq.y2_command": "state_eq.y2_command",
                    "sub.d1.y1": "y1",
                    "sub.d2.y2": "d2.y2",
                    "obj_cmp.obj": "obj",
                    "con_cmp1.con1": "con1",
                    "con_cmp2.con2": "con2"
                }
            }
        """)

    def tearDown(self):
        if not DEBUG:
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
        p = Problem(model=SellarStateConnection())
        p.setup(check=False)

        model_viewer_data = _get_viewer_data(p)

        # check expected model tree
        self.assertDictEqual(model_viewer_data['tree'], self.expected_tree)

        # check expected system pathnames
        pathnames = model_viewer_data['sys_pathnames_list']
        self.assertListEqual(sorted(pathnames), self.expected_pathnames)

        # check expected connections, after mapping cycle_arrows indices back to pathnames
        connections = model_viewer_data['connections_list']
        for conn in connections:
            if 'cycle_arrows' in conn:
                cycle_arrows = []
                for src, tgt in conn['cycle_arrows']:
                    cycle_arrows.append(' '.join([pathnames[src], pathnames[tgt]]))
                conn['cycle_arrows'] = sorted(cycle_arrows)
        self.assertListEqual(connections, self.expected_conns)

        # check expected abs2prom map
        self.assertDictEqual(model_viewer_data['abs2prom'], self.expected_abs2prom)

    def test_model_viewer_has_correct_data_from_sqlite(self):
        """
        Verify that the correct data exists when a model structure is recorded
        and then pulled out of a sqlite db file and compared to the expected
        structure.  Uses the SellarStateConnection model.
        """
        p = Problem(model=SellarStateConnection())

        r = SqliteRecorder(self.sqlite_db_filename)
        p.driver.add_recorder(r)

        p.setup(check=False)
        p.final_setup()
        r.shutdown()

        model_viewer_data = _get_viewer_data(self.sqlite_db_filename)

        # check expected model tree
        self.assertDictEqual(model_viewer_data['tree'], self.expected_tree)

        # check expected system pathnames
        pathnames = model_viewer_data['sys_pathnames_list']
        self.assertListEqual(sorted(pathnames), self.expected_pathnames)

        # check expected connections, after mapping cycle_arrows indices back to pathnames
        connections = model_viewer_data['connections_list']
        for conn in connections:
            if 'cycle_arrows' in conn:
                cycle_arrows = []
                for src, tgt in conn['cycle_arrows']:
                    cycle_arrows.append(' '.join([pathnames[src], pathnames[tgt]]))
                conn['cycle_arrows'] = sorted(cycle_arrows)
        self.assertListEqual(connections, self.expected_conns)

        # check expected abs2prom map
        self.assertDictEqual(model_viewer_data['abs2prom'], self.expected_abs2prom)

    def test_view_model_from_problem(self):
        """
        Test that an n2 html file is generated from a Problem.
        """
        p = Problem()
        p.model = SellarStateConnection()
        p.setup(check=False)
        view_model(p, outfile=self.problem_html_filename, show_browser=DEBUG)

        # Check that the html file has been created and has something in it.
        self.assertTrue(os.path.isfile(self.problem_html_filename),
                        (self.problem_html_filename + " is not a valid file."))
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
        view_model(self.sqlite_db_filename2, outfile=self.sqlite_html_filename, show_browser=DEBUG)

        # Check that the html file has been created and has something in it.
        self.assertTrue(os.path.isfile(self.sqlite_html_filename),
                        (self.problem_html_filename + " is not a valid file."))
        self.assertGreater(os.path.getsize(self.sqlite_html_filename), 100)


if __name__ == "__main__":
    unittest.main()
