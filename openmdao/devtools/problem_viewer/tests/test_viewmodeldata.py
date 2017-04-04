""" Unit tests for the problem interface."""

import unittest
import warnings
from openmdao.api import Problem
from openmdao.test_suite.components.sellar import SellarStateConnection
from openmdao.devtools.problem_viewer.problem_viewer import _get_viewer_data
import json

class TestViewModelData(unittest.TestCase):

    def test_view_model_has_correct_data(self):
        #self.maxDiff = None
        expected_tree_json = '{"name": "root", "type": "root", "promotions": {"con_cmp2.y2": "con_cmp2.y2", "obj_cmp.y2": "obj_cmp.y2", "sub.d1.y2": "d1.y2", "sub.d2.y2": "d2.y2", "sub.state_eq_group.state_eq.y2_actual": "state_eq.y2_actual", "sub.state_eq_group.state_eq.y2_command": "state_eq.y2_command"}, "subsystem_type": "group", "children": [{"name": "px", "type": "subsystem", "subsystem_type": "component", "children": [{"name": "x", "type": "unknown", "implicit": false, "dtype": "ndarray"}]}, {"name": "pz", "type": "subsystem", "subsystem_type": "component", "children": [{"name": "z", "type": "unknown", "implicit": false, "dtype": "ndarray"}]}, {"name": "sub", "type": "subsystem", "promotions": {"sub.d1.y2": "d1.y2", "sub.d2.y2": "d2.y2", "sub.state_eq_group.state_eq.y2_actual": "state_eq.y2_actual", "sub.state_eq_group.state_eq.y2_command": "state_eq.y2_command"}, "subsystem_type": "group", "children": [{"name": "state_eq_group", "type": "subsystem", "promotions": {"sub.state_eq_group.state_eq.y2_actual": "state_eq.y2_actual", "sub.state_eq_group.state_eq.y2_command": "state_eq.y2_command"}, "subsystem_type": "group", "children": [{"name": "state_eq", "type": "subsystem", "subsystem_type": "component", "children": [{"name": "y2_actual", "type": "param", "dtype": "ndarray"}, {"name": "y2_command", "type": "unknown", "implicit": true, "dtype": "ndarray"}]}]}, {"name": "d1", "type": "subsystem", "subsystem_type": "component", "children": [{"name": "z", "type": "param", "dtype": "ndarray"}, {"name": "x", "type": "param", "dtype": "ndarray"}, {"name": "y2", "type": "param", "dtype": "ndarray"}, {"name": "y1", "type": "unknown", "implicit": false, "dtype": "ndarray"}]}, {"name": "d2", "type": "subsystem", "subsystem_type": "component", "children": [{"name": "z", "type": "param", "dtype": "ndarray"}, {"name": "y1", "type": "param", "dtype": "ndarray"}, {"name": "y2", "type": "unknown", "implicit": false, "dtype": "ndarray"}]}]}, {"name": "obj_cmp", "type": "subsystem", "subsystem_type": "component", "children": [{"name": "x", "type": "param", "dtype": "ndarray"}, {"name": "y1", "type": "param", "dtype": "ndarray"}, {"name": "y2", "type": "param", "dtype": "ndarray"}, {"name": "z", "type": "param", "dtype": "ndarray"}, {"name": "obj", "type": "unknown", "implicit": false, "dtype": "ndarray"}]}, {"name": "con_cmp1", "type": "subsystem", "subsystem_type": "component", "children": [{"name": "y1", "type": "param", "dtype": "ndarray"}, {"name": "con1", "type": "unknown", "implicit": false, "dtype": "ndarray"}]}, {"name": "con_cmp2", "type": "subsystem", "subsystem_type": "component", "children": [{"name": "y2", "type": "param", "dtype": "ndarray"}, {"name": "con2", "type": "unknown", "implicit": false, "dtype": "ndarray"}]}]}'
        expected_conns_json = '[{"src": "sub.d1.y1", "tgt": "con_cmp1.y1"}, {"src": "sub.d2.y2", "tgt": "con_cmp2.y2"}, {"src": "px.x", "tgt": "obj_cmp.x"}, {"src": "sub.d1.y1", "tgt": "obj_cmp.y1"}, {"src": "sub.d2.y2", "tgt": "obj_cmp.y2"}, {"src": "pz.z", "tgt": "obj_cmp.z"}, {"src": "px.x", "tgt": "sub.d1.x"}, {"src": "sub.state_eq_group.state_eq.y2_command", "tgt": "sub.d1.y2"}, {"src": "pz.z", "tgt": "sub.d1.z"}, {"src": "sub.d1.y1", "tgt": "sub.d2.y1"}, {"src": "pz.z", "tgt": "sub.d2.z"}, {"src": "sub.d2.y2", "tgt": "sub.state_eq_group.state_eq.y2_actual", "cycle_arrows": ["sub.d1 sub.d2", "sub.state_eq_group.state_eq sub.d1"]}]'

        p = Problem()
        p.model=SellarStateConnection()
        p.setup(check=False)
        model_viewer_data = _get_viewer_data(p)
        tree_json = json.dumps(model_viewer_data['tree'])
        conns_json = json.dumps(model_viewer_data['connections_list'])


        self.assertEqual(expected_tree_json, tree_json)
        self.assertEqual(expected_conns_json, conns_json)




if __name__ == "__main__":
    unittest.main()
