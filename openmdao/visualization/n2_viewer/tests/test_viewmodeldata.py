""" Unit tests for the problem interface."""

import unittest
import os
import json
import re
import base64
import zlib

import errno
from shutil import rmtree
from tempfile import mkdtemp

from openmdao.api import Problem, IndepVarComp, ScipyOptimizeDriver
from openmdao.test_suite.components.sellar import SellarStateConnection
from openmdao.visualization.n2_viewer.n2_viewer import _get_viewer_data, n2
from openmdao.recorders.sqlite_recorder import SqliteRecorder
from openmdao.test_suite.test_examples.test_betz_limit import ActuatorDisc
from openmdao.utils.shell_proc import check_call
from openmdao.utils.assert_utils import assert_warning


# Whether to pop up a browser window for each N2
DEBUG_BROWSER = False

# set DEBUG_FILES to True if you want to view the generated HTML file(s)
DEBUG_FILES = False


class TestViewModelData(unittest.TestCase):

    def setUp(self):
        if not DEBUG_FILES:
            self.dir = mkdtemp()
        else:
            self.dir = os.getcwd()

        self.sqlite_db_filename = os.path.join(self.dir, "sellarstate_model.sqlite")
        self.sqlite_db_filename2 = os.path.join(self.dir, "sellarstate_model_view.sqlite")
        self.compare_html_filename = os.path.join(self.dir, "compare_n2.html")
        self.sqlite_html_filename = os.path.join(self.dir, "sqlite_n2.html")
        self.problem_html_filename = os.path.join(self.dir, "problem_n2.html")
        self.title_html_filename = os.path.join(self.dir, "title_n2.html")
        self.conn_html_filename = os.path.join(self.dir, "conn_n2.html")

        self.expected_tree = json.loads("""
            {
               "name":"root",
               "type":"root",
               "class":"SellarStateConnection",
               "expressions":null,
               "component_type":null,
               "subsystem_type":"group",
               "is_parallel":false,
               "linear_solver":"LN: SCIPY",
               "nonlinear_solver":"NL: Newton",
               "solve_subsystems":false,
               "children":[
                   {
                     "name":"sub",
                     "type":"subsystem",
                     "class":"Group",
                     "expressions":null,
                     "component_type":null,
                     "subsystem_type":"group",
                     "is_parallel":false,
                     "linear_solver":"LN: SCIPY",
                     "nonlinear_solver":"NL: RUNONCE",
                     "children":[
                        {
                           "name":"state_eq_group",
                           "type":"subsystem",
                           "class":"Group",
                           "expressions":null,
                           "component_type":null,
                           "subsystem_type":"group",
                           "is_parallel":false,
                           "linear_solver":"LN: SCIPY",
                           "nonlinear_solver":"NL: RUNONCE",
                           "children":[
                              {
                                 "name":"state_eq",
                                 "type":"subsystem",
                                 "class":"StateConnection",
                                 "expressions":null,
                                 "subsystem_type":"component",
                                 "is_parallel":false,
                                 "component_type":"implicit",
                                 "linear_solver":"",
                                 "nonlinear_solver":"",
                                 "children":[
                                    {
                                       "name":"y2_actual",
                                       "type":"input",
                                       "dtype":"ndarray"
                                    },
                                    {
                                       "name":"y2_command",
                                       "type":"output",
                                       "implicit":true,
                                       "dtype":"ndarray"
                                    }
                                 ]
                              }
                           ]
                        },
                        {
                           "name":"d1",
                           "type":"subsystem",
                           "class":"SellarDis1withDerivatives",
                           "expressions":null,
                           "subsystem_type":"component",
                           "is_parallel":false,
                           "component_type":"explicit",
                           "linear_solver":"",
                           "nonlinear_solver":"",
                           "children":[
                              {
                                 "name":"z",
                                 "type":"input",
                                 "dtype":"ndarray"
                              },
                              {
                                 "name":"x",
                                 "type":"input",
                                 "dtype":"ndarray"
                              },
                              {
                                 "name":"y2",
                                 "type":"input",
                                 "dtype":"ndarray"
                              },
                              {
                                 "name":"y1",
                                 "type":"output",
                                 "implicit":false,
                                 "dtype":"ndarray"
                              }
                           ]
                        },
                        {
                           "name":"d2",
                           "type":"subsystem",
                           "class":"SellarDis2withDerivatives",
                           "expressions":null,
                           "subsystem_type":"component",
                           "is_parallel":false,
                           "component_type":"explicit",
                           "linear_solver":"",
                           "nonlinear_solver":"",
                           "children":[
                              {
                                 "name":"z",
                                 "type":"input",
                                 "dtype":"ndarray"
                              },
                              {
                                 "name":"y1",
                                 "type":"input",
                                 "dtype":"ndarray"
                              },
                              {
                                 "name":"y2",
                                 "type":"output",
                                 "implicit":false,
                                 "dtype":"ndarray"
                              }
                           ]
                        }
                     ]
                  },
                  {
                     "name":"obj_cmp",
                     "type":"subsystem",
                     "class":"ExecComp",
                     "expressions":[
                        "obj = x**2 + z[1] + y1 + exp(-y2)"
                     ],
                     "subsystem_type":"component",
                     "is_parallel":false,
                     "component_type":"exec",
                     "linear_solver":"",
                     "nonlinear_solver":"",
                     "children":[
                        {
                           "name":"x",
                           "type":"input",
                           "dtype":"ndarray"
                        },
                        {
                           "name":"y1",
                           "type":"input",
                           "dtype":"ndarray"
                        },
                        {
                           "name":"y2",
                           "type":"input",
                           "dtype":"ndarray"
                        },
                        {
                           "name":"z",
                           "type":"input",
                           "dtype":"ndarray"
                        },
                        {
                           "name":"obj",
                           "type":"output",
                           "implicit":false,
                           "dtype":"ndarray"
                        }
                     ]
                  },
                  {
                     "name":"con_cmp1",
                     "type":"subsystem",
                     "class":"ExecComp",
                     "expressions":[
                        "con1 = 3.16 - y1"
                     ],
                     "subsystem_type":"component",
                     "is_parallel":false,
                     "component_type":"exec",
                     "linear_solver":"",
                     "nonlinear_solver":"",
                     "children":[
                        {
                           "name":"y1",
                           "type":"input",
                           "dtype":"ndarray"
                        },
                        {
                           "name":"con1",
                           "type":"output",
                           "implicit":false,
                           "dtype":"ndarray"
                        }
                     ]
                  },
                  {
                     "name":"con_cmp2",
                     "type":"subsystem",
                     "class":"ExecComp",
                     "expressions":[
                        "con2 = y2 - 24.0"
                     ],
                     "subsystem_type":"component",
                     "is_parallel":false,
                     "component_type":"exec",
                     "linear_solver":"",
                     "nonlinear_solver":"",
                     "children":[
                        {
                           "name":"y2",
                           "type":"input",
                           "dtype":"ndarray"
                        },
                        {
                           "name":"con2",
                           "type":"output",
                           "implicit":false,
                           "dtype":"ndarray"
                        }
                     ]
                  }
               ]
            }
        """)
        self.expected_pathnames = json.loads('["sub.d1", "sub.d2", "sub.state_eq_group.state_eq"]')
        self.expected_conns = json.loads("""
            [
                {"src": "sub.d1.y1", "tgt": "con_cmp1.y1"},
                {"src": "sub.d2.y2", "tgt": "con_cmp2.y2"},
                {"src": "_auto_ivc.v1", "tgt": "obj_cmp.x"},
                {"src": "sub.d1.y1", "tgt": "obj_cmp.y1"},
                {"src": "sub.d2.y2", "tgt": "obj_cmp.y2"},
                {"src": "_auto_ivc.v0", "tgt": "obj_cmp.z"},
                {"src": "_auto_ivc.v1", "tgt": "sub.d1.x"},
                {"src": "sub.state_eq_group.state_eq.y2_command", "tgt": "sub.d1.y2"},
                {"src": "_auto_ivc.v0", "tgt": "sub.d1.z"},
                {"src": "sub.d1.y1", "tgt": "sub.d2.y1"},
                {"src": "_auto_ivc.v0", "tgt": "sub.d2.z"},
                {"src": "sub.d2.y2", "tgt": "sub.state_eq_group.state_eq.y2_actual", "cycle_arrows": ["sub.d1 sub.d2", "sub.state_eq_group.state_eq sub.d1"]}
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
                    "x": "x",
                    "z": "z",
                    "sub.state_eq_group.state_eq.y2_command": "state_eq.y2_command",
                    "sub.d1.y1": "y1",
                    "sub.d2.y2": "d2.y2",
                    "obj_cmp.obj": "obj",
                    "con_cmp1.con1": "con1",
                    "con_cmp2.con2": "con2"
                }
            }
        """)

        self.expected_declare_partials = json.loads("""
        ["sub.state_eq_group.state_eq.y2_command > sub.state_eq_group.state_eq.y2_actual", "sub.d1.y1 > sub.d1.z", "sub.d1.y1 > sub.d1.x", "sub.d1.y1 > sub.d1.y2", "sub.d2.y2 > sub.d2.z", "sub.d2.y2 > sub.d2.y1", "obj_cmp.obj > obj_cmp.x", "obj_cmp.obj > obj_cmp.y1", "obj_cmp.obj > obj_cmp.y2", "obj_cmp.obj > obj_cmp.z", "con_cmp1.con1 > con_cmp1.y1", "con_cmp2.con2 > con_cmp2.y2"]
        """)

        self.expected_driver_name = 'Driver'
        self.expected_design_vars_names = []
        self.expected_responses_names = []

    def tearDown(self):
        if not DEBUG_FILES:
            try:
                rmtree(self.dir)
            except OSError as e:
                # If directory already deleted, keep going
                if e.errno not in (errno.ENOENT, errno.EACCES, errno.EPERM):
                    raise e

    def check_model_viewer_data(self, model_viewer_data,
                                expected_tree,
                                expected_pathnames,
                                expected_conns,
                                expected_abs2prom,
                                expected_declare_partials,
                                expected_driver_name,
                                expected_design_vars_names,
                                expected_responses_names
                                ):
        self.assertDictEqual(model_viewer_data['tree'], expected_tree)

        # check expected system pathnames
        pathnames = model_viewer_data['sys_pathnames_list']
        self.assertListEqual(sorted(pathnames), expected_pathnames)

        # check expected connections, after mapping cycle_arrows indices back to pathnames
        connections = sorted(model_viewer_data['connections_list'],
                             key=lambda x: (x['tgt'], x['src']))
        expected_conns = sorted(expected_conns,
                                key=lambda x: (x['tgt'], x['src']))
        self.assertEqual(len(connections), len(expected_conns))
        for conn in connections:
            if 'cycle_arrows' in conn and conn['cycle_arrows']:
                cycle_arrows = []
                for src, tgt in conn['cycle_arrows']:
                    cycle_arrows.append(' '.join([pathnames[src], pathnames[tgt]]))
                conn['cycle_arrows'] = sorted(cycle_arrows)
        for c, ex in zip(connections, expected_conns):
            self.assertEqual(c['src'], ex['src'])
            self.assertEqual(c['tgt'], ex['tgt'])
            self.assertEqual(c.get('cycle_arrows', []), ex.get('cycle_arrows', []))

        # check expected abs2prom map
        self.assertListEqual(sorted(model_viewer_data['abs2prom']), sorted(expected_abs2prom))

        # check expected declare_partials_list
        self.assertListEqual(sorted(model_viewer_data['declare_partials_list']),
                             sorted(expected_declare_partials))

        self.assertEqual(model_viewer_data['driver']['name'], expected_driver_name)

        self.assertListEqual(sorted(dv for dv in model_viewer_data['design_vars']),
                             sorted(expected_design_vars_names))
        self.assertListEqual(sorted(resp for resp in model_viewer_data['responses']),
                             sorted(expected_responses_names))

    def test_model_viewer_has_correct_data_from_problem(self):
        """
        Verify that the correct model structure data exists when stored as compared
        to the expected structure, using the SellarStateConnection model.
        """
        p = Problem(model=SellarStateConnection())
        p.setup()
        p.final_setup()

        model_viewer_data = _get_viewer_data(p)

        # check expected model tree
        self.check_model_viewer_data(
            model_viewer_data,
            self.expected_tree,
            self.expected_pathnames,
            self.expected_conns,
            self.expected_abs2prom,
            self.expected_declare_partials,
            self.expected_driver_name,
            self.expected_design_vars_names,
            self.expected_responses_names
        )

    def test_model_viewer_has_correct_data_from_sqlite(self):
        """
        Verify that the correct data exists when a model structure is recorded
        and then pulled out of a sqlite db file and compared to the expected
        structure.  Uses the SellarStateConnection model.
        """
        p = Problem(model=SellarStateConnection())

        r = SqliteRecorder(self.sqlite_db_filename)
        p.driver.add_recorder(r)

        p.setup()
        p.final_setup()
        r.shutdown()

        model_viewer_data = _get_viewer_data(self.sqlite_db_filename)

        # check expected model tree
        self.check_model_viewer_data(
            model_viewer_data,
            self.expected_tree,
            self.expected_pathnames,
            self.expected_conns,
            self.expected_abs2prom,
            self.expected_declare_partials,
            self.expected_driver_name,
            self.expected_design_vars_names,
            self.expected_responses_names
        )

    def test_model_viewer_has_correct_data_from_optimization_problem(self):
        """
        Verify that the correct model structure data exists when stored as compared
        to the expected structure, using the ActuatorDisc model.
        """

        # build the model
        prob = Problem()
        indeps = prob.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        indeps.add_output('a', .5)
        indeps.add_output('Area', 10.0, units='m**2')
        indeps.add_output('rho', 1.225, units='kg/m**3')
        indeps.add_output('Vu', 10.0, units='m/s')

        prob.model.add_subsystem('a_disk', ActuatorDisc(),
                                 promotes_inputs=['a', 'Area', 'rho', 'Vu'])

        # setup the optimization
        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'

        prob.model.add_design_var('a', lower=0., upper=1.)
        prob.model.add_design_var('Area', lower=0., upper=1.)

        # negative one so we maximize the objective
        prob.model.add_objective('a_disk.Cp', scaler=-1)

        prob.setup()
        prob.final_setup()
        
        model_viewer_data = _get_viewer_data(prob)

        expected_tree_betz = json.loads("""
            { 
               "name":"root",
               "type":"root",
               "class":"Group",
               "expressions":null,
               "component_type":null,
               "subsystem_type":"group",
               "is_parallel":false,
               "linear_solver":"LN: RUNONCE",
               "nonlinear_solver":"NL: RUNONCE",
               "children":[ 
                  { 
                     "name":"indeps",
                     "type":"subsystem",
                     "class":"IndepVarComp",
                     "expressions":null,
                     "subsystem_type":"component",
                     "is_parallel":false,
                     "component_type":"indep",
                     "linear_solver":"",
                     "nonlinear_solver":"",
                     "children":[ 
                        { 
                           "name":"a",
                           "type":"output",
                           "implicit":false,
                           "dtype":"ndarray"
                        },
                        { 
                           "name":"Area",
                           "type":"output",
                           "implicit":false,
                           "dtype":"ndarray"
                        },
                        { 
                           "name":"rho",
                           "type":"output",
                           "implicit":false,
                           "dtype":"ndarray"
                        },
                        { 
                           "name":"Vu",
                           "type":"output",
                           "implicit":false,
                           "dtype":"ndarray"
                        }
                     ]
                  },
                  { 
                     "name":"a_disk",
                     "type":"subsystem",
                     "class":"ActuatorDisc",
                     "expressions":null,
                     "subsystem_type":"component",
                     "is_parallel":false,
                     "component_type":"explicit",
                     "linear_solver":"",
                     "nonlinear_solver":"",
                     "children":[ 
                        { 
                           "name":"a",
                           "type":"input",
                           "dtype":"ndarray"
                        },
                        { 
                           "name":"Area",
                           "type":"input",
                           "dtype":"ndarray"
                        },
                        { 
                           "name":"rho",
                           "type":"input",
                           "dtype":"ndarray"
                        },
                        { 
                           "name":"Vu",
                           "type":"input",
                           "dtype":"ndarray"
                        },
                        { 
                           "name":"Vr",
                           "type":"output",
                           "implicit":false,
                           "dtype":"ndarray"
                        },
                        { 
                           "name":"Vd",
                           "type":"output",
                           "implicit":false,
                           "dtype":"ndarray"
                        },
                        { 
                           "name":"Ct",
                           "type":"output",
                           "implicit":false,
                           "dtype":"ndarray"
                        },
                        { 
                           "name":"thrust",
                           "type":"output",
                           "implicit":false,
                           "dtype":"ndarray"
                        },
                        { 
                           "name":"Cp",
                           "type":"output",
                           "implicit":false,
                           "dtype":"ndarray"
                        },
                        { 
                           "name":"power",
                           "type":"output",
                           "implicit":false,
                           "dtype":"ndarray"
                        }
                     ]
                  }
               ]
            }
        """)
        expected_pathnames_betz = json.loads('[]')
        expected_conns_betz = json.loads("""
                [{"src": "indeps.a", "tgt": "a_disk.a"}, {"src": "indeps.Area", "tgt": "a_disk.Area"},
                 {"src": "indeps.rho", "tgt": "a_disk.rho"}, {"src": "indeps.Vu", "tgt": "a_disk.Vu"}]
                """)
        expected_abs2prom_betz = json.loads("""
                    {"input": {"a_disk.a": "a", "a_disk.Area": "Area", "a_disk.rho": "rho", "a_disk.Vu": "Vu"},
                    "output": {"indeps.a": "a", "indeps.Area": "Area", "indeps.rho": "rho", "indeps.Vu": "Vu",
                    "a_disk.Vr": "a_disk.Vr", "a_disk.Vd": "a_disk.Vd", "a_disk.Ct": "a_disk.Ct",
                    "a_disk.thrust": "a_disk.thrust", "a_disk.Cp": "a_disk.Cp",
                    "a_disk.power": "a_disk.power"}}
                    """)
        expected_declare_partials_betz = json.loads("""
                ["a_disk.Vr > a_disk.a", "a_disk.Vr > a_disk.Vu", "a_disk.Vd > a_disk.a", "a_disk.Ct > a_disk.a", "a_disk.thrust > a_disk.a", "a_disk.thrust > a_disk.Area", "a_disk.thrust > a_disk.rho", "a_disk.thrust > a_disk.Vu", "a_disk.Cp > a_disk.a", "a_disk.power > a_disk.a", "a_disk.power > a_disk.Area", "a_disk.power > a_disk.rho", "a_disk.power > a_disk.Vu"]
        """)

        expected_driver_name = 'ScipyOptimizeDriver'
        expected_design_vars_names = ['indeps.a',  'indeps.Area']
        expected_responses_names = ['a_disk.Cp',]

        # check expected model tree
        self.check_model_viewer_data(
            model_viewer_data,
            expected_tree_betz,
            expected_pathnames_betz,
            expected_conns_betz,
            expected_abs2prom_betz,
            expected_declare_partials_betz,
            expected_driver_name,
            expected_design_vars_names,
            expected_responses_names,
        )

    def test_n2_from_problem(self):
        """
        Test that an n2 html file is generated from a Problem.
        """
        p = Problem()
        p.model = SellarStateConnection()
        p.setup()
        n2(p, outfile=self.problem_html_filename, show_browser=DEBUG_BROWSER)

        # Check that the html file has been created and has something in it.
        self.assertTrue(os.path.isfile(self.problem_html_filename),
                        (self.problem_html_filename + " is not a valid file."))
        self.assertGreater(os.path.getsize(self.problem_html_filename), 100)

    def _extract_compressed_model(self, filename):
        """
        Load an N2 html, find the compressed data string, uncompress and decode it.
        """
        file = open(filename, 'r')
        for line in file:
            if re.search('var compressedModel', line):
                b64_data = line.replace('var compressedModel = "', '').replace('";', '')
                break
        
        file.close()
        compressed_data = base64.b64decode(b64_data)
        model_data = json.loads(zlib.decompress(compressed_data).decode("utf-8"))

        return model_data

    def test_n2_from_sqlite(self):
        """
        Test that an n2 html file is generated from a sqlite file.
        """
        p = Problem()
        p.model = SellarStateConnection()
        r = SqliteRecorder(self.sqlite_db_filename2)
        p.driver.add_recorder(r)
        p.setup()
        p.final_setup()
        r.shutdown()
        n2(p, outfile=self.compare_html_filename, show_browser=DEBUG_BROWSER)
        n2(self.sqlite_db_filename2, outfile=self.sqlite_html_filename, show_browser=DEBUG_BROWSER)

        # Check that the html file has been created and has something in it.
        self.assertTrue(os.path.isfile(self.sqlite_html_filename),
                        (self.sqlite_html_filename + " is not a valid file."))
        self.assertGreater(os.path.getsize(self.sqlite_html_filename), 100)

        # Check that there are no errors when running from the command line with a recording.
        check_call('openmdao n2 --no_browser %s' % self.sqlite_db_filename2)

        # Compare models from the files generated from the Problem and the recording
        sqlite_model_data = self._extract_compressed_model(self.sqlite_html_filename)
        compare_model_data = self._extract_compressed_model(self.compare_html_filename)

        self.assertTrue(sqlite_model_data == compare_model_data,
                        'Model data from sqlite does not match data from Problem.')

    def test_n2_command(self):
        """
        Check that there are no errors when running from the command line with a script.
        """
        from openmdao.test_suite.scripts import sellar
        filename = os.path.abspath(sellar.__file__).replace('.pyc', '.py')  # PY2
        check_call('openmdao n2 --no_browser %s' % filename)

    def test_n2_set_title(self):
        """
        Test that an n2 html file is generated from a Problem.
        """
        p = Problem()
        p.model = SellarStateConnection()
        p.setup()
        n2(p, outfile=self.title_html_filename, show_browser=DEBUG_BROWSER,
           title="Sellar State Connection")

        # Check that the html file has been created and has something in it.
        self.assertTrue(os.path.isfile(self.title_html_filename),
                        (self.title_html_filename + " is not a valid file."))
        self.assertTrue('OpenMDAO Model Hierarchy and N2 diagram: Sellar State Connection'
                        in open(self.title_html_filename).read())

    def test_n2_connection_error(self):
        """
        Test that an n2 html file is generated from a Problem even if it has connection errors.
        """
        from openmdao.test_suite.scripts.bad_connection import BadConnectionModel 

        p = Problem(BadConnectionModel())

        # this would be set by the command line hook
        p.model._raise_connection_errors = False

        expected = "Group (sub): Attempted to connect from 'tgt.x' to 'cmp.x', but " + \
                   "'tgt.x' is an input. All connections must be from an output to an input."

        with assert_warning(UserWarning, expected):
            p.setup()

        n2(p, outfile=self.conn_html_filename, show_browser=DEBUG_BROWSER,
           title="Bad Connection")

        # Check that the html file has been created and has something in it.
        self.assertTrue(os.path.isfile(self.conn_html_filename),
                        (self.conn_html_filename + " is not a valid file."))
        self.assertTrue('OpenMDAO Model Hierarchy and N2 diagram: Bad Connection'
                        in open(self.conn_html_filename).read())


if __name__ == "__main__":
    unittest.main()
