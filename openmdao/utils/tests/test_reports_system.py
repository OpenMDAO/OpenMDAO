"""Unit Tests for the code that does automatic report generation"""
from importlib.util import find_spec
import unittest
import unittest.mock as mock
import pathlib
import shutil
import sys
import os
import threading
import time
from io import StringIO

import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.groups.parallel_groups import Diamond
from openmdao.core.problem import _default_prob_name, _clear_reports_dir
import openmdao.core.problem as probmod
from openmdao.core.constants import _UNDEFINED
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.reports_system import register_report, \
    list_reports, clear_reports, activate_report, _reports_registry
from openmdao.utils.testing_utils import use_tempdirs, set_env_vars, require_pyoptsparse
from openmdao.utils.assert_utils import assert_no_warning
from openmdao.utils.mpi import MPI
from openmdao.utils.tests.test_hooks import hooks_active
from openmdao.visualization.n2_viewer.n2_viewer import _default_n2_filename, _run_n2_report
from openmdao.visualization.scaling_viewer.scaling_report import _default_scaling_filename
from openmdao.visualization.opt_report.opt_report import _default_optimizer_report_filename

try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

OPT, OPTIMIZER = set_pyoptsparse_opt('SLSQP')


@use_tempdirs
class TestReportsSystem(unittest.TestCase):
    def setUp(self):
        self.n2_filename = _default_n2_filename
        self.scaling_filename = _default_scaling_filename
        self.optimizer_filename = _default_optimizer_report_filename

        # set things to a known initial state for all the test runs
        probmod._clear_problem_names()  # need to reset these to simulate separate runs
        os.environ.pop('OPENMDAO_REPORTS', None)
        # We need to remove the TESTFLO_RUNNING environment variable for these tests to run.
        # The reports code checks to see if TESTFLO_RUNNING is set and will not do anything if set
        # But we need to remember whether it was set so we can restore it
        self.testflo_running = os.environ.pop('TESTFLO_RUNNING', None)
        clear_reports()

        self.count = 0

    def tearDown(self):
        # restore what was there before running the test
        if self.testflo_running is not None:
            os.environ['TESTFLO_RUNNING'] = self.testflo_running

    def setup_and_run_simple_problem(self, driver=None, reports=_UNDEFINED, reports_dir=_UNDEFINED, linear=False):
        prob = om.Problem(reports=reports)
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')
        if linear:
            model.add_subsystem('con', om.ExecComp('y=x'))
            model.connect('f_xy', 'con.x')
            model.add_constraint('con.y', lower=0.0, linear=True)

        with assert_no_warning(om.OpenMDAOWarning):
            if driver:
                prob.driver = driver
            else:
                prob.driver = om.ScipyOptimizeDriver()

        prob.setup(check=False)
        prob.run_driver()
        prob.cleanup()

        return prob

    def setup_and_run_w_linear_only_dvs(self, driver, reports=_UNDEFINED, reports_dir=_UNDEFINED, shape=3):
        prob = om.Problem(reports=reports)
        prob.driver = driver
        model = prob.model

        ivc = model.add_subsystem('ivc', om.IndepVarComp())
        ivc.add_output('x', np.ones(shape))
        ivc.add_output('y', np.ones(shape))
        ivc.add_output('z', np.ones(shape))

        model.add_subsystem('comp', om.ExecComp('f_xy=x*2.-y*3.', shape=shape))
        model.add_subsystem('obj', om.ExecComp('obj = sum(x**2)', obj=1., x=np.ones(shape)))
        model.add_subsystem('con', om.ExecComp('y=x', shape=shape))
        model.add_subsystem('con2', om.ExecComp('y=sin(x)', shape=shape))
        model.add_subsystem('con3', om.ExecComp('y=.2*x', shape=shape), promotes_inputs=['x'])
        model.add_subsystem('con4', om.ExecComp('y=cos(x)', shape=shape), promotes_inputs=['x'])

        model.connect('ivc.x', 'comp.x')
        model.connect('ivc.y', 'comp.y')
        model.connect('comp.f_xy', 'obj.x')
        model.connect('ivc.z', 'con.x')
        model.connect('ivc.x', 'con2.x')

        model.add_design_var('ivc.x', lower=0.0, upper=1.0)
        model.add_design_var('ivc.y', lower=0.0, upper=1.0)
        model.add_design_var('ivc.z', lower=0.0, upper=1.0)
        model.add_design_var('x', lower=0.0, upper=1.0)

        model.add_objective('obj.obj')

        model.add_constraint('con.y', lower=0.0, linear=True)
        model.add_constraint('con3.y', lower=0.0, linear=True)
        model.add_constraint('con4.y', lower=0.0)
        model.add_constraint('con2.y', lower=0.0)

        prob.setup(check=False)
        prob.run_driver()
        prob.cleanup()

        return prob

    def setup_problem_w_errors(self, prob_name, driver=None, reports=_UNDEFINED, reports_dir=_UNDEFINED):
        prob = om.Problem(reports=reports, name=prob_name)
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0))
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0))
        model.add_subsystem('comp', Paraboloid(), promotes_outputs=['f_xy'])

        model.connect('p1.x', 'comp.x', src_indices=[0,1])
        model.connect('p2.y', 'comp.y')

        model.add_design_var('p1.x', lower=0.0, upper=1.0)
        model.add_design_var('p2.y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        if driver:
            prob.driver = driver
        else:
            prob.driver = om.ScipyOptimizeDriver()

        prob.setup(check=False)
        prob.run_driver()
        prob.cleanup()

        return prob

    def setup_and_run_model_with_subproblem(self, prob1_reports=_UNDEFINED,
                                            prob2_reports=_UNDEFINED):
        class _ProblemSolver(om.NonlinearRunOnce):

            def __init__(self, prob_name=None, reports=_UNDEFINED):
                super(_ProblemSolver, self).__init__()
                self.prob_name = prob_name
                self.reports = reports
                self._problem = None

            def solve(self):
                subprob = om.Problem(name=self.prob_name, reports=self.reports)
                self._problem = subprob
                subprob.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
                subprob.model.add_subsystem('comp', om.ExecComp('y=2*x'))
                subprob.model.connect('indep.x', 'comp.x')
                subprob.setup()
                subprob.run_model()

                return super().solve()

        prob = om.Problem(reports=prob1_reports)
        prob.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
        G = prob.model.add_subsystem('G', om.Group())
        G.add_subsystem('comp', om.ExecComp('y=2*x'))
        G.nonlinear_solver = _ProblemSolver(reports=prob2_reports)
        prob.model.connect('indep.x', 'G.comp.x')
        prob.setup()
        prob.run_model()  # need to do run_model in this test so sub problem is created

        return prob, G.nonlinear_solver._problem

    @hooks_active
    def test_report_generation_basic(self):
        prob = self.setup_and_run_simple_problem()

        # get the path to the problem subdirectory
        problem_reports_dir = prob.get_reports_dir()

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.optimizer_filename)
        self.assertTrue(path.is_file(), f'The optimizer report file, {str(path)}, was not found')

    @hooks_active
    @require_pyoptsparse('IPOPT')
    def test_report_generation_linear_only_dv_scaling_report_pyoptsparse(self):
        if not OPTIMIZER:
            raise unittest.SkipTest("This test requires pyOptSparseDriver.")

        driver = om.pyOptSparseDriver(optimizer='IPOPT')
        driver.declare_coloring()

        driver.opt_settings['max_iter'] = 1000
        driver.opt_settings['print_level'] = 0
        driver.opt_settings['mu_strategy'] = 'monotone'
        driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
        driver.opt_settings['tol'] = 1.0E-4
        driver.opt_settings['constr_viol_tol'] = 1.0E-4

        prob = self.setup_and_run_w_linear_only_dvs(driver=driver, reports=['scaling'], shape=(9,7))

        # get the path to the problem subdirectory
        problem_reports_dir = prob.get_reports_dir()

        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)} was not found')

    @hooks_active
    def test_report_generation_linear_only_dv_scaling_report_scipyopt(self):
        driver = om.ScipyOptimizeDriver(optimizer='SLSQP')
        driver.declare_coloring()

        prob = self.setup_and_run_w_linear_only_dvs(driver=driver, reports=['scaling'], shape=(9,7))

        # get the path to the problem subdirectory
        problem_reports_dir = prob.get_reports_dir()

        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)} was not found')

    @hooks_active
    def test_report_generation_on_error(self):
        prob_name = 'error_problem'
        try:
            self.setup_problem_w_errors(prob_name)
        except Exception as err:
            # get the path to the problem subdirectory
            problem_reports_dir = pathlib.Path(f'{prob_name}_out/reports')

            path = problem_reports_dir / self.n2_filename
            self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')

            self.assertEqual(str(err),
                "\nCollected errors for problem 'error_problem':"
                "\n   <model> <class Group>: Can't connect 'p1.x' to 'comp.x' when applying index [[0, 1]]: index 1 is out of bounds for source dimension of size 1.")
        else:
            self.fail("exception expected")

    @hooks_active
    @unittest.skipUnless(OPTIMIZER, "This test requires pyOptSparseDriver.")
    def test_report_generation_basic_pyoptsparse(self):
        # Just to try a different driver
        prob = self.setup_and_run_simple_problem(driver=om.pyOptSparseDriver(optimizer='SLSQP'))

        # get the path to the problem subdirectory
        problem_reports_dir = prob.get_reports_dir()

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.optimizer_filename)
        self.assertTrue(path.is_file(), f'The optimizer report file, {str(path)}, was not found')

    @hooks_active
    def test_report_generation_basic_doedriver(self):
        # design variable values as generated by Placket Burman DOE generator
        doe_list = [
            [('x', 0.), ('y', 0.)],
            [('x', 1.), ('y', 0.)],
            [('x', 0.), ('y', 1.)],
            [('x', 1.), ('y', 1.)]
        ]

        # Test a driver that does not generate scaling report
        prob = self.setup_and_run_simple_problem(driver=om.DOEDriver(doe_list))

        problem_reports_dir = prob.get_reports_dir()

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        # DOEDriver won't cause the creation of a scaling or optimizer report
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(path.is_file(),
                         f'The scaling report file, {str(path)}, was found but should not exist.')
        path = pathlib.Path(problem_reports_dir).joinpath(self.optimizer_filename)
        self.assertFalse(path.is_file(),
                         f'The optimizer report file, {str(path)}, was found but should not exist.')

    @hooks_active
    def test_report_generation_list_reports(self):
        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            list_reports(max_width=100)  # make width 100 to prevent word wrap
        finally:
            sys.stdout = stdout

        output = strout.getvalue()
        self.assertTrue('N2 diagram' in output,
                        '"N2 diagram" expected in list_reports output but was not found')
        self.assertTrue('Driver scaling report' in output,
                        '"Driver scaling report" expected in list_reports output but was not found')
        self.assertTrue('Summary of optimization' in output,
                        '"Summary of optimization" expected in list_reports output but was not found')

    @hooks_active
    def test_report_generation_no_reports_using_env_var(self):
        # test use of the OPENMDAO_REPORTS variable to turn off reporting
        os.environ['OPENMDAO_REPORTS'] = 'false'
        clear_reports()

        prob = self.setup_and_run_simple_problem()

        # See if the report files exist and if they have the right names
        problem_reports_dir = prob.get_reports_dir()

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertFalse(path.is_file(),
                         f'The N2 report file, {str(path)} was found but should not exist.')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(path.is_file(),
                         f'The scaling report file, {str(path)}, was found but should not exist.')
        path = pathlib.Path(problem_reports_dir).joinpath(self.optimizer_filename)
        self.assertFalse(path.is_file(),
                         f'The optimizer report file, {str(path)}, was found but should not exist.')

    @hooks_active
    def test_report_generation_selected_reports_using_env_var(self):
        # test use of the OPENMDAO_REPORTS variable to turn off selected reports
        os.environ['OPENMDAO_REPORTS'] = 'n2'
        clear_reports()

        prob = self.setup_and_run_simple_problem()

        # See if the report files exist and if they have the right names
        problem_reports_dir = prob.get_reports_dir()

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(path.is_file(),
                         f'The scaling report file, {str(path)}, was found but should not exist.')
        path = pathlib.Path(problem_reports_dir).joinpath(self.optimizer_filename)
        self.assertFalse(path.is_file(),
                         f'The optimizer report file, {str(path)}, was found but should not exist.')

    @hooks_active
    def test_report_generation_selected_reports_override_env_var(self):
        # test use of problem reports to override OPENMDAO_REPORTS
        os.environ['OPENMDAO_REPORTS'] = 'n2'
        clear_reports()

        prob = self.setup_and_run_simple_problem(reports=['optimizer', 'scaling'])

        # See if the report files exist and if they have the right names
        problem_reports_dir = prob.get_reports_dir()

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertFalse(path.is_file(),
                         f'The N2 report file, {str(path)} was found but should not exist.')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.optimizer_filename)
        self.assertTrue(path.is_file(), f'The optimizer report file, {str(path)}, was not found')

    @hooks_active
    def test_report_generation_selected_reports_override_env_var2(self):
        # test use of problem reports to override OPENMDAO_REPORTS. This time with two reports
        os.environ['OPENMDAO_REPORTS'] = 'n2,scaling'
        clear_reports()

        prob = self.setup_and_run_simple_problem(reports=False)

        # See if the report files exist and if they have the right names
        problem_reports_dir = prob.get_reports_dir()

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertFalse(path.is_file(),
                         f'The N2 report file, {str(path)}, was found but should not exist.')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(path.is_file(),
                         f'The scaling report file, {str(path)}, was found but should not exist.')
        path = pathlib.Path(problem_reports_dir).joinpath(self.optimizer_filename)
        self.assertFalse(path.is_file(),
                         f'The optimizer report file, {str(path)}, was found but should not exist.')

    @hooks_active
    def test_report_generation_user_defined_report(self):
        user_report_filename = 'user_report.txt'
        os.environ['OPENMDAO_REPORTS'] = 'User report'

        def user_defined_report(prob, report_filename):
            path = pathlib.Path(prob.get_reports_dir()).joinpath(report_filename)
            with open(path, "w") as f:
                f.write(f"Do some reporting on the Problem, {prob._name}\n")

        register_report("User report", user_defined_report,
                        "user report description",
                        'Problem', 'setup', 'post', report_filename=user_report_filename)

        prob = self.setup_and_run_simple_problem()

        path = prob.get_reports_dir() / user_report_filename

        self.assertTrue(path.is_file(), f'The user report file, {str(path)} was not found')

        # test unregister_report
        self.assertTrue('User report' in _reports_registry, "'User report' not found in registry.")
        om.unregister_report('User report')
        self.assertFalse('User report' in _reports_registry, "'User report' found in registry.")

    @hooks_active
    def test_register_report_pre_setup(self):
        user_report_filename = 'user_report.txt'
        os.environ['OPENMDAO_REPORTS'] = 'User report'

        def user_defined_report(prob, report_filename):
            path = pathlib.Path(prob.get_reports_dir()).joinpath(report_filename)
            with open(path, "w") as f:
                f.write(f"Do some reporting on the Problem, {prob._name}\n")

        with self.assertRaises(ValueError) as e:
            register_report("User report", user_defined_report,
                            "user report description",
                            'Problem', 'setup', 'pre', report_filename=user_report_filename)

        expected = 'Reports cannot be registered to execute pre-setup.'
        self.assertEqual(str(e.exception), expected)

    @hooks_active
    def test_report_generation_various_locations(self):
        # the reports can be generated pre and post for setup, final_setup, and run_driver
        # check those all work

        self.count = 0

        # A simple report
        user_report_filename = 'user_defined_{count}.txt'

        def user_defined_report(prob, report_filename):
            report_filepath = pathlib.Path(prob.get_reports_dir()).joinpath(report_filename.format(count=self.count))
            with open(report_filepath, "w") as f:
                f.write(f"Do some reporting on the Problem, {prob._name}\n")
            self.count += 1

        for method in ['setup', 'final_setup', 'run_driver']:
            for pre_or_post in ['pre', 'post']:
                if (method, pre_or_post) == ('setup', 'pre'):
                    continue
                repname = f"User defined report {method} {pre_or_post}"
                register_report(repname, user_defined_report,
                                "user defined report", 'Problem', method, pre_or_post,
                                report_filename=user_report_filename)
                activate_report(repname)

        prob = self.setup_and_run_simple_problem()

        self.count = 0
        for method in ['setup', 'final_setup', 'run_driver']:
            for pre_or_post in ['pre', 'post']:
                if (method, pre_or_post) == ('setup', 'pre'):
                    continue
                user_report_filename = f"user_defined_{self.count}.txt"
                path = pathlib.Path(prob.get_reports_dir()).joinpath(user_report_filename)
                self.assertTrue(path.is_file(),
                                f'The user defined report file, {str(path)} was not found')
                self.count += 1

    @hooks_active
    def test_report_generation_multiple_problems(self):
        prob, subprob = self.setup_and_run_model_with_subproblem()

        # The multiple problem code only runs model so no scaling or optimizer reports to look for
        for p in [prob, subprob]:
            problem_reports_dir = p.get_reports_dir()
            path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
            self.assertTrue(path.is_file(), f'N2 report file, {str(path)} was not found')

    @hooks_active
    def test_report_generation_multiple_problems_report_specific_problem(self):
        # test the ability to register a report with a specific Problem name rather
        #   than have the report run for all Problems
        os.environ['OPENMDAO_REPORTS'] = 'n2_report'

        # to simplify things, just do n2.
        clear_reports()
        register_report("n2_report", _run_n2_report, 'N2 diagram', 'Problem', 'final_setup', 'post',
                        report_filename=self.n2_filename,
                        inst_id=_default_prob_name() + '2')

        prob, subprob = self.setup_and_run_model_with_subproblem()

        self.assertEqual(_default_prob_name(), prob._name)

        # The multiple problem code only runs model so no scaling reports to look for
        problem_reports_dir = subprob.get_reports_dir()
        path = problem_reports_dir / self.n2_filename
        # for the subproblem named problem2, there should be a report but not for problem1 since
        #    we specifically asked for just the instance of problem2
        self.assertTrue(path.is_file(), f'The n2 report file, {str(path)} was not found')

        problem_reports_dir = prob.get_reports_dir()
        path = problem_reports_dir / self.n2_filename
        self.assertFalse(path.is_file(),
                         f'The N2 report file, {str(path)} was found but should not exist.')

    @hooks_active
    @set_env_vars(TESTFLO_RUNNING='true')
    def test_report_generation_test_TESTFLO_RUNNING(self):
        # need to do this here again even though it is done in setup, because otherwise
        # setup_reports won't see environment variable, TESTFLO_RUNNING
        clear_reports()

        prob = self.setup_and_run_simple_problem()

        problem_reports_dir = prob.get_reports_dir()

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertFalse(path.is_file(),
                         f'The N2 report file, {str(path)} was found but should not exist.')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(path.is_file(),
                         f'The scaling report file, {str(path)}, was found but should not exist.')
        path = pathlib.Path(problem_reports_dir).joinpath(self.optimizer_filename)
        self.assertFalse(path.is_file(),
                         f'The optimizer report file, {str(path)}, was found but should not exist.')

    @hooks_active
    def test_report_generation_basic_problem_reports_argument_false(self):
        prob = self.setup_and_run_simple_problem(reports=False)

        # get the path to the problem subdirectory
        problem_reports_dir = prob.get_reports_dir()
        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertFalse(path.is_file(),
                         f'The N2 report file, {str(path)} was found but should not exist.')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(path.is_file(),
                         f'The scaling report file, {str(path)}, was found but should not exist.')
        path = pathlib.Path(problem_reports_dir).joinpath(self.optimizer_filename)
        self.assertFalse(path.is_file(),
                         f'The optimizer report file, {str(path)}, was found but should not exist.')

    @hooks_active
    def test_report_generation_basic_problem_reports_argument_none(self):
        prob = self.setup_and_run_simple_problem(reports=None)

        # get the path to the problem subdirectory
        problem_reports_dir = prob.get_reports_dir()

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertFalse(path.is_file(),
                         f'The N2 report file, {str(path)} was found but should not exist.')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(path.is_file(),
                         f'The scaling report file, {str(path)}, was found but should not exist.')
        path = pathlib.Path(problem_reports_dir).joinpath(self.optimizer_filename)
        self.assertFalse(path.is_file(),
                         f'The optimizer report file, {str(path)}, was found but should not exist.')

    @hooks_active
    def test_report_generation_basic_problem_reports_argument_n2_only(self):
        prob = self.setup_and_run_simple_problem(reports='n2')

        # get the path to the problem subdirectory
        problem_reports_dir = prob.get_reports_dir()

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(path.is_file(),
                         f'The scaling report file, {str(path)}, was found but should not exist.')
        path = pathlib.Path(problem_reports_dir).joinpath(self.optimizer_filename)
        self.assertFalse(path.is_file(),
                         f'The optimizer report file, {str(path)}, was found but should not exist.')

    @hooks_active
    def test_report_generation_basic_problem_reports_argument_n2_and_scaling(self):
        prob = self.setup_and_run_simple_problem(reports=['n2','scaling'])

        # get the path to the problem subdirectory
        problem_reports_dir = prob.get_reports_dir()

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.optimizer_filename)
        self.assertFalse(path.is_file(),
                         f'The optimizer report file, {str(path)}, was found but should not exist.')

    @hooks_active
    def test_report_generation_problem_reports_argument_multiple_problems(self):
        prob, subprob = self.setup_and_run_model_with_subproblem(prob2_reports=None)

        # Only problem1 reports should have been generated

        # The multiple problem code only runs model so no scaling reports to look for
        problem_reports_dir = prob.get_reports_dir()
        path = problem_reports_dir / self.n2_filename
        self.assertTrue(path.is_file(), f'The problem1 N2 report file, {str(path)} was not found')

        problem_reports_dir = subprob.get_reports_dir()
        self.assertFalse(problem_reports_dir.is_dir(),
                         'The problem2 report dir was found but should not exist.')
        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertFalse(path.is_file(),
                         f'The problem2 n2 report file, {str(path)}, was found but should not exist.')

    @hooks_active
    def test_report_generation_basic_problem_reports_dir_argument(self):
        custom_reports_dir = 'user_dir'

        prob = self.setup_and_run_simple_problem(reports=False, reports_dir=custom_reports_dir)

        # get the path to the problem subdirectory
        problem_reports_dir = pathlib.Path(custom_reports_dir).joinpath(prob._name)
        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertFalse(path.is_file(),
                         f'The N2 report file, {str(path)} was found but should not exist.')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(path.is_file(),
                         f'The scaling report file, {str(path)}, was found but should not exist.')

    @hooks_active
    def test_report_generation_extra_compute_totals_from_scaling_report(self):
        clear_reports()
        if find_spec('pyoptsparse') is None:
            raise unittest.SkipTest("pyoptsparse is required.")
        prob = self.setup_and_run_simple_problem(driver=om.pyOptSparseDriver(optimizer='SLSQP'),
                                                 reports=['scaling'], linear=True)

        self.assertEqual(prob.driver.result.deriv_evals, 3)

        # See if the report files exist and if they have the right names
        problem_reports_dir = prob.get_reports_dir()

        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)} was not found')


@use_tempdirs
@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestReportsSystemMPI(unittest.TestCase):
    N_PROCS = 2

    def setUp(self):
        self.n2_filename = _default_n2_filename
        self.scaling_filename = _default_scaling_filename
        self.optimizer_filename = _default_optimizer_report_filename

        # set things to a known initial state for all the test runs
        probmod._clear_problem_names()  # need to reset these to simulate separate runs

        os.environ.pop('OPENMDAO_REPORTS', None)
        # We need to remove the TESTFLO_RUNNING environment variable for these tests to run.
        # The reports code checks to see if TESTFLO_RUNNING is set and will not do anything
        # if it is set.
        # But we need to remember whether it was set so we can restore it
        self.testflo_running = os.environ.pop('TESTFLO_RUNNING', None)
        clear_reports()

        self.count = 0  # used to keep a count of reports generated

    def tearDown(self):
        # restore what was there before running the test
        if self.testflo_running is not None:
            os.environ['TESTFLO_RUNNING'] = self.testflo_running

    @hooks_active
    def test_reports_system_mpi_basic(self):
        prob = om.Problem()
        prob.model = Diamond()
        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-8)

        prob.model.add_design_var('iv.x', lower=0, upper=10)
        prob.model.add_objective('sub.c2.y1')
        prob.model.add_constraint('sub.c3.y1', upper=0)

        prob.setup()
        prob.set_solver_print(level=0)

        prob.run_driver()

        if prob.comm.rank == 0:
            problem_reports_dir = prob.get_reports_dir()

            path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
            self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
            path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
            self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')
            path = pathlib.Path(problem_reports_dir).joinpath(self.optimizer_filename)
            self.assertTrue(path.is_file(), f'The optimizer report file, {str(path)}, was not found')


@use_tempdirs
class TestSetupReportsDirRace(unittest.TestCase):
    """
    Tests for the reports directory handling in Problem.setup (issue #3746).

    During setup, stale reports must be removed, but the reports directory itself must not be
    deleted while other ranks/processes sharing the output tree may be concurrently creating it,
    because Path.mkdir(exist_ok=True) can raise a spurious FileExistsError if the directory
    vanishes between mkdir's EEXIST and its internal is_dir() check (python/cpython#142916).
    """

    # generous timeout for the orchestration events below.  The events fire microseconds apart
    # in a normal run; the timeout only bounds a hang if orchestration is broken.
    TIMEOUT = 30.0

    def setUp(self):
        probmod._clear_problem_names()  # need to reset these to simulate separate runs
        os.environ.pop('OPENMDAO_REPORTS', None)
        # The reports code does nothing if TESTFLO_RUNNING is set, so remove it for these tests
        # and remember its value so it can be restored.
        self.testflo_running = os.environ.pop('TESTFLO_RUNNING', None)
        clear_reports()

    def tearDown(self):
        if self.testflo_running is not None:
            os.environ['TESTFLO_RUNNING'] = self.testflo_running

    def _make_problem(self, name, reports='n2'):
        prob = om.Problem(name=name, reports=reports)
        prob.model.add_subsystem('comp', om.ExecComp('y = 2.0*x'))
        return prob

    def test_setup_no_spurious_fileexists_from_concurrent_mkdir(self):
        # Deterministic regression test for issue #3746.
        #
        # Interleaving exercised (all filesystem operations and exceptions are real; the
        # events below only fix the order in which the two actors reach them):
        #
        #   "rank 1" (worker thread): calls prob.get_reports_dir(force=True), the same
        #       production call every rank makes during Problem.setup.  Its exists() check
        #       sees no reports dir, so it proceeds into Path.mkdir(parents=True,
        #       exist_ok=True).
        #   "rank 0" (main thread): calls prob.setup() with active reports.
        #
        #   1. rank 1 reaches os.mkdir for the reports dir and waits for rank 0 to create it.
        #   2. rank 0's setup creates the reports dir.
        #   3. rank 1's os.mkdir now raises a genuine FileExistsError from the kernel.
        #   4. if rank 0's setup deletes the reports directory itself (the pre-fix behavior),
        #      that deletion is allowed to complete here, inside pathlib's window between
        #      os.mkdir raising EEXIST and its is_dir() verification.
        #   5. rank 1's pathlib mkdir then runs its real is_dir() check.
        #
        # On code that deletes the reports directory during setup, step 5 finds no directory
        # and mkdir(exist_ok=True) raises FileExistsError for a path that does not exist.
        # With setup only clearing the directory's contents, no deletion ever occurs, the
        # is_dir() check sees the directory, and the mkdir call succeeds.
        prob = self._make_problem('race_prob')
        self.assertTrue(len(prob._reports) > 0)  # active reports required for this scenario
        prob.setup()

        leaf = prob.get_outputs_dir() / 'reports'
        if leaf.is_dir():
            os.rmdir(leaf)  # start with no reports dir, as on a first run
        leaf_str = os.fspath(leaf)

        rank1_entered = threading.Event()   # rank 1 is inside os.mkdir for the reports dir
        created = threading.Event()         # rank 0's setup has created the reports dir
        eexist = threading.Event()          # rank 1 has its genuine EEXIST in hand
        deleted = threading.Event()         # rank 0's setup has deleted the reports dir
        rank1_done = threading.Event()      # rank 1's mkdir call has fully completed
        setup_done = threading.Event()      # rank 0's setup() has returned

        real_mkdir = os.mkdir
        real_rmtree = shutil.rmtree
        result = {}

        def instrumented_mkdir(path, mode=0o777, *args, **kwargs):
            if os.fspath(path) != leaf_str:
                return real_mkdir(path, mode, *args, **kwargs)
            if threading.current_thread().name == 'rank1':
                rank1_entered.set()
                if not created.wait(self.TIMEOUT):
                    raise RuntimeError('orchestration timeout waiting for creation')
                try:
                    real_mkdir(path, mode, *args, **kwargs)
                except OSError:
                    eexist.set()
                    # give the setup-side deletion (if the implementation performs one) time
                    # to land inside pathlib's EEXIST -> is_dir() window
                    t0 = time.perf_counter()
                    while not (deleted.is_set() or setup_done.is_set()):
                        if time.perf_counter() - t0 > self.TIMEOUT:
                            raise RuntimeError('orchestration timeout in EEXIST window')
                        time.sleep(0.001)
                    raise
            else:
                # rank 0 re-creating the dir after a deletion must not beat rank 1's
                # is_dir() check, else the race window closes by luck
                if deleted.is_set() and not rank1_done.is_set():
                    if not rank1_done.wait(self.TIMEOUT):
                        raise RuntimeError('orchestration timeout waiting for rank 1')
                real_mkdir(path, mode, *args, **kwargs)
                created.set()

        def instrumented_rmtree(path, *args, **kwargs):
            if os.fspath(path) == leaf_str:
                if not eexist.wait(self.TIMEOUT):
                    raise RuntimeError('orchestration timeout waiting for EEXIST')
                real_rmtree(path, *args, **kwargs)
                deleted.set()
            else:
                real_rmtree(path, *args, **kwargs)

        def rank1():
            try:
                result['dir'] = prob.get_reports_dir(force=True)
            except BaseException as exc:
                result['exc'] = exc
            finally:
                rank1_done.set()

        with mock.patch('os.mkdir', instrumented_mkdir), \
                mock.patch('shutil.rmtree', instrumented_rmtree):
            t = threading.Thread(target=rank1, name='rank1')
            t.start()
            self.assertTrue(rank1_entered.wait(self.TIMEOUT), 'rank 1 never reached mkdir')
            try:
                prob.setup()
            finally:
                setup_done.set()
                t.join(self.TIMEOUT)

        self.assertFalse(t.is_alive(), 'rank 1 thread did not finish')
        if 'exc' in result:
            raise result['exc']
        self.assertTrue(pathlib.Path(result['dir']).is_dir())

    def test_setup_clears_stale_report_files(self):
        # stale files in an existing reports dir must be gone after setup, and the reports
        # dir must exist afterward when reports are active
        prob = self._make_problem('stale_prob')
        prob.setup()

        reports_dir = prob.get_outputs_dir() / 'reports'
        self.assertTrue(reports_dir.is_dir())
        stale = reports_dir / 'stale_report.html'
        stale.write_text('from a previous run')
        stale_sub = reports_dir / 'stale_subdir'
        stale_sub.mkdir()
        (stale_sub / 'nested.html').write_text('nested stale content')

        prob.setup()

        self.assertTrue(reports_dir.is_dir())
        self.assertFalse(stale.exists())
        self.assertFalse(stale_sub.exists())

    def test_setup_removes_stale_reports_dir_when_reports_inactive(self):
        # when no reports are active, nothing will create the reports dir, so a stale one
        # from a previous run must be removed entirely, as before
        prob = self._make_problem('inactive_prob')
        prob.setup()

        reports_dir = prob.get_outputs_dir() / 'reports'
        self.assertTrue(reports_dir.is_dir())
        (reports_dir / 'stale_report.html').write_text('from a previous run')

        probmod._clear_problem_names()
        prob2 = self._make_problem('inactive_prob', reports=False)
        self.assertEqual(len(prob2._reports), 0)
        prob2.setup()

        self.assertFalse(reports_dir.exists())

    def test_clear_reports_dir_tolerates_concurrent_removal(self):
        # a peer process removing an entry (or the whole dir) first is a benign outcome
        workdir = pathlib.Path('clear_tol')
        workdir.mkdir()
        for fname in ('a.html', 'b.html'):
            (workdir / fname).write_text('x')

        real_unlink = os.unlink
        calls = {'n': 0}

        def racing_unlink(path, *args, **kwargs):
            calls['n'] += 1
            if calls['n'] == 1:
                real_unlink(path, *args, **kwargs)  # a peer removed this entry first
                raise FileNotFoundError(2, 'No such file or directory', os.fspath(path))
            real_unlink(path, *args, **kwargs)

        with mock.patch('os.unlink', racing_unlink):
            _clear_reports_dir(workdir)

        self.assertTrue(workdir.is_dir())
        self.assertEqual(list(workdir.iterdir()), [])

        # the whole directory having been removed by a peer is also benign
        _clear_reports_dir(workdir / 'no_such_dir')

    def test_clear_reports_dir_propagates_real_errors(self):
        # only the benign concurrent-removal outcome is tolerated; real filesystem errors
        # must propagate
        workdir = pathlib.Path('clear_err')
        workdir.mkdir()
        (workdir / 'a.html').write_text('x')

        def denied_unlink(path, *args, **kwargs):
            raise PermissionError(13, 'Permission denied', os.fspath(path))

        with mock.patch('os.unlink', denied_unlink):
            with self.assertRaises(PermissionError):
                _clear_reports_dir(workdir)

    @unittest.skipIf(sys.platform == 'win32', 'symlink creation requires privileges on Windows')
    def test_clear_reports_dir_unlinks_symlinks_without_following(self):
        workdir = pathlib.Path('clear_sym')
        workdir.mkdir()
        target_dir = pathlib.Path('sym_target')
        target_dir.mkdir()
        keep = target_dir / 'keep.txt'
        keep.write_text('do not delete')

        os.symlink(target_dir.resolve(), workdir / 'link_to_dir')
        os.symlink('no_such_file', workdir / 'dangling')

        _clear_reports_dir(workdir)

        self.assertEqual(list(workdir.iterdir()), [])
        self.assertTrue(keep.is_file())  # symlinked-to contents must be untouched

    @unittest.skipIf(sys.platform == 'win32',
                     'directory file descriptors are not supported on Windows')
    def test_setup_preserves_reports_dir_identity(self):
        # supplementary architecture coverage: with active reports, setup must not replace
        # the reports directory with a new one.  A deleted-and-recreated directory can reuse
        # the same inode number, so instead of comparing st_ino, watch the original
        # directory's link count through an open descriptor: it drops to 0 if the directory
        # is unlinked.
        prob = self._make_problem('ident_prob')
        prob.setup()

        reports_dir = prob.get_outputs_dir() / 'reports'
        fd = os.open(reports_dir, os.O_RDONLY)
        try:
            prob.setup()

            self.assertTrue(reports_dir.is_dir())
            self.assertGreater(os.fstat(fd).st_nlink, 0,
                               'the original reports directory was deleted during setup')
        finally:
            os.close(fd)


if __name__ == '__main__':
    unittest.main()
