"""Unit Tests for the code that does automatic report generation"""
import unittest
import pathlib
import sys
import os
from io import StringIO

import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.groups.parallel_groups import Diamond
from openmdao.core.problem import _default_prob_name
import openmdao.core.problem as probmod
from openmdao.core.constants import _UNDEFINED
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.reports_system import set_reports_dir, _reports_dir, register_report, \
    list_reports, clear_reports, _reset_reports_dir, activate_report, _reports_registry
from openmdao.utils.testing_utils import use_tempdirs
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

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


@use_tempdirs
class TestReportsSystem(unittest.TestCase):
    def setUp(self):
        self.n2_filename = _default_n2_filename
        self.scaling_filename = _default_scaling_filename
        self.optimizer_filename = _default_optimizer_report_filename

        # set things to a known initial state for all the test runs
        probmod._clear_problem_names()  # need to reset these to simulate separate runs
        os.environ.pop('OPENMDAO_REPORTS', None)
        os.environ.pop('OPENMDAO_REPORTS_DIR', None)
        # We need to remove the TESTFLO_RUNNING environment variable for these tests to run.
        # The reports code checks to see if TESTFLO_RUNNING is set and will not do anything if set
        # But we need to remember whether it was set so we can restore it
        self.testflo_running = os.environ.pop('TESTFLO_RUNNING', None)
        clear_reports()
        set_reports_dir(_reports_dir)

        self.count = 0

    def tearDown(self):
        # restore what was there before running the test
        if self.testflo_running is not None:
            os.environ['TESTFLO_RUNNING'] = self.testflo_running

    def setup_and_run_simple_problem(self, driver=None, reports=_UNDEFINED, reports_dir=_UNDEFINED):
        if reports_dir is not _UNDEFINED:
            set_reports_dir(reports_dir)

        prob = om.Problem(reports=reports)
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        with assert_no_warning(om.OpenMDAOWarning):
            if driver:
                prob.driver = driver
            else:
                prob.driver = om.ScipyOptimizeDriver()

        prob.setup(check=False)
        prob.run_driver()
        prob.cleanup()

        return prob

    def setup_problem_w_errors(self, prob_name, driver=None, reports=_UNDEFINED, reports_dir=_UNDEFINED):
        if reports_dir is not _UNDEFINED:
            set_reports_dir(reports_dir)

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

        return prob._name, G.nonlinear_solver._problem._name

    @hooks_active
    def test_report_generation_basic(self):
        prob = self.setup_and_run_simple_problem()

        # get the path to the problem subdirectory
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(prob._name)

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.optimizer_filename)
        self.assertTrue(path.is_file(), f'The optimizer report file, {str(path)}, was not found')

    @hooks_active
    def test_report_generation_on_error(self):
        prob_name = 'error_problem'
        try:
            prob = self.setup_problem_w_errors(prob_name)
        except Exception as err:
            # get the path to the problem subdirectory
            problem_reports_dir = pathlib.Path(_reports_dir).joinpath(prob_name)

            path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
            self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')

            self.assertEqual(str(err),
                "\nCollected errors for problem 'error_problem':"
                "\n   <model> <class Group>: When connecting 'p1.x' to 'comp.x': index 1 is out of "
                "bounds for source dimension of size 1.")
        else:
            self.fail("exception expected")

    @hooks_active
    @unittest.skipUnless(OPTIMIZER, "This test requires pyOptSparseDriver.")
    def test_report_generation_basic_pyoptsparse(self):
        # Just to try a different driver
        prob = self.setup_and_run_simple_problem(driver=pyOptSparseDriver(optimizer='SLSQP'))

        # get the path to the problem subdirectory
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(prob._name)

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

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(prob._name)

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
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(prob._name)

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
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(prob._name)

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
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(prob._name)

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
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(prob._name)

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
    def test_report_generation_set_reports_dir_using_env_var(self):
        # test use of setting a custom reports directory other than the default of "."
        custom_dir = 'custom_reports_dir'
        os.environ['OPENMDAO_REPORTS_DIR'] = custom_dir
        _reset_reports_dir()  # this will use current value of OPENMDAO_REPORTS_DIR

        prob = self.setup_and_run_simple_problem()

        # See if the report files exist and if they have the right names
        reports_dir = custom_dir
        problem_reports_dir = pathlib.Path(reports_dir).joinpath(prob._name)

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.optimizer_filename)
        self.assertTrue(path.is_file(), f'The optimizer report file, {str(path)}, was not found')

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
                        'Problem', 'setup', 'pre', report_filename=user_report_filename)

        prob = self.setup_and_run_simple_problem()

        path = pathlib.Path(_reports_dir).joinpath(prob._name, user_report_filename)

        self.assertTrue(path.is_file(), f'The user report file, {str(path)} was not found')

        # test unregister_report
        self.assertTrue('User report' in _reports_registry, "'User report' not found in registry.")
        om.unregister_report('User report')
        self.assertFalse('User report' in _reports_registry, "'User report' found in registry.")

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
                repname = f"User defined report {method} {pre_or_post}"
                register_report(repname, user_defined_report,
                                "user defined report", 'Problem', method, pre_or_post,
                                report_filename=user_report_filename)
                activate_report(repname)

        prob = self.setup_and_run_simple_problem()

        self.count = 0
        for _ in ['setup', 'final_setup', 'run_driver']:
            for _ in ['pre', 'post']:
                user_report_filename = f"user_defined_{self.count}.txt"
                path = pathlib.Path(prob.get_reports_dir()).joinpath(user_report_filename)
                self.assertTrue(path.is_file(),
                                f'The user defined report file, {str(path)} was not found')
                self.count += 1

    @hooks_active
    def test_report_generation_multiple_problems(self):
        probname, subprobname = self.setup_and_run_model_with_subproblem()

        # The multiple problem code only runs model so no scaling or optimizer reports to look for
        for problem_name in [probname, subprobname]:
            problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{problem_name}')
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

        probname, subprobname = self.setup_and_run_model_with_subproblem()

        self.assertEqual(_default_prob_name(), probname)

        # The multiple problem code only runs model so no scaling reports to look for
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{subprobname}')
        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        # for the subproblem named problem2, there should be a report but not for problem1 since
        #    we specifically asked for just the instance of problem2
        self.assertTrue(path.is_file(), f'The n2 report file, {str(path)} was not found')

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{probname}')
        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertFalse(path.is_file(),
                         f'The N2 report file, {str(path)} was found but should not exist.')

    @hooks_active
    def test_report_generation_test_TESTFLO_RUNNING(self):
        # need to do this here again even though it is done in setup, because otherwise
        # setup_reports won't see environment variable, TESTFLO_RUNNING
        os.environ['TESTFLO_RUNNING'] = 'true'
        clear_reports()

        prob = self.setup_and_run_simple_problem()

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(prob._name)

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
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(prob._name)
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
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(prob._name)

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
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(prob._name)

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
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(prob._name)

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.optimizer_filename)
        self.assertFalse(path.is_file(),
                         f'The optimizer report file, {str(path)}, was found but should not exist.')

    @hooks_active
    def test_report_generation_problem_reports_argument_multiple_problems(self):
        _, _ = self.setup_and_run_model_with_subproblem(prob2_reports=None)

        # Only problem1 reports should have been generated

        # The multiple problem code only runs model so no scaling reports to look for
        problem_name = _default_prob_name()
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{problem_name}')
        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The problem1 N2 report file, {str(path)} was not found')

        problem_name = problem_name + '2'
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{problem_name}')
        self.assertFalse(problem_reports_dir.is_dir(),
                         f'The problem2 report dir was found but should not exist.')
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
        os.environ.pop('OPENMDAO_REPORTS_DIR', None)
        # We need to remove the TESTFLO_RUNNING environment variable for these tests to run.
        # The reports code checks to see if TESTFLO_RUNNING is set and will not do anything
        # if it is set.
        # But we need to remember whether it was set so we can restore it
        self.testflo_running = os.environ.pop('TESTFLO_RUNNING', None)
        clear_reports()
        set_reports_dir(_reports_dir)

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
            problem_reports_dir = pathlib.Path(_reports_dir).joinpath(prob._name)

            path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
            self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
            path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
            self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')
            path = pathlib.Path(problem_reports_dir).joinpath(self.optimizer_filename)
            self.assertTrue(path.is_file(), f'The optimizer report file, {str(path)}, was not found')


if __name__ == '__main__':
    unittest.main()
