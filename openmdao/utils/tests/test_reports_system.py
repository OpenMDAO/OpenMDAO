"""Unit Tests for the code that does automatic report generation"""
import unittest
import pathlib
import sys
import os
from io import StringIO

import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar_feature import SellarMDA
import openmdao.core.problem
from openmdao.utils.assert_utils import assert_warning
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.reports_system import set_reports_dir, _reports_dir, register_report, \
    get_reports_dir, list_reports, clear_reports, run_n2_report, setup_default_reports
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.mpi import MPI
from openmdao.utils.tests.test_hooks import hooks_active
from openmdao.visualization.n2_viewer.n2_viewer import _default_n2_filename
from openmdao.visualization.scaling_viewer.scaling_report import _default_scaling_filename

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

        # set things to a known initial state for all the test runs
        openmdao.core.problem._problem_names = []  # need to reset these to simulate separate runs
        os.environ.pop('OPENMDAO_REPORTS', None)
        os.environ.pop('OPENMDAO_REPORTS_DIR', None)
        # We need to remove the TESTFLO_RUNNING environment variable for these tests to run.
        # The reports code checks to see if TESTFLO_RUNNING is set and will not do anything if set
        # But we need to remember whether it was set so we can restore it
        self.testflo_running = os.environ.pop('TESTFLO_RUNNING', None)
        clear_reports()
        set_reports_dir(_reports_dir)
        setup_default_reports()

        self.count = 0

    def tearDown(self):
        # restore what was there before running the test
        if self.testflo_running is not None:
            os.environ['TESTFLO_RUNNING'] = self.testflo_running

    def setup_and_run_simple_problem(self, driver=None):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        if driver:
            prob.driver = driver

        else:
            prob.driver = om.ScipyOptimizeDriver()

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        return prob

    def setup_and_run_model_with_subproblem(self):
        class _ProblemSolver(om.NonlinearRunOnce):

            def __init__(self, prob_name=None):
                super(_ProblemSolver, self).__init__()
                self.prob_name = prob_name
                self._problem = None

            def solve(self):
                subprob = om.Problem(name=self.prob_name)
                self._problem = subprob
                subprob.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
                subprob.model.add_subsystem('comp', om.ExecComp('y=2*x'))
                subprob.model.connect('indep.x', 'comp.x')
                subprob.setup()
                subprob.run_model()

                return super().solve()

        prob = om.Problem()
        prob.model.add_subsystem('indep', om.IndepVarComp('x', 1.0))
        G = prob.model.add_subsystem('G', om.Group())
        G.add_subsystem('comp', om.ExecComp('y=2*x'))
        G.nonlinear_solver = _ProblemSolver()
        prob.model.connect('indep.x', 'G.comp.x')
        prob.setup()
        prob.run_model()  # need to do run_model in this test so sub problem is created

        probname = prob._name
        subprobname = G.nonlinear_solver._problem._name
        return probname, subprobname

    @hooks_active
    def test_report_generation_basic(self):
        prob = self.setup_and_run_simple_problem()

        # get the path to the problem subdirectory
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')

    @hooks_active
    def test_report_generation_basic_pyoptsparse(self):
        # Just to try a different driver
        prob = self.setup_and_run_simple_problem(driver=pyOptSparseDriver(optimizer='SLSQP'))

        # get the path to the problem subdirectory
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')

    @hooks_active
    def test_report_generation_basic_doedriver(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', om.IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.add_design_var('x', lower=0.0, upper=1.0)
        model.add_design_var('y', lower=0.0, upper=1.0)
        model.add_objective('f_xy')

        prob.driver = om.DOEDriver(om.PlackettBurmanGenerator())

        prob.setup()
        prob.run_driver()
        prob.cleanup()

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        # DOEDriver won't cause the creation of a scaling report
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(path.is_file(),
                         f'The scaling report file, {str(path)}, was found but should not have')

    @hooks_active
    def test_report_generation_list_reports(self):
        stdout = sys.stdout
        strout = StringIO()
        sys.stdout = strout
        try:
            list_reports()
        finally:
            sys.stdout = stdout

        output = strout.getvalue()
        self.assertTrue('N2 diagram' in output, '"N2 diagram" expected in list_reports output but was not found')
        self.assertTrue('Driver scaling report' in output, '"Driver scaling report" expected in list_reports output but was not found')

    @hooks_active
    def test_report_generation_no_reports(self):
        # test use of the OPENMDAO_REPORTS variable to turn off reporting
        os.environ['OPENMDAO_REPORTS'] = 'false'
        clear_reports()
        setup_default_reports()  # So it sees the OPENMDAO_REPORTS var

        prob = self.setup_and_run_simple_problem()

        # See if the report files exist and if they have the right names
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertFalse(path.is_file(),
                         f'The N2 report file, {str(path)} was found but should not have')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(path.is_file(),
                         f'The scaling report file, {str(path)}, was found but should not have')

    @hooks_active
    def test_report_generation_selected_reports(self):
        # test use of the OPENMDAO_REPORTS variable to turn off selected reports
        os.environ['OPENMDAO_REPORTS'] = 'n2'
        clear_reports()
        setup_default_reports()  # So it sees the OPENMDAO_REPORTS var

        prob = self.setup_and_run_simple_problem()

        # See if the report files exist and if they have the right names
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(path.is_file(),
                         f'The scaling report file, {str(path)}, was found but should not have')

    @hooks_active
    def test_report_generation_set_reports_dir(self):
        # test use of setting a custom reports directory other than the default of "."
        custom_dir = 'custom_reports_dir'
        os.environ['OPENMDAO_REPORTS_DIR'] = custom_dir

        prob = self.setup_and_run_simple_problem()

        # See if the report files exist and if they have the right names
        reports_dir = custom_dir
        problem_reports_dir = pathlib.Path(reports_dir).joinpath(f'{prob._name}_reports')

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertTrue(path.is_file(), f'The scaling report file, {str(path)}, was not found')

    @hooks_active
    def test_report_generation_user_defined_report(self):
        user_report_filename = 'user_report.txt'

        def user_defined_report(prob):
            problem_reports_dirpath = get_reports_dir(prob)
            user_report_filepath = str(
                pathlib.Path(problem_reports_dirpath).joinpath(user_report_filename))

            pathlib.Path(problem_reports_dirpath).mkdir(parents=True, exist_ok=True)

            with open(user_report_filepath, "w") as f:
                f.write(f"Do some reporting on the Problem, {prob._name}\n")

        register_report("User defined report", user_defined_report, "user defined report",
                        'Problem', 'setup', 'pre')

        prob = self.setup_and_run_simple_problem()

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

        path = pathlib.Path(problem_reports_dir).joinpath(user_report_filename)
        self.assertTrue(path.is_file(), f'The user defined report file, {str(path)} was not found')

    @hooks_active
    def test_report_generation_various_locations(self):
        # the reports can be generated pre and post for setup, final_setup, and run_driver
        # check those all work

        self.count = 0

        # A simple report
        def user_defined_report(prob):
            problem_reports_dirpath = get_reports_dir(prob)
            pathlib.Path(problem_reports_dirpath).mkdir(parents=True, exist_ok=True)

            # Use the self.count value to give the reports a unique name
            # Ideally we would be able to pass a filename in when registering a report
            #   but report registration does not support kwargs yet
            user_defined_report_filepath = str(pathlib.Path(problem_reports_dirpath).joinpath(f"user_defined_{self.count}.txt"))

            with open(user_defined_report_filepath, "w") as f:
                f.write(f"Do some reporting on the Problem, {prob._name}\n")
            self.count += 1

        for method in ['setup', 'final_setup', 'run_driver']:
            for pre_or_post in ['pre', 'post']:
                register_report(f"User defined report {method} {pre_or_post}", user_defined_report,
                                "user defined report", 'Problem', method, pre_or_post)

        prob = self.setup_and_run_simple_problem()

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

        self.count = 0
        for method in ['setup', 'final_setup', 'run_driver']:
            for pre_or_post in ['pre', 'post']:
                user_report_filename = f"user_defined_{self.count}.txt"
                path = pathlib.Path(problem_reports_dir).joinpath(user_report_filename)
                self.assertTrue(path.is_file(),
                                f'The user defined report file, {str(path)} was not found')
                self.count += 1

    @hooks_active
    def test_report_generation_multiple_problems(self):
        probname, subprobname = self.setup_and_run_model_with_subproblem()

        # The multiple problem code only runs model so no scaling reports to look for
        for problem_name in [probname, subprobname]:
            problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{problem_name}_reports')
            path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
            self.assertTrue(path.is_file(), f'N2 report file, {str(path)} was not found')

    @hooks_active
    def test_report_generation_multiple_problems_report_specific_problem(self):
        # test the ability to register a report with a specific Problem name rather
        #   than have the report run for all Problems

        # to simplify things, just do n2.
        clear_reports()
        register_report("n2_report", run_n2_report, 'N2 diagram', 'Problem', 'final_setup', 'post',
                        inst_id='problem2')

        probname, subprobname = self.setup_and_run_model_with_subproblem()

        # The multiple problem code only runs model so no scaling reports to look for
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{subprobname}_reports')
        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        # for the subproblem named problem2, there should be a report but not for problem1 since
        #    we specifically asked for just the instance of problem2
        self.assertTrue(path.is_file(), f'The n2 report file, {str(path)} was not found')

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{probname}_reports')
        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertFalse(path.is_file(),
                         f'The N2 report file, {str(path)} was found but should not have')

    @hooks_active
    def test_report_generation_test_TESTFLO_RUNNING(self):
        # need to do this here again even though it is done in setup, because otherwise
        # setup_default_reports won't see environment variable, TESTFLO_RUNNING
        os.environ['TESTFLO_RUNNING'] = 'true'
        clear_reports()
        setup_default_reports()

        prob = self.setup_and_run_simple_problem()

        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertFalse(path.is_file(),
                         f'The N2 report file, {str(path)} was found but should not have')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(path.is_file(),
                         f'The scaling report file, {str(path)}, was found but should not have')

@use_tempdirs
@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestReportsSystemMPI(unittest.TestCase):
    N_PROCS = 2

    def setUp(self):
        self.n2_filename = _default_n2_filename
        self.scaling_filename = _default_scaling_filename

        # set things to a known initial state for all the test runs
        openmdao.core.problem._problem_names = []  # need to reset these to simulate separate runs

        os.environ.pop('OPENMDAO_REPORTS', None)
        os.environ.pop('OPENMDAO_REPORTS_DIR', None)
        # We need to remove the TESTFLO_RUNNING environment variable for these tests to run. The reports code
        #   checks to see if TESTFLO_RUNNING is set and will not do anything if it is set
        # But we need to remember whether it was set so we can restore it
        self.testflo_running = os.environ.pop('TESTFLO_RUNNING', None)
        clear_reports()
        set_reports_dir(_reports_dir)
        setup_default_reports()

        self.count = 0  # used to keep a count of reports generated

    def tearDown(self):
        # restore what was there before running the test
        if self.testflo_running is not None:
            os.environ['TESTFLO_RUNNING'] = self.testflo_running

    @hooks_active
    def test_reports_system_mpi_basic(self): #  example taken from TestScipyOptimizeDriverMPI

        prob = om.Problem()
        prob.model = SellarMDA()
        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-8)

        prob.model.add_design_var('x', lower=0, upper=10)
        prob.model.add_design_var('z', lower=0, upper=10)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1', upper=0)
        prob.model.add_constraint('con2', upper=0)

        prob.setup()
        prob.set_solver_print(level=0)

        msg = "ScipyOptimizeDriver: The scaling report currently is not supported when running under MPI."
        with assert_warning(om.MPIWarning, msg):
            prob.run_driver()

        # get the path to the problem subdirectory
        problem_reports_dir = pathlib.Path(_reports_dir).joinpath(f'{prob._name}_reports')

        path = pathlib.Path(problem_reports_dir).joinpath(self.n2_filename)
        self.assertTrue(path.is_file(), f'The N2 report file, {str(path)} was not found')
        path = pathlib.Path(problem_reports_dir).joinpath(self.scaling_filename)
        self.assertFalse(path.is_file(),
                         f'The scaling report file, {str(path)}, was found but should not have')


if __name__ == '__main__':
    unittest.main()
