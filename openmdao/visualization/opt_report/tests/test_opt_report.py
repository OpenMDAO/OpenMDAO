import os
import pathlib
import unittest

import numpy as np

import openmdao.api as om

from openmdao.core.constants import INF_BOUND
from openmdao.core.driver import Driver
from openmdao.test_suite.components.expl_comp_array import TestExplCompArrayDense
from openmdao.test_suite.components.simple_comps import DoubleArrayComp
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped
from openmdao.utils.assert_utils import assert_warning

from openmdao.utils.mpi import MPI
from openmdao.utils.om_warnings import DriverWarning
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from openmdao.utils.tests.test_hooks import hooks_active

from openmdao.visualization.opt_report.opt_report import opt_report, \
    _default_optimizer_report_filename

try:
    import pyDOE2
except ImportError:
    pyDOE2 = None


@use_tempdirs
class TestOptimizationReport(unittest.TestCase):

    def setup_problem_and_run_driver(self, driver,
                                     vars_lower=-INF_BOUND, vars_upper=INF_BOUND,
                                     cons_lower=-INF_BOUND, cons_upper=INF_BOUND,
                                     final_setup_only=False,
                                     optimizer=None
                                     ):
        # build a simple model that can be used for testing opt reports with different optimizers
        self.prob = prob = om.Problem(reports='optimizer')
        prob.model.add_subsystem('parab', Paraboloid(), promotes_inputs=['x', 'y'])
        prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

        prob.model.set_input_defaults('x', 3.0)
        prob.model.set_input_defaults('y', -4.0)

        prob.driver = driver()

        if optimizer:
            driver_opts = self.prob.driver.options
            driver_opts['optimizer'] = optimizer
            # silence scipy & pyoptsparse
            if 'disp' in driver_opts:
                driver_opts['disp'] = False
            if 'print_results' in driver_opts:
                driver_opts['print_results'] = False

        prob.model.add_design_var('x', lower=vars_lower, upper=vars_upper)
        prob.model.add_design_var('y', lower=vars_lower, upper=vars_upper)
        prob.model.add_objective('parab.f_xy')

        prob.model.add_constraint('const.g', lower=cons_lower, upper=cons_upper, alias='ALIAS_TEST')

        prob.setup()

        if final_setup_only:
            prob.final_setup()
        else:
            prob.run_driver()

    def setup_sellar_problem(self):
        prob = om.Problem()
        prob.model = SellarDerivativesGrouped()

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = "SLSQP"
        prob.driver.options['print_results'] = False
        prob.driver.opt_settings['ACC'] = 1e-13
        prob.set_solver_print(level=0)
        prob.model.add_constraint('con2', upper=0.0)
        prob.model.add_objective('obj')

        return prob

    def check_opt_result(self, opt_result=None, expected=None):
        """
        Check that the data in the opt_result dict is valid for the run.
        """
        if opt_result is None:
            opt_result = self.prob.driver.opt_result
        if expected is None:
            expected = {}

        self.assertTrue(opt_result['runtime'] > 0.0,
                        f"Unexpected value for runtime: {opt_result['runtime']} (should be > 0.0)")

        for key in ['iter_count', 'obj_calls', 'deriv_calls']:
            if key in expected:
                self.assertTrue( opt_result[key] == expected[key] ,
                    f"Unexpected value for {key}: {opt_result[key]}. Expected {expected[key]}")
            else:
                self.assertTrue( opt_result[key] >= 1,
                    f"Unexpected value for {key}: {opt_result[key]}. Expected value to be >= 1")

        self.assertTrue(opt_result['exit_status'] == expected['exit_status'] if 'exit_status' in expected
                        else opt_result['exit_status'] == 'SUCCESS',
                        f"Unexpected value for exit_status: {opt_result['exit_status']}")

    def check_opt_report(self, prob=None, expected=None):
        """
        Parse the opt_result data from the report and verify.
        """
        if prob is None:
            prob = self.prob
        report_file_path = str(pathlib.Path(prob.get_reports_dir()).joinpath(_default_optimizer_report_filename))

        check_rows = {
            # 'runtime': 'Wall clock run time:',
            'iter_count':  'Number of driver iterations:',
            'obj_calls':   'Number of objective calls:',
            'deriv_calls': 'Number of derivative calls:',
            'exit_status': 'Exit status:'
        }

        def second_cell(line):
            """
            get value from second cell of table row
            """
            value = line.split('>', 4)[-1].split('<', 1)[0].strip()
            if value == '':
                # The report will show an empty cell when the value is 'None'
                value = None
            else:
                try:
                    value = int(value)
                except ValueError as err:
                    pass
            return value

        reported = {'runtime': 1}  # we won't parse/check runtime

        with open(report_file_path, 'r') as f:
            for line in f.readlines():
                line = line.lstrip()
                if line.startswith('<tr'):
                    parts = line.split('>', 1)
                    if parts[1].startswith('<td'):
                        for key, row in check_rows.items():
                            if row in line:
                                reported[key] = second_cell(line)
                elif line.startswith('</table>'):
                    # skip the rest of the file
                    break

        self.check_opt_result(reported, expected)

    # First test all the different Drivers
    def test_opt_report_run_once_driver(self):
        self.setup_problem_and_run_driver(Driver,
                                          vars_lower=-50, vars_upper=50.,
                                          cons_lower=-1,
                                          )
        expect = {'obj_calls': 0, 'deriv_calls': 0}
        self.check_opt_result(expected=expect)

        expected_warning_msg = "The optimizer report is not applicable for Driver type 'Driver', " \
                               "which does not support optimization"
        with assert_warning(DriverWarning, expected_warning_msg):
            opt_report(self.prob)

        outfilepath = str(pathlib.Path(self.prob.get_reports_dir()).joinpath(_default_optimizer_report_filename))
        self.assertFalse(os.path.exists(outfilepath))

    def test_opt_report_scipyopt_SLSQP(self):
        self.setup_problem_and_run_driver(om.ScipyOptimizeDriver,
                                          vars_lower=-50, vars_upper=50.,
                                          cons_lower=0, cons_upper=10.,
                                          optimizer='SLSQP',
                                          )
        self.check_opt_result()
        opt_report(self.prob)
        self.check_opt_report()

    def test_opt_report_scipyopt_COBYLA(self):
        self.setup_problem_and_run_driver(om.ScipyOptimizeDriver,
                                          vars_lower=-50, vars_upper=50.,
                                          cons_lower=0, cons_upper=10.,
                                          optimizer='COBYLA',
                                          )
        expect = {'deriv_calls': None}
        self.check_opt_result(expected=expect)
        opt_report(self.prob)
        self.check_opt_report(expected=expect)

    def test_opt_report_scipyopt_COBYLA_nobounds(self):
        self.setup_problem_and_run_driver(om.ScipyOptimizeDriver,
                                          # no bounds on design vars
                                          cons_lower=0, cons_upper=10.,
                                          optimizer='COBYLA',
                                          )
        expect = {'deriv_calls': None}
        self.check_opt_result(expected=expect)
        opt_report(self.prob)
        self.check_opt_report(expected=expect)

    @require_pyoptsparse('SNOPT')
    def test_opt_report_pyoptsparse_snopt(self):
        self.setup_problem_and_run_driver(om.pyOptSparseDriver,
                                          vars_lower=-50, vars_upper=50.,
                                          cons_lower=0, cons_upper=10.,
                                          optimizer='SNOPT'
                                          )
        self.check_opt_result()
        opt_report(self.prob)
        self.check_opt_report()

    @require_pyoptsparse('SLSQP')
    def test_opt_report_pyoptsparse_SLSQP(self):
        self.setup_problem_and_run_driver(om.pyOptSparseDriver,
                                          vars_lower=-50, vars_upper=50.,
                                          cons_lower=0, cons_upper=10.,
                                          optimizer='SLSQP'
                                          )
        self.check_opt_result()
        opt_report(self.prob)
        self.check_opt_report()

    def test_opt_report_DOE(self):
        # no report should be generated for this
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])

        model.add_design_var('x', lower=-10, upper=10)
        model.add_design_var('y', lower=-10, upper=10)
        model.add_objective('f_xy')

        prob.driver = om.DOEDriver(om.UniformGenerator(num_samples=5))

        prob.setup()

        prob.set_val('x', 0.0)
        prob.set_val('y', 0.0)

        prob.run_driver()
        prob.cleanup()

        self.check_opt_result(prob.driver.opt_result, expected={'obj_calls': 0, 'deriv_calls': 0})

        expected_warning_msg = "The optimizer report is not applicable for Driver type 'DOEDriver', " \
                               "which does not support optimization"
        with assert_warning(DriverWarning, expected_warning_msg):
            opt_report(prob)
        outfilepath = str(pathlib.Path(prob.get_reports_dir()).joinpath(_default_optimizer_report_filename))
        self.assertFalse(os.path.exists(outfilepath))

    @unittest.skipUnless(pyDOE2, "requires 'pyDOE2', install openmdao[doe]")
    def test_opt_report_genetic_algorithm(self):
        self.setup_problem_and_run_driver(om.SimpleGADriver,
                                          vars_lower=-50, vars_upper=50.,
                                          cons_lower=0, cons_upper=10.,
                                          )
        expect = {'deriv_calls': 0}
        self.check_opt_result(expected=expect)
        opt_report(self.prob)
        self.check_opt_report(expected=expect)

    @unittest.skipUnless(pyDOE2, "requires 'pyDOE2', install openmdao[doe]")
    def test_opt_report_differential_evolution(self):
        prob = om.Problem()

        exec_comp = om.ExecComp(['y = x**2',
                                 'z = a + x**2'],
                                a={'shape': (1,)},
                                y={'shape': (101,)},
                                x={'shape': (101,)},
                                z={'shape': (101,)})

        prob.model.add_subsystem('exec', exec_comp)

        prob.model.add_design_var('exec.a', lower=-1000, upper=1000)
        prob.model.add_objective('exec.y', index=50)
        prob.model.add_constraint('exec.z', indices=[-1], lower=0)
        prob.model.add_constraint('exec.z', indices=[0], upper=300, alias="ALIAS_TEST")

        prob.driver = om.DifferentialEvolutionDriver()

        prob.setup()

        prob.set_val('exec.x', np.linspace(-10, 10, 101))

        prob.run_driver()

        expect = {'deriv_calls': 0}
        self.check_opt_result(prob.driver.opt_result, expected=expect)
        opt_report(prob)
        self.check_opt_report(prob, expected=expect)

    # Next test all the different variations of variable types (scalar and array) for both
    #  desvars and constraints. Also includes commented out code that can be uncommented to
    #  force the creation of reports with different visuals for the desvars and constraints
    #  as bounds are violated and satisfied

    @require_pyoptsparse('SLSQP')
    def test_opt_report_array_desvars(self):
        # test the calculations and plotting when there is a desvar that is an array
        prob = self.setup_sellar_problem()
        model = prob.model

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 4.0]))
        # use these when testing to see how plots look for different situations
        # model.add_design_var('z')
        # model.add_design_var('z', lower=np.array([-10.0, 0.0]))
        # model.add_design_var('z', upper=np.array([10.0, 4.0]))

        model.add_design_var('x')
        model.add_constraint('con1', upper=0.0)

        prob.setup(check=False, mode='rev')
        prob.run_driver()
        opt_report(prob)

    @require_pyoptsparse('SLSQP')
    def test_opt_report_scalar_desvars(self):
        # Same test as test_opt_report_array_desvars but the goal is to look
        # at the visuals for the scalar variable, 'x'.
        # Includes commented out code
        # that can be uncommented to see how the visuals for scalars look for the many situations.
        # Only run final_setup, so we can set where the values of the desvars are, and they won't
        # change when running the driver.
        prob = self.setup_sellar_problem()
        model = prob.model

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 4.0]))
        model.add_design_var('x', lower=0.0)  # lower only, at boundary
        # use these when testing to see how plots look for different situations
        # model.add_design_var('x')  # no lower or upper
        # model.add_design_var('x', lower=2.0)  # lower only, violated
        # model.add_design_var('x', lower=-1.0)  # lower only, satisfied
        # model.add_design_var('x', upper=1.0)  # upper only, at boundary
        # model.add_design_var('x', upper=-1.0)  # upper only, violated
        # model.add_design_var('x', upper=2.0)  # upper only, satisfied
        # model.add_design_var('x', lower=0.0, upper=2.0)  # lower and upper, satisfied
        # model.add_design_var('x', lower=1.5, upper=2.0)  # lower and upper, less than lower
        # model.add_design_var('x', lower=-1, upper=0.5)  # lower and upper, greater than upper
        model.add_constraint('con1', upper=0.0)

        prob.setup(check=False, mode='rev')
        prob.final_setup()
        opt_report(prob)

    @require_pyoptsparse('SLSQP')
    def test_opt_report_scalar_desvars_way_out_of_bounds(self):
        # test the visual where the desvar so far out of bounds that the visual
        # is drawn differently using ellipsis to indicate that the value is from being
        # within bounds
        prob = self.setup_sellar_problem()
        model = prob.model

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 4.0]))
        # use these when testing to see how plots look for different situations
        model.add_design_var('x', lower=10.0, upper=15.0)  # much less than lower
        # model.add_design_var('x', lower=-10.0, upper=-5.0)  # much greater than upper
        model.add_constraint('con1', upper=0.0)

        prob.setup(check=False, mode='rev')
        prob.final_setup()
        opt_report(prob)

    @require_pyoptsparse('SLSQP')
    def test_opt_report_scalar_constraint(self):
        # test the visual of the constraints
        prob = self.setup_sellar_problem()
        model = prob.model

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 4.0]))
        model.add_design_var('x', lower=10.0, upper=15.0)
        model.add_constraint('con1', upper=0.0)  # upper violated
        # use these when testing to see how plots look for different situations
        # model.add_constraint('con1', upper=1.0)  # upper at boundary
        # model.add_constraint('con1', upper=2.0)  # upper satisfied
        # model.add_constraint('con1', lower=2.0)  # lower violated
        # model.add_constraint('con1', lower=1.0)  # upper at boundary
        # model.add_constraint('con1', lower=-1.0)  # upper satisfied
        # model.add_constraint('con1', lower=-1.0, upper=2.0)  # lower and upper satisfied
        # model.add_constraint('con1', lower=1.5, upper=2.0)  # lower and upper - less than lower
        # model.add_constraint('con1', lower=11.5, upper=12.0)  # lower & upper - << lower
        # model.add_constraint('con1', lower=-1, upper=0.5)  # lower and upper - greater than upper
        # model.add_constraint('con1', lower=-6, upper=-5.5)  # lower and upper - >> upper

        prob.setup(check=False, mode='rev')
        prob.final_setup()
        opt_report(prob)

    def test_opt_report_scalar_equality_constraint(self):
        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', val=50.)
        model.set_input_defaults('y', val=50.)

        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        # model.add_constraint('c', equals=-15.0)  # constraint violated
        # use these when testing to see how visual looks for different situations
        model.add_constraint('c', equals=1.0)  # constraint satisfied

        prob.setup()
        prob.final_setup()
        opt_report(prob)

    def test_opt_report_array_constraint(self):
        size = 200  # how many items in the array

        class Adder(om.ExplicitComponent):
            """
            Add 10 to every item in the input vector
            """

            def __init__(self, size):
                super().__init__()
                self.size = size

            def setup(self):
                self.add_input('x', val=np.zeros(self.size, float))
                self.add_output('y', val=np.zeros(self.size, float))

            def compute(self, inputs, outputs):
                outputs['y'] = inputs['x'] + 10.

        class Summer(om.ExplicitComponent):
            """
            Aggregation component that collects all the values from the vectors and computes a total
            """

            def __init__(self, size):
                super().__init__()
                self.size = size

            def setup(self):
                self.add_input('y', val=np.zeros(self.size))
                self.add_output('sum', 0.0, shape=1)

            def compute(self, inputs, outputs):
                outputs['sum'] = np.sum(inputs['y'])

        prob = om.Problem()
        prob.driver = om.ScipyOptimizeDriver()

        prob.model.add_subsystem('des_vars', om.IndepVarComp('x', np.ones(size) * 1000),
                                 promotes=['x'])
        prob.model.add_subsystem('plus', Adder(size=size), promotes=['x', 'y'])
        prob.model.add_subsystem('summer', Summer(size=size), promotes=['y'])

        prob.model.add_design_var('x', lower=-50.0, upper=50.0)
        prob.model.add_objective('summer.sum')
        cons = []
        for i in range(size):
            cons.append((i - size / 2.) ** 2)
        prob.model.add_constraint('x', upper=np.array(cons) / 100)
        # use these when testing to see how visual looks for different situations
        # prob.model.add_constraint('x', lower=-70 + np.array(cons) / 100)
        # prob.model.add_constraint('x', lower=-70 + np.array(cons) / 100,
        #                           upper=80 + np.array(cons) / 50)

        prob.setup()

        prob['x'] = np.arange(size) - 100.0
        prob.final_setup()
        opt_report(prob)

    def test_opt_report_array_equality_constraint(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('obj', om.ExecComp('o = areas[0, 0] + areas[1, 1]',
                                               areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('widths', lower=-50.0, upper=50.0)
        model.add_objective('o')
        model.add_constraint('areas', equals=np.array([1.0, 21.0, 1.0, 17.5]))  # Some met
        # use these when testing to see how visual looks for different situations
        # model.add_constraint('areas', equals=np.array([24.0, 21.0, 3.5, 17.5]))  # none met
        # model.add_constraint('areas', equals=np.array([1.0, 1.0, 1.0, 1.0]))  # All met

        prob.setup()
        prob.final_setup()
        opt_report(prob)

    def test_opt_report_ref_arrays(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp(name="x", val=np.ones((2, ))))
        model.add_subsystem('comp', DoubleArrayComp())
        model.connect('px.x', 'comp.x1')

        model.add_design_var('px.x', ref=np.array([2.0, 3.0]), ref0=np.array([0.5, 1.5]))
        model.add_objective('comp.y1', ref=np.array([[7.0, 11.0]]), ref0=np.array([5.2, 6.3]))
        model.add_constraint('comp.y2', lower=0.0, upper=1.0, ref=np.array([[2.0, 4.0]]),
                             ref0=np.array([1.2, 2.3]))

        prob.setup()
        prob.run_driver()
        opt_report(prob)

    @require_pyoptsparse('SLSQP')
    def test_opt_report_multiple_con_alias(self):
        prob = self.prob = om.Problem(reports='optimizer')
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('widths', np.zeros((2, 2))),
                            promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('obj', om.ExecComp('o = areas[0, 0] + areas[1, 1]',
                                               areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        model.add_design_var('widths', lower=-50.0, upper=50.0)
        model.add_objective('o')

        model.add_constraint('areas', equals=24.0, indices=[0], flat_indices=True)
        model.add_constraint('areas', equals=21.0, indices=[1], flat_indices=True, alias='a2')
        model.add_constraint('areas', equals=3.5, indices=[2], flat_indices=True, alias='a3')
        model.add_constraint('areas', equals=17.5, indices=[3], flat_indices=True, alias='a4')

        prob.driver = om.pyOptSparseDriver(optimizer='SLSQP')
        prob.driver.options['print_results'] = False

        prob.setup(mode='fwd')

        prob.run_driver()

        opt_report(self.prob)
        report_file_path = str(pathlib.Path(prob.get_reports_dir()).joinpath(_default_optimizer_report_filename))

        with open(report_file_path, 'r') as f:
            for line in f.readlines():
                if 'areas' in line:
                    self.assertTrue("equality-constraint-violated" not in line)

    @hooks_active
    def test_opt_report_hook(self):
        testflo_running = os.environ.pop('TESTFLO_RUNNING', None)

        try:
            self.setup_problem_and_run_driver(om.ScipyOptimizeDriver,
                                            vars_lower=-50, vars_upper=50.,
                                            cons_lower=0, cons_upper=10.,
                                            optimizer='SLSQP',)

            # check that the report was run properly via the hook
            # and has the expected opt_result data
            self.check_opt_report()
        finally:
            if testflo_running is not None:
                os.environ['TESTFLO_RUNNING'] = testflo_running


@use_tempdirs
@unittest.skipUnless(MPI, "MPI is required.")
class TestMPIScatter(unittest.TestCase):
    N_PROCS = 2

    def test_opt_report_mpi(self):

        prob = om.Problem()
        model = prob.model

        model.set_input_defaults('x', 50.0)
        model.set_input_defaults('y', 50.0)

        from openmdao.drivers.tests.test_scipy_optimizer import DummyComp
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', DummyComp(), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-6, disp=False)

        model.add_design_var('x', lower=-50.0, upper=50.0)
        model.add_design_var('y', lower=-50.0, upper=50.0)
        model.add_objective('f_xy')
        model.add_constraint('c', lower=-15.0)

        prob.setup()
        prob.run_driver()
        opt_report(prob)

if __name__ == '__main__':
    unittest.main()
