import os
import unittest

import numpy as np

import openmdao.api as om

from openmdao.core.constants import INF_BOUND
from openmdao.core.driver import Driver
from openmdao.test_suite.components.expl_comp_array import TestExplCompArrayDense
from openmdao.test_suite.components.simple_comps import DoubleArrayComp
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped

from openmdao.utils.mpi import MPI
from openmdao.utils.testing_utils import use_tempdirs

from openmdao.visualization.opt_report.opt_report import opt_report, \
    _default_optimizer_report_filename
from openmdao.utils.general_utils import set_pyoptsparse_opt

OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

@use_tempdirs
@unittest.skipIf(tabulate is None, reason="package 'tabulate' is not installed")
class TestOptimizationReport(unittest.TestCase):

    def setup_problem_and_run_driver(self, driver,
                                     vars_lower=-INF_BOUND, vars_upper=INF_BOUND,
                                     cons_lower=-INF_BOUND, cons_upper=INF_BOUND,
                                     final_setup_only=False,
                                     optimizer=None
                                     ):
        # build a simple model that can be used for testing opt reports with different optimizers
        self.prob = prob = om.Problem()
        prob.model.add_subsystem('parab', Paraboloid(), promotes_inputs=['x', 'y'])
        prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

        prob.model.set_input_defaults('x', 3.0)
        prob.model.set_input_defaults('y', -4.0)

        prob.driver = driver()

        if optimizer:
            self.prob.driver.options['optimizer'] = optimizer

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
        prob.driver.opt_settings['ACC'] = 1e-13
        prob.set_solver_print(level=0)
        prob.model.add_constraint('con2', upper=0.0)
        prob.model.add_objective('obj')

        return prob

    def test_opt_report_check_file_created(self):
        self.setup_problem_and_run_driver(Driver,
                                          vars_lower=-50, vars_upper=50.,
                                          cons_lower=-1,
                                          )
        opt_report(self.prob)
        self.assertTrue(os.path.exists(_default_optimizer_report_filename))

    # First test all the different Drivers
    def test_opt_report_run_once_driver(self):
        self.setup_problem_and_run_driver(Driver,
                                          vars_lower=-50, vars_upper=50.,
                                          cons_lower=-1,
                                          )
        opt_report(self.prob)

    def test_opt_report_scipyopt_SLSQP(self):
        self.setup_problem_and_run_driver(om.ScipyOptimizeDriver,
                                          vars_lower=-50, vars_upper=50.,
                                          cons_lower=0, cons_upper=10.,
                                          optimizer='SLSQP',
                                          )
        opt_report(self.prob)

    def test_opt_report_scipyopt_COBYLA(self):
        self.setup_problem_and_run_driver(om.ScipyOptimizeDriver,
                                          vars_lower=-50, vars_upper=50.,
                                          cons_lower=0, cons_upper=10.,
                                          optimizer='COBYLA',
                                          )
        opt_report(self.prob)

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_opt_report_pyoptsparse_snopt(self):
        self.setup_problem_and_run_driver(om.pyOptSparseDriver,
                                          vars_lower=-50, vars_upper=50.,
                                          cons_lower=0, cons_upper=10.,
                                          optimizer='SNOPT'
                                          )
        opt_report(self.prob)

    def test_opt_report_pyoptsparse_SLSQP(self):
        self.setup_problem_and_run_driver(om.pyOptSparseDriver,
                                          vars_lower=-50, vars_upper=50.,
                                          cons_lower=0, cons_upper=10.,
                                          optimizer='SLSQP'
                                          )
        opt_report(self.prob)

    def test_opt_report_genetic_algorithm(self):
        self.setup_problem_and_run_driver(om.SimpleGADriver,
                                          vars_lower=-50, vars_upper=50.,
                                          cons_lower=0, cons_upper=10.,
                                          )
        opt_report(self.prob)

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
        opt_report(prob)

    # Next test all the different variations of variable types (scalar and array) for both
    #  desvars and constraints. Also includes commented out code that can be uncommented to
    #  force the creation of reports with different visuals for the desvars and constraints
    #  as bounds are violated and satisfied

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
