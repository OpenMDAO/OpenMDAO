import unittest

import numpy as np
import openmdao.api as om
from openmdao.test_suite.components.expl_comp_array import TestExplCompArrayDense
from openmdao.test_suite.components.simple_comps import DoubleArrayComp
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal

from openmdao.test_suite.components.paraboloid import Paraboloid

from openmdao.visualization.opt_report.opt_report import opt_report
from openmdao.core.constants import INF_BOUND

# @use_tempdirs
class TestOptimizationReport(unittest.TestCase):

    def setup_problem_and_run_driver(self, optimizer,
                      vars_lower=-INF_BOUND, vars_upper=INF_BOUND,
                      cons_lower=-INF_BOUND, cons_upper=INF_BOUND
                      ):
        # build the model
        self.prob = prob = om.Problem()
        prob.model.add_subsystem('parab', Paraboloid(), promotes_inputs=['x', 'y'])

        # define the component whose output will be constrained
        prob.model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])

        # Design variables 'x' and 'y' span components, so we need to provide a common initial
        # value for them.
        prob.model.set_input_defaults('x', 3.0)
        prob.model.set_input_defaults('y', -4.0)

        # setup the optimization
        prob.driver = optimizer()

        prob.model.add_design_var('x', lower=vars_lower, upper=vars_upper)
        prob.model.add_design_var('y', lower=vars_lower, upper=vars_upper)
        prob.model.add_objective('parab.f_xy')

        # to add the constraint to the model
        prob.model.add_constraint('const.g', lower=cons_lower, upper=cons_upper, alias='ALIAS_TEST')

        prob.setup()

        prob.run_driver()
        # return prob


    # def test_opt_report_run_once_driver(self):
    #     prob = om.Problem()
    #     prob.model = model = SellarDerivatives()
    #
    #     model.add_design_var('z')
    #     model.add_objective('obj')
    #     model.add_constraint('con1', lower=0)
    #     prob.set_solver_print(level=0)
    #
    #     prob.setup()
    #     prob.run_driver()


    def test_opt_report_run_once_driver(self):
        # example problem where the optimum is at a constraint
        from openmdao.core.driver import Driver

        self.setup_problem_and_run_driver(Driver,
                                  vars_lower=-50, vars_upper=50.,
                                  cons_lower=0, cons_upper=10.,
                                  )
        opt_report(self.prob)


    def test_vector_scaled_derivs(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('px', om.IndepVarComp(name="x", val=np.ones((2, ))))
        comp = model.add_subsystem('comp', DoubleArrayComp())
        model.connect('px.x', 'comp.x1')

        model.add_design_var('px.x', ref=np.array([2.0, 3.0]), ref0=np.array([0.5, 1.5]))
        model.add_objective('comp.y1', ref=np.array([[7.0, 11.0]]), ref0=np.array([5.2, 6.3]))
        model.add_constraint('comp.y2', lower=0.0, upper=1.0, ref=np.array([[2.0, 4.0]]), ref0=np.array([1.2, 2.3]))

        prob.setup()
        prob.run_driver()
        opt_report(prob)

    def test_simple_array_comp2D_eq_con(self):

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('widths', np.zeros((2, 2))), promotes=['*'])
        model.add_subsystem('comp', TestExplCompArrayDense(), promotes=['*'])
        model.add_subsystem('obj', om.ExecComp('o = areas[0, 0] + areas[1, 1]', areas=np.zeros((2, 2))),
                            promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

        model.add_design_var('widths', lower=-50.0, upper=50.0)
        # model.add_design_var('widths', lower=-50.0 * np.ones((2, 2)), upper=50.0 * np.ones((2, 2)))
        model.add_objective('o')
        model.add_constraint('areas', equals=np.array([24.0, 21.0, 3.5, 17.5]))

        prob.setup()

        failed = prob.run_driver()
        opt_report(prob)



    def test_opt_report_pyoptsparse_compare_outputs(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.test_suite.components.sellar import SellarDerivativesGrouped

        prob = om.Problem()
        model = prob.model = SellarDerivativesGrouped()

        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = "SLSQP"
        prob.driver.opt_settings['ACC'] = 1e-13

        model.add_design_var('z', lower=np.array([-10.0, 0.0]), upper=np.array([10.0, 10.0]))
        model.add_design_var('x', lower=0.0, upper=10.0)
        model.add_objective('obj')
        model.add_constraint('con1', upper=0.0)
        model.add_constraint('con2', upper=0.0)

        prob.set_solver_print(level=0)

        prob.setup(check=False, mode='rev')
        prob.run_driver()
        opt_report(prob)

    def test_opt_report_scipyopt_cons_bound(self):
        # example problem where the optimum is at a constraint
        self.setup_problem_and_run_driver(om.ScipyOptimizeDriver,
                                  vars_lower=-50, vars_upper=50.,
                                  cons_lower=0, cons_upper=10.,
                                  )
        self.prob.driver.options['optimizer'] = 'SLSQP'
        opt_report(self.prob)

    def test_opt_report_scipyopt_var_bound(self):
        # example problem where the optimum is not at a constraint
        self.setup_problem_and_run_driver(om.ScipyOptimizeDriver,
                                  vars_lower=0, vars_upper=10.,
                                  cons_lower=0, cons_upper=10.,
                                  )
        self.prob.driver.options['optimizer'] = 'SLSQP'
        opt_report(self.prob)

    def test_opt_report_scipyopt_only_var_lower_bound(self):
        # example problem where the there are only lower bounds
        self.setup_problem_and_run_driver(om.ScipyOptimizeDriver,
                                  vars_lower=0,
                                  cons_lower=0, cons_upper=10.,
                                  )
        self.prob.driver.options['optimizer'] = 'SLSQP'
        opt_report(self.prob)

    def test_opt_report_scipyopt_equality_cons_run_driver(self):
        # example problem where the there is an equality constraint
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
        model.add_constraint('c', equals=-15.0)

        prob.setup()
        failed = prob.run_driver()

        opt_report(prob)

    def test_opt_report_scipyopt_equality_cons_no_run_driver(self):
        # testing report when run driver is not run
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
        model.add_constraint('c', equals=-15.0)

        prob.setup()
        prob.final_setup()

        opt_report(prob)









    def test_opt_report_pyoptsparse_snopt(self):
        self.setup_problem_and_run_driver(om.pyOptSparseDriver,
                                  vars_lower=-50, vars_upper=50.,
                                  cons_lower=0, cons_upper=10.,
                                  )
        self.prob.driver.options['optimizer'] = 'SNOPT'

        opt_report(self.prob)

    def test_opt_report_pyoptsparse_SLSQP(self):
        self.setup_problem_and_run_driver(om.pyOptSparseDriver,
                                  vars_lower=-50, vars_upper=50.,
                                  cons_lower=0, cons_upper=10.,
                                  )
        self.prob.driver.options['optimizer'] = 'SLSQP'

        opt_report(self.prob)

    # def test_opt_report_genetic_algorithm_bombs(self):
    #     self.setup_problem_and_run_driver(om.SimpleGADriver)
    #     opt_report(self.prob)
    #  Told Ken about the bug and put in a story

    def test_opt_report_genetic_algorithm(self):
        self.setup_problem_and_run_driver(om.SimpleGADriver,
                                  vars_lower=-50, vars_upper=50.,
                                  cons_lower=0, cons_upper=10.,
                                  )
        opt_report(self.prob)

    def test_opt_report_differential_evolution(self):
        prob = om.Problem()

        exec = om.ExecComp(['y = x**2',
                            'z = a + x**2'],
                            a={'shape': (1,)},
                            y={'shape': (101,)},
                            x={'shape': (101,)},
                            z={'shape': (101,)})

        prob.model.add_subsystem('exec', exec)

        prob.model.add_design_var('exec.a', lower=-1000, upper=1000)
        prob.model.add_objective('exec.y', index=50)
        prob.model.add_constraint('exec.z', indices=[-1], lower=0)
        prob.model.add_constraint('exec.z', indices=[0], upper=300, alias="ALIAS_TEST")

        prob.driver = om.DifferentialEvolutionDriver()

        prob.setup()

        prob.set_val('exec.x', np.linspace(-10, 10, 101))

        prob.run_driver()
        opt_report(prob)

    def test_exception_handling(self):
        pass


    def test_opt_report_array_vars(self):
        size = 100  # how many items in the array

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

        prob.model.add_subsystem('des_vars', om.IndepVarComp('x', np.ones(size) * 1000), promotes=['x'])
        prob.model.add_subsystem('plus', Adder(size=size), promotes=['x', 'y'])
        # prob.model.add_subsystem('summer', Summer(size=size), promotes_outputs=['sum'])
        prob.model.add_subsystem('summer', Summer(size=size), promotes=['y'])
        # prob.model.promotes('summer', inputs=[('x', 'y')])

        prob.model.add_design_var('x', lower=-50.0, upper=50.0)
        prob.model.add_objective('summer.sum')
        cons = []
        for i in range(size):
            cons.append( ( i - size /2.) ** 2)

        # prob.model.add_constraint('x', upper=-15.0)
        prob.model.add_constraint('x', lower=np.array(cons))

        prob.driver = om.ScipyOptimizeDriver()

        prob.setup()

        prob['x'] = np.arange(size)

        prob.final_setup()
        # prob.run_driver()

        opt_report(prob)


from openmdao.utils.mpi import MPI

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
