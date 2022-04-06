import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal

from openmdao.test_suite.components.paraboloid import Paraboloid

from openmdao.visualization.opt_report.opt_report import opt_report


# @use_tempdirs
class TestOptimizationReport(unittest.TestCase):

    def setup_problem(self, optimizer):
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

        prob.model.add_design_var('x', lower=-50, upper=50)
        prob.model.add_design_var('y', lower=-50, upper=50)
        prob.model.add_objective('parab.f_xy')

        # to add the constraint to the model
        prob.model.add_constraint('const.g', lower=0, upper=10., alias='ALIAS_TEST')

        prob.setup()

        prob.run_driver()
        return prob

    def test_opt_report_scipyopt(self):
        prob = self.setup_problem(om.ScipyOptimizeDriver)
        prob.driver.options['optimizer'] = 'SLSQP'
        opt_report(self.prob)

    def test_opt_report_pyoptsparse(self):
        prob = self.setup_problem(om.pyOptSparseDriver)
        # prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['optimizer'] = 'SNOPT'

        opt_report(self.prob)

    def test_opt_report_genetic_algorithm(self):
        prob = self.setup_problem(om.SimpleGADriver)
        opt_report(self.prob)

    def test_opt_report_differential_evolution(self):
        prob = self.setup_problem(om.DifferentialEvolutionDriver)
        opt_report(self.prob)

    def test_exception_handling(self):
        pass

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
