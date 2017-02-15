""" Unit tests for the Pyoptsparse Driver."""

import unittest

from openmdao.api import Problem, Group, IndepVarComp, ExecComp
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.utils.general_utils import set_pyoptsparse_opt


# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class TestPyoptSparse(unittest.TestCase):

    def setUp(self):
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is None:
            raise unittest.SkipTest("pyoptsparse is not providing SNOPT or SLSQP")

    def test_simple_paraboloid_upper(self):

        prob = Problem()
        root = prob.root = Group()

        root.add('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add('comp', Paraboloid(), promotes=['*'])
        root.add('con', ExecComp('c = - x + y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False
        prob.driver.add_desvar('x', lower=-50.0, upper=50.0)
        prob.driver.add_desvar('y', lower=-50.0, upper=50.0)

        prob.driver.add_objective('f_xy')
        prob.driver.add_constraint('c', upper=-15.0)

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

    def test_simple_paraboloid_lower(self):

        prob = Problem()
        root = prob.root = Group()

        root.add_subsystem('p1', IndepVarComp('x', 50.0), promotes=['*'])
        root.add_subsystem('p2', IndepVarComp('y', 50.0), promotes=['*'])
        root.add_subsystem('comp', Paraboloid(), promotes=['*'])
        root.add_subsystem('con', ExecComp('c = x - y'), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = OPTIMIZER
        if OPTIMIZER == 'SLSQP':
            prob.driver.opt_settings['ACC'] = 1e-9
        prob.driver.options['print_results'] = False

        root.add_design_var('x', lower=-50.0, upper=50.0)
        root.add_design_var('y', lower=-50.0, upper=50.0)

        root.add_objective('f_xy')
        root.add_constraint('c', lower=15.0)

        prob.setup(check=False)
        prob.run()

        # Minimum should be at (7.166667, -7.833334)
        assert_rel_error(self, prob['x'], 7.16667, 1e-6)
        assert_rel_error(self, prob['y'], -7.833334, 1e-6)

if __name__ == "__main__":
    unittest.main()