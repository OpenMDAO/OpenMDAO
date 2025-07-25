""" Unit tests for the Pyoptsparse Driver."""

import unittest
from packaging.version import Version

import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from openmdao.drivers.pyoptsparse_driver import pyoptsparse_version
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.general_utils import set_pyoptsparse_opt


_, snopt_opt = set_pyoptsparse_opt('SNOPT', fallback=True)


@require_pyoptsparse('IPOPT')
@unittest.skipIf(pyoptsparse_version is None or
                 pyoptsparse_version < Version('2.13.0'),
                 reason='Requires pyoptsparse 2.13.0 or later.')
@use_tempdirs
class TestComputeLagrangeMultipliers(unittest.TestCase):

    def _make_problem(self, driver, y_lower=-50, f_xy_ref=1.0, x_ref=1.0, y_ref=1.0, c_ref=1.0):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
        model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
        model.add_subsystem('comp', Paraboloid(), promotes=['*'])
        model.add_subsystem('con', om.ExecComp('c = - x + y'), promotes=['*'])

        prob.set_solver_print(level=0)

        prob.driver = driver

        model.add_design_var('x', lower=-50.0, upper=50.0, ref=x_ref)
        model.add_design_var('y', lower=y_lower, upper=50.0, ref=y_ref)
        model.add_objective('f_xy', ref=f_xy_ref)
        model.add_constraint('c', upper=-15, ref=c_ref)

        prob.setup()

        return prob

    def test_simple_paraboloid_upper_inequality_constraint_unscaled(self):
        """
        Test the computed Lagrange multipliers against IPOPT. Note IPOPT and
        SNOPT use opposite sign conventions on Lagrange multipliers.
        """
        drivers = {'pos_IPOPT': om.pyOptSparseDriver(optimizer='IPOPT', print_results=False),
                   'pos_SLSQP': om.pyOptSparseDriver(optimizer='SLSQP', print_results=False),
                   'pos_SNOPT': om.pyOptSparseDriver(optimizer=snopt_opt, print_results=False),
                   'scipy_SLSQP': om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False)}

        prob = self._make_problem(driver=drivers['pos_IPOPT'])
        result = prob.run_driver()
        self.assertEqual(result.exit_status, 'SUCCESS',
                         msg='Failed to converge baseline IPOPT optimization.')
        reference_multipliers = prob.driver.pyopt_solution.lambdaStar

        for driver_name, driver in drivers.items():
            with self.subTest(f'{driver_name=}'):
                prob = self._make_problem(driver=driver)
                result =  prob.run_driver()
                self.assertEqual(result.exit_status, 'SUCCESS',
                                 msg=f'Failed to converge optimization with {driver_name}.')
                _, active_cons = prob.driver.compute_lagrange_multipliers()
                assert_near_equal(active_cons['c']['multipliers'], reference_multipliers['c'], tolerance=1.0E-5)
                self.assertEqual(active_cons['c']['indices'], [0])
                self.assertEqual(active_cons['c']['active_bounds'], [1])

    def test_simple_paraboloid_upper_inequality_constraint_scaled(self):
        """
        Test the computed Lagrange multipliers against IPOPT. Note IPOPT and
        SNOPT use opposite sign conventions on Lagrange multipliers.
        """
        drivers = {'pos_IPOPT': om.pyOptSparseDriver(optimizer='IPOPT', print_results=False),
                   'pos_SLSQP': om.pyOptSparseDriver(optimizer='SLSQP', print_results=False),
                   'pos_SNOPT': om.pyOptSparseDriver(optimizer=snopt_opt, print_results=False),
                   'scipy_SLSQP': om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False),}

        prob = self._make_problem(driver=drivers['pos_IPOPT'], f_xy_ref=0.1, c_ref=15, x_ref=10, y_ref=10)
        result = prob.run_driver()
        self.assertEqual(result.exit_status, 'SUCCESS',
                         msg='Failed to converge baseline IPOPT optimization.')
        reference_multipliers = prob.driver.pyopt_solution.lambdaStar
        active_dvs, active_cons = prob.driver.compute_lagrange_multipliers(driver_scaling=True)
        assert_near_equal(active_cons['c']['multipliers'], reference_multipliers['c'], tolerance=1.0E-5)

        for driver_name, driver in drivers.items():
            with self.subTest(f'{driver_name=}'):
                prob = self._make_problem(driver=driver, f_xy_ref=0.1, c_ref=15, x_ref=10, y_ref=10)
                result =  prob.run_driver()
                self.assertEqual(result.exit_status, 'SUCCESS',
                                 msg=f'Failed to converge optimization with {driver_name}.')
                _, active_cons = prob.driver.compute_lagrange_multipliers(driver_scaling=True)
                assert_near_equal(active_cons['c']['multipliers'], reference_multipliers['c'], tolerance=1.0E-5)
                self.assertEqual(active_cons['c']['indices'], [0])
                self.assertEqual(active_cons['c']['active_bounds'], [1])
                # Now test the unscaled multipliers
                _, unscaled_active_cons = prob.driver.compute_lagrange_multipliers(driver_scaling=False)
                assert_near_equal(unscaled_active_cons['c']['multipliers'], 0.5, tolerance=1.0E-5)

    def test_simple_paraboloid_upper_inequality_constraint_lower_y_bound(self):
        """
        Test the computed Lagrange multipliers against IPOPT. Note IPOPT and
        SNOPT use opposite sign conventions on Lagrange multipliers.
        """
        drivers = {'pos_IPOPT': om.pyOptSparseDriver(optimizer='IPOPT', print_results=False),
                   'pos_SLSQP': om.pyOptSparseDriver(optimizer='SLSQP', print_results=False),
                   'pos_SNOPT': om.pyOptSparseDriver(optimizer=snopt_opt, print_results=False),
                   'scipy_SLSQP': om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False),}

        prob = self._make_problem(driver=drivers['pos_IPOPT'], y_lower=-7.0)
        result = prob.run_driver()
        self.assertEqual(result.exit_status, 'SUCCESS',
                         msg='Failed to converge baseline IPOPT optimization.')
        reference_multipliers = prob.driver.pyopt_solution.lambdaStar

        for driver_name, driver in drivers.items():
            with self.subTest(f'{driver_name=}'):
                prob = self._make_problem(driver=driver, y_lower=-7.0)
                result =  prob.run_driver()
                self.assertEqual(result.exit_status, 'SUCCESS',
                                 msg=f'Failed to converge optimization with {driver_name}.')
                active_dvs, active_cons = prob.driver.compute_lagrange_multipliers()
                assert_near_equal(active_cons['c']['multipliers'], reference_multipliers['c'], tolerance=1.0E-5)
                self.assertEqual(active_cons['c']['indices'], [0])
                self.assertEqual(active_cons['c']['active_bounds'], [1])

                # In this case, we also have a multiplier on the y lower bound
                self.assertEqual(active_dvs['y']['indices'], [0])
                self.assertEqual(active_dvs['y']['active_bounds'], [-1])

                # if we decrease y bound by 0.01 units, we should see the objective increase by lambda[y] * 0.01 units.
                # we use a loose tolerance when testing this due to FD across the optimization
                lambda_y = active_dvs['y']['multipliers']
                f_nom = prob.get_val('f_xy')

                prob = self._make_problem(driver=driver, y_lower=-7.01)
                result =  prob.run_driver()
                self.assertEqual(result.exit_status, 'SUCCESS',
                                 msg=f'Failed to converge optimization with {driver_name}.')
                f_perturbed = prob.get_val('f_xy')
                assert_near_equal((f_perturbed - f_nom) / 0.01, lambda_y, tolerance=1.0E-1)

    def test_simple_paraboloid_upper_inequality_constraint_lower_y_bound_scaled(self):
        """
        Test the computed Lagrange multipliers against IPOPT. Note IPOPT and
        SNOPT use opposite sign conventions on Lagrange multipliers.
        """
        drivers = {'pos_IPOPT': om.pyOptSparseDriver(optimizer='IPOPT', print_results=False),
                   'pos_SLSQP': om.pyOptSparseDriver(optimizer='SLSQP', print_results=False),
                   'pos_SNOPT': om.pyOptSparseDriver(optimizer=snopt_opt, print_results=False),
                   'scipy_SLSQP': om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False),}

        prob = self._make_problem(driver=drivers['pos_IPOPT'], y_lower=-7.0, f_xy_ref=0.1,
                                  c_ref=15, x_ref=10, y_ref=10)
        result = prob.run_driver()
        self.assertEqual(result.exit_status, 'SUCCESS',
                         msg='Failed to converge baseline IPOPT optimization.')
        reference_multipliers = prob.driver.pyopt_solution.lambdaStar

        for driver_name, driver in drivers.items():
            for use_sparse in [True, False]:
                with self.subTest(f'{driver_name=} {use_sparse=}'):
                    prob = self._make_problem(driver=driver, y_lower=-7.0, f_xy_ref=0.1,
                                            c_ref=15, x_ref=10, y_ref=10)
                    result =  prob.run_driver()
                    self.assertEqual(result.exit_status, 'SUCCESS',
                                    msg=f'Failed to converge optimization with {driver_name}.')
                    active_dvs, active_cons = prob.driver.compute_lagrange_multipliers(driver_scaling=True)
                    assert_near_equal(active_cons['c']['multipliers'], reference_multipliers['c'], tolerance=1.0E-5)
                    self.assertEqual(active_cons['c']['indices'], [0])
                    self.assertEqual(active_cons['c']['active_bounds'], [1])

                    # In this case, we also have a multiplier on the y lower bound
                    self.assertEqual(active_dvs['y']['indices'], [0])
                    self.assertEqual(active_dvs['y']['active_bounds'], [-1])

                    # if we decrease y bound by 0.01 units, we should see the objective increase by lambda[y] * 0.01 units.
                    # we use a loose tolerance when testing this due to FD across the optimization
                    active_dvs, active_cons = prob.driver.compute_lagrange_multipliers(driver_scaling=False,
                                                                                       use_sparse_solve=use_sparse)
                    lambda_y = active_dvs['y']['multipliers']
                    f_nom = prob.get_val('f_xy')

                    prob = self._make_problem(driver=driver, y_lower=-7.01)
                    result =  prob.run_driver()
                    self.assertEqual(result.exit_status, 'SUCCESS',
                                    msg=f'Failed to converge optimization with {driver_name}.')
                    f_perturbed = prob.get_val('f_xy')
                    assert_near_equal((f_perturbed - f_nom) / 0.01, lambda_y, tolerance=1.0E-1)


if __name__ == "__main__":
    unittest.main()
