""" Unit tests for the Pyoptsparse Driver."""

import unittest

import numpy as np
from packaging.version import Version
from parameterized import parameterized

import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from openmdao.drivers.pyoptsparse_driver import pyoptsparse_version
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.general_utils import set_pyoptsparse_opt


_, snopt_opt = set_pyoptsparse_opt('SNOPT', fallback=True)


@require_pyoptsparse
@unittest.skipIf(pyoptsparse_version is None or
                 pyoptsparse_version < Version('2.16.0'),
                 reason='Requires pyoptsparse 2.16.0 or later for Uno support.')
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

    @parameterized.expand([
        ('IPOPT',),
        ('Uno',),
    ])
    def test_simple_paraboloid_upper_inequality_constraint_unscaled(self, reference_optimizer):
        """
        Test the computed Lagrange multipliers against IPOPT or Uno reference.
        Note IPOPT and SNOPT use opposite sign conventions on Lagrange multipliers.
        """
        drivers = {'pos_IPOPT': om.pyOptSparseDriver(optimizer='IPOPT', print_results=False),
                   'pos_SLSQP': om.pyOptSparseDriver(optimizer='SLSQP', print_results=False),
                   'pos_SNOPT': om.pyOptSparseDriver(optimizer=snopt_opt, print_results=False),
                   'scipy_SLSQP': om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False),
                   'pos_Uno': om.pyOptSparseDriver(optimizer='Uno', print_results=False)}

        # Get reference multipliers using the parametrized optimizer
        if reference_optimizer == 'IPOPT':
            prob = self._make_problem(driver=drivers['pos_IPOPT'])
        elif reference_optimizer == 'Uno':
            prob = self._make_problem(driver=drivers['pos_Uno'])

        result = prob.run_driver()
        self.assertEqual(result.exit_status, 'SUCCESS',
                         msg=f'Failed to converge baseline {reference_optimizer} optimization.')
        reference_multipliers = prob.driver.pyopt_solution.lambdaStar

        for driver_name, driver in drivers.items():
            with self.subTest(f'reference={reference_optimizer}, tested={driver_name}'):
                prob = self._make_problem(driver=driver)
                result =  prob.run_driver()
                self.assertEqual(result.exit_status, 'SUCCESS',
                                 msg=f'Failed to converge optimization with {driver_name}.')
                _, active_cons = prob.driver.compute_lagrange_multipliers()
                assert_near_equal(active_cons['c']['multipliers'], reference_multipliers['c'], tolerance=1.0E-5)
                self.assertEqual(active_cons['c']['indices'], [0])
                self.assertEqual(active_cons['c']['active_bounds'], [1])

    @parameterized.expand([
        ('IPOPT',),
        ('Uno',),
    ])
    def test_simple_paraboloid_upper_inequality_constraint_scaled(self, reference_optimizer):
        """
        Test the computed Lagrange multipliers against IPOPT or Uno reference.
        Note IPOPT and SNOPT use opposite sign conventions on Lagrange multipliers.
        """
        drivers = {'pos_IPOPT': om.pyOptSparseDriver(optimizer='IPOPT', print_results=False),
                   'pos_SLSQP': om.pyOptSparseDriver(optimizer='SLSQP', print_results=False),
                   'pos_SNOPT': om.pyOptSparseDriver(optimizer=snopt_opt, print_results=False),
                   'scipy_SLSQP': om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False),
                   'pos_Uno': om.pyOptSparseDriver(optimizer='Uno', print_results=False)}

        # Get reference multipliers using the parametrized optimizer
        if reference_optimizer == 'IPOPT':
            prob = self._make_problem(driver=drivers['pos_IPOPT'], f_xy_ref=0.1, c_ref=15, x_ref=10, y_ref=10)
        elif reference_optimizer == 'Uno':
            prob = self._make_problem(driver=drivers['pos_Uno'], f_xy_ref=0.1, c_ref=15, x_ref=10, y_ref=10)

        result = prob.run_driver()
        self.assertEqual(result.exit_status, 'SUCCESS',
                         msg=f'Failed to converge baseline {reference_optimizer} optimization.')
        reference_multipliers = prob.driver.pyopt_solution.lambdaStar
        active_dvs, active_cons = prob.driver.compute_lagrange_multipliers(driver_scaling=True)
        assert_near_equal(active_cons['c']['multipliers'], reference_multipliers['c'], tolerance=1.0E-5)

        for driver_name, driver in drivers.items():
            with self.subTest(f'reference={reference_optimizer}, tested={driver_name}'):
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

    @parameterized.expand([
        ('IPOPT',),
        ('Uno',),
    ])
    def test_simple_paraboloid_upper_inequality_constraint_lower_y_bound(self, reference_optimizer):
        """
        Test the computed Lagrange multipliers against IPOPT or Uno reference.
        Note IPOPT and SNOPT use opposite sign conventions on Lagrange multipliers.
        """
        drivers = {'pos_IPOPT': om.pyOptSparseDriver(optimizer='IPOPT', print_results=False),
                   'pos_SLSQP': om.pyOptSparseDriver(optimizer='SLSQP', print_results=False),
                   'pos_SNOPT': om.pyOptSparseDriver(optimizer=snopt_opt, print_results=False),
                   'scipy_SLSQP': om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False),
                   'pos_Uno': om.pyOptSparseDriver(optimizer='Uno', print_results=False)}

        # Get reference multipliers using the parametrized optimizer
        if reference_optimizer == 'IPOPT':
            prob = self._make_problem(driver=drivers['pos_IPOPT'], y_lower=-7.0)
        elif reference_optimizer == 'Uno':
            prob = self._make_problem(driver=drivers['pos_Uno'], y_lower=-7.0)

        result = prob.run_driver()
        self.assertEqual(result.exit_status, 'SUCCESS',
                         msg=f'Failed to converge baseline {reference_optimizer} optimization.')
        reference_multipliers = prob.driver.pyopt_solution.lambdaStar

        for driver_name, driver in drivers.items():
            with self.subTest(f'reference={reference_optimizer}, tested={driver_name}'):
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

    @parameterized.expand([
        ('IPOPT',),
        ('Uno',),
    ])
    def test_simple_paraboloid_upper_inequality_constraint_lower_y_bound_scaled(self, reference_optimizer):
        """
        Test the computed Lagrange multipliers against IPOPT or Uno reference.
        Note IPOPT and SNOPT use opposite sign conventions on Lagrange multipliers.
        """
        drivers = {'pos_IPOPT': om.pyOptSparseDriver(optimizer='IPOPT', print_results=False),
                   'pos_SLSQP': om.pyOptSparseDriver(optimizer='SLSQP', print_results=False),
                   'pos_SNOPT': om.pyOptSparseDriver(optimizer=snopt_opt, print_results=False),
                   'scipy_SLSQP': om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False),
                   'pos_Uno': om.pyOptSparseDriver(optimizer='Uno', print_results=False)}

        # Get reference multipliers using the parametrized optimizer
        if reference_optimizer == 'IPOPT':
            prob = self._make_problem(driver=drivers['pos_IPOPT'], y_lower=-7.0, f_xy_ref=0.1,
                                      c_ref=15, x_ref=10, y_ref=10)
        elif reference_optimizer == 'Uno':
            prob = self._make_problem(driver=drivers['pos_Uno'], y_lower=-7.0, f_xy_ref=0.1,
                                      c_ref=15, x_ref=10, y_ref=10)

        result = prob.run_driver()
        self.assertEqual(result.exit_status, 'SUCCESS',
                         msg=f'Failed to converge baseline {reference_optimizer} optimization.')
        reference_multipliers = prob.driver.pyopt_solution.lambdaStar

        for driver_name, driver in drivers.items():
            for use_sparse in [True, False]:
                with self.subTest(f'reference={reference_optimizer}, tested={driver_name} {use_sparse=}'):
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


class _QuadraticComp(om.ExplicitComponent):
    """
    f = 1/2 x^T x - c^T x and g = A x.
    """

    def initialize(self):
        self.options.declare('A')
        self.options.declare('c')

    def setup(self):
        A = self.options['A']
        m, n = A.shape
        self.add_input('x', shape=n)
        self.add_output('f', val=0.0)
        self.add_output('g', shape=m)
        self.declare_partials('f', 'x')
        self.declare_partials('g', 'x', val=A)

    def compute(self, inputs, outputs):
        x = inputs['x']
        outputs['f'] = 0.5 * x @ x - self.options['c'] @ x
        outputs['g'] = self.options['A'] @ x

    def compute_partials(self, inputs, partials):
        partials['f', 'x'] = inputs['x'] - self.options['c']


class TestComputeLagrangeMultipliersAccuracy(unittest.TestCase):
    """
    Compare the multipliers with exact values on problems built from a known solution.

    The problem is minimize 1/2 x^T x - c^T x subject to A x = b. Its data are built from a
    chosen solution and chosen multipliers, and the model is evaluated at that solution, so
    no optimizer is needed.
    """

    def _make_problem(self, m=30, n=40, cond=1.0E3, dependent=False):
        rng = np.random.default_rng(1)
        U, _ = np.linalg.qr(rng.standard_normal((m, m)))
        V, _ = np.linalg.qr(rng.standard_normal((n, n)))
        # The singular values of the constraint jacobian are spread between 1 and 1 / cond.
        A = U @ np.diag(np.geomspace(1.0, 1.0 / cond, m)) @ V[:, :m].T
        if dependent:
            A = np.vstack((A, A[3] + A[7]))
        x_opt = rng.standard_normal(n)
        lam = rng.uniform(1.0, 2.0, A.shape[0])
        # Stationarity of the Lagrangian: (x - c) + A^T lam = 0
        c = x_opt + A.T @ lam

        prob = om.Problem()
        prob.model.add_subsystem('comp', _QuadraticComp(A=A, c=c), promotes=['*'])
        prob.model.add_design_var('x')
        prob.model.add_objective('f')
        prob.model.add_constraint('g', equals=A @ x_opt)
        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP')
        prob.setup()
        prob.set_val('x', x_opt)
        prob.run_model()

        return prob, A, lam

    def test_multipliers_are_exact(self):
        for cond in (1.0E1, 1.0E3, 1.0E5):
            for use_sparse in (True, False):
                with self.subTest(cond=cond, use_sparse=use_sparse):
                    prob, _, lam = self._make_problem(cond=cond)
                    _, active_cons = prob.driver.compute_lagrange_multipliers(
                        use_sparse_solve=use_sparse)
                    assert_near_equal(np.ravel(active_cons['g']['multipliers']), lam,
                                      tolerance=1.0E-8)

    def test_dependent_constraints(self):
        # One constraint is the sum of two others, so the multipliers are not unique. Both
        # solves return the solution of least norm.
        prob, A, lam = self._make_problem(cond=1.0E2, dependent=True)
        expected = np.linalg.lstsq(A.T, A.T @ lam, rcond=None)[0]

        for use_sparse in (True, False):
            with self.subTest(use_sparse=use_sparse):
                _, active_cons = prob.driver.compute_lagrange_multipliers(
                    use_sparse_solve=use_sparse)
                mult = np.ravel(active_cons['g']['multipliers'])
                assert_near_equal(mult, expected, tolerance=1.0E-8)
                # The multipliers satisfy the stationarity condition.
                assert_near_equal(A.T @ mult, A.T @ lam, tolerance=1.0E-10)

    def test_sparse_lstsq(self):
        import scipy.sparse as sp
        from openmdao.core.driver import _sparse_lstsq

        rng = np.random.default_rng(3)

        # Full column rank, with an inconsistent right-hand side
        mat = rng.standard_normal((25, 12))
        rhs = rng.standard_normal(25)
        assert_near_equal(_sparse_lstsq(sp.csr_matrix(mat), rhs),
                          np.linalg.lstsq(mat, rhs, rcond=None)[0], tolerance=1.0E-10)

        # Rank deficient
        mat[:, 5] = mat[:, 2] - mat[:, 9]
        assert_near_equal(_sparse_lstsq(sp.csr_matrix(mat), rhs),
                          np.linalg.lstsq(mat, rhs, rcond=None)[0], tolerance=1.0E-8)

        # More unknowns than equations
        mat = rng.standard_normal((6, 10))
        rhs = rng.standard_normal(6)
        assert_near_equal(_sparse_lstsq(sp.csr_matrix(mat), rhs),
                          np.linalg.lstsq(mat, rhs, rcond=None)[0], tolerance=1.0E-8)

        # Zero right-hand side
        assert_near_equal(_sparse_lstsq(sp.csr_matrix(mat), np.zeros(6)), np.zeros(10))


if __name__ == "__main__":
    unittest.main()
