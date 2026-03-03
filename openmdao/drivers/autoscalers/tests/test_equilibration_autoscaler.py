"""Tests for EquilibrationAutoscaler."""
import io
import contextlib
import unittest
import warnings

import numpy as np

import openmdao.api as om
from openmdao.drivers.autoscalers import EquilibrationAutoscaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dv_meta(size, lower=None, upper=None, total_scaler=None, total_adder=None):
    """Return a minimal design variable metadata dict."""
    return {
        'size': size, 'global_size': size,
        'lower': lower, 'upper': upper,
        'total_scaler': total_scaler, 'total_adder': total_adder,
        'discrete': False, 'distributed': False,
    }


def _out_meta(size, total_scaler=None, total_adder=None):
    """Return a minimal constraint or objective metadata dict."""
    return {
        'size': size, 'global_size': size,
        'total_scaler': total_scaler, 'total_adder': total_adder,
        'discrete': False, 'distributed': False,
    }


class _MockProblem:
    def __init__(self, jac):
        self._jac = jac

    def compute_totals(self, of, wrt, return_format='flat_dict', driver_scaling=False):
        return self._jac


class _MockDriver:
    def __init__(self, dvs, cons, objs, jac=None):
        self._designvars = dvs
        self._cons = cons
        self._objs = objs
        self._jac = jac or {}

    def _problem(self):
        return _MockProblem(self._jac)


def _configure(autoscaler, driver):
    """Call configure while suppressing the condition number printout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        autoscaler.configure(driver)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Init and property tests
# ---------------------------------------------------------------------------

class TestEquilibrationAutoscalerInit(unittest.TestCase):
    """Test constructor parameters and configure_requires_run_model."""

    def test_default_params(self):
        """Defaults: norm='inf', max_iter=20, tol=1e-3, block_mode=True, override_scaling=True."""
        a = EquilibrationAutoscaler()
        self.assertEqual(a._norm, 'inf')
        self.assertEqual(a._max_iter, 20)
        self.assertAlmostEqual(a._tol, 1e-3)
        self.assertTrue(a._block_mode)
        self.assertTrue(a._override_scaling)

    def test_custom_params(self):
        """Custom params are stored correctly."""
        a = EquilibrationAutoscaler(norm='fro', max_iter=10, tol=1e-6,
                                    block_mode=False, override_scaling=False)
        self.assertEqual(a._norm, 'fro')
        self.assertEqual(a._max_iter, 10)
        self.assertAlmostEqual(a._tol, 1e-6)
        self.assertFalse(a._block_mode)
        self.assertFalse(a._override_scaling)

    def test_invalid_norm_raises(self):
        """Unsupported norm raises ValueError at construction."""
        with self.assertRaises(ValueError) as ctx:
            EquilibrationAutoscaler(norm='l2')
        self.assertIn('l2', str(ctx.exception))

    def test_configure_requires_run_model(self):
        """configure_requires_run_model is always True."""
        self.assertTrue(EquilibrationAutoscaler().configure_requires_run_model)
        self.assertTrue(
            EquilibrationAutoscaler(norm='fro', block_mode=False).configure_requires_run_model
        )

    def test_original_dicts_initialised_empty(self):
        """_original_scalers and _original_adders start as empty dicts."""
        a = EquilibrationAutoscaler()
        self.assertEqual(a._original_scalers, {})
        self.assertEqual(a._original_adders, {})


# ---------------------------------------------------------------------------
# setup() tests
# ---------------------------------------------------------------------------

class TestEquilibrationAutoscalerSetup(unittest.TestCase):
    """Test that setup() saves original scalers and adders correctly."""

    def test_setup_saves_original_scalers_and_adders(self):
        """setup() stores a snapshot of total_scaler/total_adder for all VOI types."""
        dvs = {'x': _dv_meta(1, total_scaler=2.0, total_adder=1.0)}
        cons = {'c': _out_meta(1, total_scaler=0.5, total_adder=0.1)}
        objs = {'obj': _out_meta(1, total_scaler=3.0)}
        driver = _MockDriver(dvs, cons, objs)

        a = EquilibrationAutoscaler()
        a.setup(driver)

        self.assertEqual(a._original_scalers['design_var']['x'], 2.0)
        self.assertEqual(a._original_adders['design_var']['x'], 1.0)
        self.assertEqual(a._original_scalers['constraint']['c'], 0.5)
        self.assertEqual(a._original_adders['constraint']['c'], 0.1)
        self.assertEqual(a._original_scalers['objective']['obj'], 3.0)

    def test_setup_saves_none_scalers(self):
        """setup() preserves None as the original scaler when none was set."""
        dvs = {'x': _dv_meta(1)}
        driver = _MockDriver(dvs, {}, {'obj': _out_meta(1)})

        a = EquilibrationAutoscaler()
        a.setup(driver)

        self.assertIsNone(a._original_scalers['design_var']['x'])
        self.assertIsNone(a._original_adders['design_var']['x'])


# ---------------------------------------------------------------------------
# _compute_row_factors / _compute_col_factors tests
# ---------------------------------------------------------------------------

class TestEquilibrationRowColFactors(unittest.TestCase):
    """Analytically verify row/column scaling factor computation for both norms."""

    def test_inf_norm_row_factors(self):
        """norm='inf': r_i = 1 / sqrt(max_j |J_ij|)."""
        a = EquilibrationAutoscaler(norm='inf')
        # Row 0 max = 4, row 1 max = 9
        J = np.array([[4.0, 0.0], [0.0, 9.0]])
        r = a._compute_row_factors(J)
        np.testing.assert_allclose(r, [1.0 / np.sqrt(4.0), 1.0 / np.sqrt(9.0)])

    def test_fro_norm_row_factors(self):
        """norm='fro': r_i = 1 / sqrt(||row_i||_2) where ||row||_2 = sqrt(sum J_ij^2)."""
        a = EquilibrationAutoscaler(norm='fro')
        # Row 0: ||row||_2 = sqrt(3^2 + 4^2) = 5 → r_0 = 1/sqrt(5)
        J = np.array([[3.0, 4.0]])
        r = a._compute_row_factors(J)
        np.testing.assert_allclose(r, [1.0 / np.sqrt(5.0)])

    def test_inf_norm_col_factors(self):
        """norm='inf': c_j = 1 / sqrt(max_i |J_ij|)."""
        a = EquilibrationAutoscaler(norm='inf')
        J = np.array([[4.0, 0.0], [0.0, 9.0]])
        c = a._compute_col_factors(J)
        np.testing.assert_allclose(c, [1.0 / np.sqrt(4.0), 1.0 / np.sqrt(9.0)])

    def test_fro_norm_col_factors(self):
        """norm='fro': c_j = 1 / sqrt(||col_j||_2) where ||col||_2 = sqrt(sum J_ij^2)."""
        a = EquilibrationAutoscaler(norm='fro')
        # Col 0: ||col||_2 = sqrt(3^2 + 4^2) = 5 → c_0 = 1/sqrt(5)
        J = np.array([[3.0], [4.0]])
        c = a._compute_col_factors(J)
        np.testing.assert_allclose(c, [1.0 / np.sqrt(5.0)])

    def test_zero_row_warns_and_returns_unit_factor(self):
        """A zero row issues a RuntimeWarning and produces r_i = 1.0."""
        a = EquilibrationAutoscaler(norm='inf')
        J = np.array([[0.0, 0.0], [1.0, 2.0]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            r = a._compute_row_factors(J)
        self.assertAlmostEqual(r[0], 1.0)        # fallback
        self.assertAlmostEqual(r[1], 1.0 / np.sqrt(2.0))
        rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(rw), 1)

    def test_zero_col_warns_and_returns_unit_factor(self):
        """A zero column issues a RuntimeWarning and produces c_j = 1.0."""
        a = EquilibrationAutoscaler(norm='inf')
        J = np.array([[0.0, 4.0], [0.0, 1.0]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            c = a._compute_col_factors(J)
        self.assertAlmostEqual(c[0], 1.0)        # fallback for zero column
        self.assertAlmostEqual(c[1], 1.0 / np.sqrt(4.0))
        rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(rw), 1)

    def test_multiple_zero_rows_count_reported(self):
        """Warning message correctly counts multiple zero rows."""
        a = EquilibrationAutoscaler(norm='fro')
        J = np.array([[0.0], [0.0], [1.0]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            a._compute_row_factors(J)
        rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(rw), 1)
        self.assertIn('2', str(rw[0].message))


# ---------------------------------------------------------------------------
# _equilibrate tests
# ---------------------------------------------------------------------------

class TestEquilibrationEquilibrate(unittest.TestCase):
    """Test the iterative equilibration loop."""

    def test_identity_converges_in_one_iteration(self):
        """Identity matrix has row/col norms already = 1 → breaks on first check."""
        a = EquilibrationAutoscaler(norm='inf', tol=1e-6)
        J = np.eye(3)
        total_row, total_col, n_iter = a._equilibrate(J)
        # All r_i = c_j = 1 on first iter → r - 1 = 0 < tol → break
        self.assertEqual(n_iter, 1)
        np.testing.assert_allclose(total_row, np.ones(3))
        np.testing.assert_allclose(total_col, np.ones(3))
        # J unchanged
        np.testing.assert_allclose(J, np.eye(3))

    def test_diagonal_converges_in_two_iterations(self):
        """Diagonal matrix converges in exactly 2 iterations (scale, then verify=1)."""
        # J = diag([4, 9]):
        # iter 1: r=[0.5, 1/3], c=[0.5, 1/3], J→I. r-1≠0, continue.
        # iter 2: r=[1, 1], c=[1, 1]. r-1=0 < tol, break.
        a = EquilibrationAutoscaler(norm='inf', tol=1e-6)
        J = np.diag([4.0, 9.0])
        total_row, total_col, n_iter = a._equilibrate(J)
        self.assertEqual(n_iter, 2)
        np.testing.assert_allclose(total_row, [0.5, 1.0 / 3.0])
        np.testing.assert_allclose(total_col, [0.5, 1.0 / 3.0])
        # Equilibrated J should be identity
        np.testing.assert_allclose(J, np.eye(2), atol=1e-14)

    def test_accumulated_scalers_correct_for_diagonal(self):
        """total_row * total_col product recovers original diagonal."""
        a = EquilibrationAutoscaler(norm='inf', tol=1e-8)
        d = np.array([16.0, 0.0625])  # [16, 1/16]
        J = np.diag(d)
        total_row, total_col, _ = a._equilibrate(J)
        # total_row[i] * total_col[i] = 1/d[i] (for diagonal case)
        recovered = 1.0 / (total_row * total_col)
        np.testing.assert_allclose(recovered, d, rtol=1e-6)

    def test_non_convergence_warns(self):
        """Failing to converge in max_iter issues a RuntimeWarning."""
        # Use max_iter=1 and a non-trivial off-diagonal matrix that won't converge
        a = EquilibrationAutoscaler(norm='fro', max_iter=1, tol=1e-12)
        J = np.array([[2.0, 1.0], [1.0, 3.0]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            total_row, total_col, n_iter = a._equilibrate(J)
        self.assertEqual(n_iter, 1)
        rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(rw), 1)
        self.assertIn('converge', str(rw[0].message).lower())

    def test_fro_norm_converges(self):
        """Frobenius norm variant converges on a simple diagonal case."""
        a = EquilibrationAutoscaler(norm='fro', tol=1e-6)
        J = np.diag([4.0, 9.0])
        total_row, total_col, n_iter = a._equilibrate(J)
        self.assertGreater(n_iter, 0)
        self.assertLessEqual(n_iter, a._max_iter)
        # After convergence, scaled J should have row/col Frobenius norms ≈ 1
        for i in range(J.shape[0]):
            np.testing.assert_allclose(np.sqrt(np.sum(J[i, :] ** 2)), 1.0, atol=1e-5)
        for j in range(J.shape[1]):
            np.testing.assert_allclose(np.sqrt(np.sum(J[:, j] ** 2)), 1.0, atol=1e-5)

    def test_j_modified_in_place(self):
        """_equilibrate modifies J in place."""
        a = EquilibrationAutoscaler(norm='inf')
        J = np.diag([4.0, 9.0])
        J_before = J.copy()
        a._equilibrate(J)
        # J should have changed from its original values
        self.assertFalse(np.allclose(J, J_before))


# ---------------------------------------------------------------------------
# _collapse tests
# ---------------------------------------------------------------------------

class TestEquilibrationCollapse(unittest.TestCase):
    """Test geometric-mean collapse in block mode and passthrough in per-element mode."""

    def test_block_mode_geometric_mean(self):
        """Block mode returns geometric mean: exp(mean(log(arr)))."""
        a = EquilibrationAutoscaler(block_mode=True)
        # geometric mean of [2, 8] = sqrt(16) = 4.0
        result = a._collapse(np.array([2.0, 8.0]))
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 4.0)

    def test_block_mode_singleton(self):
        """Geometric mean of a single element returns that element."""
        a = EquilibrationAutoscaler(block_mode=True)
        result = a._collapse(np.array([7.5]))
        self.assertAlmostEqual(result, 7.5)

    def test_block_mode_uniform_array(self):
        """Geometric mean of identical elements returns that element."""
        a = EquilibrationAutoscaler(block_mode=True)
        result = a._collapse(np.array([3.0, 3.0, 3.0]))
        self.assertAlmostEqual(result, 3.0)

    def test_per_element_mode_passthrough(self):
        """Per-element mode returns a copy of the input array."""
        a = EquilibrationAutoscaler(block_mode=False)
        arr = np.array([0.5, 2.0, 0.25])
        result = a._collapse(arr)
        np.testing.assert_allclose(result, arr)
        # Verify it is a copy, not the same object
        self.assertIsNot(result, arr)

    def test_per_element_mode_returns_ndarray(self):
        """Per-element collapse returns an ndarray."""
        a = EquilibrationAutoscaler(block_mode=False)
        result = a._collapse(np.array([1.0, 2.0]))
        self.assertIsInstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# _apply_scalers tests
# ---------------------------------------------------------------------------

class TestEquilibrationApplyScalers(unittest.TestCase):
    """Test that _apply_scalers writes to metadata correctly in both modes."""

    def _make_autoscaler_with_originals(self, voi_type, name, orig_scaler, orig_adder,
                                        override_scaling):
        a = EquilibrationAutoscaler(override_scaling=override_scaling)
        a._original_scalers = {voi_type: {name: orig_scaler}}
        a._original_adders = {voi_type: {name: orig_adder}}
        return a

    def test_override_replaces_scaler_and_adder(self):
        """override_scaling=True: both total_scaler and total_adder are replaced."""
        a = self._make_autoscaler_with_originals('constraint', 'c', 5.0, 2.0,
                                                  override_scaling=True)
        meta = {'c': {'total_scaler': 5.0, 'total_adder': 2.0, 'discrete': False}}
        a._apply_scalers('constraint', meta, {'c': 0.25}, {'c': None})
        self.assertAlmostEqual(meta['c']['total_scaler'], 0.25)
        self.assertIsNone(meta['c']['total_adder'])

    def test_override_sets_array_scaler(self):
        """override_scaling=True with ndarray scaler is written directly."""
        a = self._make_autoscaler_with_originals('constraint', 'c', None, None,
                                                  override_scaling=True)
        meta = {'c': {'total_scaler': None, 'total_adder': None, 'discrete': False}}
        eq_scaler = np.array([0.1, 0.2])
        a._apply_scalers('constraint', meta, {'c': eq_scaler}, {'c': None})
        np.testing.assert_allclose(meta['c']['total_scaler'], [0.1, 0.2])

    def test_compose_multiplies_scaler_keeps_adder(self):
        """override_scaling=False: total_scaler = eq * orig; total_adder unchanged."""
        a = self._make_autoscaler_with_originals('constraint', 'c', 4.0, 0.5,
                                                  override_scaling=False)
        meta = {'c': {'total_scaler': 4.0, 'total_adder': 0.5, 'discrete': False}}
        a._apply_scalers('constraint', meta, {'c': 0.5}, {'c': None})
        self.assertAlmostEqual(meta['c']['total_scaler'], 2.0)   # 0.5 * 4.0
        self.assertAlmostEqual(meta['c']['total_adder'], 0.5)    # unchanged

    def test_compose_with_none_original_uses_eq_scaler(self):
        """Compose mode with None original scaler just uses the equilibration scaler."""
        a = self._make_autoscaler_with_originals('constraint', 'c', None, None,
                                                  override_scaling=False)
        meta = {'c': {'total_scaler': None, 'total_adder': None, 'discrete': False}}
        a._apply_scalers('constraint', meta, {'c': 0.25}, {'c': None})
        self.assertAlmostEqual(meta['c']['total_scaler'], 0.25)

    def test_discrete_variable_is_skipped(self):
        """Discrete variables are not touched."""
        a = self._make_autoscaler_with_originals('constraint', 'c', 1.0, 0.0,
                                                  override_scaling=True)
        meta = {'c': {'total_scaler': 1.0, 'total_adder': 0.0, 'discrete': True}}
        a._apply_scalers('constraint', meta, {'c': 99.0}, {'c': 99.0})
        self.assertAlmostEqual(meta['c']['total_scaler'], 1.0)  # unchanged


# ---------------------------------------------------------------------------
# configure() integration tests
# ---------------------------------------------------------------------------

class TestEquilibrationConfigureIntegration(unittest.TestCase):
    """End-to-end configure() tests using a MockDriver with known Jacobians."""

    def test_configure_unit_jac_scalers_are_one(self):
        """When every J entry is 1, all row/col norms are already 1 → scalers = 1.0."""
        # J assembled = [[1], [1]] (2x1): rows=[1], [1]; col=[max(1,1)=1]
        # All r,c = 1 → converge in iter 1 → all scalers = 1.0
        dvs = {'x': _dv_meta(1)}
        cons = {'c': _out_meta(1)}
        objs = {'obj': _out_meta(1)}
        jac = {('c', 'x'): np.array([[1.0]]), ('obj', 'x'): np.array([[1.0]])}
        driver = _MockDriver(dvs, cons, objs, jac=jac)

        a = EquilibrationAutoscaler(norm='inf')
        a.setup(driver)
        _configure(a, driver)

        self.assertAlmostEqual(dvs['x']['total_scaler'], 1.0)
        self.assertAlmostEqual(cons['c']['total_scaler'], 1.0)
        self.assertAlmostEqual(objs['obj']['total_scaler'], 1.0)

    def test_configure_diagonal_jac_exact_scalers(self):
        """Diagonal 1x1 Jacobian: verify exact equilibration scalers analytically.

        J = [[4.0]], norm='inf':
          iter 1: r=[0.5], c=[0.5], J=[[1]]. max|r-1|=0.5 > tol.
          iter 2: r=[1],   c=[1].   max|r-1|=0 < tol. Break.
          total_row=[0.5], total_col=[0.5].
          dv_scaler = 0.5, con_scaler = 0.5.
        """
        dvs = {'x': _dv_meta(1)}
        cons = {'c': _out_meta(1)}
        jac = {('c', 'x'): np.array([[4.0]])}
        driver = _MockDriver(dvs, cons, {}, jac=jac)

        a = EquilibrationAutoscaler(norm='inf', tol=1e-6)
        a.setup(driver)
        _configure(a, driver)

        self.assertAlmostEqual(dvs['x']['total_scaler'], 0.5, places=8)
        self.assertAlmostEqual(cons['c']['total_scaler'], 0.5, places=8)

    def test_configure_2x2_diagonal_jac(self):
        """Two separate 1x1 constraint-DV pairs: each gets its own independent scaler.

        J assembled = [[4, 0], [0, 9]] from two decoupled constraint-DV pairs:
          total_row = [0.5, 1/3], total_col = [0.5, 1/3]
        In block mode (size 1 vars), collapse of single-element slice = element itself.
        """
        dvs = {'x': _dv_meta(1), 'y': _dv_meta(1)}
        cons = {'c0': _out_meta(1), 'c1': _out_meta(1)}
        # Off-diagonal blocks are absent → zero in assembled J
        jac = {
            ('c0', 'x'): np.array([[4.0]]),
            ('c1', 'y'): np.array([[9.0]]),
        }
        driver = _MockDriver(dvs, cons, {}, jac=jac)

        a = EquilibrationAutoscaler(norm='inf', tol=1e-6)
        a.setup(driver)
        _configure(a, driver)

        self.assertAlmostEqual(dvs['x']['total_scaler'], 0.5, places=8)
        self.assertAlmostEqual(dvs['y']['total_scaler'], 1.0 / 3.0, places=8)
        self.assertAlmostEqual(cons['c0']['total_scaler'], 0.5, places=8)
        self.assertAlmostEqual(cons['c1']['total_scaler'], 1.0 / 3.0, places=8)

    def test_configure_sets_has_scaling(self):
        """_has_scaling is True after configure."""
        dvs = {'x': _dv_meta(1)}
        cons = {'c': _out_meta(1)}
        jac = {('c', 'x'): np.array([[1.0]])}
        driver = _MockDriver(dvs, cons, {}, jac=jac)

        a = EquilibrationAutoscaler()
        a.setup(driver)
        _configure(a, driver)

        self.assertTrue(a._has_scaling)

    def test_configure_no_dv_adder(self):
        """Equilibration does not produce DV adders; all DV adders are None."""
        dvs = {'x': _dv_meta(1), 'y': _dv_meta(1)}
        cons = {'c': _out_meta(1)}
        jac = {('c', 'x'): np.array([[2.0]]), ('c', 'y'): np.array([[3.0]])}
        driver = _MockDriver(dvs, cons, {}, jac=jac)

        a = EquilibrationAutoscaler()
        a.setup(driver)
        _configure(a, driver)

        self.assertIsNone(dvs['x']['total_adder'])
        self.assertIsNone(dvs['y']['total_adder'])

    def test_configure_no_constraint_adder(self):
        """Constraint and objective adders are also None after configure."""
        dvs = {'x': _dv_meta(1)}
        cons = {'c': _out_meta(1)}
        objs = {'obj': _out_meta(1)}
        jac = {('c', 'x'): np.array([[2.0]]), ('obj', 'x'): np.array([[3.0]])}
        driver = _MockDriver(dvs, cons, objs, jac=jac)

        a = EquilibrationAutoscaler()
        a.setup(driver)
        _configure(a, driver)

        self.assertIsNone(cons['c']['total_adder'])
        self.assertIsNone(objs['obj']['total_adder'])

    def test_configure_scalers_are_positive(self):
        """All computed scalers are strictly positive."""
        dvs = {'x': _dv_meta(2)}
        cons = {'c': _out_meta(2)}
        objs = {'obj': _out_meta(1)}
        jac = {
            ('c', 'x'): np.array([[1.0, 2.0], [3.0, 4.0]]),
            ('obj', 'x'): np.array([[0.5, 1.5]]),
        }
        driver = _MockDriver(dvs, cons, objs, jac=jac)

        a = EquilibrationAutoscaler(block_mode=True)
        a.setup(driver)
        _configure(a, driver)

        self.assertGreater(dvs['x']['total_scaler'], 0.0)
        self.assertGreater(cons['c']['total_scaler'], 0.0)
        self.assertGreater(objs['obj']['total_scaler'], 0.0)

    def test_configure_per_element_mode_produces_arrays(self):
        """Per-element mode produces ndarray scalers for vector variables."""
        dvs = {'x': _dv_meta(2)}
        cons = {'c': _out_meta(2)}
        jac = {('c', 'x'): np.array([[2.0, 0.0], [0.0, 8.0]])}
        driver = _MockDriver(dvs, cons, {}, jac=jac)

        a = EquilibrationAutoscaler(block_mode=False)
        a.setup(driver)
        _configure(a, driver)

        dv_s = dvs['x']['total_scaler']
        con_s = cons['c']['total_scaler']
        self.assertIsInstance(dv_s, np.ndarray)
        self.assertEqual(dv_s.shape, (2,))
        self.assertIsInstance(con_s, np.ndarray)
        self.assertEqual(con_s.shape, (2,))
        # All positive
        self.assertTrue(np.all(dv_s > 0))
        self.assertTrue(np.all(con_s > 0))

    def test_configure_compose_mode_multiplies_scalers(self):
        """override_scaling=False: equilibration scaler is multiplied with existing scaler."""
        # Set pre-existing scaler = 2.0 on DV; equilibration on J=[[1]] → eq_scaler=1.0
        # Composed: 1.0 * 2.0 = 2.0
        dvs = {'x': _dv_meta(1, total_scaler=2.0, total_adder=0.5)}
        cons = {'c': _out_meta(1, total_scaler=3.0, total_adder=-1.0)}
        jac = {('c', 'x'): np.array([[1.0]])}
        driver = _MockDriver(dvs, cons, {}, jac=jac)

        a = EquilibrationAutoscaler(norm='inf', override_scaling=False)
        a.setup(driver)
        _configure(a, driver)

        # eq_scaler for DV and con are both 1.0 (J=[[1]]) so composed = orig * 1.0
        self.assertAlmostEqual(dvs['x']['total_scaler'], 2.0)
        self.assertAlmostEqual(cons['c']['total_scaler'], 3.0)

    def test_configure_compose_mode_preserves_adder(self):
        """override_scaling=False: total_adder is unchanged for all VOI types."""
        dvs = {'x': _dv_meta(1, total_scaler=2.0, total_adder=0.5)}
        cons = {'c': _out_meta(1, total_scaler=3.0, total_adder=-1.0)}
        objs = {'obj': _out_meta(1, total_scaler=1.0, total_adder=0.25)}
        jac = {('c', 'x'): np.array([[1.0]]), ('obj', 'x'): np.array([[1.0]])}
        driver = _MockDriver(dvs, cons, objs, jac=jac)

        a = EquilibrationAutoscaler(norm='inf', override_scaling=False)
        a.setup(driver)
        _configure(a, driver)

        self.assertAlmostEqual(dvs['x']['total_adder'], 0.5)
        self.assertAlmostEqual(cons['c']['total_adder'], -1.0)
        self.assertAlmostEqual(objs['obj']['total_adder'], 0.25)

    def test_configure_zero_row_warns(self):
        """configure() warns when a constraint row is all zeros."""
        dvs = {'x': _dv_meta(1)}
        cons = {'c': _out_meta(1)}
        jac = {('c', 'x'): np.array([[0.0]])}
        driver = _MockDriver(dvs, cons, {}, jac=jac)

        a = EquilibrationAutoscaler()
        a.setup(driver)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _configure(a, driver)

        rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertGreater(len(rw), 0)

    def test_configure_fro_norm_produces_valid_scalers(self):
        """norm='fro' variant runs without error and produces positive scalers."""
        dvs = {'x': _dv_meta(1)}
        cons = {'c': _out_meta(1)}
        objs = {'obj': _out_meta(1)}
        jac = {('c', 'x'): np.array([[3.0]]), ('obj', 'x'): np.array([[4.0]])}
        driver = _MockDriver(dvs, cons, objs, jac=jac)

        a = EquilibrationAutoscaler(norm='fro')
        a.setup(driver)
        _configure(a, driver)

        self.assertGreater(dvs['x']['total_scaler'], 0.0)
        self.assertGreater(cons['c']['total_scaler'], 0.0)
        self.assertGreater(objs['obj']['total_scaler'], 0.0)
        self.assertTrue(a._has_scaling)

    def test_configure_empty_of_or_wrt_returns_early(self):
        """configure() returns without error when of_list or wrt_list is empty."""
        dvs = {'x': _dv_meta(1)}
        # No constraints or objectives → of_list is empty
        driver = _MockDriver(dvs, {}, {}, jac={})

        a = EquilibrationAutoscaler()
        a.setup(driver)
        # Should not raise
        _configure(a, driver)
        # _has_scaling stays False when no of_list
        self.assertFalse(a._has_scaling)


# ---------------------------------------------------------------------------
# Condition number diagnostic tests
# ---------------------------------------------------------------------------

class TestEquilibrationConditionNumbers(unittest.TestCase):
    """Test that configure() prints expected condition number strings."""

    def test_output_contains_expected_strings(self):
        """configure() prints unscaled, equilibration-scaled, and improvement info."""
        dvs = {'x': _dv_meta(1)}
        cons = {'c': _out_meta(1)}
        jac = {('c', 'x'): np.array([[5.0]])}
        driver = _MockDriver(dvs, cons, {}, jac=jac)

        a = EquilibrationAutoscaler(norm='inf')
        a.setup(driver)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            a.configure(driver)
        output = buf.getvalue()

        self.assertIn('Unscaled', output)
        self.assertIn('Equilibration-scaled', output)
        self.assertIn('Improvement', output)
        self.assertIn('norm=inf', output)

    def test_output_contains_fro_label(self):
        """norm='fro' is reflected in the diagnostic label."""
        dvs = {'x': _dv_meta(1)}
        cons = {'c': _out_meta(1)}
        jac = {('c', 'x'): np.array([[2.0]])}
        driver = _MockDriver(dvs, cons, {}, jac=jac)

        a = EquilibrationAutoscaler(norm='fro')
        a.setup(driver)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            a.configure(driver)
        output = buf.getvalue()

        self.assertIn('norm=fro', output)


# ---------------------------------------------------------------------------
# End-to-end tests with real OpenMDAO driver
# ---------------------------------------------------------------------------

class TestEquilibrationEndToEnd(unittest.TestCase):
    """Integration tests using ScipyOptimizeDriver with EquilibrationAutoscaler."""

    def _build_ill_conditioned_problem(self, autoscaler):
        """
        Build a problem with large variable magnitude disparities.

        The design variables span very different scales (x in [0,1],
        y in [0,1000]) to give the autoscaler a meaningful scaling task.
        """
        prob = om.Problem()
        prob.model.add_subsystem(
            'comp',
            om.ExecComp(
                ['c = x + 0.001*y', 'obj = x**2 + 1e-6*y**2'],
                x={'val': 0.5}, y={'val': 500.0}
            )
        )
        prob.model.add_design_var('comp.x', lower=0.0, upper=1.0)
        prob.model.add_design_var('comp.y', lower=0.0, upper=1000.0)
        prob.model.add_objective('comp.obj')
        prob.model.add_constraint('comp.c', upper=0.6)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-8)
        prob.driver.autoscaler = autoscaler
        return prob

    def test_inf_norm_block_mode_runs_and_converges(self):
        """Ruiz equilibration (norm='inf'), block mode: reaches feasibility."""
        prob = self._build_ill_conditioned_problem(
            EquilibrationAutoscaler(norm='inf', block_mode=True)
        )
        prob.setup()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            prob.run_driver()
        c_val = prob.get_val('comp.c')[0]
        self.assertLessEqual(c_val, 0.6 + 1e-6)

    def test_fro_norm_block_mode_runs_and_converges(self):
        """Sinkhorn-Knopp (norm='fro'), block mode: reaches feasibility."""
        prob = self._build_ill_conditioned_problem(
            EquilibrationAutoscaler(norm='fro', block_mode=True)
        )
        prob.setup()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            prob.run_driver()
        c_val = prob.get_val('comp.c')[0]
        self.assertLessEqual(c_val, 0.6 + 1e-6)

    def test_per_element_mode_runs_and_converges(self):
        """Per-element mode: optimizer reaches feasibility without raising errors."""
        prob = self._build_ill_conditioned_problem(
            EquilibrationAutoscaler(norm='inf', block_mode=False)
        )
        prob.setup()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            prob.run_driver()
        c_val = prob.get_val('comp.c')[0]
        self.assertLessEqual(c_val, 0.6 + 1e-6)

    def test_compose_mode_runs(self):
        """Compose mode with user-specified ref values runs without errors."""
        prob = om.Problem()
        prob.model.add_subsystem(
            'comp',
            om.ExecComp(['c = x - y', 'obj = x**2 + y**2'],
                        x={'val': 1.0}, y={'val': 1.0})
        )
        prob.model.add_design_var('comp.x', lower=-10.0, upper=10.0, ref=10.0)
        prob.model.add_design_var('comp.y', lower=-10.0, upper=10.0, ref=10.0)
        prob.model.add_objective('comp.obj')
        prob.model.add_constraint('comp.c', upper=0.0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-8)
        prob.driver.autoscaler = EquilibrationAutoscaler(block_mode=True,
                                                         override_scaling=False)
        prob.setup()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            prob.run_driver()
        # Just verify it ran to completion without exception
        self.assertIsNotNone(prob.get_val('comp.obj'))

    def test_unbounded_dvs_run_without_bounds(self):
        """Equilibration handles unbounded DVs (no bounds needed, unlike PJRN)."""
        prob = om.Problem()
        prob.model.add_subsystem(
            'comp',
            om.ExecComp(['c = x**2 + y - 1.0', 'obj = x**2 + y**2'],
                        x={'val': 0.5}, y={'val': 0.5})
        )
        # No bounds on DVs
        prob.model.add_design_var('comp.x')
        prob.model.add_design_var('comp.y')
        prob.model.add_objective('comp.obj')
        prob.model.add_constraint('comp.c', upper=0.0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-8)
        prob.driver.autoscaler = EquilibrationAutoscaler(norm='inf', block_mode=True)
        prob.setup()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            prob.run_driver()
        # Verify no exception and constraint satisfied
        c_val = prob.get_val('comp.c')[0]
        self.assertLessEqual(c_val, 0.0 + 1e-5)


if __name__ == '__main__':
    unittest.main()
