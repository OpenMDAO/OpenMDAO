"""Tests for PJRNAutoscaler."""
import io
import contextlib
import unittest
import warnings

import numpy as np

import openmdao.api as om
from openmdao.core.constants import INF_BOUND
from openmdao.drivers.autoscalers import PJRNAutoscaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dv_meta(size, lower, upper, total_scaler=None, total_adder=None):
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
    def __init__(self, dvs, cons, objs, dv_vals=None, jac=None):
        self._designvars = dvs
        self._cons = cons
        self._objs = objs
        self._dv_vals = dv_vals if dv_vals is not None else {
            n: np.ones(m['size']) for n, m in dvs.items()
        }
        self._jac = jac or {}

    def get_design_var_values(self, driver_scaling=True):
        return self._dv_vals

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

class TestPJRNAutoscalerInit(unittest.TestCase):
    """Test constructor parameters and configure_requires_run_model."""

    def test_default_params(self):
        """Default block_mode=True, override_scaling=True, large_range_tol=1e10."""
        a = PJRNAutoscaler()
        self.assertTrue(a._block_mode)
        self.assertTrue(a._override_scaling)
        self.assertEqual(a._large_range_tol, 1e10)

    def test_custom_params(self):
        """Custom block_mode=False, override_scaling=False, large_range_tol=1e6."""
        a = PJRNAutoscaler(block_mode=False, override_scaling=False, large_range_tol=1e6)
        self.assertFalse(a._block_mode)
        self.assertFalse(a._override_scaling)
        self.assertEqual(a._large_range_tol, 1e6)

    def test_configure_requires_run_model(self):
        """configure_requires_run_model is always True."""
        self.assertTrue(PJRNAutoscaler().configure_requires_run_model)
        self.assertTrue(
            PJRNAutoscaler(block_mode=False, override_scaling=False).configure_requires_run_model
        )


# ---------------------------------------------------------------------------
# setup() tests
# ---------------------------------------------------------------------------

class TestPJRNAutoscalerSetup(unittest.TestCase):
    """Test that setup() saves original scalers and adders correctly."""

    def test_setup_saves_original_scalers_and_adders(self):
        """setup() stores a snapshot of total_scaler/total_adder for all VOI types."""
        dvs = {'x': _dv_meta(1, lower=0.0, upper=10.0, total_scaler=2.0, total_adder=1.0)}
        cons = {'c': _out_meta(1, total_scaler=0.5, total_adder=0.1)}
        objs = {'obj': _out_meta(1, total_scaler=3.0)}
        driver = _MockDriver(dvs, cons, objs)

        a = PJRNAutoscaler()
        a.setup(driver)

        self.assertEqual(a._original_scalers['design_var']['x'], 2.0)
        self.assertEqual(a._original_adders['design_var']['x'], 1.0)
        self.assertEqual(a._original_scalers['constraint']['c'], 0.5)
        self.assertEqual(a._original_adders['constraint']['c'], 0.1)
        self.assertEqual(a._original_scalers['objective']['obj'], 3.0)

    def test_setup_saves_none_scalers(self):
        """setup() preserves None as the original scaler when none was set."""
        dvs = {'x': _dv_meta(1, lower=0.0, upper=10.0)}
        driver = _MockDriver(dvs, {}, {'obj': _out_meta(1)})

        a = PJRNAutoscaler()
        a.setup(driver)

        self.assertIsNone(a._original_scalers['design_var']['x'])
        self.assertIsNone(a._original_adders['design_var']['x'])


# ---------------------------------------------------------------------------
# _build_kx_inv tests
# ---------------------------------------------------------------------------

class TestPJRNBuildKxInv(unittest.TestCase):
    """Test _build_kx_inv: projection matrix and adder construction."""

    def _setup(self, dvs):
        driver = _MockDriver(dvs, {}, {'obj': _out_meta(1)})
        a = PJRNAutoscaler()
        a.setup(driver)
        return a

    def test_bounded_scalar_dv(self):
        """Single bounded DV: kx_inv = ub - lb, adder = -lb."""
        dvs = {'x': _dv_meta(1, lower=2.0, upper=12.0)}
        a = self._setup(dvs)

        kx_inv, adders = a._build_kx_inv(dvs, ['x'])

        np.testing.assert_allclose(kx_inv['x'], [10.0])
        np.testing.assert_allclose(adders['x'], [-2.0])

    def test_bounded_vector_dv(self):
        """Vector DV with differing element bounds."""
        dvs = {'x': _dv_meta(2,
                              lower=np.array([0.0, -5.0]),
                              upper=np.array([10.0, 5.0]))}
        a = self._setup(dvs)

        kx_inv, adders = a._build_kx_inv(dvs, ['x'])

        np.testing.assert_allclose(kx_inv['x'], [10.0, 10.0])
        # adder = -lb: [0.0, 5.0]
        np.testing.assert_allclose(adders['x'], [0.0, 5.0])

    def test_lb_zero_gives_zero_adder(self):
        """When lb = 0 the adder element is 0."""
        dvs = {'x': _dv_meta(1, lower=0.0, upper=5.0)}
        a = self._setup(dvs)
        _, adders = a._build_kx_inv(dvs, ['x'])
        np.testing.assert_allclose(adders['x'], [0.0])

    def test_unbounded_dv_uses_unit_fallback(self):
        """Unbounded DV falls back to a characteristic range of 1.0 and warns."""
        dvs = {'x': _dv_meta(1, lower=-INF_BOUND, upper=INF_BOUND)}
        a = self._setup(dvs)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            kx_inv, adders = a._build_kx_inv(dvs, ['x'])

        np.testing.assert_allclose(kx_inv['x'], [1.0])
        # No finite lb, so adder is None
        self.assertIsNone(adders['x'])
        # Exactly one warning mentioning 'unbounded'
        rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(rw), 1)
        self.assertIn('unbounded', str(rw[0].message).lower())

    def test_partially_unbounded_dv(self):
        """Mix of bounded and unbounded elements in one vector DV."""
        dvs = {'x': _dv_meta(2,
                              lower=np.array([0.0, -INF_BOUND]),
                              upper=np.array([10.0, INF_BOUND]))}
        a = self._setup(dvs)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            kx_inv, adders = a._build_kx_inv(dvs, ['x'])

        # Element 0: 10 - 0 = 10; Element 1: unbounded → fallback = 1.0
        np.testing.assert_allclose(kx_inv['x'], [10.0, 1.0])
        # Element 0 has finite lb=0 → adder[0]=0; element 1 is unbounded → adder[1]=0
        np.testing.assert_allclose(adders['x'], [0.0, 0.0])
        rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(rw), 1)

    def test_dymos_sentinel_bounds_treated_as_unbounded(self):
        """Dymos registers ±1e21 as 'infinite' bounds; these must trigger the fallback."""
        # Dymos INF_BOUND = 1e21, which is finite in IEEE 754 but is a sentinel
        dymos_inf = 1.0e21
        dvs = {'x': _dv_meta(1, lower=-dymos_inf, upper=dymos_inf)}
        a = self._setup(dvs)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            kx_inv, adders = a._build_kx_inv(dvs, ['x'])

        # Range = 2e21 > large_range_tol (1e10) → treated as unbounded → fallback = 1.0
        np.testing.assert_allclose(kx_inv['x'], [1.0])
        self.assertIsNone(adders['x'])
        rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(rw), 1)

    def test_custom_large_range_tol(self):
        """A custom large_range_tol is respected: range just below it is kept as-is."""
        # With large_range_tol=1e6, a range of 5e5 is fine but 2e6 is treated as unbounded.
        dvs_ok = {'x': _dv_meta(1, lower=0.0, upper=5e5)}
        dvs_large = {'x': _dv_meta(1, lower=0.0, upper=2e6)}

        a_ok = PJRNAutoscaler(large_range_tol=1e6)
        driver_ok = _MockDriver(dvs_ok, {}, {'obj': _out_meta(1)})
        a_ok.setup(driver_ok)

        a_large = PJRNAutoscaler(large_range_tol=1e6)
        driver_large = _MockDriver(dvs_large, {}, {'obj': _out_meta(1)})
        a_large.setup(driver_large)

        kx_inv_ok, _ = a_ok._build_kx_inv(dvs_ok, ['x'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            kx_inv_large, _ = a_large._build_kx_inv(dvs_large, ['x'])

        np.testing.assert_allclose(kx_inv_ok['x'], [5e5])
        np.testing.assert_allclose(kx_inv_large['x'], [1.0])
        rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(rw), 1)


# ---------------------------------------------------------------------------
# _compute_dv_scalers tests
# ---------------------------------------------------------------------------

class TestPJRNComputeDVScalers(unittest.TestCase):
    """Test _compute_dv_scalers in both modes."""

    def test_block_mode_scalar_dv(self):
        """Block mode: single-element DV → scaler = 1/(ub-lb)."""
        dvs = {'x': _dv_meta(1, lower=0.0, upper=10.0)}
        a = PJRNAutoscaler(block_mode=True)
        kx_inv = {'x': np.array([10.0])}
        scalers = a._compute_dv_scalers(dvs, kx_inv)
        self.assertAlmostEqual(scalers['x'], 0.1)

    def test_block_mode_vector_dv(self):
        """Block mode: vector DV → scaler = 1/max(ub-lb) across elements."""
        dvs = {'x': _dv_meta(2, lower=np.array([0.0, 0.0]), upper=np.array([10.0, 20.0]))}
        a = PJRNAutoscaler(block_mode=True)
        kx_inv = {'x': np.array([10.0, 20.0])}
        scalers = a._compute_dv_scalers(dvs, kx_inv)
        # max range = 20 → scaler = 1/20
        self.assertAlmostEqual(scalers['x'], 0.05)

    def test_per_element_mode(self):
        """Per-element mode: vector DV → per-element scaler = 1/(ub-lb)."""
        dvs = {'x': _dv_meta(2, lower=np.array([0.0, 0.0]), upper=np.array([10.0, 20.0]))}
        a = PJRNAutoscaler(block_mode=False)
        kx_inv = {'x': np.array([10.0, 20.0])}
        scalers = a._compute_dv_scalers(dvs, kx_inv)
        np.testing.assert_allclose(scalers['x'], [0.1, 0.05])

    def test_multiple_dvs(self):
        """Multiple DVs each get their own scaler."""
        dvs = {
            'x': _dv_meta(1, lower=0.0, upper=4.0),
            'y': _dv_meta(1, lower=-5.0, upper=5.0),
        }
        a = PJRNAutoscaler(block_mode=True)
        kx_inv = {'x': np.array([4.0]), 'y': np.array([10.0])}
        scalers = a._compute_dv_scalers(dvs, kx_inv)
        self.assertAlmostEqual(scalers['x'], 0.25)
        self.assertAlmostEqual(scalers['y'], 0.1)


# ---------------------------------------------------------------------------
# _compute_output_scalers tests
# ---------------------------------------------------------------------------

class TestPJRNComputeOutputScalers(unittest.TestCase):
    """Test projected Jacobian row-norm scaler computation."""

    def test_block_mode_single_element(self):
        """Block mode, scalar constraint: scaler = 1/||J_proj||."""
        # kx_inv = [10], J = [[3]] → J_proj = [[30]] → norm = 30
        a = PJRNAutoscaler(block_mode=True)
        out_meta = {'c': _out_meta(1)}
        kx_inv = {'x': np.array([10.0])}
        jac = {('c', 'x'): np.array([[3.0]])}
        scalers = a._compute_output_scalers(out_meta, ['x'], jac, kx_inv)
        self.assertAlmostEqual(scalers['c'], 1.0 / 30.0)

    def test_block_mode_multi_element(self):
        """Block mode, vector constraint: scaler = 1/||J_proj||_F."""
        # J = [[1,2],[3,4]], kx_inv=[10,20]
        # J_proj = [[10,40],[30,80]]
        # ||J_proj||_F = sqrt(100+1600+900+6400) = sqrt(9000)
        a = PJRNAutoscaler(block_mode=True)
        out_meta = {'c': _out_meta(2)}
        kx_inv = {'x': np.array([10.0, 20.0])}
        jac = {('c', 'x'): np.array([[1.0, 2.0], [3.0, 4.0]])}
        scalers = a._compute_output_scalers(out_meta, ['x'], jac, kx_inv)
        self.assertAlmostEqual(scalers['c'], 1.0 / np.sqrt(9000.0))

    def test_per_element_mode(self):
        """Per-element mode: each row gets its own scaler."""
        # J = [[1,2],[3,4]], kx_inv=[10,20]
        # J_proj = [[10,40],[30,80]]
        # Row 0 norm: sqrt(100+1600) = sqrt(1700)
        # Row 1 norm: sqrt(900+6400) = sqrt(7300)
        a = PJRNAutoscaler(block_mode=False)
        out_meta = {'c': _out_meta(2)}
        kx_inv = {'x': np.array([10.0, 20.0])}
        jac = {('c', 'x'): np.array([[1.0, 2.0], [3.0, 4.0]])}
        scalers = a._compute_output_scalers(out_meta, ['x'], jac, kx_inv)
        expected = np.array([1.0 / np.sqrt(1700.0), 1.0 / np.sqrt(7300.0)])
        np.testing.assert_allclose(scalers['c'], expected)

    def test_multiple_dvs_accumulates_projection(self):
        """Projection is summed across all DV columns."""
        # Two DVs: x (size 1, kx_inv=4), y (size 1, kx_inv=3)
        # J_c_x = [[2]], J_c_y = [[1]]
        # J_proj row: [2*4, 1*3] = [8, 3]
        # norm = sqrt(64+9) = sqrt(73)
        a = PJRNAutoscaler(block_mode=True)
        out_meta = {'c': _out_meta(1)}
        kx_inv = {'x': np.array([4.0]), 'y': np.array([3.0])}
        jac = {('c', 'x'): np.array([[2.0]]), ('c', 'y'): np.array([[1.0]])}
        scalers = a._compute_output_scalers(out_meta, ['x', 'y'], jac, kx_inv)
        self.assertAlmostEqual(scalers['c'], 1.0 / np.sqrt(73.0))

    def test_zero_norm_warns_and_sets_unit_scaler(self):
        """Zero projected norm triggers a warning and falls back to scaler = 1.0."""
        a = PJRNAutoscaler(block_mode=True)
        out_meta = {'c': _out_meta(1)}
        kx_inv = {'x': np.array([10.0])}
        jac = {('c', 'x'): np.array([[0.0]])}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            scalers = a._compute_output_scalers(out_meta, ['x'], jac, kx_inv)

        self.assertAlmostEqual(scalers['c'], 1.0)
        rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertGreater(len(rw), 0)

    def test_zero_norm_per_element_warns_and_sets_unit_scaler(self):
        """Per-element zero-norm rows get scaler = 1.0 individually."""
        a = PJRNAutoscaler(block_mode=False)
        out_meta = {'c': _out_meta(2)}
        kx_inv = {'x': np.array([10.0])}
        # Row 0 is zero, row 1 is non-zero
        jac = {('c', 'x'): np.array([[0.0], [5.0]])}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            scalers = a._compute_output_scalers(out_meta, ['x'], jac, kx_inv)

        # Row 0: zero → 1.0; Row 1: 50 → 1/50
        np.testing.assert_allclose(scalers['c'], [1.0, 1.0 / 50.0])
        rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertGreater(len(rw), 0)


# ---------------------------------------------------------------------------
# _apply_scalers tests
# ---------------------------------------------------------------------------

class TestPJRNApplyScalers(unittest.TestCase):
    """Test that _apply_scalers writes to metadata correctly in both modes."""

    def _make_autoscaler_with_originals(self, voi_type, name, orig_scaler, orig_adder,
                                        override_scaling):
        a = PJRNAutoscaler(override_scaling=override_scaling)
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
        pjrn = np.array([0.1, 0.2])
        a._apply_scalers('constraint', meta, {'c': pjrn}, {'c': None})
        np.testing.assert_allclose(meta['c']['total_scaler'], [0.1, 0.2])

    def test_compose_multiplies_scaler_keeps_adder(self):
        """override_scaling=False: total_scaler = pjrn * orig; total_adder unchanged."""
        a = self._make_autoscaler_with_originals('constraint', 'c', 4.0, 0.5,
                                                  override_scaling=False)
        meta = {'c': {'total_scaler': 4.0, 'total_adder': 0.5, 'discrete': False}}
        a._apply_scalers('constraint', meta, {'c': 0.5}, {'c': None})
        self.assertAlmostEqual(meta['c']['total_scaler'], 2.0)   # 0.5 * 4.0
        self.assertAlmostEqual(meta['c']['total_adder'], 0.5)    # unchanged

    def test_compose_with_none_original_uses_pjrn_scaler(self):
        """Compose mode with None original scaler just uses the PJRN scaler."""
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
# configure() integration tests (MockDriver)
# ---------------------------------------------------------------------------

class TestPJRNConfigureIntegration(unittest.TestCase):
    """End-to-end configure() tests using a MockDriver with known Jacobians."""

    def test_configure_block_mode_scalar_problem(self):
        """Block mode with scalar DV, constraint, objective: verify exact scalers."""
        # DV x: lb=0, ub=10 → kx_inv=[10] → dv_scaler=1/10
        # J_con_x = [[3.0]] → J_proj = [[30]] → con_scaler = 1/30
        # J_obj_x = [[5.0]] → J_proj = [[50]] → obj_scaler = 1/50
        dvs = {'x': _dv_meta(1, lower=0.0, upper=10.0)}
        cons = {'c': _out_meta(1)}
        objs = {'obj': _out_meta(1)}
        jac = {('c', 'x'): np.array([[3.0]]), ('obj', 'x'): np.array([[5.0]])}
        driver = _MockDriver(dvs, cons, objs, dv_vals={'x': np.array([1.0])}, jac=jac)

        a = PJRNAutoscaler(block_mode=True)
        a.setup(driver)
        _configure(a, driver)

        self.assertAlmostEqual(dvs['x']['total_scaler'], 1.0 / 10.0)
        self.assertAlmostEqual(cons['c']['total_scaler'], 1.0 / 30.0)
        self.assertAlmostEqual(objs['obj']['total_scaler'], 1.0 / 50.0)
        # DV adder: -lb = 0
        np.testing.assert_allclose(dvs['x']['total_adder'], [0.0])
        # Output adders: None (PJRN does not add an offset to outputs)
        self.assertIsNone(cons['c']['total_adder'])
        self.assertIsNone(objs['obj']['total_adder'])

    def test_configure_per_element_mode(self):
        """Per-element mode with vector DV and vector constraint."""
        # DV x (size 2): lb=[0,0], ub=[10,20] → kx_inv=[10,20]
        # J_con = [[1,2],[3,4]] → J_proj = [[10,40],[30,80]]
        # Row norms: sqrt(1700), sqrt(7300)
        dvs = {'x': _dv_meta(2,
                              lower=np.array([0.0, 0.0]),
                              upper=np.array([10.0, 20.0]))}
        cons = {'c': _out_meta(2)}
        objs = {'obj': _out_meta(1)}
        jac = {
            ('c', 'x'): np.array([[1.0, 2.0], [3.0, 4.0]]),
            ('obj', 'x'): np.array([[5.0, 1.0]]),
        }
        driver = _MockDriver(dvs, cons, objs,
                             dv_vals={'x': np.array([1.0, 1.0])}, jac=jac)

        a = PJRNAutoscaler(block_mode=False)
        a.setup(driver)
        _configure(a, driver)

        np.testing.assert_allclose(dvs['x']['total_scaler'], [0.1, 0.05])
        np.testing.assert_allclose(
            cons['c']['total_scaler'],
            [1.0 / np.sqrt(1700.0), 1.0 / np.sqrt(7300.0)]
        )
        # Obj row: [5*10, 1*20] = [50, 20] → norm = sqrt(2500+400) = sqrt(2900)
        np.testing.assert_allclose(
            objs['obj']['total_scaler'],
            [1.0 / np.sqrt(2900.0)]
        )

    def test_configure_dv_adder_from_lower_bound(self):
        """DV adder = -lb so that (x + adder)*scaler maps lb → 0."""
        dvs = {'x': _dv_meta(2,
                              lower=np.array([2.0, -3.0]),
                              upper=np.array([12.0, 7.0]))}
        cons = {'c': _out_meta(1)}
        objs = {'obj': _out_meta(1)}
        jac = {('c', 'x'): np.array([[1.0, 1.0]]),
               ('obj', 'x'): np.array([[1.0, 1.0]])}
        driver = _MockDriver(dvs, cons, objs, jac=jac)

        a = PJRNAutoscaler(block_mode=True)
        a.setup(driver)
        _configure(a, driver)

        # adder = -lb = [-2, 3]
        np.testing.assert_allclose(dvs['x']['total_adder'], [-2.0, 3.0])

    def test_configure_sets_has_scaling(self):
        """_has_scaling is True after configure."""
        dvs = {'x': _dv_meta(1, lower=0.0, upper=10.0)}
        cons = {'c': _out_meta(1)}
        objs = {'obj': _out_meta(1)}
        jac = {('c', 'x'): np.array([[1.0]]), ('obj', 'x'): np.array([[1.0]])}
        driver = _MockDriver(dvs, cons, objs, jac=jac)

        a = PJRNAutoscaler()
        a.setup(driver)
        _configure(a, driver)

        self.assertTrue(a._has_scaling)

    def test_configure_compose_mode_multiplies_scalers(self):
        """override_scaling=False: PJRN scaler is multiplied with existing total_scaler."""
        # User has pre-set: DV scaler=2, con scaler=3, obj scaler=4
        # PJRN will compute: DV 1/10, con 1/30, obj 1/50
        # Composed: DV 2/10=0.2, con 3/30=0.1, obj 4/50=0.08
        dvs = {'x': _dv_meta(1, lower=0.0, upper=10.0, total_scaler=2.0, total_adder=1.0)}
        cons = {'c': _out_meta(1, total_scaler=3.0, total_adder=0.5)}
        objs = {'obj': _out_meta(1, total_scaler=4.0)}
        jac = {('c', 'x'): np.array([[3.0]]), ('obj', 'x'): np.array([[5.0]])}
        driver = _MockDriver(dvs, cons, objs, dv_vals={'x': np.array([1.0])}, jac=jac)

        a = PJRNAutoscaler(block_mode=True, override_scaling=False)
        a.setup(driver)
        _configure(a, driver)

        self.assertAlmostEqual(dvs['x']['total_scaler'], 0.2)
        self.assertAlmostEqual(cons['c']['total_scaler'], 0.1)
        self.assertAlmostEqual(objs['obj']['total_scaler'], 0.08)

    def test_configure_compose_mode_preserves_adder(self):
        """In compose mode total_adder is not modified for any VOI type."""
        dvs = {'x': _dv_meta(1, lower=0.0, upper=10.0, total_scaler=2.0, total_adder=1.0)}
        cons = {'c': _out_meta(1, total_scaler=3.0, total_adder=0.5)}
        objs = {'obj': _out_meta(1, total_scaler=4.0, total_adder=-1.0)}
        jac = {('c', 'x'): np.array([[3.0]]), ('obj', 'x'): np.array([[5.0]])}
        driver = _MockDriver(dvs, cons, objs, dv_vals={'x': np.array([1.0])}, jac=jac)

        a = PJRNAutoscaler(block_mode=True, override_scaling=False)
        a.setup(driver)
        _configure(a, driver)

        # Adders are all unchanged in compose mode
        self.assertAlmostEqual(dvs['x']['total_adder'], 1.0)
        self.assertAlmostEqual(cons['c']['total_adder'], 0.5)
        self.assertAlmostEqual(objs['obj']['total_adder'], -1.0)

    def test_configure_unbounded_dv_warns(self):
        """configure() warns when a DV has unbounded or zero-range elements."""
        dvs = {'x': _dv_meta(1, lower=-INF_BOUND, upper=INF_BOUND)}
        cons = {'c': _out_meta(1)}
        objs = {'obj': _out_meta(1)}
        jac = {('c', 'x'): np.array([[1.0]]), ('obj', 'x'): np.array([[1.0]])}
        driver = _MockDriver(dvs, cons, objs, dv_vals={'x': np.array([3.0])}, jac=jac)

        a = PJRNAutoscaler()
        a.setup(driver)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _configure(a, driver)

        rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertGreater(len(rw), 0)
        self.assertIn('unbounded', str(rw[0].message).lower())

    def test_configure_zero_jacobian_warns_and_uses_unit_scaler(self):
        """configure() warns for zero projected rows and leaves scaler as 1.0."""
        dvs = {'x': _dv_meta(1, lower=0.0, upper=10.0)}
        cons = {'c': _out_meta(1)}
        objs = {'obj': _out_meta(1)}
        jac = {('c', 'x'): np.array([[0.0]]), ('obj', 'x'): np.array([[0.0]])}
        driver = _MockDriver(dvs, cons, objs, jac=jac)

        a = PJRNAutoscaler()
        a.setup(driver)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _configure(a, driver)

        rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertGreater(len(rw), 0)
        self.assertAlmostEqual(cons['c']['total_scaler'], 1.0)
        self.assertAlmostEqual(objs['obj']['total_scaler'], 1.0)


# ---------------------------------------------------------------------------
# Condition number diagnostic tests
# ---------------------------------------------------------------------------

class TestPJRNConditionNumbers(unittest.TestCase):
    """Test that _print_condition_numbers produces reasonable output."""

    def test_output_contains_condition_numbers(self):
        """configure() prints unscaled and scaled condition numbers."""
        dvs = {'x': _dv_meta(1, lower=0.0, upper=10.0)}
        cons = {'c': _out_meta(1)}
        objs = {'obj': _out_meta(1)}
        jac = {('c', 'x'): np.array([[3.0]]), ('obj', 'x'): np.array([[5.0]])}
        driver = _MockDriver(dvs, cons, objs, jac=jac)

        a = PJRNAutoscaler()
        a.setup(driver)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            a.configure(driver)
        output = buf.getvalue()

        self.assertIn('Unscaled', output)
        self.assertIn('PJRN-scaled', output)
        self.assertIn('Improvement', output)


# ---------------------------------------------------------------------------
# End-to-end tests with real OpenMDAO driver
# ---------------------------------------------------------------------------

class TestPJRNEndToEnd(unittest.TestCase):
    """Integration tests using ScipyOptimizeDriver with PJRNAutoscaler."""

    def _build_ill_conditioned_problem(self, autoscaler):
        """
        Build a problem with large variable magnitude disparities.

        The design variables span very different scales (x in [0,1],
        y in [0,1000]) to give PJRN a meaningful scaling task.
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

    def test_block_mode_runs_and_converges(self):
        """Block mode: optimizer reaches feasibility without raising errors."""
        prob = self._build_ill_conditioned_problem(PJRNAutoscaler(block_mode=True))
        prob.setup()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            prob.run_driver()
        # Constraint satisfaction: c = x + 0.001*y <= 0.6
        c_val = prob.get_val('comp.c')[0]
        self.assertLessEqual(c_val, 0.6 + 1e-6)

    def test_per_element_mode_runs_and_converges(self):
        """Per-element mode: optimizer reaches feasibility without raising errors."""
        prob = self._build_ill_conditioned_problem(PJRNAutoscaler(block_mode=False))
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
        prob.driver.autoscaler = PJRNAutoscaler(block_mode=True, override_scaling=False)
        prob.setup()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            prob.run_driver()
        # Just verify it ran to completion without exception
        self.assertIsNotNone(prob.get_val('comp.obj'))

    def test_dv_scaling_maps_bounds_to_unit_interval(self):
        """After configure, (lb + adder)*scaler ≈ 0 and (ub + adder)*scaler ≈ 1."""
        prob = om.Problem()
        prob.model.add_subsystem(
            'comp',
            om.ExecComp(['c = x', 'obj = x**2'], x={'val': 2.0})
        )
        prob.model.add_design_var('comp.x', lower=-5.0, upper=15.0)
        prob.model.add_objective('comp.obj')
        prob.model.add_constraint('comp.c', upper=10.0)

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP')
        prob.driver.autoscaler = PJRNAutoscaler(block_mode=True)
        prob.setup()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            prob.run_driver()

        dv_meta = prob.driver._designvars['comp.x']
        scaler = np.atleast_1d(dv_meta['total_scaler'])
        adder = np.atleast_1d(dv_meta['total_adder'])

        # (lb + adder) * scaler = (-5 + 5) * (1/20) = 0
        self.assertAlmostEqual(float((-5.0 + adder[0]) * scaler[0]), 0.0, places=10)
        # (ub + adder) * scaler = (15 + 5) * (1/20) = 1
        self.assertAlmostEqual(float((15.0 + adder[0]) * scaler[0]), 1.0, places=10)


if __name__ == '__main__':
    unittest.main()
