import unittest

import numpy as np

import openmdao.math as omath
from openmdao.utils.assert_utils import assert_near_equal

try:
    import jax
except ModuleNotFoundError:
    jax = None


class TestSmooth(unittest.TestCase):

    @unittest.skipIf(jax is None, 'jax is not available.')
    def test_tanh_act(self):
        f = omath.act_tanh(6, mu=1.0E-5, z=6, a=-10, b=10)
        assert_near_equal(np.asarray(f), 0.0)

        f = omath.act_tanh(6, mu=1.0E-5, z=6, a=-10, b=0)
        assert_near_equal(np.asarray(f), -5.0)

        f = omath.act_tanh(-10, mu=1.0E-5, z=6, a=-10, b=0)
        assert_near_equal(np.asarray(f), -10)

        f = omath.act_tanh(10, mu=1.0E-5, z=6, a=-10, b=20)
        assert_near_equal(np.asarray(f), 20)

    @unittest.skipIf(jax is None, 'jax is not available.')
    def test_smooth_max(self):
        x = np.linspace(0, 1, 1000)
        sin = np.sin(x)
        cos = np.cos(x)

        smax = omath.smooth_max(sin, cos, mu=1.0E-6)

        idxs_sgt = np.where(sin > cos)
        idxs_cgt = np.where(sin < cos)

        assert_near_equal(smax[idxs_sgt], sin[idxs_sgt])
        assert_near_equal(smax[idxs_cgt], cos[idxs_cgt])

    @unittest.skipIf(jax is None, 'jax is not available.')
    def test_smooth_min(self):
        x = np.linspace(0, 1, 1000)
        sin = np.sin(x)
        cos = np.cos(x)

        smin = omath.smooth_min(sin, cos, mu=1.0E-6)

        idxs_sgt = np.where(sin > cos)
        idxs_cgt = np.where(sin < cos)

        assert_near_equal(smin[idxs_sgt], cos[idxs_sgt])
        assert_near_equal(smin[idxs_cgt], sin[idxs_cgt])

    @unittest.skipIf(jax is None, 'jax is not available.')
    def test_smooth_abs(self):
        x = np.linspace(-0.5, 0.5, 1000)

        sabs = omath.smooth_abs(x)
        abs = np.abs(x)

        idxs_compare = np.where(abs > 0.1)
        assert_near_equal(sabs[idxs_compare], abs[idxs_compare], tolerance=1.0E-9)

    @unittest.skipIf(jax is None, 'jax is not available.')
    def test_ks_max(self):
        x = np.random.random(1000)

        ksmax = omath.ks_max(x, rho=1.E6)
        npmax = np.max(x)

        assert_near_equal(ksmax, npmax, tolerance=1.0E-6)

    @unittest.skipIf(jax is None, 'jax is not available.')
    def test_ks_min(self):
        x = np.random.random(1000)

        ksmin = omath.ks_min(x, rho=1.E6)
        npmin = np.min(x)

        assert_near_equal(ksmin, npmin, tolerance=1.0E-6)


if __name__ == '__main__':
    unittest.main()
