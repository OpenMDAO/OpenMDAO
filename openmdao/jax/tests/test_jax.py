import unittest

import numpy as np
from openmdao.utils.assert_utils import assert_near_equal

from openmdao.utils.jax_utils import jax

if jax is not None:
    from openmdao.jax import act_tanh, smooth_abs, smooth_max, smooth_min, ks_max, \
        ks_min, smooth_round


@unittest.skipIf(jax is None, 'jax is not available.')
class TestJax(unittest.TestCase):

    def test_tanh_act(self):
        f = act_tanh(6, mu=1.0E-5, z=6, a=-10, b=10)
        assert_near_equal(np.asarray(f), 0.0)

        f = act_tanh(6, mu=1.0E-5, z=6, a=-10, b=0)
        assert_near_equal(np.asarray(f), -5.0)

        f = act_tanh(-10, mu=1.0E-5, z=6, a=-10, b=0)
        assert_near_equal(np.asarray(f), -10)

        f = act_tanh(10, mu=1.0E-5, z=6, a=-10, b=20)
        assert_near_equal(np.asarray(f), 20)

    def test_smooth_max(self):
        x = np.linspace(0, 1, 1000)
        sin = np.sin(x)
        cos = np.cos(x)

        smax = np.asarray(smooth_max(sin, cos, mu=1.0E-6))

        idxs_sgt = np.where(sin > cos)
        idxs_cgt = np.where(sin < cos)

        assert_near_equal(smax[idxs_sgt], sin[idxs_sgt])
        assert_near_equal(smax[idxs_cgt], cos[idxs_cgt])

    def test_smooth_min(self):
        x = np.linspace(0, 1, 1000)
        sin = np.sin(x)
        cos = np.cos(x)

        smin = np.asarray(smooth_min(sin, cos, mu=1.0E-6))

        idxs_sgt = np.where(sin > cos)
        idxs_cgt = np.where(sin < cos)

        assert_near_equal(smin[idxs_sgt], cos[idxs_sgt])
        assert_near_equal(smin[idxs_cgt], sin[idxs_cgt])

    def test_smooth_abs(self):
        x = np.linspace(-0.5, 0.5, 1000)

        sabs = np.asarray(smooth_abs(x))
        abs = np.abs(x)

        idxs_compare = np.where(abs > 0.1)
        assert_near_equal(sabs[idxs_compare], abs[idxs_compare], tolerance=1.0E-9)

    def test_smooth_round(self):
        x = np.linspace(10, -10, 100)

        round_x = np.asarray(smooth_round(x, mu=0.1))

        expected_x = np.array([10.,   9.997,   9.872,   9.107,   9.002,   9.,   8.997,
                               8.848,   8.089,   8.002,   8.,   7.996,   7.82,   7.074,
                               7.001,   7.,   6.995,   6.788,   6.061,   6.001,   6.,
                               5.994,   5.752,   5.051,   5.001,   5.,   4.993,   4.713,
                               4.042,   4.001,   4.,   3.991,   3.67,   3.034,   3.001,
                               3.,   2.989,   2.624,   2.028,   2.001,   2.,   1.987,
                               1.575,   1.023,   1.,   1.,   0.984,   0.525,   0.019,
                               0.,  -0.,  -0.019,  -0.525,  -0.984,  -1.,  -1.,
                               -1.023,  -1.575,  -1.987,  -2.,  -2.001,  -2.028,  -2.624,
                               -2.989,  -3.,  -3.001,  -3.034,  -3.67,  -3.991,  -4.,
                               -4.001,  -4.042,  -4.713,  -4.993,  -5.,  -5.001,  -5.051,
                               -5.752,  -5.994,  -6.,  -6.001,  -6.061,  -6.788,  -6.995,
                               -7.,  -7.001,  -7.074,  -7.82,  -7.996,  -8.,  -8.002,
                               -8.089,  -8.848,  -8.997,  -9.,  -9.002,  -9.107,  -9.872,
                               -9.997, -10.])

        assert_near_equal(round_x, expected_x, tolerance=1e-4)

    def test_ks_max(self):
        x = np.random.random(1000)

        ksmax = np.asarray(ks_max(x, rho=1.E6))
        npmax = np.max(x)

        assert_near_equal(ksmax, npmax, tolerance=1.0E-6)

    def test_ks_min(self):
        x = np.random.random(1000)

        ksmin = np.asarray(ks_min(x, rho=1.E6))
        npmin = np.min(x)

        assert_near_equal(ksmin, npmin, tolerance=2.0E-5)


if __name__ == '__main__':
    unittest.main()
