"""
Unit tests for the spline distribution class.
"""
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from openmdao.utils.spline_distributions import SplineDistribution

class SplineDistributionsTestCase(unittest.TestCase):

    def test_cell_centered(self):

        xcp = np.array([1.0, 2.0, 4.0, 6.0, 10.0, 12.0])
        s_dist = SplineDistribution()

        calculated_midpoints = np.array([3.7, 7.0, 10.3, 13.6, 16.9])
        assert_array_almost_equal(calculated_midpoints, s_dist.cell_centered(xcp, 5))

    def test_cell_centered_error_msg(self):

        xcp = np.array([1.0, 2.0, 4.0])
        s_dist = SplineDistribution()


        with self.assertRaises(KeyError) as cm:
            s_dist.cell_centered(xcp, 3)

        msg = "'Number of points must be less than input_points.'"
        self.assertEqual(msg, str(cm.exception))

    def test_sin_distribution(self):

        xcp = np.array([1.0, 2.0, 4.0])
        s_dist = SplineDistribution()

        calculated_midpoints = np.array(
            [1.        , 0.93973688, 0.77347408, 0.54128967, 0.29915229,
             0.10542975, 0.00681935, 0.02709138, 0.16135921, 0.37725726,
             0.62274274, 0.83864079, 0.97290862, 0.99318065, 0.89457025,
             0.70084771, 0.45871033, 0.22652592, 0.06026312, 0.        ])

        assert_array_almost_equal(calculated_midpoints, s_dist.sine_distribution(xcp, 20))

    def test_node_centered(self):

        xcp = np.array([1.0, 2.0, 4.0])
        s_dist = SplineDistribution()

        calculated_midpoints = np.array([1., 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.])

        assert_array_almost_equal(calculated_midpoints, s_dist.node_centered(xcp, 10))