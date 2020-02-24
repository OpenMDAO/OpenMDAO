"""
Unit tests for the spline distribution class.
"""
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from openmdao.utils.spline_distributions import cell_centered, sine_distribution, node_centered


class SplineDistributionsTestCase(unittest.TestCase):

    def test_cell_centered(self):

        calculated_midpoints = np.array([2.1, 4.3, 6.5, 8.7, 10.9])

        dist = cell_centered(5, start=1.0, end=12.0)

        assert_array_almost_equal(calculated_midpoints, dist)

    def test_sin_distribution(self):

        calculated = np.array([0.        , 0.03015369, 0.11697778, 0.25      , 0.41317591,
                               0.58682409, 0.75      , 0.88302222, 0.96984631, 1.        ])

        dist = sine_distribution(10)

        assert_array_almost_equal(calculated, dist)

        calculated = np.array([0.14644661, 0.21321178, 0.28869087, 0.37059048, 0.45642213,
                               0.54357787, 0.62940952, 0.71130913, 0.78678822, 0.85355339])


        dist = sine_distribution(10, phase=np.pi/2.0)

        assert_array_almost_equal(calculated, dist)

    def test_node_centered(self):

        calculated_midpoints = np.array([1., 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.])

        assert_array_almost_equal(calculated_midpoints, node_centered(10, start=1.0, end=4.0))


class SplineDistributionFeatureTestCase(unittest.TestCase):

    def test_spline_distribution_example(self):

        import numpy as np

        import openmdao.api as om
        from openmdao.utils.spline_distributions import cell_centered, sine_distribution, node_centered

        x_cp = np.linspace(0., 1., 6)
        y_cp = np.array([5.0, 12.0, 14.0, 16.0, 21.0, 29.0])
        n = 20
        x = sine_distribution(n, start=0.0, end=1.0, phase=np.pi)

        prob = om.Problem()

        comp = om.SplineComp(method='akima', x_cp_val=x_cp, x_interp_val=x)
        prob.model.add_subsystem('akima1', comp)

        comp.add_spline(y_cp_name='ycp', y_interp_name='y_val', y_cp_val=y_cp)

        prob.setup(force_alloc_complex=True)
        prob.run_model()

        akima_y = np.array([[5.         , 5.32381994,  6.28062691 , 7.79410646 , 9.64169506, 11.35166363,
                             12.26525921, 12.99152288, 13.77257256, 14.58710327, 15.41289673, 16.28341046,
                             17.96032258, 20.14140712, 22.31181718, 24.40891577, 26.27368825, 27.74068235,
                             28.67782484, 29.        ]])

        assert_array_almost_equal(akima_y.flatten(), prob['akima1.y_val'].flatten())


if __name__ == '__main__':
    unittest.main()