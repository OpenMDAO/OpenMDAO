"""Helper function to create non uniform distributions for SplineComp."""

import numpy as np


class SplineDistribution(object):
    """
    Class to provide helper functions for distribution of interpolation points.

    """

    def cell_centered(self, num_points, start=0.0, end=1.0):
        """
        Cell centered distribution of control points.

        Parameters
        ----------
        input_points : list or ndarray
            Control points to distribute.
        num_points : int
            Number of points to distribute.

        Returns
        -------
        ndarray
            Values to interpolate at.
        """

        interp_grid = np.linspace(start, end, num_points)

        return np.array((interp_grid[1:] + interp_grid[:-1] / 2))

    def sine_distribution(self, num_points, start=0.0, end=1.0, phase=np.pi):
        """
        Sine distribution of control points.

        Parameters
        ----------
        input_points : list or ndarray
            Control points to distribute.
        num_points : int
            Number of points to distribute.
        phase : float
            Phase of the sine wave

        Returns
        -------
        ndarray
            Values to interpolate at.
        """
        t_vec = np.linspace(start, end, num_points)

        return np.array(0.5 * (1.0 + np.sin(-0.5 * phase + t_vec * phase)))

    def node_centered(self, num_points, start=0.0, end=1.0):
        """
        Distribute control points.

        Parameters
        ----------
        input_points : list or ndarray
            Control points to distribute.
        num_points : int
            Number of points to distribute.

        Returns
        -------
        ndarray
            Values to interpolate at.
        """
        return np.linspace(start, end, num_points + 1)
