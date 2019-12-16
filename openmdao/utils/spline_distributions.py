"""Helper function to create non uniform distributions for SplineComp."""

import numpy as np


class SplineDistribution(object):
    """
    Class to provide helper functions for distribution of interpolation points.

    Attributes
    ----------
    None
    """

    def cell_centered(self, input_points, num_points):
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
        if num_points >= len(input_points):
            raise KeyError('Number of points must be less than input_points.')

        interp_grid = np.linspace(min(input_points), max(input_points), num_points + 1)

        return np.array((interp_grid[1:] + interp_grid[:-1] / 2))

    def sine_distribution(self, input_points, num_points, phase=np.pi):
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
        t_vec = np.linspace(min(input_points), max(input_points), num_points)

        return np.array(0.5 * (1.0 + np.sin(-0.5 * phase + t_vec * phase)))

    def node_centered(self, input_points, num_points):
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
        return np.linspace(min(input_points), max(input_points), num_points + 1)
