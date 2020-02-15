"""Helper function to create non uniform distributions for SplineComp."""

import numpy as np


def cell_centered(num_cells, start=0.0, end=1.0):
    """
    Cell centered distribution of control points.

    Parameters
    ----------
    num_cells : int
        Number of cells.
    start : int or float
        Minimum value to interpolate at.
    end : int or float
        Maximum value to interpolate at.

    Returns
    -------
    ndarray
        Values to interpolate at.
    """
    interp_grid = np.linspace(start, end, num=num_cells + 1)

    return np.array(0.5 * (interp_grid[1:] + interp_grid[:-1]))


def sine_distribution(num_points, start=0.0, end=1.0, phase=np.pi):
    """
    Sine distribution of control points.

    Parameters
    ----------
    num_points : int
        Number of points to predict at.
    start : int or float
        Minimum value to interpolate at.
    end : int or float
        Maximum value to interpolate at.
    phase : float
        Phase of the sine wave

    Returns
    -------
    ndarray
        Values to interpolate at.
    """
    t_vec = np.linspace(start, end, num_points)

    return np.array(0.5 * (1.0 + np.sin(-0.5 * phase + t_vec * phase)))


def node_centered(num_points, start=0.0, end=1.0):
    """
    Distribute control points.

    Parameters
    ----------
    num_points : int
        Number of points to predict.
    start : int or float
        Minimum value to interpolate at.
    end : int or float
        Maximum value to interpolate at.

    Returns
    -------
    ndarray
        Values to interpolate at.
    """
    return np.linspace(start, end, num_points + 1)
