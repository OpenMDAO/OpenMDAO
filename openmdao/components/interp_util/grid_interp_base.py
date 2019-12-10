"""Base class for grid interpolation methods."""
from __future__ import division, print_function, absolute_import

import numpy as np

from openmdao.components.interp_util.outofbounds_error import OutOfBoundsError


class GridInterpBase(object):
    """
    Interpolation on a regular grid in arbitrary dimensions.

    The data must be defined on a regular grid; the grid spacing however may be uneven. Several
    interpolation methods are supported. These are defined in the child classes. Gradients are
    provided for all interpolation methods. Gradients with respect to grid values are also
    available optionally.

    Attributes
    ----------
    bounds_error : bool
        If True, when interpolated values are requested outside of the domain of the input data,
        a ValueError is raised. If False, then the methods are allowed to extrapolate.
        Default is True (raise an exception).
    grid : tuple
        Collection of points that determine the regular grid.
    interp_method : string
        Name of interpolation method.
    training_data_gradients : bool
        Flag that tells interpolation objects wether to compute gradients with respect to the
        grid values.
    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.
    _all_gradients : ndarray
        Cache of computed gradients.
    _interp_config : dict
        Configuration object that stores the number of points required for each interpolation
        method.
    _xi : ndarray
        Cache of current evaluation point.
    """

    def __init__(self, points, values, interp_method="slinear", bounds_error=True):
        """
        Initialize instance of interpolation class.

        Parameters
        ----------
        points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
            The points defining the regular grid in n dimensions.
        values : array_like, shape (m1, ..., mn, ...)
            The data on the regular grid in n dimensions.
        interp_method : str, optional
            Name of interpolation method.
        bounds_error : bool, optional
            If True, when interpolated values are requested outside of the domain of the input
            data, a ValueError is raised. If False, then the methods are allowed to extrapolate.
            Default is True (raise an exception).
        """
        configs = self._interp_methods()
        self._all_methods, self._interp_config = configs
        if interp_method not in self._all_methods:
            all_m = ', '.join(['"' + m + '"' for m in self._all_methods])
            raise ValueError('Interpolation method "%s" is not defined. Valid methods are '
                             '%s.' % (interp_method, all_m))
        self.interp_method = interp_method
        self.bounds_error = bounds_error

        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = np.asarray(values)

        if len(points) > values.ndim:
            raise ValueError("There are %d point arrays, but values has %d "
                             "dimensions" % (len(points), values.ndim))

        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)

        if np.iscomplexobj(values[:]):
            msg = "Interpolation method '%s' does not support complex values." % interp_method
            raise ValueError(msg)

        for i, p in enumerate(points):
            n_p = len(p)
            if not np.all(np.diff(p) > 0.):
                raise ValueError("The points in dimension %d must be strictly "
                                 "ascending" % i)
            if not np.asarray(p).ndim == 1:
                raise ValueError("The points in dimension %d must be "
                                 "1-dimensional" % i)
            if not values.shape[i] == n_p:
                raise ValueError("There are %d points and %d values in "
                                 "dimension %d" % (len(p), values.shape[i], i))

        self.grid = tuple([np.asarray(p) for p in points])
        self.values = values
        self._xi = None
        self._all_gradients = None
        self.training_data_gradients = False

    def _interp_methods(self):
        """
        Method-specific settings for interpolation and for testing.

        Returns
        -------
        list
            Valid interpolation name strings.
        dict
            Configuration object that stores the number of points required for each method.
        """
        return None

    def interpolate(self, xi):
        """
        Interpolate at the sample coordinates.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data.

        Returns
        -------
        ndarray
            Value of interpolant at all sample points.
        """
        # cache latest evaluation point for gradient method's use later
        self._xi = xi

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if np.isnan(p).any():
                    raise OutOfBoundsError("One of the requested xi contains a NaN",
                                           i, np.NaN, self.grid[i][0], self.grid[i][-1])

                eps = 1e-14 * self.grid[i][-1]
                if not np.logical_and(np.all(self.grid[i][0] <= p + eps),
                                      np.all(p - eps <= self.grid[i][-1])):
                    p1 = np.where(self.grid[i][0] > p)[0]
                    p2 = np.where(p > self.grid[i][-1])[0]
                    # First violating entry is enough to direct the user.
                    violated_idx = set(p1).union(p2).pop()
                    value = p[violated_idx]
                    raise OutOfBoundsError("One of the requested xi is out of bounds",
                                           i, value, self.grid[i][0], self.grid[i][-1])
        return None

    def gradient(self, xi):
        """
        Compute the gradients at the specified point.

        The gradients are computed as the interpolation itself is performed,
        but are cached and returned separately by this method.

        If the point for evaluation differs from the point used to produce
        the currently cached gradient, the interpolation is re-performed in
        order to return the correct gradient.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at

        Returns
        -------
        gradient : ndarray of shape (..., ndim)
            Vector of gradients of the interpolated values with respect to each value in xi.
        """
        if (self._xi is None) or (not np.array_equal(xi, self._xi)):
            # If inputs have changed since last computation, then re-interpolate.
            self.interpolate(xi)

        return self._all_gradients.reshape(np.asarray(xi).shape)

    def training_gradients(self, pt):
        """
        Compute the training gradient for the vector of training points.

        Parameters
        ----------
        pt : ndarray
            Training point values.

        Returns
        -------
        ndarray
            Gradient of output with respect to training point values.
        """
        pass
