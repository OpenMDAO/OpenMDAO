"""
Base class for interpolation methods that calculate values for each dimension independently.

Based on Tables in NPSS, and was added to bridge the gap between some of the slower scipy
implementations.
"""
from __future__ import division, print_function, absolute_import
from six.moves import range

import numpy as np

from openmdao.components.interp_util.interp_akima import InterpAkima
from openmdao.components.interp_util.interp_cubic import InterpCubic
from openmdao.components.interp_util.interp_lagrange2 import InterpLagrange2
from openmdao.components.interp_util.interp_lagrange3 import InterpLagrange3
from openmdao.components.interp_util.interp_scipy import InterpScipy
from openmdao.components.interp_util.interp_slinear import InterpLinear

from openmdao.components.interp_util.outofbounds_error import OutOfBoundsError

INTERP_METHODS = {
    'slinear': InterpLinear,
    'lagrange2': InterpLagrange2,
    'lagrange3': InterpLagrange3,
    'cubic': InterpCubic,
    'akima': InterpAkima,
    'scipy_cubic': InterpScipy,
    'scipy_slinear': InterpScipy,
    'scipy_quintic': InterpScipy,
}


class InterpND(object):
    """
    Interpolation on a regular grid of arbitrary dimensions.

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
    training_data_gradients : bool
        Flag that tells interpolation objects wether to compute gradients with respect to the
        grid values.
    table : <InterpTable>
        Table object that contains algorithm that performs the interpolation.
    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.
    _d_dx : ndarray
        Cache of computed gradients with respect to evaluation point.
    _d_dgrid : ndarray
        Cache of computed gradients with respect to grid.
    _d_dvalues : ndarray
        Cache of computed gradients with respect to table values.
    _interp : class
        Class specified as interpolation algorithm, used to regenerate if needed.
    _interp_config : dict
        Configuration object that stores the number of points required for each interpolation
        method.
    _interp_options : dict
        Dictionary of cached interpolator-specific options.
    _xi : ndarray
        Cache of current evaluation point.
    """

    def __init__(self, points, values, interp_method="slinear", bounds_error=True, **kwargs):
        """
        Initialize instance of interpolation class.

        Parameters
        ----------
        points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
            The points defining the regular grid in n dimensions.
        values : array_like, shape (m1, ..., mn, ...)
            The data on the regular grid in n dimensions.
        interp_method : str or list of str, optional
            Name of interpolation method(s).
        bounds_error : bool, optional
            If True, when interpolated values are requested outside of the domain of the input
            data, a ValueError is raised. If False, then the methods are allowed to extrapolate.
            Default is True (raise an exception).
        **kwargs : dict
            Interpolator-specific options to pass onward.
        """
        if interp_method not in INTERP_METHODS:
            all_m = ', '.join(['"' + m + '"' for m in INTERP_METHODS])
            raise ValueError('Interpolation method "%s" is not defined. Valid methods are '
                             '%s.' % (interp_method, all_m))
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
        self._d_dx = None
        self._d_dgrid = None
        self._d_dvalues = None
        self.training_data_gradients = False

        # Cache spline coefficients.
        interp = INTERP_METHODS[interp_method]

        if interp_method.startswith('scipy'):
            kwargs['interp_method'] = interp_method

        table = interp(self.grid, self.values, interp, **kwargs)
        table.check_config()
        self.table = table
        self._interp = interp
        self._interp_options = kwargs

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

        if self.training_data_gradients:
            # If the table grid or values are component inputs, then we need to create a new table
            # each iteration.
            interp = self._interp
            self.table = interp(self.grid, self.values, interp, **self._interp_options)

        table = self.table
        if table._vectorized:
            result, derivs, d_values, d_grid = table.evaluate_vectorized(xi)

        else:
            xi = np.atleast_2d(xi)
            n_nodes, nx = xi.shape
            result = np.empty((n_nodes, ), dtype=xi.dtype)
            derivs = np.empty((n_nodes, nx), dtype=xi.dtype)

            # TODO: it might be possible to vectorize over n_nodes.
            for j in range(n_nodes):
                val, d_x, d_values, d_grid = table.evaluate(xi[j, :])
                result[j] = val
                derivs[j, :] = d_x.flatten()

        # Cache derivatives
        self._d_dx = derivs

        return result

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

        return self._d_dx.reshape(np.asarray(xi).shape)

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
        grid = self.grid
        interp = self._interp

        if self.table._vectorized:
            return self.table.training_gradients(pt)

        else:
            for i, axis in enumerate(self.grid):
                ngrid = axis.size
                values = np.zeros(ngrid)
                deriv_i = np.zeros(ngrid)

                for j in range(ngrid):
                    values[j] = 1.0
                    table = interp([grid[i]], values, self._interp, **self._interp_options)
                    deriv_i[j], _, _, _ = table.evaluate(pt[i:i + 1])
                    values[j] = 0.0

                if i == 0:
                    deriv_running = deriv_i.copy()
                else:
                    deriv_running = np.outer(deriv_running, deriv_i)

        return deriv_running
