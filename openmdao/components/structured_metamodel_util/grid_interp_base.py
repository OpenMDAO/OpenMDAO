"""Base class for grid interpolation methods."""
from __future__ import division, print_function, absolute_import

import numpy as np


class GridInterpBase(object):
    """
    Interpolation on a regular grid in arbitrary dimensions.

    The data must be defined on a regular grid; the grid spacing however may be uneven. First,
    third and fifth order spline interpolation are supported. After setting up the interpolator
    object, the interpolation order (*slinear*, *cubic*, and *quintic*) may be chosen at each
    evaluation. Additionally, gradients are provided for the spline interpolation methods.

    Attributes
    ----------
    bounds_error : bool
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.
        Default is True (raise an exception).
    fill_value : float
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated. Note that gradient values will always be
        extrapolated rather than set to the fill_value if bounds_error=False
        for any points outside of the interpolation domain.
        Default is `np.nan`.
    grid : tuple
        Collection of points that determine the regular grid.
    order : string
        Name of interpolation order.
    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.
    _all_gradients : ndarray
        Cache of computed gradients.
    _g_order : string
        Name of interpolation order used to compute the last gradient.
    _interp_config : dict
        Configuration object that stores limitations of each interpolation
        order.
    _ki : list
        Interpolation order to be used in each dimension.
    _spline_dim_error : bool
        If spline_dim_error=True and an order `k` spline interpolation method
        is used, then if any dimension has fewer points than `k` + 1, an error
        will be raised. If spline_dim_error=False, then the spline interpolant
        order will be reduced as needed on a per-dimension basis. Default
        is True (raise an exception).
    _xi : ndarray
        Current evaluation point.
    """

    def __init__(self, points, values, order="slinear", bounds_error=True,
                 fill_value=np.nan, spline_dim_error=True):
        """
        Initialize instance of interpolation class.

        Parameters
        ----------
        points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
            The points defining the regular grid in n dimensions.
        values : array_like, shape (m1, ..., mn, ...)
            The data on the regular grid in n dimensions.
        order : str, optional
            The order of interpolation to perform. Supported are 'slinear',
            'cubic',  and 'quintic'. This parameter will become
            the default for the object's interpolate order. Default is "linear".
        bounds_error : bool, optional
            If True, when interpolated values are requested outside of the
            domain of the input data, a ValueError is raised.
            If False, then `fill_value` is used.
            Default is True (raise an exception).
        fill_value : number, optional
            If provided, the value to use for points outside of the
            interpolation domain. If None, values outside
            the domain are extrapolated. Note that gradient values will always be
            extrapolated rather than set to the fill_value if bounds_error=False
            for any points outside of the interpolation domain.
            Default is `np.nan`.
        spline_dim_error : bool, optional
            If spline_dim_error=True and an order `k` spline interpolation method
            is used, then if any dimension has fewer points than `k` + 1, an error
            will be raised. If spline_dim_error=False, then the spline interpolant
            order will be reduced as needed on a per-dimension basis. Default
            is True (raise an exception).
        """
        configs = self._interp_orders()
        self._all_orders, self._interp_config = configs
        if order not in self._all_orders:
            all_m = ', '.join(['"' + m + '"' for m in self._all_orders])
            raise ValueError('Order "%s" is not defined. Valid order are '
                             '%s.' % (order, all_m))
        self.order = order
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
            raise ValueError("order '%s' does not support complex values." % order)

        self.fill_value = fill_value
        if fill_value is not None:
            fill_value_dtype = np.asarray(fill_value).dtype
            if (hasattr(values, 'dtype') and not
                    np.can_cast(fill_value_dtype, values.dtype,
                                casting='same_kind')):
                raise ValueError("fill_value must be either 'None' or "
                                 "of a type compatible with values")

        k = self._interp_config[order]
        self._ki = []
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

            self._ki.append(k)
            if n_p <= k:
                if not spline_dim_error:
                    self._ki[-1] = n_p - 1
                else:
                    raise ValueError("There are %d points in dimension %d,"
                                     " but order %s requires at least %d "
                                     "points per "
                                     "dimension."
                                     "" % (n_p, i, order, k + 1))

        self.grid = tuple([np.asarray(p) for p in points])
        self.values = values
        self._xi = None
        self._all_gradients = None
        self._spline_dim_error = spline_dim_error
        self._g_order = None

    def _interp_orders(self):
        """
        Method-specific settings for interpolation and for testing.

        Returns
        -------
        list
            Valid interpolation name strings.
        dict
            Configuration object that stores limitations of each interpolation
            order.
        """
        interpolator_configs = {
            "slinear": 1,
            "cubic": 3,
            "quintic": 5,
        }

        all_orders = list(interpolator_configs.keys())

        return all_orders, interpolator_configs

    def order(self):
        """
        Return a list of valid interpolation order names.

        Returns
        -------
        list
            Valid interpolation name strings.
        """
        return ['slinear', 'cubic', 'quintic']

    def interpolate(self, xi, order=None, compute_gradients=True):
        """
        Interpolate at the sample coordinates.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        order : str, optional
            The order of interpolation to perform. Supported are 'slinear', 'cubic', and
            'quintic'. Default is None, which will use the order defined at the construction
            of the interpolation object instance.
        compute_gradients : bool, optional
            If a spline interpolation method is chosen, this determines whether gradient
            calculations should be made and cached. Default is True.

        Returns
        -------
        array_like
            Value of interpolant at all sample points.
        """
        pass

    def gradient(self, xi, order=None):
        """
        Return the computed gradients at the specified point.

        The gradients are computed as the interpolation itself is performed,
        but are cached and returned separately by this method.

        If the point for evaluation differs from the point used to produce
        the currently cached gradient, the interpolation is re-performed in
        order to return the correct gradient.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        order : str, optional
            The order of interpolation to perform. Supported are 'slinear',
            'cubic', and 'quintic'. Default is None, which will use the order
            defined at the construction of the interpolation object instance.

        Returns
        -------
        gradient : ndarray of shape (..., ndim)
            gradient vector of the gradients of the interpolated values with
            respect to each value in xi
        """
        pass

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