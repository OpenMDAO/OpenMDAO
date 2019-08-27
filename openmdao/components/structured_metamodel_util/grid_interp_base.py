"""Base class for grid interpolation methods."""
from __future__ import division, print_function, absolute_import

import numpy as np


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
        a ValueError is raised. If False, then `fill_value` is used.
        Default is True (raise an exception).
    fill_value : float
        If provided, the value to use for points outside of the interpolation domain. If None,
        values outside the domain are extrapolated. Note that gradient values will always be
        extrapolated rather than set to the fill_value if bounds_error=False for any points
        outside of the interpolation domain.
        Default is `np.nan`.
    grid : tuple
        Collection of points that determine the regular grid.
    interp_method : string
        Name of interpolation method.
    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.
    _all_gradients : ndarray
        Cache of computed gradients.
    _g_method : string
        Name of interpolation method used to compute the last gradient.
    _interp_config : dict
        Configuration object that stores the number of points required for each interpolation
        method.
    _ki : list
        Interpolation order to be used in each dimension.
    _spline_dim_error : bool
        If spline_dim_error=True and an order `k` spline interpolation method
        is used, then if any dimension has fewer points than `k` + 1, an error
        will be raised. If spline_dim_error=False, then the spline interpolant
        order will be reduced as needed on a per-dimension basis. Default
        is True (raise an exception).
    _xi : ndarray
        Cache of current evaluation point.
    """

    def __init__(self, points, values, interp_method="slinear", bounds_error=True,
                 fill_value=np.nan, spline_dim_error=True):
        """
        Initialize instance of interpolation class.

        Parameters
        ----------
        points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
            The points defining the regular grid in n dimensions.
        values : array_like, shape (m1, ..., mn, ...)
            The data on the regular grid in n dimensions.
        interp_method : str, optional
            The interpolation method to perform. Supported are 'slinear',
            'cubic',  and 'quintic'. This parameter will become
            the default for the object's ``interpolate`` method. Default is "slinear".
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

        self.fill_value = fill_value
        if fill_value is not None:
            fill_value_dtype = np.asarray(fill_value).dtype
            if (hasattr(values, 'dtype') and not
                    np.can_cast(fill_value_dtype, values.dtype,
                                casting='same_kind')):
                raise ValueError("fill_value must be either 'None' or "
                                 "of a type compatible with values")

        k = self._interp_config[interp_method]
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
                                     " but method %s requires at least %d "
                                     "points per "
                                     "dimension."
                                     "" % (n_p, i, interp_method, k + 1))

        self.grid = tuple([np.asarray(p) for p in points])
        self.values = values
        self._xi = None
        self._all_gradients = None
        self._spline_dim_error = spline_dim_error
        self.training_data_gradients = False

    def interpolate(self, xi, interp_method=None, compute_gradients=True):
        """
        Interpolate at the sample coordinates.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        interp_method : str, optional
            The interpolation method to perform. Supported are 'slinear', 'cubic', and
            'quintic'. Default is None, which will use the method defined at the construction
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

    def gradient(self, xi, interp_method=None):
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
        interp_method : str, optional
            The interpolation method to perform. Supported are 'slinear',
            'cubic', and 'quintic'. Default is None, which will use the method
            defined at the construction of the interpolation object instance.

        Returns
        -------
        gradient : ndarray of shape (..., ndim)
            Vector of gradients of the interpolated values with respect to each value in xi.
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