"""Grid interpolation using scipy splines."""
from __future__ import division, print_function, absolute_import

from six.moves import range, zip

from scipy import __version__ as scipy_version
try:
    from scipy.interpolate._bsplines import make_interp_spline as _make_interp_spline
except ImportError:
    def _make_interp_spline(*args, **kwargs):
        msg = "'MetaModelStructuredComp' requires scipy>=0.19, but the currently" \
              " installed version is %s." % scipy_version
        raise RuntimeError(msg)

import numpy as np
from scipy.interpolate.interpnd import _ndim_coords_from_arrays

from openmdao.components.structured_metamodel_util.grid_interp_base import GridInterpBase
from openmdao.components.structured_metamodel_util.outofbounds_error import OutOfBoundsError


class ScipyGridInterp(GridInterpBase):
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

    def _interp_methods(self):
        """
        Method-specific settings for interpolation and for testing.

        Returns
        -------
        list
            Valid interpolation name strings.
        dict
            Configuration object that stores limitations of each interpolation
            method.
        """
        interpolator_configs = {
            "slinear": 1,
            "cubic": 3,
            "quintic": 5,
        }

        all_methods = list(interpolator_configs.keys())

        return all_methods, interpolator_configs

    def interp_method(self):
        """
        Return a list of valid interpolation method names.

        Returns
        -------
        list
            Valid interpolation name strings.
        """
        return ['slinear', 'cubic', 'quintic']

    def interpolate(self, xi, interp_method=None, compute_gradients=True):
        """
        Interpolate at the sample coordinates.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        interp_method : str, optional
            The method of interpolation to perform. Supported are 'slinear', 'cubic', and
            'quintic'. Default is None, which will use the method defined at the construction
            of the interpolation object instance.
        compute_gradients : bool, optional
            When True, gradients should be computed and cached.

        Returns
        -------
        array_like
            Value of interpolant at all sample points.
        """
        # cache latest evaluation point for gradient method's use later
        self._xi = xi

        interp_method = self.interp_method if interp_method is None else interp_method
        if interp_method not in self._all_methods:
            all_m = ', '.join(['"' + m + '"' for m in self._all_methods])
            raise ValueError('Interpolation method "%s" is not defined. Valid methods are '
                             '%s.' % (interp_method, all_m))

        ndim = len(self.grid)
        self.ndim = ndim
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             "%d, but this RegularGridInterp has "
                             "dimension %d" % (xi.shape[1], ndim))

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if np.isnan(p).any():
                    raise OutOfBoundsError("One of the requested xi contains a NaN",
                                           i, np.NaN, self.grid[i][0], self.grid[i][-1])

                if not np.logical_and(np.all(self.grid[i][0] <= p),
                                      np.all(p <= self.grid[i][-1])):
                    p1 = np.where(self.grid[i][0] > p)[0]
                    p2 = np.where(p > self.grid[i][-1])[0]
                    # First violating entry is enough to direct the user.
                    violated_idx = set(p1).union(p2).pop()
                    value = p[violated_idx]
                    raise OutOfBoundsError("One of the requested xi is out of bounds",
                                           i, value, self.grid[i][0], self.grid[i][-1])

        _, out_of_bounds = self._find_indices(xi.T)

        ki = self._ki
        interp_method = self.interp_method if interp_method is None else interp_method
        if interp_method != self.interp_method:
            # re-validate dimensions vs spline order

            k = self._interp_config[interp_method]
            ki = []
            for i, p in enumerate(self.grid):
                n_p = len(p)
                ki.append(k)
                if n_p <= k:
                    ki[-1] = n_p - 1

        interpolator = _make_interp_spline
        result = self._evaluate_splines(self.values[:].T,
                                        xi,
                                        interpolator,
                                        interp_method,
                                        ki,
                                        compute_gradients=compute_gradients)

        if not self.bounds_error and self.fill_value is not None:
            result[out_of_bounds] = self.fill_value

        return result.reshape(xi_shape[:-1] + self.values.shape[ndim:])

    def _evaluate_splines(self, data_values, xi, interpolator, interp_method,
                          ki, compute_gradients=True):
        """
        Perform interpolation using the given interpolator.

        Parameters
        ----------
        data_values : array_like
            The data on the regular grid in n dimensions.
        xi : ndarray
            The coordinates to sample the gridded data at
        interpolator : <scipy.interpolate.BSpline>
            A BSpline object that is used for interpolation.
        interp_method : str, optional
            The interpolation method to perform. Supported are 'slinear', 'cubic', and
            'quintic'. Default is None, which will use the order defined at the construction
            of the interpolation object instance.
        ki : list
            List of spline interpolation orders.
        compute_gradients : bool, optional
            If a spline interpolation method is chosen, this determines whether gradient
            calculations should be made and cached. Default is True.

        Returns
        -------
        array_like
            Value of interpolant at all sample points.
        """
        # for spline based methods

        # requires floating point input
        xi = xi.astype(np.float)

        # ensure xi is 2D list of points to evaluate
        if xi.ndim == 1:
            xi = xi.reshape((1, xi.size))
        m, n = xi.shape

        # create container arrays for output and gradients
        result = np.empty(m)
        if compute_gradients:
            all_gradients = np.empty_like(xi)

        # Non-stationary procedure: difficult to vectorize this part entirely
        # into numpy-level operations. Unfortunately this requires explicit
        # looping over each point in xi.

        # can at least vectorize the first pass across all points in the
        # last variable of xi. This provides one dimension of the entire
        # gradient output array.
        i = n - 1
        first_values, first_derivs = self._do_spline_fit(interpolator,
                                                         self.grid[i],
                                                         data_values,
                                                         xi[:, i],
                                                         ki[i],
                                                         compute_gradients)

        # the rest of the dimensions have to be on a per point-in-xi basis
        for j, x in enumerate(xi):
            gradient = np.empty_like(x)
            values = data_values[:]

            # Main process: Apply 1D interpolate in each dimension
            # sequentially, starting with the last dimension. These are then
            # "folded" into the next dimension in-place.
            for i in range(n - 1, 0, -1):
                if i == n - 1:
                    values = first_values[j]
                    if compute_gradients:
                        local_derivs = first_derivs[j]
                else:
                    # Interpolate and collect gradients for each 1D in this
                    # last dimensions. This collapses each 1D sequence into a
                    # scalar.
                    values, local_derivs = self._do_spline_fit(interpolator,
                                                               self.grid[i],
                                                               values,
                                                               x[i],
                                                               ki[i],
                                                               compute_gradients)

                # Chain rule: to compute gradients of the output w.r.t. xi
                # across the dimensions, apply interpolation to the collected
                # gradients. This is equivalent to multiplication by
                # dResults/dValues at each level.
                if compute_gradients:
                    gradient[i] = self._evaluate_splines(local_derivs,
                                                         x[: i],
                                                         interpolator,
                                                         interp_method,
                                                         ki,
                                                         compute_gradients=False)

            # All values have been folded down to a single dimensional array
            # compute the final interpolated results, and gradient w.r.t. the
            # first dimension
            output_value, gradient[0] = self._do_spline_fit(interpolator,
                                                            self.grid[0],
                                                            values,
                                                            x[0],
                                                            ki[0],
                                                            compute_gradients)

            if compute_gradients:
                all_gradients[j] = gradient
            result[j] = output_value

        # Cache the computed gradients for return by the gradient method
        if compute_gradients:
            self._all_gradients = all_gradients
            # indicate what order was used to compute these
            self._g_order = interp_method
        return result

    def _do_spline_fit(self, interpolator, x, y, pt, k, compute_gradients):
        """
        Do a single interpolant call, and compute the gradient if needed.

        Parameters
        ----------
        interpolator : <scipy.interpolate.BSpline>
            A BSpline object that is used for interpolation.
        x : array_like, shape (n,)
            Abscissas.
        y : array_like, shape (n, ...)
            Ordinates.
        pt : array_like
            Points to evaluate the spline at.
        k : float
            Spline interpolation order.
        compute_gradients : bool
            If a spline interpolation method is chosen, this determines whether gradient
            calculations should be made and cached.

        Returns
        -------
        array_like
            Value of interpolant at point of interest.
        None or array_like, optional
            Value of gradient of interpolant at point of interest.
        """
        local_interp = interpolator(x, y, k=k, axis=0)
        values = local_interp(pt)
        local_derivs = None
        if compute_gradients:
            local_derivs = local_interp(pt, 1)
        return values, local_derivs

    def _find_indices(self, xi):
        """
        Find the correct search indices for table lookups.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at.

        Returns
        -------
        list of ndarray
            Indices
        list of ndarray
            Norm of the distance to the lower edge
        list of bool
            Out of bounds flags.
        """
        # find relevant edges between which xi are situated
        indices = []
        # check for out of bounds xi
        out_of_bounds = np.zeros((xi.shape[1]), dtype=bool)
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            i = np.searchsorted(grid, x) - 1
            i[i < 0] = 0
            i[i > grid.size - 2] = grid.size - 2
            indices.append(i)

            if not self.bounds_error:
                out_of_bounds += x < grid[0]
                out_of_bounds += x > grid[-1]
        return indices, out_of_bounds

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
            gradient vector of the gradients of the interpolated values with
            respect to each value in xi
        """
        # Determine if the needed gradients have been cached already
        if not interp_method:
            interp_method = self.interp_method

        if (self._xi is None) or \
                (not np.array_equal(xi, self._xi)) or \
                (interp_method != self._g_order):
            # if not, compute the interpolation to get the gradients
            self.interpolate(xi, interp_method=interp_method)
        gradients = self._all_gradients
        gradients = gradients.reshape(np.asarray(xi).shape)
        return gradients

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
        for i, axis in enumerate(self.grid):
            e_i = np.eye(axis.size)
            interp = _make_interp_spline(axis, e_i, k=self._ki[i], axis=0)
            if i == 0:
                val = interp(pt[i])
            else:
                val = np.outer(val, interp(pt[i]))

        return val
