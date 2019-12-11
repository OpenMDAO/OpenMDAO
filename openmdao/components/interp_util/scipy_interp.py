"""Grid interpolation using scipy splines."""
from __future__ import division, print_function, absolute_import

from six.moves import range

from scipy import __version__ as scipy_version
try:
    from scipy.interpolate._bsplines import make_interp_spline as _make_interp_spline
except ImportError:
    def _make_interp_spline(*args, **kwargs):
        msg = "'MetaModelStructuredComp' requires scipy>=0.19, but the currently" \
              " installed version is %s." % scipy_version
        raise RuntimeError(msg)

import numpy as np

from openmdao.components.interp_util.grid_interp_base import GridInterpBase


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
        If True, when interpolated values are requested outside of the domain of the input data,
        a ValueError is raised. If False, then the methods are allowed to extrapolate.
        Default is True (raise an exception).
    grid : tuple
        Collection of points that determine the regular grid.
    order : string
        Name of interpolation order.
    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.
    _all_gradients : ndarray
        Cache of computed gradients.
    _interp_config : dict
        Configuration object that stores the number of points required for each interpolation
        method.
    _ki : list
        Interpolation order to be used in each dimension.
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
        interp_method : str, optional
            Name of interpolation method.
        bounds_error : bool, optional
            If True, when interpolated values are requested outside of the domain of the input
            data, a ValueError is raised. If False, then the methods are allowed to extrapolate.
            Default is True (raise an exception).
        **kwargs : dict
            Interpolator-specific options to pass onward.
        """
        if kwargs:
            raise KeyError("SciPy interpolator does not support {} options.".format([x for x in kwargs]))

        super(ScipyGridInterp, self).__init__(points, values, interp_method=interp_method,
                                              bounds_error=bounds_error)

        # ScipyGridInterp supports automatic order reduction.
        self._ki = []
        # Order is the number of required points minus one.
        k = self._interp_config[interp_method] - 1
        for p in points:
            n_p = len(p)
            self._ki.append(k)
            if n_p <= k:
                self._ki[-1] = n_p - 1

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
        interpolator_configs = {
            "slinear": 2,
            "cubic": 4,
            "quintic": 6,
        }

        all_methods = list(interpolator_configs.keys())
        return all_methods, interpolator_configs

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
        super(ScipyGridInterp, self).interpolate(xi)

        result = self._evaluate_splines(self.values[:].T, xi, self._ki)

        return result

    def _evaluate_splines(self, data_values, xi, ki, compute_gradients=True):
        """
        Perform interpolation using the scipy interpolator.

        Parameters
        ----------
        data_values : array_like
            The data on the regular grid in n dimensions.
        xi : ndarray
            The coordinates to sample the gridded data at
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
        first_values, first_derivs = self._do_spline_fit(self.grid[i],
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
                    values, local_derivs = self._do_spline_fit(self.grid[i],
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
                                                         ki,
                                                         compute_gradients=False)

            # All values have been folded down to a single dimensional array
            # compute the final interpolated results, and gradient w.r.t. the
            # first dimension
            output_value, gradient[0] = self._do_spline_fit(self.grid[0],
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

        return result

    def _do_spline_fit(self, x, y, pt, k, compute_gradients):
        """
        Do a single interpolant call, and compute the gradient if needed.

        Parameters
        ----------
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
        local_interp = _make_interp_spline(x, y, k=k, axis=0)
        values = local_interp(pt)
        local_derivs = None
        if compute_gradients:
            local_derivs = local_interp(pt, 1)
        return values, local_derivs

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
