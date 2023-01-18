"""Grid interpolation using scipy splines."""

from scipy import __version__ as scipy_version
try:
    from scipy.interpolate._bsplines import make_interp_spline as _make_interp_spline
except ImportError:
    def _make_interp_spline(*args, **kwargs):
        msg = "'MetaModelStructuredComp' requires scipy>=0.19, but the currently" \
              " installed version is %s." % scipy_version
        raise RuntimeError(msg)

import numpy as np

from openmdao.components.interp_util.interp_algorithm import InterpAlgorithm
from openmdao.utils.options_dictionary import OptionsDictionary

SCIPY_ORDERS = {
    "scipy_slinear": 2,
    "scipy_cubic": 4,
    "scipy_quintic": 6,
}


class InterpScipy(InterpAlgorithm):
    """
    Interpolation on a regular grid in arbitrary dimensions.

    The data must be defined on a regular grid; the grid spacing however may be uneven. First,
    third and fifth order spline interpolation are supported. After setting up the interpolator
    object, the interpolation order (*slinear*, *cubic*, and *quintic*) may be chosen at each
    evaluation. Additionally, gradients are provided for the spline interpolation methods.

    Parameters
    ----------
    grid : tuple(ndarray)
        Tuple containing x grid locations for this dimension and all subtable dimensions.
    values : ndarray
        Array containing the table values for all dimensions.
    interp : class
        Interpolation class to be used for subsequent table dimensions.
    **kwargs : dict
        Interpolator-specific options to pass onward.

    Attributes
    ----------
    _d_dx : ndarray
        Cache of computed gradients with respect to evaluation point.
    _interp_config : dict
        Configuration object that stores the number of points required for each interpolation
        method.
    _ki : list
        Interpolation order to be used in each dimension.
    _supports_d_dvalues : bool
        If True, this algorithm can compute the derivatives with respect to table values.
    _xi : ndarray
        Cache of current evaluation point.
    """

    def __init__(self, grid, values, interp=None, **kwargs):
        """
        Initialize table and subtables.
        """
        self.options = OptionsDictionary(parent_name=type(self).__name__)
        self.initialize()
        self.options.update(kwargs)

        self._vectorized = True

        interp_method = self.options['interp_method']
        self._name = interp_method
        self._supports_d_dvalues = True
        self._full_slice = None

        self.grid = grid
        self.values = values

        # InterpScipy supports automatic order reduction.
        self._ki = []

        # Order is the number of required points minus one.
        k = SCIPY_ORDERS[interp_method] - 1
        for p in grid:
            n_p = len(p)
            self._ki.append(k)
            if n_p <= k:
                self._ki[-1] = n_p - 1

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('interp_method', default='scipy_slinear',
                             values=["scipy_slinear", "scipy_cubic", "scipy_quintic"],
                             desc='Interpolation method to use for scipy.')

    def check_config(self):
        """
        Verify that we have enough points for this interpolation algorithm.
        """
        # Scipy supports automatic order reduction, so don't raise an error.
        # TODO - Make the auto order reduction an option so it can be turned off.
        pass

    def evaluate_vectorized(self, x):
        """
        Interpolate across all table dimensions for all requested samples.

        Parameters
        ----------
        x : ndarray
            The coordinates to sample the gridded data at. First array element is the point to
            interpolate here. Remaining elements are interpolated on sub tables.

        Returns
        -------
        ndarray
            Interpolated values.
        ndarray
            Derivative of interpolated values with respect to this independents.
        ndarray
            Derivative of interpolated values with respect to values.
        ndarray
            Derivative of interpolated values with respect to grid.
        """
        result, d_dx, d_values, d_grid = self.interpolate(x)

        return result, d_dx, d_values, d_grid

    def interpolate(self, x, idx=None, slice_idx=None):
        """
        Compute the interpolated value over this grid dimension.

        This method must be defined by child classes.

        Parameters
        ----------
        x : ndarray
            The coordinates to sample the gridded data at. First array element is the point to
            interpolate here. Remaining elements are interpolated on sub tables.
        idx : int
            Interval index for x.
        slice_idx : list of <slice>
            Slice object containing indices of data points requested by parent interpolating
            tables.

        Returns
        -------
        ndarray
            Interpolated values.
        ndarray
            Derivative of interpolated values with respect to this independent and child
            independents.
        ndarray
            Derivative of interpolated values with respect to values for this and subsequent table
            dimensions.
        ndarray
            Derivative of interpolated values with respect to grid for this and subsequent table
            dimensions.
        """
        result = self._evaluate_splines(self.values[:].T, x, self._ki)

        return result, self._d_dx, None, None

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
        xi = xi.astype(float)

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
            self._d_dx = all_gradients
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
