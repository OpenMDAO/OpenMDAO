"""
Base class for interpolation methods that work on a semi-structured grid.
"""
import numpy as np

from openmdao.components.interp_util.interp_akima import InterpAkimaSemi
from openmdao.components.interp_util.interp_lagrange2 import InterpLagrange2Semi
from openmdao.components.interp_util.interp_lagrange3 import InterpLagrange3Semi
from openmdao.components.interp_util.interp_slinear import InterpLinearSemi
from openmdao.components.interp_util.outofbounds_error import OutOfBoundsError


INTERP_METHODS = {
    'slinear': InterpLinearSemi,
    'lagrange2': InterpLagrange2Semi,
    'lagrange3': InterpLagrange3Semi,
    'akima': InterpAkimaSemi,
}

TABLE_METHODS = [method for method in INTERP_METHODS.keys()]


class InterpNDSemi(object):
    """
    Interpolation on a semi-structured grid of arbitrary dimensions.

    Parameters
    ----------
    points : ndarray
        The points defining the semi-structured grid in n dimensions.
    values : ndarray
        Values (or initial values) for all points in the semi-structured grid.
    method : str
        Name of interpolation method.
    extrapolate : bool
        When False, raise an exception for any point that is extrapolated.
        When True, raise a warning for any point that is extrapolated.
        Note, default is True because semi-structured grids are often sparse in at least one
        dimension.
    **kwargs : dict
        Interpolator-specific options to pass onward.

    Attributes
    ----------
    extrapolate : bool
        When False, raise an exception for any point that is extrapolated.
        When True, raise a warning for any point that is extrapolated.
        Default is True (just warn).
    grid : tuple
        Collection of points that determine the regular grid.
    table : <InterpTable>
        Table object that contains algorithm that performs the interpolation.
    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.
    _compute_d_dvalues : bool
        When set to True, compute gradients with respect to the grid values.
    _d_dx : ndarray
        Cache of computed gradients with respect to evaluation point.
    _d_dvalues : ndarray
        Cache of computed gradients with respect to table values.
    _interp : class
        Class specified as interpolation algorithm, used to regenerate if needed.
    _interp_options : dict
        Dictionary of cached interpolator-specific options.
    _xi : ndarray
        Cache of current evaluation point.
    """

    def __init__(self, points, values, method="slinear", extrapolate=True, **kwargs):
        """
        Initialize an InterpNDSemi object.

        This object can be setup and used to interpolate on a curve or multi-dimensional table.
        """
        if not isinstance(method, str):
            msg = "Argument 'method' should be a string."
            raise ValueError(msg)
        elif method not in INTERP_METHODS:
            all_m = ', '.join(['"' + m + '"' for m in INTERP_METHODS])
            raise ValueError('Interpolation method "%s" is not defined. Valid methods are '
                             '%s.' % (method, all_m))

        if hasattr(values, 'dtype') and hasattr(values, 'astype'):
            if not np.issubdtype(values.dtype, np.inexact):
                values = values.astype(float)

        if len(points) > len(values):
            raise ValueError(f"There are {len(points)} point arrays, but {len(values)} values.")

        if np.iscomplexobj(values[:]):
            msg = "Interpolation method '%s' does not support complex values." % method
            raise ValueError(msg)

        self.grid = points
        self.values = values
        self.extrapolate = extrapolate

        self._xi = None
        self._d_dx = None
        self._d_dvalues = None
        self._compute_d_dvalues = False

        # Cache spline coefficients.
        interp = INTERP_METHODS[method]

        table = interp(self.grid, values, interp, **kwargs)
        table.check_config()

        self.table = table
        self._interp = interp
        self._interp_options = kwargs

    def interpolate(self, x, compute_derivative=False):
        """
        Interpolate at the sample coordinates.

        Parameters
        ----------
        x : ndarray or tuple
            Locations to interpolate.
        compute_derivative : bool
            Set to True to compute derivatives with respect to x.

        Returns
        -------
        ndarray
            Value of interpolant at all sample points.
        ndarray
            Value of derivative of interpolated output with respect to input x. (Only when
            compute_derivative is True).
        """
        xnew = self._interpolate(x)

        if compute_derivative:
            return xnew, self._d_dx
        else:
            return xnew

    def _interpolate(self, xi):
        """
        Interpolate at the sample coordinates.

        This method is called from OpenMDAO, and is not meant for standalone use.

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

        if not self.extrapolate:
            for i, p in enumerate(xi.T):
                if np.isnan(p).any():
                    raise OutOfBoundsError("One of the requested xi contains a NaN",
                                           i, np.NaN, self.grid[i][0], self.grid[i][-1])

        if self._compute_d_dvalues:
            # If the table grid or values are component inputs, then we need to create a new table
            # each iteration.
            interp = self._interp
            self.table = interp(self.grid, self.values, interp, extrapolate=self.extrapolate,
                                compute_d_dvalues=True, **self._interp_options)

        table = self.table

        xi = np.atleast_2d(xi)
        n_nodes, nx = xi.shape
        result = np.empty((n_nodes, ), dtype=xi.dtype)
        derivs_x = np.empty((n_nodes, nx), dtype=xi.dtype)
        if self._compute_d_dvalues:
            derivs_val = np.zeros((n_nodes, len(self.values)), dtype=xi.dtype)

        # Loop over n_nodes because there isn't a way to vectorize.
        for j in range(n_nodes):
            val, d_x, d_values_tuple, extrapolate = table.interpolate(xi[j, :])
            result[j] = val
            derivs_x[j, :] = d_x.ravel()
            if self._compute_d_dvalues:
                d_values, idx = d_values_tuple
                derivs_val[j, idx] = d_values.ravel()

        # Cache derivatives
        self._d_dx = derivs_x
        if self._compute_d_dvalues:
            self._d_dvalues = derivs_val

        return result

    def gradient(self, xi):
        """
        Compute the gradients at the specified point.

        Most of the gradients are computed as the interpolation itself is performed,
        but are cached and returned separately by this method.

        If the point for evaluation differs from the point used to produce
        the currently cached gradient, the interpolation is re-performed in
        order to return the correct gradient.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at.

        Returns
        -------
        ndarray
            Vector of gradients of the interpolated values with respect to each value in xi.
        """
        if (self._xi is None) or (not np.array_equal(xi, self._xi)):
            # If inputs have changed since last computation, then re-interpolate.
            self.interpolate(xi)

        return self._d_dx

    def training_gradients(self):
        """
        Compute the training gradient for the vector of training points.

        Returns
        -------
        ndarray
            Gradient of output with respect to training point values.
        """
        return self._d_dvalues
