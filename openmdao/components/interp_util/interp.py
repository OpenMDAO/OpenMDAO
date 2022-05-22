"""
Base class for interpolation methods that calculate values for each dimension independently.

Based on Tables in NPSS, and was added to bridge the gap between some of the slower scipy
implementations.
"""
import numpy as np

from openmdao.components.interp_util.interp_akima import InterpAkima, Interp1DAkima
from openmdao.components.interp_util.interp_bsplines import InterpBSplines
from openmdao.components.interp_util.interp_cubic import InterpCubic
from openmdao.components.interp_util.interp_lagrange2 import InterpLagrange2, Interp3DLagrange2,\
    Interp2DLagrange2, Interp1DLagrange2
from openmdao.components.interp_util.interp_lagrange3 import InterpLagrange3, Interp3DLagrange3, \
    Interp2DLagrange3, Interp1DLagrange3
from openmdao.components.interp_util.interp_scipy import InterpScipy
from openmdao.components.interp_util.interp_slinear import InterpLinear, Interp3DSlinear, \
    Interp1DSlinear, Interp2DSlinear

from openmdao.components.interp_util.outofbounds_error import OutOfBoundsError
from openmdao.utils.om_warnings import warn_deprecation
from openmdao.utils.array_utils import shape_to_len


INTERP_METHODS = {
    'slinear': InterpLinear,
    'lagrange2': InterpLagrange2,
    'lagrange3': InterpLagrange3,
    'cubic': InterpCubic,
    'akima': InterpAkima,
    'scipy_cubic': InterpScipy,
    'scipy_slinear': InterpScipy,
    'scipy_quintic': InterpScipy,
    'bsplines': InterpBSplines,
    '1D-slinear': Interp1DSlinear,
    '2D-slinear': Interp2DSlinear,
    '3D-slinear': Interp3DSlinear,
    '1D-lagrange2': Interp1DLagrange2,
    '2D-lagrange2': Interp2DLagrange2,
    '3D-lagrange2': Interp3DLagrange2,
    '1D-lagrange3': Interp1DLagrange3,
    '2D-lagrange3': Interp2DLagrange3,
    '3D-lagrange3': Interp3DLagrange3,
    '1D-akima': Interp1DAkima,
    'trilinear': Interp3DSlinear,  # Deprecated
    'akima1D': Interp1DAkima,  # Deprecated
}

TABLE_METHODS = ['slinear', 'lagrange2', 'lagrange3', 'cubic', 'akima',
                 'scipy_cubic', 'scipy_slinear', 'scipy_quintic',
                 'trilinear', 'akima1D',  # These two are Deprecated
                 '3D-slinear', '2D-slinear', '1D-slinear',
                 '1D-akima',
                 '3D-lagrange2', '2D-lagrange2', '1D-lagrange2',
                 '3D-lagrange3', '2D-lagrange3', '1D-lagrange3']
SPLINE_METHODS = ['slinear', 'lagrange2', 'lagrange3', 'cubic', 'akima', 'bsplines',
                  'scipy_cubic', 'scipy_slinear', 'scipy_quintic']


class InterpND(object):
    """
    Interpolation on a regular grid of arbitrary dimensions.

    The data must be defined on a regular grid; the grid spacing however may be uneven. Several
    interpolation methods are supported. These are defined in the child classes. Gradients are
    provided for all interpolation methods. Gradients with respect to grid values are also
    available optionally.

    Parameters
    ----------
    method : str
        Name of interpolation method.
    points : ndarray or tuple of ndarray
        The points defining the regular grid in n dimensions.
        For 1D interpolation, this can be an ndarray of table locations.
        For table interpolation, it can be a tuple or an ndarray. If it is a tuple, it should
        contain one ndarray for each table dimension.
        For spline evaluation, num_cp can be specified instead of points.
    values : ndarray or tuple of ndarray or None
        These must be specified for interpolation.
        The data on the regular grid in n dimensions.
    x_interp : ndarray or None
        If we are always interpolating at a fixed set of locations, then they can be
        specified here.
    extrapolate : bool
        If False, when interpolated values are requested outside of the domain of the input
        data, a ValueError is raised. If True, then the methods are allowed to extrapolate.
        Default is True (raise an exception).
    num_cp : None or int
        Optional. When specified, use a linear distribution of num_cp control points. If you
        are using 'bsplines' as the method, then num_cp must be set instead of points.
    **kwargs : dict
        Interpolator-specific options to pass onward.

    Attributes
    ----------
    extrapolate : bool
        If False, when interpolated values are requested outside of the domain of the input data,
        a ValueError is raised. If True, then the methods are allowed to extrapolate.
        Default is True.
    grid : tuple
        Collection of points that determine the regular grid.
    table : <InterpTable>
        Table object that contains algorithm that performs the interpolation.
    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.
    x_interp : ndarray
        Cached non-decreasing vector of points to be interpolated when used as an order-reducing
        spline.
    _compute_d_dvalues : bool
        When set to True, compute gradients with respect to the grid values.
    _compute_d_dx : bool
        When set to True, compute gradients with respect to the interpolated point location.
    _d_dx : ndarray
        Cache of computed gradients with respect to evaluation point.
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

    def __init__(self, method="slinear", points=None, values=None, x_interp=None, extrapolate=False,
                 num_cp=None, **kwargs):
        """
        Initialize an InterpND object.

        This object can be setup and used to interpolate on a curve or multi-dimensional table.

        It can also be used to setup an interpolating spline that can be evaluated at fixed
        locations.

        For interpolation, specify values and points.

        For spline evaluation, specifiy x_interp and either points or num_cp.
        """
        if not isinstance(method, str):
            msg = "Argument 'method' should be a string."
            raise ValueError(msg)
        elif method not in INTERP_METHODS:
            all_m = ', '.join(['"' + m + '"' for m in INTERP_METHODS])
            raise ValueError('Interpolation method "%s" is not defined. Valid methods are '
                             '%s.' % (method, all_m))
        elif method == 'akima1D':
            warn_deprecation("The 'akima1D' method has been renamed to '1D-akima'.")
        elif method == 'trilinear':
            warn_deprecation("The 'trilinear' method has been renamed to '3D-slinear'.")

        self.extrapolate = extrapolate

        # The table points are always defined, by specifying either the points directly, or num_cp.
        if points is None:
            if num_cp is not None:
                points = [np.linspace(0.0, 1.0, num_cp)]
            else:
                msg = "Either 'points' or 'num_cp' must be specified."
                raise ValueError(msg)
        else:

            if isinstance(points, np.ndarray):
                points = [points]

            for i, p in enumerate(points):
                n_p = len(p)
                if not np.all(np.diff(p) > 0.):
                    raise ValueError("The points in dimension %d must be strictly "
                                     "ascending" % i)
                if not np.asarray(p).ndim == 1:
                    raise ValueError("The points in dimension %d must be "
                                     "1-dimensional" % i)

        # Table Interpolation
        if x_interp is None:

            if values is None:
                msg = "Either 'values' or 'x_interp' must be specified."
                raise ValueError(msg)

            if method == 'bsplines':
                msg = "Method 'bsplines' is not supported for table interpolation."
                raise ValueError(msg)

            if not hasattr(values, 'ndim'):
                # allow reasonable duck-typed values
                values = np.asarray(values)

            if hasattr(values, 'dtype') and hasattr(values, 'astype'):
                if not np.issubdtype(values.dtype, np.inexact):
                    values = values.astype(float)

            if len(points) > values.ndim:
                raise ValueError("There are %d point arrays, but values has %d "
                                 "dimensions" % (len(points), values.ndim))

            if (method.startswith('scipy') or method == 'akima') and \
               (np.iscomplexobj(values[:]) or np.any(np.iscomplex(points[0]))):
                msg = f"Interpolation method '{method}' does not support complex points or values."
                raise ValueError(msg)

            for i, p in enumerate(points):
                n_p = len(p)
                if values.shape[i] != n_p:
                    raise ValueError("There are %d points and %d values in "
                                     "dimension %d" % (len(p), values.shape[i], i))

        self.grid = tuple([np.asarray(p) for p in points])
        self.values = values
        self.x_interp = x_interp

        self._xi = None
        self._d_dx = None
        self._d_dvalues = None
        self._compute_d_dvalues = False
        self._compute_d_dx = True

        # Cache spline coefficients.
        interp = INTERP_METHODS[method]

        if method.startswith('scipy'):
            kwargs['interp_method'] = method

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
        self._compute_d_dx = compute_derivative
        self.table._compute_d_dx = compute_derivative
        self.table._compute_d_dvalues = False

        if isinstance(x, np.ndarray):
            if len(x.shape) < 2:
                if len(self.grid) > 1:
                    # Input is an array containing multi-D coordinates of a single point.
                    x = np.atleast_2d(x)
                else:
                    # Input is an array of separate points on a 1D table.
                    x = np.atleast_2d(x).T
        else:
            # Input is a list or tuple of separate points.
            x = np.atleast_2d(x)

        # cache latest evaluation point for gradient method's use later
        self._xi = x

        xnew = self._interpolate(x)

        if compute_derivative:
            return xnew, self._d_dx
        else:
            return xnew

    def evaluate_spline(self, values, compute_derivative=False):
        """
        Interpolate at all fixed output coordinates given the new table values.

        Parameters
        ----------
        values : ndarray(n_points)
            New data values for all points on the regular grid.
        compute_derivative : bool
            Set to True to compute derivatives with respect to x.

        Returns
        -------
        ndarray
            Value of interpolant at all sample points.
        ndarray
            Value of derivative of interpolated output with respect to values.
        """
        self._compute_d_dvalues = compute_derivative
        self.table._compute_d_dvalues = compute_derivative
        self.table._compute_d_dx = False

        if len(values.shape) == 1:
            values = np.expand_dims(values, axis=0)

        # cache latest evaluation point for gradient method's use later
        self._xi = self.x_interp.copy()

        result = self._evaluate_spline(values)
        if result.shape[0] == 1:
            # Not vectorized, so drop the extra dimension.
            result = result.ravel()

        if compute_derivative:
            d_dvalues = self.spline_gradient()
            if d_dvalues.shape[0] == 1:
                d_dvalues = d_dvalues[0]
            return result, d_dvalues
        else:
            return result

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
        if not self.extrapolate:
            for i, p in enumerate(xi.T):
                if np.isnan(p).any():
                    raise OutOfBoundsError("One of the requested xi contains a NaN",
                                           i, np.NaN, self.grid[i][0], self.grid[i][-1])

                eps = 1e-14 * self.grid[i][-1]
                if np.any(p < self.grid[i][0] - eps) or np.any(p > self.grid[i][-1] + eps):
                    p1 = np.where(self.grid[i][0] > p)[0]
                    p2 = np.where(p > self.grid[i][-1])[0]
                    # First violating entry is enough to direct the user.
                    violated_idx = set(p1).union(p2).pop()
                    value = p[violated_idx]
                    raise OutOfBoundsError("One of the requested xi is out of bounds",
                                           i, value, self.grid[i][0], self.grid[i][-1])

        if self._compute_d_dvalues:
            # If the table grid or values are component inputs, then we need to create a new table
            # each iteration.
            interp = self._interp
            self.table = interp(self.grid, self.values, interp, **self._interp_options)
            if not self.table._supports_d_dvalues:
                raise RuntimeError(f'Method {self.table._name} does not support the '
                                   '"training_data_gradients" option.')

            self.table._compute_d_dvalues = True

        table = self.table
        if table.vectorized(xi):
            result, derivs_x, derivs_val, derivs_grid = table.evaluate_vectorized(xi)

        else:
            n_nodes, nx = xi.shape
            result = np.empty((n_nodes, ), dtype=xi.dtype)
            derivs_x = np.empty((n_nodes, nx), dtype=xi.dtype)
            derivs_val = None

            # TODO: it might be possible to vectorize over n_nodes.
            for j in range(n_nodes):
                val, d_x, d_values, d_grid = table.evaluate(xi[j, ...])
                result[j] = val
                derivs_x[j, :] = d_x.ravel()
                if d_values is not None:
                    if derivs_val is None:
                        dv_shape = [n_nodes]
                        dv_shape.extend(self.values.shape)
                        derivs_val = np.zeros(dv_shape, dtype=xi.dtype)
                    in_slice = table._full_slice
                    full_slice = [slice(j, j + 1)]
                    full_slice.extend(in_slice)
                    shape = derivs_val[tuple(full_slice)].shape
                    derivs_val[tuple(full_slice)] = d_values.reshape(shape)

        # Cache derivatives
        self._d_dx = derivs_x
        self._d_dvalues = derivs_val

        return result

    def _evaluate_spline(self, values):
        """
        Interpolate at all fixed output coordinates given the new table values.

        This method is called from OpenMDAO, and is not meant for standalone use.

        Parameters
        ----------
        values : ndarray(n_nodes x n_points)
            The data on the regular grid in n dimensions.

        Returns
        -------
        ndarray
            Value of interpolant at all sample points.
        """
        xi = self.x_interp
        self.values = values

        table = self.table
        if table._vectorized:

            if table._name == 'bsplines':
                # bsplines is fully vectorized.
                table.values = values
                result, _, derivs_val, _ = table.evaluate_vectorized(xi)

            else:
                # Scipy implementation vectorized over lookups, but not over multiple table values.
                interp = self._interp
                n_nodes, _ = values.shape
                nx = shape_to_len(xi.shape)

                result = np.empty((n_nodes, nx), dtype=values.dtype)
                derivs_val = None

                for j in range(n_nodes):

                    table = interp(self.grid, values[j, :], interp, **self._interp_options)
                    table._compute_d_dvalues = False
                    table._compute_d_dx = False

                    result[j, :], _, _, _ = table.evaluate_vectorized(xi.reshape((nx, 1)))

        else:
            interp = self._interp
            n_nodes, _ = values.shape
            nx = shape_to_len(xi.shape)
            result = np.empty((n_nodes, nx), dtype=values.dtype)
            derivs_val = None

            # TODO: it might be possible to vectorize over n_nodes.
            for j in range(n_nodes):

                table = interp(self.grid, values[j, :], interp, **self._interp_options)
                table._compute_d_dvalues = True
                table._compute_d_dx = False

                for k in range(nx):
                    x_pt = np.atleast_2d(xi[k])
                    val, _, d_values, _ = table.evaluate(x_pt)
                    result[j, k] = val
                    if d_values is not None:
                        if derivs_val is None:
                            dv_shape = [n_nodes, nx]
                            dv_shape.extend(values.shape[1:])
                            derivs_val = np.zeros(dv_shape, dtype=values.dtype)
                        in_slice = table._full_slice
                        full_slice = [slice(j, j + 1), slice(k, k + 1)]
                        full_slice.extend(in_slice)
                        shape = derivs_val[tuple(full_slice)].shape
                        derivs_val[tuple(full_slice)] = d_values.reshape(shape)

        # Cache derivatives
        self._d_dvalues = derivs_val

        self.table = table
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

        return self._gradient().reshape(np.asarray(xi).shape)

    def _gradient(self):
        """
        Return the pre-computed gradients.

        Returns
        -------
        ndarray
            Vector of gradients of the interpolated values with respect to each value in xi.
        """
        return self._d_dx

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
        if self.table._vectorized:
            return self.table.training_gradients(pt)

        else:
            grid = self.grid
            interp = self._interp
            opts = self._interp_options

            for i, axis in enumerate(grid):
                ngrid = axis.size
                values = np.zeros(ngrid)
                deriv_i = np.zeros(ngrid)

                for j in range(ngrid):
                    values[j] = 1.0
                    table = interp([grid[i]], values, interp, **opts)
                    table._compute_d_dvalues = False
                    deriv_i[j], _, _, _ = table.evaluate(pt[i:i + 1])
                    values[j] = 0.0

                if i == 0:
                    deriv_running = deriv_i.copy()
                else:
                    deriv_running = np.outer(deriv_running, deriv_i)

            return deriv_running

    def spline_gradient(self):
        """
        Return derivative of spline with respect to its control points.

        Returns
        -------
        ndarray
            Gradient of output with respect to training point values.
        """
        vec_size, n_cp = self.values.shape
        x_interp = self.x_interp
        n_interp = len(x_interp)

        d_dvalues = self._d_dvalues
        if d_dvalues is not None:
            dy_ddata = np.zeros((vec_size, n_interp, n_cp), dtype=d_dvalues.dtype)

            if d_dvalues.shape[0] == vec_size:
                # Akima precomputes derivs at all points in vec_size.
                dy_ddata[:] = d_dvalues
            else:
                # Bsplines computed derivative is the same at all points in vec_size.
                dy_ddata[:] = np.broadcast_to(d_dvalues.toarray(), (vec_size, n_interp, n_cp))
        else:
            # Note: These derivatives are independent of control point y values, so they will never
            # be complex dtype.
            dy_ddata = np.zeros((n_interp, n_cp))

            # This way works for the rest of the interpolation methods.
            for k in range(n_interp):
                val = self.training_gradients(x_interp[k:k + 1])
                dy_ddata[k, :] = val
            dy_ddata = np.broadcast_to(dy_ddata, (vec_size, n_interp, n_cp))

        return dy_ddata
