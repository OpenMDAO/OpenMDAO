"""
Interpolation method based on Tables in NPSS.
"""
from __future__ import division, print_function, absolute_import
from six.moves import range

import numpy as np

from openmdao.components.structured_metamodel_util.grid_interp_base import GridInterpBase
from openmdao.components.structured_metamodel_util.outofbounds_error import OutOfBoundsError


class InterpLinear(object):

    def __init__(self):
        self.last_index = -1
        self.slope = 0.0

    def interpolate(self, x, idx, table):
        grid = table.grid
        subtables = table.subtables
        slope = self.slope

        if len(subtables) > 0:

            dtmp = subtables[idx].evaluate(x[1:])
            slope = (subtables[idx + 1].evaluate(x[1:]) - dtmp) / (grid[idx + 1] - grid[idx])
            self.slope = slope
            return dtmp + (x[0] - grid[idx]) * slope

        else:
            values = table.values
            last_index = self.last_index

            # If the lookup index is the same as last time and it is not '0',
            # then the slope hasn't changed, so don't need to recalculate.
            if idx != last_index or lastIndex==0:
                lastIndex = idx
                slope = (values[idx + 1] - values[idx]) / (grid[idx + 1] - grid[idx])
                self.slope = slope
                self.last_index = last_index

            return values[idx] + (x - grid[idx]) * slope


class InterpCubic(object):

    def __init__(self):
        self.last_index = -1
        self.second_derivs = None

    def compute_second_derivatives(self, grid, values):
        n = len(grid)

        # Natural spline has second deriv=0 at both ends
        sec_deriv = np.zeros(n)
        temp = np.zeros(n)

        for i in range(1, n-1):
            sig = (grid[i] - grid[i - 1]) / (grid[i + 1] - grid[i - 1])
            prtl = sig * sec_deriv[i - 1] + 2.0
            sec_deriv[i] = (sig - 1.0) / prtl
            temp[i] = (values[i + 1] - values[i]) / (grid[i + 1] - grid[i]) - \
                      (values[i] - values[i - 1]) / (grid[i] - grid[i - 1])
            temp[i] = (6.0 * temp[i] / (grid[i + 1] - grid[i - 1]) - sig*temp[i - 1]) / prtl


        for i in range(n - 2, 0, -1):
            sec_deriv[i] = sec_deriv[i] * sec_deriv[i + 1] + temp[i]

        self.second_derivs = sec_deriv

    def interpolate(self, x, idx, table):
        grid = table.grid
        subtables = table.subtables

        if len(subtables) > 0:
            n = len(grid)
            values = np.zeros(n)
            for j in range(n):
                values[j] = subtables[j].evaluate(x[1:])

            self.compute_second_derivatives(table.grid, values)
            sec_deriv = self.second_derivs

            step = grid[idx + 1] - grid[idx]
            a = (grid[idx + 1] - x[0]) / step
            b = (x[0] - grid[idx]) / step
            return a * values[idx] + b * values[idx + 1] + \
                   ((a * a * a - a) * sec_deriv[idx] + \
                    (b * b * b - b) * sec_deriv[idx + 1]) * (step * step) / 6.0

        else:
            values = table.values

            if self.second_derivs is None:
                self.compute_second_derivatives(table.grid, values)
            sec_deriv = self.second_derivs

            # Perform the interpolation
            step = grid[idx + 1] - grid[idx]
            a = (grid[idx+1] - x) / step
            b = (x - grid[idx]) / step
            return a * values[idx] + b * values[idx + 1] + \
                   ((a * a * a - a) * sec_deriv[idx] + \
                    (b * b * b - b) * sec_deriv[idx + 1]) * (step * step) / 6.0


class NPSSTable(object):

    def __init__(self, grid, values, coeffs):
        self.subtables = []

        self.grid = grid[0]
        self.interp = coeffs[0]()

        if len(grid) > 1:
            self.values = values[0, :]

            nt = len(grid[0])
            for j in range(nt):
                subtable = NPSSTable(grid[1:], values[j, :], coeffs[1:])
                self.subtables.append(subtable)

        else:
            self.values = values

        self.last_index = 0

    def bracket(self, x):
        """
        Locate the interval of the new independent.

        bracket() uses the following algorithm:
           1) determine if the value is above or below the value at _lastIndex
           2) bracket the value between _lastIndex and _lastIndex +- inc, where
              inc has an increasing value of 1,2,4,8, etc.
           3) once the value is bracketed, use bisection method within that bracket.

        The return value is nonzero if the bracket is
        below (-1), or above (1) the table.  On return, index will be set to
        the index of the bottom of the bracket interval.  bracket() assumes that
        the indeps sequence is increasing in a monotonic fashion.
        """
        grid = self.grid
        last_index = self.last_index
        high = last_index + 1
        highbound = len(grid) - 1
        inc = 1

        while x < grid[last_index]:
            high = last_index
            last_index -= inc
            if last_index < 0:
                last_index = 0

                # Check if we're off of the bottom end
                if x < grid[0]:
                    return last_index, -1
                break

            inc += inc

        if high > highbound:
            high = highbound

        while x > grid[high]:
            last_index = high
            high += inc
            if high >= highbound:

                # Check if we're off of the top end
                if x > grid[highbound]:
                    last_index = highbound
                    return last_index, 1
                high = highbound
                break
            inc += inc

        # Bisection
        while high - last_index > 1:
            low = (high + last_index) // 2
            if x < grid[low]:
                high = low
            else:
                last_index = low

        return last_index, 0

    def evaluate(self, x):

        if len(x) > 0:
            idx, extrap_switch = self.bracket(x[0])
        else:
            idx, extrap_switch = self.bracket(x)

        if extrap_switch == 0:
            result = self.interp.interpolate(x, idx, self)
        else:
            raise NotImplementedError("Still working on extrapolation.")

        return result


class NPSSGridInterp(GridInterpBase):
    """
    Interpolation on a regular grid in arbitrary dimensions.

    This method is based on the interpolation code from OTIS.

    The data must be defined on a regular grid; the grid spacing however may be uneven.

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
            the default for the object's interpolate method. Default is "linear".
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
        super(NPSSGridInterp, self).__init__(points, values, order=order,
                                             bounds_error=bounds_error, fill_value=fill_value,
                                             spline_dim_error=spline_dim_error)

        # Cache spline coefficients.
        coeffs = []
        for x in self.grid:
            if order == 'slinear':
                coef = InterpLinear
            elif order == 'cubic':
                coef = InterpCubic

            coeffs.append(coef)

        self._coeffs = coeffs
        self.table = NPSSTable(self.grid, self.values, coeffs)

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
        }

        all_orders = list(interpolator_configs.keys())

        return all_orders, interpolator_configs

    def orders(self):
        """
        Return a list of valid interpolation order names.

        Returns
        -------
        list
            Valid interpolation name strings.
        """
        return ['slinear', 'cubic']

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
            If a spline interpolation order is chosen, this determines whether gradient
            calculations should be made and cached. Default is True.

        Returns
        -------
        array_like
            Value of interpolant at all sample points.
        """
        order = self.order if order is None else order
        if order not in self._all_orders:
            all_m = ', '.join(['"' + m + '"' for m in self._all_orders])
            raise ValueError('Order"%s" is not defined. Valid order are '
                             '%s.' % (order, all_m))

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

        # TODO: Vectorize.
        xi = np.atleast_2d(xi)
        n_nodes, nx = xi.shape
        result = np.empty((n_nodes, ))
        derivs = np.empty((n_nodes, nx))

        for j in range(n_nodes):
            print('evaluating', j)
            val = self.table.evaluate(xi[j, :])
            result[j] = val
            #derivs[j, :] = deriv.flatten()

        # Cache derivatives
        #self.derivs = derivs

        # TODO: Support out-of-bounds identification.
        #if not self.bounds_error and self.fill_value is not None:
        #   result[out_of_bounds] = self.fill_value

        return result