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

    def interpolate(self, x, idx, slice_idx, table):
        grid = table.grid
        subtable = table.subtable
        slope = self.slope

        # Extrapolate high
        if idx == len(grid) - 1:
            idx -= 1

        if subtable is not None:
            nx = len(x)
            slice_idx.append(slice(idx, idx + 2))

            tshape = table.values[tuple(slice_idx)].shape
            nshape = list(tshape[:-nx])
            nshape.append(nx)
            derivs = np.empty(tuple(nshape))

            dtmp, subderiv = subtable.evaluate(x[1:], slice_idx=slice_idx)
            slope = (dtmp[..., 1:2] - dtmp[..., 0:1]) / (grid[idx + 1] - grid[idx])

            derivs[..., 0:1] = slope
            dslope_dsub = (subderiv[..., 1, :] - subderiv[..., 0, :]) / (grid[idx + 1] - grid[idx])
            derivs[..., 1:] = subderiv[..., 0, :] + (x[0] - grid[idx]) * dslope_dsub

            return dtmp[..., 0] + (x[0] - grid[idx]) * slope.flatten(), derivs

        else:
            values = table.values[tuple(slice_idx)]
            last_index = self.last_index

            # If the lookup index is the same as last time and it is not '0',
            # then the slope hasn't changed, so don't need to recalculate.
            if idx != last_index or lastIndex==0:
                lastIndex = idx
                slope = (values[..., idx + 1] - values[..., idx]) / (grid[idx + 1] - grid[idx])
                self.slope = slope
                self.last_index = last_index

            return values[..., idx] + (x - grid[idx]) * slope, np.expand_dims(slope, axis=-1)


class InterpLagrange2(object):

    def __init__(self):
        self.last_index = -1
        self.slope = None

    def interpolate(self, x, idx, slice_idx, table):
        grid = table.grid
        subtable = table.subtable
        nx = len(x)

        # Extrapolate high
        ngrid = len(grid)
        if idx > ngrid - 3:
            idx = ngrid - 3
        elif idx == ngrid - 1:
            idx -= 1

        derivs = np.empty(len(x))

        xx1 = x[0] - grid[idx]
        xx2 = x[0] - grid[idx + 1]
        xx3 = x[0] - grid[idx + 2]

        if subtable is not None:

            slice_idx.append(slice(idx, idx + 3))

            tshape = table.values[tuple(slice_idx)].shape
            nshape = list(tshape[:-nx])
            nshape.append(nx)
            derivs = np.empty(tuple(nshape))

            # Checking the lastIndex value here won't help, because our slope is not
            # guaranteed to be the same as last time even if idx == lastIndex, since
            # the numerator of the slope comes from lower level tables whose index may
            # have changed.
            c12 = grid[idx] - grid[idx + 1]
            c13 = grid[idx] - grid[idx + 2]
            c23 = grid[idx + 1] - grid[idx + 2]

            subval, subderiv = subtable.evaluate(x[1:], slice_idx=slice_idx)

            q1 = subval[..., 0] / (c12 * c13)
            q2 = subval[..., 1] / (c12 * c23)
            q3 = subval[..., 2] / (c13 * c23)

            dq1_dsub = subderiv[..., 0, :] / (c12 * c13)
            dq2_dsub = subderiv[..., 1, :]  / (c12 * c23)
            dq3_dsub = subderiv[..., 2, :] / (c13 * c23)

            derivs[..., 1:] = xx3 * (dq1_dsub * xx2 - dq2_dsub * xx1) + dq3_dsub * xx1 * xx2

        else:
            values = table.values[tuple(slice_idx)]
            last_index = self.last_index

            nshape = list(values.shape[:-1])
            nshape.append(1)
            derivs = np.empty(tuple(nshape))

            # If the lookup index is the same as last time and it is not '0',
            # then the slope hasn't changed, so don't need to recalculate.
            if idx != last_index or last_index == 0:
                self.last_index = idx
                c12 = grid[idx] - grid[idx + 1]
                c13 = grid[idx] - grid[idx + 2]
                c23 = grid[idx + 1] - grid[idx + 2]
                q1 = values[..., idx] / (c12 * c13)
                q2 = values[..., idx + 1] / (c12 * c23)
                q3 = values[..., idx + 2] / (c13 * c23)
                self.slope = (q1, q2, q3)

            q1, q2, q3 = self.slope

        derivs[..., 0] = q1 * (2.0 * x[0] - grid[idx + 1] - grid[idx + 2]) - \
                         q2 * (2.0 * x[0] - grid[idx] - grid[idx + 2]) + \
                         q3 * (2.0 * x[0] - grid[idx] - grid[idx + 1])

        return xx3 * (q1 * xx2 - q2 * xx1) + q3 * xx1 * xx2, derivs


class InterpLagrange3(object):

    def __init__(self):
        self.last_index = -1
        self.slope = None

    def interpolate(self, x, idx, slice_idx, table):
        grid = table.grid
        subtable = table.subtable
        nx = len(x)

        # Extrapolate high
        ngrid = len(grid)
        if idx > ngrid - 3:
            idx = ngrid - 3
        elif idx == 0:
            idx = 1

        derivs = np.empty(len(x))

        p1 = grid[idx - 1]
        p2 = grid[idx]
        p3 = grid[idx + 1]
        p4 = grid[idx + 2]

        xx1 = x[0] - p1
        xx2 = x[0] - p2
        xx3 = x[0] - p3
        xx4 = x[0] - p4

        if subtable is not None:

            slice_idx.append(slice(idx - 1, idx + 3))

            tshape = table.values[tuple(slice_idx)].shape
            nshape = list(tshape[:-nx])
            nshape.append(nx)
            derivs = np.empty(tuple(nshape))

            # Checking the lastIndex value here won't help, because our slope is not
            # guaranteed to be the same as last time even if idx == lastIndex, since
            # the numerator of the slope comes from lower level tables whose index may
            # have changed.
            c12 = p1 - p2
            c13 = p1 - p3
            c14 = p1 - p4
            c23 = p2 - p3
            c24 = p2 - p4
            c34 = p3 - p4

            subval, subderiv = subtable.evaluate(x[1:], slice_idx=slice_idx)

            q1 = subval[..., 0] / (c12 * c13 * c14)
            q2 = subval[..., 1] / (c12 * c23 * c24)
            q3 = subval[..., 2] / (c13 * c23 * c34)
            q4 = subval[..., 3] / (c14 * c24 * c34)

            dq1_dsub = subderiv[..., 0, :] / (c12 * c13 * c14)
            dq2_dsub = subderiv[..., 1, :] / (c12 * c23 * c24)
            dq3_dsub = subderiv[..., 2, :] / (c13 * c23 * c34)
            dq4_dsub = subderiv[..., 3, :] / (c14 * c24 * c34)

            derivs[..., 1:] = xx4 * (xx3 * (dq1_dsub * xx2 - dq2_dsub * xx1) + dq3_dsub * xx1 * xx2) - \
                dq4_dsub * xx1 * xx2 * xx3

        else:
            values = table.values[tuple(slice_idx)]
            last_index = self.last_index

            nshape = list(values.shape[:-1])
            nshape.append(1)
            derivs = np.empty(tuple(nshape))

            # If the lookup index is the same as last time and it is not '0',
            # then the slope hasn't changed, so don't need to recalculate.
            if idx != last_index or last_index == 0:
                self.last_index = idx

                c12 = p1 - p2
                c13 = p1 - p3
                c14 = p1 - p4
                c23 = p2 - p3
                c24 = p2 - p4
                c34 = p3 - p4

                q1 = values[..., idx - 1] / (c12 * c13 * c14)
                q2 = values[..., idx] / (c12 * c23 * c24)
                q3 = values[..., idx + 1] / (c13 * c23 * c34)
                q4 = values[..., idx + 2] / (c14 * c24 * c34)

                self.slope = (q1, q2, q3, q4)

            q1, q2, q3, q4 = self.slope

        derivs[..., 0] = q1 * (x[0] * (3.0 * x[0] - 2.0 * (p4 + p3 + p2)) + \
                               p4 * (p2 + p3) + p2 * p3) - \
                         q2 * (x[0] * (3.0 * x[0] - 2.0 * (p4 + p3 + p1)) + \
                               p4 * (p1 + p3) + p1 * p3) + \
                         q3 * (x[0] * (3.0 * x[0] - 2.0 * (p4 + p2 + p1)) + \
                               p4 * (p2 + p1) + p2 * p1) - \
                         q4 * (x[0] * (3.0 * x[0] - 2.0 * (p3 + p2 + p1)) + \
                               p1 * (p2 + p3) + p2 * p3)

        return xx4 * (xx3 * (q1 * xx2 - q2 * xx1) + q3 * xx1 * xx2) - q4 * xx1 * xx2 * xx3, derivs


class InterpAkima(object):

    def interpolate(self, x, idx, slice_idx, table):
        grid = table.grid
        subtable = table.subtable

        c = 0.0
        d = 0.0
        m1 = 0.0
        m2 = 0.0
        m4 = 0.0
        m5 = 0.0
        extrap = 0

        # Check for extrapolation conditions. if off upper end of table (idx = ient-1)
        # reset idx to interval lower bracket (ient-2). if off lower end of table
        # (idx = 0) interval lower bracket already set but need to check if independent
        # variable is < first value in independent array
        ngrid = len(grid)
        if idx == ngrid - 1:
            idx = ngrid - 2
            extrap = 1
        elif idx == 0 and x[0] < grid[0]:
            extrap = -1

        if idx >= 2:
            low_idx = idx - 2
        elif idx == 1:
            low_idx = idx - 1
        else:
            low_idx = idx

        if idx < ngrid - 3:
            high_idx = idx + 3
        elif idx == ngrid - 3:
            high_idx = idx + 2
        else:
            high_idx = idx + 1

        if subtable is not None:

            slice_idx.append(slice(low_idx, high_idx + 1))

            subval, subderiv = subtable.evaluate(x[1:], slice_idx=slice_idx)

            j = 0
            if idx >= 2:
                val1 = subval[..., j]
                j += 1
            if idx >= 1:
                val2 = subval[..., j]
                j += 1
            val3 = subval[..., j]
            val4 = subval[..., j + 1]
            j += 2
            if idx < ngrid - 2:
                val5 = subval[..., j]
                j += 1
            if idx < ngrid - 3:
                val6 = subval[..., j]
                j += 1

        else:
            values = table.values[tuple(slice_idx)]

            if idx >= 2:
                val1 = values[..., idx - 2]
            if idx >= 1:
                val2 = values[..., idx - 1]
            val3 = values[..., idx]
            val4 = values[..., idx + 1]
            if idx < ngrid - 2:
                val5 = values[..., idx + 2]
            if idx < ngrid - 3:
                val6 = values[..., idx + 3]

        # Calculate interval slope values
        #
        # m1 is the slope of interval (xi-2, xi-1)
        # m2 is the slope of interval (xi-1, xi)
        # m3 is the slope of interval (xi, xi+1)
        # m4 is the slope of interval (xi+1, xi+2)
        # m5 is the slope of interval (xi+2, xi+3)
        #
        # The values of m1, m2, m4 and m5 may be calculated from other slope values
        # depending on the value of idx

        m3 = (val4 - val3) / (grid[idx + 1] - grid[idx])

        if idx >= 2:
            m1 = (val2 - val1) / (grid[idx - 1] - grid[idx - 2])

        if idx >= 1:
            m2 = (val3 - val2) / (grid[idx] - grid[idx - 1])

        if idx < ngrid - 2:
            m4 = (val5 - val4) / (grid[idx + 2] - grid[idx + 1])

        if idx < ngrid - 3:
            m5 = (val6 - val5) / (grid[idx + 3] - grid[idx + 2])

        if idx == 0:
            m1 = 3*m3 - 2*m4
            m2 = 2*m3 - m4

        elif idx == 1:
            m1 = 2 * m2 - m3

        elif idx == ngrid - 3:
            m5 = 2 * m4 - m3

        elif idx == ngrid - 2:
            m4 = 2 * m3 - m2
            m5 = 3 * m3 - 2 * m2

        # Calculate cubic fit coefficients
        w2 = abs(m4 - m3)
        w3 = abs(m2 - m1)

        jj = np.where(w2 + w3  > 0)
        b = np.atleast_1d(0.5 * (m2+m3))
        bpos = np.atleast_1d((m2*w2 + m3*w3) / (w2 + w3))
        b[jj] = bpos[jj]

        w3 = abs(m5 - m4)
        w4 = abs(m3 - m2)

        jj = np.where(w3 + w4  > 0)
        bp1 = np.atleast_1d(0.5 * (m3+m4))
        bp1pos = np.atleast_1d((m3*w3 + m4*w4) / (w3 + w4))
        bp1[jj] = bp1pos[jj]

        if extrap == 0:
            a = val3
            c = (3 * m3 - 2 * b - bp1) / (grid[idx+1] - grid[idx])
            d = (b + bp1 - 2 * m3) / ((grid[idx+1] - grid[idx]) * (grid[idx+1] - grid[idx]))
            dx = x[0] - grid[idx]

        elif extrap == 1:
            a = val4
            b = bp1;
            dx = x[0] - grid[idx + 1]

        else:
            a = val3
            dx = x[0] - grid[0]

        # Evaluate dependent value and exit
        return a + b * dx + c * (dx * dx) + d * (dx * dx * dx), None


class InterpCubic(object):

    def __init__(self):
        self.second_derivs = None

    def compute_second_derivatives(self, grid, values):
        n = len(grid)

        # Natural spline has second deriv=0 at both ends
        sec_deriv = np.zeros(n)
        temp = np.zeros(values.shape)

        sig = (grid[1:n - 1] - grid[:n - 2]) / (grid[2:] - grid[:n - 2])

        vdiff = (values[..., 1:] - values[..., :n - 1]) / (grid[1:] - grid[:n - 1])
        tmp = 6.0 * (vdiff[..., 1:] - vdiff[..., :n - 2]) / (grid[2:] - grid[:n - 2])

        for i in range(1, n - 1):
            prtl = 1.0 / sig[i - 1] * sec_deriv[i - 1] + 2.0
            sec_deriv[i] = (sig[i - 1] - 1.0) * prtl
            temp[..., i] = (tmp[..., i - 1] - sig[i - 1] * temp[..., i - 1]) * prtl

        sec_deriv = np.array(np.broadcast_to(sec_deriv, temp.shape))

        for i in range(n - 2, 0, -1):
            sec_deriv[..., i] = sec_deriv[..., i] * sec_deriv[..., i + 1] + temp[..., i]

        return sec_deriv

    def interpolate(self, x, idx, slice_idx, table):
        grid = table.grid
        subtable = table.subtable

        # Extrapolate high
        if idx == len(grid) - 1:
            idx -= 1

        if subtable is not None:
            n = len(grid)
            nx = len(x)

            values, subderivs = subtable.evaluate(x[1:], slice_idx=slice_idx)
            sec_deriv = self.compute_second_derivatives(table.grid, values)

            step = grid[idx + 1] - grid[idx]
            r_step = 1.0 / step
            a = (grid[idx + 1] - x[0]) * r_step
            b = (x[0] - grid[idx]) * r_step
            fact = 1.0 / 6.0

            interp_values = a * values[..., idx] + b * values[..., idx + 1] + \
                ((a * a * a - a) * sec_deriv[..., idx] + \
                 (b * b * b - b) * sec_deriv[..., idx + 1]) * (step * step * fact)

            # Derivatives

            tshape = list(interp_values.shape)
            tshape.append(nx)
            derivs = np.empty(tuple(tshape))

            derivs[..., 0] = r_step * (values[..., idx + 1] - values[..., idx]) + \
                             (((3.0 * b * b - 1) * sec_deriv[..., idx + 1] - \
                               (3.0 * a * a - 1) * sec_deriv[..., idx]) * (step * fact))

            if nx == 2:
                dsec = self.compute_second_derivatives(table.grid, subderivs)
                derivs[..., 1] = ((a * a * a - a) * dsec[..., idx] + \
                                   (b * b * b - b) * dsec[..., idx + 1]) * (step * step * fact)

                derivs[..., 1] += a * subderivs[..., idx] + b * subderivs[..., idx + 1]

            else:
                dsec = self.compute_second_derivatives(table.grid, np.swapaxes(subderivs, -1, -2))
                derivs[..., 1:] = ((a * a * a - a) * dsec[..., idx] + \
                                   (b * b * b - b) * dsec[..., idx + 1]) * (step * step * fact)

                derivs[..., 1:] += a * subderivs[..., idx, :] + b * subderivs[..., idx + 1, :]

            return interp_values, derivs

        else:
            values = table.values

            if self.second_derivs is None:
                self.second_derivs = self.compute_second_derivatives(table.grid, values)
            sec_deriv = self.second_derivs

            # Perform the interpolation
            step = grid[idx + 1] - grid[idx]
            r_step = 1.0 / step
            a = (grid[idx + 1] - x) * r_step
            b = (x - grid[idx]) * r_step
            fact = 1.0 / 6.0

            val = a * values[..., idx] + b * values[..., idx + 1] + \
                   ((a * a * a - a) * sec_deriv[..., idx] + \
                    (b * b * b - b) * sec_deriv[..., idx + 1]) * (step * step * fact)

            deriv = r_step * (values[..., idx + 1] - values[..., idx]) + \
                    ((3.0 * b * b - 1) * sec_deriv[..., idx + 1] - \
                     (3.0 * a * a - 1) * sec_deriv[..., idx]) * (step * fact)

            return val, deriv


class NPSSTable(object):

    def __init__(self, grid, values, coeffs):
        self.subtable = None

        self.grid = grid[0]
        self.interp = coeffs[0]()
        self.values = values

        if len(grid) > 1:
            self.subtable = NPSSTable(grid[1:], values, coeffs[1:])

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

    def evaluate(self, x, slice_idx=None):

        if len(x) > 0:
            idx, extrap_switch = self.bracket(x[0])
        else:
            idx, extrap_switch = self.bracket(x)

        if True or extrap_switch == 0:
            if slice_idx is None:
                slice_idx = []
            result, deriv = self.interp.interpolate(x, idx, slice_idx, self)
        else:
            raise NotImplementedError("Still working on extrapolation.")

        return result, deriv


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
            elif order == 'lagrange2':
                coef = InterpLagrange2
            elif order == 'lagrange3':
                coef = InterpLagrange3
            elif order == 'cubic':
                coef = InterpCubic
            elif order == 'akima':
                coef = InterpAkima

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
            "slinear": 2,
            "lagrange2": 3,
            "lagrange3": 4,
            "akima": 4,
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
        return ['slinear', 'lagrange2', 'cubic', 'akima']

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
            val, deriv = self.table.evaluate(xi[j, :])
            result[j] = val
            if deriv is not None:
                derivs[j, :] = deriv.flatten()

        # Cache derivatives
        self.derivs = derivs

        # TODO: Support out-of-bounds identification.
        #if not self.bounds_error and self.fill_value is not None:
        #   result[out_of_bounds] = self.fill_value

        return result

    def gradient(self, xi, order=None):
        """
        Compute the gradients at the specified point.

        The gradients are computed as the interpolation itself is performed,
        but are cached and returned separately by this method.

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
        return self.derivs