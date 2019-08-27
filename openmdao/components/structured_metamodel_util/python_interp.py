"""
Interpolation method based on Tables in NPSS.

This was added to bridge the gap between some of the slower scipy implementations.
"""
from __future__ import division, print_function, absolute_import
from six.moves import range

import numpy as np
from scipy.interpolate.interpnd import _ndim_coords_from_arrays

from openmdao.components.structured_metamodel_util.grid_interp_base import GridInterpBase
from openmdao.components.structured_metamodel_util.outofbounds_error import OutOfBoundsError


class InterpLinear(object):
    """
    Interpolate using a linear polynomial.

    Attributes
    ----------
    last_index : int
        Index of previously computed approximation, for caching slope computation on leaf table.
    slope : double
        Cached slope value for leaf table.
    """

    def __init__(self):
        self.last_index = None
        self.slope = 0.0

    def interpolate(self, x, idx, slice_idx, table):
        """
        Compute the interpolated value over this grid dimension.

        x : ndarray
            The coordinates to sample the gridded data at. First array element is the point to
            interpolate here. Remaining elements are interpolated on sub tables.
        idx : integer
            Interval index for x.
        slice_idx : List of <slice>
            Slice object containing indices of data points requested by parent interpolating
            tables.
        table : <InterpTable>
            Table object that contains the grid and data.
        """
        grid = table.grid
        subtable = table.subtable

        # Extrapolate high
        if idx == len(grid) - 1:
            idx -= 1

        if subtable is not None:
            # Interpolate between values that come from interpolating the subtables in the
            # subsequent dimensions.
            nx = len(x)
            slice_idx.append(slice(idx, idx + 2))

            tshape = table.values[tuple(slice_idx)].shape
            nshape = list(tshape[:-nx])
            nshape.append(nx)
            derivs = np.empty(tuple(nshape), dtype=x.dtype)

            dtmp, subderiv, extrap_flag = subtable.evaluate(x[1:], slice_idx=slice_idx)
            slope = (dtmp[..., 1] - dtmp[..., 0]) / (grid[idx + 1] - grid[idx])

            derivs[..., 0] = slope
            dslope_dsub = (subderiv[..., 1, :] - subderiv[..., 0, :]) / (grid[idx + 1] - grid[idx])
            derivs[..., 1:] = subderiv[..., 0, :] + (x[0] - grid[idx]) * dslope_dsub

            return dtmp[..., 0] + (x[0] - grid[idx]) * slope, derivs, extrap_flag

        else:
            values = table.values[tuple(slice_idx)]
            last_index = self.last_index
            slope = self.slope

            # If idx is the same as last time, then the slope hasn't changed.
            if idx != last_index:
                self.last_index = idx
                slope = (values[..., idx + 1] - values[..., idx]) / (grid[idx + 1] - grid[idx])
                self.slope = slope

            return values[..., idx] + (x - grid[idx]) * slope, np.expand_dims(slope, axis=-1), \
                   False


class InterpLagrange2(object):
    """
    Interpolate using a second order Lagrange polynomial.

    Attributes
    ----------
    last_index : int
        Index of previously computed approximation, for caching slope computation on leaf table.
    slope : double
        Cached slope value for leaf table.
    """

    def __init__(self):
        self.last_index = None
        self.slope = None

    def interpolate(self, x, idx, slice_idx, table):
        """
        Compute the interpolated value over this grid dimension.

        x : ndarray
            The coordinates to sample the gridded data at. First array element is the point to
            interpolate here. Remaining elements are interpolated on sub tables.
        idx : integer
            Interval index for x.
        slice_idx : List of <slice>
            Slice object containing indices of data points requested by parent interpolating
            tables.
        table : <InterpTable>
            Table object that contains the grid and data.
        """
        grid = table.grid
        subtable = table.subtable

        # Extrapolate high
        ngrid = len(grid)
        if idx > ngrid - 3:
            idx = ngrid - 3

        derivs = np.empty(len(x))

        xx1 = x[0] - grid[idx]
        xx2 = x[0] - grid[idx + 1]
        xx3 = x[0] - grid[idx + 2]

        if subtable is not None:
            # Interpolate between values that come from interpolating the subtables in the
            # subsequent dimensions.
            nx = len(x)
            slice_idx.append(slice(idx, idx + 3))

            tshape = table.values[tuple(slice_idx)].shape
            nshape = list(tshape[:-nx])
            nshape.append(nx)
            derivs = np.empty(tuple(nshape), dtype=x.dtype)

            c12 = grid[idx] - grid[idx + 1]
            c13 = grid[idx] - grid[idx + 2]
            c23 = grid[idx + 1] - grid[idx + 2]

            subval, subderiv, extrap_flag = subtable.evaluate(x[1:], slice_idx=slice_idx)

            q1 = subval[..., 0] / (c12 * c13)
            q2 = subval[..., 1] / (c12 * c23)
            q3 = subval[..., 2] / (c13 * c23)

            dq1_dsub = subderiv[..., 0, :] / (c12 * c13)
            dq2_dsub = subderiv[..., 1, :]  / (c12 * c23)
            dq3_dsub = subderiv[..., 2, :] / (c13 * c23)

            derivs[..., 1:] = xx3 * (dq1_dsub * xx2 - dq2_dsub * xx1) + dq3_dsub * xx1 * xx2

        else:
            extrap_flag = False
            values = table.values[tuple(slice_idx)]
            last_index = self.last_index

            nshape = list(values.shape[:-1])
            nshape.append(1)
            derivs = np.empty(tuple(nshape), dtype=x.dtype)

            # If idx is the same as last time, then the slope hasn't changed.
            if idx != last_index:
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

        return xx3 * (q1 * xx2 - q2 * xx1) + q3 * xx1 * xx2, derivs, extrap_flag


class InterpLagrange3(object):
    """
    Interpolate using a third order Lagrange polynomial.

    Attributes
    ----------
    last_index : int
        Index of previously computed approximation, for caching slope computation on leaf table.
    slope : double
        Cached slope value for leaf table.
    """

    def __init__(self):
        self.last_index = -1
        self.slope = None

    def interpolate(self, x, idx, slice_idx, table):
        """
        Compute the interpolated value over this grid dimension.

        x : ndarray
            The coordinates to sample the gridded data at. First array element is the point to
            interpolate here. Remaining elements are interpolated on sub tables.
        idx : integer
            Interval index for x.
        slice_idx : List of <slice>
            Slice object containing indices of data points requested by parent interpolating
            tables.
        table : <InterpTable>
            Table object that contains the grid and data.
        """
        grid = table.grid
        subtable = table.subtable

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
            # Interpolate between values that come from interpolating the subtables in the
            # subsequent dimensions.
            nx = len(x)
            slice_idx.append(slice(idx - 1, idx + 3))

            tshape = table.values[tuple(slice_idx)].shape
            nshape = list(tshape[:-nx])
            nshape.append(nx)
            derivs = np.empty(tuple(nshape), dtype=x.dtype)

            c12 = p1 - p2
            c13 = p1 - p3
            c14 = p1 - p4
            c23 = p2 - p3
            c24 = p2 - p4
            c34 = p3 - p4

            subval, subderiv, extrap_flag = subtable.evaluate(x[1:], slice_idx=slice_idx)

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
            extrap_flag = False
            values = table.values[tuple(slice_idx)]
            last_index = self.last_index

            nshape = list(values.shape[:-1])
            nshape.append(1)
            derivs = np.empty(tuple(nshape), dtype=x.dtype)

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

        return xx4 * (xx3 * (q1 * xx2 - q2 * xx1) + q3 * xx1 * xx2) - q4 * xx1 * xx2 * xx3, \
               derivs, extrap_flag


def abs_complex(x):
    """
    Compute the absolute value of a complex-stepped vector.

    Rather than taking a Euclidian norm, simply negate the values that are less than zero.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    ndarray
        Complex-step absolute value of the array.
    """
    idx_neg = np.where(x < 0)
    x[idx_neg] = -x[idx_neg]
    return x


def dv_abs_complex(x, x_deriv):
    """
    Apply the complex-step derivative of the absolute value function.

    Parameters
    ----------
    x : ndarray
        Input array, used for determining which elements to negate.
    x_deriv : ndarray
        Incominng partial derivative array, may have one additional dimension.

    Returns
    -------
    ndarray
        Absolute value applied to x_deriv.
    """
    idx_neg = np.where(x < 0)

    # Special case when x is (1, ) and x_deriv is (1, n).
    if len(x_deriv.shape) == 1:
        if idx_neg[0].size != 0:
            return -x_deriv

    x_deriv[idx_neg] = -x_deriv[idx_neg]

    return x_deriv


class InterpAkima(object):
    """
    Interpolate using an Akima polynomial.
    """

    def interpolate(self, x, idx, slice_idx, table):
        """
        Compute the interpolated value over this grid dimension.

        x : ndarray
            The coordinates to sample the gridded data at. First array element is the point to
            interpolate here. Remaining elements are interpolated on sub tables.
        idx : integer
            Interval index for x.
        slice_idx : List of <slice>
            Slice object containing indices of data points requested by parent interpolating
            tables.
        table : <InterpTable>
            Table object that contains the grid and data.
        """
        grid = table.grid
        subtable = table.subtable
        nx = len(x)

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
            high_idx = idx + 4
        elif idx == ngrid - 3:
            high_idx = idx + 3
        else:
            high_idx = idx + 2

        if subtable is not None:
            # Interpolate between values that come from interpolating the subtables in the
            # subsequent dimensions.

            slice_idx.append(slice(low_idx, high_idx))

            tshape = table.values[tuple(slice_idx)].shape
            nshape = list(tshape[:-nx])
            nshape.append(nx)
            derivs = np.empty(tuple(nshape), dtype=x.dtype)

            subval, subderiv, extrap_flag = subtable.evaluate(x[1:], slice_idx=slice_idx)

            j = 0
            if idx >= 2:
                val1 = subval[..., j]
                dval1 = subderiv[..., j, :]
                j += 1
            if idx >= 1:
                val2 = subval[..., j]
                dval2 = subderiv[..., j, :]
                j += 1
            val3 = subval[..., j]
            val4 = subval[..., j + 1]
            dval3 = subderiv[..., j, :]
            dval4 = subderiv[..., j + 1, :]
            j += 2
            if idx < ngrid - 2:
                val5 = subval[..., j]
                dval5 = subderiv[..., j, :]
                j += 1
            if idx < ngrid - 3:
                val6 = subval[..., j]
                dval6 = subderiv[..., j, :]
                j += 1

        else:
            extrap_flag = False
            values = table.values[tuple(slice_idx)]

            nshape = list(values.shape[:-1])
            nshape.append(1)
            derivs = np.empty(tuple(nshape), dtype=x.dtype)

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
            m1 = 3 * m3 - 2 * m4
            m2 = 2 * m3 - m4

        elif idx == 1:
            m1 = 2 * m2 - m3

        elif idx == ngrid - 3:
            m5 = 2 * m4 - m3

        elif idx == ngrid - 2:
            m4 = 2 * m3 - m2
            m5 = 3 * m3 - 2 * m2

        m1 = np.atleast_1d(m1)
        m2 = np.atleast_1d(m2)
        m3 = np.atleast_1d(m3)
        m4 = np.atleast_1d(m4)
        m5 = np.atleast_1d(m5)

        # Calculate cubic fit coefficients
        w2 = abs_complex(m4 - m3)
        w31 = abs_complex(m2 - m1)

        # Special case to avoid divide by zero.
        jj1 = np.where(w2 + w31  > 0)
        b = 0.5 * (m2 + m3)

        # We need to suppress some warnings that occur when we divide by zero.  We replace all
        # values where this happens, so it never affects the result.
        old_settings = np.seterr()
        np.seterr(invalid='ignore', divide='ignore')

        bpos = np.atleast_1d((m2 * w2 + m3 * w31) / (w2 + w31))
        b[jj1] = bpos[jj1]

        w32 = abs_complex(m5 - m4)
        w4 = abs_complex(m3 - m2)

        # Special case to avoid divide by zero.
        jj2 = np.where(w32 + w4  > 0)
        bp1 = 0.5 * (m3 + m4)

        bp1pos = np.atleast_1d((m3 * w32 + m4 * w4) / (w32 + w4))
        bp1[jj2] = bp1pos[jj2]

        if extrap == 0:
            h = 1.0 / (grid[idx + 1] - grid[idx])
            a = val3
            c = (3 * m3 - 2 * b - bp1) * h
            d = (b + bp1 - 2 * m3) * h * h
            dx = x[0] - grid[idx]

        elif extrap == 1:
            a = val4
            b = bp1
            dx = x[0] - grid[idx + 1]

        else:
            a = val3
            dx = x[0] - grid[0]

        derivs[..., 0] = b + dx * (2.0 * c + 3.0 * d * dx)

        # Propagate derivatives from sub table.
        if subtable is not None:
            shape = dval3.shape
            cd_term = 0

            dm3 = (dval4 - dval3) / (grid[idx + 1] - grid[idx])

            if idx >= 2:
                dm1 = (dval2 - dval1) / (grid[idx - 1] - grid[idx - 2])
            else:
                dm1 = np.zeros(shape, dtype=x.dtype)

            if idx >= 1:
                dm2 = (dval3 - dval2) / (grid[idx] - grid[idx - 1])
            else:
                dm2 = np.zeros(shape, dtype=x.dtype)

            if idx < ngrid - 2:
                dm4 = (dval5 - dval4) / (grid[idx + 2] - grid[idx + 1])
            else:
                dm4 = np.zeros(shape, dtype=x.dtype)

            if idx < ngrid - 3:
                dm5 = (dval6 - dval5) / (grid[idx + 3] - grid[idx + 2])
            else:
                dm5 = np.zeros(shape, dtype=x.dtype)

            if idx == 0:
                dm1 = 3 * dm3 - 2 * dm4
                dm2 = 2 * dm3 - dm4

            elif idx == 1:
                dm1 = 2 * dm2 - dm3

            elif idx == ngrid - 3:
                dm5 = 2 * dm4 - dm3

            elif idx == ngrid - 2:
                dm4 = 2 * dm3 - dm2
                dm5 = 3 * dm3 - 2 * dm2

            # Calculate cubic fit coefficients
            dw2 = dv_abs_complex(m4 - m3, dm4 - dm3)
            dw3 = dv_abs_complex(m2 - m1, dm2 - dm1)

            # Special case to avoid divide by zero.

            if len(nshape) > 1:
                w2 = w2[..., np.newaxis]
                w31 = w31[..., np.newaxis]
                m2e = m2[..., np.newaxis]
                m3e = m3[..., np.newaxis]
                bpos = bpos[..., np.newaxis]
            else:
                m2e = m2
                m3e = m3

            db = 0.5 * (dm2 + dm3)

            dbpos = ((dm2 * w2 + m2e * dw2 + dm3 * w31 + m3e * dw3)  - bpos * (dw2 + dw3)) / \
                    (w2 + w31)

            if nx > 2:

                if len(val3.shape) == 0:
                    if len(jj1[0]) > 0:
                        db[:] = dbpos
                else:
                    for j in range(nx - 1):
                        db[jj1, j] = dbpos[jj1, j]

            else:
                db[jj1] = dbpos[jj1]

            dw3 = dv_abs_complex(m5 - m4, dm5 - dm4)
            dw4 = dv_abs_complex(m3 - m2, dm3 - dm2)

            # Special case to avoid divide by zero.
            if len(nshape) > 1:
                w32 = w32[..., np.newaxis]
                w4 = w4[..., np.newaxis]
                m3e = m3[..., np.newaxis]
                m4e = m4[..., np.newaxis]
                bp1pos = bp1pos[..., np.newaxis]
            else:
                m3e = m3
                m4e = m4

            dbp1 = 0.5 * (dm3 + dm4)

            dbp1pos = ((dm3 * w32 + m3e * dw3 + dm4 * w4 + m4e * dw4) - bp1pos * (dw3 + dw4) ) / \
                       (w32 + w4)

            if nx > 2:

                if len(val3.shape) == 0:
                    if len(jj2[0]) > 0:
                        dbp1[:] = dbp1pos
                else:
                    for j in range(nx - 1):
                        dbp1[jj2, j] = dbp1pos[jj2, j]

            else:
                dbp1[jj2] = dbp1pos[jj2]

            if extrap == 0:
                da = dval3
                dc = (3 * dm3 - 2 * db - dbp1) * h
                dd = (db + dbp1 - 2 * dm3) * h * h
                cd_term = dx * (dc + dx * dd)

            elif extrap == 1:
                da = dval4
                db = dbp1

            else:
                da = dval3

            derivs[..., 1:] = da + dx * (db + cd_term)

        # Restore numpy warnings to previous setting.
        np.seterr(**old_settings)

        # Evaluate dependent value and exit
        return a + dx * (b + dx * (c + dx * d)), derivs, extrap_flag


class InterpCubic(object):
    """
    Interpolate using a cubic spline.

    Continuity of derivatives between segments is assured, but a linear solution is
    required to attain this.

    Attributes
    ----------
    second_derivs : ndarray
        Cache of all second derivatives for the leaf table only.
    """

    def __init__(self):
        self.second_derivs = None

    def compute_second_derivatives(self, grid, values, x):
        n = len(grid)

        # Natural spline has second deriv=0 at both ends
        sec_deriv = np.zeros(n, dtype=x.dtype)
        temp = np.zeros(values.shape, dtype=x.dtype)

        sig = (grid[1:n - 1] - grid[:n - 2]) / (grid[2:] - grid[:n - 2])

        vdiff = (values[..., 1:] - values[..., :n - 1]) / (grid[1:] - grid[:n - 1])
        tmp = 6.0 * (vdiff[..., 1:] - vdiff[..., :n - 2]) / (grid[2:] - grid[:n - 2])

        for i in range(1, n - 1):
            prtl = sig[i - 1] * sec_deriv[..., i - 1] + 2.0
            sec_deriv[i] = (sig[i - 1] - 1.0) / prtl
            temp[..., i] = (tmp[..., i - 1] - sig[i - 1] * temp[..., i - 1]) / prtl

        sec_deriv = np.array(np.broadcast_to(sec_deriv, temp.shape), dtype=x.dtype)

        for i in range(n - 2, 0, -1):
            sec_deriv[..., i] = sec_deriv[..., i] * sec_deriv[..., i + 1] + temp[..., i]

        return sec_deriv

    def interpolate(self, x, idx, slice_idx, table):
        """
        Compute the interpolated value over this grid dimension.

        x : ndarray
            The coordinates to sample the gridded data at. First array element is the point to
            interpolate here. Remaining elements are interpolated on sub tables.
        idx : integer
            Interval index for x.
        slice_idx : List of <slice>
            Slice object containing indices of data points requested by parent interpolating
            tables.
        table : <InterpTable>
            Table object that contains the grid and data.
        """
        grid = table.grid
        subtable = table.subtable

        # Extrapolate high
        if idx == len(grid) - 1:
            idx -= 1

        if subtable is not None:
            # Interpolate between values that come from interpolating the subtables in the
            # subsequent dimensions.
            n = len(grid)
            nx = len(x)

            values, subderivs, extrap_flag = subtable.evaluate(x[1:], slice_idx=slice_idx)
            sec_deriv = self.compute_second_derivatives(table.grid, values, x)

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
            derivs = np.empty(tuple(tshape), dtype=x.dtype)

            derivs[..., 0] = r_step * (values[..., idx + 1] - values[..., idx]) + \
                             (((3.0 * b * b - 1) * sec_deriv[..., idx + 1] - \
                               (3.0 * a * a - 1) * sec_deriv[..., idx]) * (step * fact))

            if nx == 2:
                dsec = self.compute_second_derivatives(table.grid, subderivs, x)
                derivs[..., 1] = ((a * a * a - a) * dsec[..., idx] + \
                                   (b * b * b - b) * dsec[..., idx + 1]) * (step * step * fact)

                derivs[..., 1] += a * subderivs[..., idx] + b * subderivs[..., idx + 1]

            else:
                dsec = self.compute_second_derivatives(table.grid, np.swapaxes(subderivs, -1, -2),
                                                       x)
                derivs[..., 1:] = ((a * a * a - a) * dsec[..., idx] + \
                                   (b * b * b - b) * dsec[..., idx + 1]) * (step * step * fact)

                derivs[..., 1:] += a * subderivs[..., idx, :] + b * subderivs[..., idx + 1, :]

            return interp_values, derivs, extrap_flag

        else:
            values = table.values

            if self.second_derivs is None:
                self.second_derivs = self.compute_second_derivatives(table.grid, values, x)
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

            return val, deriv, False


class InterpTable(object):

    def __init__(self, grid, values, interps):
        self.subtable = None

        self.grid = grid[0]
        self.interp = interps[0]()
        self.values = values

        if len(grid) > 1:
            self.subtable = InterpTable(grid[1:], values, interps[1:])

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
            idx, extrap_flag = self.bracket(x[0])
        else:
            idx, extrap_flag = self.bracket(x)

        if slice_idx is None:
            slice_idx = []

        result, deriv, sub_extrap_flag = self.interp.interpolate(x, idx, slice_idx, self)

        return result, deriv, extrap_flag or sub_extrap_flag


class PythonGridInterp(GridInterpBase):
    """
    Interpolation on a regular grid in arbitrary dimensions.

    This class includes methods based on the interpolation code from NPSS implemented solely in
    Python.

    The data must be defined on a regular grid; the grid spacing however may be uneven.

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
        super(PythonGridInterp, self).__init__(points, values, interp_method=interp_method,
                                               bounds_error=bounds_error, fill_value=fill_value,
                                               spline_dim_error=spline_dim_error)

        # Cache spline coefficients.
        interps = []
        for x in self.grid:
            if interp_method == 'slinear':
                interp = InterpLinear
            elif interp_method == 'lagrange2':
                interp = InterpLagrange2
            elif interp_method == 'lagrange3':
                interp = InterpLagrange3
            elif interp_method == 'cubic':
                interp = InterpCubic
            else:
                interp = InterpAkima

            interps.append(interp)

        self._interps = interps
        self.table = InterpTable(self.grid, self.values, interps)

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
            "slinear": 2,
            "lagrange2": 3,
            "lagrange3": 4,
            "akima": 4,
            "cubic": 3,
        }

        all_methods = list(interpolator_configs.keys())

        return all_methods, interpolator_configs

    def interpolate(self, xi, interp_method=None, compute_gradients=True):
        """
        Interpolate at the sample coordinates.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        interp_method : str, optional
            The interpolation method to perform. Supported are 'slinear', 'cubic', and
            'quintic'. Default is None, which will use the order defined at the construction
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
        self._xi_shape = xi_shape = xi.shape

        if xi_shape[-1] != len(self.grid):
            raise ValueError("The requested sample points xi have dimension "
                             "%d, but this RegularGridInterp has "
                             "dimension %d" % (xi_shape[1], ndim))

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

        # TODO: Vectorize.
        xi = np.atleast_2d(self._xi)
        n_nodes, nx = xi.shape
        result = np.empty((n_nodes, ), dtype=xi.dtype)
        extrap_flag = np.empty((n_nodes, ))
        derivs = np.empty((n_nodes, nx), dtype=xi.dtype)

        for j in range(n_nodes):
            if self.training_data_gradients:
                # If the table values are inputs, then we need to create a new table each time.
                self.table = InterpTable(self.grid, self.values, self._interps)
            val, deriv, extrap_flag = self.table.evaluate(xi[j, :])
            result[j] = val
            derivs[j, :] = deriv.flatten()

        # Cache derivatives
        self._all_gradients = derivs

        if not self.bounds_error and self.fill_value is not None:
            out_of_bounds = np.where(extrap_flag != 0)
            result[out_of_bounds] = self.fill_value

        return result

    def gradient(self, xi, interp_method=None):
        """
        Compute the gradients at the specified point.

        The gradients are computed as the interpolation itself is performed,
        but are cached and returned separately by this method.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        interp_method : str, optional
            The interpolation method to perform. Supported are 'slinear',
            'cubic', and 'quintic'. Default is None, which will use the order
            defined at the construction of the interpolation object instance.

        Returns
        -------
        gradient : ndarray of shape (..., ndim)
            Vector of gradients of the interpolated values with respect to each value in xi.
        """
        return self._all_gradients

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

        for i, axis in enumerate(self.grid):
            ngrid = axis.size
            values = np.zeros(ngrid)
            deriv_i = np.zeros(ngrid)

            for j in range(ngrid):
                values[j] = 1.0
                table = InterpTable([grid[i]], values, [self._interps[i]])
                deriv_i[j], _, _ = table.evaluate(pt[i:i + 1])
                values[j] = 0.0

            if i == 0:
                deriv_running = deriv_i.copy()
            else:
                deriv_running = np.outer(deriv_running, deriv_i)

        return deriv_running