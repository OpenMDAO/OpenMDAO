"""
Interpolate using am Akima spline.

Based on NPSS implementation, with improvements from Andrew Ning (BYU).
"""
from __future__ import division, print_function, absolute_import
from six.moves import range

import numpy as np

from openmdao.components.interp_util.interp_algorithm import InterpAlgorithm
from openmdao.utils.array_utils import abs_complex, dv_abs_complex


def abs_smooth_complex(x, delta_x):
    """
    Compute the absolute value of a complex-stepped vector.

    Rather than taking a Euclidian norm, simply negate the values that are less than zero.

    Parameters
    ----------
    x : ndarray
        Input array.
    delta_x : float
        Half width of the rounded section.

    Returns
    -------
    ndarray
        Complex-step absolute value of the array.
    """
    y = x**2 / (2.0 * delta_x) + delta_x / 2.0
    idx_neg = np.where(x <= -delta_x)
    idx_pos = np.where(x >= delta_x)
    y[idx_neg] = -x[idx_neg]
    y[idx_pos] = x[idx_pos]
    return y


def dv_abs_smooth_complex(x, x_deriv, delta_x):
    """
    Apply the complex-step derivative of the absolute value function.

    Parameters
    ----------
    x : ndarray
        Input array, used for determining which elements to negate.
    x_deriv : ndarray
        Incoming partial derivative array, may have one additional dimension.
    delta_x : float
        Half width of the rounded section.

    Returns
    -------
    ndarray
        Absolute value applied to x_deriv.
    """
    # Special case when x is (1, ) and x_deriv is (1, n).
    if len(x_deriv.shape) == 1:
        if x[0] >= delta_x:
            return x_deriv
        elif x[0] <= -delta_x:
            return -x_deriv
        else:
            return 2.0 * x[0] * x_deriv / (2.0 * delta_x)

    y_deriv = 2.0 * x * x_deriv / (2.0 * delta_x)
    idx_neg = np.where(x <= -delta_x)
    idx_pos = np.where(x >= delta_x)

    y_deriv[idx_neg] = -x_deriv[idx_neg]
    y_deriv[idx_pos] = x_deriv[idx_pos]

    return y_deriv


def abs_smooth_dv(x, x_deriv, delta_x):
    """
    Compute the absolute value in a smooth differentiable manner.

    The valley is rounded off using a quadratic function.

    Parameters
    ----------
    x : float
        Quantity value
    x_deriv : float
        Derivative value
    delta_x : float
        Half width of the rounded section.

    Returns
    -------
    float
        Smooth absolute value of the quantity.
    float
        Smooth absolute value of the derivative.
    """
    if x >= delta_x:
        y_deriv = x_deriv
        y = x

    elif x <= -delta_x:
        y_deriv = -x_deriv
        y = -x

    else:
        y_deriv = 2.0 * x * x_deriv / (2.0 * delta_x)
        y = x**2 / (2.0 * delta_x) + delta_x / 2.0

    return y, y_deriv


class InterpAkima(InterpAlgorithm):
    """
    Interpolate using an Akima polynomial.
    """

    def __init__(self, grid, values, interp, **kwargs):
        """
        Initialize table and subtables.

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
        """
        super(InterpAkima, self).__init__(grid, values, interp, **kwargs)
        self.k = 4
        self._name = 'akima'

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('delta_x', default=0.,
                             desc="half-width of the smoothing interval added in the valley of "
                             "absolute-value function. This allows the derivatives with respect to"
                             " the data points (dydxpt, dydypt) to also be C1 continuous. Set "
                             "parameter to 0 to get the original Akima function (but only if you "
                             "don't need dydxpt, dydypt")
        self.options.declare('eps', default=1e-30,
                             desc='Value that triggers division-by-zero safeguard.')

    def interpolate(self, x, idx, slice_idx):
        """
        Compute the interpolated value over this grid dimension.

        Parameters
        ----------
        x : ndarray
            The coordinates to sample the gridded data at. First array element is the point to
            interpolate here. Remaining elements are interpolated on sub tables.
        idx : integer
            Interval index for x.
        slice_idx : List of <slice>
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
        grid = self.grid
        subtable = self.subtable
        eps = self.options['eps']
        delta_x = self.options['delta_x']
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

            tshape = self.values[tuple(slice_idx)].shape
            nshape = list(tshape[:-nx])
            nshape.append(nx)
            derivs = np.empty(tuple(nshape), dtype=x.dtype)

            subval, subderiv, _, _ = subtable.evaluate(x[1:], slice_idx=slice_idx)

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
            values = self.values[tuple(slice_idx)]

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
            m2 = 2 * m3 - m4
            m1 = 2 * m2 - m3

        elif idx == 1:
            m1 = 2 * m2 - m3

        elif idx == ngrid - 3:
            m5 = 2 * m4 - m3

        elif idx == ngrid - 2:
            m4 = 2 * m3 - m2
            m5 = 2 * m4 - m3

        m1 = np.atleast_1d(m1)
        m2 = np.atleast_1d(m2)
        m3 = np.atleast_1d(m3)
        m4 = np.atleast_1d(m4)
        m5 = np.atleast_1d(m5)

        # Calculate cubic fit coefficients
        if delta_x > 0:
            w2 = abs_smooth_complex(m4 - m3, delta_x)
            w31 = abs_smooth_complex(m2 - m1, delta_x)
        else:
            w2 = abs_complex(m4 - m3)
            w31 = abs_complex(m2 - m1)

        # Special case to avoid divide by zero.
        jj1 = np.where(w2 + w31 > eps)
        b = 0.5 * (m2 + m3)

        # We need to suppress some warnings that occur when we divide by zero.  We replace all
        # values where this happens, so it never affects the result.
        old_settings = np.seterr()
        np.seterr(invalid='ignore', divide='ignore')

        bpos = np.atleast_1d((m2 * w2 + m3 * w31) / (w2 + w31))
        b[jj1] = bpos[jj1]

        if delta_x > 0:
            w32 = abs_smooth_complex(m5 - m4, delta_x)
            w4 = abs_smooth_complex(m3 - m2, delta_x)
        else:
            w32 = abs_complex(m5 - m4)
            w4 = abs_complex(m3 - m2)

        # Special case to avoid divide by zero.
        jj2 = np.where(w32 + w4 > eps)
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
            if delta_x > 0:
                dw2 = dv_abs_smooth_complex(m4 - m3, dm4 - dm3, delta_x)
                dw3 = dv_abs_smooth_complex(m2 - m1, dm2 - dm1, delta_x)
            else:
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

            dbpos = ((dm2 * w2 + m2e * dw2 + dm3 * w31 + m3e * dw3) - bpos * (dw2 + dw3)) / \
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

            if delta_x > 0:
                dw3 = dv_abs_smooth_complex(m5 - m4, dm5 - dm4, delta_x)
                dw4 = dv_abs_smooth_complex(m3 - m2, dm3 - dm2, delta_x)
            else:
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

            dbp1pos = ((dm3 * w32 + m3e * dw3 + dm4 * w4 + m4e * dw4) - bp1pos * (dw3 + dw4)) / \
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
        return a + dx * (b + dx * (c + dx * d)), derivs, None, None
