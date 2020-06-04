"""
Interpolate using am Akima spline.

Based on NPSS implementation, with improvements from Andrew Ning (BYU).
"""
import numpy as np

from openmdao.components.interp_util.interp_algorithm import InterpAlgorithm
from openmdao.utils.array_utils import abs_complex, dv_abs_complex


def abs_smooth_complex(x, delta_x):
    """
    Compute the absolute value of a complex-stepped vector.

    Rather than taking a Euclidean norm, simply negate the values that are less than zero.

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
            return x, x_deriv
        elif x[0] <= -delta_x:
            return -x, -x_deriv
        else:
            return 0.5 * (x[0]**2 / delta_x + delta_x), x[0] * x_deriv / delta_x

    y_deriv = x * x_deriv / delta_x
    y = 0.5 * (x[0]**2 / delta_x + delta_x)
    idx_neg = np.where(x <= -delta_x)
    idx_pos = np.where(x >= delta_x)

    y_deriv[idx_neg] = -x_deriv[idx_neg]
    y_deriv[idx_pos] = x_deriv[idx_pos]
    y[idx_neg] = -x[idx_neg]
    y[idx_pos] = x[idx_pos]

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

        # Complex Step
        if self.values.dtype == np.complex:
            dtype = self.values.dtype
        else:
            dtype = x.dtype

        c = 0.0
        d = 0.0
        m1 = 0.0
        m2 = 0.0
        m4 = 0.0
        m5 = 0.0
        extrap = 0
        deriv_dv = None

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
            deriv_dx = np.empty(tuple(nshape), dtype=x.dtype)

            subval, subderiv, deriv_dv, _ = subtable.evaluate(x[1:], slice_idx=slice_idx)
            if deriv_dv is not None:
                self._full_slice = subtable._full_slice

            j = 0
            if idx >= 2:
                val1 = subval[..., j]
                dval1 = subderiv[..., j, :]
                if self._compute_d_dvalues:
                    dval1_dv = deriv_dv[..., j, :]
                j += 1
            if idx >= 1:
                val2 = subval[..., j]
                dval2 = subderiv[..., j, :]
                if self._compute_d_dvalues:
                    dval2_dv = deriv_dv[..., j, :]
                j += 1
            val3 = subval[..., j]
            val4 = subval[..., j + 1]
            dval3 = subderiv[..., j, :]
            dval4 = subderiv[..., j + 1, :]
            if self._compute_d_dvalues:
                dval3_dv = deriv_dv[..., j, :]
                dval4_dv = deriv_dv[..., j + 1, :]

            j += 2
            if idx < ngrid - 2:
                val5 = subval[..., j]
                dval5 = subderiv[..., j, :]
                if self._compute_d_dvalues:
                    dval5_dv = deriv_dv[..., j, :]
                j += 1
            if idx < ngrid - 3:
                val6 = subval[..., j]
                dval6 = subderiv[..., j, :]
                if self._compute_d_dvalues:
                    dval6_dv = deriv_dv[..., j, :]
                j += 1

        else:
            values = self.values[tuple(slice_idx)]

            nshape = list(values.shape[:-1])
            nshape.append(1)
            deriv_dx = np.empty(tuple(nshape), dtype=dtype)
            if self._compute_d_dvalues:
                n_this = high_idx - low_idx
                nshape = list(values.shape[:-1])
                nshape.append(n_this)
                n_flat = np.prod(nshape)
                deriv_dv = np.eye(n_flat, dtype=dtype)

                new_shape = []
                new_shape.extend(nshape)
                new_shape.append(n_flat)
                deriv_dv = deriv_dv.reshape(new_shape)

                slice_idx.append(slice(low_idx, high_idx))
                self._full_slice = slice_idx

            j = 0
            if idx >= 2:
                val1 = values[..., idx - 2]
                if self._compute_d_dvalues:
                    idx_val1 = j
                    j += 1
            if idx >= 1:
                val2 = values[..., idx - 1]
                if self._compute_d_dvalues:
                    idx_val2 = j
                    j += 1
            val3 = values[..., idx]
            if self._compute_d_dvalues:
                idx_val3 = j
                j += 1
            val4 = values[..., idx + 1]
            if self._compute_d_dvalues:
                idx_val4 = j
                j += 1
            if idx < ngrid - 2:
                val5 = values[..., idx + 2]
                if self._compute_d_dvalues:
                    idx_val5 = j
                    j += 1
            if idx < ngrid - 3:
                val6 = values[..., idx + 3]
                if self._compute_d_dvalues:
                    idx_val6 = j

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

        compute_local_train = self._compute_d_dvalues and subtable is None

        m3 = (val4 - val3) / (grid[idx + 1] - grid[idx])
        if compute_local_train:
            dm3_dv = (deriv_dv[..., idx_val4, :] - deriv_dv[..., idx_val3, :]) / \
                (grid[idx + 1] - grid[idx])
            dm1_dv = 0
            dm2_dv = 0
            dm4_dv = 0
            dm5_dv = 0

        if idx >= 2:
            m1 = (val2 - val1) / (grid[idx - 1] - grid[idx - 2])
            if compute_local_train:
                dm1_dv = (deriv_dv[..., idx_val2, :] - deriv_dv[..., idx_val1, :]) / \
                    (grid[idx - 1] - grid[idx - 2])

        if idx >= 1:
            m2 = (val3 - val2) / (grid[idx] - grid[idx - 1])
            if compute_local_train:
                dm2_dv = (deriv_dv[..., idx_val3, :] - deriv_dv[..., idx_val2, :]) / \
                    (grid[idx] - grid[idx - 1])

        if idx < ngrid - 2:
            m4 = (val5 - val4) / (grid[idx + 2] - grid[idx + 1])
            if compute_local_train:
                dm4_dv = (deriv_dv[..., idx_val5, :] - deriv_dv[..., idx_val4, :]) / \
                    (grid[idx + 2] - grid[idx + 1])

        if idx < ngrid - 3:
            m5 = (val6 - val5) / (grid[idx + 3] - grid[idx + 2])
            if compute_local_train:
                dm5_dv = (deriv_dv[..., idx_val6, :] - deriv_dv[..., idx_val5, :]) / \
                    (grid[idx + 3] - grid[idx + 2])

        if idx == 0:
            m2 = 2 * m3 - m4
            m1 = 2 * m2 - m3
            if compute_local_train:
                dm2_dv = 2.0 * dm3_dv - dm4_dv
                dm1_dv = 2.0 * dm2_dv - dm3_dv

        elif idx == 1:
            m1 = 2 * m2 - m3
            if compute_local_train:
                dm1_dv = 2.0 * dm2_dv - dm3_dv

        elif idx == ngrid - 3:
            m5 = 2 * m4 - m3
            if compute_local_train:
                dm5_dv = 2.0 * dm4_dv - dm3_dv

        elif idx == ngrid - 2:
            m4 = 2 * m3 - m2
            m5 = 2 * m4 - m3
            if compute_local_train:
                dm4_dv = 2.0 * dm3_dv - dm2_dv
                dm5_dv = 2.0 * dm4_dv - dm3_dv

        m1 = np.atleast_1d(m1)
        m2 = np.atleast_1d(m2)
        m3 = np.atleast_1d(m3)
        m4 = np.atleast_1d(m4)
        m5 = np.atleast_1d(m5)

        # Calculate cubic fit coefficients
        if delta_x > 0:
            if compute_local_train:
                w2, dw2_dv = dv_abs_smooth_complex(m4 - m3, dm4_dv - dm3_dv, delta_x)
                w31, dw31_dv = dv_abs_smooth_complex(m2 - m1, dm2_dv - dm1_dv, delta_x)
            else:
                w2 = abs_smooth_complex(m4 - m3, delta_x)
                w31 = abs_smooth_complex(m2 - m1, delta_x)
        else:
            if compute_local_train:
                w2, dw2_dv = dv_abs_complex(m4 - m3, dm4_dv - dm3_dv)
                w31, dw31_dv = dv_abs_complex(m2 - m1, dm2_dv - dm1_dv)
            else:
                w2 = abs_complex(m4 - m3)
                w31 = abs_complex(m2 - m1)

        # Special case to avoid divide by zero.
        jj1 = np.where(w2 + w31 > eps)
        b = 0.5 * (m2 + m3)
        if compute_local_train:
            db_dv = 0.5 * (dm2_dv + dm3_dv)
            db_dv = np.atleast_2d(db_dv)

        # We need to suppress some warnings that occur when we divide by zero.  We replace all
        # values where this happens, so it never affects the result.
        old_settings = np.seterr()
        np.seterr(invalid='ignore', divide='ignore')

        bpos = np.atleast_1d((m2 * w2 + m3 * w31) / (w2 + w31))
        if compute_local_train:
            if len(m2.shape) > 1:

                w2n = w2[..., np.newaxis]
                w31n = w31[..., np.newaxis]
                m2n = m2[..., np.newaxis]
                m3n = m3[..., np.newaxis]

                dbpos_dv = ((m2n * dw2_dv + dm2_dv * w2n + m3n * dw31_dv + dm3_dv * w31n) *
                            (w2n + w31n) -
                            (m2n * w2n + m3n * w31n) * (dw2_dv + dw31_dv)) / (w2n + w31n) ** 2
                dbpos_dv = np.atleast_2d(dbpos_dv)

            else:
                dbpos_dv = ((m2 * dw2_dv + dm2_dv * w2 + m3 * dw31_dv + dm3_dv * w31) *
                            (w2 + w31) -
                            (m2 * w2 + m3 * w31) * (dw2_dv + dw31_dv)) / (w2 + w31) ** 2
                dbpos_dv = np.atleast_2d(dbpos_dv)

        b[jj1] = bpos[jj1]
        if compute_local_train:
            db_dv[jj1] = dbpos_dv[jj1]

        if delta_x > 0:
            if compute_local_train:
                w32, dw32_dv = dv_abs_smooth_complex(m5 - m4, dm5_dv - dm4_dv, delta_x)
                w4, dw4_dv = dv_abs_smooth_complex(m3 - m2, dm3_dv - dm2_dv, delta_x)
            else:
                w32 = abs_smooth_complex(m5 - m4, delta_x)
                w4 = abs_smooth_complex(m3 - m2, delta_x)
        else:
            if compute_local_train:
                w32, dw32_dv = dv_abs_complex(m5 - m4, dm5_dv - dm4_dv)
                w4, dw4_dv = dv_abs_complex(m3 - m2, dm3_dv - dm2_dv)
            else:
                w32 = abs_complex(m5 - m4)
                w4 = abs_complex(m3 - m2)

        # Special case to avoid divide by zero.
        jj2 = np.where(w32 + w4 > eps)
        bp1 = 0.5 * (m3 + m4)
        if compute_local_train:
            dbp1_dv = 0.5 * (dm3_dv + dm4_dv)
            dbp1_dv = np.atleast_2d(dbp1_dv)

        bp1pos = np.atleast_1d((m3 * w32 + m4 * w4) / (w32 + w4))
        if compute_local_train:
            if len(m2.shape) > 1:

                w32n = w32[..., np.newaxis]
                w4n = w4[..., np.newaxis]
                m4n = m4[..., np.newaxis]

                dbp1pos_dv = ((m3n * dw32_dv + dm3_dv * w32n + m4n * dw4_dv + dm4_dv * w4n) *
                              (w32n + w4n) -
                              (m3n * w32n + m4n * w4n) * (dw32_dv + dw4_dv)) / (w32n + w4n) ** 2
                dbp1pos_dv = np.atleast_2d(dbp1pos_dv)
            else:
                dbp1pos_dv = ((m3 * dw32_dv + dm3_dv * w32 + m4 * dw4_dv + dm4_dv * w4) *
                              (w32 + w4) -
                              (m3 * w32 + m4 * w4) * (dw32_dv + dw4_dv)) / (w32 + w4) ** 2
                dbp1pos_dv = np.atleast_2d(dbp1pos_dv)

        bp1[jj2] = bp1pos[jj2]
        if compute_local_train:
            dbp1_dv[jj2] = dbp1pos_dv[jj2]

        if extrap == 0:
            h = 1.0 / (grid[idx + 1] - grid[idx])
            a = val3
            c = (3 * m3 - 2 * b - bp1) * h
            d = (b + bp1 - 2 * m3) * h * h
            dx = x[0] - grid[idx]

            if compute_local_train:
                da_dv = deriv_dv[..., idx_val3, :]
                dc_dv = (3 * dm3_dv - 2 * db_dv - dbp1_dv) * h
                dd_dv = (db_dv + dbp1_dv - 2 * dm3_dv) * h * h

        elif extrap == 1:
            a = val4
            b = bp1
            dx = x[0] - grid[idx + 1]

            if compute_local_train:
                da_dv = deriv_dv[..., idx_val4, :]
                db_dv = dbp1_dv
                dc_dv = dd_dv = 0

        else:
            a = val3
            dx = x[0] - grid[0]

            if compute_local_train:
                da_dv = deriv_dv[..., idx_val3, :]
                dc_dv = dd_dv = 0

        deriv_dx[..., 0] = b + dx * (2.0 * c + 3.0 * d * dx)
        if compute_local_train:
            deriv_dv = da_dv + dx * (db_dv + dx * (dc_dv + dx * dd_dv))

        # Propagate derivatives from sub table.
        if subtable is not None and (self._compute_d_dx or self._compute_d_dvalues):
            shape = dval3.shape
            cd_term = 0

            if self._compute_d_dx:
                dm3 = (dval4 - dval3) / (grid[idx + 1] - grid[idx])
            if self._compute_d_dvalues:
                dm3_dv = (dval4_dv - dval3_dv) / (grid[idx + 1] - grid[idx])

            if idx >= 2:
                if self._compute_d_dx:
                    dm1 = (dval2 - dval1) / (grid[idx - 1] - grid[idx - 2])
                if self._compute_d_dvalues:
                    dm1_dv = (dval2_dv - dval1_dv) / (grid[idx - 1] - grid[idx - 2])
            else:
                dm1 = 0

            if idx >= 1:
                if self._compute_d_dx:
                    dm2 = (dval3 - dval2) / (grid[idx] - grid[idx - 1])
                if self._compute_d_dvalues:
                    dm2_dv = (dval3_dv - dval2_dv) / (grid[idx] - grid[idx - 1])
            else:
                dm2 = 0

            if idx < ngrid - 2:
                if self._compute_d_dx:
                    dm4 = (dval5 - dval4) / (grid[idx + 2] - grid[idx + 1])
                if self._compute_d_dvalues:
                    dm4_dv = (dval5_dv - dval4_dv) / (grid[idx + 2] - grid[idx + 1])
            else:
                dm4 = 0

            if idx < ngrid - 3:
                if self._compute_d_dx:
                    dm5 = (dval6 - dval5) / (grid[idx + 3] - grid[idx + 2])
                if self._compute_d_dvalues:
                    dm5_dv = (dval6_dv - dval5_dv) / (grid[idx + 3] - grid[idx + 2])
            else:
                dm5 = 0

            if idx == 0:
                if self._compute_d_dx:
                    dm1 = 3 * dm3 - 2 * dm4
                    dm2 = 2 * dm3 - dm4
                if self._compute_d_dvalues:
                    dm1_dv = 3 * dm3_dv - 2 * dm4_dv
                    dm2_dv = 2 * dm3_dv - dm4_dv

            elif idx == 1:
                if self._compute_d_dx:
                    dm1 = 2 * dm2 - dm3
                if self._compute_d_dvalues:
                    dm1_dv = 2 * dm2_dv - dm3_dv

            elif idx == ngrid - 3:
                if self._compute_d_dx:
                    dm5 = 2 * dm4 - dm3
                if self._compute_d_dvalues:
                    dm5_dv = 2 * dm4_dv - dm3_dv

            elif idx == ngrid - 2:
                if self._compute_d_dx:
                    dm4 = 2 * dm3 - dm2
                    dm5 = 3 * dm3 - 2 * dm2
                if self._compute_d_dvalues:
                    dm4_dv = 2 * dm3_dv - dm2_dv
                    dm5_dv = 3 * dm3_dv - 2 * dm2_dv

            # Calculate cubic fit coefficients
            if delta_x > 0:
                if self._compute_d_dx:
                    _, dw2 = dv_abs_smooth_complex(m4 - m3, dm4 - dm3, delta_x)
                    _, dw3 = dv_abs_smooth_complex(m2 - m1, dm2 - dm1, delta_x)
                if self._compute_d_dvalues:
                    _, dw2_dv = dv_abs_smooth_complex(m4 - m3, dm4_dv - dm3_dv, delta_x)
                    _, dw3_dv = dv_abs_smooth_complex(m2 - m1, dm2_dv - dm1_dv, delta_x)
            else:
                if self._compute_d_dx:
                    _, dw2 = dv_abs_complex(m4 - m3, dm4 - dm3)
                    _, dw3 = dv_abs_complex(m2 - m1, dm2 - dm1)
                if self._compute_d_dvalues:
                    _, dw2_dv = dv_abs_complex(m4 - m3, dm4_dv - dm3_dv)
                    _, dw3_dv = dv_abs_complex(m2 - m1, dm2_dv - dm1_dv)

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

            if self._compute_d_dx:
                db = 0.5 * (dm2 + dm3)

                dbpos = ((dm2 * w2 + m2e * dw2 + dm3 * w31 + m3e * dw3) - bpos * (dw2 + dw3)) / \
                    (w2 + w31)

            if self._compute_d_dvalues:
                db_dv = 0.5 * (dm2_dv + dm3_dv)

                dbpos_dv = ((dm2_dv * w2 + m2e * dw2_dv + dm3_dv * w31 + m3e * dw3_dv) -
                            bpos * (dw2_dv + dw3_dv)) / \
                    (w2 + w31)

            if nx > 2:

                if len(val3.shape) == 0:
                    if len(jj1[0]) > 0:
                        if self._compute_d_dx:
                            db[:] = dbpos
                        if self._compute_d_dvalues:
                            db_dv[:] = dbpos_dv
                else:
                    for j in range(nx - 1):
                        if self._compute_d_dx:
                            db[jj1, j] = dbpos[jj1, j]
                        if self._compute_d_dvalues:
                            db_dv[jj1, j] = dbpos_dv[jj1, j]

            else:
                if self._compute_d_dx:
                    db[jj1] = dbpos[jj1]
                if self._compute_d_dvalues:
                    db_dv[jj1] = dbpos_dv[jj1]

            if delta_x > 0:
                if self._compute_d_dx:
                    _, dw3 = dv_abs_smooth_complex(m5 - m4, dm5 - dm4, delta_x)
                    _, dw4 = dv_abs_smooth_complex(m3 - m2, dm3 - dm2, delta_x)
                if self._compute_d_dvalues:
                    _, dw3_dv = dv_abs_smooth_complex(m5 - m4, dm5_dv - dm4_dv, delta_x)
                    _, dw4_dv = dv_abs_smooth_complex(m3 - m2, dm3_dv - dm2_dv, delta_x)
            else:
                if self._compute_d_dx:
                    _, dw3 = dv_abs_complex(m5 - m4, dm5 - dm4)
                    _, dw4 = dv_abs_complex(m3 - m2, dm3 - dm2)
                if self._compute_d_dvalues:
                    _, dw3_dv = dv_abs_complex(m5 - m4, dm5_dv - dm4_dv)
                    _, dw4_dv = dv_abs_complex(m3 - m2, dm3_dv - dm2_dv)

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

            if self._compute_d_dx:
                dbp1 = 0.5 * (dm3 + dm4)

                dbp1pos = ((dm3 * w32 + m3e * dw3 + dm4 * w4 + m4e * dw4) -
                           bp1pos * (dw3 + dw4)) / \
                    (w32 + w4)

            if self._compute_d_dvalues:
                dbp1_dv = 0.5 * (dm3_dv + dm4_dv)

                dbp1pos_dv = ((dm3_dv * w32 + m3e * dw3_dv + dm4_dv * w4 + m4e * dw4_dv) -
                              bp1pos * (dw3_dv + dw4_dv)) / \
                    (w32 + w4)

            if nx > 2:

                if len(val3.shape) == 0:
                    if len(jj2[0]) > 0:
                        if self._compute_d_dx:
                            dbp1[:] = dbp1pos
                        if self._compute_d_dvalues:
                            dbp1_dv[:] = dbp1pos_dv

                else:
                    for j in range(nx - 1):
                        if self._compute_d_dx:
                            dbp1[jj2, j] = dbp1pos[jj2, j]
                        if self._compute_d_dvalues:
                            dbp1_dv[jj2, j] = dbp1pos_dv[jj2, j]

            else:
                if self._compute_d_dx:
                    dbp1[jj2] = dbp1pos[jj2]
                if self._compute_d_dvalues:
                    dbp1_dv[jj2] = dbp1pos_dv[jj2]

            if extrap == 0:
                if self._compute_d_dx:
                    da = dval3
                    dc = (3 * dm3 - 2 * db - dbp1) * h
                    dd = (db + dbp1 - 2 * dm3) * h * h
                    cd_term = dx * (dc + dx * dd)
                if self._compute_d_dvalues:
                    da_dv = dval3_dv
                    dc_dv = (3 * dm3_dv - 2 * db_dv - dbp1_dv) * h
                    dd_dv = (db_dv + dbp1_dv - 2 * dm3_dv) * h * h
                    cd_term_dv = dx * (dc_dv + dx * dd_dv)

            elif extrap == 1:
                if self._compute_d_dx:
                    da = dval4
                    db = dbp1
                if self._compute_d_dvalues:
                    da_dv = dval4_dv
                    db_dv = dbp1_dv

            else:
                if self._compute_d_dx:
                    da = dval3
                if self._compute_d_dvalues:
                    da_dv = dval3_dv

            if self._compute_d_dx:
                deriv_dx[..., 1:] = da + dx * (db + cd_term)
            if self._compute_d_dvalues:
                deriv_dv = da_dv + dx * (db_dv + cd_term_dv)

        # Restore numpy warnings to previous setting.
        np.seterr(**old_settings)

        # Evaluate dependent value and exit
        return a + dx * (b + dx * (c + dx * d)), deriv_dx, deriv_dv, None
