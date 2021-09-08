"""
Interpolate using a third order Lagrange polynomial.

Based on NPSS implementation.
"""
import numpy as np

from openmdao.components.interp_util.interp_algorithm import InterpAlgorithm, InterpAlgorithmSemi


class InterpLagrange3(InterpAlgorithm):
    """
    Interpolate using a third order Lagrange polynomial.

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

    def __init__(self, grid, values, interp, **kwargs):
        """
        Initialize table and subtables.
        """
        super().__init__(grid, values, interp, **kwargs)
        self.k = 4
        self._name = 'lagrange3'

    def interpolate(self, x, idx, slice_idx):
        """
        Compute the interpolated value over this grid dimension.

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
        grid = self.grid
        subtable = self.subtable

        # Complex Step
        if self.values.dtype == complex:
            dtype = self.values.dtype
        else:
            dtype = x.dtype

        # Shift if we don't have 2 points on each side.
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

        c12 = p1 - p2
        c13 = p1 - p3
        c14 = p1 - p4
        c23 = p2 - p3
        c24 = p2 - p4
        c34 = p3 - p4

        if subtable is not None:
            # Interpolate between values that come from interpolating the subtables in the
            # subsequent dimensions.
            nx = len(x)
            slice_idx.append(slice(idx - 1, idx + 3))

            tshape = self.values[tuple(slice_idx)].shape
            nshape = list(tshape[:-nx])
            nshape.append(nx)
            derivs = np.empty(tuple(nshape), dtype=dtype)

            subval, subderiv, _, _ = subtable.evaluate(x[1:], slice_idx=slice_idx)

            q1 = subval[..., 0] / (c12 * c13 * c14)
            q2 = subval[..., 1] / (c12 * c23 * c24)
            q3 = subval[..., 2] / (c13 * c23 * c34)
            q4 = subval[..., 3] / (c14 * c24 * c34)

            dq1_dsub = subderiv[..., 0, :] / (c12 * c13 * c14)
            dq2_dsub = subderiv[..., 1, :] / (c12 * c23 * c24)
            dq3_dsub = subderiv[..., 2, :] / (c13 * c23 * c34)
            dq4_dsub = subderiv[..., 3, :] / (c14 * c24 * c34)

            derivs[..., 1:] = xx4 * (xx3 * (dq1_dsub * xx2 - dq2_dsub * xx1) +
                                     dq3_dsub * xx1 * xx2) - dq4_dsub * xx1 * xx2 * xx3

        else:
            values = self.values[tuple(slice_idx)]

            nshape = list(values.shape[:-1])
            nshape.append(1)
            derivs = np.empty(tuple(nshape), dtype=dtype)

            q1 = values[..., idx - 1] / (c12 * c13 * c14)
            q2 = values[..., idx] / (c12 * c23 * c24)
            q3 = values[..., idx + 1] / (c13 * c23 * c34)
            q4 = values[..., idx + 2] / (c14 * c24 * c34)

        derivs[..., 0] = q1 * (x[0] * (3.0 * x[0] - 2.0 * (p4 + p3 + p2)) +
                               p4 * (p2 + p3) + p2 * p3) - \
            q2 * (x[0] * (3.0 * x[0] - 2.0 * (p4 + p3 + p1)) +
                  p4 * (p1 + p3) + p1 * p3) + \
            q3 * (x[0] * (3.0 * x[0] - 2.0 * (p4 + p2 + p1)) +
                  p4 * (p2 + p1) + p2 * p1) - \
            q4 * (x[0] * (3.0 * x[0] - 2.0 * (p3 + p2 + p1)) +
                  p1 * (p2 + p3) + p2 * p3)

        return xx4 * (xx3 * (q1 * xx2 - q2 * xx1) + q3 * xx1 * xx2) - q4 * xx1 * xx2 * xx3, \
            derivs, None, None


class InterpLagrange3Semi(InterpAlgorithmSemi):
    """
    Interpolate on a semi structured grid using a second order Lagrange polynomial.

    Parameters
    ----------
    grid : tuple(ndarray)
        Tuple containing ndarray of x grid locations for each table dimension.
    values : ndarray
        Array containing the values at all points in grid.
    interp : class
        Interpolation class to be used for subsequent table dimensions.
    extrapolate : bool
        When False, raise an error if extrapolation occurs in this dimension.
    compute_d_dvalues : bool
        When True, compute gradients with respect to the table values.
    idx : list or None
        Maps values to their indices in the training data input. Only used during recursive
        calls.
    idim : int
        Integer corresponding to table depth. Used for error messages.
    **kwargs : dict
        Interpolator-specific options to pass onward.
    """

    def __init__(self, grid, values, interp, extrapolate=True, compute_d_dvalues=False, idx=None,
                 idim=0, **kwargs):
        """
        Initialize table and subtables.
        """
        super().__init__(grid, values, interp, extrapolate=extrapolate,
                         compute_d_dvalues=compute_d_dvalues, idx=idx, idim=idim, **kwargs)
        self.k = 4
        self._name = 'lagrange3'

    def interpolate(self, x):
        """
        Compute the interpolated value over this grid dimension.

        Parameters
        ----------
        x : ndarray
            Coordinate of the point being interpolated. First element is component in this
            dimension. Remaining elements are interpolated on sub tables.

        Returns
        -------
        ndarray
            Interpolated values.
        ndarray
            Derivative of interpolated values with respect to this independent and child
            independents.
        tuple(ndarray, list)
            Derivative of interpolated values with respect to values for this and subsequent table
            dimensions. Second term is the indices into the value array.
        bool
            True if the coordinate is extrapolated in this dimension.
        """
        grid = self.grid
        subtables = self.subtables

        idx, flag = self.bracket(x[0])
        extrap = flag != 0

        # Complex Step
        if self.values.dtype == complex:
            dtype = self.values.dtype
        else:
            dtype = x.dtype

        # Shift if we don't have 2 points on each side.
        ngrid = len(grid)
        if idx > ngrid - 3:
            idx = ngrid - 3
        elif idx == 0:
            idx = 1

        derivs = np.empty(len(x), dtype=dtype)

        if subtables is not None:
            # Interpolate between values that come from interpolating the subtables in the
            # subsequent dimensions.
            val0, dx0, dvalue0, flag0 = subtables[idx - 1].interpolate(x[1:])
            val1, dx1, dvalue1, flag1 = subtables[idx].interpolate(x[1:])
            val2, dx2, dvalue2, flag2 = subtables[idx + 1].interpolate(x[1:])
            val3, dx3, dvalue3, flag3 = subtables[idx + 2].interpolate(x[1:])

            # Extrapolation detection.
            flags = (flag0, flag1, flag2, flag3)
            if extrap or flags == (False, False, False, False):
                # If we are already extrapolating, no change needed.
                # If no sub-points are extrapolating, no change needed.
                pass
            elif flags == (False, False, False, True) and idx > 0:
                # We are near the right edge of our sub-region, so slide to the left.
                idx -= 1
                val_a, dx_a, dvalue_a, flag_a = subtables[idx - 1].interpolate(x[1:])
                if flag_a:
                    # Nothing we can do; there just aren't enough points here.
                    idx += 1
                    extrap = True
                else:
                    val3, dx3, dvalue3 = val2, dx2, dvalue2
                    val2, dx2, dvalue2 = val1, dx1, dvalue1
                    val1, dx1, dvalue1 = val0, dx0, dvalue0
                    val0, dx0, dvalue0 = val_a, dx_a, dvalue_a
            elif flags == (True, False, False, False) and idx < ngrid - 3:
                # We are near the left edge of our sub-region, so slide to the right.
                idx += 1
                val_a, dx_a, dvalue_a, flag_a = subtables[idx + 2].interpolate(x[1:])
                if flag_a:
                    # Nothing we can do; there just aren't enough points here.
                    idx -= 1
                    extrap = True
                else:
                    val0, dx0, dvalue0 = val1, dx1, dvalue1
                    val1, dx1, dvalue1 = val2, dx2, dvalue2
                    val2, dx2, dvalue2 = val3, dx3, dvalue3
                    val3, dx3, dvalue3 = val_a, dx_a, dvalue_a
            else:
                # All other cases, we are in an extrapolation sub-region.
                extrap = True

        p1 = grid[idx - 1]
        p2 = grid[idx]
        p3 = grid[idx + 1]
        p4 = grid[idx + 2]

        xx1 = x[0] - p1
        xx2 = x[0] - p2
        xx3 = x[0] - p3
        xx4 = x[0] - p4

        c12 = p1 - p2
        c13 = p1 - p3
        c14 = p1 - p4
        c23 = p2 - p3
        c24 = p2 - p4
        c34 = p3 - p4

        fact1 = 1.0 / (c12 * c13 * c14)
        fact2 = 1.0 / (c12 * c23 * c24)
        fact3 = 1.0 / (c13 * c23 * c34)
        fact4 = 1.0 / (c14 * c24 * c34)

        if subtables is not None:

            derivs = np.empty(len(dx0) + 1, dtype=dtype)

            q1 = val0 * fact1
            q2 = val1 * fact2
            q3 = val2 * fact3
            q4 = val3 * fact4

            dq1_dsub = dx0 * fact1
            dq2_dsub = dx1 * fact2
            dq3_dsub = dx2 * fact3
            dq4_dsub = dx3 * fact4

            derivs[1:] = xx4 * (xx3 * (dq1_dsub * xx2 - dq2_dsub * xx1) +
                                dq3_dsub * xx1 * xx2) - dq4_dsub * xx1 * xx2 * xx3

            d_value = None
            if self._compute_d_dvalues:
                dvalue0, idx0 = dvalue0
                dvalue1, idx1 = dvalue1
                dvalue2, idx2 = dvalue2
                dvalue3, idx3 = dvalue3
                n = len(dvalue0)

                d_value = np.empty(n * 4, dtype=dtype)
                d_value[:n] = dvalue0 * xx2 * xx3 * xx4 * fact1
                d_value[n:n * 2] = -dvalue1 * xx1 * xx3 * xx4 * fact2
                d_value[n * 2:n * 3] = dvalue2 * xx1 * xx2 * xx4 * fact3
                d_value[n * 3:n * 4] = -dvalue3 * xx1 * xx2 * xx3 * fact4

                idx0.extend(idx1)
                idx0.extend(idx2)
                idx0.extend(idx3)
                d_value = (d_value, idx0)

        else:
            values = self.values
            derivs = np.empty(1, dtype=dtype)

            q1 = values[idx - 1] * fact1
            q2 = values[idx] * fact2
            q3 = values[idx + 1] * fact3
            q4 = values[idx + 2] * fact4

            d_value = None
            if self._compute_d_dvalues:
                d_value = np.empty(4, dtype=dtype)
                d_value[0] = xx2 * xx3 * xx4 * fact1
                d_value[1] = -xx1 * xx3 * xx4 * fact2
                d_value[2] = xx1 * xx2 * xx4 * fact3
                d_value[3] = -xx1 * xx2 * xx3 * fact4

                d_value = (d_value,
                           [self._idx[idx - 1], self._idx[idx],
                            self._idx[idx + 1], self._idx[idx + 2]])

        derivs[0] = q1 * (x[0] * (3.0 * x[0] - 2.0 * (p4 + p3 + p2)) +
                          p4 * (p2 + p3) + p2 * p3) - \
            q2 * (x[0] * (3.0 * x[0] - 2.0 * (p4 + p3 + p1)) +
                  p4 * (p1 + p3) + p1 * p3) + \
            q3 * (x[0] * (3.0 * x[0] - 2.0 * (p4 + p2 + p1)) +
                  p4 * (p2 + p1) + p2 * p1) - \
            q4 * (x[0] * (3.0 * x[0] - 2.0 * (p3 + p2 + p1)) +
                  p1 * (p2 + p3) + p2 * p3)

        return xx4 * (xx3 * (q1 * xx2 - q2 * xx1) + q3 * xx1 * xx2) - q4 * xx1 * xx2 * xx3, \
            derivs, d_value, extrap
