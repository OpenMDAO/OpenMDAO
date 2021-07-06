"""
Interpolate using a second order Lagrange polynomial.

Based on NPSS implementation.
"""
import numpy as np

from openmdao.components.interp_util.interp_algorithm import InterpAlgorithm, InterpAlgorithmSemi


class InterpLagrange2(InterpAlgorithm):
    """
    Interpolate using a second order Lagrange polynomial.
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
        super().__init__(grid, values, interp, **kwargs)
        self.k = 3
        self._name = 'lagrange2'

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

        # Extrapolate high
        ngrid = len(grid)
        if idx > ngrid - 3:
            idx = ngrid - 3

        derivs = np.empty(len(x), dtype=dtype)

        xx1 = x[0] - grid[idx]
        xx2 = x[0] - grid[idx + 1]
        xx3 = x[0] - grid[idx + 2]

        c12 = grid[idx] - grid[idx + 1]
        c13 = grid[idx] - grid[idx + 2]
        c23 = grid[idx + 1] - grid[idx + 2]

        if subtable is not None:
            # Interpolate between values that come from interpolating the subtables in the
            # subsequent dimensions.
            nx = len(x)
            slice_idx.append(slice(idx, idx + 3))

            tshape = self.values[tuple(slice_idx)].shape
            nshape = list(tshape[:-nx])
            nshape.append(nx)
            derivs = np.empty(tuple(nshape), dtype=dtype)

            subval, subderiv, _, _ = subtable.evaluate(x[1:], slice_idx=slice_idx)

            q1 = subval[..., 0] / (c12 * c13)
            q2 = subval[..., 1] / (c12 * c23)
            q3 = subval[..., 2] / (c13 * c23)

            dq1_dsub = subderiv[..., 0, :] / (c12 * c13)
            dq2_dsub = subderiv[..., 1, :] / (c12 * c23)
            dq3_dsub = subderiv[..., 2, :] / (c13 * c23)

            derivs[..., 1:] = xx3 * (dq1_dsub * xx2 - dq2_dsub * xx1) + dq3_dsub * xx1 * xx2

        else:
            values = self.values[tuple(slice_idx)]

            nshape = list(values.shape[:-1])
            nshape.append(1)
            derivs = np.empty(tuple(nshape), dtype=dtype)

            q1 = values[..., idx] / (c12 * c13)
            q2 = values[..., idx + 1] / (c12 * c23)
            q3 = values[..., idx + 2] / (c13 * c23)

        derivs[..., 0] = q1 * (2.0 * x[0] - grid[idx + 1] - grid[idx + 2]) - \
            q2 * (2.0 * x[0] - grid[idx] - grid[idx + 2]) + \
            q3 * (2.0 * x[0] - grid[idx] - grid[idx + 1])

        return xx3 * (q1 * xx2 - q2 * xx1) + q3 * xx1 * xx2, derivs, None, None


class InterpLagrange2Semi(InterpAlgorithmSemi):
    """
    Interpolate on a semi structured grid using a second order Lagrange polynomial.
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
        super().__init__(grid, values, interp, **kwargs)
        self.k = 3
        self._name = 'lagrange2'

    def interpolate(self, x):
        """
        Compute the interpolated value over this grid dimension.

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
            Derivative of interpolated values with respect to this independent and child
            independents.
        tuple(ndarray, list)
            Derivative of interpolated values with respect to values for this and subsequent table
            dimensions. Second term is the indices into the value array.
        """
        grid = self.grid
        subtables = self.subtables

        idx, _ = self.bracket(x[0])

        # Complex Step
        if self.values.dtype == complex:
            dtype = self.values.dtype
        else:
            dtype = x.dtype

        # Extrapolate high
        ngrid = len(grid)
        if idx > ngrid - 3:
            idx = ngrid - 3

        derivs = np.empty(len(x), dtype=dtype)

        xx1 = x[0] - grid[idx]
        xx2 = x[0] - grid[idx + 1]
        xx3 = x[0] - grid[idx + 2]

        c12 = grid[idx] - grid[idx + 1]
        c13 = grid[idx] - grid[idx + 2]
        c23 = grid[idx + 1] - grid[idx + 2]

        if subtables is not None:
            # Interpolate between values that come from interpolating the subtables in the
            # subsequent dimensions.
            val0, dx0, dvalue0 = subtables[idx].evaluate(x[1:])
            val1, dx1, dvalue1 = subtables[idx + 1].evaluate(x[1:])
            val2, dx2, dvalue2 = subtables[idx + 2].evaluate(x[1:])

            derivs = np.empty(len(dx0) + 1, dtype=dtype)

            q1 = val0 / (c12 * c13)
            q2 = val1 / (c12 * c23)
            q3 = val2 / (c13 * c23)

            dq1_dsub = dx0 / (c12 * c13)
            dq2_dsub = dx1 / (c12 * c23)
            dq3_dsub = dx2 / (c13 * c23)

            derivs[1:] = xx3 * (dq1_dsub * xx2 - dq2_dsub * xx1) + dq3_dsub * xx1 * xx2

            d_value = None
            if self._compute_d_dvalues:
                dvalue0, idx0 = dvalue0
                dvalue1, idx1 = dvalue1
                dvalue2, idx2 = dvalue2
                n = len(dvalue0)

                d_value = np.empty(n * 3, dtype=dtype)
                d_value[:n] = dvalue0 * xx3 * xx2 / (c12 * c13)
                d_value[n:n * 2] = -dvalue1 * xx3 * xx1 / (c12 * c23)
                d_value[n * 2:n * 3] = dvalue2 * xx1 * xx2 / (c13 * c23)

                idx0.extend(idx1)
                idx0.extend(idx2)
                d_value = (d_value, idx0)

        else:
            values = self.values

            q1 = values[idx] / (c12 * c13)
            q2 = values[idx + 1] / (c12 * c23)
            q3 = values[idx + 2] / (c13 * c23)

            derivs = np.empty(1, dtype=dtype)

            d_value = None
            if self._compute_d_dvalues:
                d_value = np.empty(3, dtype=dtype)
                d_value[0] = xx3 * xx2 / (c12 * c13)
                d_value[1] = -xx3 * xx1 / (c12 * c23)
                d_value[2] = xx1 * xx2 / (c13 * c23)

                d_value = (d_value,
                           [self._idx[idx], self._idx[idx + 1], self._idx[idx + 2]])

        derivs[0] = q1 * (xx2 + xx3) - \
            q2 * (xx1 + xx3) + \
            q3 * (xx1 + xx2)

        return xx3 * (q1 * xx2 - q2 * xx1) + q3 * xx1 * xx2, derivs, d_value