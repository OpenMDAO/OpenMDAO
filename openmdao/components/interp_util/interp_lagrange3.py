"""
Interpolate using a third order Lagrange polynomial.

Based on NPSS implementation.
"""
import numpy as np

from openmdao.components.interp_util.interp_algorithm import InterpAlgorithm


class InterpLagrange3(InterpAlgorithm):
    """
    Interpolate using a third order Lagrange polynomial.
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

        # Complex Step
        if self.values.dtype == np.complex:
            dtype = self.values.dtype
        else:
            dtype = x.dtype

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

            tshape = self.values[tuple(slice_idx)].shape
            nshape = list(tshape[:-nx])
            nshape.append(nx)
            derivs = np.empty(tuple(nshape), dtype=dtype)

            c12 = p1 - p2
            c13 = p1 - p3
            c14 = p1 - p4
            c23 = p2 - p3
            c24 = p2 - p4
            c34 = p3 - p4

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
