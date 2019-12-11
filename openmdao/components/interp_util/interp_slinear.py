"""
Interpolate using a linear polynomial.

Based on NPSS implementation.
"""
from __future__ import division, print_function, absolute_import

import numpy as np

from openmdao.components.interp_util.interp_base import InterpTableBase


class InterpLinear(InterpTableBase):
    """
    Interpolate using a linear polynomial.
    """

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
            independents
        """
        grid = self.grid
        subtable = self.subtable

        # Extrapolate high
        if idx == len(grid) - 1:
            idx -= 1

        h = 1.0 / (grid[idx + 1] - grid[idx])

        if subtable is not None:
            # Interpolate between values that come from interpolating the subtables in the
            # subsequent dimensions.
            nx = len(x)
            slice_idx.append(slice(idx, idx + 2))

            tshape = self.values[tuple(slice_idx)].shape
            nshape = list(tshape[:-nx])
            nshape.append(nx)
            derivs = np.empty(tuple(nshape), dtype=x.dtype)

            dtmp, subderiv = subtable.evaluate(x[1:], slice_idx=slice_idx)
            slope = (dtmp[..., 1] - dtmp[..., 0]) * h

            derivs[..., 0] = slope
            dslope_dsub = (subderiv[..., 1, :] - subderiv[..., 0, :]) * h
            derivs[..., 1:] = subderiv[..., 0, :] + (x[0] - grid[idx]) * dslope_dsub

            return dtmp[..., 0] + (x[0] - grid[idx]) * slope, derivs

        else:
            values = self.values[tuple(slice_idx)]
            slope = (values[..., idx + 1] - values[..., idx]) * h

            return values[..., idx] + (x - grid[idx]) * slope, np.expand_dims(slope, axis=-1)