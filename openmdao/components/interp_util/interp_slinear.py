"""
Interpolate using a linear polynomial.

Based on NPSS implementation.
"""
import numpy as np

from openmdao.components.interp_util.interp_algorithm import InterpAlgorithm, InterpAlgorithmSemi


class InterpLinear(InterpAlgorithm):
    """
    Interpolate using a linear polynomial.

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
        self.k = 2
        self._name = 'slinear'

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

            dtmp, subderiv, _, _ = subtable.evaluate(x[1:], slice_idx=slice_idx)
            slope = (dtmp[..., 1] - dtmp[..., 0]) * h

            derivs[..., 0] = slope
            dslope_dsub = (subderiv[..., 1, :] - subderiv[..., 0, :]) * h
            derivs[..., 1:] = subderiv[..., 0, :] + (x[0] - grid[idx]) * dslope_dsub

            return dtmp[..., 0] + (x[0] - grid[idx]) * slope, derivs, None, None

        else:
            values = self.values[tuple(slice_idx)]
            slope = (values[..., idx + 1] - values[..., idx]) * h

            return values[..., idx] + (x - grid[idx]) * slope, np.expand_dims(slope, axis=-1), \
                None, None


class InterpLinearSemi(InterpAlgorithmSemi):
    """
    Interpolate on a semi structured grid using a linear polynomial.

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
        self.k = 2
        self._name = 'slinear'

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

        # Extrapolate high
        if idx == len(grid) - 1:
            idx -= 1

        h = 1.0 / (grid[idx + 1] - grid[idx])

        if subtables is not None:
            # Interpolate between values that come from interpolating the subtables in the
            # subsequent dimensions.
            val0, dx0, dvalue0, flag0 = subtables[idx].interpolate(x[1:])
            val1, dx1, dvalue1, flag1 = subtables[idx + 1].interpolate(x[1:])

            # Extrapolation detection.
            # (Not much we can do for linear.)
            extrap = extrap or flag1 or flag0

            slope = (val1 - val0) * h

            derivs = np.empty(len(dx0) + 1, dtype=x.dtype)
            derivs[0] = slope
            dslope_dsub = (dx1 - dx0) * h
            derivs[1:] = dx0 + (x[0] - grid[idx]) * dslope_dsub

            d_value = None
            if self._compute_d_dvalues:
                dvalue0, idx0 = dvalue0
                dvalue1, idx1 = dvalue1
                n = len(dvalue0)

                d_value = np.empty(n * 2, dtype=x.dtype)
                d_value[:n] = dvalue0 * (1.0 - (x[0] - grid[idx]) * h)
                d_value[n:] = dvalue1 * (x[0] - grid[idx]) * h

                idx0.extend(idx1)
                d_value = (d_value, idx0)

            return val0 + (x[0] - grid[idx]) * slope, derivs, d_value, extrap

        else:
            values = self.values
            slope = (values[idx + 1] - values[idx]) * h

            d_value = None
            if self._compute_d_dvalues:
                d_value = np.empty(2, dtype=x.dtype)
                d_value[1] = h * (x - grid[idx])
                d_value[0] = 1.0 - d_value[1]

                d_value = (d_value, [self._idx[idx], self._idx[idx + 1]])

            return values[idx] + (x - grid[idx]) * slope, slope, d_value, extrap
