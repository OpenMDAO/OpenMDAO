"""
Interpolate using a cubic spline polynomial.

Based on NPSS implementation.
"""

import numpy as np

from openmdao.components.interp_util.interp_algorithm import InterpAlgorithm


class InterpCubic(InterpAlgorithm):
    """
    Interpolate using a cubic spline.

    Continuity of derivatives between segments is assured, but a linear solution is
    required to attain this.

    Attributes
    ----------
    second_derivs : ndarray
        Cache of all second derivatives for the leaf table only.
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
        super().__init__(grid, values, interp)
        self.second_derivs = None
        self.k = 4
        self._name = 'cubic'

    def compute_coeffs(self, grid, values, x):
        """
        Compute cubic spline coefficients that give continuity of second derivatives.

        This requires solution of a tri-diagonal system, which is done with a forward and
        a reverse pass.

        Parameters
        ----------
        grid : tuple(ndarray)
            Tuple containing x grid locations for this dimension.
        values : ndarray
            Array containing the values to be interpolated.
        x : ndarray
            The coordinates to sample the gridded data at. Only needed to query the dtype for
            complex step.

        Returns
        -------
        ndarray
            Coefficients for cubic spline.
        """
        n = len(grid)

        # Complex Step
        if values.dtype == np.complex:
            dtype = values.dtype
        else:
            dtype = x.dtype

        # Natural spline has second deriv=0 at both ends
        sec_deriv = np.zeros(n, dtype=dtype)
        temp = np.zeros(values.shape, dtype=dtype)

        # Subdiagonal stripe.
        mu = (grid[1:n - 1] - grid[:n - 2]) / (grid[2:] - grid[:n - 2])

        # Right hand sides.
        vdiff = (values[..., 1:] - values[..., :n - 1]) / (grid[1:] - grid[:n - 1])
        tmp = 6.0 * (vdiff[..., 1:] - vdiff[..., :n - 2]) / (grid[2:] - grid[:n - 2])

        for i in range(1, n - 1):
            prtl = mu[i - 1] * sec_deriv[..., i - 1] + 2.0
            sec_deriv[i] = (mu[i - 1] - 1.0) / prtl
            temp[..., i] = (tmp[..., i - 1] - mu[i - 1] * temp[..., i - 1]) / prtl

        sec_deriv = np.array(np.broadcast_to(sec_deriv, temp.shape), dtype=dtype)

        for i in range(n - 2, 0, -1):
            sec_deriv[..., i] = sec_deriv[..., i] * sec_deriv[..., i + 1] + temp[..., i]

        return sec_deriv

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

        # Extrapolate high
        if idx == len(grid) - 1:
            idx -= 1

        if subtable is not None:
            # Interpolate between values that come from interpolating the subtables in the
            # subsequent dimensions.
            nx = len(x)

            values, subderivs, _, _ = subtable.evaluate(x[1:], slice_idx=slice_idx)
            sec_deriv = self.compute_coeffs(grid, values, x)

            step = grid[idx + 1] - grid[idx]
            r_step = 1.0 / step
            a = (grid[idx + 1] - x[0]) * r_step
            b = (x[0] - grid[idx]) * r_step
            fact = 1.0 / 6.0

            interp_values = a * values[..., idx] + b * values[..., idx + 1] + \
                ((a * a * a - a) * sec_deriv[..., idx] +
                 (b * b * b - b) * sec_deriv[..., idx + 1]) * (step * step * fact)

            # Derivatives

            tshape = list(interp_values.shape)
            tshape.append(nx)
            derivs = np.empty(tuple(tshape), dtype=x.dtype)

            derivs[..., 0] = r_step * (values[..., idx + 1] - values[..., idx]) + \
                (((3.0 * b * b - 1) * sec_deriv[..., idx + 1] -
                  (3.0 * a * a - 1) * sec_deriv[..., idx]) * (step * fact))

            if nx == 2:
                dsec = self.compute_coeffs(grid, subderivs, x)
                derivs[..., 1] = ((a * a * a - a) * dsec[..., idx] +
                                  (b * b * b - b) * dsec[..., idx + 1]) * (step * step * fact)

                derivs[..., 1] += a * subderivs[..., idx] + b * subderivs[..., idx + 1]

            else:
                dsec = self.compute_coeffs(grid, np.swapaxes(subderivs, -1, -2), x)
                derivs[..., 1:] = ((a * a * a - a) * dsec[..., idx] +
                                   (b * b * b - b) * dsec[..., idx + 1]) * (step * step * fact)

                derivs[..., 1:] += a * subderivs[..., idx, :] + b * subderivs[..., idx + 1, :]

            return interp_values, derivs, None, None

        values = self.values

        # The coefficients can be cached because it is computed for every grid segment in
        # the model.
        if self.second_derivs is None:
            self.second_derivs = self.compute_coeffs(grid, values, x)
        sec_deriv = self.second_derivs

        # Perform the interpolation
        step = grid[idx + 1] - grid[idx]
        r_step = 1.0 / step
        a = (grid[idx + 1] - x) * r_step
        b = (x - grid[idx]) * r_step
        fact = 1.0 / 6.0

        val = a * values[..., idx] + b * values[..., idx + 1] + \
            ((a * a * a - a) * sec_deriv[..., idx] +
             (b * b * b - b) * sec_deriv[..., idx + 1]) * (step * step * fact)

        deriv = r_step * (values[..., idx + 1] - values[..., idx]) + \
            ((3.0 * b * b - 1) * sec_deriv[..., idx + 1] -
             (3.0 * a * a - 1) * sec_deriv[..., idx]) * (step * fact)

        return val, deriv, None, None
