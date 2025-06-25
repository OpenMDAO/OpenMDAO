"""
Interpolate using a second order Lagrange polynomial.

Based on NPSS implementation.
"""
import numpy as np

from openmdao.components.interp_util.interp_algorithm import InterpAlgorithm, \
    InterpAlgorithmSemi, InterpAlgorithmFixed


class InterpLagrange2(InterpAlgorithm):
    """
    Interpolate using a second order Lagrange polynomial.

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
        self.k = 3
        self._name = 'lagrange2'

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

        # Extrapolate high
        ngrid = len(grid)
        if idx > ngrid - 3:
            idx = ngrid - 3

        derivs = np.empty(len(x), dtype=dtype)

        if subtables is not None:
            # Interpolate between values that come from interpolating the subtables in the
            # subsequent dimensions.
            val0, dx0, dvalue0, flag0 = subtables[idx].interpolate(x[1:])
            val1, dx1, dvalue1, flag1 = subtables[idx + 1].interpolate(x[1:])
            val2, dx2, dvalue2, flag2 = subtables[idx + 2].interpolate(x[1:])

            # Extrapolation detection.
            flags = (flag0, flag1, flag2)
            if extrap or flags == (False, False, False):
                # If we are already extrapolating, no change needed.
                # If no sub-points are extrapolating, no change needed.
                pass
            elif flags == (False, False, True) and idx > 0:
                # We are near the right edge of our sub-region, so slide to the left.
                idx -= 1
                val_a, dx_a, dvalue_a, flag_a = subtables[idx].interpolate(x[1:])
                if flag_a:
                    # Nothing we can do; there just aren't enough points here.
                    idx += 1
                    extrap = True
                else:
                    val2, dx2, dvalue2 = val1, dx1, dvalue1
                    val1, dx1, dvalue1 = val0, dx0, dvalue0
                    val0, dx0, dvalue0 = val_a, dx_a, dvalue_a
            else:
                # All other cases, we are in an extrapolation sub-region.
                extrap = True

        xx1 = (x[0] - grid[idx]).item()
        xx2 = (x[0] - grid[idx + 1]).item()
        xx3 = (x[0] - grid[idx + 2]).item()

        c12 = (grid[idx] - grid[idx + 1]).item()
        c13 = (grid[idx] - grid[idx + 2]).item()
        c23 = (grid[idx + 1] - grid[idx + 2]).item()

        if subtables is not None:

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

        return xx3 * (q1 * xx2 - q2 * xx1) + q3 * xx1 * xx2, derivs, d_value, extrap


class Interp3DLagrange2(InterpAlgorithmFixed):
    """
    Interpolate on a fixed 3D grid using a second order Lagrange polynomial.

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

    Attributes
    ----------
    coeffs : dict of ndarray
        Cache of all computed coefficients.
    vec_coeff : None or ndarray
        Cache of all computed coefficients when running vectorized.
    """

    def __init__(self, grid, values, interp, **kwargs):
        """
        Initialize table and subtables.
        """
        super().__init__(grid, values, interp)
        self.coeffs = {}
        self.vec_coeff = None
        self.k = 3
        self.dim = 3
        self.last_index = [0] * self.dim
        self._name = '3D-lagrange2'
        self._vectorized = False

    def vectorized(self, x):
        """
        Return whether this table will be run vectorized for the given requested input.

        Parameters
        ----------
        x : float
            Value of new independent to interpolate.

        Returns
        -------
        bool
            Returns True if this table can be run vectorized.
        """
        # If we only have 1 point, use the non-vectorized implementation, which has faster
        # bracketing than the numpy version.
        return x.shape[0] > 1

    def interpolate(self, x, idx):
        """
        Compute the interpolated value.

        Parameters
        ----------
        x : ndarray
            The coordinates to sample the gridded data at. First array element is the point to
            interpolate here. Remaining elements are interpolated on sub tables.
        idx : int
            Interval index for x.

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
        i_x, i_y, i_z = idx

        # Extrapolation
        # Shift if we don't have 2 points on each side.
        n = len(grid[0])
        if i_x > n - 3:
            i_x = n - 3
        elif i_x < 0:
            i_x = 0

        n = len(grid[1])
        if i_y > n - 3:
            i_y = n - 3
        elif i_y < 0:
            i_y = 0

        n = len(grid[2])
        if i_z > n - 3:
            i_z = n - 3
        elif i_z < 0:
            i_z = 0

        idx = (i_x, i_y, i_z)

        # Complex Step
        if self.values.dtype == complex:
            dtype = self.values.dtype
        else:
            dtype = x.dtype

        if idx not in self.coeffs:
            self.coeffs[idx] = self.compute_coeffs(idx, dtype)
        a = self.coeffs[idx]

        x, y, z = x

        # Taking powers of the "deltas" instead of the actual table inputs eliminates numerical
        # problems that arise from the scaling of each axis.
        x = x - grid[0][i_x]
        y = y - grid[1][i_y]
        z = z - grid[2][i_z]

        # Compute interpolated value using the 16 coefficients.

        xx = np.array([1.0, x, x * x], dtype=dtype)
        yy = np.array([1.0, y, y * y], dtype=dtype)
        zz = np.array([1.0, z, z * z], dtype=dtype)
        val = np.einsum('ijk,i,j,k->', a, xx, yy, zz)

        # Compute derivatives using the 27 coefficients.

        dx = np.array([0.0, 1.0, 2.0 * x])
        dy = np.array([0.0, 1.0, 2.0 * y])
        dz = np.array([0.0, 1.0, 2.0 * z])

        d_x = np.empty((3, ), dtype=dtype)
        d_x[0] = np.einsum('i,j,k,ijk->', dx, yy, zz, a)
        d_x[1] = np.einsum('i,j,k,ijk->', xx, dy, zz, a)
        d_x[2] = np.einsum('i,j,k,ijk->', xx, yy, dz, a)

        return val, d_x, None, None

    def compute_coeffs(self, idx, dtype):
        """
        Compute the tri-lagrange3 interpolation coefficients for this block.

        Parameters
        ----------
        idx : int
            List of interval indices for x.
        dtype : object
            The dtype for vector allocation; used for complex step.

        Returns
        -------
        ndarray
            Interpolation coefficients.
        """
        grid = self.grid
        values = self.values

        i_x, i_y, i_z = idx

        x = grid[0]
        y = grid[1]
        z = grid[2]
        x1, x2, x3 = x[i_x:i_x + 3]
        y1, y2, y3 = y[i_y:i_y + 3]
        z1, z2, z3 = z[i_z:i_z + 3]

        cx12 = x1 - x2
        cx13 = x1 - x3
        cx23 = x2 - x3

        cy12 = y1 - y2
        cy13 = y1 - y3
        cy23 = y2 - y3

        cz12 = z1 - z2
        cz13 = z1 - z3
        cz23 = z2 - z3

        # Normalize for numerical stability
        x2 -= x1
        x3 -= x1

        y2 -= y1
        y3 -= y1

        z2 -= z1
        z3 -= z1

        termx = np.array([[x2 * x3,
                          0.0,
                          0.0],
                          [x2 + x3,
                           x3,
                           x2],
                          [1.0 / (cx12 * cx13),
                           -1.0 / (cx12 * cx23),
                           1.0 / (cx13 * cx23)]], dtype=dtype)

        termy = np.array([[y2 * y3,
                          0.0,
                          0.0],
                          [y2 + y3,
                           y3,
                           y2],
                          [1.0 / (cy12 * cy13),
                           -1.0 / (cy12 * cy23),
                           1.0 / (cy13 * cy23)]], dtype=dtype)

        termz = np.array([[z2 * z3,
                          0.0,
                          0.0],
                          [z2 + z3,
                           z3,
                           z2],
                          [1.0 / (cz12 * cz13),
                           -1.0 / (cz12 * cz23),
                           1.0 / (cz13 * cz23)]], dtype=dtype)

        termx[1, :] *= -termx[2, :]
        termy[1, :] *= -termy[2, :]
        termz[1, :] *= -termz[2, :]

        termx[0, :] *= termx[2, :]
        termy[0, :] *= termy[2, :]
        termz[0, :] *= termz[2, :]

        all_val = values[i_x: i_x + 3, i_y: i_y + 3, i_z: i_z + 3]

        # There are 27 coefficients to compute, and each of them is a sum of 27 terms. These come
        # from multiplying the expression for lagrange interpolation, the  core of which is:
        # (x-x1)(x-x2)(x-x3)(y-y1)(y-y2)(y-y3)(z-z1)(z-z2)(z-z3)
        # and expressing it in terms of powers of x, y, and z.
        # This can efficiently be done in a single call to einsum.
        a = np.einsum("mi,nj,pk,ijk->mnp", termx, termy, termz, all_val)

        return a

    def interpolate_vectorized(self, x_vec, idx):
        """
        Compute the interpolated value.

        Parameters
        ----------
        x_vec : ndarray
            The coordinates to interpolate on this grid.
        idx : int
            List of interval indices for x.

        Returns
        -------
        ndarray
            Interpolated values.
        ndarray
            Derivative of interpolated values with respect to independents.
        ndarray
            Derivative of interpolated values with respect to values.
        ndarray
            Derivative of interpolated values with respect to grid.
        """
        grid = self.grid
        i_x, i_y, i_z = idx

        # extrapolate low
        i_x[i_x < 0] = 0
        i_y[i_y < 0] = 0
        i_z[i_z < 0] = 0

        # extrapolate high
        nx, ny, nz = self.values.shape
        i_x[i_x > nx - 3] = nx - 3
        i_y[i_y > ny - 3] = ny - 3
        i_z[i_z > nz - 3] = nz - 3

        # Complex Step
        if self.values.dtype == complex:
            dtype = self.values.dtype
        else:
            dtype = x_vec.dtype

        if self.vec_coeff is None:
            self.coeffs = set()
            grid = self.grid
            self.vec_coeff = np.empty((nx, ny, nz, 3, 3, 3), dtype=dtype)

        needed = set(zip(i_x, i_y, i_z))
        uncached = needed.difference(self.coeffs)
        if len(uncached) > 0:
            unc = np.array(list(uncached))
            uncached_idx = (unc[:, 0], unc[:, 1], unc[:, 2])
            a = self.compute_coeffs_vectorized(uncached_idx, dtype)
            self.vec_coeff[unc[:, 0], unc[:, 1], unc[:, 2], ...] = a
            self.coeffs.update(uncached)
        a = self.vec_coeff[i_x, i_y, i_z, :]

        # Taking powers of the "deltas" instead of the actual table inputs eliminates numerical
        # problems that arise from the scaling of each axis.
        x = x_vec[:, 0] - grid[0][i_x]
        y = x_vec[:, 1] - grid[1][i_y]
        z = x_vec[:, 2] - grid[2][i_z]

        # Compute interpolated value using the 27 coefficients.

        vec_size = len(i_x)
        xx = np.empty((vec_size, 3), dtype=dtype)
        xx[:, 0] = 1.0
        xx[:, 1] = x
        xx[:, 2] = x * x

        yy = np.empty((vec_size, 3), dtype=dtype)
        yy[:, 0] = 1.0
        yy[:, 1] = y
        yy[:, 2] = y * y

        zz = np.empty((vec_size, 3), dtype=dtype)
        zz[:, 0] = 1.0
        zz[:, 1] = z
        zz[:, 2] = z * z

        val = np.einsum('qijk,qi,qj,qk->q', a, xx, yy, zz)

        # Compute derivatives using the 27 coefficients.

        dx = np.empty((vec_size, 2), dtype=dtype)
        dx[:, 0] = 1.0
        dx[:, 1] = 2.0 * x

        dy = np.empty((vec_size, 2), dtype=dtype)
        dy[:, 0] = 1.0
        dy[:, 1] = 2.0 * y

        dz = np.empty((vec_size, 2), dtype=dtype)
        dz[:, 0] = 1.0
        dz[:, 1] = 2.0 * z

        d_x = np.empty((vec_size, 3), dtype=dtype)
        d_x[:, 0] = np.einsum('qijk,qi,qj,qk->q', a[:, 1:, ...], dx, yy, zz)
        d_x[:, 1] = np.einsum('qijk,qi,qj,qk->q', a[:, :, 1:, :], xx, dy, zz)
        d_x[:, 2] = np.einsum('qijk,qi,qj,qk->q', a[:, :, :, 1:], xx, yy, dz)

        return val, d_x, None, None

    def compute_coeffs_vectorized(self, idx, dtype):
        """
        Compute the tri-lagrange3 interpolation coefficients for this block.

        Parameters
        ----------
        idx : int
            List of interval indices for x.
        dtype : object
            The dtype for vector allocation; used for complex step.

        Returns
        -------
        ndarray
            Interpolation coefficients.
        """
        grid = self.grid
        values = self.values
        a = np.zeros((3, 3, 3), dtype=dtype)

        i_x, i_y, i_z = idx
        vec_size = len(i_x)

        x = grid[0]
        y = grid[1]
        z = grid[2]

        x1 = x[i_x]
        x2 = x[i_x + 1]
        x3 = x[i_x + 2]
        y1 = y[i_y]
        y2 = y[i_y + 1]
        y3 = y[i_y + 2]
        z1 = z[i_z]
        z2 = z[i_z + 1]
        z3 = z[i_z + 2]

        cx12 = x1 - x2
        cx13 = x1 - x3
        cx23 = x2 - x3

        cy12 = y1 - y2
        cy13 = y1 - y3
        cy23 = y2 - y3

        cz12 = z1 - z2
        cz13 = z1 - z3
        cz23 = z2 - z3

        # Normalize for numerical stability
        x2 -= x1
        x3 -= x1

        y2 -= y1
        y3 -= y1

        z2 -= z1
        z3 -= z1

        termx = np.empty((vec_size, 3, 3), dtype=dtype)
        termx[:, 0, 0] = x2 * x3
        termx[:, 0, 1] = 0.0
        termx[:, 0, 2] = 0.0
        termx[:, 1, 0] = x2 + x3
        termx[:, 1, 1] = x3
        termx[:, 1, 2] = x2
        termx[:, 2, 0] = 1.0 / (cx12 * cx13)
        termx[:, 2, 1] = -1.0 / (cx12 * cx23)
        termx[:, 2, 2] = 1.0 / (cx13 * cx23)

        termy = np.empty((vec_size, 3, 3), dtype=dtype)
        termy[:, 0, 0] = y2 * y3
        termy[:, 0, 1] = 0.0
        termy[:, 0, 2] = 0.0
        termy[:, 1, 0] = y2 + y3
        termy[:, 1, 1] = y3
        termy[:, 1, 2] = y2
        termy[:, 2, 0] = 1.0 / (cy12 * cy13)
        termy[:, 2, 1] = -1.0 / (cy12 * cy23)
        termy[:, 2, 2] = 1.0 / (cy13 * cy23)

        termz = np.empty((vec_size, 3, 3), dtype=dtype)
        termz[:, 0, 0] = z2 * z3
        termz[:, 0, 1] = 0.0
        termz[:, 0, 2] = 0.0
        termz[:, 1, 0] = z2 + z3
        termz[:, 1, 1] = z3
        termz[:, 1, 2] = z2
        termz[:, 2, 0] = 1.0 / (cz12 * cz13)
        termz[:, 2, 1] = -1.0 / (cz12 * cz23)
        termz[:, 2, 2] = 1.0 / (cz13 * cz23)

        termx[:, 1, :] *= -termx[:, 2, :]
        termy[:, 1, :] *= -termy[:, 2, :]
        termz[:, 1, :] *= -termz[:, 2, :]

        termx[:, 0, :] *= termx[:, 2, :]
        termy[:, 0, :] *= termy[:, 2, :]
        termz[:, 0, :] *= termz[:, 2, :]

        all_val = np.empty((vec_size, 3, 3, 3), dtype=dtype)
        # The only loop in this algorithm, but it doesn't seem to have much impact on time.
        # Broadcasting out the index slices would be a bit complicated.
        for j in range(vec_size):
            all_val[j, ...] = values[i_x[j]: i_x[j] + 3,
                                     i_y[j]: i_y[j] + 3,
                                     i_z[j]: i_z[j] + 3]

        # There are 27 coefficients to compute, and each of them is a sum of 27 terms. These come
        # from multiplying the expression for lagrange interpolation, the  core of which is:
        # (x-x1)(x-x2)(x-x3)(y-y1)(y-y2)(y-y3)(z-z1)(z-z2)(z-z3)
        # and expressing it in terms of powers of x, y, and z.
        # This can efficiently be done in a single call to einsum for all requested cells
        # simultaneously.
        a = np.einsum("qmi,qnj,qpk,qijk->qmnp", termx, termy, termz, all_val)

        return a


class Interp2DLagrange2(InterpAlgorithmFixed):
    """
    Interpolate on a fixed 2D grid using a second order Lagrange polynomial.

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

    Attributes
    ----------
    coeffs : dict of ndarray
        Cache of all computed coefficients.
    vec_coeff : None or ndarray
        Cache of all computed coefficients when running vectorized.
    """

    def __init__(self, grid, values, interp, **kwargs):
        """
        Initialize table and subtables.
        """
        super().__init__(grid, values, interp)
        self.coeffs = {}
        self.vec_coeff = None
        self.k = 3
        self.dim = 2
        self.last_index = [0] * self.dim
        self._name = '2D-lagrange2'
        self._vectorized = False

    def vectorized(self, x):
        """
        Return whether this table will be run vectorized for the given requested input.

        Parameters
        ----------
        x : float
            Value of new independent to interpolate.

        Returns
        -------
        bool
            Returns True if this table can be run vectorized.
        """
        # If we only have 1 point, use the non-vectorized implementation, which has faster
        # bracketing than the numpy version.
        return x.shape[0] > 1

    def interpolate(self, x, idx):
        """
        Compute the interpolated value.

        Parameters
        ----------
        x : ndarray
            The coordinates to sample the gridded data at. First array element is the point to
            interpolate here. Remaining elements are interpolated on sub tables.
        idx : int
            Interval index for x.

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
        i_x, i_y = idx

        # Extrapolation
        # Shift if we don't have 2 points on each side.
        n = len(grid[0])
        if i_x > n - 3:
            i_x = n - 3
        elif i_x < 0:
            i_x = 0

        n = len(grid[1])
        if i_y > n - 3:
            i_y = n - 3
        elif i_y < 0:
            i_y = 0

        idx = (i_x, i_y)

        # Complex Step
        if self.values.dtype == complex:
            dtype = self.values.dtype
        else:
            dtype = x.dtype

        if idx not in self.coeffs:
            self.coeffs[idx] = self.compute_coeffs(idx, dtype)
        a = self.coeffs[idx]

        x, y = x

        # Taking powers of the "deltas" instead of the actual table inputs eliminates numerical
        # problems that arise from the scaling of each axis.
        x = x - grid[0][i_x]
        y = y - grid[1][i_y]

        # Compute interpolated value using the 9 coefficients.

        xx = np.array([1.0, x, x * x], dtype=dtype)
        yy = np.array([1.0, y, y * y], dtype=dtype)
        val = np.einsum('ij,i,j->', a, xx, yy)

        # Compute derivatives using the 9 coefficients.

        dx = np.array([0.0, 1.0, 2.0 * x])
        dy = np.array([0.0, 1.0, 2.0 * y])

        d_x = np.empty((2, ), dtype=dtype)
        d_x[0] = np.einsum('i,j,ij->', dx, yy, a)
        d_x[1] = np.einsum('i,j,ij->', xx, dy, a)

        return val, d_x, None, None

    def compute_coeffs(self, idx, dtype):
        """
        Compute the lagrange3 interpolation coefficients for this block.

        Parameters
        ----------
        idx : int
            List of interval indices for x.
        dtype : object
            The dtype for vector allocation; used for complex step.

        Returns
        -------
        ndarray
            Interpolation coefficients.
        """
        grid = self.grid
        values = self.values

        i_x, i_y = idx

        x = grid[0]
        y = grid[1]
        x1, x2, x3 = x[i_x:i_x + 3]
        y1, y2, y3 = y[i_y:i_y + 3]

        cx12 = x1 - x2
        cx13 = x1 - x3
        cx23 = x2 - x3

        cy12 = y1 - y2
        cy13 = y1 - y3
        cy23 = y2 - y3

        # Normalize for numerical stability
        x2 -= x1
        x3 -= x1

        y2 -= y1
        y3 -= y1

        termx = np.array([[x2 * x3,
                          0.0,
                          0.0],
                          [x2 + x3,
                           x3,
                           x2],
                          [1.0 / (cx12 * cx13),
                           -1.0 / (cx12 * cx23),
                           1.0 / (cx13 * cx23)]], dtype=dtype)

        termy = np.array([[y2 * y3,
                          0.0,
                          0.0],
                          [y2 + y3,
                           y3,
                           y2],
                          [1.0 / (cy12 * cy13),
                           -1.0 / (cy12 * cy23),
                           1.0 / (cy13 * cy23)]], dtype=dtype)

        termx[1, :] *= -termx[2, :]
        termy[1, :] *= -termy[2, :]

        termx[0, :] *= termx[2, :]
        termy[0, :] *= termy[2, :]

        all_val = values[i_x: i_x + 3, i_y: i_y + 3]

        # There are 9 coefficients to compute, and each of them is a sum of 9 terms. These come
        # from multiplying the expression for lagrange interpolation, the  core of which is:
        # (x-x1)(x-x2)(x-x3)(y-y1)(y-y2)(y-y3)
        # and expressing it in terms of powers of x, y.
        # This can efficiently be done in a single call to einsum.
        a = np.einsum("mi,nj,ij->mn", termx, termy, all_val)

        return a

    def interpolate_vectorized(self, x_vec, idx):
        """
        Compute the interpolated value.

        Parameters
        ----------
        x_vec : ndarray
            The coordinates to interpolate on this grid.
        idx : int
            List of interval indices for x.

        Returns
        -------
        ndarray
            Interpolated values.
        ndarray
            Derivative of interpolated values with respect to independents.
        ndarray
            Derivative of interpolated values with respect to values.
        ndarray
            Derivative of interpolated values with respect to grid.
        """
        grid = self.grid
        i_x, i_y = idx

        # extrapolate low
        i_x[i_x < 0] = 0
        i_y[i_y < 0] = 0

        # extrapolate high
        nx, ny = self.values.shape
        i_x[i_x > nx - 3] = nx - 3
        i_y[i_y > ny - 3] = ny - 3

        # Complex Step
        if self.values.dtype == complex:
            dtype = self.values.dtype
        else:
            dtype = x_vec.dtype

        if self.vec_coeff is None:
            self.coeffs = set()
            grid = self.grid
            self.vec_coeff = np.empty((nx, ny, 3, 3), dtype=dtype)

        needed = set(zip(i_x, i_y))
        uncached = needed.difference(self.coeffs)
        if len(uncached) > 0:
            unc = np.array(list(uncached))
            uncached_idx = (unc[:, 0], unc[:, 1])
            a = self.compute_coeffs_vectorized(uncached_idx, dtype)
            self.vec_coeff[unc[:, 0], unc[:, 1], ...] = a
            self.coeffs.update(uncached)
        a = self.vec_coeff[i_x, i_y, :]

        # Taking powers of the "deltas" instead of the actual table inputs eliminates numerical
        # problems that arise from the scaling of each axis.
        x = x_vec[:, 0] - grid[0][i_x]
        y = x_vec[:, 1] - grid[1][i_y]

        # Compute interpolated value using the 9 coefficients.

        vec_size = len(i_x)
        xx = np.empty((vec_size, 3), dtype=dtype)
        xx[:, 0] = 1.0
        xx[:, 1] = x
        xx[:, 2] = x * x

        yy = np.empty((vec_size, 3), dtype=dtype)
        yy[:, 0] = 1.0
        yy[:, 1] = y
        yy[:, 2] = y * y

        val = np.einsum('qij,qi,qj->q', a, xx, yy)

        # Compute derivatives using the 16 coefficients.

        dx = np.empty((vec_size, 2), dtype=dtype)
        dx[:, 0] = 1.0
        dx[:, 1] = 2.0 * x

        dy = np.empty((vec_size, 2), dtype=dtype)
        dy[:, 0] = 1.0
        dy[:, 1] = 2.0 * y

        d_x = np.empty((vec_size, 2), dtype=dtype)
        d_x[:, 0] = np.einsum('qij,qi,qj->q', a[:, 1:, ...], dx, yy)
        d_x[:, 1] = np.einsum('qij,qi,qj->q', a[:, :, 1:], xx, dy)

        return val, d_x, None, None

    def compute_coeffs_vectorized(self, idx, dtype):
        """
        Compute the tri-lagrange3 interpolation coefficients for this block.

        Parameters
        ----------
        idx : int
            List of interval indices for x.
        dtype : object
            The dtype for vector allocation; used for complex step.

        Returns
        -------
        ndarray
            Interpolation coefficients.
        """
        grid = self.grid
        values = self.values
        a = np.zeros((3, 3), dtype=dtype)

        i_x, i_y = idx
        vec_size = len(i_x)

        x = grid[0]
        y = grid[1]

        x1 = x[i_x]
        x2 = x[i_x + 1]
        x3 = x[i_x + 2]
        y1 = y[i_y]
        y2 = y[i_y + 1]
        y3 = y[i_y + 2]

        cx12 = x1 - x2
        cx13 = x1 - x3
        cx23 = x2 - x3

        cy12 = y1 - y2
        cy13 = y1 - y3
        cy23 = y2 - y3

        # Normalize for numerical stability
        x2 -= x1
        x3 -= x1

        y2 -= y1
        y3 -= y1

        termx = np.empty((vec_size, 3, 3), dtype=dtype)
        termx[:, 0, 0] = x2 * x3
        termx[:, 0, 1] = 0.0
        termx[:, 0, 2] = 0.0
        termx[:, 1, 0] = x2 + x3
        termx[:, 1, 1] = x3
        termx[:, 1, 2] = x2
        termx[:, 2, 0] = 1.0 / (cx12 * cx13)
        termx[:, 2, 1] = -1.0 / (cx12 * cx23)
        termx[:, 2, 2] = 1.0 / (cx13 * cx23)

        termy = np.empty((vec_size, 3, 3), dtype=dtype)
        termy[:, 0, 0] = y2 * y3
        termy[:, 0, 1] = 0.0
        termy[:, 0, 2] = 0.0
        termy[:, 1, 0] = y2 + y3
        termy[:, 1, 1] = y3
        termy[:, 1, 2] = y2
        termy[:, 2, 0] = 1.0 / (cy12 * cy13)
        termy[:, 2, 1] = -1.0 / (cy12 * cy23)
        termy[:, 2, 2] = 1.0 / (cy13 * cy23)

        termx[:, 1, :] *= -termx[:, 2, :]
        termy[:, 1, :] *= -termy[:, 2, :]

        termx[:, 0, :] *= termx[:, 2, :]
        termy[:, 0, :] *= termy[:, 2, :]

        all_val = np.empty((vec_size, 3, 3), dtype=dtype)
        # The only loop in this algorithm, but it doesn't seem to have much impact on time.
        # Broadcasting out the index slices would be a bit complicated.
        for j in range(vec_size):
            all_val[j, ...] = values[i_x[j]: i_x[j] + 3,
                                     i_y[j]: i_y[j] + 3]

        # There are 9 coefficients to compute, and each of them is a sum of 9 terms. These come
        # from multiplying the expression for lagrange interpolation, the  core of which is:
        # (x-x1)(x-x2)(x-x3)(y-y1)(y-y2)(y-y3)
        # and expressing it in terms of powers of x, y, and z.
        # This can efficiently be done in a single call to einsum for all requested cells
        # simultaneously.
        a = np.einsum("qmi,qnj,qij->qmn", termx, termy, all_val)

        return a


class Interp1DLagrange2(InterpAlgorithmFixed):
    """
    Interpolate on a fixed 1D grid using a second order Lagrange polynomial.

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

    Attributes
    ----------
    coeffs : dict of ndarray
        Cache of all computed coefficients.
    vec_coeff : None or ndarray
        Cache of all computed coefficients when running vectorized.
    """

    def __init__(self, grid, values, interp, **kwargs):
        """
        Initialize table and subtables.
        """
        super().__init__(grid, values, interp)
        self.coeffs = {}
        self.vec_coeff = None
        self.k = 3
        self.dim = 1
        self.last_index = [0] * self.dim
        self._name = '2D-lagrange2'
        self._vectorized = False

    def vectorized(self, x):
        """
        Return whether this table will be run vectorized for the given requested input.

        Parameters
        ----------
        x : float
            Value of new independent to interpolate.

        Returns
        -------
        bool
            Returns True if this table can be run vectorized.
        """
        # If we only have 1 point, use the non-vectorized implementation, which has faster
        # bracketing than the numpy version.
        return x.shape[0] > 1

    def interpolate(self, x, idx):
        """
        Compute the interpolated value.

        Parameters
        ----------
        x : ndarray
            The coordinates to sample the gridded data at. First array element is the point to
            interpolate here. Remaining elements are interpolated on sub tables.
        idx : int
            Interval index for x.

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
        i_x = idx[0]

        # Extrapolation
        # Shift if we don't have 2 points on each side.
        n = len(grid[0])
        if i_x > n - 3:
            i_x = n - 3
        elif i_x < 0:
            i_x = 0

        # Complex Step
        if self.values.dtype == complex:
            dtype = self.values.dtype
        else:
            dtype = x.dtype

        if i_x not in self.coeffs:
            self.coeffs[i_x] = self.compute_coeffs(i_x, dtype)
        a = self.coeffs[i_x]

        x = x[0]

        # Taking powers of the "deltas" instead of the actual table inputs eliminates numerical
        # problems that arise from the scaling of each axis.
        x = x - grid[0][i_x]

        # Compute interpolated value using the 3 coefficients.
        val = a[0] + x * (a[1] + x * a[2])

        # Compute derivatives using the 3 coefficients.
        d_x = a[1] + x * 2.0 * a[2]

        return val, d_x, None, None

    def compute_coeffs(self, idx, dtype):
        """
        Compute the lagrange3 interpolation coefficients for this block.

        Parameters
        ----------
        idx : int
            List of interval indices for x.
        dtype : object
            The dtype for vector allocation; used for complex step.

        Returns
        -------
        ndarray
            Interpolation coefficients.
        """
        grid = self.grid
        values = self.values

        x = grid[0]
        x1, x2, x3 = x[idx:idx + 3]

        cx12 = x1 - x2
        cx13 = x1 - x3
        cx23 = x2 - x3

        # Normalize for numerical stability
        x2 -= x1
        x3 -= x1

        termx = np.array([[x2 * x3,
                          0.0,
                          0.0],
                          [x2 + x3,
                           x3,
                           x2],
                          [1.0 / (cx12 * cx13),
                           -1.0 / (cx12 * cx23),
                           1.0 / (cx13 * cx23)]], dtype=dtype)

        termx[1, :] *= -termx[2, :]
        termx[0, :] *= termx[2, :]

        all_val = values[idx: idx + 3]

        # There are 3 coefficients to compute, and each of them is a sum of 3 terms. These come
        # from multiplying the expression for lagrange interpolation, the  core of which is:
        # (x-x1)(x-x2)(x-x3)
        # and expressing it in terms of powers of x.
        # This can efficiently be done in a single call to einsum.
        a = np.einsum("mi,i->m", termx, all_val)

        return a

    def interpolate_vectorized(self, x_vec, idx):
        """
        Compute the interpolated value.

        Parameters
        ----------
        x_vec : ndarray
            The coordinates to interpolate on this grid.
        idx : int
            List of interval indices for x.

        Returns
        -------
        ndarray
            Interpolated values.
        ndarray
            Derivative of interpolated values with respect to independents.
        ndarray
            Derivative of interpolated values with respect to values.
        ndarray
            Derivative of interpolated values with respect to grid.
        """
        grid = self.grid
        i_x = idx[0]

        # extrapolate low
        i_x[i_x < 0] = 0

        # extrapolate high
        nx = len(self.values)
        i_x[i_x > nx - 3] = nx - 3

        # Complex Step
        if self.values.dtype == complex:
            dtype = self.values.dtype
        else:
            dtype = x_vec.dtype

        if self.vec_coeff is None:
            self.coeffs = set()
            grid = self.grid
            self.vec_coeff = np.empty((nx, 3), dtype=dtype)

        needed = set(i_x)
        uncached = needed.difference(self.coeffs)
        if len(uncached) > 0:
            uncached_idx = np.array(list(uncached))
            a = self.compute_coeffs_vectorized(uncached_idx, dtype)
            self.vec_coeff[uncached_idx, ...] = a
            self.coeffs.update(uncached)
        a = self.vec_coeff[i_x, :]

        # Taking powers of the "deltas" instead of the actual table inputs eliminates numerical
        # problems that arise from the scaling of each axis.
        x = x_vec[:, 0] - grid[0][i_x]

        # Compute interpolated value using the 9 coefficients.

        vec_size = len(i_x)
        xx = np.empty((vec_size, 3), dtype=dtype)
        xx[:, 0] = 1.0
        xx[:, 1] = x
        xx[:, 2] = x * x

        val = np.einsum('qi,qi->q', a, xx)

        # Compute derivatives using the 16 coefficients.

        dx = np.empty((vec_size, 2), dtype=dtype)
        dx[:, 0] = 1.0
        dx[:, 1] = 2.0 * x

        d_x = np.empty((vec_size, 1), dtype=dtype)
        d_x[:, 0] = np.einsum('qi,qi->q', a[:, 1:], dx)

        return val, d_x, None, None

    def compute_coeffs_vectorized(self, idx, dtype):
        """
        Compute the tri-lagrange3 interpolation coefficients for this block.

        Parameters
        ----------
        idx : int
            List of interval indices for x.
        dtype : object
            The dtype for vector allocation; used for complex step.

        Returns
        -------
        ndarray
            Interpolation coefficients.
        """
        grid = self.grid
        values = self.values
        a = np.zeros((3, ), dtype=dtype)

        vec_size = len(idx)

        x = grid[0]

        x1 = x[idx]
        x2 = x[idx + 1]
        x3 = x[idx + 2]

        cx12 = x1 - x2
        cx13 = x1 - x3
        cx23 = x2 - x3

        # Normalize for numerical stability
        x2 -= x1
        x3 -= x1

        termx = np.empty((vec_size, 3, 3), dtype=dtype)
        termx[:, 0, 0] = x2 * x3
        termx[:, 0, 1] = 0.0
        termx[:, 0, 2] = 0.0
        termx[:, 1, 0] = x2 + x3
        termx[:, 1, 1] = x3
        termx[:, 1, 2] = x2
        termx[:, 2, 0] = 1.0 / (cx12 * cx13)
        termx[:, 2, 1] = -1.0 / (cx12 * cx23)
        termx[:, 2, 2] = 1.0 / (cx13 * cx23)

        termx[:, 1, :] *= -termx[:, 2, :]
        termx[:, 0, :] *= termx[:, 2, :]

        all_val = np.empty((vec_size, 3), dtype=dtype)
        # The only loop in this algorithm, but it doesn't seem to have much impact on time.
        # Broadcasting out the index slices would be a bit complicated.
        for j in range(vec_size):
            all_val[j, ...] = values[idx[j]: idx[j] + 3]

        # There are 9 coefficients to compute, and each of them is a sum of 3 terms. These come
        # from multiplying the expression for lagrange interpolation, the  core of which is:
        # (x-x1)(x-x2)(x-x3)
        # and expressing it in terms of powers of x.
        # This can efficiently be done in a single call to einsum for all requested cells
        # simultaneously.
        a = np.einsum("qmi,qi->qm", termx, all_val)

        return a
