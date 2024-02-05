"""
Interpolate using a linear polynomial.

Based on NPSS implementation.
"""
import numpy as np

from openmdao.components.interp_util.interp_algorithm import InterpAlgorithm, \
    InterpAlgorithmSemi, InterpAlgorithmFixed


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
            level = len(slice_idx)
            if level == 0:
                nshape = (nx, )
            else:
                nshape = [2] * level
                nshape.append(nx)

            derivs = np.empty(tuple(nshape), dtype=x.dtype)

            slice_idx.append(slice(idx, idx + 2))
            dtmp, subderiv, _, _ = subtable.evaluate(x[1:], slice_idx=slice_idx)
            slope = (dtmp[..., 1] - dtmp[..., 0]) * h

            derivs[..., 0] = slope
            dslope_dsub = (subderiv[..., 1, :] - subderiv[..., 0, :]) * h
            delx = x[0] - grid[idx]
            derivs[..., 1:] = subderiv[..., 0, :] + delx * dslope_dsub

            return dtmp[..., 0] + delx * slope, derivs, None, None

        else:
            values = self.values[tuple(slice_idx)]
            slope = (values[..., idx + 1] - values[..., idx]) * h

            return values[..., idx] + (x - grid[idx]) * slope, slope[..., None], \
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

            slope = ((val1 - val0) * h).item()

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
                d_value[1] = (h * (x - grid[idx])).item()
                d_value[0] = (1.0 - d_value[1]).item()

                d_value = (d_value, [self._idx[idx], self._idx[idx + 1]])

            return values[idx] + (x - grid[idx]) * slope, slope, d_value, extrap


class Interp1DSlinear(InterpAlgorithmFixed):
    """
    Interpolate on a 1D grid using trilinear interpolation.

    Parameters
    ----------
    grid : tuple(ndarray)
        Tuple (x, ) of grid locations.
    values : ndarray
        Array containing the table values.
    interp : class
        Unused, but kept for API compatibility.
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
        self.k = 2
        self.dim = 1
        self.last_index = [0] * self.dim
        self._name = '1D-slinear'

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
            The coordinate to interpolate on this grid.
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
        x = x[0]
        grid = self.grid[0]
        idx_key = tuple(idx)
        idx = idx[0]

        # Complex Step
        if self.values.dtype == complex:
            dtype = self.values.dtype
        else:
            dtype = x.dtype

        # Extrapolation
        n = len(grid)
        if idx == n - 1:
            idx = n - 2
        elif idx == -1:
            idx = 0

        if idx_key not in self.coeffs:
            self.coeffs[idx_key] = self.compute_coeffs(idx, dtype)
        a = self.coeffs[idx_key]

        val = a[0] + a[1] * (x - grid[idx])

        d_x = np.array([a[1]], dtype=dtype)

        return val, d_x, None, None

    def compute_coeffs(self, idx, dtype):
        """
        Compute the interpolation coefficients for this block.

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
        grid = self.grid[0]
        values = self.values
        a = np.empty(2, dtype=dtype)

        i_x = idx

        x0 = grid[i_x]
        x1 = grid[i_x + 1]

        c0 = values[i_x]
        c1 = values[i_x + 1]

        a[0] = c0
        a[1] = (c1 - c0) / (x1 - x0)

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
        x = x_vec[:, 0]
        grid = self.grid[0]
        vec_size = len(x)
        i_x = idx[0]

        # Complex Step
        if self.values.dtype == complex:
            dtype = self.values.dtype
        else:
            dtype = x.dtype

        # extrapolate low
        i_x[i_x == -1] = 0

        # extrapolate high
        nx = self.values.shape[0]
        i_x[i_x == nx - 1] = nx - 2

        if self.vec_coeff is None:
            self.coeffs = set()
            self.vec_coeff = np.empty((nx, 2), dtype=dtype)

        needed = set([item for item in i_x])
        uncached = needed.difference(self.coeffs)
        if len(uncached) > 0:
            unc = np.array(list(uncached))
            uncached_idx = (unc, )
            a = self.compute_coeffs_vectorized(uncached_idx, dtype)
            self.vec_coeff[unc, ...] = a
            self.coeffs.update(uncached)
        a = self.vec_coeff[i_x, :]

        val = a[:, 0] + a[:, 1] * (x - grid[i_x])

        d_x = np.empty((vec_size, 1), dtype=dtype)
        d_x[:, 0] = a[:, 1]

        return val, d_x, None, None

    def compute_coeffs_vectorized(self, idx, dtype):
        """
        Compute the interpolation coefficients for this block.

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
        grid = self.grid[0]
        values = self.values

        i_x = idx[0]
        vec_size = len(i_x)
        a = np.empty((vec_size, 2))

        x0 = grid[i_x]
        x1 = grid[i_x + 1]

        c0 = values[i_x]
        c1 = values[i_x + 1]

        a[:, 0] = c0
        rec_vol = 1.0 / (x1 - x0)
        a[:, 1] = (c1 - c0) * rec_vol

        return a


class Interp2DSlinear(InterpAlgorithmFixed):
    """
    Interpolate on a 2D grid using trilinear interpolation.

    Parameters
    ----------
    grid : tuple(ndarray)
        Tuple (x, y) of grid locations.
    values : ndarray
        Array containing the table values.
    interp : class
        Unused, but kept for API compatibility.
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
        self.k = 2
        self.dim = 2
        self.last_index = [0] * self.dim
        self._name = '2D-slinear'

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
        x, y = x
        idx_key = tuple(idx)

        # Complex Step
        if self.values.dtype == complex:
            dtype = self.values.dtype
        else:
            dtype = x.dtype

        if idx_key not in self.coeffs:
            self.coeffs[idx_key] = self.compute_coeffs(idx, dtype)
        a = self.coeffs[idx_key]

        val = a[0] + (a[1] + a[3] * y) * x + a[2] * y

        d_x = np.empty((2, ), dtype=dtype)
        d_x[0] = a[1] + y * a[3]
        d_x[1] = a[2] + x * a[3]

        return val, d_x, None, None

    def compute_coeffs(self, idx, dtype):
        """
        Compute the interpolation coefficients for this block.

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
        a = np.empty(4, dtype=dtype)

        i_x, i_y = idx

        # Extrapolation
        n = len(grid[0])
        if i_x == n - 1:
            i_x = n - 2
        elif i_x == -1:
            i_x = 0

        n = len(grid[1])
        if i_y == n - 1:
            i_y = n - 2
        elif i_y == -1:
            i_y = 0

        x0 = grid[0][i_x]
        x1 = grid[0][i_x + 1]
        y0 = grid[1][i_y]
        y1 = grid[1][i_y + 1]

        c00 = values[i_x, i_y]
        c01 = values[i_x, i_y + 1]
        c10 = values[i_x + 1, i_y]
        c11 = values[i_x + 1, i_y + 1]

        a[0] = c00 * x1 * y1 - \
            c01 * x1 * y0 - \
            c10 * x0 * y1 + \
            c11 * x0 * y0

        a[1] = (c10 - c00) * y1 + (c01 - c11) * y0

        a[2] = (c01 - c00) * x1 + (c10 - c11) * x0

        a[3] = c00 + c11 - c01 - c10

        rec_vol = 1.0 / ((x0 - x1) * (y0 - y1))
        return a * rec_vol

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
        x = x_vec[:, 0]
        y = x_vec[:, 1]
        grid = self.grid
        vec_size = len(x)
        i_x, i_y = idx

        # Complex Step
        if self.values.dtype == complex:
            dtype = self.values.dtype
        else:
            dtype = x.dtype

        # extrapolate low
        i_x[i_x == -1] = 0
        i_y[i_y == -1] = 0

        # extrapolate high
        nx, ny = self.values.shape
        i_x[i_x == nx - 1] = nx - 2
        i_y[i_y == ny - 1] = ny - 2

        if self.vec_coeff is None:
            self.coeffs = set()
            self.vec_coeff = np.empty((nx, ny, 4), dtype=dtype)

        needed = set([item for item in zip(i_x, i_y)])
        uncached = needed.difference(self.coeffs)
        if len(uncached) > 0:
            unc = np.array(list(uncached))
            uncached_idx = (unc[:, 0], unc[:, 1])
            a = self.compute_coeffs_vectorized(uncached_idx, dtype)
            self.vec_coeff[unc[:, 0], unc[:, 1], ...] = a
            self.coeffs.update(uncached)
        a = self.vec_coeff[i_x, i_y, :]

        val = a[:, 0] + (a[:, 1] + a[:, 3] * y) * x + a[:, 2] * y

        d_x = np.empty((vec_size, 2), dtype=dtype)
        d_x[:, 0] = a[:, 1] + y * a[:, 3]
        d_x[:, 1] = a[:, 2] + x * a[:, 3]

        return val, d_x, None, None

    def compute_coeffs_vectorized(self, idx, dtype):
        """
        Compute the interpolation coefficients for this block.

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
        vec_size = len(i_y)
        a = np.empty((vec_size, 4), dtype=dtype)

        x0 = grid[0][i_x]
        x1 = grid[0][i_x + 1]
        y0 = grid[1][i_y]
        y1 = grid[1][i_y + 1]

        c00 = values[i_x, i_y]
        c01 = values[i_x, i_y + 1]
        c10 = values[i_x + 1, i_y]
        c11 = values[i_x + 1, i_y + 1]

        a[:, 0] = c00 * x1 * y1 - \
            c01 * x1 * y0 - \
            c10 * x0 * y1 + \
            c11 * x0 * y0

        a[:, 1] = (c10 - c00) * y1 + (c01 - c11) * y0

        a[:, 2] = (c01 - c00) * x1 + (c10 - c11) * x0

        a[:, 3] = c00 + c11 - c01 - c10

        rec_vol = 1.0 / ((x0 - x1) * (y0 - y1))
        return np.einsum('i,ij->ij', rec_vol, a)


class Interp3DSlinear(InterpAlgorithmFixed):
    """
    Interpolate on a 3D grid using trilinear interpolation.

    Parameters
    ----------
    grid : tuple(ndarray)
        Tuple (x, y, z) of grid locations.
    values : ndarray
        Array containing the table values.
    interp : class
        Unused, but kept for API compatibility.
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
        self.k = 2
        self.dim = 3
        self.last_index = [0] * self.dim
        self._name = '3D-slinear'

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
        x, y, z = x
        idx_key = tuple(idx)

        # Complex Step
        if self.values.dtype == complex:
            dtype = self.values.dtype
        else:
            dtype = x.dtype

        if idx_key not in self.coeffs:
            self.coeffs[idx_key] = self.compute_coeffs(idx, dtype)
        a = self.coeffs[idx_key]

        val = a[0] + \
            (a[1] + (a[4] + a[7] * z) * y) * x + \
            a[2] * y + \
            (a[3] + a[5] * x + a[6] * y) * z

        d_x = np.empty((3, ), dtype=dtype)
        d_x[0] = a[1] + y * a[4] + z * (a[5] + y * a[7])
        d_x[1] = a[2] + x * a[4] + z * (a[6] + x * a[7])
        d_x[2] = a[3] + x * a[5] + y * (a[6] + x * a[7])

        return val, d_x, None, None

    def compute_coeffs(self, idx, dtype):
        """
        Compute the interpolation coefficients for this block.

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
        a = np.empty(8, dtype=dtype)

        i_x, i_y, i_z = idx

        # Extrapolation
        n = len(grid[0])
        if i_x == n - 1:
            i_x = n - 2
        elif i_x == -1:
            i_x = 0

        n = len(grid[1])
        if i_y == n - 1:
            i_y = n - 2
        elif i_y == -1:
            i_y = 0

        n = len(grid[2])
        if i_z == n - 1:
            i_z = n - 2
        elif i_z == -1:
            i_z = 0

        x0 = grid[0][i_x]
        x1 = grid[0][i_x + 1]
        y0 = grid[1][i_y]
        y1 = grid[1][i_y + 1]
        z0 = grid[2][i_z]
        z1 = grid[2][i_z + 1]

        c000 = values[i_x, i_y, i_z]
        c100 = values[i_x + 1, i_y, i_z]
        c010 = values[i_x, i_y + 1, i_z]
        c001 = values[i_x, i_y, i_z + 1]
        c110 = values[i_x + 1, i_y + 1, i_z]
        c011 = values[i_x, i_y + 1, i_z + 1]
        c101 = values[i_x + 1, i_y, i_z + 1]
        c111 = values[i_x + 1, i_y + 1, i_z + 1]

        a[0] = -c000 * x1 * y1 * z1 + \
            c001 * x1 * y1 * z0 + \
            c010 * x1 * y0 * z1 - \
            c011 * x1 * y0 * z0 + \
            c100 * x0 * y1 * z1 - \
            c101 * x0 * y1 * z0 - \
            c110 * x0 * y0 * z1 + \
            c111 * x0 * y0 * z0

        a[1] = c000 * y1 * z1 - \
            c001 * y1 * z0 - \
            c010 * y0 * z1 + \
            c011 * y0 * z0 - \
            c100 * y1 * z1 + \
            c101 * y1 * z0 + \
            c110 * y0 * z1 - \
            c111 * y0 * z0

        a[2] = c000 * x1 * z1 - \
            c001 * x1 * z0 - \
            c010 * x1 * z1 + \
            c011 * x1 * z0 - \
            c100 * x0 * z1 + \
            c101 * x0 * z0 + \
            c110 * x0 * z1 - \
            c111 * x0 * z0

        a[3] = c000 * x1 * y1 - \
            c001 * x1 * y1 - \
            c010 * x1 * y0 + \
            c011 * x1 * y0 - \
            c100 * x0 * y1 + \
            c101 * x0 * y1 + \
            c110 * x0 * y0 - \
            c111 * x0 * y0

        a[4] = -c000 * z1 + \
            c001 * z0 + \
            c010 * z1 - \
            c011 * z0 + \
            c100 * z1 - \
            c101 * z0 - \
            c110 * z1 + \
            c111 * z0

        a[5] = -c000 * y1 + \
            c001 * y1 + \
            c010 * y0 - \
            c011 * y0 + \
            c100 * y1 - \
            c101 * y1 - \
            c110 * y0 + \
            c111 * y0

        a[6] = -c000 * x1 + \
            c001 * x1 + \
            c010 * x1 - \
            c011 * x1 + \
            c100 * x0 - \
            c101 * x0 - \
            c110 * x0 + \
            c111 * x0

        a[7] = c000 - c001 - c010 + c011 - c100 + c101 + c110 - c111

        rec_vol = 1.0 / ((x0 - x1) * (y0 - y1) * (z0 - z1))
        return a * rec_vol

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
        x = x_vec[:, 0]
        y = x_vec[:, 1]
        z = x_vec[:, 2]
        grid = self.grid
        vec_size = len(x)
        i_x, i_y, i_z = idx

        # Complex Step
        if self.values.dtype == complex:
            dtype = self.values.dtype
        else:
            dtype = x.dtype

        # extrapolate low
        i_x[i_x == -1] = 0
        i_y[i_y == -1] = 0
        i_z[i_z == -1] = 0

        # extrapolate high
        nx, ny, nz = self.values.shape
        i_x[i_x == nx - 1] = nx - 2
        i_y[i_y == ny - 1] = ny - 2
        i_z[i_z == nz - 1] = nz - 2

        if self.vec_coeff is None:
            self.coeffs = set()
            self.vec_coeff = np.empty((nx, ny, nz, 8), dtype=dtype)

        needed = set([item for item in zip(i_x, i_y, i_z)])
        uncached = needed.difference(self.coeffs)
        if len(uncached) > 0:
            unc = np.array(list(uncached))
            uncached_idx = (unc[:, 0], unc[:, 1], unc[:, 2])
            a = self.compute_coeffs_vectorized(uncached_idx, dtype)
            self.vec_coeff[unc[:, 0], unc[:, 1], unc[:, 2], ...] = a
            self.coeffs.update(uncached)
        a = self.vec_coeff[i_x, i_y, i_z, :]

        val = a[:, 0] + \
            (a[:, 1] + (a[:, 4] + a[:, 7] * z) * y) * x + \
            a[:, 2] * y + \
            (a[:, 3] + a[:, 5] * x + a[:, 6] * y) * z

        d_x = np.empty((vec_size, 3), dtype=dtype)
        d_x[:, 0] = a[:, 1] + y * a[:, 4] + z * (a[:, 5] + y * a[:, 7])
        d_x[:, 1] = a[:, 2] + x * a[:, 4] + z * (a[:, 6] + x * a[:, 7])
        d_x[:, 2] = a[:, 3] + x * a[:, 5] + y * (a[:, 6] + x * a[:, 7])

        return val, d_x, None, None

    def compute_coeffs_vectorized(self, idx, dtype):
        """
        Compute the interpolation coefficients for this block.

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
        vec_size = len(i_y)
        a = np.empty((vec_size, 8), dtype=dtype)

        x0 = grid[0][i_x]
        x1 = grid[0][i_x + 1]
        y0 = grid[1][i_y]
        y1 = grid[1][i_y + 1]
        z0 = grid[2][i_z]
        z1 = grid[2][i_z + 1]

        c000 = values[i_x, i_y, i_z]
        c100 = values[i_x + 1, i_y, i_z]
        c010 = values[i_x, i_y + 1, i_z]
        c001 = values[i_x, i_y, i_z + 1]
        c110 = values[i_x + 1, i_y + 1, i_z]
        c011 = values[i_x, i_y + 1, i_z + 1]
        c101 = values[i_x + 1, i_y, i_z + 1]
        c111 = values[i_x + 1, i_y + 1, i_z + 1]

        a[:, 0] = -c000 * x1 * y1 * z1 + \
            c001 * x1 * y1 * z0 + \
            c010 * x1 * y0 * z1 - \
            c011 * x1 * y0 * z0 + \
            c100 * x0 * y1 * z1 - \
            c101 * x0 * y1 * z0 - \
            c110 * x0 * y0 * z1 + \
            c111 * x0 * y0 * z0

        a[:, 1] = c000 * y1 * z1 - \
            c001 * y1 * z0 - \
            c010 * y0 * z1 + \
            c011 * y0 * z0 - \
            c100 * y1 * z1 + \
            c101 * y1 * z0 + \
            c110 * y0 * z1 - \
            c111 * y0 * z0

        a[:, 2] = c000 * x1 * z1 - \
            c001 * x1 * z0 - \
            c010 * x1 * z1 + \
            c011 * x1 * z0 - \
            c100 * x0 * z1 + \
            c101 * x0 * z0 + \
            c110 * x0 * z1 - \
            c111 * x0 * z0

        a[:, 3] = c000 * x1 * y1 - \
            c001 * x1 * y1 - \
            c010 * x1 * y0 + \
            c011 * x1 * y0 - \
            c100 * x0 * y1 + \
            c101 * x0 * y1 + \
            c110 * x0 * y0 - \
            c111 * x0 * y0

        a[:, 4] = -c000 * z1 + \
            c001 * z0 + \
            c010 * z1 - \
            c011 * z0 + \
            c100 * z1 - \
            c101 * z0 - \
            c110 * z1 + \
            c111 * z0

        a[:, 5] = -c000 * y1 + \
            c001 * y1 + \
            c010 * y0 - \
            c011 * y0 + \
            c100 * y1 - \
            c101 * y1 - \
            c110 * y0 + \
            c111 * y0

        a[:, 6] = -c000 * x1 + \
            c001 * x1 + \
            c010 * x1 - \
            c011 * x1 + \
            c100 * x0 - \
            c101 * x0 - \
            c110 * x0 + \
            c111 * x0

        a[:, 7] = c000 - c001 - c010 + c011 - c100 + c101 + c110 - c111

        rec_vol = 1.0 / ((x0 - x1) * (y0 - y1) * (z0 - z1))
        return np.einsum('i,ij->ij', rec_vol, a)
