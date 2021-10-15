"""
Trilinear interpolation.

Interpolates on a regular grid by performing trilinear interpolation in a 3D cube. Coefficients for
a given cell are solved once and cached for later.

See https://en.wikipedia.org/wiki/Trilinear_interpolation
"""
import numpy as np

from openmdao.components.interp_util.interp_algorithm import InterpAlgorithmFixed


class InterpTrilinear(InterpAlgorithmFixed):
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
        self._name = 'trilinear'

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
        Compute the trilinear interpolation coefficients for this block.

        Parameters
        ----------
        idx : int
            List of interval indices for x.
        dtype : numpy.dtype
            Determines whether to allocate complex.

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

        # Complex Step
        if self.values.dtype == complex:
            dtype = self.values.dtype
        else:
            dtype = x.dtype

        for j, i_n in enumerate(idx):

            # extrapolate low
            if -1 in i_n:
                extrap_idx = np.where(i_n == -1)[0]
                if len(extrap_idx) > 0:
                    i_n[extrap_idx] = 0

            # extrapolate high
            ngrid = len(grid[j])
            if ngrid - 1 in i_n:
                extrap_idx = np.where(i_n == ngrid - 1)[0]
                if len(extrap_idx) > 0:
                    i_n[extrap_idx] = ngrid - 2

        if self.vec_coeff is None:
            self.coeffs = set()
            grid = self.grid
            self.vec_coeff = np.empty((len(grid[0]), len(grid[1]), len(grid[2]), 8))

        needed = set([item for item in zip(idx[0], idx[1], idx[2])])
        uncached = needed.difference(self.coeffs)
        if len(uncached) > 0:
            unc = np.array(list(uncached))
            uncached_idx = (unc[:, 0], unc[:, 1], unc[:, 2])
            a = self.compute_coeffs_vectorized(uncached_idx, dtype)
            self.vec_coeff[unc[:, 0], unc[:, 1], unc[:, 2], ...] = a
            self.coeffs = self.coeffs.union(uncached)
        a = self.vec_coeff[idx[0], idx[1], idx[2], :]

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
        Compute the trilinear interpolation coefficients for this block.

        Parameters
        ----------
        idx : int
            List of interval indices for x.
        dtype : numpy.dtype
            Determines whether to allocate complex.

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
