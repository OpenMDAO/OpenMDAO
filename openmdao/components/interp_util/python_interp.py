"""
Base class for interpolation methods that calculate values for each dimension independently.

Based on Tables in NPSS, and was added to bridge the gap between some of the slower scipy
implementations.
"""
from __future__ import division, print_function, absolute_import
from six.moves import range
from six import iteritems

import numpy as np

from openmdao.components.interp_util.grid_interp_base import GridInterpBase
from openmdao.components.interp_util.interp_akima import InterpAkima
from openmdao.components.interp_util.interp_cubic import InterpCubic
from openmdao.components.interp_util.interp_lagrange2 import InterpLagrange2
from openmdao.components.interp_util.interp_lagrange3 import InterpLagrange3
from openmdao.components.interp_util.interp_slinear import InterpLinear


class PythonGridInterp(GridInterpBase):
    """
    Interpolation on a regular grid in arbitrary dimensions.

    This class includes methods based on the interpolation code from NPSS implemented solely in
    Python.

    The data must be defined on a regular grid; the grid spacing however may be uneven.

    Attributes
    ----------
    bounds_error : bool
        If True, when interpolated values are requested outside of the domain of the input data,
        a ValueError is raised. If False, then the methods are allowed to extrapolate.
        Default is True (raise an exception).
    grid : tuple
        Collection of points that determine the regular grid.
    interp_method : string
        Name of interpolation method.
    table : <InterpTable>
        Table object that contains algorithm that performs the interpolation.
    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.
    _all_gradients : ndarray
        Cache of computed gradients.
    _interp_config : dict
        Configuration object that stores the number of points required for each interpolation
        method.
    _interp : class
        Class specified as interpolation algorithm, used to regenerate if needed.
    _xi : ndarray
        Cache of current evaluation point.
    _interp_options : dict
        Dictionary of interpolator options
    """

    def __init__(self, points, values, interp_method="slinear", bounds_error=True, **kwargs):
        """
        Initialize instance of interpolation class.

        Parameters
        ----------
        points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
            The points defining the regular grid in n dimensions.
        values : array_like, shape (m1, ..., mn, ...)
            The data on the regular grid in n dimensions.
        interp_method : str, optional
            Name of interpolation method.
        bounds_error : bool, optional
            If True, when interpolated values are requested outside of the domain of the input
            data, a ValueError is raised. If False, then the methods are allowed to extrapolate.
            Default is True (raise an exception).
        **kwargs : dict
            Interpolator-specific options to pass onward.
        """
        super(PythonGridInterp, self).__init__(points, values, interp_method=interp_method,
                                               bounds_error=bounds_error)

        k = self._interp_config[interp_method]
        for i, p in enumerate(points):
            n_p = len(p)
            if n_p < k:
                raise ValueError("There are %d points in dimension %d,"
                                 " but method %s requires at least %d "
                                 "points per "
                                 "dimension."
                                 "" % (n_p, i, interp_method, k + 1))

        # Cache spline coefficients.
        if interp_method == 'slinear':
            interp = InterpLinear
        elif interp_method == 'lagrange2':
            interp = InterpLagrange2
        elif interp_method == 'lagrange3':
            interp = InterpLagrange3
        elif interp_method == 'cubic':
            interp = InterpCubic
        else:
            interp = InterpAkima

        self._interp = interp
        self._interp_options = kwargs
        self.table = interp(self.grid, self.values, interp, **kwargs)

    def _interp_methods(self):
        """
        Method-specific settings for interpolation and for testing.

        Returns
        -------
        list
            Valid interpolation name strings.
        dict
            Configuration object that stores the number of points required for each method.
        """
        interpolator_configs = {
            "slinear": 2,
            "lagrange2": 3,
            "lagrange3": 4,
            "akima": 4,
            "cubic": 4,
        }

        all_methods = list(interpolator_configs.keys())

        return all_methods, interpolator_configs

    def interpolate(self, xi):
        """
        Interpolate at the sample coordinates.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data.

        Returns
        -------
        ndarray
            Value of interpolant at all sample points.
        """
        super(PythonGridInterp, self).interpolate(xi)
        interp = self._interp

        xi = np.atleast_2d(self._xi)
        n_nodes, nx = xi.shape
        result = np.empty((n_nodes, ), dtype=xi.dtype)
        derivs = np.empty((n_nodes, nx), dtype=xi.dtype)

        # TODO: it might be possible to vectorize over n_nodes.
        for j in range(n_nodes):
            if self.training_data_gradients:
                # If the table values are inputs, then we need to create a new table each time.
                self.table = interp(self.grid, self.values, interp, **self._interp_options)

            val, deriv = self.table.evaluate(xi[j, :])
            result[j] = val
            derivs[j, :] = deriv.flatten()

        # Cache derivatives
        self._all_gradients = derivs

        return result

    def training_gradients(self, pt):
        """
        Compute the training gradient for the vector of training points.

        Parameters
        ----------
        pt : ndarray
            Training point values.

        Returns
        -------
        ndarray
            Gradient of output with respect to training point values.
        """
        grid = self.grid
        interp = self._interp

        for i, axis in enumerate(self.grid):
            ngrid = axis.size
            values = np.zeros(ngrid)
            deriv_i = np.zeros(ngrid)

            for j in range(ngrid):
                values[j] = 1.0
                table = interp([grid[i]], values, self._interp, **self._interp_options)
                deriv_i[j], _ = table.evaluate(pt[i:i + 1])
                values[j] = 0.0

            if i == 0:
                deriv_running = deriv_i.copy()
            else:
                deriv_running = np.outer(deriv_running, deriv_i)

        return deriv_running
