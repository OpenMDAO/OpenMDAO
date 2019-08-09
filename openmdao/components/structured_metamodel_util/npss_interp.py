"""
Interpolation method based on Tables in NPSS.
"""
from __future__ import division, print_function, absolute_import
from six.moves import range

import numpy as np

from openmdao.components.structured_metamodel_util.grid_interp_base import GridInterpBase
from openmdao.components.structured_metamodel_util.outofbounds_error import OutOfBoundsError


class NPSSTable(object):
    
    def __init__(self, grid, values):
        

class NPSSGridInterp(GridInterpBase):
    """
    Interpolation on a regular grid in arbitrary dimensions.

    This method is based on the interpolation code from OTIS.

    The data must be defined on a regular grid; the grid spacing however may be uneven.

    Attributes
    ----------
    bounds_error : bool
        If True, when interpolated values are requested outside of the
        domain of the input data, a ValueError is raised.
        If False, then `fill_value` is used.
        Default is True (raise an exception).
    fill_value : float
        If provided, the value to use for points outside of the
        interpolation domain. If None, values outside
        the domain are extrapolated. Note that gradient values will always be
        extrapolated rather than set to the fill_value if bounds_error=False
        for any points outside of the interpolation domain.
        Default is `np.nan`.
    grid : tuple
        Collection of points that determine the regular grid.
    order : string
        Name of interpolation order.
    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.
    _all_gradients : ndarray
        Cache of computed gradients.
    _g_order : string
        Name of interpolation order used to compute the last gradient.
    _interp_config : dict
        Configuration object that stores limitations of each interpolation
        order.
    _ki : list
        Interpolation order to be used in each dimension.
    _spline_dim_error : bool
        If spline_dim_error=True and an order `k` spline interpolation method
        is used, then if any dimension has fewer points than `k` + 1, an error
        will be raised. If spline_dim_error=False, then the spline interpolant
        order will be reduced as needed on a per-dimension basis. Default
        is True (raise an exception).
    _xi : ndarray
        Current evaluation point.
    """