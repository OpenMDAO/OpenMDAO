"""
This module defines the Python equivalents of the functions used by OTIS for interpolation of
tabular data.

    Authors:  C R Hargraves/ S W Paris:  Boeing Aerospace - OTIS 1 Development
              Rob Falck: NASA GRC - Conversion to Python
"""
from __future__ import division, print_function, absolute_import
from six.moves import range

import numpy as np

from openmdao.components.structured_metamodel_util.grid_interp_base import GridInterpBase
from openmdao.components.structured_metamodel_util.outofbounds_error import OutOfBoundsError

# C provides the dependences of the ith quintic coefficient on the j-1th function derivative
# at the left endpoint.  It is defined as a private module variable so it need only be computed once
C = np.array([0.5, -0.9375, 0.0, 0.625, 0.0, -0.1875,
              0.3125, -0.4375, -0.375, 0.625, 0.0625, -0.1875,
              0.0625, -0.0625, -0.125, 0.125, 0.0625, -0.0625,
              0.5, 0.9375, 0.0, -0.625, 0.0, 0.1875,
              -0.3125, -0.4375, 0.375, 0.625, -0.0625, -0.1875,
              0.0625, 0.0625, -0.125, -0.125, 0.0625, 0.0625]).reshape((6, 6)).T


def _qcoff5(x):
    """
    Precompute coefficients for quintic spline.

    Parameters
    ----------
    x : ndarray
        Independent variable values, monotonically increasing

    Returns
    -------
    ndarray
       Dependence of quintic coefficients for each interval on function values
       at endpoints of interval and immediately adjancent intervals.
    """
    n = len(x)
    b = np.zeros(shape=(6, 4))
    a = np.zeros(shape=(n, 6, 4))

    # 1st interval is quadratic
    el = 0.5 * (x[1] - x[0])
    d32 = x[2] - x[1]
    d21 = x[1] - x[0]
    d31 = x[2] - x[0]
    b[0, 0] = 1.e0
    b[1, 0] = -el*(d31+d21)/(d31*d21)
    b[1, 1] = el*d31/(d32*d21)
    b[1, 2] = -el*d21/(d31*d32)
    b[2, 0] = 2.e0*el*el/(d21*d31)
    b[2, 1] = -2.e0*el*el/(d32*d21)
    b[2, 2] = 2.e0*el*el/(d32*d31)
    b[3, 1] = 1.e0
    b[4, 0] = -el*d32/(d21*d31)
    b[4, 1] = el*(d32-d21)/(d32*d21)
    b[4, 2] = el*d21/(d31*d32)
    b[5, 0] = b[2, 0]
    b[5, 1] = b[2, 1]
    b[5, 2] = b[2, 2]
    a[0, ...] = np.dot(C, b)

    for k in range(3, n):
        el =.5 * (x[k-1]-x[k-2])
        b[:] = 0.0
        d21 = x[k-2]-x[k-3]
        d31 = x[k-1]-x[k-3]
        d32 = x[k-1]-x[k-2]
        d42 = x[k]-x[k-2]
        d43 = x[k]-x[k-1]
        b[0, 1] = 1.e0
        b[1, 0] = -el*d32/(d21*d31)
        b[1, 1] = el*(d32-d21)/(d32*d21)
        b[1, 2] = el*d21/(d31*d32)
        b[3, 2] = 1.e0
        b[4, 1] = -el*d43/(d32*d42)
        b[4, 2] = el*(d43-d32)/(d43*d32)
        b[4, 3] = el*d32/(d42*d43)

        # Second derivative stuff
        b[2, 0] = el*el*2.e0/(d21*d31)
        b[2, 1] = -el*el*2.e0/(d32*d21)
        b[2, 2] = el*el*2.e0/(d31*d32)
        b[5, 1] = el*el*2.e0/(d32*d42)
        b[5, 2] = -el*el*2.e0/(d43*d32)
        b[5, 3] = el*el*2.e0/(d42*d43)
        a[k-2, ...] = np.dot(C, b)

    # Last interval is quadratic
    el = 0.5 * (x[n-1] - x[n-2])
    b[:] = 0.0
    d32 = x[n-2] - x[n-3]
    d42 = x[n-1] - x[n-3]
    d43 = x[n-1] - x[n-2]
    b[0, 2] = 1.0
    b[1, 1] = -el*d43/(d32*d42)
    b[1, 2] = el*(d43-d32)/(d43*d32)
    b[1, 3] = el*d32/(d42*d43)
    b[3, 3] = 1.0
    b[4, 1] = el*d43/(d32*d42)
    b[4, 2] = -el*d42/(d43*d32)
    b[4, 3] = el*(d43+d42)/(d43*d42)
    b[2, 1] = 2.0*el*el/(d32*d42)
    b[2, 2] = -2.0*el*el/(d43*d32)
    b[2, 3] = 2.0*el*el/(d42*d43)
    b[5, 1] = b[2, 1]
    b[5, 2] = b[2, 2]
    b[5, 3] = b[2, 3]
    a[n-2, ...] = np.dot(C, b)
    return a


def _qcoff3(x):
    """
    Precompute coefficients for cubic spline.

    Parameters
    ----------
    x : ndarray
        Independent variable values, monotonically increasing

    Returns
    -------
    ndarray
       Dependence of cubic coefficients for each interval on function values
       at endpoints of interval and immediately adjancent intervals.
    """
    n = len(x)
    b = np.zeros(shape=(6, 4))
    a = np.zeros(shape=(n, 6, 4))

    # 1st interval is quadratic
    el = 0.5 * (x[1]-x[0])
    d32=x[2]-x[1]
    d21=x[1]-x[0]
    d31=x[2]-x[0]
    b[0,0]=1.e0
    b[1,0]=-el*(d31+d21)/(d31*d21)
    b[1,1]= el*d31/(d32*d21)
    b[1,2]=-el*d21/(d31*d32)
    b[2,0]= 2.e0*el*el/(d21*d31)
    b[2,1]=-2.e0*el*el/(d32*d21)
    b[2,2]= 2.e0*el*el/(d32*d31)
    b[3,1]=1.e0
    b[4,0]=-el*d32/(d21*d31)
    b[4,1]= el*(d32-d21)/(d32*d21)
    b[4,2]= el*d21/(d31*d32)
    b[5,0]= b[2,0]
    b[5,1]= b[2,1]
    b[5,2]= b[2,2]
    a[0, ...] = np.dot(C, b)

    for k in range(3,n):
        el=.5 * (x[k-1]-x[k-2])
        b= b * 0.0
        d21=x[k-2]-x[k-3]
        d31=x[k-1]-x[k-3]
        d32=x[k-1]-x[k-2]
        d42=x[k]-x[k-2]
        d43=x[k]-x[k-1]
        b[0,1]= 1.e0
        b[1,0]=-el*d32/(d21*d31)
        b[1,1]= el*(d32-d21)/(d32*d21)
        b[1,2]= el*d21/(d31*d32)
        b[3,2]= 1.e0
        b[4,1]=-el*d43/(d32*d42)
        b[4,2]= el*(d43-d32)/(d43*d32)
        b[4,3]= el*d32/(d42*d43)

        # Second derivative stuff
        b[2,0]=      el*(-4.0*b[1,0]-2.0*b[4,0])/d32
        b[2,1]=-el*el*6.0/(d32*d32)+el*(-4.0*b[1,1]-2.0*b[4,1])/d32
        b[2,2]= el*el*6.0/(d32*d32)+el*(-4.0*b[1,2]-2.0*b[4,2])/d32
        b[2,3]=      el*(-4.0*b[1,3]-2.0*b[4,3])/d32
        b[5,0]=      el*(2.0*b[1,0]+4.0*b[4,0])/d32
        b[5,1]= el*el*6.0/(d32*d32)+el*(2.0*b[1,1]+4.0*b[4,1])/d32
        b[5,2]=-el*el*6.0/(d32*d32)+el*(2.0*b[1,2]+4.0*b[4,2])/d32
        b[5,3]=      el*(2.0*b[1,3]+4.0*b[4,3])/d32
        a[k-2, ...] = np.dot(C, b)

    # Last interval is quadratic
    el = 0.5 * (x[n-1] - x[n-2])
    b= b * 0.0
    d32 = x[n-2] - x[n-3]
    d42 = x[n-1] - x[n-3]
    d43 = x[n-1] - x[n-2]
    b[0,2]= 1.0
    b[1,1]=-el*d43/(d32*d42)
    b[1,2]= el*(d43-d32)/(d43*d32)
    b[1,3]= el*d32/(d42*d43)
    b[3,3]= 1.0
    b[4,1]= el*d43/(d32*d42)
    b[4,2]=-el*d42/(d43*d32)
    b[4,3]= el*(d43+d42)/(d43*d42)
    b[2,1]= 2.0*el*el/(d32*d42)
    b[2,2]=-2.0*el*el/(d43*d32)
    b[2,3]= 2.0*el*el/(d42*d43)
    b[5,1]=b[2,1]
    b[5,2]=b[2,2]
    b[5,3]=b[2,3]
    a[n-2, ...] = np.dot(C, b)
    return a


def _qcoff1(x):
    """
    Precompute coefficients for linear spline.

    Parameters
    ----------
    x : ndarray
        Independent variable values, monotonically increasing

    Returns
    -------
    ndarray
       Dependence of linear coefficients for each interval on function values
       at endpoints of interval and immediately adjancent intervals.
    """
    n = len(x)
    b = np.zeros(shape=(6, 4))
    a = np.zeros(shape=(n, 6, 4))

    b[0, 0] = 1.0
    b[3, 1] = 1.0
    b[1, 0] = -0.5
    b[1, 1] = 0.5
    b[4, 0] = -0.5
    b[4, 1] = 0.5

    a[0, ...] = np.dot(C, b)

    if n <= 2:
        return a

    b[:] = 0.0
    b[0, 1] = 1.0
    b[3, 2] = 1.0
    b[1, 1] = -0.5
    b[1, 2] = 0.5
    b[4, 1] = -0.5
    b[4, 2] = 0.5

    cxbf = np.dot(C, b)

    for k in range(3, n):
        a[k - 2, ...] = cxbf

    b[:] = 0.0
    b[0, 2] = 1.0
    b[3, 3] = 1.0
    b[1, 2] = -0.5
    b[1, 3] = 0.5
    b[4, 2] = -0.5
    b[4, 3] = 0.5

    a[n - 2, ...] = np.dot(C, b)
    return a


def _qcoffc(x):
    """
    Precompute coefficients for chamfered spline.

    Parameters
    ----------
    x : ndarray
        Independent variable values, monotonically increasing

    Returns
    -------
    ndarray
       Dependence of coefficients for chamfered splines for each interval on function values
       at endpoints of interval and immediately adjancent intervals.
    """
    n = len(x)
    b = np.zeros(shape=(6, 4))
    a = np.zeros(shape=(n, 6, 4))

    # 1st interval is quadratic
    el = 0.5 * (x[1] - x[0])
    b[0, 0] = 1.0
    b[3, 1] = 1.0
    b[1, 0] = -0.5
    b[1, 1] = 0.5
    b[4, 0] = 0.5
    b[4, 1] = 0.5
    a[0, ...] = np.dot(C, b)
    if n <= 2:
        return

    for k in range(3, n, 2):
        b[:] = 0.0
        el =.5 * (x[k - 1] - x[k - 2])
        d21 = x[k - 2] - x[k - 3]
        d43 = x[k] - x[k-1]
        b[0, 1]= 1.0
        b[1, 0] = -el/d21
        b[1, 1] = el/d21
        b[3, 2] = 1.0
        b[4, 2] = -el/d43
        b[5, 4] = el/d43
        a[k - 2, ...] = np.dot(C, b)

        b[:] = 0.0
        b[1, 2] = 1.0
        b[4, 3] = 1.0
        b[2, 2] = -0.50
        b[2, 3] = 0.50
        b[5, 2] = -0.50
        b[5, 3] = 0.50
        a[k - 1, ...] = np.dot(C, b)

    b[:] = 0.0
    b[1, 3] = 1.0
    b[4, 4] = 1.0
    b[2, 3] = -0.50
    b[2, 4] = 0.50
    b[5, 3] = -0.50
    b[5, 4] = 0.50
    a[n - 2, ...] = np.dot(C, b)
    return a


class OtisGridInterp(GridInterpBase):
    """
    Interpolation on a regular grid in arbitrary dimensions.

    This method is based on the interpolation code from OTIS.

    The data must be defined on a regular grid; the grid spacing however may be uneven. First,
    third and fifth order spline interpolation are supported. After setting up the interpolator
    object, the interpolation order (*slinear*, *cubic*, and *quintic*) may be chosen at each
    evaluation. Additionally, gradients are provided for the spline interpolation methods.

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
    _coeffs : ndarray
        Cached spline coefficients.
    _g_order : string
        Name of interpolation method used to compute the last gradient.
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

    def __init__(self, points, values, order="slinear", bounds_error=True,
                 fill_value=np.nan, spline_dim_error=True):
        """
        Initialize instance of interpolation class.

        Parameters
        ----------
        points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
            The points defining the regular grid in n dimensions.
        values : array_like, shape (m1, ..., mn, ...)
            The data on the regular grid in n dimensions.
        order : str, optional
            The order of interpolation to perform. Supported are 'slinear',
            'cubic',  and 'quintic'. This parameter will become
            the default for the object's interpolate method. Default is "linear".
        bounds_error : bool, optional
            If True, when interpolated values are requested outside of the
            domain of the input data, a ValueError is raised.
            If False, then `fill_value` is used.
            Default is True (raise an exception).
        fill_value : number, optional
            If provided, the value to use for points outside of the
            interpolation domain. If None, values outside
            the domain are extrapolated. Note that gradient values will always be
            extrapolated rather than set to the fill_value if bounds_error=False
            for any points outside of the interpolation domain.
            Default is `np.nan`.
        spline_dim_error : bool, optional
            If spline_dim_error=True and an order `k` spline interpolation method
            is used, then if any dimension has fewer points than `k` + 1, an error
            will be raised. If spline_dim_error=False, then the spline interpolant
            order will be reduced as needed on a per-dimension basis. Default
            is True (raise an exception).
        """
        super(OtisGridInterp, self).__init__(points, values, order=order,
                                             bounds_error=bounds_error, fill_value=fill_value,
                                             spline_dim_error=spline_dim_error)

        # Cache spline coefficients.
        coeffs = []
        for x in self.grid:
            if order == 'slinear':
                coef = _qcoff1(x)
            elif order == 'cubic':
                coef = _qcoff3(x)
            elif order == 'quintic':
                coef = _qcoff5(x)
            else:
                coef = _qcoffc(x)

            coeffs.append(coef)

        self._coeffs = coeffs

    def _interp_orders(self):
        """
        Method-specific settings for interpolation and for testing.

        Returns
        -------
        list
            Valid interpolation name strings.
        dict
            Configuration object that stores limitations of each interpolation
            order.
        """
        interpolator_configs = {
            "slinear": 1,
            "cubic": 3,
            "quintic": 5,
            #"chamfered": 5,
        }

        all_orders = list(interpolator_configs.keys())

        return all_orders, interpolator_configs

    def orders(self):
        """
        Return a list of valid interpolation order names.

        Returns
        -------
        list
            Valid interpolation name strings.
        """
        return ['slinear', 'cubic', 'quintic', ]#'chamfered']

    def interpolate(self, xi, order=None, compute_gradients=True):
        """
        Interpolate at the sample coordinates.

        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        order : str, optional
            The order of interpolation to perform. Supported are 'slinear', 'cubic', and
            'quintic'. Default is None, which will use the order defined at the construction
            of the interpolation object instance.
        compute_gradients : bool, optional
            If a spline interpolation order is chosen, this determines whether gradient
            calculations should be made and cached. Default is True.

        Returns
        -------
        array_like
            Value of interpolant at all sample points.
        """
        # cache latest evaluation point for gradient method's use later
        self._xi = xi

        order = self.order if order is None else order
        if order not in self._all_orders:
            all_m = ', '.join(['"' + m + '"' for m in self._all_orders])
            raise ValueError('Order"%s" is not defined. Valid order are '
                             '%s.' % (order, all_m))

        if self.bounds_error:
            for i, p in enumerate(xi.T):
                if np.isnan(p).any():
                    raise OutOfBoundsError("One of the requested xi contains a NaN",
                                           i, np.NaN, self.grid[i][0], self.grid[i][-1])

                if not np.logical_and(np.all(self.grid[i][0] <= p),
                                      np.all(p <= self.grid[i][-1])):
                    p1 = np.where(self.grid[i][0] > p)[0]
                    p2 = np.where(p > self.grid[i][-1])[0]
                    # First violating entry is enough to direct the user.
                    violated_idx = set(p1).union(p2).pop()
                    value = p[violated_idx]
                    raise OutOfBoundsError("One of the requested xi is out of bounds",
                                           i, value, self.grid[i][0], self.grid[i][-1])

        # TODO: Vectorize.
        xi = np.atleast_2d(xi)
        n_nodes, nx = xi.shape
        result = np.empty((n_nodes, ))
        derivs = np.empty((n_nodes, nx))

        for j in range(n_nodes):
            val, deriv = self.q_r_evl(self.grid, self._coeffs, self.values, xi[j, :])
            result[j] = val
            derivs[j, :] = deriv.flatten()

        # Cache derivatives
        self.derivs = derivs.flatten()

        # TODO: Support out-of-bounds identification.
        #if not self.bounds_error and self.fill_value is not None:
        #   result[out_of_bounds] = self.fill_value

        return result

    def gradient(self, xi, order=None):
        """
        Return the computed gradients at the specified point.

        The gradients are computed as the interpolation itself is performed,
        but are cached and returned separately by this method.
        Parameters
        ----------
        xi : ndarray of shape (..., ndim)
            The coordinates to sample the gridded data at
        order : str, optional
            The order of interpolation to perform. Supported are 'slinear',
            'cubic', and 'quintic'. Default is None, which will use the order
            defined at the construction of the interpolation object instance.

        Returns
        -------
        gradient : ndarray of shape (..., ndim)
            gradient vector of the gradients of the interpolated values with
            respect to each value in xi
        """
        return self.derivs

    def q_r_evl(self, x_data, coeffs, f_data, x_new):
        """
        Does a 2D quintic spline interpolation.
        The spline coefficients a are generated by the qcoff routines.

        x_data: A list of arrays of values for independent variables at which the points to be
            interpolated are defined.

        coeffs: A list of arrays of quintic coefficient dependencies for the first independent variable.  This
           array should have been generated by a qcoff routine.

        f_data: The values to be interpolated.  Z should be sized len(x1) x len(x2) x len(x3) x len(x4) x nf.
           It is reshaped locally.

        x_new:  The value at the first independent variable at which interpolation is desired.

        Returns:

        f:  If y is a 1D array, f is the interpolated value at h
            If y is a 2D array, f is an array of the interpolated value of each function at h
        """
        lengths = [len(item) for item in x_data]
        nl = len(lengths)

        nf = f_data.size // np.prod(lengths)
        nx = lengths[-1]
        f = np.zeros(nf)
        df_dx = np.zeros((nf, nl))

        lengths.append(nf)
        f_rs = f_data.reshape(lengths)

        interval = self.insrch(x_data[-1], x_new[-1])

        if nx <= 4:
            k1 = 0
            k2 = nx - 1
            ins = interval
            if ins < 0:
                ins = 0
            elif ins >= nx:
                ins = nx - 2
        else:
            # Standard case of more than 4 points
            k1 = interval - 1
            k2 = interval + 2
            ins = 1

            # Upper edge check
            if k2 > nx:

                if nl == 2:
                    k2 = nx - 1
                    k1 = nx - 4
                else:
                    k2 = nx
                    k1 = nx - 3

                ins = 2

            # Lower Edge Check
            elif k1 < 0:
                k1 = 0
                k2 = 3
                ins = 0

        nk = k2 - k1 + 1
        coef_flat = coeffs[-1][k1:k1 + 4, ...]
        for l in range(nf):
            if nl == 2:
                e, de_dx = self.q1evl(x_data[0], coeffs[0], f_rs[..., k1:k1 + nk, l], x_new[0])
            else:
                e, de_dx = self.q_r_evl(x_data[:-1], coeffs[:-1], f_rs[..., k1:k1 + nk, l],
                                        x_new[:-1])
            val, dval_dx = self.qs1evl(x_data[-1][k1:k1+nk], coef_flat, e, de_dx, x_new[-1], ins)
            f[l] = val
            df_dx[l, :] = dval_dx.flatten()

        return f, df_dx

    def insrch(self, xt, x):
        """
        Inputs:
           xt : A 1D array of independent variable values, monotonically increasing

           x  : A floating point value
        Returns:
           k : The interval of xt in which x lies

           k = 0 : x < xt[0]
           k = i : xt[k] <= x < xt[k+1]
           k = len(x)-1: xt[-1] <= x
        """
        lxt = len(xt)

        if x <= xt[0]:
            return 0
        elif x >= xt[-1]:
            return lxt-1

        ilo = 0
        ihi = lxt - 2

        k=ilo+int((x-xt[ilo])*(ihi-ilo)/(xt[ihi+1]-xt[ilo]))

        if k<0 or k>=lxt:
            raise ValueError("Big Trouble in insrch")

        # Interval is to the left
        if x <= xt[k+1]:
            ig = int(k)
            for i in range(ig,-1,-1):
                k = i
                if x > xt[k]:
                    return k

        # Interval is to the right
        else:
            ig = k+1
            for i in range(ig,lxt-1):
                k = i
                if x < xt[k+1]:
                    return k

        raise ValueError("insrch failed to find interval")

    def q1evl(self, x, a, y, h):
        """
        Does a 1D quintic spline interpolation of values defined by the arrays x and y.
        The spline coefficients a are generated by the qcoff routines.

        Arguments:

        x:  An array of monotonically increasing values of the independent variable

        a:  An array of quintic coefficients for each interval of function values

        y:  An array of values at the points in the x array to be interpolated.
            y is of dimension n x m where n is the length of x and m is the number
            of functions to be interpolated

        h:  The independent variable value at which interpolation is desired

        Returns:

        f - If y is a 1D array, f is the interpolated value at h
            If y is a 2D array, f is an array of the interpolated value of each function at h
        """

        n = len(x)
        if len(y.shape) == 1:
            y = y.reshape([len(y), 1])
        m = y.shape[1]

        sk = np.zeros(4)
        f = np.zeros(m)
        df_dh = np.zeros((m, 1))

        if n <= 1:
            f[:] = y[0, :]
            return f, np.zeros((1, ))

        # Find the interval of x on which h lies
        interval = self.insrch(x, h)
        inshft = 0
        if interval <= 0:
            interval = 0
            inshft = 1
        elif interval >= n-2:
            interval = n-2
            inshft = -1

        # Normalize the interval from -1 to 1
        eli = 2.0 / (x[interval+1] - x[interval])
        t = eli * (h - 0.5 * (x[interval+1]+x[interval]))
        k1 = interval - 1 + inshft
        k2 = min(k1+3, n) + 1

        # Assemble local function values
        for j in range(m):
            nterm = -1
            for k in range(k1, k2):
                nterm = nterm + 1
                if k > -1:
                    sk[nterm] = y[k, j]

            e = np.einsum('ij,j->i', a[interval, ...], sk)
            f[j] = e[0] + t*(e[1] + t*(e[2] + t*(e[3] + t*(e[4] + t*e[5]))))
            df_dh[j] = eli * (e[1] + 2 * t * (e[2] + 3 * t * (e[3] + 4 * t *(e[4] + 5 * t *e[5]))))

        return f, df_dh

    def qs1evl(self, x, a, y, dy_dhh, h, interval):
        """
        This is a special version of q1evl that does only 4 points and does not require an interval
        search.

        x:  An array of monotonically increasing values of the independent variable

        a:  An array of quintic coefficients for each interval of function values

        y:  An array of values at the points in the x array to be interpolated

        dy_dhh:  Derivative of points with respect to other independent variables.

        h:  The independent variable value at which interpolation is desired

        interval:  The interval on which h lies
        """
        n = len(x)
        s = np.zeros(4)

        if n <= 1:
            return y[0]

        # Normalize the interval from -1 to 1
        eli = 2.0 / (x[interval + 1] - x[interval])
        t = eli * (h - 0.5 * (x[interval + 1] + x[interval]))
        ns = 0

        if n == 3 and interval > 0:
            ns = 1

        # Assemble local function values
        s[ns:ns + n] = y[:n]

        # Multiply by influence matrix to calculate quintic coefficients
        de_ds = a[interval, ...]
        e = np.einsum('ij,j->i', de_ds, s)
        de_dhh = np.einsum('ij,jk->ik', de_ds, dy_dhh)

        # Calculate function values by evaluating quintic
        f = e[0] + t * (e[1] + t * (e[2] + t * (e[3] + t * (e[4] + t * e[5]))))
        df_dh = eli * (e[1] + 2 * t * (e[2] + 3 * t * (e[3] + 4 * t *(e[4] + 5 * t *e[5]))))
        df_dhh = de_dhh[0] + t * (de_dhh[1] + t * (de_dhh[2] + t * (de_dhh[3] + t * (de_dhh[4] +
                 t * de_dhh[5]))))

        return f, np.hstack((df_dhh, df_dh))






