"""
Interpolation usng simple B-splines
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from openmdao.components.interp_util.interp_algorithm import InterpAlgorithm

CITATIONS = """
@conference {Hwang2012c,
        title = {GeoMACH: Geometry-Centric MDAO of Aircraft Configurations with High Fidelity},
        booktitle = {Proceedings of the 14th AIAA/ISSMO Multidisciplinary Analysis Optimization
                     Conference},
        year = {2012},
        note = {<p>AIAA 2012-5605</p>},
        month = {September},
        address = {Indianapolis, IN},
        author = {John T. Hwang and Joaquim R. R. A. Martins}
}
"""


def get_bspline_mtx(num_cp, t_vec, order=4):
    """
    Compute matrix of B-spline coefficients.

    Parameters
    ----------
    num_cp : int
        Number of control points.
    t_vec : int
        Interpolated point locations.
    order : int(4)
        B-spline order.

    Returns
    -------
    csr_matrix
        Sparse matrix of B-spline coefficients.
    """
    knots = np.zeros(num_cp + order)
    knots[order - 1:num_cp + 1] = np.linspace(0, 1, num_cp - order + 2)
    knots[num_cp + 1:] = 1.0

    basis = np.zeros(order)
    arange = np.arange(order)

    num_pt = len(t_vec)
    data = np.zeros((num_pt, order))
    rows = np.zeros((num_pt, order), int)
    cols = np.zeros((num_pt, order), int)

    for ipt in range(num_pt):
        t = t_vec[ipt]

        i0 = -1
        for ind in range(order, num_cp + 1):
            if (knots[ind - 1] <= t) and (t < knots[ind]):
                i0 = ind - order
        if t == knots[-1]:
            i0 = num_cp - order

        basis[:] = 0.
        basis[-1] = 1.

        for i in range(2, order + 1):
            ll = i - 1
            j1 = order - ll
            j2 = order
            n = i0 + j1
            if knots[n + ll] != knots[n]:
                basis[j1 - 1] = (knots[n + ll] - t) / (knots[n + ll] - knots[n]) * basis[j1]
            else:
                basis[j1 - 1] = 0.
            for j in range(j1 + 1, j2):
                n = i0 + j
                if knots[n + ll - 1] != knots[n - 1]:
                    basis[j - 1] = (t - knots[n - 1]) / \
                        (knots[n + ll - 1] - knots[n - 1]) * basis[j - 1]
                else:
                    basis[j - 1] = 0.
                if knots[n + ll] != knots[n]:
                    basis[j - 1] += (knots[n + ll] - t) / (knots[n + ll] - knots[n]) * basis[j]
            n = i0 + j2
            if knots[n + ll - 1] != knots[n - 1]:
                basis[j2 - 1] = (t - knots[n - 1]) / \
                    (knots[n + ll - 1] - knots[n - 1]) * basis[j2 - 1]
            else:
                basis[j2 - 1] = 0.

        data[ipt, :] = basis
        rows[ipt, :] = ipt
        cols[ipt, :] = i0 + arange

    data, rows, cols = data.flatten(), rows.flatten(), cols.flatten()

    return csr_matrix((data, (rows, cols)), shape=(num_pt, num_cp))


class InterpBSplines(InterpAlgorithm):
    """
    Interpolate using B-spline.

    Attributes
    ----------
    jac : ndarray
        Matrix of b-spline coefficients.
    """

    def __init__(self, grid, values, interp=None, **kwargs):
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
        super(InterpBSplines, self).__init__(grid, values, interp)

        self._vectorized = True
        self.k = 2
        self._name = 'bsplines'
        self._jac = None

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('order', default=4,
                             desc='B-spline order.')

    def evaluate_vectorized(self, x):
        """
        Interpolate across all table dimensions for all requested samples.

        Parameters
        ----------
        x : ndarray
            The coordinates to sample the gridded data at. First array element is the point to
            interpolate here. Remaining elements are interpolated on sub tables.

        Returns
        -------
        ndarray
            Interpolated values.
        ndarray
            Derivative of interpolated values with respect to this independents.
        ndarray
            Derivative of interpolated values with respect to values.
        ndarray
            Derivative of interpolated values with respect to grid.
        """
        n_cp = len(self.grid)
        if self._jac is None:
            self._jac = get_bspline_mtx(n_cp, x / self.grid[-1], order=self.options['order']).tocoo()

        result = np.einsum('ij,kj->ki', self._jac.toarray(), self.values)

        return result, None, self._jac, None

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
        return self.jac.toarray().flatten()