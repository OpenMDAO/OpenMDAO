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
            self._jac = self.get_bspline_mtx(n_cp, x / self.grid[-1],
                                             order=self.options['order']).tocoo()

        result = np.einsum('ij,kj->ki', self._jac.toarray(), self.values)

        if self._compute_d_dx:
            d_derivs = np.einsum('ijk,lj->lik', self._djac_dx, self.values)
        else:
            d_derivs = None

        return result, d_derivs, self._jac, None

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

    def get_bspline_mtx(self, num_cp, t_vec, order=4):
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
        knots = np.zeros(num_cp + order, dtype=t_vec.dtype)
        knots[order - 1:num_cp + 1] = np.linspace(0, 1, num_cp - order + 2)
        knots[num_cp + 1:] = 1.0

        basis = np.zeros(order, dtype=t_vec.dtype)
        arange = np.arange(order)

        num_pt = len(t_vec)
        data = np.zeros((num_pt, order), dtype=t_vec.dtype)
        rows = np.zeros((num_pt, order), int)
        cols = np.zeros((num_pt, order), int)

        if self._compute_d_dx:
            dbasis_dt = np.zeros((order, num_pt), dtype=t_vec.dtype)
            ddata_dt = np.zeros((num_pt, order, num_pt), dtype=t_vec.dtype)

        for ipt in range(num_pt):
            t = t_vec[ipt]

            i0 = -1
            for ind in range(order, num_cp + 1):
                if (knots[ind - 1].real <= t.real) and (t.real < knots[ind].real):
                    i0 = ind - order
            if t.real == knots[-1].real:
                i0 = num_cp - order

            basis[:] = 0.
            basis[-1] = 1.

            for i in range(2, order + 1):
                ll = i - 1
                j1 = order - ll
                j2 = order
                n = i0 + j1

                if knots[n + ll].real != knots[n].real:
                    basis[j1 - 1] = (knots[n + ll] - t) / (knots[n + ll] - knots[n]) * basis[j1]

                    if self._compute_d_dx:
                        dbasis_dt[j1 - 1, :] = (knots[n + ll] - t) / (knots[n + ll] - knots[n]) * \
                            dbasis_dt[j1, :]
                        dbasis_dt[j1 - 1, ipt] += basis[j1] / (knots[n + ll] - knots[n])

                else:
                    basis[j1 - 1] = 0.
                    if self._compute_d_dx:
                        dbasis_dt[j1 - 1, :] = 0.0

                for j in range(j1 + 1, j2):
                    n = i0 + j

                    if knots[n + ll - 1].real != knots[n - 1].real:
                        basis[j - 1] = (t - knots[n - 1]) / \
                            (knots[n + ll - 1] - knots[n - 1]) * basis[j - 1]

                        if self._compute_d_dx:
                            dbasis_dt[j - 1, :] *= (t - knots[n - 1]) / \
                                (knots[n + ll - 1] - knots[n - 1])
                            dbasis_dt[j - 1, ipt] += basis[j - 1] / (knots[n + ll - 1] - knots[n - 1])

                    else:
                        basis[j - 1] = 0.
                        if self._compute_d_dx:
                            dbasis_dt[j - 1, :] = 0.0

                    if knots[n + ll].real != knots[n].real:
                        basis[j - 1] += (knots[n + ll] - t) / (knots[n + ll] - knots[n]) * basis[j]

                        if self._compute_d_dx:
                            dbasis_dt[j - 1, :] += (knots[n + ll] - t) / \
                                (knots[n + ll] - knots[n]) * dbasis_dt[j, :]
                            dbasis_dt[j - 1, ipt] -= basis[j] / (knots[n + ll] - knots[n])

                n = i0 + j2
                if knots[n + ll - 1].real != knots[n - 1].real:
                    basis[j2 - 1] = (t - knots[n - 1]) / \
                        (knots[n + ll - 1] - knots[n - 1]) * basis[j2 - 1]

                    if self._compute_d_dx:
                        dbasis_dt[j2 - 1, :] *= (t - knots[n - 1]) / \
                            (knots[n + ll - 1] - knots[n - 1])
                        dbasis_dt[j2 - 1, :] +=  basis[j2 - 1] / \
                            (knots[n + ll - 1] - knots[n - 1])

                else:
                    basis[j2 - 1] = 0.
                    if self._compute_d_dx:
                        dbasis_dt[j2 - 1, :] = 0.0

            data[ipt, :] = basis
            rows[ipt, :] = ipt
            cols[ipt, :] = i0 + arange

            if self._compute_d_dx:
                ddata_dt[ipt, ...] = dbasis_dt

        data, rows, cols = data.flatten(), rows.flatten(), cols.flatten()
        if self._compute_d_dx:
            # This is slow. Need a multi-D sparse way to do it.
            ddata_dt = ddata_dt.flatten()
            djac_dx = np.zeros((num_pt, num_cp, num_pt), dtype=t_vec.dtype)
            for val, row, col in zip(ddata_dt, rows, cols):
                djac_dx[row, col, :] = val

            self._djac_dx = djac_dx

        return csr_matrix((data, (rows, cols)), shape=(num_pt, num_cp))