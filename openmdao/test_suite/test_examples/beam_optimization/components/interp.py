"""
Simple B-spline component for interpolation of beam thickness as a function of length along the beam.
"""

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from openmdao.api import ExplicitComponent


def get_bspline_mtx(num_cp, num_pt, order=4, distribution='sine'):
    knots = np.zeros(num_cp + order)
    knots[order-1:num_cp+1] = np.linspace(0, 1, num_cp - order + 2)
    knots[num_cp+1:] = 1.0

    t_vec = np.linspace(0, 1, num_pt)
    if distribution == 'uniform':
        pass
    elif distribution == 'sine':
        t_vec = 0.5 + 0.5 * np.sin(-np.pi / 2. + t_vec * np.pi)

    basis = np.zeros(order)
    arange = np.arange(order)
    data = np.zeros((num_pt, order))
    rows = np.zeros((num_pt, order), int)
    cols = np.zeros((num_pt, order), int)

    for ipt in range(num_pt):
        t = t_vec[ipt]

        i0 = -1
        for ind in range(order, num_cp+1):
            if (knots[ind-1] <= t) and (t < knots[ind]):
                i0 = ind - order
        if t == knots[-1]:
            i0 = num_cp - order

        basis[:] = 0.
        basis[-1] = 1.

        for i in range(2, order+1):
            l = i - 1
            j1 = order - l
            j2 = order
            n = i0 + j1
            if knots[n+l] != knots[n]:
                basis[j1-1] = (knots[n+l] - t) / \
                              (knots[n+l] - knots[n]) * basis[j1]
            else:
                basis[j1-1] = 0.
            for j in range(j1+1, j2):
                n = i0 + j
                if knots[n+l-1] != knots[n-1]:
                    basis[j-1] = (t - knots[n-1]) / \
                                (knots[n+l-1] - knots[n-1]) * basis[j-1]
                else:
                    basis[j-1] = 0.
                if knots[n+l] != knots[n]:
                    basis[j-1] += (knots[n+l] - t) / \
                                  (knots[n+l] - knots[n]) * basis[j]
            n = i0 + j2
            if knots[n+l-1] != knots[n-1]:
                basis[j2-1] = (t - knots[n-1]) / \
                              (knots[n+l-1] - knots[n-1]) * basis[j2-1]
            else:
                basis[j2-1] = 0.

        data[ipt, :] = basis
        rows[ipt, :] = ipt
        cols[ipt, :] = i0 + arange

    data, rows, cols = data.flatten(), rows.flatten(), cols.flatten()

    return csr_matrix((data, (rows, cols)), shape=(num_pt, num_cp))


class BsplinesComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('num_control_points', types=int)
        self.metadata.declare('num_points', types=int)
        self.metadata.declare('bspline_order', 4, types=int)
        self.metadata.declare('in_name', types=str)
        self.metadata.declare('out_name', types=str)
        self.metadata.declare('distribution', 'sine', values = ['sine', 'uniform'])

    def setup(self):
        meta = self.metadata
        num_control_points = meta['num_control_points']
        num_points = meta['num_points']
        bspline_order = meta['bspline_order']
        in_name = meta['in_name']
        out_name = meta['out_name']
        distribution = meta['distribution']

        self.add_input(in_name, val=np.random.rand(num_control_points, ))
        self.add_output(out_name, val=np.random.rand(num_points, ))

        jac = get_bspline_mtx(num_control_points, num_points,
            order=bspline_order, distribution=distribution).tocoo()

        data, rows, cols = jac.data, jac.row, jac.col

        self.jac = csc_matrix((data, (rows, cols)),
            shape=(num_points, num_control_points))

        self.declare_partials(out_name, in_name, val=data, rows=rows, cols=cols)

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        meta = self.metadata
        num_control_points = meta['num_control_points']
        num_points = meta['num_points']

        out_shape = (num_points, )
        in_shape = (num_control_points, )

        out = self.jac * inputs[meta['in_name']].reshape(np.prod(in_shape))
        outputs[meta['out_name']] = out.reshape(out_shape)