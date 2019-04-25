"""
Simple interpolant based on Andrew Ning's implementation of Akima splines.

Includes derivatives wrt training points.  Akima spline is regenerated during each compute.

https://github.com/andrewning/akima
"""
from six.moves import range
from six import string_types

import numpy as np

from openmdao.api import ExplicitComponent


def abs_smooth_dv(x, x_deriv, delta_x):
    """
    Compute the absolute value in a smooth differentiable manner.

    The valley is rounded off using a quadratic function.

    Parameters
    ----------
    x : float
        Quantity value
    x_deriv : float
        Derivative value
    delta_x : float
        Half width of the rounded section.

    Returns
    -------
    float
        Smooth absolute value of the quantity.
    float
        Smooth absolute value of the derivative.
    """
    if x >= delta_x:
        y_deriv = x_deriv
        y = x

    elif x <= -delta_x:
        y_deriv = -x_deriv
        y = -x

    else:
        y_deriv = 2.0 * x * x_deriv / (2.0 * delta_x)
        y = x**2 / (2.0 * delta_x) + delta_x / 2.0

    return y, y_deriv


class AkimaSplineComp(ExplicitComponent):
    """
    Interpolant component based on Andrew Ning's implementation of Akima splines.

    Assumes a uniform distribution of points. Control points fully span the distribution.
    Output (computational) points can be at either the end points or segment center
    points.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        super(AkimaSplineComp, self).__init__(**kwargs)

        self.x_grid = None
        self.x_cp_grid = None

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('num_control_points', types=int, default=10,
                             desc="Number of control points.")
        self.options.declare('num_points', types=int, default=20,
                             desc="Number of interpolated points.")
        self.options.declare('vec_size', types=int, default=1,
                             desc='The number of independent rows to interpolate.')
        self.options.declare('ycp_name', types=str, default='y_cp',
                             desc="Name to use for the input variable (control points).")
        self.options.declare('y_name', types=str, default='y',
                             desc="Name to use for the output variable (interpolated points).")
        self.options.declare('xcp_name', types=str, default=None, allow_none=True,
                             desc="Name to use for the input grid (control points). When set to"
                             "None (default), then the points will be generated internally.")
        self.options.declare('x_name', types=str, default=None, allow_none=True,
                             desc="Name to use for the output grid (interpolated points).")
        self.options.declare('units', types=string_types, default=None, allow_none=True,
                             desc="Units to use for the y and ycp variables.")
        self.options.declare('x_units', types=string_types, default=None, allow_none=True,
                             desc="Units to use for the x and xcp variables.")
        self.options.declare('delta_x', default=0.1,
                             desc="half-width of the smoothing interval added in the valley of "
                             "absolute-value function this allows the derivatives with respect to "
                             "the data points (dydxpt, dydypt) to also be C1 continuous. Set to "
                             "parameter to 0 to get the original Akima function (but only ifyou "
                             "don't need dydxpt, dydypt")
        self.options.declare('evaluate_at', default='end', values=['end', 'cell_center'],
                             desc="Where the return values are evaluate on the spline. When set "
                             "to end, compute num_points values spanning the full interval set by "
                             "the training points. When set to 'cell_center', compute values at "
                             "centerpoints of num_points cells. Only used if xcp_name is not "
                             "given.")
        self.options.declare('eps', default=1e-30,
                             desc='Value that triggers division-by-zero safeguard.')

    def setup(self):
        """
        Set up the B-spline component.
        """
        opts = self.options
        num_control_points = opts['num_control_points']
        num_points = opts['num_points']
        vec_size = opts['vec_size']
        ycp_name = opts['ycp_name']
        y_name = opts['y_name']
        xcp_name = opts['xcp_name']
        x_name = opts['x_name']
        units = opts['units']
        evaluate_at = opts['evaluate_at']

        if xcp_name is not None:
            self.add_input(xcp_name, val=np.random.rand(num_control_points),
                           units=opts['x_units'])

        else:
            self.x_cp_grid = np.linspace(0., 1., num_control_points)

        if x_name is not None:
            self.add_input(x_name, val=np.random.rand(num_points),
                           units=opts['x_units'])

        else:
            if evaluate_at == 'cell_center':
                x_ends = np.linspace(0., 1., num_points + 1)
                self.x_grid = 0.5 * (x_ends[:-1] + x_ends[1:])

            elif evaluate_at == 'end':
                self.x_grid = np.linspace(0., 1., num_points)

        self.add_input(ycp_name, val=np.random.rand(vec_size, num_control_points),
                       units=units)

        self.add_output(y_name, val=np.random.rand(vec_size, num_points), units=units)

        # Derivatives
        row = np.repeat(np.arange(num_points), num_control_points)
        col = np.tile(np.arange(num_control_points), num_points)

        ntot = num_points * num_control_points
        rows = np.tile(row, vec_size) + np.repeat(num_points * np.arange(vec_size), ntot)
        cols = np.tile(col, vec_size) + np.repeat(num_control_points * np.arange(vec_size), ntot)

        self.declare_partials(of=y_name, wrt=ycp_name, rows=rows, cols=cols)

        if xcp_name is not None:
            rows = np.tile(row, vec_size) + np.repeat(num_points * np.arange(vec_size), ntot)
            cols = np.tile(col, vec_size)

            self.declare_partials(of=y_name, wrt=xcp_name, rows=rows, cols=cols)

        if x_name is not None:
            row_col = np.arange(num_points * vec_size)

            self.declare_partials(of=y_name, wrt=x_name, rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        """
        Compute values at the B-spline interpolation points.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        opts = self.options
        ycp_name = opts['ycp_name']
        y_name = opts['y_name']
        y_cp = inputs[ycp_name]

        if self.x_cp_grid is not None:
            x_cp = self.x_cp_grid
        else:
            x_cp = inputs[opts['xcp_name']]

        if self.x_grid is not None:
            x = self.x_grid
        else:
            x = inputs[opts['x_name']]

        # Train on control points.
        self.akima_setup_dv(x_cp, y_cp[0, :])

        # Evaluate at computational points.
        y = self.akima_iterpolate(x, x_cp)
        outputs[y_name][0, :] = y

    def compute_partials(self, inputs, partials):
        """
        Return the pre-computed partials.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        partials : Jacobian
            sub-jac components written to partials[output_name, input_name]
        """
        opts = self.options
        y_name = opts['y_name']
        ycp_name = opts['ycp_name']
        xcp_name = opts['xcp_name']
        x_name = opts['x_name']

        partials[y_name, ycp_name] = self.dy_dycp.flatten()
        if xcp_name is not None:
            partials[y_name, xcp_name] = self.dy_dxcp.flatten()
        if x_name is not None:
            partials[y_name, x_name] = self.dy_dx.flatten()

    def akima_setup_dv(self, xpt, ypt):
        """
        Train the akima spline and save the derivatives.

        Conversion of fortran function AKIMA_DV.
        """
        opts = self.options
        delta_x = self.options['delta_x']
        eps = self.options['eps']
        ncp = opts['num_control_points']
        nbdirs = 2 * ncp

        xptd = np.vstack([np.eye(ncp, dtype=ypt.dtype),
                          np.zeros((ncp, ncp), dtype=ypt.dtype)])
        yptd = np.vstack([np.zeros((ncp, ncp), dtype=ypt.dtype),
                          np.eye(ncp, dtype=ypt.dtype)])

        md = np.zeros((nbdirs, ncp + 3), dtype=ypt.dtype)
        m = np.zeros((ncp + 3, ), dtype=ypt.dtype)
        td = np.zeros((nbdirs, ncp), dtype=ypt.dtype)
        t = np.zeros((ncp, ), dtype=ypt.dtype)

        # Compute segment slopes
        md[:, 2:ncp + 1] = ((yptd[:, 1:] - yptd[:, :-1]) * (xpt[1:] - xpt[:-1]) -
                            (ypt[1:] - ypt[:-1]) * (xptd[:, 1:] - xptd[:, :-1])) / \
            (xpt[1:] - xpt[:-1]) ** 2

        m[2:ncp + 1] = (ypt[1:] - ypt[:-1]) / (xpt[1:] - xpt[:-1])

        # Estimation for end points.
        md[:, 1] = 2.0 * md[:, 2] - md[:, 3]
        md[:, 0] = 2.0 * md[:, 1] - md[:, 2]
        md[:, ncp + 1] = 2.0 * md[:, ncp] - md[:, ncp - 1]
        md[:, ncp + 2] = 2.0 * md[:, ncp + 1] - md[:, ncp]

        m[1] = 2.0 * m[2] - m[3]
        m[0] = 2.0 * m[1] - m[2]
        m[ncp + 1] = 2.0 * m[ncp] - m[ncp - 1]
        m[ncp + 2] = 2.0 * m[ncp + 1] - m[ncp]

        # Slope at points.
        for i in range(2, ncp + 1):
            m1d = md[:, i - 2]
            m2d = md[:, i - 1]
            m3d = md[:, i]
            m4d = md[:, i + 1]
            arg1d = m4d - m3d

            m1 = m[i - 2]
            m2 = m[i - 1]
            m3 = m[i]
            m4 = m[i + 1]
            arg1 = m4 - m3

            w1, w1d = abs_smooth_dv(arg1, arg1d, delta_x)

            arg1d = m2d - m1d
            arg1 = m2 - m1

            w2, w2d = abs_smooth_dv(arg1, arg1d, delta_x)

            if w1 < eps and w2 < eps:
                # Special case to avoid divide by zero.
                td[:, i - 2] = 0.5 * (m2d + m3d)
                t[i - 2] = 0.5 * (m2 + m3)

            else:
                td[:, i - 2] = ((w1d * m2 + w1 * m2d + w2d * m3 + w2 * m3d) *
                                (w1 + w2) - (w1 * m2 + w2 * m3) * (w1d + w2d)) \
                    / (w1 + w2) ** 2

                t[i - 2] = (w1 * m2 + w2 * m3) / (w1 + w2)

        # Polynomial Coefficients
        dx = xpt[1:] - xpt[:-1]
        dx2 = dx**2
        dxd = xptd[:, 1:] - xptd[:, :-1]
        t1 = t[:-1]
        t2 = t[1:]

        p0 = ypt
        p1 = t1
        p2 = (3.0 * m[2:ncp + 1] - 2.0 * t1 - t2) / dx
        p3 = (t1 + t2 - 2.0 * m[2:ncp + 1]) / dx2

        p0d = yptd
        p1d = td[:, :-1]
        p2d = ((3.0 * md[:, 2:ncp + 1] - 2.0 * td[:, :-1] - td[:, 1:]) * dx -
               (3.0 * m[2:ncp + 1] - 2.0 * t1 - t2) * dxd) / dx2
        p3d = ((td[:, :-1] + td[:, 1:] - 2.0 * md[:, 2:ncp + 1]) * dx2 -
               (t1 + t2 - 2.0 * m[2:ncp + 1]) * 2 * dx * dxd) / (dx2)**2

        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        self.dp0_dxcp = p0d[:ncp, :].T
        self.dp0_dycp = p0d[ncp:, :].T
        self.dp1_dxcp = p1d[:ncp, :].T
        self.dp1_dycp = p1d[ncp:, :].T
        self.dp2_dxcp = p2d[:ncp, :].T
        self.dp2_dycp = p2d[ncp:, :].T
        self.dp3_dxcp = p3d[:ncp, :].T
        self.dp3_dycp = p3d[ncp:, :].T

    def akima_iterpolate(self, x, xcp):
        """
        Interpolate the spline at the given points, returning values and derivatives.

        Conversion of fortran function INTERP.
        """
        opts = self.options
        ncp = opts['num_control_points']
        n = opts['num_points']

        p0 = self.p0
        p1 = self.p1
        p2 = self.p2
        p3 = self.p3

        dp0_dxcp = self.dp0_dxcp
        dp1_dxcp = self.dp1_dxcp
        dp2_dxcp = self.dp2_dxcp
        dp3_dxcp = self.dp3_dxcp
        dp0_dycp = self.dp0_dycp
        dp1_dycp = self.dp1_dycp
        dp2_dycp = self.dp2_dycp
        dp3_dycp = self.dp3_dycp

        j_idx = np.zeros(n, dtype=np.int)
        for i in range(n):

            # Find location in array (use end segments if out of bounds)
            if x[i] < xcp[0]:
                j = 0

            else:
                # Linear search for now
                for j in range(ncp - 2, -1, -1):
                    if x[i] >= xcp[j]:
                        break

            j_idx[i] = j

        # Evaluate polynomial (and derivative)
        dx = x - xcp[j_idx]
        dx2 = dx * dx
        dx3 = dx2 * dx

        y = p0[j_idx] + p1[j_idx] * dx + p2[j_idx] * dx2 + p3[j_idx] * dx3

        dydx = p1[j_idx] + 2.0 * p2[j_idx] * dx + 3.0 * p3[j_idx] * dx2
        self.dy_dx = dydx

        dydxcp = dp0_dxcp[j_idx, :] + np.einsum('ij,i->ij', dp1_dxcp[j_idx, :], dx) + \
            np.einsum('ij,i->ij', dp2_dxcp[j_idx, :], dx2) + \
            np.einsum('ij,i->ij', dp3_dxcp[j_idx, :], dx3)

        for i in range(n):
            j = j_idx[i]
            dydxcp[i, j] -= dydx[i]

        dydycp = dp0_dycp[j_idx, :] + np.einsum('ij,i->ij', dp1_dycp[j_idx, :], dx) + \
            np.einsum('ij,i->ij', dp2_dycp[j_idx, :], dx2) + \
            np.einsum('ij,i->ij', dp3_dycp[j_idx, :], dx3)

        self.dy_dxcp = dydxcp
        self.dy_dycp = dydycp

        return y
