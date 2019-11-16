"""
Simple interpolant based on Andrew Ning's implementation of Akima splines.

Includes derivatives wrt training points.  Akima spline is regenerated during each compute.

https://github.com/andrewning/akima
"""
from six.moves import range
from six import string_types

import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


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

    Attributes
    ----------
    x_grid : None or ndarray
        Cached training grid.
    x_cp_grid : None or ndarray
        Cached interpolation grid.
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
                             desc='Number of independent rows to interpolate.')
        self.options.declare('name', types=str, default='var',
                             desc="Name to use for the interpolated variable.")
        self.options.declare('input_x', types=bool, default=False,
                             desc="When True, the interpolated x grid is a component input.")
        self.options.declare('input_xcp', types=bool, default=False,
                             desc="When True, the x control point grid is a component input.")
        self.options.declare('units', types=string_types, default=None, allow_none=True,
                             desc="Units to use for the y and ycp variables.")
        self.options.declare('x_units', types=string_types, default=None, allow_none=True,
                             desc="Units to use for the x and xcp variables.")
        self.options.declare('delta_x', default=0.1,
                             desc="half-width of the smoothing interval added in the valley of "
                             "absolute-value function. This allows the derivatives with respect to"
                             " the data points (dydxpt, dydypt) to also be C1 continuous. Set "
                             "parameter to 0 to get the original Akima function (but only if you "
                             "don't need dydxpt, dydypt")
        self.options.declare('eval_at', default='end', values=['end', 'cell_center'],
                             desc="Where the return values are evaluated on the spline. When set "
                             "to 'end', compute num_points values spanning the full interval set "
                             "by the training points. When set to 'cell_center', compute values "
                             "at centerpoints of num_points cells. Only used if input_x is "
                             "True.")
        self.options.declare('eps', default=1e-30,
                             desc='Value that triggers division-by-zero safeguard.')

    def setup(self):
        """
        Set up the akima spline component.
        """
        opts = self.options
        num_control_points = opts['num_control_points']
        num_points = opts['num_points']
        vec_size = opts['vec_size']
        units = opts['units']
        eval_at = opts['eval_at']
        input_xcp = opts['input_xcp']
        input_x = opts['input_x']

        name = opts['name']
        x_name = name + ":x"
        xcp_name = name + ":x_cp"
        y_name = name + ":y"
        ycp_name = name + ":y_cp"

        if input_xcp:
            self.add_input(xcp_name, val=np.random.rand(num_control_points),
                           units=opts['x_units'])

        else:
            self.x_cp_grid = np.linspace(0., 1., num_control_points)
            self.add_output(xcp_name, val=self.x_cp_grid,
                            units=opts['x_units'])

        if input_x:
            self.add_input(x_name, val=np.random.rand(num_points),
                           units=opts['x_units'])

        else:
            if eval_at == 'cell_center':
                x_ends = np.linspace(0., 1., num_points + 1)
                self.x_grid = 0.5 * (x_ends[:-1] + x_ends[1:])

            elif eval_at == 'end':
                self.x_grid = np.linspace(0., 1., num_points)

            self.add_output(x_name, val=self.x_grid,
                            units=opts['x_units'])

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

        if input_xcp:
            rows = np.tile(row, vec_size) + np.repeat(num_points * np.arange(vec_size), ntot)
            cols = np.tile(col, vec_size)

            self.declare_partials(of=y_name, wrt=xcp_name, rows=rows, cols=cols)

        if input_x:
            row_col = np.arange(num_points)
            rows = np.tile(row_col, vec_size) + np.repeat(num_points * np.arange(vec_size),
                                                          num_points)
            cols = np.tile(row_col, vec_size)

            self.declare_partials(of=y_name, wrt=x_name, rows=rows, cols=cols)

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
        name = opts['name']
        x_name = name + ":x"
        xcp_name = name + ":x_cp"
        y_name = name + ":y"
        ycp_name = name + ":y_cp"

        if opts['input_xcp']:
            x_cp = inputs[xcp_name]
        else:
            x_cp = self.x_cp_grid

        if opts['input_x']:
            x = inputs[x_name]
        else:
            x = self.x_grid

        # Train on control points.
        self.akima_setup_dv(x_cp, inputs[ycp_name])

        # Evaluate at computational points.
        outputs[y_name] = self.akima_iterpolate(x, x_cp)

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
        name = opts['name']
        x_name = name + ":x"
        xcp_name = name + ":x_cp"
        y_name = name + ":y"
        ycp_name = name + ":y_cp"

        partials[y_name, ycp_name] = self.dy_dycp.flatten()

        if opts['input_xcp']:
            partials[y_name, xcp_name] = self.dy_dxcp.flatten()

        if opts['input_x']:
            partials[y_name, x_name] = self.dy_dx.flatten()

    def akima_setup_dv(self, xpt, ypt):
        """
        Train the akima spline and save the derivatives.

        Conversion of fortran function AKIMA_DV.

        Parameters
        ----------
        xpt : ndarray
            Values at which the akima spline was trained.
        ypt : ndarray
            Training values for the akima spline.
        """
        opts = self.options
        delta_x = opts['delta_x']
        eps = opts['eps']
        vec_size = opts['vec_size']
        ncp = opts['num_control_points']
        nbdirs = 2 * ncp

        # Poly points and derivs
        p1 = np.empty((vec_size, ncp - 1), dtype=ypt.dtype)
        p2 = np.empty((vec_size, ncp - 1), dtype=ypt.dtype)
        p3 = np.empty((vec_size, ncp - 1), dtype=ypt.dtype)
        p0d = np.empty((vec_size, nbdirs, ncp - 1), dtype=ypt.dtype)
        p1d = np.empty((vec_size, nbdirs, ncp - 1), dtype=ypt.dtype)
        p2d = np.empty((vec_size, nbdirs, ncp - 1), dtype=ypt.dtype)
        p3d = np.empty((vec_size, nbdirs, ncp - 1), dtype=ypt.dtype)

        md = np.zeros((nbdirs, ncp + 3), dtype=ypt.dtype)
        m = np.zeros((ncp + 3, ), dtype=ypt.dtype)
        td = np.zeros((nbdirs, ncp), dtype=ypt.dtype)
        t = np.zeros((ncp, ), dtype=ypt.dtype)

        xptd = np.vstack([np.eye(ncp, dtype=ypt.dtype),
                          np.zeros((ncp, ncp), dtype=ypt.dtype)])
        yptd = np.vstack([np.zeros((ncp, ncp), dtype=ypt.dtype),
                          np.eye(ncp, dtype=ypt.dtype)])

        dx = xpt[1:] - xpt[:-1]
        dx2 = dx**2
        dxd = xptd[:, 1:] - xptd[:, :-1]

        p0 = ypt[:, :-1]

        # TODO - It is possible to vectorize this further if some more effort is put in here.
        # Returns might be marginal though, and counterbalanced by increased memory. For
        # future investigation, here are the vectorized slopes:
        # md_f[:, :, 2:ncp + 1] = ((yptd[:, 1:] - yptd[:, :-1]) * (xpt[1:] - xpt[:-1]) -
        #                          np.einsum('ij,kj->ikj', ypt[:, 1:] - ypt[:, :-1],
        #                                    xptd[:, 1:] - xptd[:, :-1])) / \
        #     (xpt[1:] - xpt[:-1]) ** 2
        # m_f[:, 2:ncp + 1] = (ypt[:, 1:] - ypt[:, :-1]) / (xpt[1:] - xpt[:-1])

        for jj in range(vec_size):

            ypt_jj = ypt[jj, :]

            # Compute segment slopes
            md[:, 2:ncp + 1] = ((yptd[:, 1:] - yptd[:, :-1]) * dx -
                                (ypt_jj[1:] - ypt_jj[:-1]) * dxd) / dx2

            m[2:ncp + 1] = (ypt_jj[1:] - ypt_jj[:-1]) / dx

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
            for i in range(2, ncp + 2):
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
            t1 = t[:-1]
            t2 = t[1:]

            p1[jj, :] = t1
            p2[jj, :] = (3.0 * m[2:ncp + 1] - 2.0 * t1 - t2) / dx
            p3[jj, :] = (t1 + t2 - 2.0 * m[2:ncp + 1]) / dx2

            p0d[jj, ...] = yptd[:, :-1]
            p1d[jj, ...] = td[:, :-1]
            p2d[jj, ...] = ((3.0 * md[:, 2:ncp + 1] - 2.0 * td[:, :-1] - td[:, 1:]) * dx -
                            (3.0 * m[2:ncp + 1] - 2.0 * t1 - t2) * dxd) / dx2
            p3d[jj, ...] = ((td[:, :-1] + td[:, 1:] - 2.0 * md[:, 2:ncp + 1]) * dx2 -
                            (t1 + t2 - 2.0 * m[2:ncp + 1]) * 2 * dx * dxd) / (dx2)**2

        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        self.dp0_dxcp = p0d[:, :ncp, :].transpose((0, 2, 1))
        self.dp0_dycp = p0d[:, ncp:, :].transpose((0, 2, 1))
        self.dp1_dxcp = p1d[:, :ncp, :].transpose((0, 2, 1))
        self.dp1_dycp = p1d[:, ncp:, :].transpose((0, 2, 1))
        self.dp2_dxcp = p2d[:, :ncp, :].transpose((0, 2, 1))
        self.dp2_dycp = p2d[:, ncp:, :].transpose((0, 2, 1))
        self.dp3_dxcp = p3d[:, :ncp, :].transpose((0, 2, 1))
        self.dp3_dycp = p3d[:, ncp:, :].transpose((0, 2, 1))

    def akima_iterpolate(self, x, xcp):
        """
        Interpolate the spline at the given points, returning values and derivatives.

        Conversion of fortran function INTERP.

        Parameters
        ----------
        x : ndarray
            Values at which to interpolate.
        xcp : ndarray
            Values at which the akima spline was trained.

        Returns
        -------
        ndarray
            Interpolated values.
        """
        opts = self.options
        ncp = opts['num_control_points']
        n = opts['num_points']
        vec_size = opts['vec_size']

        p0 = self.p0
        p1 = self.p1
        p2 = self.p2
        p3 = self.p3

        # All vectorized points uses same grid, so find these once.
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

        dx = x - xcp[j_idx]
        dx2 = dx * dx
        dx3 = dx2 * dx

        # Evaluate polynomial (and derivative)
        y = p0[:, j_idx] + p1[:, j_idx] * dx + p2[:, j_idx] * dx2 + p3[:, j_idx] * dx3

        dydx = p1[:, j_idx] + 2.0 * p2[:, j_idx] * dx + 3.0 * p3[:, j_idx] * dx2

        dydxcp = self.dp0_dxcp[:, j_idx, :] + \
            np.einsum('kij,i->kij', self.dp1_dxcp[:, j_idx, :], dx) + \
            np.einsum('kij,i->kij', self.dp2_dxcp[:, j_idx, :], dx2) + \
            np.einsum('kij,i->kij', self.dp3_dxcp[:, j_idx, :], dx3)

        for jj in range(vec_size):
            for i in range(n):
                j = j_idx[i]
                dydxcp[jj, i, j] -= dydx[jj, i]

        dydycp = self.dp0_dycp[:, j_idx, :] + \
            np.einsum('kij,i->kij', self.dp1_dycp[:, j_idx, :], dx) + \
            np.einsum('kij,i->kij', self.dp2_dycp[:, j_idx, :], dx2) + \
            np.einsum('kij,i->kij', self.dp3_dycp[:, j_idx, :], dx3)

        self.dy_dx = dydx
        self.dy_dxcp = dydxcp
        self.dy_dycp = dydycp

        return y
