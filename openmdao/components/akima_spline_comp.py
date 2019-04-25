"""
Simple interpolant based on Andrew Ning's implementation of Akima splines.

Includes derivatives wrt training points.  Akima spline is regenerated during each compute.

https://github.com/andrewning/akima
"""

import numpy as np

from openmdao.api import ExplicitComponent


class AkimaSplineComp(ExplicitComponent):
    """
    Simple B-spline component for interpolation.
    """

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
        self.options.declare('in_name', types=str, default='h_cp',
                             desc="Name to use for the input variable (control points).")
        self.options.declare('out_name', types=str, default='h',
                             desc="Name to use for the output variable (interpolated points).")
        self.options.declare('units', types=string_types, default=None, allow_none=True,
                             desc="Units to use for the input and output variables.")
        self.options.declare('delta_x', default=0.1,
                             desc="half-width of the smoothing interval added in the valley of "
                             "absolute-value function this allows the derivatives with respect to "
                             "the data points (dydxpt, dydypt) to also be C1 continuous. Set to "
                             "parameter to 0 to get the original Akima function (but only ifyou "
                             "don't need dydxpt, dydypt")
        self.options.declare('evaluate_at', default=['end'], values=['end', 'cell_center'],
                             "Where the return values are evaluate on the spline. When set to "
                             "end, compute num_points values spanning the full interval set by "
                             "the training points. When set to 'cell_center', compute values at "
                             "centerpoints of num_points cells.")

    def setup(self):
        """
        Set up the B-spline component.
        """
        opts = self.options
        num_control_points = opts['num_control_points']
        num_points = opts['num_points']
        vec_size = opts['vec_size']
        in_name = opts['in_name']
        out_name = opts['out_name']
        units = opts['units']
        evaluate_at = opts['evaluate_at']

        if evaluate_at == 'cell_center':
            x_ends = np.linspace(0., 1., num_points + 1)
            self.x_grid = 0.5 * (x_ends[:-1] + x_ends[1:])

        elif evaluate_at == 'end':
            self.x_grid = np.linspace(0., 1., num_points)

        self.add_input(in_name, val=np.random.rand(vec_size, num_control_points), units=units)
        self.add_output(out_name, val=np.random.rand(vec_size, num_points), units=units)

