import numpy as np

from openmdao.components.interp_base import InterpBase


class SplineComp(InterpBase):

    def __init__(self, **kwargs):
        # self, method, x_cp_val, x_interp, x_cp_name='x_cp', x_interp_name='x_interp',
        #          x_units=None, vec_size=1.0, interp_options={}):
        """
        Initialize all attributes.

        Parameters
        ----------
        method : str
            Interpolation method. Valid values are ['akima', 'bspline', (more to come)]
        x_cp_val : list or ndarray
            List/array of x control point values, must be monotonically increasing.
        x_interp : list or ndarray
            List/array of x interpolated point values.
        x_cp_name : str
            Name for the x control points input.
        x_interp_name : str
            Name of the x interpolated points input.
        x_units : str or None
            Units of the x variable.
        vec_size : int
            The number of independent splines to interpolate.
        interp_options : dict
            Dict contains the name and value of options specific to the chosen interpolation method.
        """

        super(SplineComp, self).__init__(**kwargs)

        super(SplineComp, self).add_input(name=self.options['x_interp_name'],
                                          val=self.options['x_interp'])

        self.pnames.append(self.options['x_interp_name'])
        self.params.append(np.asarray(self.options['x_cp_val']))

    def _declare_options(self):
        """
        Declare options.
        """
        super(SplineComp, self)._declare_options()
        self.options.declare('x_cp_val', default=None, desc='List/array of x control'
                             'point values, must be monotonically increasing.')
        self.options.declare('x_interp', default=None, desc='List/array of x '
                             'interpolated point values.')
        self.options.declare('x_cp_name', default='x_cp',
                             desc='Name for the x control points input.')
        self.options.declare('x_interp_name', default='x_interp',
                             desc='Name of the x interpolated points input.')
        self.options.declare('x_units', default=None, desc='Units of the x variable.')
        self.options.declare('interp_options', types=dict, default={},
                             desc='Dict contains the name and value of options specific to the '
                             'chosen interpolation method.')


        # self.options.declare('delta_x', default=0.1,
        #                      desc="half-width of the smoothing interval added in the valley of "
        #                      "absolute-value function. This allows the derivatives with respect "
        #                      " to the data points (dydxpt, dydypt) to also be C1 continuous. Set "
        #                      "parameter to 0 to get the original Akima function (but only if you "
        #                      "don't need dydxpt, dydypt")
        # self.options.declare('eps', default=1e-30,
        #                      desc='Value that triggers division-by-zero safeguard.')

    def add_spline(self, y_cp_name, y_interp_name, y_cp_val=None, y_units=None):
        """
        Add a single spline output to this component.

        Parameters
        ----------
        y_cp_name : str
            Name for the y control points input.
        y_interp_name : str
            Name of the y interpolated points output.
        y_cp_val : list or ndarray
            List/array of default y control point values.
        y_units : str or None
            Units of the y variable.
        """
        if not y_cp_name:
            raise ValueError("y_cp_name cannot have an empty string")
        elif not y_interp_name:
            raise ValueError("y_interp cannot have an empty string")


        super(SplineComp, self).add_output(y_interp_name, 1.0 * np.ones(self.options['vec_size']),
                                           units=y_units)

        self.training_outputs[y_interp_name] = y_cp_val