from six import iteritems, itervalues
import numpy as np

from openmdao.components.interp_base import InterpBase


class SplineComp(InterpBase):
    """
    Interpolation component that can use any of OpenMDAO's interpolation methods.

    Attributes
    ----------
    interp_to_cp : dict
        Dictionary of relationship between the interpolated data and its control points.
    """

    def __init__(self, **kwargs):
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

        self.add_input(name=self.options['x_interp_name'], val=self.options['x_interp'])
        self.add_input(name=self.options['x_cp_name'], val=self.options['x_cp_val'])

        self.pnames.append(self.options['x_interp_name'])
        self.params.append(np.asarray(self.options['x_cp_val']))

        self.options['extrapolate'] = True
        self.options['training_data_gradients'] = True

        self.interp_to_cp = {}

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
            msg = "{}: y_cp_name cannot be an empty string."
            raise ValueError(msg.format(self.msginfo))
        elif not y_interp_name:
            msg = "{}: y_interp_name cannot be an empty string."
            raise ValueError(msg.format(self.msginfo))

        self.add_output(y_interp_name, 1.0 * np.ones(self.options['vec_size']),
                                           units=y_units)
        if y_cp_val is None:
            y_cp_val = self.options['x_cp_val']

            self.add_input(name=y_cp_name, val=np.linspace(0., 1., len(y_cp_val)))
            self.training_outputs[y_interp_name] = y_cp_val
        else:
            self.add_input(name=y_cp_name, val=y_cp_val)
            self.training_outputs[y_interp_name] = y_cp_val

        self.interp_to_cp[y_interp_name] = y_cp_name

    def _setup_partials(self, recurse=True):
        """
        Process all partials and approximations that the user declared.

        Metamodel needs to declare its partials after inputs and outputs are known.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(SplineComp, self)._setup_partials()
        arange = np.arange(self.options['vec_size'])
        pnames = tuple(self.pnames)
        dct = {
            'rows': arange,
            'cols': arange,
            'dependent': True,
        }

        for name in self._outputs:
            self._declare_partials(of=name, wrt=pnames, dct=dct)
            self._declare_partials(of=name, wrt=self.interp_to_cp[name], dct={'dependent': True})

        # The scipy methods do not support complex step.
        if self.options['method'].startswith('scipy'):
            self.set_check_partial_options('*', method='fd')


    def compute(self, inputs, outputs):
        """
        Perform the interpolation at run time.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        outputs : Vector
            unscaled, dimensional output variables read via outputs[key]
        """
        pt = np.array([inputs[pname].flatten() for pname in self.pnames]).T
        for out_name, interp in iteritems(self.interps):
            interp.values = inputs[self.interp_to_cp[out_name]]
            interp.training_data_gradients = True

            try:
                val = interp.interpolate(pt)

            except ValueError as err:
                raise ValueError("{}: Error interpolating output '{}':\n{}".format(self.msginfo,
                                                                                   out_name,
                                                                                   str(err)))
            outputs[out_name] = val


    def compute_partials(self, inputs, partials):
        """
        Collect computed partial derivatives and return them.

        Checks if the needed derivatives are cached already based on the
        inputs vector. Refreshes the cache by re-computing the current point
        if necessary.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        partials : Jacobian
            sub-jac components written to partials[output_name, input_name]
        """
        pt = np.array([inputs[pname].flatten() for pname in self.pnames]).T
        dy_ddata = np.zeros(self.grad_shape)
        interp = next(itervalues(self.interps))
        for j in range(self.options['vec_size']):
            val = interp.training_gradients(pt[j, :])
            dy_ddata[j] = val.reshape(self.grad_shape[1:])

        for out_name in self.interps:
            dval = self.interps[out_name].gradient(pt).T
            for i, p in enumerate(self.pnames):
                partials[out_name, p] = dval[i, :]

            partials[out_name, self.interp_to_cp[out_name]] = dy_ddata