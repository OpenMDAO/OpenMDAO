"""Define the SplineComp class."""
from six import iteritems, itervalues
import numpy as np

from openmdao.components.interp_util.python_interp import PythonGridInterp
from openmdao.components.interp_util.scipy_interp import ScipyGridInterp
from openmdao.components.interp_base import InterpBase
from openmdao.core.problem import Problem


class SplineComp(InterpBase):
    """
    Interpolation component that can use any of OpenMDAO's interpolation methods.

    Attributes
    ----------
    interp_to_cp : dict
        Dictionary of relationship between the interpolated data and its control points.
    n_interp : int
        Number of points to interpolate at
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            Interpolator options to pass onward.
        """
        super(SplineComp, self).__init__(**kwargs)

        self.add_input(name=self.options['x_interp_name'], val=self.options['x_interp'])
        self.add_input(name=self.options['x_cp_name'], val=self.options['x_cp_val'])

        self.pnames.append(self.options['x_interp_name'])
        self.params.append(np.asarray(self.options['x_cp_val']))

        self.options['extrapolate'] = True
        self.options['training_data_gradients'] = True
        self.n_interp = len(self.options['x_interp'])

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

        self.add_output(y_interp_name, np.ones((self.options['vec_size'], self.n_interp)),
                        units=y_units)
        if y_cp_val is None:
            y_cp_val = self.options['x_cp_val']

            self.add_input(name=y_cp_name, val=np.linspace(0., 1., len(y_cp_val)))
            self.training_outputs[y_interp_name] = y_cp_val
        else:
            self.add_input(name=y_cp_name, val=y_cp_val)
            self.training_outputs[y_interp_name] = y_cp_val

        self.interp_to_cp[y_interp_name] = y_cp_name

    def _setup_var_data(self, recurse=True):
        """
        Instantiate surrogates for the output variables that use the default surrogate.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        interp_method = self.options['method']
        if interp_method.startswith('scipy'):
            interp = ScipyGridInterp
            interp_method = interp_method[6:]
        else:
            interp = PythonGridInterp

        opts = {}
        if 'interp_options' in self.options:
            opts = self.options['interp_options']
        for name, cp_points in iteritems(self.training_outputs):
            if self.options['vec_size'] > 1:
                cp_points = cp_points[0, :]
            self.interps[name] = interp(self.params, cp_points,
                                        interp_method=interp_method,
                                        bounds_error=not self.options['extrapolate'],
                                        **opts)

        if self.options['training_data_gradients']:
            self.grad_shape = tuple([self.options['vec_size']] + [i.size for i in self.params])

        super(SplineComp, self)._setup_var_data(recurse=recurse)

    def _setup_partials(self, recurse=True):
        """
        Process all partials and approximations that the user declared.

        SplineComp needs to declare its partials after inputs and outputs are known.

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
            for i in range(0, self.options['vec_size']):
                if self.options['vec_size'] > 1:
                    interp.values = inputs[self.interp_to_cp[out_name]][i]
                else:
                    interp.values = inputs[self.interp_to_cp[out_name]]
                interp.training_data_gradients = True

                try:
                    val = interp.interpolate(pt)
                    outputs[out_name][i, :] = val

                except ValueError as err:
                    raise ValueError("{}: Error interpolating output '{}':\n{}".format(self.msginfo,
                                                                                       out_name,
                                                                                       str(err)))

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


def interp(method, x_data, y_data, x):
    """
    Compute y and its derivatives for a given x by interpolating on x_data and y_data.

    Parameters
    ----------
    method : str
        Method to use, choose from all available openmmdao methods.
    x_data : ndarray or list
        Input data for x, should be monotonically increasing. For higher dimensional grids,
        x_data should be a list containing the x data for each dimension.
    y_data : ndarray
        Input values for y. For higher dimensional grids, the index order should be the same as
        in x_data.
    x : float or iterable or ndarray
        Location(s) at which to interpolate.

    Returns
    -------
    float or ndarray
        Interpolated values y
    ndarray
        Derivative of y with respect to x
    ndarray
        Derivative of y with respect to x_data
    ndarray
        Derivative of y with respect to y_data
    """
    prob = Problem()

    comp = SplineComp(method=method, x_cp_val=x_data, x_cp_name='xcp', x_interp=x,
                      x_interp_name='x')
    comp.add_spline(y_cp_name='ycp', y_interp_name='y', y_cp_val=y_data)

    prob.model.add_subsystem('spline1', comp)
    prob.setup(force_alloc_complex=True)
    prob.run_model()

    return prob['spline1.y'], prob['spline1.x'], prob['spline1.xcp'], prob['spline1.ycp']
