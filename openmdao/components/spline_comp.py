"""Define the SplineComp class."""
from six import iteritems, itervalues
from six import string_types
import numpy as np

from openmdao.components.interp_util.interp import InterpND
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

        self.options['extrapolate'] = True

        self.interp_to_cp = {}

    def setup(self):
        """
        Set up the spline component.
        """
        self.add_input(name=self.options['x_interp_name'], val=self.options['x_interp'])
        self.add_input(name=self.options['x_cp_name'], val=self.options['x_cp_val'])

        self.params.append(np.asarray(self.options['x_cp_val']))

    def _declare_options(self):
        """
        Declare options.
        """
        super(SplineComp, self)._declare_options()
        self.options.declare('x_cp_val', types=(list, np.ndarray), desc='List/array of x control '
                             'point values, must be monotonically increasing.')
        self.options.declare('x_interp', types=(list, np.ndarray), desc='List/array of x '
                             'interpolated point values.')
        self.options.declare('x_cp_name', types=string_types, default="'x_cp'",
                             desc='Name for the x control points input.')
        self.options.declare('x_interp_name', types=str, default="'x_interp'",
                             desc='Name of the x interpolated points input.')
        self.options.declare('x_units', types=string_types, default=None, allow_none=True,
                             desc='Units of the x variable.')
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

        vec_size = self.options['vec_size']
        n_interp = len(self.options['x_interp'])
        n_cp = len(self.options['x_cp_val'])

        self.add_output(y_interp_name, np.ones((vec_size, n_interp)), units=y_units)

        if y_cp_val is None:
            y_cp_val = np.ones((vec_size, n_cp))

        elif len(y_cp_val.shape) < 2:
            y_cp_val = y_cp_val.reshape((1, n_cp))

        self.add_input(name=y_cp_name, val=y_cp_val)
        self.training_outputs[y_interp_name] = y_cp_val

        self.interp_to_cp[y_interp_name] = y_cp_name

        rows = np.arange(vec_size * n_interp)
        cols = np.tile(np.arange(n_interp), vec_size)

        #self.declare_partials(y_interp_name, self.options['x_interp_name'], rows=rows, cols=cols)

        row = np.repeat(np.arange(n_interp), n_cp)
        col = np.tile(np.arange(n_cp), n_interp)
        rows = np.tile(row, vec_size) + np.repeat(n_interp * np.arange(vec_size), n_interp * n_cp)
        cols = np.tile(col, vec_size) + np.repeat(n_cp * np.arange(vec_size), n_interp * n_cp)

        self.declare_partials(y_interp_name, y_cp_name, rows=rows, cols=cols)

    def _setup_var_data(self, recurse=True):
        """
        Instantiate surrogates for the output variables that use the default surrogate.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        interp_method = self.options['method']

        opts = {}
        if 'interp_options' in self.options:
            opts = self.options['interp_options']
        for name, train_data in iteritems(self.training_outputs):
            # Separate data for each vec_size, but we only need to do sizing, so just pass
            # in the first.  Most interps aren't vectorized.
            train_data = train_data[0, :]
            self.interps[name] = InterpND(self.params, train_data,
                                          interp_method=interp_method,
                                          x_interp=self.options['x_interp'],
                                          bounds_error=not self.options['extrapolate'], **opts)

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
        #pt = np.array([inputs[pname].flatten() for pname in self.pnames]).T
        for out_name, interp in iteritems(self.interps):
            values = inputs[self.interp_to_cp[out_name]]
            interp.training_data_gradients = True

            #try:
            outputs[out_name] = interp.evaluate_spline(values)

            #except ValueError as err:
                #msg = "{}: Error interpolating output '{}':\n{}"
                #raise ValueError(msg.format(self.msginfo, out_name, str(err)))

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
        vec_size = self.options['vec_size']
        n_interp = len(self.options['x_interp'])
        n_cp = len(self.options['x_cp_val'])

        pt = np.array([inputs[pname].flatten() for pname in self.pnames]).T

        for out_name, interp in iteritems(self.interps):
            cp_name = self.interp_to_cp[out_name]
            #dval = interp.gradient(pt).T
            #for i, p in enumerate(self.pnames):
                #partials[out_name, p] = dval

            dy_ddata = np.zeros((vec_size, n_interp, n_cp))

            d_dvalues = interp._d_dvalues
            if d_dvalues is not None:
                if d_dvalues.shape[0] == vec_size:
                    # Akima precomputes derivs at all points in vec_size.
                    dy_ddata[:] = d_dvalues
                else:
                    # Bsplines computed derivative is the same at all points in vec_size.
                    dy_ddata[:] = np.broadcast_to(d_dvalues.toarray(), (vec_size, n_interp, n_cp))
            else:
                # This way works for most of the interpolation methods.
                for j in range(self.options['vec_size']):
                    val = interp.training_gradients(pt[j, :])
                    dy_ddata[j] = val.reshape(self.grad_shape[1:])

            partials[out_name, cp_name] = dy_ddata.flatten()
