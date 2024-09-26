"""Define the MetaModelSemiStructuredComp class."""
import inspect

import numpy as np

from openmdao.components.interp_util.outofbounds_error import OutOfBoundsError
from openmdao.components.interp_util.interp_semi import InterpNDSemi, TABLE_METHODS
from openmdao.core.analysis_error import AnalysisError
from openmdao.core.explicitcomponent import ExplicitComponent


class MetaModelSemiStructuredComp(ExplicitComponent):
    """
    Interpolation Component generated from semi-structured data on a regular grid.

    Produces smooth fits through provided training data using polynomial splines of various
    orders. Analytic derivatives are automatically computed.

    For multi-dimensional data, fits are computed on a separable per-axis basis. If a particular
    dimension does not have enough training data points to support a selected spline method (e.g. 3
    sample points, but an fifth order quintic spline is specified) the order of the
    fitted spline with be automatically reduced for that dimension alone.

    Extrapolation is supported, but disabled by default. It can be enabled via initialization
    option.

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Component options.

    Attributes
    ----------
    interps : dict
        Dictionary of interpolations for each output.
    pnames : list
        Cached list of input names.
    training_inputs : dict of ndarray
        Dictionary of grid point locations corresponding to the training values in
        self.training_outputs.
    training_outputs : dict of ndarray
        Dictionary of training data each output.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.
        """
        super().__init__(**kwargs)

        self.pnames = []
        self.training_inputs = {}
        self.training_outputs = {}
        self.interps = {}

        self._no_check_partials = True

    def initialize(self):
        """
        Initialize the component.
        """
        self.options.declare('extrapolate', types=bool, default=True,
                             desc='Sets whether extrapolation should be performed '
                                  'when an input is out of bounds.')
        self.options.declare('training_data_gradients', types=bool, default=False,
                             desc='When True, compute gradients with respect to training data '
                             'values.')
        self.options.declare('vec_size', types=int, default=1,
                             desc='Number of points to evaluate at once.')
        self.options.declare('method', values=TABLE_METHODS, default='slinear',
                             desc='Spline interpolation method to use for all outputs.')

    def add_input(self, name, training_data, val=1.0, **kwargs):
        """
        Add an input to this component and a corresponding training input.

        Parameters
        ----------
        name : str
            Name of the input.
        training_data : ndarray
            Training data grid sample points for this input variable. Must be of length m, where m
            is the total number of points in the table..
        val : float or ndarray
            Initial value for the input.
        **kwargs : dict
            Additional agruments for add_input.
        """
        n = self.options['vec_size']

        # Currently no support for vector inputs, apart from vec_size
        if not np.isscalar(val):

            if len(val) not in [1, n] or len(val.shape) > 1:
                msg = "{}: Input {} must either be scalar, or of length equal to vec_size."
                raise ValueError(msg.format(self.msginfo, name))

        super().add_input(name, val * np.ones(n), **kwargs)

        self.training_inputs[name] = training_data

        self.pnames.append(name)

    def add_output(self, name, training_data=None, **kwargs):
        """
        Add an output to this component and a corresponding training output.

        Parameters
        ----------
        name : str
            Name of the output.
        training_data : ndarray
            Training data sample points for this output variable. Must be of length m, where m is
            the total number of points in the table.
        **kwargs : dict
            Additional agruments for add_output.
        """
        n = self.options['vec_size']
        super().add_output(name, np.ones(n), **kwargs)

        if self.options['training_data_gradients']:
            if training_data is None:
                for item in self.training_inputs.values():
                    n_train = len(item)
                    # Training datasets are all the same length, so grab the first.
                    break
                training_data = np.ones(n_train)
            super().add_input("%s_train" % name, val=training_data, **kwargs)

        elif training_data is None:
            msg = f"Training data is required for output '{name}'."
            raise ValueError(msg)

        self.training_outputs[name] = training_data

    def _setup_var_data(self):
        """
        Instantiate surrogates for the output variables that use the default surrogate.
        """
        interp_method = self.options['method']

        # Make sure all training data is sized correctly.
        size = len(self.training_inputs[self.pnames[0]])
        for data_dict in [self.training_inputs, self.training_outputs]:
            for name, data in data_dict.items():
                size2 = len(data)
                if size2 != size:
                    msg = f"Size mismatch: training data for '{name}' is length {size2}, but" + \
                        f" data for '{self.pnames[0]}' is length {size}."
                    raise ValueError(msg)

        grid = np.array([col for col in self.training_inputs.values()]).T

        for name, train_data in self.training_outputs.items():
            self.interps[name] = InterpNDSemi(grid, train_data, method=interp_method,
                                              extrapolate=self.options['extrapolate'])

        super()._setup_var_data()

    def _setup_partials(self):
        """
        Process all partials and approximations that the user declared.

        Metamodel needs to declare its partials after inputs and outputs are known.
        """
        super()._setup_partials()
        arange = np.arange(self.options['vec_size'])
        wrtnames = tuple(self.pnames)
        pattern_meta = {
            'rows': arange,
            'cols': arange,
            'dependent': True,
        }

        for of in self._var_rel_names['output']:
            self._resolve_partials_patterns(of=of, wrt=wrtnames, pattern_meta=pattern_meta)
            if self.options['training_data_gradients']:
                self._resolve_partials_patterns(of=of, wrt="%s_train" % of,
                                                pattern_meta={'dependent': True})

        # The scipy methods do not support complex step.
        if self.options['method'].startswith('scipy'):
            self.set_check_partial_options('*', method='fd')

    def compute(self, inputs, outputs):
        """
        Perform the interpolation at run time.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        """
        pt = np.array([inputs[pname].ravel() for pname in self.pnames]).T
        for out_name, interp in self.interps.items():
            if self.options['training_data_gradients']:
                # Training point values may have changed every time we compute.
                interp.values = inputs["%s_train" % out_name]
                interp._compute_d_dvalues = True

            try:
                val = interp._interpolate(pt)

            except OutOfBoundsError as err:
                varname_causing_error = '.'.join((self.pathname, self.pnames[err.idx]))
                errmsg = (f"{self.msginfo}: Error interpolating output '{out_name}' "
                          f"because input '{varname_causing_error}' required extrapolation while "
                          f"interpolating dimension {err.idx + 1}, where its value '{err.value}'"
                          f" exceeded the range ('{err.lower}', '{err.upper}')")
                raise AnalysisError(errmsg, inspect.getframeinfo(inspect.currentframe()),
                                    self.msginfo)

            except ValueError as err:
                raise ValueError(f"{self.msginfo}: Error interpolating output '{out_name}':\n"
                                 f"{str(err)}")
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
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Sub-jac components written to partials[output_name, input_name].
        """
        pt = np.array([inputs[pname].ravel() for pname in self.pnames]).T

        for out_name, interp in self.interps.items():
            dval = interp.gradient(pt).T
            for i, p in enumerate(self.pnames):
                partials[out_name, p] = dval[i, :]

            if self.options['training_data_gradients']:
                train_name = f"{out_name}_train"
                d_dvalues = interp._d_dvalues
                partials[out_name, train_name] = d_dvalues
