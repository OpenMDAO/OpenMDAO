"""Define the MetaModelStructured class."""
from __future__ import division, print_function, absolute_import

from six import raise_from, iteritems, itervalues
from six.moves import range

import numpy as np

from openmdao.components.interp_util.outofbounds_error import OutOfBoundsError
from openmdao.components.interp_util.interp import InterpND
from openmdao.core.analysis_error import AnalysisError
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.utils.general_utils import warn_deprecation


ALL_METHODS = ('cubic', 'slinear', 'lagrange2', 'lagrange3', 'akima',
               'scipy_cubic', 'scipy_slinear', 'scipy_quintic')


class MetaModelStructuredComp(ExplicitComponent):
    """
    Interpolation Component generated from data on a regular grid.

    Produces smooth fits through provided training data using polynomial splines of various
    orders. Analytic derivatives are automatically computed.

    For multi-dimensional data, fits are computed on a separable per-axis basis. If a particular
    dimension does not have enough training data points to support a selected spline method (e.g. 3
    sample points, but an fifth order quintic spline is specified) the order of the
    fitted spline with be automatically reduced for that dimension alone.

    Extrapolation is supported, but disabled by default. It can be enabled via initialization
    option.

    Attributes
    ----------
    grad_shape : tuple
        Cached shape of the gradient of the outputs wrt the training inputs.
    interps : dict
        Dictionary of interpolations for each output.
    params : list
        List containing training data for each input.
    pnames : list
        Cached list of input names.
    training_outputs : dict
        Dictionary of training data each output.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        super(MetaModelStructuredComp, self).__init__(**kwargs)

        self.pnames = []
        self.params = []
        self.training_outputs = {}
        self.interps = {}
        self.grad_shape = ()

    def initialize(self):
        """
        Initialize the component.
        """
        self.options.declare('extrapolate', types=bool, default=False,
                             desc='Sets whether extrapolation should be performed '
                                  'when an input is out of bounds.')
        self.options.declare('training_data_gradients', types=bool, default=False,
                             desc='Sets whether gradients with respect to output '
                                  'training data should be computed.')
        self.options.declare('vec_size', types=int, default=1,
                             desc='Number of points to evaluate at once.')
        self.options.declare('method', values=ALL_METHODS, default='scipy_cubic',
                             desc='Spline interpolation method to use for all outputs.')

    def add_input(self, name, val=1.0, training_data=None, **kwargs):
        """
        Add an input to this component and a corresponding training input.

        Parameters
        ----------
        name : string
            Name of the input.
        val : float or ndarray
            Initial value for the input.
        training_data : ndarray
            training data sample points for this input variable.
        **kwargs : dict
            Additional agruments for add_input.
        """
        n = self.options['vec_size']

        # Currently no support for vector inputs, apart from vec_size
        if not np.isscalar(val):

            if len(val) not in [1, n] or len(val.shape) > 1:
                msg = "{}: Input {} must either be scalar, or of length equal to vec_size."
                raise ValueError(msg.format(self.msginfo, name))

        super(MetaModelStructuredComp, self).add_input(name, val * np.ones(n), **kwargs)

        self.pnames.append(name)
        self.params.append(np.asarray(training_data))

    def add_output(self, name, val=1.0, training_data=None, **kwargs):
        """
        Add an output to this component and a corresponding training output.

        Parameters
        ----------
        name : string
            Name of the output.
        val : float or ndarray
            Initial value for the output.
        training_data : ndarray
            training data sample points for this output variable.
        **kwargs : dict
            Additional agruments for add_output.
        """
        n = self.options['vec_size']

        # Currently no support for vector outputs, apart from vec_size
        if not np.isscalar(val):

            if len(val) not in [1, n] or len(val.shape) > 1:
                msg = "{}: Output {} must either be scalar, or of length equal to vec_size."
                raise ValueError(msg.format(self.msginfo, name))

        super(MetaModelStructuredComp, self).add_output(name, val * np.ones(n), **kwargs)

        self.training_outputs[name] = training_data

        if self.options['training_data_gradients']:
            super(MetaModelStructuredComp, self).add_input("%s_train" % name,
                                                           val=training_data, **kwargs)

    def _setup_var_data(self, recurse=True):
        """
        Instantiate surrogates for the output variables that use the default surrogate.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        interp_method = self.options['method']

        for name, train_data in iteritems(self.training_outputs):
            self.interps[name] = InterpND(self.params, train_data,
                                          interp_method=interp_method,
                                          bounds_error=not self.options['extrapolate'])

        if self.options['training_data_gradients']:
            self.grad_shape = tuple([self.options['vec_size']] + [i.size for i in self.params])

        super(MetaModelStructuredComp, self)._setup_var_data(recurse=recurse)

    def _setup_partials(self, recurse=True):
        """
        Process all partials and approximations that the user declared.

        Metamodel needs to declare its partials after inputs and outputs are known.

        Parameters
        ----------
        recurse : bool
            Whether to call this method in subsystems.
        """
        super(MetaModelStructuredComp, self)._setup_partials()
        arange = np.arange(self.options['vec_size'])
        pnames = tuple(self.pnames)
        dct = {
            'rows': arange,
            'cols': arange,
            'dependent': True,
        }

        for name in self._outputs:
            self._declare_partials(of=name, wrt=pnames, dct=dct)
            if self.options['training_data_gradients']:
                self._declare_partials(of=name, wrt="%s_train" % name, dct={'dependent': True})

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
            if self.options['training_data_gradients']:
                # Training point values may have changed every time we compute.
                interp.values = inputs["%s_train" % out_name]
                interp.training_data_gradients = True

            try:
                val = interp.interpolate(pt)

            except OutOfBoundsError as err:
                varname_causing_error = '.'.join((self.pathname, self.pnames[err.idx]))
                errmsg = "{}: Error interpolating output '{}' because input '{}' " \
                    "was out of bounds ('{}', '{}') with " \
                    "value '{}'".format(self.msginfo, out_name, varname_causing_error,
                                        err.lower, err.upper, err.value)
                raise_from(AnalysisError(errmsg), None)

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
        dy_data = None

        for out_name, interp in iteritems(self.interps):
            dval = interp.gradient(pt).T
            for i, p in enumerate(self.pnames):
                partials[out_name, p] = dval[i, :]

            if self.options['training_data_gradients']:

                if dy_data is None or interp._d_dvalues is not None:
                    dy_ddata = np.zeros(self.grad_shape)

                    if interp._d_dvalues is not None:
                        # Akima must be handled individually.
                        dy_ddata[:] = interp._d_dvalues

                    else:
                        # This way works for most of the interpolation methods.
                        for j in range(self.options['vec_size']):
                            val = interp.training_gradients(pt[j, :])
                            dy_ddata[j] = val.reshape(self.grad_shape[1:])

                partials[out_name, "%s_train" % out_name] = dy_ddata


class MetaModelStructured(MetaModelStructuredComp):
    """
    Deprecated.
    """

    def __init__(self, *args, **kwargs):
        """
        Capture Initialize to throw warning.

        Parameters
        ----------
        *args : list
            Deprecated arguments.
        **kwargs : dict
            Deprecated arguments.
        """
        warn_deprecation("'MetaModelStructured' has been deprecated. Use "
                         "'MetaModelStructuredComp' instead.")
        super(MetaModelStructured, self).__init__(*args, **kwargs)
