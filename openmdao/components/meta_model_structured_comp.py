"""Define the MetaModelStructured class."""
from __future__ import division, print_function, absolute_import

from six import raise_from, iteritems, itervalues
from six.moves import range

import numpy as np

from openmdao.components.structured_metamodel_util.outofbounds_error import OutOfBoundsError
from openmdao.components.structured_metamodel_util.python_interp import PythonGridInterp
from openmdao.components.structured_metamodel_util.scipy_interp import ScipyGridInterp
from openmdao.core.analysis_error import AnalysisError
from openmdao.utils.general_utils import warn_deprecation
from openmdao.components.interp_base import InterpBase

ALL_METHODS = ('cubic', 'slinear', 'lagrange2', 'lagrange3', 'akima',
               'scipy_cubic', 'scipy_slinear', 'scipy_quintic')


class MetaModelStructuredComp(InterpBase):
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
