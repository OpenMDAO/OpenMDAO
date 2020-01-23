"""Define the InterpBase class."""
from openmdao.core.explicitcomponent import ExplicitComponent

ALL_METHODS = ('cubic', 'slinear', 'lagrange2', 'lagrange3', 'akima',
               'scipy_cubic', 'scipy_slinear', 'scipy_quintic')


class InterpBase(ExplicitComponent):
    """
    Base class for interpolation components.

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
        **kwargs : dict
            Interpolator options to pass onward.
        """
        super(InterpBase, self).__init__(**kwargs)
        self.pnames = []
        self.params = []
        self.training_outputs = {}
        self.interps = {}
        self.grad_shape = ()

    def _declare_options(self):
        """
        Initialize the component.
        """
        super(InterpBase, self)._declare_options()
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
