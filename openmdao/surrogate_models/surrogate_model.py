"""
Class definition for SurrogateModel, the base class for all surrogate models.
"""
from openmdao.utils.options_dictionary import OptionsDictionary


class SurrogateModel(object):
    """
    Base class for surrogate models.

    Attributes
    ----------
    options : <OptionsDictionary>
        Dictionary with general pyoptsparse options.
    trained : bool
        True when surrogate has been trained.
    _parent_name : str or None
        Absolute pathname of metamodel component that owns this surrogate.
    """

    def __init__(self, **kwargs):
        """
        Initialize all attributes.

        Parameters
        ----------
        **kwargs : dict
            options dictionary.
        """
        self.trained = False

        self.options = OptionsDictionary(parent_name=type(self).__name__)
        self._declare_options()
        self.options.update(kwargs)
        self._parent_name = None

    def train(self, x, y):
        """
        Train the surrogate model with the given set of inputs and outputs.

        Parameters
        ----------
        x : array-like
            Training input locations
        y : array-like
            Model responses at given inputs.
        """
        self.trained = True

    def predict(self, x):
        """
        Calculate a predicted value of the response based on the current trained model.

        Parameters
        ----------
        x : array-like
            Point(s) at which the surrogate is evaluated.
        """
        if not self.trained:
            msg = "{0} has not been trained, so no prediction can be made."\
                .format(type(self).__name__)
            raise RuntimeError(msg)

    def vectorized_predict(self, x):
        """
        Calculate predicted values of the response based on the current trained model.

        Parameters
        ----------
        x : array-like
            Vectorized point(s) at which the surrogate is evaluated.
        """
        pass

    def linearize(self, x):
        """
        Calculate the jacobian of the interpolant at the requested point.

        Parameters
        ----------
        x : array-like
            Point at which the surrogate Jacobian is evaluated.
        """
        pass

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        pass

    def _setup_var_data(self, parent_name=None):
        """
        Set up surrogate to create valid informative error messages.

        Parameters
        ----------
        parent_name : str
            Absolute pathname of metamodel component that owns this surrogate.
        """
        self._parent_name = parent_name

    def _raise(self, msg, exc_type=RuntimeError):
        """
        Raise the given exception type, with parent's name prepended to the message.

        Parameters
        ----------
        msg : str
            The error message.
        exc_type : class
            The type of the exception to be raised.
        """
        if self._parent_name is None:
            full_msg = msg
        else:
            full_msg = '{}: {}'.format(self._parent_name, msg)
        raise exc_type(full_msg)


class MultiFiSurrogateModel(SurrogateModel):
    """
    Base class for surrogate models using multi-fidelity training data.
    """

    def train(self, x, y):
        """
        Calculate a predicted value of the response based on the current trained model.

        Parameters
        ----------
        x : array-like
            Point(s) at which the surrogate is evaluated.
        y : array-like
            Model responses at given inputs.
        """
        super(MultiFiSurrogateModel, self).train(x, y)
        self.train_multifi([x], [y])

    def train_multifi(self, x, y):
        """
        Train the surrogate model, based on the given multi-fidelity training data.

        Parameters
        ----------
        x : list of (m samples, n inputs) ndarrays
            Values representing the multi-fidelity training case inputs.
        y : list of ndarray
            output training values which corresponds to the multi-fidelity
            training case input given by x.
        """
        pass
