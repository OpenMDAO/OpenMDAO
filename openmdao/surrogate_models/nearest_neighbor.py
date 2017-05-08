# Based off of the N-Dimensional Interpolation library by Stephen Marone.
# https://github.com/SMarone/NDInterp

from collections import OrderedDict
from openmdao.surrogate_models.surrogate_model import SurrogateModel
from openmdao.surrogate_models.nn_interpolators.linear_interpolator import \
    LinearInterpolator
from openmdao.surrogate_models.nn_interpolators.weighted_interpolator import \
    WeightedInterpolator
from openmdao.surrogate_models.nn_interpolators.rbf_interpolator import \
    RBFInterpolator

_interpolators = OrderedDict([('linear', LinearInterpolator),
                              ('weighted', WeightedInterpolator),
                              ('rbf', RBFInterpolator)])


class NearestNeighbor(SurrogateModel):
    """
    Surrogate model that approximates values using a nearest neighbor
    approximation. `interpolant_type` argument must be one of 'linear',
    'weighted', or 'rbf'.

    Args
    ----

    interpolant_type : str
        One of 'linear', 'weighted', or 'rbf'. Determines the type of
        interpolant used.

    kwargs :
        Additional keyword arguments to be passed to the constructor for the
        interpolant.

    """
    def __init__(self, interpolant_type='rbf', **kwargs):
        super(NearestNeighbor, self).__init__()

        if interpolant_type not in _interpolators.keys():
            msg = "NearestNeighbor: interpolant_type '{0}' not supported." \
                  " interpolant_type must be one of {1}.".format(
                      interpolant_type, list(_interpolators.keys())
                  )
            raise ValueError(msg)

        self.interpolant_init_args = kwargs

        self.interpolant_type = interpolant_type
        self.interpolant = None

    def train(self, x, y):
        """
        Train the surrogate model with the given set of inputs and outputs.

        Args
        ----
        x : array-like
            Training input locations

        y : array-like
            Model responses at given inputs.
        """
        super(NearestNeighbor, self).train(x, y)
        self.interpolant = _interpolators[self.interpolant_type](x, y, **self.interpolant_init_args)

    def predict(self, x, **kwargs):
        """
        Calculates a predicted value of the response based on the current
        trained model for the supplied list of inputs.

        Args
        ----
        x : array-like
            Point(s) at which the surrogate is evaluated.

        kwargs :
            Additional keyword arguments passed to the interpolant.
        """
        super(NearestNeighbor, self).predict(x)
        return self.interpolant(x, **kwargs)

    def linearize(self, x, **kwargs):
        """
        Calculates the jacobian of the interpolant at the requested point.

        Args
        ----
        x : array-like
            Point at which the surrogate Jacobian is evaluated.

        kwargs :
            Additional keyword arguments passed to the interpolant.
        """
        jac = self.interpolant.gradient(x, **kwargs)
        if jac.shape[0] == 1 and len(jac.shape) > 2:
            return jac[0, ...]
        return jac
